import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2
import numpy as np
from collections import deque, Counter
import os
import zipfile
import shutil
import time
import urllib.request
import argparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, XCLIPProcessor, XCLIPModel
from tensorflow.keras.models import load_model

# ── Google Colab / IPython optional imports ──────────────────────────────────
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    from IPython.display import clear_output
    IN_IPYTHON = True
except ImportError:
    IN_IPYTHON = False

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()          # fallback when running as a Colab cell

MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


# ============================================================================
# ATTENTION FUSION MODEL (for Overall Risk Score)
# ============================================================================

class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Linear(3, 3)

    def forward(self, x):
        attn_logits = self.attention_layer(x)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        overall_risk = (attn_weights * x).sum(dim=-1, keepdim=True)
        return overall_risk, attn_weights


# ============================================================================
# MODULE 1: GENDER CLASSIFICATION
# ============================================================================

class GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderClassifier, self).__init__()
        self.backbone = mobilenet_v2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(p=0.3), nn.Linear(512, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(p=0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class VisualTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.history = {}
        self.max_lost = 5
        self.min_hits = 5
        self.iou_threshold = 0.3

    def get_appearance(self, crop):
        if crop.size == 0: return np.zeros((24,), dtype=np.float32)
        crop = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def update(self, detections):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['lost'] += 1
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]

        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for t, tid in enumerate(track_ids):
                track = self.tracks[tid]
                for d, det in enumerate(detections):
                    box = det[0]
                    iou = self.iou(track['box'], box)
                    vis = cv2.compareHist(track['hist'], self.get_appearance(det[1]), cv2.HISTCMP_CORREL)
                    cost_matrix[t, d] = 1.0 - (0.6 * iou + 0.4 * max(0, vis))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            used_detections = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.6:
                    self.update_track(track_ids[r], detections[c])
                    used_detections.add(c)

            for i in range(len(detections)):
                if i not in used_detections:
                    self.create_track(detections[i])

        elif len(detections) > 0:
            for d in detections: self.create_track(d)

        self.remove_duplicates()

        for tid, t in self.tracks.items():
            self.history[tid] = t.copy()

        return self.get_visible_tracks()

    def remove_duplicates(self):
        ids = list(self.tracks.keys())
        to_remove = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a = ids[i]
                id_b = ids[j]
                if id_a in to_remove or id_b in to_remove: continue
                iou = self.iou(self.tracks[id_a]['box'], self.tracks[id_b]['box'])
                if iou > 0.65:
                    if self.tracks[id_a]['hits'] >= self.tracks[id_b]['hits']:
                        to_remove.add(id_b)
                    else:
                        to_remove.add(id_a)
        for tid in to_remove:
            del self.tracks[tid]

    def create_track(self, det):
        box, crop, gender, f_prob = det
        c = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        self.tracks[self.next_id] = {
            'id': self.next_id, 'box': box, 'centroid': c,
            'hist': self.get_appearance(crop),
            'gender_votes': deque([gender], maxlen=20),
            'female_probs': deque([f_prob], maxlen=50),
            'hits': 1, 'lost': 0, 'max_risk_score': 0.0, 'risk_reason': "Safe", 'is_threat': False
        }
        self.history[self.next_id] = self.tracks[self.next_id]
        self.next_id += 1

    def update_track(self, tid, det):
        box, crop, gender, f_prob = det
        c = ((box[0]+box[2])//2, (box[1]+box[3])//2)
        t = self.tracks[tid]
        t['box'] = box; t['centroid'] = c
        t['hist'] = 0.8 * t['hist'] + 0.2 * self.get_appearance(crop)
        t['gender_votes'].append(gender)
        t['female_probs'].append(f_prob)
        t['hits'] += 1
        t['lost'] = 0

    def record_risk(self, victim_id, threat_id, risk_score, reason):
        if victim_id in self.tracks:
            if risk_score > self.tracks[victim_id]['max_risk_score']:
                self.tracks[victim_id]['max_risk_score'] = risk_score
                self.tracks[victim_id]['risk_reason'] = reason
            self.history[victim_id] = self.tracks[victim_id].copy()

        if threat_id in self.tracks:
            self.tracks[threat_id]['is_threat'] = True
            self.history[threat_id] = self.tracks[threat_id].copy()

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def get_visible_tracks(self):
        visible = []
        for tid, t in self.tracks.items():
            if t['hits'] >= self.min_hits and t['lost'] < 2:
                avg_f_prob = sum(t['female_probs']) / len(t['female_probs'])
                final_g = "Female" if avg_f_prob > 0.35 else "Male"
                visible.append({'id': tid, 'box': t['box'], 'centroid': t['centroid'], 'gender': final_g, 'f_conf': avg_f_prob})
        return visible

    def get_final_history(self):
        clean = {}
        for tid, t in self.history.items():
            if t['hits'] > 15 or t['max_risk_score'] > 0 or t['is_threat']:
                t['final_gender'] = Counter(t['gender_votes']).most_common(1)[0][0]
                clean[tid] = t
        return clean


class DynamicSafetyLogic:
    def calculate_risk(self, people, frame_width):
        females = [p for p in people if p['gender'] == 'Female']
        total_crowd = len(people)
        reports = []

        # 1. Global Bystander Damping
        bystander_count = max(0, total_crowd - 2)
        bystander_damping = 1.0 / (1.0 + (0.03 * bystander_count))

        for f in females:
            f_id = f['id']
            local_males = 0
            threat_id = None
            local_females = 0

            # --- LOCAL ANALYSIS (250px Bubble) ---
            for other in people:
                if other['id'] == f_id: continue
                d = dist.euclidean(f['centroid'], other['centroid'])

                if d < 250:
                    if other['gender'] == 'Male':
                        local_males += 1
                        threat_id = other['id']
                    else:
                        local_females += 1

            # --- BASE RISK CALCULATION ---
            threat_score = np.tanh(0.8 * local_males)
            mitigation = np.tanh(0.5 * local_females)
            base_risk = max(0, threat_score - (0.5 * mitigation))

            # --- CONTEXTUAL ADJUSTMENT ---
            risk_val = 0.0
            reason = "Safe"

            # Case 1: Isolated (Alone) - High Risk
            if total_crowd == 1:
                risk_val = 0.90
                reason = "High Risk: Isolated/Alone"

            # Case 2: Harassment (Male Proximity)
            elif local_males > 0:
                # Apply Global Bystander Damping
                risk_val = base_risk * bystander_damping

                if risk_val > 0.7:
                    reason = f"CRITICAL: Harassed by {local_males} Male(s)"
                elif risk_val > 0.4:
                    reason = f"WARNING: Proximity ({total_crowd} people nearby)"
                else:
                    reason = f"Caution: Male Proximity"

            # Case 3: Safe Crowd
            else:
                risk_val = 0.1 * bystander_damping
                reason = "Safe (Crowd)"

            reports.append({
                'id': f_id, 'risk': risk_val, 'reason': reason,
                'box': f['box'], 'threat_id': threat_id
            })

        return reports


def run_gender_module(fname):
    print("\n" + "="*30)
    print(" MODULE 1: GENDER CLASSIFICATION")
    print("="*30)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'best_mobilenet_gender_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Returning 0.0")
        return 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen_model = GenderClassifier(2)
    gen_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    gen_model.to(device).eval()

    print("Loading Advanced Detector (YOLOv5m)...")
    detector = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    detector.classes=[0]; detector.conf = 0.40; detector.iou = 0.50
    detector.to(device)

    tracker = VisualTracker()
    logic = DynamicSafetyLogic()

    cap = cv2.VideoCapture(fname)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3)); h = int(cap.get(4))

    out_filename = f"final_gender_{os.path.basename(fname)}"
    out_path = os.path.join(OUTPUTS_DIR, out_filename)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    gender_risk_history = []

    print(f"Processing {fname}...")
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        idx += 1

        results = detector(frame)
        raw_detections = []

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1,y1,x2,y2 = map(int, xyxy)

            box_area = (x2-x1) * (y2-y1)
            if box_area > (w * h * 0.60): continue
            if (x2-x1) < 30 or (y2-y1) < 50: continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            t_crop = trans(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(gen_model(t_crop), 1)
                f_prob = probs[0][0].item()
                g = "Female" if f_prob > 0.5 else "Male"

            raw_detections.append(([x1,y1,x2,y2], crop, g, f_prob))

        active_tracks = tracker.update(raw_detections)
        risk_reports = logic.calculate_risk(active_tracks, w)

        female_scores = [r['risk'] for r in risk_reports]
        final_scene_risk = max(female_scores) if female_scores else 0.0
        gender_risk_history.append(final_scene_risk)

        for r in risk_reports:
            tracker.record_risk(r['id'], r['threat_id'], r['risk'], r['reason'])
            if r['risk'] > 0.5:
                x1,y1,x2,y2 = r['box']
                color = (0,0,255) if r['risk'] > 0.8 else (0,165,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 4)
                cv2.putText(frame, f"RISK {r['risk']:.2f}", (x1, y1-10), 0, 0.6, color, 2)

        for p in active_tracks:
            is_risky = any(r['id'] == p['id'] and r['risk'] > 0.5 for r in risk_reports)
            if not is_risky:
                c = (255,100,100) if p['gender']=='Female' else (100,255,100)
                cv2.rectangle(frame, (p['box'][0],p['box'][1]), (p['box'][2],p['box'][3]), c, 2)
                cv2.putText(frame, f"{p['gender'][0]}{p['id']}", (p['box'][0],p['box'][1]-5), 0, 0.6, c, 2)

        overlay_color = (0, 255, 0)
        if final_scene_risk > 0.8: overlay_color = (0, 0, 255)
        elif final_scene_risk > 0.5: overlay_color = (0, 165, 255)

        cv2.putText(frame, f"SCENE RISK: {final_scene_risk:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
        out.write(frame)

    cap.release(); out.release()
    print(f"Saved → {out_path}")

    if IN_COLAB:
        colab_files.download(out_path)

    # --- FINAL REPORT (HYBRID APPROACH) ---
    gender_risk_history = np.array(gender_risk_history)
    overall_gender_risk = 0.0

    if len(gender_risk_history) > 0:
        avg_risk = float(np.mean(gender_risk_history))
        peak_risk = float(np.max(gender_risk_history))
        duration_ratio = float(np.mean(gender_risk_history > 0.5))

        total = avg_risk + peak_risk + duration_ratio + 1e-6
        w_avg = avg_risk / total
        w_peak = peak_risk / total
        w_dur = duration_ratio / total

        overall_gender_risk = (
            w_avg * avg_risk +
            w_peak * peak_risk +
            w_dur * duration_ratio
        )

        print("\n GENDER MODULE - OVERALL RISK REPORT")
        print("=" * 45)
        print(f"Total Frames Analyzed: {len(gender_risk_history)}")
        print(f"Average Gender Risk  : {avg_risk:.2f}")
        print(f"Peak Gender Risk     : {peak_risk:.2f}")
        print(f"High-Risk Duration   : {duration_ratio*100:.1f}%")
        print("-" * 45)
        print(f"FINAL GENDER RISK    : {overall_gender_risk:.2f}")

    return overall_gender_risk


# ============================================================================
# MODULE 2: VIOLENCE DETECTION
# ============================================================================

def run_violence_module(VIDEO_PATH):
    print("\n" + "="*30)
    print(" MODULE 2: VIOLENCE DETECTION")
    print("="*30)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    ZIP_PATH    = os.path.join(MODELS_DIR, 'final_violence_model.zip')
    EXTRACT_PATH = os.path.join(MODELS_DIR, 'custom_violence_model')

    if not os.path.exists(ZIP_PATH):
        print("WARNING: final_violence_model.zip not found in models/. Using Base Model.")
        MODEL_DIR = "MCG-NJU/videomae-base-finetuned-kinetics"
    else:
        if os.path.exists(EXTRACT_PATH): shutil.rmtree(EXTRACT_PATH)
        os.makedirs(EXTRACT_PATH)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref: zip_ref.extractall(EXTRACT_PATH)
        MODEL_DIR = EXTRACT_PATH

    out_filename = f"final_violence_{os.path.basename(VIDEO_PATH)}"
    OUTPUT_PATH  = os.path.join(OUTPUTS_DIR, out_filename)

    # --- SETTINGS ---
    FRAME_STRIDE = 5
    VIOLENCE_CLASS_ID = 1
    DANGEROUS_ACTIONS = [
        "punching", "kicking", "shooting", "fighting", "slapping",
        "strangling", "dragging", "pulling", "hitting"
    ]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- LOAD MODELS ---
    try:
        mae_processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
        mae_model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    except:
        MODEL_DIR = "MCG-NJU/videomae-base-finetuned-kinetics"
        mae_processor = VideoMAEImageProcessor.from_pretrained(MODEL_DIR)
        mae_model = VideoMAEForVideoClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    mae_model.eval()

    XCLIP_MODEL_NAME = "microsoft/xclip-base-patch32"
    xclip_processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_NAME)
    xclip_model = XCLIPModel.from_pretrained(XCLIP_MODEL_NAME).to(DEVICE)
    xclip_model.eval()

    XCLIP_LABELS = [
        "person punching someone", "person kicking someone", "person shooting a gun",
        "people fighting", "person slapping someone", "person strangling someone",
        "person dragging someone", "person pulling someone",
        "normal conversation", "person walking", "person sitting", "people hugging"
    ]

    # --- PROCESSING LOOP ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_buffer = []
    BUFFER_SIZE = 16
    PREDICTION_INTERVAL = 8
    score_history = deque(maxlen=50)
    adaptive_threshold = 0.50

    current_status = "Calibrating..."
    current_action = ""
    risk_score = 0.0
    status_color = (0, 255, 255)

    frame_count = 0
    buffer_fill_count = 0
    all_risk_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % FRAME_STRIDE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frame_buffer.append(pil_image)
            if len(frame_buffer) > BUFFER_SIZE: frame_buffer.pop(0)
            buffer_fill_count += 1

        if len(frame_buffer) == BUFFER_SIZE and buffer_fill_count % PREDICTION_INTERVAL == 0:
            inputs = mae_processor(list(frame_buffer), return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = mae_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            mae_prob = probs[0][VIOLENCE_CLASS_ID].item()

            xclip_indices = np.linspace(0, len(frame_buffer)-1, 8, dtype=int)
            clip_frames = [frame_buffer[i] for i in xclip_indices]
            inputs = xclip_processor(text=XCLIP_LABELS, images=clip_frames, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = xclip_model(**inputs)
                probs = outputs.logits_per_video.softmax(dim=1)

            best_idx = probs.argmax().item()
            clip_conf = probs[0][best_idx].item()
            current_action = XCLIP_LABELS[best_idx]

            is_dangerous = any(k in current_action for k in DANGEROUS_ACTIONS)
            action_risk = clip_conf if is_dangerous else 0.0
            raw_risk = 1.0 - ((1.0 - mae_prob) * (1.0 - action_risk))
            risk_score = max(0.0, min(1.0, raw_risk))

            all_risk_scores.append(risk_score)

            score_history.append(risk_score)
            if len(score_history) > 5:
                avg_noise = sum(score_history) / len(score_history)
                adaptive_threshold = max(0.40, min(0.85, 0.40 + (0.3 * avg_noise)))

            if risk_score > 0.85:
                current_status = "VIOLENCE DETECTED"
                status_color = (0, 0, 255)
            elif risk_score > adaptive_threshold:
                current_status = "SUSPICIOUS ACTIVITY"
                status_color = (0, 165, 255)
            else:
                current_status = "SAFE"
                status_color = (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45; thickness = 1; padding = 8; line_spacing = 4

        txt_status = f"STATUS: {current_status}"
        txt_risk   = f"RISK: {risk_score:.0%} (Limit: {adaptive_threshold:.0%})"
        txt_action = f"ACTION: {current_action}"

        s_size = cv2.getTextSize(txt_status, font, scale, thickness)[0]
        r_size = cv2.getTextSize(txt_risk,   font, scale, thickness)[0]
        a_size = cv2.getTextSize(txt_action, font, scale, thickness)[0]
        box_w = max(s_size[0], r_size[0], a_size[0]) + (padding * 2)
        box_h = s_size[1] + r_size[1] + a_size[1] + (padding * 2) + (line_spacing * 2)
        margin = 15
        top_left     = (width - box_w - margin, margin)
        bottom_right = (width - margin, margin + box_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        x = width - box_w - margin + padding
        y = margin + s_size[1] + padding
        cv2.putText(frame, txt_status, (x, y), font, scale, status_color, thickness)
        y += r_size[1] + line_spacing
        risk_c = (0, int(255*(1-risk_score)), int(255*risk_score))
        cv2.putText(frame, txt_risk,   (x, y), font, scale, risk_c, thickness)
        y += a_size[1] + line_spacing
        cv2.putText(frame, txt_action, (x, y), font, scale, (255, 255, 255), thickness)

        if risk_score > 0.85:
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 4)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved → {OUTPUT_PATH}")

    if IN_COLAB:
        colab_files.download(OUTPUT_PATH)

    overall_risk = 0.0
    if len(all_risk_scores) > 0:
        all_risk_scores.sort(reverse=True)
        top_n = max(1, int(len(all_risk_scores) * 0.1))
        top_scores = all_risk_scores[:top_n]
        overall_risk = sum(top_scores) / top_n
        overall_risk = min(1.0, max(0.0, overall_risk))

    print(f"VIOLENCE RISK (S_action): {overall_risk:.2f}")
    return overall_risk


# ============================================================================
# MODULE 3: EMOTION RECOGNITION
# ============================================================================

def run_emotion_module(INPUT_VIDEO):
    print("\n" + "="*30)
    print(" MODULE 3: EMOTION RECOGNITION")
    print("="*30)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --- LOAD EMOTION MODEL ---
    model_path = None
    for name in ['emotion_model.h5', 'emotion_model.hdf5']:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print(f"CRITICAL WARNING: No emotion_model.h5 found in {MODELS_DIR}/. Code will crash on 'model.predict'.")
    else:
        global model, class_labels
        model = load_model(model_path, compile=False)
        class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # ---------- CONFIG ----------
    out_filename = f"final_emotion_{os.path.basename(INPUT_VIDEO)}"
    OUTPUT_VIDEO = os.path.join(OUTPUTS_DIR, out_filename)

    DNN_PROTO      = os.path.join(MODELS_DIR, "deploy.prototxt")
    DNN_MODEL_FILE = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    DNN_CONF_THRESHOLD  = 0.3
    TEMPORAL_WINDOW     = 5
    TEMPORAL_MIN_PRESENCE = 1
    TARGET_W            = 960
    DISPLAY_EVERY       = 12

    # ---------- DOWNLOAD FACE DNN WEIGHTS (auto) ----------
    def download(url, path):
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)

    download(PROTO_URL, DNN_PROTO)
    download(MODEL_URL, DNN_MODEL_FILE)

    # ---------- LOAD FACE DNN ----------
    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL_FILE)

    # ---------- VIDEO IO ----------
    cap = cv2.VideoCapture(INPUT_VIDEO)
    assert cap.isOpened(), "Video not found or cannot be opened"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20

    scale = TARGET_W / orig_w
    OUT_W = TARGET_W
    OUT_H = int(orig_h * scale)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (OUT_W, OUT_H)
    )

    # ---------- HELPERS ----------
    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, xB-xA+1) * max(0, yB-yA+1)
        areaA  = (a[2]-a[0]+1)*(a[3]-a[1]+1)
        areaB  = (b[2]-b[0]+1)*(b[3]-b[1]+1)
        return inter / float(areaA + areaB - inter + 1e-6)

    recent_detections = deque(maxlen=TEMPORAL_WINDOW)

    # ---------- EMOTION RISK SETUP ----------
    NEGATIVE_EMOTIONS   = ["angry", "disgust", "fear", "sad", "surprise"]
    class_labels_lower  = [c.lower() for c in class_labels]
    NEG_IDX             = [class_labels_lower.index(e) for e in NEGATIVE_EMOTIONS]

    neg_baseline       = 0.0
    BASELINE_MOMENTUM  = 0.98
    emotion_risk_prev  = 0.0
    RISK_ALPHA         = 0.3
    DELTA              = 0.5
    emotion_risk_history = []

    # ---------- PROCESS ----------
    frame_id = 0
    start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        frame = cv2.resize(frame, (OUT_W, OUT_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w  = gray.shape

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300,300)), 1.0, (300,300), (104,177,123)
        )
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf < DNN_CONF_THRESHOLD:
                continue
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w-1, x2); y2 = min(h-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))

        recent_detections.append(boxes)
        valid_boxes = []

        for b in boxes:
            count = 0
            for past in recent_detections:
                if any(iou(b, p) > 0.4 for p in past):
                    count += 1
            if count >= TEMPORAL_MIN_PRESENCE:
                valid_boxes.append(b)

        faces, coords = [], []
        for (x1,y1,x2,y2) in valid_boxes:
            pad = int(0.2 * (y2 - y1))
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(w-1, x2 + pad); y2p = min(h-1, y2 + pad)
            face = gray[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue
            face = cv2.resize(face, (48,48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            coords.append((x1,y1,x2,y2))

        if len(faces) == 0:
            writer.write(frame)
            continue

        batch = np.vstack(faces)
        preds = model.predict(batch, verbose=0)

        for (x1,y1,x2,y2), p in zip(coords, preds):
            idx   = int(np.argmax(p))
            label = class_labels[idx]
            conf  = float(p[idx])

            # -------- Dynamic Emotion Risk --------
            neg_mass     = float(np.sum(p[NEG_IDX]))
            neg_baseline = BASELINE_MOMENTUM * neg_baseline + (1-BASELINE_MOMENTUM)*neg_mass
            deviation    = max(0.0, neg_mass - neg_baseline)
            risk_raw     = min(1.0, deviation / DELTA)

            emotion_risk      = RISK_ALPHA*risk_raw + (1-RISK_ALPHA)*emotion_risk_prev
            emotion_risk_prev = emotion_risk
            emotion_risk_history.append(emotion_risk)

            risk_pct = int(emotion_risk * 100)
            text     = f"{label} {conf:.2f} | Risk {risk_pct}%"
            color    = (0, int(255*(1-emotion_risk)), int(255*emotion_risk))

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, text, (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)

        if frame_id % DISPLAY_EVERY == 0:
            if IN_IPYTHON:
                clear_output(wait=True)
            print(f"Frame {frame_id}, FPS ~ {frame_id/(time.time()-start):.2f}")

    cap.release()
    writer.release()
    print(f"Saved → {OUTPUT_VIDEO}")

    if IN_COLAB:
        colab_files.download(OUTPUT_VIDEO)

    # ============================
    # FINAL OVERALL EMOTION RISK
    # ============================
    emotion_risk_history = np.array(emotion_risk_history)
    overall_emotion_risk = 0.0

    if len(emotion_risk_history) > 0:
        avg_risk       = float(np.mean(emotion_risk_history))
        peak_risk      = float(np.max(emotion_risk_history))
        duration_ratio = float(np.mean(emotion_risk_history > 0.6))

        total   = avg_risk + peak_risk + duration_ratio + 1e-6
        w_avg   = avg_risk / total
        w_peak  = peak_risk / total
        w_dur   = duration_ratio / total

        overall_emotion_risk = (
            w_avg * avg_risk +
            w_peak * peak_risk +
            w_dur * duration_ratio
        )

        print("\n EMOTION MODULE — OVERALL RISK REPORT")
        print("="*45)
        print(f"Average Emotion Risk : {avg_risk:.2f}")
        print(f"Peak Emotion Risk    : {peak_risk:.2f}")
        print(f"High-Risk Duration   : {duration_ratio*100:.1f}%")
        print("-"*45)
        print(f"FINAL EMOTION RISK   : {overall_emotion_risk:.2f}")

    print("Done →", OUTPUT_VIDEO)
    return overall_emotion_risk


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main_pipeline(video_path):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Gender
    S_gender = run_gender_module(video_path)

    # 2. Violence
    S_action = run_violence_module(video_path)

    # 3. Emotion
    S_emotion = run_emotion_module(video_path)

    # 4. Attention-Based Fusion
    FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "attention_fusion_model.pth")
    if not os.path.exists(FUSION_MODEL_PATH):
        FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "saved_models", "attention_fusion_model.pth")

    if os.path.exists(FUSION_MODEL_PATH):
        fusion_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fusion_model = AttentionFusion()
        checkpoint   = torch.load(FUSION_MODEL_PATH, map_location=fusion_device, weights_only=True)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.to(fusion_device)
        fusion_model.eval()

        input_scores = torch.FloatTensor([[S_gender, S_action, S_emotion]]).to(fusion_device)

        with torch.no_grad():
            S_final_tensor, attn_weights = fusion_model(input_scores)
            S_final  = S_final_tensor.item()
            w_gender = attn_weights[0][0].item()
            w_action = attn_weights[0][1].item()
            w_emotion= attn_weights[0][2].item()

        S_final       = min(1.0, max(0.0, S_final))
        fusion_method = "Attention-Based Fusion"

    else:
        print("WARNING: attention_fusion_model.pth not found in models/. Using fixed weights.")
        w_gender, w_action, w_emotion = 0.30, 0.50, 0.20
        S_final = (w_action * S_action) + (w_gender * S_gender) + (w_emotion * S_emotion)
        S_final = min(1.0, max(0.0, S_final))
        fusion_method = "Fixed Weights (fallback)"

    # 5. Final Report
    print("\n" + "="*50)
    print(" FINAL INTEGRATED SECURITY REPORT")
    print("="*50)
    print(f"Video Source        : {video_path}")
    print(f"Fusion Method       : {fusion_method}")
    print("-" * 50)
    print(f"S_gender  (Safety)  : {S_gender:.4f}")
    print(f"S_action  (Violence): {S_action:.4f}")
    print(f"S_emotion (Distress): {S_emotion:.4f}")
    print("-" * 50)
    print(f"Attention Weights:")
    print(f"  Gender:  {w_gender:.3f}")
    print(f"  Action:  {w_action:.3f}")
    print(f"  Emotion: {w_emotion:.3f}")
    print("-" * 50)
    print(f"OVERALL THREAT SCORE (S_final): {S_final:.4f}")
    print("-" * 50)

    if S_final > 0.75:
        print(">>> SYSTEM ALERT: CRITICAL THREAT DETECTED")
    elif S_final > 0.40:
        print(">>> SYSTEM ALERT: SUSPICIOUS ACTIVITY")
    else:
        print(">>> SYSTEM STATUS: SAFE")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if IN_COLAB:
        # ── Colab: prompt user to upload a video ────────────────────────────
        print("--- UPLOAD INPUT VIDEO ---")
        uploaded = colab_files.upload()
        if not uploaded:
            print("No video uploaded.")
        else:
            VIDEO_PATH = list(uploaded.keys())[0]
            main_pipeline(VIDEO_PATH)
    else:
        # ── Terminal: accept video path as CLI argument ──────────────────────
        parser = argparse.ArgumentParser(
            description="CARE-W: Context-Aware Risk Evaluation for Women's Safety"
        )
        parser.add_argument(
            "--video", required=True,
            help="Path to the input surveillance video (mp4, avi, mov …)"
        )
        args = parser.parse_args()

        if not os.path.isfile(args.video):
            print(f"[ERROR] Video file not found: {args.video}")
        else:
            main_pipeline(args.video)
