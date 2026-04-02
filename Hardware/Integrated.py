import os
# --- JETSON MEMORY HACK ---
# Forces TensorFlow (Emotion) to use CPU so PyTorch can use the GPU for VideoMAE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import numpy as np
import math
import time
import threading
import subprocess
import zipfile
import shutil
from collections import deque
import urllib.request
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write

# ============================================================================
# 0. HARDWARE AUDIO ALARM SETUP
# ============================================================================
def create_alarm_sound(filename="alarm.wav"):
    if not os.path.exists(filename):
        print("Generating audio alarm file...")
        sample_rate = 44100
        t = np.linspace(0, 1.0, sample_rate)
        audio = np.sin(2 * np.pi * 800 * t) * (t % 0.2 < 0.1) + np.sin(2 * np.pi * 1000 * t) * (t % 0.2 >= 0.1)
        write(filename, sample_rate, (audio * 32767).astype(np.int16))

# ============================================================================
# 1. MODEL ARCHITECTURES (This is what went missing!)
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

class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Linear(3, 3)
    def forward(self, x):
        attn_logits = self.attention_layer(x)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        overall_risk = (attn_weights * x).sum(dim=-1, keepdim=True)
        return overall_risk, attn_weights

# --- GLOBAL VARIABLES FOR ASYNC THREADING ---
S_action_global = 0.0
action_thread_running = False

def run_videomae_thread(frames_list, processor, model, device):
    global S_action_global, action_thread_running
    try:
        inputs = processor(frames_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        S_action_global = probs[0][1].item() # Assuming class 1 is violence
    except Exception as e:
        print(f"VideoMAE Error: {e}")
    finally:
        action_thread_running = False

# ============================================================================
# MAIN LIVE PIPELINE
# ============================================================================
def main():
    print("\n" + "="*50)
    print(" INITIALIZING OPTIMIZED ASYNC PIPELINE ")
    print("="*50)

    create_alarm_sound("alarm.wav")
    last_alarm_time = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    # ---------------------------------------------------------
    # LOAD GENDER MODULE (YOLOv5n + MobileNetV2)
    # ---------------------------------------------------------
    print("\n[1/4] Loading Gender Module...")
    yolo_detector = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    yolo_detector.classes = [0]
    yolo_detector.conf = 0.25
   
    gen_model = GenderClassifier(2)
    if os.path.exists('best_mobilenet_gender_model.pth'):
        gen_model.load_state_dict(torch.load('best_mobilenet_gender_model.pth', map_location=device), strict=False)
    gen_model.to(device).eval()
    gen_trans = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---------------------------------------------------------
    # LOAD EMOTION MODULE (SSD + Keras CNN)
    # ---------------------------------------------------------
    print("\n[2/4] Loading Emotion Module...")
    if not os.path.exists("deploy.prototxt"):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", "deploy.prototxt")
    if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        urllib.request.urlretrieve("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", "res10_300x300_ssd_iter_140000.caffemodel")
   
    face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    emotion_model = load_model('emotion_model.h5', compile=False)
    NEG_IDX = [0, 1, 2, 5, 6] # Angry, Disgust, Fear, Sad, Surprise

    # ---------------------------------------------------------
    # LOAD CUSTOM VIOLENCE MODULE (VideoMAE ZIP)
    # ---------------------------------------------------------
    print("\n[3/4] Loading Violence Module (Custom VideoMAE)...")
    ZIP_PATH = "final_violence_model.zip"
    EXTRACT_PATH = "custom_violence_model"

    if not os.path.exists(ZIP_PATH):
        print(f"WARNING: {ZIP_PATH} not found. Using Base HuggingFace Model.")
        MAE_MODEL_DIR = "MCG-NJU/videomae-base-finetuned-kinetics"
    else:
        print("Extracting custom violence model...")
        if os.path.exists(EXTRACT_PATH):
            shutil.rmtree(EXTRACT_PATH)
        os.makedirs(EXTRACT_PATH)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        MAE_MODEL_DIR = EXTRACT_PATH

    try:
        mae_processor = VideoMAEImageProcessor.from_pretrained(MAE_MODEL_DIR)
        mae_model = VideoMAEForVideoClassification.from_pretrained(MAE_MODEL_DIR).to(device)
    except Exception as e:
        print(f"Error loading custom model ({e}). Falling back to base model...")
        MAE_MODEL_DIR = "MCG-NJU/videomae-base-finetuned-kinetics"
        mae_processor = VideoMAEImageProcessor.from_pretrained(MAE_MODEL_DIR)
        mae_model = VideoMAEForVideoClassification.from_pretrained(MAE_MODEL_DIR).to(device)
       
    mae_model.eval()
    video_buffer = deque(maxlen=16)
    last_frame_grab_time = time.time()

    # ---------------------------------------------------------
    # LOAD FUSION MODULE
    # ---------------------------------------------------------
    print("\n[4/4] Loading Attention Fusion Module...")
    fusion_model = AttentionFusion()
    use_attention = False
    if os.path.exists("attention_fusion_model.pth"):
        checkpoint = torch.load("attention_fusion_model.pth", map_location=device, weights_only=True)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.to(device).eval()
        use_attention = True
        print("-> Using Custom Attention Fusion")
    else:
        print("-> Using Fixed Weights Fallback")

    # ---------------------------------------------------------
    # START CAMERA (Fixed for Jetson Hardware)
    # ---------------------------------------------------------
    gstreamer_str = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=1280, height=720, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
   
    if not cap.isOpened():
        print("CSI Camera failed. Run: sudo systemctl restart nvargus-daemon")
        return

    print("\n--- LIVE SYSTEM RUNNING. PRESS 'q' TO QUIT ---")

    cached_people = []
    cached_faces = []
    S_gender, S_emotion, S_final = 0.0, 0.0, 0.0
    frame_count = 0

    global action_thread_running, S_action_global

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            frame_count += 1
            current_time = time.time()

            # =========================================================
            # 1. ASYNC VIOLENCE MODULE (Time-Synced Buffer)
            # =========================================================
            if current_time - last_frame_grab_time >= 0.1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_buffer.append(Image.fromarray(frame_rgb))
                last_frame_grab_time = current_time

            if len(video_buffer) == 16 and not action_thread_running:
                action_thread_running = True
                buffer_copy = list(video_buffer)
                threading.Thread(target=run_videomae_thread, args=(buffer_copy, mae_processor, mae_model, device)).start()

            # =========================================================
            # 2. GENDER & EMOTION MODULE
            # =========================================================
            if frame_count % 3 == 0:
                results = yolo_detector(frame)
                cached_people = []
                for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2 = map(int, xyxy)
                    if (x2 - x1) < 15 or (y2 - y1) < 25: continue
                   
                    # TIGHT CROP FIX: Focuses on the upper body/face for better gender classification
                    center_y = y1 + (y2 - y1) // 3  
                    crop_y1 = max(0, int(center_y - (y2 - y1) * 0.3))
                    crop_y2 = min(h - 1, int(center_y + (y2 - y1) * 0.3))
                    crop_x1 = max(0, int(x1 + (x2 - x1) * 0.1))
                    crop_x2 = min(w - 1, int(x2 - (x2 - x1) * 0.1))

                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                   
                    if crop.size > 0:
                        t_crop = gen_trans(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            f_prob = torch.softmax(gen_model(t_crop), 1)[0][0].item()
                            gender = "Female" if f_prob > 0.5 else "Male"
                        cached_people.append({'box': (x1, y1, x2, y2), 'centroid': ((x1+x2)//2, (y1+y2)//2), 'gender': gender})

                # --- THESIS SPATIAL MATH (Eq 3.1 - 3.8) ---
                N_total = len(cached_people)
                D_bystander = 1.0 / (1.0 + 0.03 * (N_total - 2)) if N_total >= 2 else 1.0
                females = [p for p in cached_people if p['gender'] == 'Female']
                males = [p for p in cached_people if p['gender'] == 'Male']
               
                frame_gender_risk = 0.0
                for f in females:
                    cx_f, cy_f = f['centroid']
                    if N_total == 1:
                        risk = 0.90
                    else:
                        N_local_male = sum(1 for m in males if math.hypot(m['centroid'][0]-cx_f, m['centroid'][1]-cy_f) <= 250)
                        N_local_female = sum(1 for of in females if of is not f and math.hypot(of['centroid'][0]-cx_f, of['centroid'][1]-cy_f) <= 250)
                       
                        if N_local_male == 0:
                            risk = 0.1 * D_bystander
                        else:
                            risk = max(0.0, math.tanh(0.8 * N_local_male) - (0.5 * math.tanh(0.5 * N_local_female))) * D_bystander
                    frame_gender_risk = max(frame_gender_risk, risk)
                S_gender = frame_gender_risk

            if frame_count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
                face_net.setInput(blob)
                face_dets = face_net.forward()
               
                cached_faces, faces_crops = [], []
                for i in range(face_dets.shape[2]):
                    if face_dets[0, 0, i, 2] > 0.4:
                        box = face_dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        face_crop = gray[max(0,y1):min(h-1,y2), max(0,x1):min(w-1,x2)]
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (48, 48)).astype("float32") / 255.0
                            faces_crops.append(np.expand_dims(np.expand_dims(face_resized, axis=-1), axis=0))
                            cached_faces.append((x1, y1, x2, y2))

                S_emotion = 0.0
                if len(faces_crops) > 0:
                    batch = np.vstack(faces_crops)
                    preds = emotion_model.predict(batch, verbose=0)
                    for p in preds:
                        S_emotion = max(S_emotion, min(1.0, float(np.sum(p[NEG_IDX])) / 0.5))

            # =========================================================
            # 3. FUSION & AUDIO ALARM
            # =========================================================
            if use_attention:
                inp = torch.FloatTensor([[S_gender, S_action_global, S_emotion]]).to(device)
                with torch.no_grad():
                    S_final_tensor, _ = fusion_model(inp)
                    S_final = min(1.0, max(0.0, S_final_tensor.item()))
            else:
                S_final = min(1.0, max(0.0, (0.3 * S_gender) + (0.5 * S_action_global) + (0.2 * S_emotion)))

            if S_final > 0.70:
                if current_time - last_alarm_time > 1.5:
                    subprocess.Popen(['aplay', '-q', 'alarm.wav'])
                    last_alarm_time = current_time

            # =========================================================
            # 4. VISUALIZATION
            # =========================================================
            status_text = "SAFE"
            border_color = (0, 255, 0)

            if S_final > 0.70:
                status_text = "CRITICAL THREAT"
                border_color = (0, 0, 255)
            elif S_final > 0.40:
                status_text = "SUSPICIOUS"
                border_color = (0, 165, 255)

            for p in cached_people:
                x1, y1, x2, y2 = p['box']
                if S_final > 0.70:
                    box_color = (0, 0, 255)
                elif S_final > 0.40:
                    box_color = (0, 165, 255)
                else:
                    box_color = (203, 192, 255) if p['gender'] == 'Female' else (255, 100, 0)
               
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, p['gender'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for (x1, y1, x2, y2) in cached_faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)
               
            cv2.rectangle(frame, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.putText(frame, f"S_gender  (Math)       : {S_gender:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"S_action  (VideoMAE)   : {S_action_global:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"S_emotion (Face CNN)   : {S_emotion:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"OVERALL: {int(S_final*100)}% - {status_text}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
           
            if S_final > 0.70:
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)

            cv2.imshow("Live Multi-Modal Security Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
