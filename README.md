# A Unified Risk Scoring Framework for Detecting Violence against Women through the Integration of Gender Composition, Behavioral, and Emotional Signals

**A Unified Risk Scoring Framework for Detecting
Violence against Women through the Integration of
Gender Composition, Behavioral, and Emotional
Signals** is a real-time,  Tri-Stream Fusion video analysis pipeline for detecting threats against women in surveillance footage. It fuses three independent AI modules — gender & proximity risk, violence detection, and emotion distress analysis — into a single final threat score using a learned attention-based fusion model.

> Works in **Google Colab** and from a **local terminal** with no code changes.

---

## Overview

Given a surveillance video, CARE-W runs three parallel analysis streams and fuses them into one risk score:

| Module | Model | What it detects |
|--------|-------|-----------------|
| **M1 — Gender & Proximity** | MobileNetV2 + YOLOv5m + Visual Tracker | Isolated women, male proximity clustering |
| **M2 — Violence Detection** | Fine-tuned VideoMAE + X-CLIP (zero-shot) | Physical violence, suspicious actions |
| **M3 — Emotion Distress** | CNN (7-class) + OpenCV DNN face detector | Fear, anger, distress in facial expressions |
| **Fusion** | Attention Fusion (learned weights) | Weighted combination → S_final ∈ [0, 1] |

**Final alert levels:**

| S_final | Alert |
|---------|-------|
| > 0.75 | `CRITICAL THREAT DETECTED` |
| > 0.40 | `SUSPICIOUS ACTIVITY DETECTED` |
| ≤ 0.40 | `SAFE` |

---

## Architecture

```
Input Video
    │
    ├──► [M1] YOLOv5m Person Detector
    │         └─► MobileNetV2 Gender Classifier
    │             └─► Visual Tracker (IoU + colour histogram)
    │                 └─► DynamicSafetyLogic (proximity bubble)
    │                     └─► S_gender ∈ [0,1]
    │
    ├──► [M2] VideoMAE  (16-frame sliding window, Fight/NonFight)
    │         └─► X-CLIP zero-shot (12 action labels)
    │             └─► Combined risk → S_action ∈ [0,1]
    │
    └──► [M3] OpenCV DNN Face Detector (ResNet SSD)
              └─► CNN Emotion Classifier (7 classes, FER-2013)
                  └─► Temporal deviation scoring → S_emotion ∈ [0,1]

  [S_gender, S_action, S_emotion]
              │
              ▼
      AttentionFusion  (learned Linear(3→3) + softmax)
              │
              ▼
          S_final ∈ [0,1]  ──►  Alert Level
```

---

## Repository Layout

```
FYP2026-WomenSafety/
├── pipeline.py          ← full pipeline (runs in Colab and terminal)
├── requirements.txt
├── README.md
└── models/
    ├── README.md        ← model download instructions
    ├── best_mobilenet_gender_model.pth
    ├── emotion_model.h5
    ├── final_violence_model.zip
    └── attention_fusion_model.pth
```

---

## Setup

### 1 — Clone the repo

```bash
git clone https://github.com/PRANEETHAPADALA/FYP2026-WomenSafety
```

### 2 — Add model weights

Download your trained model files and place them inside the `models/` folder.
See [models/README.md](models/README.md) for the exact file names and fallback behaviour.

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended.** The pipeline auto-detects CUDA; it falls back to CPU if unavailable.
> Python 3.9+ required.

---

## Running the Pipeline

### Option A — Google Colab

1. Upload the entire folder to your Google Drive (or clone it directly in Colab).
2. Open a new Colab notebook and run:

```python
# Mount Drive (if using Drive)
from google.drive import drive
drive.mount('/content/drive')

# Add the repo to the Python path
import sys
sys.path.insert(0, '/content/drive/MyDrive/CARE-W')   # adjust path if needed

# Run — Colab will prompt you to upload a video via a file picker
%run pipeline.py
```

Or in a single cell:

```python
import subprocess
subprocess.run(["python", "/content/drive/MyDrive/CARE-W/pipeline.py"])
```

> When running in Colab, the script detects the environment automatically.
> It will show a **file upload button** to select your video, and will **auto-download** each annotated output video when done.

---

### Option B — Local Terminal

```bash
python pipeline.py --video path/to/your/video.mp4
```

**Examples:**

```bash
# Analyse a specific file
python pipeline.py --video surveillance_clip.avi

# Use an absolute path
python pipeline.py --video "C:/Users/you/Videos/test.mp4"
```

Output annotated videos are saved to the `outputs/` folder that is created automatically next to `pipeline.py`.

---

## Output

After the pipeline finishes you will see a report like:

```
==================================================
 FINAL INTEGRATED SECURITY REPORT
==================================================
Video Source        : test_clip.mp4
Fusion Method       : Attention-Based Fusion
--------------------------------------------------
S_gender  (Safety)  : 0.7821
S_action  (Violence): 0.6540
S_emotion (Distress): 0.4312
--------------------------------------------------
Attention Weights:
  Gender:  0.412
  Action:  0.381
  Emotion: 0.207
--------------------------------------------------
OVERALL THREAT SCORE (S_final): 0.6803
--------------------------------------------------
>>> SYSTEM ALERT: SUSPICIOUS ACTIVITY
```

Three annotated output videos are saved (one per module):

| File | Contents |
|------|---------|
| `outputs/final_gender_<name>.mp4` | Bounding boxes, gender labels, per-frame scene risk |
| `outputs/final_violence_<name>.mp4` | Status overlay, risk %, detected action label |
| `outputs/final_emotion_<name>.mp4` | Face boxes, emotion label, per-face risk score |

---

## Module Details

<details>
<summary><b>M1 — Gender & Proximity Risk</b></summary>

- YOLOv5m detects all persons in each frame (class 0, conf ≥ 0.40).
- A fine-tuned MobileNetV2 classifies each crop as Female / Male.
- `VisualTracker` links detections across frames using IoU + HSV colour histogram matching (Hungarian algorithm).
- `DynamicSafetyLogic` computes per-female risk inside a **250 px proximity bubble**, with bystander damping: `1 / (1 + 0.03 × crowd)`.
- The module score `S_gender` is a weighted combination of average risk, peak risk, and high-risk frame duration.

</details>

<details>
<summary><b>M2 — Violence Detection</b></summary>

- A 16-frame sliding window (stride = 5) is scored by the fine-tuned **VideoMAE** model (Fight / NonFight).
- Simultaneously, 8 sampled frames are scored against 12 action text prompts by **X-CLIP** (zero-shot).
- The module score `S_action` is the mean of the top-10% highest-risk windows.
- The decision threshold adapts dynamically to the running average noise level.

</details>

<details>
<summary><b>M3 — Emotion Distress Analysis</b></summary>

- OpenCV ResNet SSD detects faces (conf ≥ 0.30) with temporal smoothing across 5 frames.
- Each 48×48 greyscale face crop is classified into 7 emotions by a lightweight CNN.
- Risk is the **deviation** of negative-emotion mass (angry, disgust, fear, sad, surprise) from a personal exponential moving average baseline, smoothed with EMA (α = 0.3).
- The module score `S_emotion` is a weighted sum of average, peak, and duration of high-risk frames.

</details>

<details>
<summary><b>Attention-Based Fusion</b></summary>

A small learned module (`Linear(3→3)` + softmax) assigns dynamic weights to `[S_gender, S_action, S_emotion]` depending on the actual risk values, producing a final weighted score. Falls back to fixed weights (0.30 / 0.50 / 0.20) if `attention_fusion_model.pth` is missing.

</details>

---

## Related Work

- [YOLOv5](https://github.com/ultralytics/yolov5) — real-time person detection
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) — masked video autoencoder
- [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP) — zero-shot video-language model
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — facial emotion recognition dataset

---

## Citation

If you use this in your research, please cite:

```bibtex
@misc{carew2026,
  title   = {A Unified Risk Scoring Framework for Detecting
Violence against Women through the Integration of
Gender Composition, Behavioral, and Emotional
Signals},
  author  = {Ganesh Naik, Dr. Anusha Jayasimhan, Vijaya Lakshmi A, Hishitha K, Padala Praneetha},
  year    = {2026},
  note    = {Final Year Project 2026, SSN College of Engineering}
}
```

---

## License

This repository is released for **academic and non-commercial research use only**.
