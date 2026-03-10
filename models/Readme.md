# Model Weights

Place all model weight files in **this folder** (`models/`) before running the pipeline.

The pipeline looks for these files at startup:

| File | Size (approx.) | Description |
|------|---------------|-------------|
| `best_mobilenet_gender_model.pth` | ~14 MB | Fine-tuned MobileNetV2 — gender classification |
| `emotion_model.h5` | ~20 MB | CNN trained on FER-2013 — 7-class facial emotion |
| `final_violence_model.zip` | ~340 MB | Fine-tuned VideoMAE — Fight / NonFight classifier |
| `attention_fusion_model.pth` | <1 MB | Learned attention weights for final score fusion |

> **Auto-downloaded** (no action needed):
> - `deploy.prototxt` — OpenCV face detector config (downloaded automatically on first run)
> - `res10_300x300_ssd_iter_140000.caffemodel` — OpenCV ResNet SSD face detector (downloaded automatically on first run)
> - `yolov5m.pt` — YOLOv5m person detector (downloaded automatically by `torch.hub` on first run)

---

## Fallback behaviour when a file is missing

| Missing file | What happens |
|---|---|
| `best_mobilenet_gender_model.pth` | Gender module returns `S_gender = 0.0` and skips |
| `emotion_model.h5` | Emotion module prints a warning and will crash at `model.predict` |
| `final_violence_model.zip` | Violence module falls back to the public `MCG-NJU/videomae-base-finetuned-kinetics` base model |
| `attention_fusion_model.pth` | Fusion falls back to fixed weights: gender=0.30, action=0.50, emotion=0.20 |

---

Download final_violence_model.zip from this link  -  https://drive.google.com/file/d/1tnrUjs5TLbIJHBIl3bHOk6uaCaBXlk5e/view?usp=sharing

## Expected folder layout after setup

```
FYP2026-WomenSafety/
├── pipeline.py
├── requirements.txt
├── README.md
└── models/
    ├── README.md                               ← this file
    ├── best_mobilenet_gender_model.pth
    ├── emotion_model.h5
    ├── final_violence_model.zip
    ├── attention_fusion_model.pth
    ├── deploy.prototxt                         (auto-downloaded)
    └── res10_300x300_ssd_iter_140000.caffemodel (auto-downloaded)
```
