# Autonomous ADAS Detection Pipeline

An end-to-end **Advanced Driver Assistance System (ADAS)** pipeline built on top of NVIDIA H100 GPUs using the PBS job scheduler. The system performs real-time object detection, lane detection, multi-object tracking, monocular depth estimation, Time-To-Collision (TTC) collision alerts, night enhancement, and overtaking decision fusion — all in a single unified inference pipeline.

---

## Project Overview

Developed as part of an IBM Internship 2026 project, this system evolved over three phases from a baseline object detector to a full multi-modal ADAS perception stack.

| Phase | Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| Week 1 | YOLOv11m — 24 epochs | 82.34% | 61.47% |
| Week 2 | YOLOv11m — 500 epochs | 91.71% | 69.31% |
| **Phase 3 (Final)** | **YOLOv11x — 300 epochs** | **95.42%** | **79.80%** |

---

## Full ADAS Pipeline Architecture

```
[Input Frame]
      │
      ▼
[Stage 0]  Night Enhancement → Zero-DCE++ / CLAHE (DAY / NIGHT_LIT / NIGHT_UNLIT)
      │
      ▼
[Stage 1]  YOLOv11x Detection (11 classes, 640px) → Raw detections
      │         └─► Traffic Light State Classifier (ResNet-18 / HSV fallback)
      │
[Stage 2]  CLRNet R101 Lane Detection → Polylines + Ego-Lane
      │         └─► Lane Type Classifier → solid / dashed / double_solid
      │
[Stage 3]  ByteTrack Multi-Object Tracker → Tracks with velocity
      │
[Stage 4]  Depth Anything V2 (ViT-S) → Metric depth map → TTC alerts
      │           BRAKE (TTC < 1.5s)  |  WARNING (TTC < 3.0s)
      │
[Stage 5]  Overtaking Analyzer → 5-Rule Safety Fusion → POSSIBLE / NOT POSSIBLE
      │
      ▼
[Output]  Annotated MP4 + decisions.jsonl (per-frame ADAS log)
```

---

## Key Features

- **YOLOv11x Architecture:** Largest YOLOv11 model (57M params), fine-tuned on 11-class unified KITTI dataset. Achieves **95.42% mAP@0.5**.
- **CLRNet Lane Detection:** Cross-Layer Refinement Network with ResNet-101 backbone, pre-trained on CULane (80.13 F1@50).
- **Depth Anything V2:** Metric monocular depth estimation (ViT-S encoder) for real-world TTC in metres.
- **Custom ByteTrack:** Self-contained two-stage IoU tracker with exponentially-smoothed velocity — no external package required.
- **Night Enhancement:** Zero-DCE++ neural curve estimator with CLAHE fallback for low-light scenes.
- **Overtaking Decision:** Rule-based fusion of lane type, curvature, oncoming traffic, depth gap, and lighting conditions.
- **HPC Pipeline Ready:** Fully scripted for PBS cluster (data prep → training → inference → export).
- **ONNX Export Ready:** Models exportable to ONNX for deployment on edge devices or local PCs.

---

## Final Results (Phase 3 — YOLOv11x)

| Metric | Score |
|---|---|
| **mAP@0.5** | **95.42%** |
| **mAP@0.5:0.95** | **79.80%** |
| **Precision** | **95.12%** |
| **Recall** | **92.65%** |
| **Best Epoch** | 266 / 300 |
| **Inference Speed (H100)** | ~3,300+ FPS |
| **Video frames processed** | 3,604 frames |
| **Output video size** | 27 MB |

---

## Repository Structure

```
autonomous_detection/
├── configs/
│   └── yolov11/
│       └── yolov11_kitti.yaml          # YOLO dataset config for KITTI
├── data/
│   ├── datasets.py                     # PyTorch dataset loaders (KITTI, nuScenes, YOLO)
│   ├── prepare_kitti.py                # KITTI → YOLO format converter
│   ├── prepare_bdd100k.py              # BDD100K → YOLO format converter
│   ├── prepare_lisa_det.py             # LISA Traffic Light → YOLO converter
│   ├── prepare_merged.py               # Multi-dataset merger (11-class unified taxonomy)
│   └── augmentations.py               # Albumentations augmentation pipelines
├── evaluation/
│   ├── evaluate_kitti.py               # KITTI evaluation (mAP, confusion matrix)
│   ├── evaluate_final.py               # Full 5-benchmark evaluation suite
│   └── metrics.py                      # Custom metric tracking
├── inference/
│   ├── adas_final.py                   # ★ Master ADAS pipeline orchestrator
│   ├── collision.py                    # TTC collision detection via depth
│   ├── night_enhance.py               # Zero-DCE++ night enhancement
│   ├── overtaking.py                  # 5-rule overtaking decision fusion
│   ├── tracker.py                     # Custom ByteTrack implementation
│   ├── visualizer.py                  # HUD + BEV visualization utilities
│   ├── pipeline.py                    # Core detection pipeline
│   └── advanced_pipeline.py           # Extended ADAS pipeline
├── models/
│   ├── lane_detector.py               # CLRNet / UFLDv2 lane detector wrappers
│   ├── aux_classifiers.py             # Traffic light + lane type ResNet-18 classifiers
│   ├── detector_2d.py                 # 2D detection model builder
│   └── tracker.py                     # Tracker model wrapper
├── training/
│   ├── train.py                       # Single-GPU training entry point
│   ├── train_aux_classifiers.py       # Auxiliary classifier training
│   ├── train_ddp.py                   # Multi-GPU DDP training
│   └── pbs/                           # PBS job scripts for HPC cluster
│       ├── yolo11x_merged.pbs         # Job A: Main detector (8 GPU)
│       ├── clrnet_culane.pbs          # Job B: CLRNet lane detection (4 GPU)
│       └── aux_and_eval.pbs           # Job C+D+F: Classifiers + evaluation (1 GPU)
├── main.py                            # Unified CLI entry point
├── FINAL_PRODUCT_GUIDE.md             # Phase 3 product design guide
├── KRISH_HANDOVER.md                  # Step-by-step HPC execution guide
├── METHODOLOGY_REPORT.md             # Full architecture & methodology report
├── WEEK2_RESULTS.md                   # Week 2 training results documentation
└── README.md                          # This file
```

---

## Pipeline Execution Summary

### Phase 1–2: Environment & Dataset
- Configured `auto_det` conda environment on HPC (PyTorch 2.8, CUDA 12.8)
- Downloaded and converted KITTI dataset (~12GB, 7,481 images)
- Remapped 8-class KITTI → 11-class unified taxonomy using `prepare_merged.py`

### Phase 3: YOLOv11m Baseline (Week 1–2)
- Trained YOLOv11m for 50 → 500 epochs on KITTI
- Achieved **91.71% mAP@0.5** with 94.71% precision

### Phase 4: YOLOv11x Upgrade (Phase 3)
- Upgraded to YOLOv11x (57M params) on merged 11-class KITTI dataset
- Trained 300 epochs with AdamW + cosine LR + early stopping (patience=50)
- Achieved **95.42% mAP@0.5** — best epoch at 266

### Phase 5: Full ADAS Pipeline
- Downloaded CLRNet R101 pre-trained weights (CULane, 80.13 F1@50)
- Downloaded Depth Anything V2 ViT-S metric weights (AbsRel < 0.12)
- Ran `adas_final.py` on dashcam video: **3,604 frames** processed
- Output: annotated video + per-frame `decisions.jsonl` log

---

## Quick Start (Inference on Your Own Video)

```bash
# On HPC (after downloading weights)
python inference/adas_final.py \
  --source  your_video.mp4 \
  --weights runs/final/yolo11x_merged-2/weights/best.pt \
  --save    output.mp4 \
  --log     decisions.jsonl
```

## Model Weights

| Model | File | Purpose |
|---|---|---|
| YOLOv11x (KITTI 11-class) | `weights/yolo11x_best.pt` | Main object detector |
| CLRNet R101 | `weights/clrnet_r101_culane.pth` | Lane detection |
| Depth Anything V2 | `weights/depth_anything_v2_metric_vkitti_vits.pth` | Metric depth / TTC |
