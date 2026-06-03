# Autonomous Object & Vehicle Detection

SOTA pipeline for 2D/3D vehicle detection + tracking, supporting HPC training.
Uses the best open-source models (YOLOv11, BEVFormer, BEVFusion, Sparse4D v3).

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                                 │
│  Camera images  │  LiDAR point cloud  │  Multi-camera video     │
└────────┬────────┴──────────┬──────────┴────────────┬────────────┘
         │                  │                        │
┌────────▼────────┐  ┌──────▼──────────┐  ┌────────▼────────────┐
│   2D DETECTION  │  │ 3D BEV DETECTION│  │  DETECTION+TRACKING │
│                 │  │                 │  │                      │
│ YOLOv11m/l/x   │  │  BEVFormer      │  │  YOLO11 + ByteTrack  │
│ RT-DETR        │  │  (camera-only)  │  │  BotSort             │
│ Grounding DINO  │  │  BEVFusion      │  │                      │
│ (open-vocab)    │  │  (LiDAR+Cam)   │  │                      │
└────────┬────────┘  │  Sparse4D v3   │  └────────┬────────────┘
         │           │  (temporal)     │           │
         │           └──────┬──────────┘           │
         └──────────────────┼──────────────────────┘
                            │
               ┌────────────▼────────────┐
               │     EVALUATION          │
               │  mAP@0.5, mAP@0.5:0.95 │
               │  NDS (nuScenes)         │
               │  Confusion Matrix       │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │     DEPLOYMENT          │
               │  ONNX / TensorRT        │
               │  CoreML / OpenVINO      │
               └─────────────────────────┘
```

---

## Quick Start

### 1. Setup
```bash
bash scripts/setup_env.sh
```

### 2. Run pre-trained detection (zero setup)
```bash
# Image
python inference/pipeline.py --source your_image.jpg --model yolo11m

# Video with tracking
python inference/pipeline.py --source your_video.mp4 --model yolo11m --track --save

# Webcam
python inference/pipeline.py --source 0 --model yolo11m --track --show

# Open-vocabulary (no training needed)
python inference/pipeline.py \
    --source image.jpg \
    --model grounding_dino \
    --prompt "car . truck . bus . pedestrian . cyclist ."
```

### 3. Prepare data
```bash
# KITTI (download first from https://www.cvlibs.net/datasets/kitti)
python data/prepare_kitti.py --data_root data/kitti --convert_yolo

# nuScenes mini (download first from https://www.nuscenes.org)
python data/prepare_nuscenes.py \
    --data_root data/nuscenes \
    --output_root data/nuscenes_yolo \
    --version v1.0-mini
```

### 4. Train (single GPU)
```bash
python training/train.py --config configs/yolov11/yolov11_kitti.yaml
```

### 5. Train (multi-GPU / HPC)
```bash
# 4 GPUs, single node
torchrun --nproc_per_node=4 training/train_ddp.py \
    --config configs/yolov11/yolov11_kitti.yaml

# HPC cluster (SLURM)
sbatch training/slurm/train_single_node.sh    # 8 GPUs
sbatch training/slurm/train_multi_node.sh     # 32 GPUs
sbatch training/slurm/bevformer_train.sh      # BEVFormer 3D
```

### 6. Evaluate
```bash
python evaluation/evaluate_kitti.py \
    --weights runs/train/exp/weights/best.pt \
    --data_root data/kitti
```

### 7. Export
```bash
# ONNX
python inference/export.py --weights best.pt --format onnx

# TensorRT (NVIDIA, fastest)
python inference/export.py --weights best.pt --format engine --half

# TensorRT INT8 (4x faster, slight accuracy drop)
python inference/export.py --weights best.pt --format engine --int8
```

---

## Model Selection Guide

### 2D Detection (for camera-only, real-time)

| Need | Model | Command |
|------|-------|---------|
| Fastest (edge, Pi, Jetson) | `yolo11n` | `--model yolo11n` |
| Best balance | `yolo11m` | `--model yolo11m` |
| Best accuracy | `yolo11x` | `--model yolo11x` |
| Crowded scenes, no NMS | `rtdetr-l` | `--model rtdetr-l` |
| Open-vocabulary (no retraining) | Grounding DINO 1.5 | `--model grounding_dino` |

### 3D Detection (BEV perception)

| Sensors | Model | Performance |
|---------|-------|-------------|
| Camera only | BEVFormer-Base | NDS=51.7 |
| Camera only (temporal) | Sparse4D v3 | NDS=67.7 |
| LiDAR + Camera | BEVFusion | NDS=72.9 |

---

## Datasets

| Dataset | Classes | Frames | Format | Use |
|---------|---------|--------|--------|-----|
| KITTI | 8 | 7,481 | 2D+3D | Quick training/eval |
| nuScenes mini | 10 | 404 | 3D BEV | BEVFormer testing |
| nuScenes trainval | 10 | 28,130 | 3D BEV | Full 3D training |
| Udacity (Kaggle) | 11 | 15,000 | 2D | Extra 2D data |

---

## Project Structure

```
autonomous_detection/
├── configs/
│   ├── yolov11/yolov11_kitti.yaml      ← YOLO training config
│   ├── yolov11/yolov11_nuscenes.yaml
│   └── bevformer/bevformer_base.py     ← BEVFormer config reference
├── data/
│   ├── datasets.py                     ← Dataset loaders (KITTI, nuScenes, YOLO-format)
│   ├── prepare_kitti.py                ← KITTI → YOLO conversion
│   ├── prepare_nuscenes.py             ← nuScenes → YOLO conversion
│   └── augmentations.py               ← Albumentations pipeline
├── models/
│   ├── detector_2d.py                  ← YOLOv11, RT-DETR, Grounding DINO wrappers
│   ├── detector_3d.py                  ← BEVFormer, BEVFusion, Sparse4D wrappers
│   └── tracker.py                     ← ByteTrack, SimpleIoUTracker
├── training/
│   ├── train.py                        ← Main training entry point
│   ├── train_ddp.py                    ← Distributed training
│   └── slurm/                         ← SLURM job scripts for HPC
│       ├── train_single_node.sh        ← 8 GPU single node
│       ├── train_multi_node.sh         ← 32 GPU multi-node
│       └── bevformer_train.sh          ← BEVFormer HPC job
├── inference/
│   ├── pipeline.py                     ← Full detect+track pipeline
│   ├── visualizer.py                  ← Draw boxes, BEV viz
│   └── export.py                      ← ONNX/TensorRT export
├── evaluation/
│   ├── metrics.py                      ← mAP, confusion matrix
│   └── evaluate_kitti.py              ← KITTI evaluation script
├── notebooks/
│   ├── 01_quickstart_yolov11.ipynb    ← Start here
│   └── 02_bevformer_3d_detection.ipynb ← 3D detection deep dive
├── scripts/
│   ├── setup_env.sh                   ← One-shot environment setup
│   └── download_nuscenes.sh
├── requirements.txt
└── requirements_hpc.txt
```

---

## Learning Roadmap

**Week 1 — 2D Detection Basics**
- Notebook 01: Run YOLOv11, understand outputs
- Train on KITTI, reach >85% mAP@0.5
- Enable ByteTrack tracking on dashcam video
- Export to ONNX, benchmark latency

**Week 2 — Advanced 2D**
- Try RT-DETR vs YOLO11 on crowded scenes
- Grounding DINO for open-vocabulary detection
- Custom augmentations with Albumentations

**Week 3 — 3D Detection (BEV)**
- Notebook 02: Understand BEVFormer architecture
- Study spatial cross-attention and temporal attention
- Prepare nuScenes mini dataset
- Run BEVFormer inference (pre-trained weights)

**Week 4 — HPC Training**
- Submit BEVFormer training job to HPC (SLURM)
- Monitor with TensorBoard / WandB
- Compare camera-only (BEVFormer) vs fusion (BEVFusion)

---

## Key Papers

| Model | Paper | Year | Key Idea |
|-------|-------|------|----------|
| YOLOv11 | arxiv.org/abs/2410.22898 | 2024 | YOLO evolution, 22% fewer params vs v8 |
| YOLOv10 | arxiv.org/abs/2405.14458 | 2024 | NMS-free detection |
| RT-DETR | arxiv.org/abs/2304.08069 | 2023 | Real-time DETR |
| BEVFormer | arxiv.org/abs/2203.17270 | 2022 | Spatial-temporal BEV from cameras |
| BEVFusion | arxiv.org/abs/2205.13542 | 2022 | LiDAR+Camera fusion in BEV |
| Sparse4D v3 | arxiv.org/abs/2311.11722 | 2024 | 4D anchors + temporal tracking |
| Grounding DINO | arxiv.org/abs/2303.05499 | 2023 | Language-guided detection |
| ByteTrack | arxiv.org/abs/2110.06864 | 2021 | Associate every detection |

---

## References / External Repos

- YOLOv11: https://github.com/ultralytics/ultralytics
- YOLOv10: https://github.com/THU-MIG/yolov10
- BEVFormer: https://github.com/fundamentalvision/BEVFormer
- BEVFusion: https://github.com/mit-han-lab/bevfusion
- Sparse4D: https://github.com/linxuewu/Sparse4D
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- MMDetection3D: https://github.com/open-mmlab/mmdetection3d
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
