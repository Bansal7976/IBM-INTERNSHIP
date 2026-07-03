# Autonomous Detection Pipeline

This repository contains a full pipeline for 2D object detection for autonomous driving scenarios, utilizing YOLOv11 and the ByteTrack algorithm. The system is designed to detect and track key objects in driving scenes (Cars, Pedestrians, Cyclists, etc.) and features an advanced Time-To-Collision (TTC) estimation system to generate ADAS (Advanced Driver Assistance Systems) collision warnings.

## Project Overview

The project was developed and executed on a High-Performance Computing (HPC) cluster utilizing NVIDIA H100 GPUs, executing through the PBS job scheduler. 

### Key Features
- **YOLOv11m Architecture:** State-of-the-art real-time detection, pretrained on COCO and fine-tuned on the KITTI dataset.
- **ByteTrack Tracking:** Fast, association-based multi-object tracking.
- **TTC Collision Warning:** Proximal distance estimation based on bounding box scales to generate `SAFE`, `WARNING`, and `CRITICAL` collision alerts on moving vehicles.
- **HPC Pipeline Ready:** Fully scripted for a PBS cluster environment (data preparation, training, evaluation, inference, and TRT/ONNX export).
- **Edge Deployment Ready:** Supports exporting models to ONNX and TensorRT (`.engine`) formats for optimized inference on NVIDIA hardware.

## Pipeline Execution Details

The project execution consisted of six distinct phases:

1. **Phase 1: Environment Setup**
   - Configured an isolated Conda environment (`auto_det`) on the HPC.
   - Installed Ultralytics, PyTorch (CUDA 12.8), ByteTrack dependencies, and OpenCV.

2. **Phase 2: Dataset Preparation**
   - Downloaded the **KITTI Object Detection** dataset (~12GB) directly to the HPC.
   - Developed custom parsers to convert KITTI annotations (`.txt` files) to YOLO normalized formats (`.txt` YOLO format + `kitti.yaml`).
   - Re-organized files into standard `images/train`, `images/val`, `labels/train`, and `labels/val` structures.

3. **Phase 3: Model Training**
   - **Model:** YOLOv11m (`yolo11m.pt`)
   - **Hardware:** NVIDIA H100 80GB
   - **Epochs:** 50
   - **Batch Size:** 32
   - **Results:** Reached convergence with exceptional metrics. Achieved **91.8% mAP@0.5** and **69.7% mAP@0.5:0.95**. The best weights were saved as `best.pt`.

4. **Phase 4: Evaluation**
   - Evaluated the best model weights on the KITTI validation split.
   - Generated Precision-Recall curves and normalized Confusion Matrices.

5. **Phase 5: ADAS Inference & Tracking**
   - Processed raw dashcam driving footage through the detection pipeline.
   - Implemented real-time Multi-Object Tracking (ByteTrack).
   - Displayed collision warnings using TTC estimation.
   - **Performance:** Reached **>3,300 FPS** inference speeds on the H100 GPU during batch video processing.

6. **Phase 6: Model Export**
   - Exported the PyTorch weights (`best.pt`) to the universal **ONNX format (`best.onnx`)** (approx. 77MB) for flexible deployment on Edge devices or local Windows PCs.

## Repository Structure

```
autonomous_detection/
├── configs/
│   └── yolov11/
│       └── yolov11_kitti.yaml      # YOLO dataset configuration for KITTI
├── data/
│   ├── datasets.py                 # PyTorch/Ultralytics dataset loaders
│   ├── prepare_kitti.py            # Scripts to convert KITTI to YOLO format
│   └── augmentations.py            # Albumentations pipelines
├── evaluation/
│   ├── evaluate_kitti.py           # Evaluation script for mAP & Confusion Matrix
│   └── metrics.py                  # Custom metric tracking components
├── inference/
│   ├── pipeline.py                 # Core tracking & inference logic
│   └── advanced_pipeline.py        # Extended pipeline with TTC estimation and ADAS HUD
├── models/
│   └── build.py                    # Model builder utilities
├── scripts/
│   └── auto_setup.py               # Environment validation utilities
├── main.py                         # Unified entry point for data prep, training, and testing
└── README.md                       # This file
```

## Results Summary

| Metric | Score |
|---|---|
| **mAP@0.5** | **91.8%** |
| **mAP@0.5:0.95** | **69.7%** |
| **Precision** | **91.6%** |
| **Recall** | **86.2%** |
| **Inference Speed (H100)** | **~3,332 FPS** |

The exported ONNX models can be utilized locally without needing an HPC cluster to run the inference scripts on local dashcam files.
