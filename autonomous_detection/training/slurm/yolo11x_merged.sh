#!/bin/bash
# JOB A — Final detector: YOLOv11x on merged KITTI+BDD100K @ 1280px
# Submit: sbatch training/slurm/yolo11x_merged.sh
#SBATCH --job-name=yolo11x_merged
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/yolo11x_merged_%j.log

set -euo pipefail
mkdir -p logs

module load cuda/11.8 2>/dev/null || true
source activate autodet 2>/dev/null || conda activate autodet

cd "$SLURM_SUBMIT_DIR"

# Prerequisite check
if [ ! -f data/merged_yolo/merged.yaml ]; then
    echo "ERROR: run 'python data/prepare_merged.py' first"
    exit 1
fi

yolo detect train \
  model=yolo11x.pt \
  data=data/merged_yolo/merged.yaml \
  imgsz=1280 \
  epochs=300 \
  batch=128 \
  device=0,1,2,3,4,5,6,7 \
  optimizer=AdamW \
  lr0=0.0005 \
  cos_lr=True \
  warmup_epochs=5 \
  mosaic=1.0 \
  mixup=0.15 \
  copy_paste=0.3 \
  hsv_v=0.6 \
  degrees=5.0 \
  perspective=0.0005 \
  close_mosaic=20 \
  patience=50 \
  save_period=25 \
  workers=32 \
  project=runs/final \
  name=yolo11x_merged

echo "=== TRAINING COMPLETE ==="
yolo detect val \
  model=runs/final/yolo11x_merged/weights/best.pt \
  data=data/merged_yolo/merged.yaml imgsz=1280 batch=64 device=0
