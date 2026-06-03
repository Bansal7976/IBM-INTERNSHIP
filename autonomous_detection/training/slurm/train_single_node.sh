#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM: Single-node multi-GPU training (e.g., 1x DGX-A100 with 8 GPUs)
# Submit: sbatch training/slurm/train_single_node.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vehicle_det_single
#SBATCH --partition=gpu             # change to your partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # 4 workers per GPU × 8 GPUs
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ─── Environment Setup ────────────────────────────────────────────────────────
module purge
module load cuda/11.8                # match your cluster's CUDA version
module load python/3.10

# Activate virtualenv or conda
source activate autonomous_det       # or: source /path/to/venv/bin/activate

# Project root
PROJECT_DIR="$HOME/autonomous_detection"
cd "$PROJECT_DIR"

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG="configs/yolov11/yolov11_kitti.yaml"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPUs:        $SLURM_GPUS_ON_NODE"
echo "Config:      $CONFIG"
echo "Started at:  $(date)"

# ─── Training ────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    training/train_ddp.py \
    --config "$CONFIG"

echo "Finished at: $(date)"
