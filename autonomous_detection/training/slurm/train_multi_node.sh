#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM: Multi-node multi-GPU training (e.g., 4 nodes × 8 GPUs = 32 GPUs)
# Submit: sbatch training/slurm/train_multi_node.sh
#
# This is for training large models like BEVFormer/BEVFusion on full nuScenes.
# Effective batch size = 4 per GPU × 32 GPUs = 128 total.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vehicle_det_multi
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=72:00:00
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err

# ─── Environment Setup ────────────────────────────────────────────────────────
module purge
module load cuda/11.8
module load python/3.10
source activate autonomous_det

PROJECT_DIR="$HOME/autonomous_detection"
cd "$PROJECT_DIR"

# ─── Distributed Setup ───────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=8   # GPUs per node

export MASTER_ADDR
export MASTER_PORT

echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_NODELIST"
echo "Master addr:  $MASTER_ADDR:$MASTER_PORT"
echo "World size:   $((NNODES * NPROC_PER_NODE))"
echo "Started at:   $(date)"

# ─── Training ────────────────────────────────────────────────────────────────
# For custom PyTorch DDP model:
srun torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    training/train_ddp.py \
    --config configs/yolov11/yolov11_kitti.yaml

# ─── For MMDetection3D (BEVFormer / BEVFusion) ───────────────────────────────
# Uncomment and set correct paths for 3D detection training:
#
# MMDET3D_DIR="$HOME/BEVFormer"   # cloned repo
# cd "$MMDET3D_DIR"
#
# srun torchrun \
#     --nproc_per_node=8 \
#     --nnodes=$NNODES \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     tools/train.py \
#     projects/configs/bevformer/bevformer_base.py \
#     --launcher pytorch \
#     --work-dir "$PROJECT_DIR/work_dirs/bevformer_base"

echo "Finished at: $(date)"
