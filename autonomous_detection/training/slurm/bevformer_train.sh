#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM: BEVFormer training on nuScenes (8 GPUs, ~24h for 24 epochs)
# Requires: cloned BEVFormer repo + prepared nuScenes dataset
#
# Setup:
#   git clone https://github.com/fundamentalvision/BEVFormer.git
#   cd BEVFormer && pip install -r requirements.txt
#   # Prepare nuScenes as per BEVFormer README (create data/nuscenes symlink)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=bevformer_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bevformer_%j.out
#SBATCH --error=logs/bevformer_%j.err

module purge
module load cuda/11.3
module load python/3.8
source activate bevformer_env

BEVFORMER_DIR="$HOME/BEVFormer"
cd "$BEVFORMER_DIR"

echo "Training BEVFormer-Base on nuScenes"
echo "GPUs: 8 | Expected time: ~36h"
echo "Config: projects/configs/bevformer/bevformer_base.py"

torchrun \
    --nproc_per_node=8 \
    --master_port=29600 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --launcher pytorch \
    --work-dir "$HOME/work_dirs/bevformer_base" \
    --cfg-options "train_dataloader.dataset.data_root=$HOME/data/nuscenes"

echo "Done: $(date)"
