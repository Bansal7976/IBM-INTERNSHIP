#!/bin/bash
# JOB B — Lane detection: CLRNet ResNet-101 on CULane
# Submit: sbatch training/slurm/clrnet_culane.sh
#SBATCH --job-name=clrnet_culane
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=logs/clrnet_culane_%j.log

set -euo pipefail
mkdir -p logs

module load cuda/11.8 2>/dev/null || true
source activate autodet 2>/dev/null || conda activate autodet

cd "$SLURM_SUBMIT_DIR"

# One-time setup
if [ ! -d external/CLRNet ]; then
    git clone https://github.com/Turoad/CLRNet.git external/CLRNet
    cd external/CLRNet
    pip install -r requirements.txt
    python setup.py build develop
    cd "$SLURM_SUBMIT_DIR"
fi

if [ ! -d data/culane/driver_23_30frame ]; then
    echo "ERROR: CULane dataset missing — see FINAL_PRODUCT_GUIDE.md §8.2"
    exit 1
fi

cd external/CLRNet
ln -sfn "$SLURM_SUBMIT_DIR/data/culane" data/CULane

# Train
python main.py configs/clrnet/clr_resnet101_culane.py --gpus 0 1 2 3

# Evaluate per-category (Curve + Night numbers go in the report)
python main.py configs/clrnet/clr_resnet101_culane.py --validate \
  --load_from work_dirs/clr/r101_culane/*/ckpt/best.pth \
  2>&1 | tee "$SLURM_SUBMIT_DIR/logs/clrnet_per_category_eval.txt"

# Copy final weights to project weights dir
mkdir -p "$SLURM_SUBMIT_DIR/weights"
cp work_dirs/clr/r101_culane/*/ckpt/best.pth \
   "$SLURM_SUBMIT_DIR/weights/clrnet_r101_culane.pth"
echo "=== CLRNet weights -> weights/clrnet_r101_culane.pth ==="
