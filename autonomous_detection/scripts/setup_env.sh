#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Environment setup script for autonomous detection project.
# Run once on your machine or HPC login node.
#
# Usage: bash scripts/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on error

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="autonomous_det"
PYTHON_VERSION="3.10"

echo "=== Autonomous Detection Environment Setup ==="
echo "Project: $PROJECT_DIR"
echo "Env:     $ENV_NAME"

# ─── 1. Create conda environment ─────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
source activate "$ENV_NAME"

# ─── 2. PyTorch (CUDA 11.8) ──────────────────────────────────────────────────
echo ""
echo "=== Installing PyTorch (CUDA 11.8) ==="
# For CUDA 12.1: change cu118 → cu121
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# ─── 3. Core dependencies ─────────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
cd "$PROJECT_DIR"
pip install -r requirements.txt

# ─── 4. MMDetection3D (for BEVFormer / BEVFusion) ────────────────────────────
echo ""
echo "=== Installing MMDetection3D ==="
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.4.0"
mim install "mmdet3d>=1.3.0,<1.5.0"

# ─── 5. Clone model repos ────────────────────────────────────────────────────
echo ""
echo "=== Cloning model repositories ==="
mkdir -p "$PROJECT_DIR/external"

# BEVFormer
if [ ! -d "$PROJECT_DIR/external/BEVFormer" ]; then
    git clone https://github.com/fundamentalvision/BEVFormer.git \
        "$PROJECT_DIR/external/BEVFormer"
    cd "$PROJECT_DIR/external/BEVFormer"
    pip install -r requirements.txt
    cd "$PROJECT_DIR"
fi

# BEVFusion (MIT HAN Lab)
if [ ! -d "$PROJECT_DIR/external/bevfusion" ]; then
    git clone https://github.com/mit-han-lab/bevfusion.git \
        "$PROJECT_DIR/external/bevfusion"
    cd "$PROJECT_DIR/external/bevfusion"
    pip install -r requirements.txt
    python setup.py develop
    cd "$PROJECT_DIR"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download KITTI:    bash scripts/download_kitti.sh"
echo "  2. Download nuScenes: bash scripts/download_nuscenes.sh"
echo "  3. Prepare data:      python data/prepare_kitti.py --data_root data/kitti --convert_yolo"
echo "  4. Train:             python training/train.py --config configs/yolov11/yolov11_kitti.yaml"
echo "  5. Infer:             python inference/pipeline.py --source video.mp4 --model yolo11m --track"
