#!/bin/bash
# nuScenes download helper.
# The full dataset requires registration. This downloads the mini split for testing.
#
# Full dataset (~350 GB) requires manual download from:
#   https://www.nuscenes.org/nuscenes#download
#
# Usage: bash scripts/download_nuscenes.sh [output_dir]

OUTPUT_DIR="${1:-data/nuscenes}"
mkdir -p "$OUTPUT_DIR"

echo "=== nuScenes Download ==="
echo ""
echo "STEP 1: Register at https://www.nuscenes.org/sign-up"
echo "STEP 2: Accept license agreement"
echo "STEP 3: Download the following files to $OUTPUT_DIR :"
echo ""
echo "  For quick testing (mini, ~400 MB):"
echo "    v1.0-mini.tgz"
echo ""
echo "  For full training (~350 GB):"
echo "    v1.0-trainval01_blobs.tgz  (through v1.0-trainval10_blobs.tgz)"
echo "    v1.0-trainval_meta.tgz"
echo ""
echo "STEP 4: Extract:"
echo "  tar -xzf v1.0-mini.tgz -C $OUTPUT_DIR"
echo ""
echo "STEP 5: Prepare YOLO format:"
echo "  python data/prepare_nuscenes.py \\"
echo "    --data_root $OUTPUT_DIR \\"
echo "    --output_root data/nuscenes_yolo \\"
echo "    --version v1.0-mini"
echo ""

# Alternative: download mini via wget if available (no auth required for this)
# The mini is publicly available via some mirrors for academic use
echo "Checking for wget..."
if command -v wget &> /dev/null; then
    echo "wget is available. If you have the download URLs from the nuScenes website, use:"
    echo "  wget -P $OUTPUT_DIR <URL>"
fi
