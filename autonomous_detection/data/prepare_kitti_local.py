"""
Convert KITTI Object Detection dataset → YOLO format.

KITTI raw structure:
    data/kitti/
        training/
            image_2/     ← left colour images (.png)
            label_2/     ← annotations (.txt)
        testing/
            image_2/

KITTI label format (one object per line):
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

Output (YOLO format):
    data/kitti_yolo/
        images/train/  images/val/
        labels/train/  labels/val/
        kitti.yaml
"""

from __future__ import annotations
import os
import shutil
import random
from pathlib import Path

# ── Class mapping ─────────────────────────────────────────────────────────────

KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(KITTI_CLASSES)}

# Classes to skip entirely
IGNORE_CLASSES = {'DontCare'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def verify_kitti_structure(data_root: str) -> bool:
    """
    Verify the raw KITTI dataset folder structure exists.
    Returns True if valid, False otherwise.
    """
    data_root = Path(data_root)
    required = {
        'training/image_2': data_root / 'training' / 'image_2',
        'training/label_2': data_root / 'training' / 'label_2',
    }
    all_ok = True
    print(f'\nVerifying KITTI structure at: {data_root}')
    for name, path in required.items():
        if path.exists():
            n = len(list(path.iterdir()))
            print(f'  ✓ {name}/ — {n} files')
        else:
            print(f'  ✗ {name}/ — NOT FOUND')
            all_ok = False

    if not all_ok:
        print('\nExpected structure:')
        print('  data/kitti/')
        print('  ├── training/')
        print('  │   ├── image_2/   ← images (.png)')
        print('  │   └── label_2/   ← labels (.txt)')
    return all_ok


def _parse_kitti_label(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Parse one KITTI .txt label file → list of YOLO format strings.
    YOLO format: 'class_id cx cy w h' (all normalised 0-1)
    """
    yolo_lines = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_name = parts[0]

            if cls_name in IGNORE_CLASSES or cls_name not in CLASS_TO_IDX:
                continue

            cls_id = CLASS_TO_IDX[cls_name]

            # KITTI bbox: x1 y1 x2 y2 (pixels, 0-indexed)
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])

            # Convert to YOLO normalised cx cy w h
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            if bw <= 0 or bh <= 0:
                continue  # skip degenerate boxes

            yolo_lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    return yolo_lines


# ── Main conversion ───────────────────────────────────────────────────────────

def convert_kitti_to_yolo(
    data_root: str,
    output_root: str,
    val_split: float = 0.2,
    seed: int = 42,
) -> str:
    """
    Convert KITTI raw dataset to YOLO format.

    Args:
        data_root:   path to raw KITTI folder (contains training/)
        output_root: where to write YOLO-format dataset
        val_split:   fraction of images for validation (default 20%)
        seed:        random seed for reproducible train/val split

    Returns:
        Path to generated kitti.yaml
    """
    from PIL import Image as PILImage

    data_root   = Path(data_root)
    output_root = Path(output_root)

    img_src = data_root / 'training' / 'image_2'
    lbl_src = data_root / 'training' / 'label_2'

    if not img_src.exists():
        raise FileNotFoundError(f'Images not found: {img_src}')
    if not lbl_src.exists():
        raise FileNotFoundError(f'Labels not found: {lbl_src}')

    # Collect all images
    all_images = sorted(img_src.glob('*.png'))
    if not all_images:
        raise RuntimeError(f'No .png images found in {img_src}')

    print(f'\nTotal KITTI images found: {len(all_images)}')

    # Train / val split
    random.seed(seed)
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    n_val = max(1, int(len(all_images) * val_split))
    val_set = set(indices[:n_val])

    # Create output directory structure
    for split in ('train', 'val'):
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    train_count = val_count = skip_count = 0
    class_counts = {cls: 0 for cls in KITTI_CLASSES}

    print('Converting...')
    for i, img_path in enumerate(all_images):
        stem = img_path.stem
        split = 'val' if i in val_set else 'train'

        lbl_path = lbl_src / f'{stem}.txt'
        if not lbl_path.exists():
            skip_count += 1
            continue

        # Get image dimensions (needed for normalisation)
        try:
            with PILImage.open(img_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            skip_count += 1
            continue

        yolo_lines = _parse_kitti_label(lbl_path, img_w, img_h)

        # Copy image
        dst_img = output_root / 'images' / split / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write YOLO label
        dst_lbl = output_root / 'labels' / split / f'{stem}.txt'
        with open(dst_lbl, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Count classes
        for line in yolo_lines:
            cls_id = int(line.split()[0])
            class_counts[KITTI_CLASSES[cls_id]] += 1

        if split == 'train':
            train_count += 1
        else:
            val_count += 1

        if (i + 1) % 500 == 0:
            print(f'  Processed {i+1}/{len(all_images)}...')

    print(f'\nConversion complete:')
    print(f'  Train : {train_count} images')
    print(f'  Val   : {val_count} images')
    print(f'  Skipped: {skip_count} images')
    print(f'\nClass distribution:')
    for cls, count in class_counts.items():
        print(f'  {cls:<20}: {count:>6}')

    # Write kitti.yaml
    yaml_path = output_root / 'kitti.yaml'
    yaml_content = (
        f"# KITTI Object Detection — YOLO format\n"
        f"# Generated by prepare_kitti.py\n"
        f"\n"
        f"path: {output_root.absolute()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(KITTI_CLASSES)}\n"
        f"names: {KITTI_CLASSES}\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f'\n✓ kitti.yaml written: {yaml_path}')
    print(f'✓ Dataset ready for training!')
    return str(yaml_path)
