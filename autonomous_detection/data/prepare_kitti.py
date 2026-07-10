"""
KITTI dataset download and preparation script.

Usage:
    python data/prepare_kitti.py --data_root ./data/kitti --download

KITTI Object Detection Download (manual step required):
  Register at: https://www.cvlibs.net/datasets/kitti/eval_object.php
  Download:
    - Left color images (12 GB)
    - Training labels (5 MB)
    - Camera calibration matrices (16 MB)
"""

import os
import argparse
import zipfile
import shutil
from pathlib import Path
import urllib.request


KITTI_URLS = {
    # These require registration; left here as reference
    'images_train': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
    'labels':       'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
    'calib':        'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
}

KITTI_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']


def convert_kitti_to_yolo(data_root: str, output_root: str, classes: list = None):
    """Convert KITTI annotation format to YOLO format for Ultralytics training."""
    classes = classes or KITTI_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    data_root = Path(data_root)
    output_root = Path(output_root)

    for split_name, img_fraction in [('train', (0, 0.8)), ('val', (0.8, 1.0))]:
        (output_root / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

    label_dir = data_root / 'training' / 'label_2'
    image_dir = data_root / 'training' / 'image_2'

    all_ids = sorted([p.stem for p in image_dir.glob('*.png')])
    n = len(all_ids)
    splits = {
        'train': all_ids[:int(n * 0.8)],
        'val':   all_ids[int(n * 0.8):],
    }

    for split_name, ids in splits.items():
        converted = 0
        skipped = 0
        for sample_id in ids:
            img_src = image_dir / f'{sample_id}.png'
            lbl_src = label_dir / f'{sample_id}.txt'

            if not img_src.exists():
                skipped += 1
                continue

            # Copy image
            img_dst = output_root / 'images' / split_name / f'{sample_id}.png'
            shutil.copy2(img_src, img_dst)

            # Convert labels
            import cv2
            img = cv2.imread(str(img_src))
            h, w = img.shape[:2]

            yolo_lines = []
            if lbl_src.exists():
                with open(lbl_src) as f:
                    for line in f:
                        parts = line.strip().split()
                        cls_name = parts[0]
                        if cls_name not in class_to_idx:
                            continue
                        x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        yolo_lines.append(f'{class_to_idx[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

            lbl_dst = output_root / 'labels' / split_name / f'{sample_id}.txt'
            with open(lbl_dst, 'w') as f:
                f.write('\n'.join(yolo_lines))

            converted += 1

        print(f'{split_name}: converted {converted}, skipped {skipped}')

    # Write dataset YAML for Ultralytics
    yaml_content = f"""path: {output_root.absolute()}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    with open(output_root / 'kitti.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML saved to: {output_root / 'kitti.yaml'}")
    print("Use this with: yolo train data=kitti.yaml model=yolo11m.pt epochs=50")


def verify_kitti_structure(data_root: str) -> bool:
    root = Path(data_root)
    required = [
        root / 'training' / 'image_2',
        root / 'training' / 'label_2',
    ]
    ok = all(p.exists() for p in required)
    if not ok:
        print("Missing KITTI structure. Expected:")
        for p in required:
            status = "OK" if p.exists() else "MISSING"
            print(f"  [{status}] {p}")
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare KITTI dataset')
    parser.add_argument('--data_root', required=True, help='Path to raw KITTI data')
    parser.add_argument('--output_root', default=None, help='Output path for YOLO format (default: data_root/../kitti_yolo)')
    parser.add_argument('--convert_yolo', action='store_true', help='Convert to YOLO format')
    parser.add_argument('--classes', nargs='+', default=None, help='Classes to keep (default: all)')
    args = parser.parse_args()

    output = args.output_root or str(Path(args.data_root).parent / 'kitti_yolo')

    if not verify_kitti_structure(args.data_root):
        print("\nPlease download KITTI first:")
        print("  1. Register at https://www.cvlibs.net/datasets/kitti/eval_object.php")
        print("  2. Download data_object_image_2.zip, data_object_label_2.zip, data_object_calib.zip")
        print(f"  3. Extract all to: {args.data_root}")
        exit(1)

    if args.convert_yolo:
        print(f"Converting KITTI → YOLO format → {output}")
        convert_kitti_to_yolo(args.data_root, output, args.classes)
