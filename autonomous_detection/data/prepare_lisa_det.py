"""Convert LISA Traffic Light dataset -> YOLO detection format (FALLBACK PATH).

Use this when BDD100K is not available: LISA provides traffic-light BOUNDING
BOXES (day+night), so the detector learns the traffic_light class (id 8 in the
unified map) from LISA instead of BDD100K. States (red/yellow/green) are still
handled by the separate crop classifier (train_aux_classifiers.py).

Download (no registration wall, just a Kaggle account):
    kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data/lisa_tl
    cd data/lisa_tl && unzip lisa-traffic-light-dataset.zip

Usage:
    python data/prepare_lisa_det.py --lisa data/lisa_tl --out data/lisa_yolo
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

TRAFFIC_LIGHT_CLS = 8  # unified 11-class map id


def convert(lisa_root: str, out_dir: str, val_ratio: float = 0.1):
    root, out = Path(lisa_root), Path(out_dir)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect boxes per image across all clip annotation CSVs
    boxes_per_image: dict[Path, list] = {}
    csvs = list(root.rglob("frameAnnotationsBOX.csv"))
    if not csvs:
        raise SystemExit(f"No frameAnnotationsBOX.csv under {root} — check extraction")

    for csv_path in csvs:
        clip_dir = csv_path.parent
        with open(csv_path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                img_name = Path(row["Filename"]).name
                matches = list(clip_dir.rglob(img_name))
                if not matches:
                    continue
                x1 = int(row["Upper left corner X"]); y1 = int(row["Upper left corner Y"])
                x2 = int(row["Lower right corner X"]); y2 = int(row["Lower right corner Y"])
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                boxes_per_image.setdefault(matches[0], []).append((x1, y1, x2, y2))

    images = sorted(boxes_per_image)
    n_val = max(1, int(len(images) * val_ratio))
    val_set = set(images[::max(1, len(images) // n_val)][:n_val])

    n_boxes = 0
    for i, img_path in enumerate(tqdm(images, desc="LISA->YOLO")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        for x1, y1, x2, y2 in boxes_per_image[img_path]:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{TRAFFIC_LIGHT_CLS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1

        split = "val" if img_path in val_set else "train"
        new_name = f"lisa_{i:06d}{img_path.suffix}"
        dst = out / split / "images" / new_name
        if not dst.exists():
            try:
                dst.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst)
        (out / split / "labels" / f"lisa_{i:06d}.txt").write_text("\n".join(lines))

    print(f"\nConverted {len(images)} images, {n_boxes} traffic-light boxes")
    print(f"Merge into the unified dataset with:")
    print(f"  python data/prepare_merged.py  (add --lisa {out})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lisa", default="data/lisa_tl")
    ap.add_argument("--out", default="data/lisa_yolo")
    args = ap.parse_args()
    convert(args.lisa, args.out)
