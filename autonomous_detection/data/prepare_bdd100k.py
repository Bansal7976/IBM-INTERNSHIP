"""Convert BDD100K detection labels -> YOLO format.

BDD100K gives night/rain/fog images + traffic light & sign classes — the
robustness data KITTI lacks.

Download (register at http://bdd-data.berkeley.edu):
    bdd100k_images_100k.zip          -> data/bdd100k/images/100k/{train,val}
    bdd100k_det_20_labels_trainval.zip -> data/bdd100k/labels/det_20/det_{train,val}.json

Usage:
    python data/prepare_bdd100k.py --data_root data/bdd100k --out data/bdd100k_yolo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

# BDD100K category -> unified 11-class id (see prepare_merged.py)
BDD_TO_UNIFIED = {
    "car": 0, "truck": 1, "bus": 2,
    "pedestrian": 4, "other person": 4,
    "rider": 5, "bicycle": 5,
    "motorcycle": 6, "train": 7,
    "traffic light": 8, "traffic sign": 9,
    "other vehicle": 10, "trailer": 1,
}

IMG_W, IMG_H = 1280, 720  # BDD100K fixed resolution


def convert_split(labels_json: Path, images_dir: Path, out_dir: Path) -> dict:
    out_labels = out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    with open(labels_json) as f:
        frames = json.load(f)

    stats = {"frames": 0, "boxes": 0, "skipped_boxes": 0, "attr": {}}
    for frame in tqdm(frames, desc=labels_json.stem):
        name = frame["name"]
        img_path = images_dir / name
        if not img_path.exists():
            continue

        lines = []
        for obj in frame.get("labels", []):
            cls = BDD_TO_UNIFIED.get(obj.get("category", ""))
            box = obj.get("box2d")
            if cls is None or box is None:
                stats["skipped_boxes"] += 1
                continue
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                stats["skipped_boxes"] += 1
                continue
            # YOLO normalized xywh
            cx = (x1 + x2) / 2 / IMG_W
            cy = (y1 + y2) / 2 / IMG_H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / IMG_W:.6f} {bh / IMG_H:.6f}")
            stats["boxes"] += 1

        if not lines:
            continue

        (out_labels / f"{Path(name).stem}.txt").write_text("\n".join(lines))
        link = out_images / name
        if not link.exists():
            try:
                link.symlink_to(img_path.resolve())
            except OSError:  # Windows without admin: copy instead of symlink
                import shutil
                shutil.copy2(img_path, link)
        stats["frames"] += 1

        # Track weather/time-of-day tags for per-condition evaluation later
        attrs = frame.get("attributes", {})
        key = f'{attrs.get("timeofday", "?")}/{attrs.get("weather", "?")}'
        stats["attr"][key] = stats["attr"].get(key, 0) + 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/bdd100k")
    ap.add_argument("--out", default="data/bdd100k_yolo")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)

    for split in ("train", "val"):
        labels_json = root / "labels" / "det_20" / f"det_{split}.json"
        images_dir = root / "images" / "100k" / split
        if not labels_json.exists():
            print(f"[skip] {labels_json} not found")
            continue
        stats = convert_split(labels_json, images_dir, out / split)
        print(f"\n{split}: {stats['frames']} frames, {stats['boxes']} boxes, "
              f"{stats['skipped_boxes']} skipped")
        print("  Conditions:", dict(sorted(stats["attr"].items(),
                                           key=lambda kv: -kv[1])[:8]))

    # Dataset yaml
    yaml_text = f"""# BDD100K in unified 11-class YOLO format
path: {out.resolve()}
train: train/images
val: val/images
names:
  0: car
  1: truck
  2: bus
  3: van
  4: pedestrian
  5: cyclist
  6: motorcycle
  7: tram
  8: traffic_light
  9: traffic_sign
  10: misc
"""
    (out / "bdd100k.yaml").write_text(yaml_text)
    print(f"\nWrote {out / 'bdd100k.yaml'}")


if __name__ == "__main__":
    main()
