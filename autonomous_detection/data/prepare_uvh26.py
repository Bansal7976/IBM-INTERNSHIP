"""Convert UVH-26 (IISc Urban Vision Hackathon Dataset) -> YOLO format.

WHY THIS EXISTS
----------------
Found while researching how other projects handle the exact domain gap this
project ran into (see KRISH_HANDOVER.md PLAN C): UVH-26 (Nov 2025,
arxiv.org/abs/2511.02563, huggingface.co/datasets/iisc-aim/UVH-26) is the
newest and most India-specific detection dataset available —
26,646 real Bengaluru traffic-camera images, 1.8M boxes, 283K-316K
consensus-labeled ground truth boxes across 14 India-specific vehicle
classes with body-TYPE granularity (Hatchback/Sedan/SUV/MUV are separate
classes, not lumped into one generic "car" like IDD/DriveIndia). The
authors fine-tuned YOLOv11/DAMO-YOLO/RT-DETRv2 on it and measured up to
**31.5% mAP@50:95 improvement over COCO-trained baselines** on Indian
traffic — the same class of gain PLAN C is trying to achieve here, now with
a third, complementary, YOLOv11-native data source to combine with IDD +
DriveIndia (data/prepare_idd.py, data/prepare_driveindia.py).

Label quality note: UVH-26 ships TWO consensus-annotation versions from its
crowdsourced (565 student) hackathon — MV (majority voting) and ST (STAPLE
probabilistic consensus). Either is fine; MV is the simpler of the two and
is the default here.

DOWNLOAD
---------
    https://huggingface.co/datasets/iisc-aim/UVH-26
    (CC BY 4.0 — no registration wall, HF account only)

Extract so you have:
    data/UVH26/
      UVH-26-Train/images/000/... , UVH-26-MV-Train.json
      UVH-26-Val/images/000/...   , UVH-26-MV-Val.json

Usage:
    python data/prepare_uvh26.py --data_root data/UVH26 --out data/uvh26_yolo
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from tqdm import tqdm

# UVH-26 class name (normalized) -> unified 15-class id (see prepare_merged.py).
# Body-type granularity (Hatchback/Sedan/SUV/MUV/etc.) is deliberately
# collapsed into our coarser taxonomy -- this project's overtaking/collision
# decisions don't need to distinguish a sedan from a hatchback, just "car"-
# like road behavior vs. "van"-like (bigger, different blind spots).
NAME_TO_UNIFIED = {
    "hatchback": 0, "sedan": 0, "suv": 0,                # -> car
    "muv": 3, "van": 3, "lcv": 3, "tempo-traveller": 3,  # -> van (bigger
                                                          #   people/cargo carriers)
    "bus": 2, "mini-bus": 2,                             # -> bus
    "truck": 1,                                          # -> truck
    "three-wheeler": 11,                                 # -> autorickshaw (exact match)
    "two-wheeler": 6,                                    # -> motorcycle
    "bicycle": 5,                                        # -> cyclist
    "other": 14,                                         # -> vehicle_fallback
}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def convert_split(coco_json_path: Path, images_root: Path, out_dir: Path
                  ) -> dict:
    with open(coco_json_path) as f:
        coco = json.load(f)

    id_to_unified = {}
    unmapped = []
    for cat in coco["categories"]:
        mapped = NAME_TO_UNIFIED.get(_norm(cat["name"]))
        if mapped is not None:
            id_to_unified[cat["id"]] = mapped
        else:
            unmapped.append(cat["name"])
    if unmapped:
        print(f"[prepare_uvh26] Unmapped categories (skipped): {unmapped}")

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "boxes": 0, "skipped_boxes": 0}
    for image_id, anns in tqdm(anns_by_image.items(), desc=coco_json_path.stem):
        img_info = images_by_id.get(image_id)
        if img_info is None:
            continue
        img_path = images_root / img_info["file_name"]
        if not img_path.exists():
            continue
        w, h = img_info["width"], img_info["height"]
        if w <= 0 or h <= 0:
            continue

        lines = []
        for ann in anns:
            cls = id_to_unified.get(ann["category_id"])
            if cls is None:
                stats["skipped_boxes"] += 1
                continue
            x, y, bw, bh = ann["bbox"]   # COCO: top-left x,y + absolute width/height
            if bw < 2 or bh < 2:
                stats["skipped_boxes"] += 1
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            stats["boxes"] += 1

        if not lines:
            continue

        stem = f"uvh26_{image_id}"
        dst_img = out_images / f"{stem}{img_path.suffix}"
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines))
        stats["frames"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/UVH26")
    ap.add_argument("--out", default="data/uvh26_yolo")
    ap.add_argument("--annotation_version", choices=["MV", "ST"], default="MV",
                    help="MV=majority voting (default, simpler), "
                         "ST=STAPLE probabilistic consensus")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    if not root.exists():
        raise SystemExit(
            f"{root} not found. Download from "
            "https://huggingface.co/datasets/iisc-aim/UVH-26 (see this file's docstring).")

    splits = [
        ("train", root / "UVH-26-Train" / f"UVH-26-{args.annotation_version}-Train.json",
         root / "UVH-26-Train" / "images"),
        ("val", root / "UVH-26-Val" / f"UVH-26-{args.annotation_version}-Val.json",
         root / "UVH-26-Val" / "images"),
    ]

    for split, json_path, images_root in splits:
        if not json_path.exists():
            print(f"[skip] {json_path} not found")
            continue
        stats = convert_split(json_path, images_root, out / split)
        print(f"\n{split}: {stats['frames']} frames, {stats['boxes']} boxes, "
              f"{stats['skipped_boxes']} boxes skipped")

    print("\nMerge into the unified dataset with: python data/prepare_merged.py "
          f"(add --uvh26 {out})")


if __name__ == "__main__":
    main()
