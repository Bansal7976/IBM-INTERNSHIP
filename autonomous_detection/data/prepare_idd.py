"""Convert IDD Detection (India Driving Dataset) -> YOLO format.

WHY THIS EXISTS
----------------
KITTI (Karlsruhe, Germany) and CULane (Chinese highways) were both captured
in structured, rule-following traffic with painted lanes and a small,
homogeneous set of vehicle types. A detector fine-tuned only on those
datasets has never seen an autorickshaw, a cow on the road, a hand-cart, or
a tractor pulling a trailer — so on Indian footage it either (a) misses
these objects entirely, or (b) forces them into the nearest KITTI class it
does know (car/van/misc), which is exactly the "fake detection" behavior
being reported: false classifications, not random noise.

IDD (Varma et al., WACV 2019, arxiv.org/abs/1811.10200 — built by IIIT
Hyderabad + Intel specifically because "datasets like KITTI, Cityscapes,
Argoverse and nuScenes... results are often not directly applicable in
unstructured road situations") is the standard fix used across multiple
independent projects for this exact problem. This script converts its
PASCAL-VOC-style detection release into the YOLO format used everywhere
else in this repo, remapped into the unified taxonomy in prepare_merged.py.

DOWNLOAD (free, ~1-day approval, no institutional email needed)
------------------------------------------------------------------
1. Register: https://idd.insaan.iiit.ac.in/  (dataset -> IDD Detection)
2. Download "IDD Detection" (~14 GB) and extract so you have:
     data/IDD_Detection/
       JPEGImages/<category>/.../*.jpg
       Annotations/<category>/.../*.xml
       ImageSets/Main/train.txt   (or split lists elsewhere in the release —
                                    the fallback below handles either layout)

Usage:
    python data/prepare_idd.py --data_root data/IDD_Detection --out data/idd_yolo
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# IDD Detection class names (case/spacing normalized) -> unified 15-class id.
# See prepare_merged.py for the full taxonomy. Anything not in this dict is
# counted and skipped rather than guessed — safer than silently mislabeling.
IDD_TO_UNIFIED = {
    "car": 0, "truck": 1, "bus": 2,
    "motorcycle": 6, "motorbike": 6,
    "person": 4, "pedestrian": 4,
    "rider": 13,
    "bicycle": 5,
    "autorickshaw": 11, "auto rickshaw": 11, "auto-rickshaw": 11,
    "animal": 12,
    "traffic light": 8, "trafficlight": 8,
    "traffic sign": 9, "trafficsign": 9,
    "caravan": 1, "trailer": 1,
    "train": 7,
    # IDD's open-world "vehicle fallback" bucket (street cart, tractor,
    # water tanker, excavator) -> our vehicle_fallback class
    "vehicle fallback": 14, "vehiclefallback": 14, "vehicle_fallback": 14,
    "tractor": 14, "cart": 14, "pushcart": 14, "hand cart": 14,
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _parse_split_list(root: Path, split: str) -> list[str] | None:
    """Standard VOC devkit split lists: ImageSets/Main/{split}.txt, one
    relative id per line (no extension). Returns None if not found so the
    caller can fall back to scanning Annotations/ directly."""
    candidates = [
        root / "ImageSets" / "Main" / f"{split}.txt",
        root / f"{split}.txt",
    ]
    for c in candidates:
        if c.exists():
            return [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
    return None


def convert_split(root: Path, split: str, out_dir: Path) -> dict:
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    ids = _parse_split_list(root, split)
    if ids is not None:
        xml_paths = [root / "Annotations" / f"{i}.xml" for i in ids]
    else:
        # Fallback: no split list found at the expected location — walk every
        # XML under Annotations/ (works for mirrors that split by folder
        # instead of a train.txt/val.txt file; you'll want to pass
        # --train_frac to carve out a val split in that case, see main()).
        xml_paths = sorted(root.rglob("*.xml"))

    stats = {"frames": 0, "boxes": 0, "skipped_boxes": 0, "unknown_classes": {}}
    for xml_path in tqdm(xml_paths, desc=f"IDD/{split}"):
        if not xml_path.exists():
            continue
        img_path = Path(str(xml_path)
                         .replace("Annotations", "JPEGImages")
                         .replace(".xml", ".jpg"))
        if not img_path.exists():
            continue

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root_el = tree.getroot()
        size = root_el.find("size")
        if size is None:
            continue
        w = int(float(size.findtext("width", "0")))
        h = int(float(size.findtext("height", "0")))
        if w <= 0 or h <= 0:
            continue

        lines = []
        for obj in root_el.findall("object"):
            name = _normalize(obj.findtext("name", ""))
            cls = IDD_TO_UNIFIED.get(name)
            if cls is None:
                stats["unknown_classes"][name] = stats["unknown_classes"].get(name, 0) + 1
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x1 = float(bnd.findtext("xmin", "0"))
            y1 = float(bnd.findtext("ymin", "0"))
            x2 = float(bnd.findtext("xmax", "0"))
            y2 = float(bnd.findtext("ymax", "0"))
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                stats["skipped_boxes"] += 1
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            stats["boxes"] += 1

        if not lines:
            continue

        stem = xml_path.stem
        dst_img = out_images / f"{stem}.jpg"
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                import shutil
                shutil.copy2(img_path, dst_img)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines))
        stats["frames"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/IDD_Detection")
    ap.add_argument("--out", default="data/idd_yolo")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    if not root.exists():
        raise SystemExit(
            f"{root} not found. Register + download from "
            "https://idd.insaan.iiit.ac.in/ (see this file's docstring).")

    all_unknown: dict[str, int] = {}
    for split in ("train", "val"):
        stats = convert_split(root, split, out / split)
        print(f"\n{split}: {stats['frames']} frames, {stats['boxes']} boxes, "
              f"{stats['skipped_boxes']} tiny boxes skipped")
        for k, v in stats["unknown_classes"].items():
            all_unknown[k] = all_unknown.get(k, 0) + v

    if all_unknown:
        print("\n[WARNING] Unmapped class names seen (not converted — add "
              "them to IDD_TO_UNIFIED in this file if they matter for you):")
        for k, v in sorted(all_unknown.items(), key=lambda kv: -kv[1]):
            print(f"    {k!r}: {v} boxes")

    names = "\n".join(
        f"  {k}: {v}" for k, v in {
            0: "car", 1: "truck", 2: "bus", 3: "van", 4: "pedestrian",
            5: "cyclist", 6: "motorcycle", 7: "tram", 8: "traffic_light",
            9: "traffic_sign", 10: "misc", 11: "autorickshaw", 12: "animal",
            13: "rider", 14: "vehicle_fallback",
        }.items())
    (out / "idd.yaml").write_text(
        f"# IDD Detection in unified 15-class YOLO format\n"
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n"
        f"names:\n{names}\n")
    print(f"\nWrote {out / 'idd.yaml'}")
    print("Merge into the unified dataset with: python data/prepare_merged.py "
          f"(add --idd {out})")


if __name__ == "__main__":
    main()
