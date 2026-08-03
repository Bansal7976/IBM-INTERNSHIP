"""Adapt DriveIndia (already YOLO format) into this project's unified taxonomy.

WHY THIS EXISTS
----------------
DriveIndia (2025, TiHAN-IIT Hyderabad, arxiv.org/abs/2507.19912) is the
freshest and largest Indian traffic detection dataset available — 66,986
images across 24 classes (autorickshaw, tractor, pushcart, construction
vehicle, ambulance, police vehicle, animal, etc.), collected over 3,400+ km
of real Indian urban/rural/highway driving, and — unlike IDD — it's already
released in YOLO format. That means no bounding-box math to redo, only a
class-ID remap into this project's unified taxonomy (prepare_merged.py).

IMPORTANT — this script reads the class NAMES from DriveIndia's own
data.yaml/classes.txt rather than hardcoding assumed integer IDs. The
paper describes 24 classes but doesn't fix their exact index order in the
released files, and guessing wrong here would silently mislabel the entire
dataset with no way to detect it. Every YOLO-format release ships a
data.yaml (or classes.txt) with the authoritative id->name mapping — this
script uses that file directly, so it's correct regardless of the exact
ordering DriveIndia shipped with.

DOWNLOAD
---------
    https://tihan.iith.ac.in/tiand-datasets/  (per the DriveIndia paper's
    stated release channel — TiHAN-IIT Hyderabad dataset repository)
Extract so you have (standard Ultralytics YOLO dataset layout):
    data/DriveIndia/
      data.yaml               (or classes.txt)
      train/images  train/labels
      val/images    val/labels    (or valid/ — both are handled below)

Usage:
    python data/prepare_driveindia.py --data_root data/DriveIndia --out data/driveindia_yolo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

# DriveIndia class name (normalized) -> unified 15-class id (see prepare_merged.py).
# Names are matched case-insensitively with spaces/underscores/hyphens
# collapsed, so "Commercial Vehicle", "commercial_vehicle" etc. all match.
NAME_TO_UNIFIED = {
    "car": 0, "truck": 1, "commercial vehicle": 1, "trailer": 1,
    "bus": 2, "van": 3,
    "person": 4, "pedestrian": 4,
    "cyclist": 5, "bicycle": 5,
    "motorcycle": 6, "motorbike": 6, "two wheeler": 6,
    "tram": 7,
    "traffic light": 8, "traffic signal": 8,
    "traffic sign": 9, "speed limit sign": 9,
    "autorickshaw": 11, "auto rickshaw": 11, "three wheeler": 11,
    "animal": 12, "cattle": 12, "cow": 12, "dog": 12,
    "rider": 13,
    # DriveIndia-specific classes that map to our open-world bucket
    "tractor": 14, "pushcart": 14, "hand cart": 14, "cart": 14,
    "construction vehicle": 14, "ambulance": 14, "police vehicle": 14,
}


def _norm(name: str) -> str:
    s = re.sub(r"[_\-]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", s)


def _load_class_names(root: Path) -> dict[int, str]:
    """Read the dataset's own id->name mapping from data.yaml or classes.txt."""
    for yaml_name in ("data.yaml", "dataset.yaml", "driveindia.yaml"):
        p = root / yaml_name
        if p.exists():
            if yaml is None:
                raise SystemExit("pip install pyyaml  # needed to read data.yaml")
            cfg = yaml.safe_load(p.read_text())
            names = cfg.get("names")
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}

    classes_txt = root / "classes.txt"
    if classes_txt.exists():
        lines = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
        return {i: n for i, n in enumerate(lines)}

    raise SystemExit(
        f"No data.yaml/classes.txt found under {root} — DriveIndia's own "
        "class list is required (see this file's docstring for why we "
        "don't hardcode assumed IDs).")


def _find_split_dir(root: Path, split: str) -> Path | None:
    for name in (split, "valid" if split == "val" else split):
        d = root / name
        if (d / "images").exists():
            return d
    return None


def convert_split(root: Path, split: str, out_dir: Path, id_to_unified: dict[int, int],
                   unknown: dict[str, int], class_names: dict[int, str]) -> dict:
    src = _find_split_dir(root, split)
    if src is None:
        print(f"[skip] no {split}/images (or valid/images) under {root}")
        return {"frames": 0, "boxes": 0}

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "boxes": 0, "skipped_boxes": 0}
    for img in tqdm(sorted((src / "images").iterdir()), desc=f"DriveIndia/{split}"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = src / "labels" / f"{img.stem}.txt"
        if not label.exists():
            continue

        lines = []
        for line in label.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            new_id = id_to_unified.get(old_id)
            if new_id is None:
                name = class_names.get(old_id, f"id_{old_id}")
                unknown[name] = unknown.get(name, 0) + 1
                stats["skipped_boxes"] += 1
                continue
            lines.append(" ".join([str(new_id)] + parts[1:]))
            stats["boxes"] += 1
        if not lines:
            continue

        new_name = f"di_{img.name}"
        dst_img = out_images / new_name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img.resolve())
            except OSError:
                import shutil
                shutil.copy2(img, dst_img)
        (out_labels / f"di_{img.stem}.txt").write_text("\n".join(lines))
        stats["frames"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/DriveIndia")
    ap.add_argument("--out", default="data/driveindia_yolo")
    args = ap.parse_args()

    root, out = Path(args.data_root), Path(args.out)
    if not root.exists():
        raise SystemExit(
            f"{root} not found. Download from https://tihan.iith.ac.in/tiand-datasets/ "
            "(see this file's docstring).")

    class_names = _load_class_names(root)
    id_to_unified = {}
    for old_id, name in class_names.items():
        mapped = NAME_TO_UNIFIED.get(_norm(name))
        if mapped is not None:
            id_to_unified[old_id] = mapped

    print(f"DriveIndia classes found: {len(class_names)}, "
          f"mapped to unified taxonomy: {len(id_to_unified)}")
    unmapped_names = {name for i, name in class_names.items() if i not in id_to_unified}
    if unmapped_names:
        print(f"[WARNING] Not in NAME_TO_UNIFIED, will be skipped: {sorted(unmapped_names)}")
        print("  If any of these matter, add them to NAME_TO_UNIFIED in this file.")

    unknown: dict[str, int] = {}
    for split in ("train", "val"):
        stats = convert_split(root, split, out / split, id_to_unified, unknown, class_names)
        print(f"\n{split}: {stats['frames']} frames, {stats['boxes']} boxes, "
              f"{stats.get('skipped_boxes', 0)} boxes skipped (unmapped class)")

    print("\nMerge into the unified dataset with: python data/prepare_merged.py "
          f"(add --driveindia {out})")


if __name__ == "__main__":
    main()
