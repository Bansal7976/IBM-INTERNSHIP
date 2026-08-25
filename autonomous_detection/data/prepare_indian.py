"""Convert any Indian traffic dataset to YOLO, whatever format it ships in.

WHY ONE SCRIPT AND NOT FOUR
----------------------------
The problem statement points at IDD plus "Mendeley traffic data", and the
Mendeley candidates arrive in three different formats:

    DATS_2022        VOC XML (also ships .txt and .json variants)
    HeteroTraffic    YOLO .txt, already
    IndiaScene365    VOC XML
    Indistreet2K25   VOC XML

Writing a converter per dataset means four places for the same bug. This one
detects the format from what is actually on disk and emits YOLO with the
SOURCE dataset's own class names preserved.

Preserving the source names is deliberate. Collapsing to decision groups here
would hide which original class a box came from, and the granularity experiment
needs exactly that information. `data/prepare_taxonomy.py` does the collapsing
afterwards, at whichever granularity is being tested -- so this script has one
job and the mapping lives in one place.

WHAT IT REPORTS
---------------
Class names found and their box counts, and -- the useful part -- which of them
the decision taxonomy does not recognise. An unrecognised name is data about to
be silently discarded, and it is far cheaper to see that here than to notice a
missing class after training.

Usage:
    python data/prepare_indian.py --src data/DATS_2022 --out data/dats_yolo
    python data/prepare_indian.py --src data/HeteroTraffic --out data/hetero_yolo
    python data/prepare_taxonomy.py --src data/dats_yolo --report-only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import (  # noqa: E402
    EXCLUDED_CLASSES, DecisionTaxonomy, normalise)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# format detection
# --------------------------------------------------------------------------

def detect_format(src: Path) -> str:
    """Work out what we are looking at from the files present."""
    if any(src.rglob("*.xml")):
        return "voc"
    jsons = [p for p in src.rglob("*.json") if p.stat().st_size > 1000]
    for j in jsons:
        try:
            head = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(head, dict) and "annotations" in head and "images" in head:
            return "coco"
    if any(src.rglob("*.txt")):
        return "yolo"
    raise SystemExit(
        f"Could not tell what format {src} is in. Expected VOC .xml, "
        f"COCO .json, or YOLO .txt annotation files somewhere underneath it.")


def find_image(stem: str, image_index: dict) -> Path | None:
    return image_index.get(stem)


def build_image_index(src: Path) -> dict:
    """{stem: path} for every image under src. Later duplicates are ignored."""
    index = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem not in index:
            index[p.stem] = p
    return index


# --------------------------------------------------------------------------
# per-format readers -> a common intermediate
#   record = (image_path, [(class_name, x1, y1, x2, y2), ...], width, height)
# --------------------------------------------------------------------------

def read_voc(src: Path, image_index: dict):
    for xml_path in sorted(src.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        try:
            w = int(float(size.findtext("width")))
            h = int(float(size.findtext("height")))
        except (AttributeError, TypeError, ValueError):
            w = h = 0
        img = find_image(xml_path.stem, image_index)
        if img is None:
            continue
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if not name or bb is None:
                continue
            try:
                boxes.append((name,
                              float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                              float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
            except (TypeError, ValueError):
                continue
        yield img, boxes, w, h


def read_coco(src: Path, image_index: dict):
    for j in sorted(src.rglob("*.json")):
        try:
            data = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not (isinstance(data, dict) and "annotations" in data):
            continue
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        images = {i["id"]: i for i in data.get("images", [])}
        by_image: dict = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for img_id, anns in by_image.items():
            meta = images.get(img_id)
            if meta is None:
                continue
            img = find_image(Path(meta.get("file_name", "")).stem, image_index)
            if img is None:
                continue
            w, h = int(meta.get("width", 0)), int(meta.get("height", 0))
            boxes = []
            for a in anns:
                x, y, bw, bh = a.get("bbox", (0, 0, 0, 0))
                name = cats.get(a.get("category_id"), "")
                if name:
                    boxes.append((name, x, y, x + bw, y + bh))
            yield img, boxes, w, h


def read_yolo_names(src: Path) -> dict:
    """Class id -> name, from whichever names file the dataset ships."""
    for cand in list(src.rglob("*.yaml")) + list(src.rglob("*.yml")):
        try:
            import yaml
            cfg = yaml.safe_load(cand.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        names = (cfg or {}).get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(names))
    for stem in ("classes.txt", "obj.names", "names.txt", "labels.txt"):
        for cand in src.rglob(stem):
            lines = [ln.strip() for ln in
                     cand.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return dict(enumerate(lines))
    return {}


def read_yolo(src: Path, image_index: dict, names: dict):
    for txt in sorted(src.rglob("*.txt")):
        if txt.name in ("classes.txt", "obj.names", "names.txt", "labels.txt"):
            continue
        img = find_image(txt.stem, image_index)
        if img is None:
            continue
        try:
            from PIL import Image
            with Image.open(img) as im:
                w, h = im.size
        except Exception:                                    # noqa: BLE001
            w = h = 0
        boxes = []
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            name = names.get(cid, f"class_{cid}")
            # Already normalised; keep it that way by carrying pixel coords
            # only when the image size is known.
            if w and h:
                boxes.append((name, (cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h))
            else:
                boxes.append((name, cx - bw / 2, cy - bh / 2,
                              cx + bw / 2, cy + bh / 2))
        if not w or not h:
            w = h = 1        # coordinates are already normalised in this case
        yield img, boxes, w, h


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction held out for validation when the dataset "
                         "ships no split of its own")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    ap.add_argument("--report-only", action="store_true",
                    help="scan and report; write nothing")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} not found")

    fmt = detect_format(args.src)
    image_index = build_image_index(args.src)
    print(f"source : {args.src}")
    print(f"format : {fmt}")
    print(f"images : {len(image_index)}")
    if not image_index:
        raise SystemExit("No images found. Check that --src points at the "
                         "extracted dataset root.")

    if fmt == "voc":
        records = read_voc(args.src, image_index)
    elif fmt == "coco":
        records = read_coco(args.src, image_index)
    else:
        records = read_yolo(args.src, image_index, read_yolo_names(args.src))

    # First pass: collect everything, so the class list is known before writing.
    collected, counts = [], Counter()
    for img, boxes, w, h in records:
        if not boxes:
            continue
        collected.append((img, boxes, w, h))
        counts.update(name for name, *_ in boxes)

    if not collected:
        raise SystemExit("No annotated images found. The annotation files may "
                         "not share stems with the image files.")

    class_names = sorted(counts)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    total = sum(counts.values())

    # -- coverage report ---------------------------------------------------
    tx = DecisionTaxonomy()
    print(f"\n{len(collected)} annotated images, {total} boxes, "
          f"{len(class_names)} classes\n")
    print(f"  {'class':<24}{'boxes':>9}{'share':>8}  decision group")
    unrecognised = 0
    for name in class_names:
        g = tx.map_name(name)
        if g is None:
            status = ("excluded by design"
                      if normalise(name) in EXCLUDED_CLASSES
                      else "UNRECOGNISED -- add to models/taxonomy.py")
            if "UNRECOGNISED" in status:
                unrecognised += counts[name]
        else:
            status = g
        print(f"  {name:<24}{counts[name]:>9}{counts[name]/total*100:>7.1f}%  {status}")

    if unrecognised:
        print(f"\n  [!] {unrecognised} boxes ({unrecognised/total*100:.1f}%) "
              f"belong to classes the taxonomy does not recognise.")
        print("      Those boxes survive this conversion -- source names are")
        print("      preserved -- but prepare_taxonomy.py will drop them.")
        print("      Add them to NAME_TO_GROUP and IDD_LEVEL3_GROUPS first.")
    else:
        print("\n  Every class maps to a decision group or is excluded by design.")

    if args.report_only:
        return

    # -- write -------------------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(collected)
    n_val = int(len(collected) * args.val_frac)
    splits = {"val": collected[:n_val], "train": collected[n_val:]}

    for split, items in splits.items():
        img_dir = args.out / split / "images"
        lbl_dir = args.out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for img, boxes, w, h in items:
            lines = []
            for name, x1, y1, x2, y2 in boxes:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
                # Clamp rather than drop: an annotation running a pixel past
                # the frame edge is a real object, not a corrupt label.
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                bw, bh = min(bw, 1.0), min(bh, 1.0)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{name_to_id[name]} {cx:.6f} {cy:.6f} "
                             f"{bw:.6f} {bh:.6f}")
            if not lines:
                continue
            dst = img_dir / f"{img.stem}{img.suffix.lower()}"
            if not dst.exists():
                if args.copy:
                    shutil.copy2(img, dst)
                else:
                    try:
                        dst.symlink_to(img.resolve())
                    except OSError:
                        shutil.copy2(img, dst)
            (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines),
                                                     encoding="utf-8")
            written += 1
        print(f"  {split:<6} {written:>6} images")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    cfg = args.out / "data.yaml"
    cfg.write_text(
        f"# Converted from {args.src} ({fmt} format) by data/prepare_indian.py\n"
        f"# Source class names preserved; collapse with data/prepare_taxonomy.py\n"
        f"path: {args.out.resolve()}\n"
        f"train: train/images\nval: val/images\nnames:\n{names_block}\n",
        encoding="utf-8")
    print(f"\nconfig: {cfg}")
    print(f"\nNext:  python data/prepare_taxonomy.py --src {args.out} --report-only")


if __name__ == "__main__":
    main()
