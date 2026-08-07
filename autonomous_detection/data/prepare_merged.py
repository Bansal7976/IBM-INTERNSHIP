"""Merge KITTI + BDD100K + IDD + DriveIndia (+ optional nuScenes/LISA) into
one unified YOLO dataset.

Unified 15-class map (11 original + 4 India-specific, added to fix the
domain gap documented in the IDD paper — arxiv.org/abs/1811.10200 — and
DriveIndia — arxiv.org/abs/2507.19912 — where KITTI/CULane-trained models
misclassify or completely miss autorickshaws, animals, and handcarts/
tractors/tankers that don't exist in KITTI's 8-class taxonomy at all):
    0 car  1 truck  2 bus  3 van  4 pedestrian  5 cyclist
    6 motorcycle  7 tram  8 traffic_light  9 traffic_sign  10 misc
    11 autorickshaw  12 animal  13 rider  14 vehicle_fallback
       (rider = person ON a bicycle/motorcycle, distinct from `cyclist`
        which some sources use for the bicycle itself; vehicle_fallback =
        IDD's open-world bucket for street cart / tractor / water tanker /
        excavator — extremely common on Indian roads, absent from KITTI)

Prerequisites:
    python data/prepare_kitti.py      --data_root data/kitti       (already done)
    python data/prepare_bdd100k.py    --data_root data/bdd100k
    python data/prepare_idd.py        --data_root data/IDD_Detection
    python data/prepare_driveindia.py --data_root data/DriveIndia

Usage:
    python data/prepare_merged.py --out data/merged_yolo
    # then train:
    # yolo detect train model=yolo11x.pt data=data/merged_yolo/merged.yaml ...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

# KITTI's original 8 classes (ids from kitti_yolo) -> unified 15-class ids
KITTI_ID_TO_UNIFIED = {
    0: 0,   # Car -> car
    1: 3,   # Van -> van
    2: 1,   # Truck -> truck
    3: 4,   # Pedestrian -> pedestrian
    4: 4,   # Person_sitting -> pedestrian
    5: 5,   # Cyclist -> cyclist
    6: 7,   # Tram -> tram
    7: 10,  # Misc -> misc
}

UNIFIED_NAMES = {
    0: "car", 1: "truck", 2: "bus", 3: "van", 4: "pedestrian", 5: "cyclist",
    6: "motorcycle", 7: "tram", 8: "traffic_light", 9: "traffic_sign", 10: "misc",
    11: "autorickshaw", 12: "animal", 13: "rider", 14: "vehicle_fallback",
}


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def add_source(src_root: Path, out_root: Path, split: str, prefix: str,
               id_map: dict | None) -> int:
    """Copy one dataset split into the merged tree, remapping class ids."""
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    if not src_images.exists():
        # kitti_yolo layout: images/{train,val}
        src_images = src_root / "images" / split
        src_labels = src_root / "labels" / split
    if not src_images.exists():
        print(f"[skip] {src_images} not found")
        return 0

    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in tqdm(sorted(src_images.iterdir()), desc=f"{prefix}/{split}"):
        if img.suffix.lower() not in (".jpg", ".png", ".jpeg"):
            continue
        label = src_labels / f"{img.stem}.txt"
        if not label.exists():
            continue

        # Remap class ids
        out_lines = []
        for line in label.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            new_id = id_map.get(cid, cid) if id_map else cid
            out_lines.append(" ".join([str(new_id)] + parts[1:]))
        if not out_lines:
            continue

        # Prefix filenames to avoid collisions across datasets
        new_name = f"{prefix}_{img.name}"
        _link_or_copy(img, out_images / new_name)
        (out_labels / f"{prefix}_{img.stem}.txt").write_text("\n".join(out_lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", default="data/kitti_yolo")
    ap.add_argument("--bdd", default="data/bdd100k_yolo")
    ap.add_argument("--nuscenes", default="data/nuscenes_yolo")
    ap.add_argument("--lisa", default="data/lisa_yolo",
                    help="LISA traffic-light YOLO dir (fallback when no BDD100K)")
    ap.add_argument("--idd", default="data/idd_yolo",
                    help="IDD Detection converted via data/prepare_idd.py — "
                         "fixes fake detections on Indian-specific classes")
    ap.add_argument("--driveindia", default="data/driveindia_yolo",
                    help="DriveIndia converted via data/prepare_driveindia.py")
    ap.add_argument("--uvh26", default="data/uvh26_yolo",
                    help="UVH-26 (IISc, Nov 2025) converted via data/prepare_uvh26.py — "
                         "26.6K real Bengaluru traffic-camera images, largest/freshest "
                         "India-specific source")
    ap.add_argument("--out", default="data/merged_yolo")
    args = ap.parse_args()

    out = Path(args.out)
    totals = {}
    for split in ("train", "val"):
        n_kitti = add_source(Path(args.kitti), out, split, "kitti", KITTI_ID_TO_UNIFIED)
        n_bdd = add_source(Path(args.bdd), out, split, "bdd", None) \
            if Path(args.bdd).exists() else 0
        n_nusc = add_source(Path(args.nuscenes), out, split, "nusc", None) \
            if Path(args.nuscenes).exists() else 0
        n_lisa = add_source(Path(args.lisa), out, split, "lisa", None) \
            if Path(args.lisa).exists() else 0
        n_idd = add_source(Path(args.idd), out, split, "idd", None) \
            if Path(args.idd).exists() else 0
        n_di = add_source(Path(args.driveindia), out, split, "di", None) \
            if Path(args.driveindia).exists() else 0
        n_uvh = add_source(Path(args.uvh26), out, split, "uvh", None) \
            if Path(args.uvh26).exists() else 0
        totals[split] = (n_kitti, n_bdd, n_nusc, n_lisa, n_idd, n_di, n_uvh)

    names = "\n".join(f"  {k}: {v}" for k, v in UNIFIED_NAMES.items())
    (out / "merged.yaml").write_text(
        f"# Merged KITTI + BDD100K + IDD + DriveIndia + UVH-26 (+nuScenes/LISA), "
        f"unified {len(UNIFIED_NAMES)} classes\n"
        f"path: {out.resolve()}\n"
        f"train: train/images\nval: val/images\nnames:\n{names}\n")

    print("\n=== MERGED DATASET ===")
    for split, (k, b, n, li, idd, di, uvh) in totals.items():
        print(f"{split}: kitti={k}  bdd100k={b}  nuscenes={n}  lisa={li}  "
              f"idd={idd}  driveindia={di}  uvh26={uvh}  "
              f"total={k + b + n + li + idd + di + uvh}")
        if idd == 0 and di == 0 and uvh == 0:
            print("  [WARNING] No India-specific data merged (idd/driveindia/uvh26 "
                  "all missing) — autorickshaw/animal/rider/vehicle_fallback "
                  "classes will have ZERO training examples. See "
                  "KRISH_HANDOVER.md 'PLAN C' before training for Indian roads.")
    print(f"Config: {out / 'merged.yaml'}")
    print("\nTrain with:")
    print("  yolo detect train model=yolo11x.pt data="
          f"{out / 'merged.yaml'} imgsz=1280 epochs=300 batch=128 "
          "device=0,1,2,3,4,5,6,7")


if __name__ == "__main__":
    main()
