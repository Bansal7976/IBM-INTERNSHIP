"""Remap any YOLO-format dataset to a chosen label granularity.

This is the tool that makes the granularity experiment possible: the same
images and boxes, relabelled at three levels, so that a difference in accuracy
is attributable to the taxonomy and nothing else.

    fine       the source dataset's own classes (the control)
    semantic   grouped by visual/semantic similarity, in the spirit of IDD's
               own level-3 hierarchy (the second control -- without it we
               cannot tell whether a gain comes from grouping at all or from
               grouping THIS way)
    decision   the six decision-consequence groups (ours)

All three levels are restricted to the same shared vocabulary, so the three
datasets contain *identical* boxes and differ only in how those boxes are
labelled. Any accuracy difference is then attributable to the taxonomy. Verify
this after building: the three runs must report the same box counts.

Boxes whose source class has no mapping are DROPPED and counted, never
reassigned to a nearest guess -- silently bucketing an unrecognised class
would corrupt the labels with no way to notice.

Usage:
    python data/prepare_taxonomy.py --src data/merged_india_yolo \
        --level decision --out data/india_decision6

    # build all three at once for the experiment
    for L in fine semantic decision; do
        python data/prepare_taxonomy.py --src data/merged_india_yolo \
            --level $L --out data/india_$L
    done
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import (  # noqa: E402
    DECISION_CLASSES, IDD_LEVEL3_GROUPS, NAME_TO_GROUP, DecisionTaxonomy,
    normalise)

try:
    import yaml
except ImportError:
    yaml = None


def read_source_names(src: Path) -> dict:
    """Read {id: name} from whichever dataset yaml the source directory has."""
    for cand in sorted(src.glob("*.yaml")):
        if yaml is None:
            raise SystemExit("pip install pyyaml")
        cfg = yaml.safe_load(cand.read_text(encoding="utf-8"))
        names = (cfg or {}).get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(names))
    raise SystemExit(f"No dataset yaml with a 'names' block found in {src}")


def build_mapping(level: str, source_names: dict):
    """Return (id_map, class_list, unmapped_names) for the requested level."""
    if level == "fine":
        # Restricted to the shared vocabulary, not simply passed through. All
        # three granularities must contain exactly the same boxes -- otherwise
        # the fine model trains on objects the others never see, and the
        # comparison measures a difference in data rather than in taxonomy.
        classes, id_map, unmapped = [], {}, {}
        for sid in sorted(source_names):
            name = source_names[sid]
            if normalise(name) not in NAME_TO_GROUP:
                unmapped[name] = unmapped.get(name, 0) + 1
                continue
            id_map[sid] = len(classes)
            classes.append(name)
        return id_map, classes, unmapped

    if level == "semantic":
        classes, id_map, unmapped = [], {}, {}
        for sid in sorted(source_names):
            g = IDD_LEVEL3_GROUPS.get(normalise(source_names[sid]))
            if g is None:
                unmapped[source_names[sid]] = unmapped.get(source_names[sid], 0) + 1
                continue
            if g not in classes:
                classes.append(g)
            id_map[sid] = classes.index(g)
        classes_sorted = sorted(classes)
        # re-index against the sorted list so the ordering is deterministic
        remap = {classes.index(c): classes_sorted.index(c) for c in classes}
        return {k: remap[v] for k, v in id_map.items()}, classes_sorted, unmapped

    if level == "decision":
        tx = DecisionTaxonomy()
        id_map = tx.map_id_from_names(source_names)
        return id_map, list(DECISION_CLASSES), dict(tx.unmapped)

    raise SystemExit(f"unknown level {level!r}")


def convert_split(src: Path, out: Path, split: str, id_map: dict) -> dict:
    src_img = src / split / "images"
    src_lbl = src / split / "labels"
    if not src_img.exists():
        src_img, src_lbl = src / "images" / split, src / "labels" / split
    if not src_img.exists():
        print(f"[skip] no {split} split under {src}")
        return {"frames": 0, "boxes": 0, "dropped": 0}

    out_img, out_lbl = out / split / "images", out / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "boxes": 0, "dropped": 0}
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = src_lbl / f"{img.stem}.txt"
        if not lbl.exists():
            continue

        lines = []
        for raw in lbl.read_text(encoding="utf-8").strip().splitlines():
            parts = raw.split()
            if len(parts) != 5:
                continue
            new_id = id_map.get(int(parts[0]))
            if new_id is None:
                stats["dropped"] += 1
                continue
            lines.append(" ".join([str(new_id)] + parts[1:]))
            stats["boxes"] += 1
        if not lines:
            continue

        dst = out_img / img.name
        if not dst.exists():
            try:
                dst.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst)
        (out_lbl / f"{img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats["frames"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="source YOLO dataset directory (must contain a *.yaml)")
    ap.add_argument("--level", required=True,
                    choices=["fine", "semantic", "decision"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} not found")

    source_names = read_source_names(args.src)
    id_map, classes, unmapped = build_mapping(args.level, source_names)

    print(f"level        : {args.level}")
    print(f"source classes: {len(source_names)}  ->  target classes: {len(classes)}")
    print(f"target        : {classes}")
    if unmapped:
        print("\n[warning] source classes with no mapping (their boxes are DROPPED,")
        print("          not reassigned -- add them to models/taxonomy.py if wanted):")
        for k, v in sorted(unmapped.items()):
            print(f"    {k!r}")

    totals = {}
    for split in ("train", "val"):
        totals[split] = convert_split(args.src, args.out, split, id_map)

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (args.out / f"{args.level}.yaml").write_text(
        f"# Granularity: {args.level}  ({len(classes)} classes)\n"
        f"# Built by data/prepare_taxonomy.py from {args.src}\n"
        f"path: {args.out.resolve()}\n"
        f"train: train/images\nval: val/images\nnames:\n{names_block}\n",
        encoding="utf-8")

    print("\n--- summary ---")
    for split, s in totals.items():
        print(f"{split:<6} frames={s['frames']:>6}  boxes={s['boxes']:>7}  "
              f"dropped={s['dropped']:>6}")
    print(f"\nconfig: {args.out / (args.level + '.yaml')}")


if __name__ == "__main__":
    main()
