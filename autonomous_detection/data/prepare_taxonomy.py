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
    DECISION_CLASSES, EXCLUDED_CLASSES, IDD_LEVEL3_GROUPS, NAME_TO_GROUP,
    DecisionTaxonomy, normalise)

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


def build_target_classes(level: str, all_source_names: list) -> list:
    """The output class list, computed ONCE across every source.

    Building it per-source is wrong when sources are combined: UVH-26 carries
    no pedestrians, so a semantic class list derived from it alone omits
    "person" entirely, and every source's ids then index a different list. The
    labels would point at the wrong classes and nothing would report an error.

    Every level is restricted to the shared vocabulary, so all three arms end
    up holding exactly the same boxes.
    """
    if level == "decision":
        return list(DECISION_CLASSES)

    out, seen = [], set()
    for source_names in all_source_names:
        for cid in sorted(source_names):
            name = source_names[cid]
            key = normalise(name)
            if key not in NAME_TO_GROUP:
                continue
            if level == "fine":
                # Deduplicated on the normalised name, so "Truck" and "truck"
                # are one class. Genuinely synonymous labels across datasets
                # ("Three-Wheeler" vs "autorickshaw") stay separate: merging
                # those is the taxonomy's job, and the fine arm is supposed to
                # keep every distinction its sources drew.
                target = name
            else:
                target = IDD_LEVEL3_GROUPS.get(key)
                if target is None:
                    continue
            if normalise(target) in seen:
                continue
            seen.add(normalise(target))
            out.append(target)
    return out if level == "fine" else sorted(out)


def build_mapping(level: str, source_names: dict, target_classes: list):
    """Return (id_map, unmapped_names) mapping one source onto the target list."""
    index = {normalise(c): i for i, c in enumerate(target_classes)}
    id_map, unmapped = {}, {}
    for cid in sorted(source_names):
        name = source_names[cid]
        key = normalise(name)
        if level == "fine":
            target = key
        elif level == "semantic":
            g = IDD_LEVEL3_GROUPS.get(key)
            target = normalise(g) if g else None
        elif level == "decision":
            g = NAME_TO_GROUP.get(key)
            target = normalise(g) if g else None
        else:
            raise SystemExit(f"unknown level {level!r}")

        slot = index.get(target) if target else None
        if slot is None:
            unmapped[name] = unmapped.get(name, 0) + 1
            continue
        id_map[cid] = slot
    return id_map, unmapped


def convert_split(src: Path, out: Path, split: str, id_map: dict,
                  prefix: str = "") -> dict:
    """Convert one split of one source into `out`.

    `prefix` namespaces the output stems. Several sources write into the same
    directory, and two datasets can easily contain an image called 000123.jpg;
    without the prefix the second would silently overwrite the first's label
    file and the boxes would be attached to the wrong picture.
    """
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

        stem = f"{prefix}{img.stem}"
        dst = out_img / f"{stem}{img.suffix}"
        if not dst.exists():
            try:
                dst.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst)
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats["frames"] += 1
    return stats


def count_boxes_per_class(src: Path) -> dict:
    """{source_class_id: box count} over every split, without writing anything."""
    counts = {}
    for split in ("train", "val"):
        lbl_dir = src / split / "labels"
        if not lbl_dir.exists():
            lbl_dir = src / "labels" / split
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    counts[cid] = counts.get(cid, 0) + 1
    return counts


def report(src: Path, max_dropped_frac: float) -> int:
    """Print what each granularity would keep and drop. Writes nothing.

    Run this FIRST, before any GPU time. It answers the question that decides
    whether the experiment is worth running: how much of the data survives the
    mapping, and is anything being lost that should not be?

    Distinguishes two very different reasons a class is dropped:
      * excluded by design -- traffic lights, signs, "misc". Not path
        obstacles, so their absence is correct.
      * unrecognised -- a name the taxonomy has no entry for. That is a GAP,
        and the fix is to add it to models/taxonomy.py rather than accept the
        loss.
    """
    source_names = read_source_names(src)
    counts = count_boxes_per_class(src)
    total = sum(counts.values())
    if total == 0:
        print(f"[report] no label files found under {src}")
        return 1

    tx = DecisionTaxonomy()
    print(f"source: {src}")
    print(f"{total} boxes across {len(source_names)} classes\n")
    print(f"  {'id':>3}  {'class':<20}{'boxes':>10}{'share':>8}  "
          f"{'decision':<17}{'semantic':<17}status")

    excluded = unrecognised = kept = 0
    for cid in sorted(source_names):
        name = source_names[cid]
        n = counts.get(cid, 0)
        g = tx.map_name(name)
        s = IDD_LEVEL3_GROUPS.get(normalise(name))
        if g is not None:
            status, kept = "kept", kept + n
        elif normalise(name) in EXCLUDED_CLASSES:
            status, excluded = "excluded by design", excluded + n
        else:
            status, unrecognised = "UNRECOGNISED -- add a mapping", unrecognised + n
        print(f"  {cid:>3}  {name:<20}{n:>10}{n / total * 100:>7.1f}%  "
              f"{str(g):<17}{str(s):<17}{status}")

    dropped = excluded + unrecognised
    print(f"\n  kept                {kept:>10}  {kept / total * 100:>5.1f}%")
    print(f"  excluded by design  {excluded:>10}  {excluded / total * 100:>5.1f}%")
    print(f"  UNRECOGNISED        {unrecognised:>10}  {unrecognised / total * 100:>5.1f}%")

    # The limit applies to UNRECOGNISED boxes only. Boxes excluded by design
    # are supposed to be absent: traffic signs are numerous and can easily be a
    # fifth of all annotations, and failing the run because we correctly
    # declined to treat them as path obstacles would be nonsense. An
    # unrecognised class is the opposite -- data we meant to keep and silently
    # lost.
    if unrecognised:
        print("\n  Unrecognised classes are a GAP, not a decision. Add them to")
        print("  NAME_TO_GROUP (and IDD_LEVEL3_GROUPS, which must stay in sync)")
        print("  in models/taxonomy.py, or to EXCLUDED_CLASSES if they genuinely")
        print("  are not path obstacles.")

    if unrecognised / total > max_dropped_frac:
        print(f"\n  FAILED: {unrecognised / total * 100:.1f}% of boxes belong to "
              f"classes the taxonomy does not recognise, above the "
              f"{max_dropped_frac * 100:.0f}% limit.")
        print("  Training on this would silently discard real objects. Fix the")
        print("  mapping first, or raise --max-dropped deliberately if the loss")
        print("  is genuinely acceptable.")
        return 1

    print(f"\n  OK: {unrecognised / total * 100:.1f}% unrecognised, within the "
          f"{max_dropped_frac * 100:.0f}% limit. "
          f"({dropped / total * 100:.1f}% dropped in total, the rest by design.)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, nargs="+",
                    help="one or more source YOLO dataset directories, each "
                         "containing a *.yaml. Several sources are combined "
                         "into one dataset; for the fine arm their class lists "
                         "are unioned rather than collapsed.")
    ap.add_argument("--level", choices=["fine", "semantic", "decision"],
                    help="required unless --report-only")
    ap.add_argument("--out", type=Path, help="required unless --report-only")
    ap.add_argument("--report-only", action="store_true",
                    help="show what each granularity keeps and drops, write "
                         "nothing. Run this before spending GPU time.")
    ap.add_argument("--max-dropped", type=float, default=0.10,
                    help="fail if more than this fraction of boxes would be "
                         "dropped (default 0.10)")
    args = ap.parse_args()

    for src in args.src:
        if not src.exists():
            raise SystemExit(f"{src} not found")

    if args.report_only:
        raise SystemExit(max(report(src, args.max_dropped) for src in args.src))
    if not (args.level and args.out):
        raise SystemExit("--level and --out are required unless --report-only")

    all_names = [read_source_names(src) for src in args.src]
    classes = build_target_classes(args.level, all_names)

    print(f"level    : {args.level}")
    print(f"sources  : {[str(s) for s in args.src]}")

    totals = {"train": {"frames": 0, "boxes": 0, "dropped": 0},
              "val": {"frames": 0, "boxes": 0, "dropped": 0}}
    for i, (src, source_names) in enumerate(zip(args.src, all_names)):
        id_map, unmapped = build_mapping(args.level, source_names, classes)

        print(f"\n[{src.name}] {len(source_names)} classes -> {len(classes)}")
        if unmapped:
            by_design = [k for k in unmapped if normalise(k) in EXCLUDED_CLASSES]
            gaps = [k for k in unmapped if normalise(k) not in EXCLUDED_CLASSES]
            if by_design:
                print(f"  excluded by design (not path obstacles): {sorted(by_design)}")
            if gaps:
                print("  [WARNING] UNRECOGNISED classes -- their boxes are DROPPED,")
                print("            not reassigned. This is a gap in")
                print(f"            models/taxonomy.py, not a decision: {sorted(gaps)}")
                print("            Run --report-only to see how many boxes it costs.")

        # Namespace the stems per source: two datasets can each contain an
        # image called 000123.jpg, and without this the second would overwrite
        # the first's labels.
        prefix = f"s{i}_" if len(args.src) > 1 else ""
        for split in ("train", "val"):
            s = convert_split(src, args.out, split, id_map, prefix)
            for k in totals[split]:
                totals[split][k] += s[k]

    print(f"\ntarget classes ({len(classes)}): {classes}")
    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (args.out / f"{args.level}.yaml").write_text(
        f"# Granularity: {args.level}  ({len(classes)} classes)\n"
        f"# Built by data/prepare_taxonomy.py from "
        f"{', '.join(str(s) for s in args.src)}\n"
        f"path: {args.out.resolve()}\n"
        f"train: train/images\nval: val/images\nnames:\n{names_block}\n",
        encoding="utf-8")

    print("\n--- summary ---")
    for split, s in totals.items():
        print(f"{split:<6} frames={s['frames']:>6}  boxes={s['boxes']:>7}  "
              f"dropped={s['dropped']:>6}")
    print(f"\nconfig: {args.out / (args.level + '.yaml')}")
    print("\nThe three levels must report IDENTICAL box counts. If they do not,")
    print("the comparison measures a difference in data, not in taxonomy.")


if __name__ == "__main__":
    main()
