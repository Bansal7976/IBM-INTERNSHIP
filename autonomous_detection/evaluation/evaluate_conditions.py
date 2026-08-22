"""Per-condition evaluation: does the system still work at night?

WHAT THIS PRODUCES
------------------
One table, split by lighting condition, reporting the safety metric rather than
mAP alone:

    condition      images   SWMC   critical   recall   (raw)
    condition      images   SWMC   critical   recall   (enhanced)

Two claims in the plan were, until now, assumed rather than measured:

  1. That the pipeline's DAY / NIGHT_LIT / NIGHT_UNLIT split is meaningful.
     It is used to gate the overtaking decision and to switch on enhancement,
     so if the buckets are not real the gate is arbitrary. Here the buckets are
     produced by the pipeline's own `scene_lighting()`, so the table validates
     the classifier that ships, not a separate offline one.

  2. That low-light enhancement helps detection. It is applied on every dark
     frame today. Enhancement can just as easily amplify noise into texture
     that a domain-shifted detector reads as an object -- which is one of the
     ways phantom detections appear. The ablation runs both arms over the SAME
     images, so the comparison is controlled and the answer can come out
     negative. If it does, that is the finding.

WHY SWMC AND NOT mAP
--------------------
Night failures are overwhelmingly MISSES, and a miss of a pedestrian is not the
same event as a miss of a parked cart. mAP averages those together; SWMC prices
them apart, and the critical-error rate counts exactly the outcomes that would
have removed a protection. See evaluation/evaluate_swmc.py.

SMALL BUCKETS
-------------
Unlit-night frames are rare in most road datasets. A confident number computed
over 30 images is noise dressed as a result, so any bucket below
--min-bucket images is reported and explicitly flagged as insufficient rather
than quietly averaged into the table.

Usage:
    python evaluation/evaluate_conditions.py \
        --weights runs/granularity/decision/weights/best.pt \
        --data data/gran_decision/decision.yaml \
        --out results/conditions.json

    # skip the enhancement arm (half the runtime)
    python evaluation/evaluate_conditions.py ... --no-enhance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_swmc import (  # noqa: E402
    UNMAPPED, SWMCAccumulator, load_yaml, names_from_cfg, projection, read_gt,
    val_image_paths)

CONDITIONS = ["DAY", "NIGHT_LIT", "NIGHT_UNLIT"]


def bucket_images(images, sample_stride: int = 1):
    """Group val images by the pipeline's own lighting classifier."""
    from inference.night_enhance import scene_lighting

    buckets = {c: [] for c in CONDITIONS}
    stats = {c: [] for c in CONDITIONS}
    for i, p in enumerate(images):
        if i % sample_stride:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cond = scene_lighting(gray)
        buckets.setdefault(cond, []).append(p)
        stats.setdefault(cond, []).append(float(gray.mean()))
    return buckets, stats


def score_bucket(model, paths, gt_proj, pred_proj, imgsz, conf, iou_thr,
                 enhancer=None, lighting="DAY") -> dict:
    """SWMC over one condition bucket, optionally through the enhancer."""
    acc = SWMCAccumulator()
    batch = 8
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        frames, keep = [], []
        for p in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if enhancer is not None:
                img = enhancer.maybe_enhance(img, lighting)
            frames.append(img)
            keep.append(p)
        if not frames:
            continue
        results = model.predict(source=frames, imgsz=imgsz, conf=conf, verbose=False)
        for p, img, res in zip(keep, frames, results):
            h, w = img.shape[:2]
            gt_ids, gt_boxes = read_gt(p, w, h)
            gt_groups = [gt_proj.get(c, UNMAPPED) for c in gt_ids]
            if res.boxes is None or len(res.boxes) == 0:
                pb, pg, pc = [], [], []
            else:
                pb = res.boxes.xyxy.cpu().numpy().tolist()
                pc = res.boxes.conf.cpu().numpy().tolist()
                pg = [pred_proj.get(int(c), UNMAPPED)
                      for c in res.boxes.cls.cpu().numpy().tolist()]
            acc.add_frame(gt_groups, gt_boxes, pg, pb, pc, iou_thr)
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth val image when bucketing")
    ap.add_argument("--min-bucket", type=int, default=100,
                    help="below this many images a bucket is flagged as too "
                         "small to draw a conclusion from")
    ap.add_argument("--no-enhance", action="store_true",
                    help="skip the low-light enhancement arm")
    ap.add_argument("--out", type=Path, default=Path("results/conditions.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = Path(args.data)
    cfg = load_yaml(data_yaml)
    gt_proj = projection(names_from_cfg(cfg))

    model = YOLO(args.weights)
    pred_proj = projection({int(k): v for k, v in model.names.items()})

    images = val_image_paths(cfg, data_yaml)
    if args.limit:
        images = images[:args.limit]

    print(f"[conditions] bucketing {len(images)} val images by the pipeline's "
          f"own scene_lighting()")
    buckets, brightness = bucket_images(images, args.stride)
    for c in CONDITIONS:
        b = brightness.get(c, [])
        mean_b = f"{np.mean(b):.1f}" if b else "-"
        print(f"  {c:<12} {len(buckets.get(c, [])):>6} images   "
              f"mean brightness {mean_b}")

    enhancer = None
    if not args.no_enhance:
        try:
            from inference.night_enhance import NightEnhancer
            enhancer = NightEnhancer()
        except Exception as exc:  # noqa: BLE001
            print(f"[conditions] enhancement arm unavailable: {exc}")

    report = {"buckets": {}, "weights": str(args.weights), "data": str(args.data)}
    for cond in CONDITIONS:
        paths = buckets.get(cond, [])
        entry = {
            "n_images": len(paths),
            "mean_brightness": (round(float(np.mean(brightness[cond])), 2)
                                if brightness.get(cond) else None),
            "sufficient": len(paths) >= args.min_bucket,
        }
        if paths:
            print(f"\n--- {cond}: scoring {len(paths)} images (raw) ---")
            entry["raw"] = score_bucket(model, paths, gt_proj, pred_proj,
                                        args.imgsz, args.conf, args.iou)
            if enhancer is not None and cond != "DAY":
                print(f"--- {cond}: scoring {len(paths)} images (enhanced) ---")
                entry["enhanced"] = score_bucket(
                    model, paths, gt_proj, pred_proj, args.imgsz, args.conf,
                    args.iou, enhancer=enhancer, lighting=cond)
        report["buckets"][cond] = entry

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 78)
    print("PER-CONDITION DETECTION SAFETY  (SWMC: lower is better)")
    print("=" * 78)
    print(f"  {'condition':<13}{'arm':<10}{'images':>8}{'SWMC':>9}"
          f"{'critical':>10}{'recall':>9}{'grp acc':>9}{'phantom':>9}")
    for cond in CONDITIONS:
        e = report["buckets"].get(cond, {})
        for arm in ("raw", "enhanced"):
            s = e.get(arm)
            if not s:
                continue
            flag = "" if e.get("sufficient") else "  <- too few images"
            print(f"  {cond:<13}{arm:<10}{e['n_images']:>8}{s['swmc']:>9.4f}"
                  f"{s['critical_error_rate']:>10.4f}"
                  f"{s['detection_recall']:>9.4f}{s['group_accuracy']:>9.4f}"
                  f"{s['n_phantom']:>9}{flag}")

    # -------------------------------------------------------- interpretation
    day = report["buckets"].get("DAY", {}).get("raw")
    print()
    for cond in ("NIGHT_LIT", "NIGHT_UNLIT"):
        e = report["buckets"].get(cond, {})
        raw, enh = e.get("raw"), e.get("enhanced")
        if not (day and raw):
            continue
        if not e.get("sufficient"):
            print(f"  {cond}: {e['n_images']} images -- too few to conclude "
                  f"from; report the count, not the number.")
            continue
        delta = raw["swmc"] - day["swmc"]
        print(f"  {cond}: SWMC {raw['swmc']:.4f} vs DAY {day['swmc']:.4f} "
              f"({delta:+.4f}); critical errors "
              f"{raw['critical_error_rate']:.4f} vs "
              f"{day['critical_error_rate']:.4f}")
        if enh:
            d = enh["swmc"] - raw["swmc"]
            verdict = ("enhancement HELPS" if d < -0.005 else
                       "enhancement HURTS" if d > 0.005 else
                       "enhancement makes no measurable difference")
            print(f"    {verdict}: SWMC {raw['swmc']:.4f} -> {enh['swmc']:.4f} "
                  f"({d:+.4f}), phantoms {raw['n_phantom']} -> "
                  f"{enh['n_phantom']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
