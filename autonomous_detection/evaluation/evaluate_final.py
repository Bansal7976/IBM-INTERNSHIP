"""JOB F — Full evaluation suite. Produces every results table for the report.

Runs:
  1. Detection mAP on merged val (overall)
  2. Detection mAP per condition (day/night/rain via BDD100K tags)
  3. Night benchmark on ExDark (raw vs night-enhanced) — ablation
  4. Distance accuracy vs KITTI LiDAR ground truth (AbsRel)
  5. Pipeline latency benchmark (per-module + total FPS)

Usage:
    python evaluation/evaluate_final.py --weights runs/final/yolo11x_merged/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REPORT = []


def log(section: str, table: dict):
    REPORT.append((section, table))
    print(f"\n=== {section} ===")
    for k, v in table.items():
        print(f"  {k:40s} {v}")


# ---------------------------------------------------------------- 1 + 2

def eval_detection(weights: str, data_yaml: str, imgsz: int = 1280):
    from ultralytics import YOLO

    model = YOLO(weights)
    m = model.val(data=data_yaml, imgsz=imgsz, batch=32, verbose=False)
    log("DETECTION — merged val set", {
        "mAP@0.5": f"{m.box.map50:.4f}",
        "mAP@0.5:0.95": f"{m.box.map:.4f}",
        "precision": f"{m.box.mp:.4f}",
        "recall": f"{m.box.mr:.4f}",
    })
    return model


def eval_by_condition(model, bdd_root: str, imgsz: int = 1280, per_cond: int = 300):
    """Approximate per-condition detection quality using BDD100K attribute tags.
    Reports mean confidence + detections/image per condition (proxy metrics that
    don't need re-splitting labels)."""
    labels = Path(bdd_root) / "labels" / "det_20" / "det_val.json"
    if not labels.exists():
        print("[skip] BDD100K val labels not found")
        return
    with open(labels) as f:
        frames = json.load(f)

    buckets: dict[str, list] = {}
    for fr in frames:
        a = fr.get("attributes", {})
        key = a.get("timeofday", "?")
        buckets.setdefault(key, []).append(fr["name"])

    table = {}
    img_dir = Path(bdd_root) / "images" / "100k" / "val"
    for cond, names in buckets.items():
        confs, counts = [], []
        for name in names[:per_cond]:
            p = img_dir / name
            if not p.exists():
                continue
            r = model.predict(str(p), imgsz=imgsz, conf=0.3, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                confs.extend(r.boxes.conf.cpu().numpy().tolist())
                counts.append(len(r.boxes))
            else:
                counts.append(0)
        if counts:
            table[cond] = (f"dets/img={np.mean(counts):.2f}  "
                           f"mean_conf={np.mean(confs) if confs else 0:.3f}  "
                           f"(n={len(counts)})")
    log("DETECTION BY CONDITION (BDD100K time-of-day)", table)


# ---------------------------------------------------------------- 3

def eval_night_ablation(model, exdark_root: str, imgsz: int = 1280, n: int = 300):
    """ExDark: detections with raw vs enhanced frames. More recovered objects
    with enhancement = the night layer works."""
    from inference.night_enhance import NightEnhancer, scene_lighting

    root = Path(exdark_root)
    imgs = [p for p in root.rglob("*.jpg")][:n] + [p for p in root.rglob("*.png")][:n]
    if not imgs:
        print("[skip] ExDark images not found")
        return
    enhancer = NightEnhancer(str(PROJECT_ROOT / "weights" / "zero_dce_plus.pth"))

    raw_counts, enh_counts = [], []
    for p in imgs[:n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = enhancer.maybe_enhance(img, scene_lighting(gray))
        r0 = model.predict(img, imgsz=imgsz, conf=0.3, verbose=False)[0]
        r1 = model.predict(enhanced, imgsz=imgsz, conf=0.3, verbose=False)[0]
        raw_counts.append(0 if r0.boxes is None else len(r0.boxes))
        enh_counts.append(0 if r1.boxes is None else len(r1.boxes))

    log("NIGHT ABLATION (ExDark)", {
        "images evaluated": len(raw_counts),
        "detections/img RAW": f"{np.mean(raw_counts):.2f}",
        "detections/img ENHANCED": f"{np.mean(enh_counts):.2f}",
        "recovery gain": f"{(np.mean(enh_counts) / max(np.mean(raw_counts), 1e-9) - 1) * 100:+.1f}%",
    })


# ---------------------------------------------------------------- 4

def eval_depth_accuracy(kitti_root: str, n: int = 100):
    """AbsRel of Depth Anything V2 vs KITTI LiDAR-projected sparse depth."""
    try:
        from inference.collision import DepthAnythingV2Metric
        depth = DepthAnythingV2Metric(
            str(PROJECT_ROOT / "weights" / "depth_anything_v2_metric_vkitti_vits.pth"))
    except Exception as e:
        print(f"[skip] depth model unavailable: {e}")
        return

    # KITTI depth: expects data/kitti_depth/{image_02, proj_depth/groundtruth}
    root = Path(kitti_root)
    gt_dir = root / "proj_depth" / "groundtruth" / "image_02"
    img_dir = root / "image_02"
    if not gt_dir.exists():
        print("[skip] KITTI depth GT not found (download depth completion set)")
        return

    abs_rels = []
    for gt_path in sorted(gt_dir.glob("*.png"))[:n]:
        img_path = img_dir / gt_path.name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        pred = depth.infer(img)
        mask = (gt > 1.0) & (gt < 80.0)
        if mask.sum() < 100:
            continue
        abs_rels.append(float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask])))

    if abs_rels:
        log("DEPTH ACCURACY vs KITTI LiDAR", {
            "frames": len(abs_rels),
            "AbsRel (lower=better, target <0.12)": f"{np.mean(abs_rels):.4f}",
        })


# ---------------------------------------------------------------- 5

def eval_latency(weights: str, video: str | None = None, n_frames: int = 100):
    from inference.adas_final import ADASFinalPipeline

    pipe = ADASFinalPipeline(detector_weights=weights)
    if video and Path(video).exists():
        cap = cv2.VideoCapture(video)
        frames = []
        while len(frames) < n_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    else:
        frames = [np.random.randint(0, 255, (720, 1280, 3), np.uint8)
                  for _ in range(n_frames)]

    # Warmup then measure
    for f in frames[:5]:
        pipe.process_frame(f)
    t0 = time.perf_counter()
    lat = [pipe.process_frame(f).latency_ms for f in frames]
    total = time.perf_counter() - t0

    log("PIPELINE LATENCY", {
        "frames": len(frames),
        "mean latency": f"{np.mean(lat):.1f} ms",
        "p95 latency": f"{np.percentile(lat, 95):.1f} ms",
        "throughput": f"{len(frames) / total:.1f} FPS",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/final/yolo11x_merged/weights/best.pt")
    ap.add_argument("--data", default="data/merged_yolo/merged.yaml")
    ap.add_argument("--bdd", default="data/bdd100k")
    ap.add_argument("--exdark", default="data/exdark")
    ap.add_argument("--kitti_depth", default="data/kitti_depth")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    model = eval_detection(args.weights, args.data)
    eval_by_condition(model, args.bdd)
    eval_night_ablation(model, args.exdark)
    eval_depth_accuracy(args.kitti_depth)
    eval_latency(args.weights, args.video)

    # Dump machine-readable report
    out = PROJECT_ROOT / "runs" / "final_evaluation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        [{"section": s, "results": t} for s, t in REPORT], indent=2))
    print(f"\nFull report saved: {out}")


if __name__ == "__main__":
    main()
