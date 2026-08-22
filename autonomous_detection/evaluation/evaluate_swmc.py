"""Safety-Weighted Misclassification Cost (SWMC) evaluation.

WHY A NEW METRIC
----------------
mAP treats every confusion identically. Predicting "bus" for a truck and
predicting "barrier" for a pedestrian cost the same under mAP, but only the
second removes a protection the system would otherwise have applied. IDD-AW
(WACV 2024) made this point for segmentation with Safe mIoU, using the label
hierarchy to separate dangerous mispredictions from harmless ones. SWMC applies
the same principle to detection, with the cost derived from what the decision
layer does differently for each group.

THE COMPARISON PROBLEM, AND HOW THIS SOLVES IT
-----------------------------------------------
The granularity experiment trains three models with three different label
spaces (~24 fine / ~14 semantic / 6 decision). Their mAP numbers are NOT
comparable -- fewer classes is an easier problem, so the 6-class model wins
trivially and the result means nothing. A reviewer will say so immediately.

So every model here is evaluated in ONE common space: predictions and ground
truth are both projected onto the six decision groups before scoring, whatever
granularity the model was trained at. The question becomes the one that
matters: given the same footage and the same decision-relevant question, which
training granularity answers it best? That is a fair comparison, and the fine
model is not handicapped -- its extra resolution simply has to survive the
projection to count for anything.

WHAT IS REPORTED
----------------
    SWMC          total safety cost / number of ground-truth objects. Lower is
                  better; 0.0 is perfect. Decomposed into classification, miss
                  and phantom components so it is visible which one dominates.
    Group recall  per decision group, in the common space.
    Confusion     6x6 in the common space, rows = truth.
    Critical-error rate
                  fraction of ground-truth objects that were either missed or
                  predicted as a strictly less cautious group. This is the
                  single number to quote in the paper.

Usage:
    python evaluation/evaluate_swmc.py \
        --weights runs/india_decision/weights/best.pt \
        --data data/india_decision/decision.yaml \
        --out results/swmc_decision.json

    # compare the three granularities (each trained on its own labels, all
    # scored in the common decision space)
    python evaluation/evaluate_swmc.py --compare \
        fine=runs/india_fine/weights/best.pt:data/india_fine/fine.yaml \
        semantic=runs/india_semantic/weights/best.pt:data/india_semantic/semantic.yaml \
        decision=runs/india_decision/weights/best.pt:data/india_decision/decision.yaml \
        --out results/granularity_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import (  # noqa: E402
    CAUTION_RANK, DECISION_CLASSES, DecisionTaxonomy, false_positive_cost,
    misclassification_cost, miss_cost)

UNMAPPED = -1  # sentinel for a class name the taxonomy does not recognise


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes. Shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match_class_agnostic(gt_boxes, pred_boxes, pred_conf, iou_thr: float = 0.5):
    """Greedy IoU matching that ignores class.

    Class-agnostic on purpose. SWMC is a *classification* cost; matching by
    class first would hide every misclassification as a simultaneous miss and
    phantom, and the cost asymmetry -- the whole point -- would never be
    exercised. Here a box found in the right place but labelled wrongly is
    scored as exactly that.

    Returns (matches, unmatched_gt_idx, unmatched_pred_idx) where matches is a
    list of (gt_idx, pred_idx).
    """
    ious = iou_matrix(gt_boxes, pred_boxes)
    order = np.argsort(-np.asarray(pred_conf, dtype=np.float32)) if len(pred_conf) else []
    gt_taken, pred_taken, matches = set(), set(), []
    for pi in order:
        if len(gt_boxes) == 0:
            break
        col = ious[:, pi].copy()
        for gi in gt_taken:
            col[gi] = -1.0
        gi = int(np.argmax(col))
        if col[gi] >= iou_thr:
            gt_taken.add(gi)
            pred_taken.add(int(pi))
            matches.append((gi, int(pi)))
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in gt_taken]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in pred_taken]
    return matches, unmatched_gt, unmatched_pred


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

class SWMCAccumulator:
    """Accumulates safety cost over a dataset, in the common decision space."""

    def __init__(self):
        n = len(DECISION_CLASSES)
        self.confusion = np.zeros((n, n), dtype=np.int64)
        self.missed = np.zeros(n, dtype=np.int64)
        self.phantom = np.zeros(n, dtype=np.int64)
        self.cost_cls = 0.0
        self.cost_miss = 0.0
        self.cost_fp = 0.0
        self.n_gt = 0
        self.n_critical = 0      # missed, or predicted strictly less cautious
        self.n_unmapped_gt = 0
        self.n_unmapped_pred = 0

    def add_frame(self, gt_groups, gt_boxes, pred_groups, pred_boxes, pred_conf,
                  iou_thr: float = 0.5):
        """One image. Group arrays hold decision-class ids, or UNMAPPED."""
        keep_gt = [i for i, g in enumerate(gt_groups) if g != UNMAPPED]
        self.n_unmapped_gt += len(gt_groups) - len(keep_gt)
        keep_pr = [i for i, g in enumerate(pred_groups) if g != UNMAPPED]
        self.n_unmapped_pred += len(pred_groups) - len(keep_pr)

        gtg = [gt_groups[i] for i in keep_gt]
        gtb = [gt_boxes[i] for i in keep_gt]
        prg = [pred_groups[i] for i in keep_pr]
        prb = [pred_boxes[i] for i in keep_pr]
        prc = [pred_conf[i] for i in keep_pr]

        self.n_gt += len(gtg)
        matches, miss_idx, fp_idx = match_class_agnostic(gtb, prb, prc, iou_thr)

        for gi, pi in matches:
            t, p = DECISION_CLASSES[gtg[gi]], DECISION_CLASSES[prg[pi]]
            self.confusion[gtg[gi], prg[pi]] += 1
            self.cost_cls += misclassification_cost(t, p)
            if CAUTION_RANK.get(p, 0) < CAUTION_RANK.get(t, 0):
                self.n_critical += 1

        for gi in miss_idx:
            t = DECISION_CLASSES[gtg[gi]]
            self.missed[gtg[gi]] += 1
            self.cost_miss += miss_cost(t)
            self.n_critical += 1

        for pi in fp_idx:
            p = DECISION_CLASSES[prg[pi]]
            self.phantom[prg[pi]] += 1
            self.cost_fp += false_positive_cost(p)

    # -- results ----------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return self.cost_cls + self.cost_miss + self.cost_fp

    def summary(self) -> dict:
        n = max(self.n_gt, 1)
        correct = int(np.trace(self.confusion))
        detected = int(self.confusion.sum())
        per_group = {}
        for i, c in enumerate(DECISION_CLASSES):
            support = int(self.confusion[i].sum() + self.missed[i])
            if support == 0:
                continue
            per_group[c] = {
                "support": support,
                "recall": round(float(self.confusion[i].sum()) / support, 4),
                "group_accuracy": round(float(self.confusion[i, i]) / support, 4),
                "missed": int(self.missed[i]),
                "phantom": int(self.phantom[i]),
            }
        return {
            "swmc": round(self.total_cost / n, 4),
            "swmc_components": {
                "classification": round(self.cost_cls / n, 4),
                "miss": round(self.cost_miss / n, 4),
                "phantom": round(self.cost_fp / n, 4),
            },
            "critical_error_rate": round(self.n_critical / n, 4),
            "group_accuracy": round(correct / n, 4),
            "detection_recall": round(detected / n, 4),
            "n_ground_truth": self.n_gt,
            "n_phantom": int(self.phantom.sum()),
            "unmapped_gt_boxes": self.n_unmapped_gt,
            "unmapped_pred_boxes": self.n_unmapped_pred,
            "per_group": per_group,
            "confusion": self.confusion.tolist(),
            "confusion_axis": list(DECISION_CLASSES),
        }

    def print_report(self, title: str = "SWMC") -> dict:
        s = self.summary()
        print(f"\n=== {title} ===")
        print(f"  ground-truth objects      {s['n_ground_truth']}")
        print(f"  SWMC (lower is better)    {s['swmc']:.4f}")
        print(f"      classification        {s['swmc_components']['classification']:.4f}")
        print(f"      miss                  {s['swmc_components']['miss']:.4f}")
        print(f"      phantom               {s['swmc_components']['phantom']:.4f}")
        print(f"  critical-error rate       {s['critical_error_rate']:.4f}")
        print(f"  decision-group accuracy   {s['group_accuracy']:.4f}")
        print(f"  detection recall          {s['detection_recall']:.4f}")
        print(f"  phantom detections        {s['n_phantom']}")
        if s["unmapped_gt_boxes"] or s["unmapped_pred_boxes"]:
            print(f"  [!] unmapped boxes skipped: gt={s['unmapped_gt_boxes']} "
                  f"pred={s['unmapped_pred_boxes']}")

        head = f"  {'group':<17}{'support':>9}{'recall':>9}{'correct':>9}"
        print("\n" + head + f"{'missed':>9}{'phantom':>9}")
        for g, r in s["per_group"].items():
            print(f"  {g:<17}{r['support']:>9}{r['recall']:>9.3f}"
                  f"{r['group_accuracy']:>9.3f}{r['missed']:>9}{r['phantom']:>9}")

        print("\n  confusion (row = truth, col = predicted, common space):")
        w = 19
        print(" " * w + "".join(f"{c[:9]:>10}" for c in DECISION_CLASSES))
        for c, row in zip(DECISION_CLASSES, s["confusion"]):
            print(f"  {c:<{w - 2}}" + "".join(f"{v:>10}" for v in row))
        return s


# --------------------------------------------------------------------------
# dataset plumbing
# --------------------------------------------------------------------------

def load_yaml(path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def names_from_cfg(cfg: dict) -> dict:
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names or []))


def projection(source_names: dict) -> dict:
    """{source_class_id: decision_class_id}, UNMAPPED where unrecognised.

    A source class with no mapping is recorded and skipped, never assigned to
    a nearest guess -- a silent bucket would show up as a fictitious accuracy
    gain in exactly the experiment this metric exists to adjudicate.
    """
    tx = DecisionTaxonomy()
    out = {}
    for sid, name in source_names.items():
        g = tx.map_name(name)
        out[sid] = tx.index(g) if g else UNMAPPED
    tx.report_unmapped()
    return out


def val_image_paths(cfg: dict, data_yaml: Path) -> list:
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    vdir = root / cfg.get("val", "val/images")
    if not vdir.exists():
        raise SystemExit(f"val images not found: {vdir}")
    return sorted(p for p in vdir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def label_path_for(img_path: Path) -> Path:
    """YOLO convention: .../images/... -> .../labels/...

    Rewrites only the LAST path component named "images". A plain string
    replace would also rewrite a dataset root like `indian_images/`, silently
    pointing at a directory that does not exist -- which surfaces as every
    object being scored as a miss rather than as an error.
    """
    parts = list(img_path.parent.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts) / f"{img_path.stem}.txt"


def read_gt(img_path: Path, w: int, h: int):
    """YOLO-format label file -> (class ids, xyxy boxes in pixels)."""
    lbl = label_path_for(img_path)
    if not lbl.exists():
        return [], []
    ids, boxes = [], []
    for line in lbl.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        ids.append(int(parts[0]))
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return ids, boxes


def evaluate(weights: str, data_yaml, imgsz: int = 1280, conf: float = 0.25,
             iou_thr: float = 0.5, limit: int = 0) -> dict:
    from ultralytics import YOLO

    data_yaml = Path(data_yaml)
    cfg = load_yaml(data_yaml)
    gt_proj = projection(names_from_cfg(cfg))

    model = YOLO(weights)
    pred_proj = projection({int(k): v for k, v in model.names.items()})

    images = val_image_paths(cfg, data_yaml)
    if limit:
        images = images[:limit]
    print(f"[swmc] {len(images)} val images | {len(gt_proj)} source classes "
          f"-> {len(DECISION_CLASSES)} decision groups")

    acc = SWMCAccumulator()
    batch = 16
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict(source=[str(p) for p in chunk], imgsz=imgsz,
                                conf=conf, verbose=False)
        for img_path, res in zip(chunk, results):
            h, w = res.orig_shape
            gt_ids, gt_boxes = read_gt(img_path, w, h)
            gt_groups = [gt_proj.get(c, UNMAPPED) for c in gt_ids]

            if res.boxes is None or len(res.boxes) == 0:
                pb, pg, pc = [], [], []
            else:
                pb = res.boxes.xyxy.cpu().numpy().tolist()
                pc = res.boxes.conf.cpu().numpy().tolist()
                pg = [pred_proj.get(int(c), UNMAPPED)
                      for c in res.boxes.cls.cpu().numpy().tolist()]
            acc.add_frame(gt_groups, gt_boxes, pg, pb, pc, iou_thr)

        if (start // batch) % 20 == 0:
            print(f"  {min(start + batch, len(images))}/{len(images)}", flush=True)

    return acc.print_report(f"SWMC - {Path(weights).parent.parent.name}")


def add_map(weights: str, data_yaml, imgsz: int, out: dict) -> dict:
    """Native-space mAP alongside SWMC, so both views appear in one table.

    Reported but explicitly NOT comparable across granularities -- it is the
    number each model gets on its own easier-or-harder label space, included
    because a reviewer will want to see it and because its divergence from
    SWMC is itself the finding.
    """
    try:
        from ultralytics import YOLO
        m = YOLO(weights).val(data=str(data_yaml), imgsz=imgsz, verbose=False)
        out["native_map50"] = round(float(m.box.map50), 4)
        out["native_map50_95"] = round(float(m.box.map), 4)
        out["native_num_classes"] = len(names_from_cfg(load_yaml(data_yaml)))
    except Exception as exc:  # noqa: BLE001 - mAP is a nice-to-have here
        print(f"[swmc] native mAP unavailable: {exc}")
    return out


def _is_drive_colon(spec: str, i: int) -> bool:
    """True if spec[i] is the colon of a Windows drive letter, as in "C:\\runs".

    A drive colon is preceded by exactly one alphabetic character -- one that
    itself starts the path -- and followed by a separator. In "best.pt:/data"
    the colon is preceded by "t" but "p" precedes that, so it is a real
    separator, not a drive.
    """
    if i == 0 or not spec[i - 1].isalpha():
        return False
    if i >= 2 and (spec[i - 2].isalnum() or spec[i - 2] in "._-"):
        return False
    return spec[i + 1:i + 2] in ("\\", "/")


def split_weights_data(spec: str):
    """Split "WEIGHTS:DATA_YAML" on the separator colon.

    Neither the first nor the last colon is reliable on Windows, where a path
    carries its own colon in "C:\\runs\\best.pt". Drive colons are skipped and
    the first remaining colon separates the two paths.
    """
    for i, ch in enumerate(spec):
        if ch == ":" and not _is_drive_colon(spec, i) and spec[:i] and spec[i + 1:]:
            return spec[:i], spec[i + 1:]
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--compare", nargs="+", default=None,
                    metavar="LABEL=WEIGHTS:DATA_YAML",
                    help="score several models in the common decision space")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/swmc.json"))
    args = ap.parse_args()

    runs = []
    if args.compare:
        for spec in args.compare:
            label, _, rest = spec.partition("=")
            weights, data = split_weights_data(rest)
            if not (label and weights and data):
                raise SystemExit(f"bad --compare spec {spec!r}, "
                                 "expected LABEL=WEIGHTS:DATA_YAML")
            runs.append((label, weights, data))
    elif args.weights and args.data:
        runs.append((Path(args.weights).stem, args.weights, args.data))
    else:
        raise SystemExit("give --weights and --data, or --compare")

    report = {}
    for label, weights, data in runs:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        s = evaluate(weights, data, args.imgsz, args.conf, args.iou, args.limit)
        if not args.no_map:
            s = add_map(weights, data, args.imgsz, s)
        s["weights"] = str(weights)
        s["data"] = str(data)
        report[label] = s

    if len(report) > 1:
        print(f"\n{'=' * 70}\nGRANULARITY COMPARISON (common decision space)\n{'=' * 70}")
        print(f"  {'setting':<14}{'classes':>9}{'nat.mAP50':>11}{'SWMC':>9}"
              f"{'critical':>10}{'grp acc':>9}{'recall':>9}")
        for label, s in report.items():
            print(f"  {label:<14}{s.get('native_num_classes', '-'):>9}"
                  f"{s.get('native_map50', float('nan')):>11.4f}"
                  f"{s['swmc']:>9.4f}{s['critical_error_rate']:>10.4f}"
                  f"{s['group_accuracy']:>9.4f}{s['detection_recall']:>9.4f}")
        print("\n  native mAP is NOT comparable across rows (different label"
              "\n  spaces). SWMC, critical-error rate, group accuracy and recall"
              "\n  are all measured in the same six-group space and ARE comparable.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
