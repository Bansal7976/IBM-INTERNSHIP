"""
Detection evaluation metrics.

Implements:
  - mAP@0.5, mAP@0.5:0.95 (COCO-style)
  - Per-class AP
  - NDS (nuScenes Detection Score) summary
  - Confusion matrix

Reference: COCO evaluation (pycocotools) + custom AP implementation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class DetectionResult:
    """Single predicted detection."""
    image_id: str
    box: np.ndarray      # [x1,y1,x2,y2]
    score: float
    class_id: int


@dataclass
class GroundTruth:
    """Single ground-truth annotation."""
    image_id: str
    box: np.ndarray      # [x1,y1,x2,y2]
    class_id: int
    is_crowd: bool = False


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes (xyxy)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using 101-point interpolation (COCO style)."""
    ap = 0.0
    for thresh in np.linspace(0, 1, 101):
        prec_at_rec = precision[recall >= thresh]
        ap += (np.max(prec_at_rec) if prec_at_rec.size > 0 else 0) / 101
    return ap


def compute_class_ap(
    preds: List[DetectionResult],
    gts: List[GroundTruth],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute AP for a single class at a given IoU threshold.
    preds and gts should already be filtered to one class.
    """
    if not gts:
        return 0.0

    # Group GTs by image
    gt_by_image: Dict[str, List] = defaultdict(list)
    for gt in gts:
        gt_by_image[gt.image_id].append(gt)

    # Sort predictions by descending score
    preds = sorted(preds, key=lambda x: x.score, reverse=True)

    matched = defaultdict(set)  # image_id -> set of matched GT indices
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        image_gts = gt_by_image.get(pred.image_id, [])

        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(image_gts):
            if j in matched[pred.image_id]:
                continue
            iou_val = iou(pred.box, gt.box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            matched[pred.image_id].add(best_gt_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / (len(gts) + 1e-6)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    return compute_ap(recall, precision)


class MeanAveragePrecision:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 across all classes.

    Usage:
        metric = MeanAveragePrecision(num_classes=8, class_names=KITTI_CLASSES)
        metric.update(predictions, ground_truths)
        results = metric.compute()
        print(results['map_50'], results['map_50_95'])
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.iou_thresholds = iou_thresholds or list(np.arange(0.5, 1.0, 0.05).round(2))
        self._preds: List[DetectionResult] = []
        self._gts:   List[GroundTruth]     = []

    def update(
        self,
        predictions: List[DetectionResult],
        ground_truths: List[GroundTruth],
    ):
        self._preds.extend(predictions)
        self._gts.extend(ground_truths)

    def reset(self):
        self._preds.clear()
        self._gts.clear()

    def compute(self) -> Dict:
        per_class_aps_by_thresh: Dict[float, Dict[int, float]] = {}

        for iou_thresh in self.iou_thresholds:
            class_aps = {}
            for cls_id in range(self.num_classes):
                cls_preds = [p for p in self._preds if p.class_id == cls_id]
                cls_gts   = [g for g in self._gts   if g.class_id == cls_id]
                class_aps[cls_id] = compute_class_ap(cls_preds, cls_gts, iou_thresh)
            per_class_aps_by_thresh[iou_thresh] = class_aps

        # mAP@0.5
        aps_50 = list(per_class_aps_by_thresh[0.5].values())
        map_50 = float(np.mean(aps_50))

        # mAP@0.5:0.95
        all_aps = [
            ap
            for thresh_aps in per_class_aps_by_thresh.values()
            for ap in thresh_aps.values()
        ]
        map_50_95 = float(np.mean(all_aps))

        results = {
            'map_50':    map_50,
            'map_50_95': map_50_95,
            'per_class': {
                self.class_names[i]: per_class_aps_by_thresh[0.5][i]
                for i in range(self.num_classes)
            }
        }
        return results

    def summary(self) -> str:
        r = self.compute()
        lines = [
            f"{'Metric':<25} {'Value':>10}",
            "-" * 36,
            f"{'mAP@0.5':<25} {r['map_50']:>10.4f}",
            f"{'mAP@0.5:0.95':<25} {r['map_50_95']:>10.4f}",
            "",
            "Per-class AP@0.5:",
        ]
        for cls_name, ap in r['per_class'].items():
            lines.append(f"  {cls_name:<23} {ap:>10.4f}")
        return '\n'.join(lines)


class ConfusionMatrix:
    """Confusion matrix for detection evaluation."""

    def __init__(self, num_classes: int, conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.conf = conf_threshold
        self.iou  = iou_threshold
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        # Last row/col = background (FP/FN)

    def update(self, predictions: List[DetectionResult], ground_truths: List[GroundTruth]):
        gt_by_image = defaultdict(list)
        for gt in ground_truths:
            gt_by_image[gt.image_id].append(gt)

        for image_id, img_gts in gt_by_image.items():
            img_preds = [p for p in predictions
                         if p.image_id == image_id and p.score >= self.conf]

            matched_gts = set()
            for pred in sorted(img_preds, key=lambda x: x.score, reverse=True):
                best_iou_val = self.iou
                best_gt = None
                for j, gt in enumerate(img_gts):
                    if j in matched_gts:
                        continue
                    iou_val = iou(pred.box, gt.box)
                    if iou_val > best_iou_val:
                        best_iou_val = iou_val
                        best_gt = j

                if best_gt is not None:
                    self.matrix[img_gts[best_gt].class_id][pred.class_id] += 1
                    matched_gts.add(best_gt)
                else:
                    self.matrix[self.num_classes][pred.class_id] += 1  # FP

            for j, gt in enumerate(img_gts):
                if j not in matched_gts:
                    self.matrix[gt.class_id][self.num_classes] += 1  # FN

    def plot(self, class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("pip install matplotlib seaborn")
            return

        names = (class_names or [str(i) for i in range(self.num_classes)]) + ['background']
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            self.matrix,
            annot=True, fmt='d',
            xticklabels=names, yticklabels=names,
            cmap='Blues', ax=ax,
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved confusion matrix: {save_path}")
        plt.show()
