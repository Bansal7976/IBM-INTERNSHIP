"""
Evaluate a trained model on KITTI validation split.

Usage:
    python evaluation/evaluate_kitti.py \
        --weights runs/train/exp/weights/best.pt \
        --data_root /path/to/kitti \
        --split val

Outputs: mAP@0.5, mAP@0.5:0.95, per-class AP, confusion matrix
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.datasets import KITTIDetectionDataset, KITTI_CLASSES
from evaluation.metrics import MeanAveragePrecision, ConfusionMatrix, DetectionResult, GroundTruth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to model weights (.pt)')
    p.add_argument('--data_root', required=True, help='KITTI data root')
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--conf', type=float, default=0.001, help='Low conf to keep all for PR curve')
    p.add_argument('--iou', type=float, default=0.6)
    p.add_argument('--device', default='cuda')
    p.add_argument('--classes', nargs='+', default=None)
    p.add_argument('--save_dir', default='runs/eval/kitti')
    return p.parse_args()


def evaluate(args):
    from ultralytics import YOLO
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    classes = args.classes or KITTI_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    dataset = KITTIDetectionDataset(
        data_root=args.data_root,
        split=args.split,
        classes=classes,
        img_size=(640, 640),
    )

    model = YOLO(args.weights)
    metric = MeanAveragePrecision(num_classes=len(classes), class_names=classes)
    cm     = ConfusionMatrix(num_classes=len(classes), conf_threshold=0.25, iou_threshold=0.5)

    print(f"Evaluating on {len(dataset)} KITTI {args.split} samples...")

    all_preds = []
    all_gts   = []

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image_id = sample.image_id

        # Ground truths
        for box, label in zip(sample.boxes, sample.labels):
            all_gts.append(GroundTruth(
                image_id=image_id,
                box=box,
                class_id=int(label),
            ))

        # Predictions
        results = model.predict(
            sample.image,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            r = results[0]
            boxes   = r.boxes.xyxy.cpu().numpy()
            scores  = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cid in zip(boxes, scores, cls_ids):
                if cid < len(classes):
                    all_preds.append(DetectionResult(
                        image_id=image_id,
                        box=box,
                        score=float(score),
                        class_id=int(cid),
                    ))

    metric.update(all_preds, all_gts)
    cm.update(all_preds, all_gts)

    print("\n" + "="*50)
    print("KITTI Evaluation Results")
    print("="*50)
    print(metric.summary())

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cm.plot(class_names=classes, save_path=str(save_dir / 'confusion_matrix.png'))

    results = metric.compute()
    print(f"\nSaved confusion matrix to: {save_dir / 'confusion_matrix.png'}")

    return results


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
