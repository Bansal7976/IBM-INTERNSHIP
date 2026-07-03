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
    import torch, numpy as np
    from tqdm import tqdm

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    classes = args.classes or KITTI_CLASSES

    # KITTIDetectionDataset expects the YOLO-converted root (kitti_yolo)
    data_path = Path(args.data_root)
    if (data_path / 'images').exists():
        kitti_yolo_root = str(data_path)           # already kitti_yolo dir
    else:
        kitti_yolo_root = str(data_path.parent / 'kitti_yolo')

    dataset = KITTIDetectionDataset(
        data_root=kitti_yolo_root,
        split=args.split,
        img_size=640,
    )

    model  = YOLO(args.weights)
    metric = MeanAveragePrecision(num_classes=len(classes), class_names=classes)
    cm     = ConfusionMatrix(num_classes=len(classes), conf_threshold=0.25, iou_threshold=0.5)

    print(f"Evaluating on {len(dataset)} KITTI {args.split} samples...")
    print(f"Device: {device}")

    all_preds = []
    all_gts   = []

    for idx in tqdm(range(len(dataset))):
        sample   = dataset[idx]
        image_id = sample['image_id']
        boxes_np = sample['boxes']   # Nx5: [x1,y1,x2,y2,cls_id]

        for row in boxes_np:
            x1, y1, x2, y2, cls_id = row
            all_gts.append(GroundTruth(
                image_id=image_id,
                box=np.array([x1, y1, x2, y2]),
                class_id=int(cls_id),
            ))

        img_path = str(dataset.img_dir / f'{image_id}.png')
        results = model.predict(img_path, conf=args.conf, iou=args.iou,
                                device=device, verbose=False)
        if results and results[0].boxes is not None:
            r = results[0]
            for box, score, cid in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy().astype(int)):
                if cid < len(classes):
                    all_preds.append(DetectionResult(
                        image_id=image_id, box=box,
                        score=float(score), class_id=int(cid),
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

    results_dict = metric.compute()
    print(f"\nSaved confusion matrix → {save_dir / 'confusion_matrix.png'}")
    return results_dict


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
