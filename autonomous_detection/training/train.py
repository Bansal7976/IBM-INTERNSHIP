"""
Main training entry point for 2D detection (YOLOv11 / RT-DETR).

Quick start:
    # Single GPU
    python training/train.py --config configs/yolov11/yolov11_kitti.yaml

    # Multi-GPU (single node)
    torchrun --nproc_per_node=4 training/train.py --config configs/yolov11/yolov11_kitti.yaml

    # Resume from checkpoint
    python training/train.py --config configs/yolov11/yolov11_kitti.yaml --resume runs/train/exp/weights/last.pt
"""

import argparse
import os
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Train 2D vehicle detector')
    p.add_argument('--config', required=True, help='Path to training config YAML')
    p.add_argument('--resume', default=None, help='Resume from checkpoint path')
    p.add_argument('--device', default=None, help='Override device (e.g. "0,1,2,3" or "cpu")')
    p.add_argument('--batch', type=int, default=None, help='Override batch size')
    p.add_argument('--epochs', type=int, default=None, help='Override epochs')
    p.add_argument('--workers', type=int, default=None, help='Override dataloader workers')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_ultralytics(cfg: dict, args):
    """Train YOLOv11 / RT-DETR using Ultralytics API."""
    from ultralytics import YOLO

    model_cfg = cfg['model']
    train_cfg = cfg['training']

    # Build model: load pretrained weights for fine-tuning
    weights = model_cfg.get('weights', model_cfg.get('name', 'yolo11m') + '.pt')
    model = YOLO(weights)

    # CLI overrides take precedence
    batch   = args.batch   or train_cfg.get('batch_size', 16)
    epochs  = args.epochs  or train_cfg.get('epochs', 50)
    workers = args.workers or train_cfg.get('workers', 8)
    device  = args.device  or train_cfg.get('device', '0')

    results = model.train(
        data=cfg['dataset']['yaml'],
        epochs=epochs,
        imgsz=cfg['dataset'].get('img_size', 640),
        batch=batch,
        device=device,
        workers=workers,
        project=train_cfg.get('project', 'runs/train'),
        name=train_cfg.get('name', 'exp'),
        pretrained=model_cfg.get('pretrained', True),
        resume=bool(args.resume),
        amp=train_cfg.get('amp', True),
        multi_scale=train_cfg.get('multi_scale', False),
        freeze=model_cfg.get('freeze_layers'),
        lr0=train_cfg.get('lr0', 0.01),
        lrf=train_cfg.get('lrf', 0.01),
        warmup_epochs=train_cfg.get('warmup_epochs', 3),
        optimizer=train_cfg.get('optimizer', 'AdamW'),
        weight_decay=train_cfg.get('weight_decay', 0.0005),
        save_period=train_cfg.get('save_period', 5),
        # Augmentation
        mosaic=train_cfg.get('mosaic', 1.0),
        mixup=train_cfg.get('mixup', 0.15),
        copy_paste=train_cfg.get('copy_paste', 0.3),
        degrees=train_cfg.get('degrees', 10.0),
        translate=train_cfg.get('translate', 0.1),
        scale=train_cfg.get('scale', 0.5),
        fliplr=train_cfg.get('fliplr', 0.5),
        hsv_h=train_cfg.get('hsv_h', 0.015),
        hsv_s=train_cfg.get('hsv_s', 0.7),
        hsv_v=train_cfg.get('hsv_v', 0.4),
    )

    print(f"\nTraining complete. Best weights: {results.save_dir}/weights/best.pt")
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Config: {args.config}")
    print(f"Model:  {cfg['model'].get('name', cfg['model'].get('weights', 'unknown'))}")
    print(f"Data:   {cfg['dataset'].get('yaml', cfg['dataset'].get('name'))}")

    framework = cfg.get('framework', 'ultralytics').lower()

    if framework == 'ultralytics':
        train_ultralytics(cfg, args)
    elif framework == 'mmdet3d':
        print("For MMDetection3D models (BEVFormer, BEVFusion), use:")
        print("  python tools/train.py <config> --launcher pytorch")
        print("  See training/slurm/ for HPC job scripts")
    else:
        raise ValueError(f"Unknown framework: {framework}")


if __name__ == '__main__':
    main()
