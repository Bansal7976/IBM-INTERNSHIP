"""
Autonomous Vehicle & Object Detection System — Main Entry Point
Complete execution pipeline for training, inference, and evaluation.

Usage:
    # Train 2D detection on KITTI
    python main.py train --config configs/yolov11/yolov11_kitti.yaml --device 0,1,2,3

    # Run inference on video with tracking and collision detection
    python main.py infer --source video.mp4 --model yolo11m --weights runs/train/exp/weights/best.pt --track

    # Evaluate on KITTI validation set
    python main.py eval --weights runs/train/exp/weights/best.pt --data_root data/kitti --split val

    # Export model to ONNX
    python main.py export --weights best.pt --format onnx
"""

import argparse
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import train_ultralytics
from inference.pipeline import DetectionPipeline
from evaluation.evaluate_kitti import evaluate as eval_kitti
from inference.export import export_model
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Autonomous Vehicle Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train YOLOv11 on KITTI
  python main.py train --config configs/yolov11/yolov11_kitti.yaml

  # Run detection + tracking on video
  python main.py infer --source video.mp4 --model yolo11m --track --save

  # Evaluate trained model
  python main.py eval --weights best.pt --data_root data/kitti

  # Export to ONNX
  python main.py export --weights best.pt --format onnx
        ''')

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # ── TRAIN ────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser('train', help='Train detection model')
    train_parser.add_argument('--config', required=True, help='Path to training config YAML')
    train_parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    train_parser.add_argument('--device', default=None, help='Device override (e.g. 0,1,2,3)')
    train_parser.add_argument('--batch', type=int, default=None, help='Batch size override')
    train_parser.add_argument('--epochs', type=int, default=None, help='Epochs override')

    # ── INFER ────────────────────────────────────────────────────────────────
    infer_parser = subparsers.add_parser('infer', help='Run inference on image/video')
    infer_parser.add_argument('--source', required=True, help='Image, video, or webcam (0)')
    infer_parser.add_argument('--model', default='yolo11m', help='Model type')
    infer_parser.add_argument('--weights', default=None, help='Fine-tuned weights path')
    infer_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    infer_parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    infer_parser.add_argument('--track', action='store_true', help='Enable tracking')
    infer_parser.add_argument('--tracker', default='bytetrack', choices=['bytetrack', 'botsort'])
    infer_parser.add_argument('--save', action='store_true', help='Save output')
    infer_parser.add_argument('--save_dir', default='runs/inference', help='Save directory')
    infer_parser.add_argument('--device', default=None, help='Device (auto-detected: cuda or cpu)')
    infer_parser.add_argument('--show', action='store_true', help='Show preview')

    # ── EVAL ─────────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser('eval', help='Evaluate on benchmark dataset')
    eval_parser.add_argument('--weights', required=True, help='Trained model weights')
    eval_parser.add_argument('--data_root', required=True, help='Dataset root path')
    eval_parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    eval_parser.add_argument('--conf', type=float, default=0.001, help='Low conf for PR curve')
    eval_parser.add_argument('--device', default=None, help='Device (auto-detected: cuda or cpu)')
    eval_parser.add_argument('--save_dir', default='runs/eval', help='Save directory')

    # ── EXPORT ───────────────────────────────────────────────────────────────
    export_parser = subparsers.add_parser('export', help='Export model to ONNX/TensorRT')
    export_parser.add_argument('--weights', required=True, help='Model weights path')
    export_parser.add_argument('--format', default='onnx',
                              choices=['onnx', 'engine', 'coreml', 'tflite', 'openvino'])
    export_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    export_parser.add_argument('--half', action='store_true', help='FP16 quantization')
    export_parser.add_argument('--int8', action='store_true', help='INT8 quantization')
    export_parser.add_argument('--dynamic', action='store_true', help='Dynamic shapes')

    # ── PREPARE ──────────────────────────────────────────────────────────────
    prep_parser = subparsers.add_parser('prepare', help='Prepare datasets')
    prep_parser.add_argument('--dataset', required=True, choices=['kitti', 'nuscenes'])
    prep_parser.add_argument('--data_root', required=True, help='Raw dataset path')
    prep_parser.add_argument('--output_root', default=None, help='Output path')
    prep_parser.add_argument('--version', default='v1.0-mini', help='nuScenes version')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # ────────────────────────────────────────────────────────────────────────
    # TRAIN
    # ────────────────────────────────────────────────────────────────────────
    if args.command == 'train':
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION — TRAINING')
        print('='*70 + '\n')

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        print(f'Config:   {args.config}')
        print(f'Model:    {cfg["model"].get("name")}')
        print(f'Dataset:  {cfg["dataset"].get("name")}')
        print(f'Epochs:   {cfg["training"].get("epochs")}')
        print()

        train_ultralytics(cfg, args)

    # ────────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ────────────────────────────────────────────────────────────────────────
    elif args.command == 'infer':
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION — INFERENCE')
        print('='*70 + '\n')

        pipeline = DetectionPipeline(
            model_type=args.model,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            enable_tracking=args.track,
            tracker_type=args.tracker,
        )

        print(f'Model:     {args.model}')
        print(f'Weights:   {args.weights or "(pre-trained COCO)"}')
        print(f'Tracking:  {args.track}')
        print(f'Source:    {args.source}')
        print()

        pipeline.run_video(
            source=args.source,
            save=args.save,
            save_dir=args.save_dir,
            show=args.show,
        )

    # ────────────────────────────────────────────────────────────────────────
    # EVAL
    # ────────────────────────────────────────────────────────────────────────
    elif args.command == 'eval':
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION — EVALUATION')
        print('='*70 + '\n')

        print(f'Weights:   {args.weights}')
        print(f'Dataset:   {args.data_root}')
        print(f'Split:     {args.split}')
        print()

        eval_kitti(argparse.Namespace(
            weights=args.weights,
            data_root=args.data_root,
            split=args.split,
            conf=args.conf,
            device=args.device,
            save_dir=args.save_dir,
            classes=None,
        ))

    # ────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ────────────────────────────────────────────────────────────────────────
    elif args.command == 'export':
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION — EXPORT')
        print('='*70 + '\n')

        print(f'Weights:   {args.weights}')
        print(f'Format:    {args.format}')
        print(f'Imgsz:     {args.imgsz}')
        print(f'FP16:      {args.half}')
        print(f'INT8:      {args.int8}')
        print()

        export_model(
            weights=args.weights,
            fmt=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            dynamic=args.dynamic,
        )

    # ────────────────────────────────────────────────────────────────────────
    # PREPARE
    # ────────────────────────────────────────────────────────────────────────
    elif args.command == 'prepare':
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION — PREPARE DATASET')
        print('='*70 + '\n')

        if args.dataset == 'kitti':
            from data.prepare_kitti import convert_kitti_to_yolo, verify_kitti_structure
            if not verify_kitti_structure(args.data_root):
                return
            output = args.output_root or str(Path(args.data_root).parent / 'kitti_yolo')
            print(f'Converting KITTI → YOLO format → {output}\n')
            convert_kitti_to_yolo(args.data_root, output)

        elif args.dataset == 'nuscenes':
            from data.prepare_nuscenes import prepare_nuscenes_yolo
            output = args.output_root or str(Path(args.data_root).parent / 'nuscenes_yolo')
            print(f'Converting nuScenes → YOLO format → {output}\n')
            prepare_nuscenes_yolo(
                args.data_root,
                output,
                version=args.version,
            )

        print(f'[OK] Dataset preparation complete.')


if __name__ == '__main__':
    main()
