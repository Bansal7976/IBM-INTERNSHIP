"""
Example usage patterns for the Autonomous Detection System.
Copy-paste ready code snippets.

Run: python examples.py <example_name>
"""

import sys
import cv2
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Basic Object Detection (Zero Setup)
# ════════════════════════════════════════════════════════════════════════════════

def example_basic_detection():
    """
    Run YOLOv11 on an image using pre-trained COCO weights.
    No training needed — works out of the box.
    """
    from ultralytics import YOLO

    print('\n' + '='*70)
    print('  EXAMPLE 1: Basic Object Detection')
    print('='*70 + '\n')

    # Load pre-trained model
    model = YOLO('yolo11m.pt')  # auto-downloads

    # Detect on image
    image_url = 'https://ultralytics.com/images/bus.jpg'
    results = model.predict(image_url, conf=0.25, verbose=False)

    # Print results
    r = results[0]
    print(f'Detections: {len(r.boxes)}')
    for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        class_name = r.names[int(cls)]
        print(f'  {class_name:15s} confidence={float(score):.3f}')

    # Save annotated image
    annotated_img = r.plot()
    cv2.imwrite('example1_detection.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f'\nAnnotated image saved: example1_detection.jpg')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Multi-Object Tracking on Video
# ════════════════════════════════════════════════════════════════════════════════

def example_tracking():
    """
    Run object detection + ByteTrack on a video.
    Assigns persistent track IDs to vehicles.
    """
    from ultralytics import YOLO

    print('\n' + '='*70)
    print('  EXAMPLE 2: Multi-Object Tracking')
    print('='*70 + '\n')

    model = YOLO('yolo11m.pt')

    # Use a sample video (replace with your own)
    video_source = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Example

    print(f'Tracking on: {video_source}')
    print('Press Q to stop\n')

    # Track on video (or webcam: source=0)
    results = model.track(
        source=video_source,
        conf=0.3,
        iou=0.45,
        tracker='bytetrack.yaml',
        persist=True,
        device=0,
        stream=True,
    )

    for i, result in enumerate(results):
        if i % 10 == 0:  # Print every 10 frames
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().tolist()
                print(f'Frame {i}: {len(track_ids)} tracks')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print('\nTracking complete.')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Custom Model Fine-Tuning on KITTI
# ════════════════════════════════════════════════════════════════════════════════

def example_training():
    """
    Fine-tune YOLOv11 on KITTI dataset.
    Requires: KITTI dataset downloaded and converted to YOLO format.
    """
    from ultralytics import YOLO

    print('\n' + '='*70)
    print('  EXAMPLE 3: Model Fine-Tuning (KITTI)')
    print('='*70 + '\n')

    # Check if dataset exists
    kitti_yaml = Path('data/kitti_yolo/kitti.yaml')
    if not kitti_yaml.exists():
        print(f'ERROR: Dataset not found at {kitti_yaml}')
        print('Prepare KITTI first:')
        print('  python main.py prepare --dataset kitti --data_root ./data/kitti')
        return

    # Load pre-trained model
    model = YOLO('yolo11m.pt')

    # Train on KITTI
    print('Training on KITTI (this will take ~2-4 hours on a single A100 GPU)\n')

    results = model.train(
        data=str(kitti_yaml),
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        project='runs/train',
        name='kitti_finetune',
        amp=True,
        patience=20,  # early stopping
    )

    print(f'\nTraining complete!')
    print(f'Best weights: {results.save_dir}/weights/best.pt')
    print(f'mAP@50: {results.results_dict["metrics/mAP50(B)"]:.4f}')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Evaluate on Benchmark Dataset
# ════════════════════════════════════════════════════════════════════════════════

def example_evaluation():
    """
    Evaluate a fine-tuned model on KITTI validation set.
    Reports mAP, confusion matrix, per-class performance.
    """
    from evaluation.evaluate_kitti import evaluate as eval_kitti
    import argparse

    print('\n' + '='*70)
    print('  EXAMPLE 4: Model Evaluation')
    print('='*70 + '\n')

    weights = Path('runs/train/kitti_finetune/weights/best.pt')
    if not weights.exists():
        print(f'ERROR: Weights not found at {weights}')
        print('Train a model first using Example 3')
        return

    print(f'Evaluating: {weights}')
    print(f'Dataset: KITTI validation split\n')

    # Run evaluation
    args = argparse.Namespace(
        weights=str(weights),
        data_root='data/kitti',
        split='val',
        conf=0.001,
        device='cuda',
        save_dir='runs/eval',
        classes=None,
    )

    results = eval_kitti(args)

    print(f'\nEvaluation Results:')
    print(f'  mAP@0.5: {results["map_50"]:.4f}')
    print(f'  mAP@0.5:0.95: {results["map_50_95"]:.4f}')
    print(f'\nPer-class AP@0.5:')
    for class_name, ap in results['per_class'].items():
        print(f'  {class_name:20s}: {ap:.4f}')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Export Model for Production
# ════════════════════════════════════════════════════════════════════════════════

def example_export():
    """
    Export trained model to ONNX (universal) or TensorRT (NVIDIA-optimized).
    ONNX works on CPU/GPU, TensorRT is 3–5× faster on NVIDIA GPUs.
    """
    from inference.export import export_model

    print('\n' + '='*70)
    print('  EXAMPLE 5: Model Export')
    print('='*70 + '\n')

    weights = Path('runs/train/kitti_finetune/weights/best.pt')

    # ONNX (universal, works everywhere)
    print('Exporting to ONNX...')
    onnx_path = export_model(
        weights=str(weights),
        fmt='onnx',
        imgsz=640,
        simplify=True,
    )
    print(f'[OK] ONNX: {onnx_path}\n')

    # TensorRT (NVIDIA GPU only, 3–5× faster)
    print('Exporting to TensorRT (FP16)...')
    trt_path = export_model(
        weights=str(weights),
        fmt='engine',
        imgsz=640,
        half=True,  # FP16 quantization
    )
    print(f'[OK] TensorRT: {trt_path}\n')

    print('Use exported models:')
    print('  from ultralytics import YOLO')
    print(f'  model = YOLO("{onnx_path}")')
    print('  results = model.predict("video.mp4")')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Full ADAS Perception Pipeline
# ════════════════════════════════════════════════════════════════════════════════

def example_adas_pipeline():
    """
    Complete ADAS perception: detection + tracking + collision detection.
    Combines all modules into one integrated system.
    """
    from inference.advanced_pipeline import ADASPerceptionSystem
    import cv2

    print('\n' + '='*70)
    print('  EXAMPLE 6: Full ADAS Perception Pipeline')
    print('='*70 + '\n')

    # Initialize system
    print('Initializing ADAS perception...')
    adas = ADASPerceptionSystem(
        detector='yolo11m',
        detector_weights=None,  # pre-trained COCO
        enable_lane=True,
        enable_depth=True,
        enable_collision=True,
        device='cuda',
    )

    print('ADAS system ready.\n')

    # Process a test image
    test_image = 'https://ultralytics.com/images/bus.jpg'
    import urllib.request
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        urllib.request.urlretrieve(test_image, f.name)
        frame = cv2.imread(f.name)

    print('Processing frame...')
    result = adas.process_frame(frame)

    print(f'\nResults:')
    print(f'  Detections: {len(result.detections)}')
    for det in result.detections:
        print(f'    - {det["class_name"]} ({det["score"]:.2f})')

    print(f'  Collision alerts: {len(result.collision_alerts)}')
    for alert in result.collision_alerts:
        print(f'    - {alert["class"]}: TTC={alert["ttc_s"]:.1f}s ({alert["risk"]})')

    print(f'  Latency: {result.latency_ms:.1f}ms')

    # Visualize
    annotated = adas.visualize(result)
    cv2.imwrite('example6_adas.jpg', annotated)
    print(f'\nAnnotated image saved: example6_adas.jpg')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: Open-Vocabulary Detection (Grounding DINO)
# ════════════════════════════════════════════════════════════════════════════════

def example_grounding_dino():
    """
    Zero-shot detection using text prompts.
    Detects arbitrary objects without training.
    """
    print('\n' + '='*70)
    print('  EXAMPLE 7: Open-Vocabulary Detection (Grounding DINO)')
    print('='*70 + '\n')

    from models.detector_2d import GroundingDINODetector
    import cv2
    import urllib.request
    import tempfile

    # Download test image
    test_url = 'https://ultralytics.com/images/bus.jpg'
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        urllib.request.urlretrieve(test_url, f.name)
        image = cv2.imread(f.name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Grounding DINO
    print('Loading Grounding DINO...')
    detector = GroundingDINODetector(backbone='swinb')

    # Detect objects by text prompt
    prompt = 'car . truck . bus . person . bicycle .'
    print(f'Prompt: {prompt}\n')

    detections = detector.predict(image, prompt=prompt)

    print(f'Detections: {len(detections)}')
    for det in detections:
        print(f'  {det.class_name:20s} confidence={det.score:.3f}')

    print(f'\nNo training required! Works on any object class via text prompt.')


# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: Multi-GPU Distributed Training
# ════════════════════════════════════════════════════════════════════════════════

def example_ddp_training():
    """
    Multi-GPU training using PyTorch DDP (Distributed Data Parallel).
    Scales across multiple GPUs on a single node or multiple nodes.
    """
    print('\n' + '='*70)
    print('  EXAMPLE 8: Multi-GPU Distributed Training')
    print('='*70 + '\n')

    print('Single node, 4 GPUs:')
    print('  torchrun --nproc_per_node=4 training/train_ddp.py \\')
    print('      --config configs/yolov11/yolov11_kitti.yaml\n')

    print('Multi-node (HPC), 32 GPUs (4 nodes × 8 GPUs):')
    print('  sbatch training/slurm/train_multi_node.sh\n')

    print('Key concepts:')
    print('  - Each GPU holds complete model copy')
    print('  - Gradients synchronized after backward pass')
    print('  - Effective batch size = batch_per_gpu × num_gpus')
    print('  - Linear LR scaling: lr = base_lr × (batch / 256)\n')

    print('Enable on your config:')
    print('  device: "0,1,2,3"  # 4 GPUs, single node\n')


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    examples = {
        '1': ('Basic Detection', example_basic_detection),
        '2': ('Tracking', example_tracking),
        '3': ('Training', example_training),
        '4': ('Evaluation', example_evaluation),
        '5': ('Export', example_export),
        '6': ('ADAS Pipeline', example_adas_pipeline),
        '7': ('Grounding DINO', example_grounding_dino),
        '8': ('DDP Training', example_ddp_training),
    }

    if len(sys.argv) < 2:
        print('\n' + '='*70)
        print('  AUTONOMOUS DETECTION SYSTEM — EXAMPLES')
        print('='*70)
        print('\nUsage: python examples.py <number>\n')
        print('Available examples:')
        for num, (name, _) in sorted(examples.items()):
            print(f'  {num}. {name}')
        print('\nExample: python examples.py 1\n')
        sys.exit(0)

    ex_num = sys.argv[1]
    if ex_num not in examples:
        print(f'ERROR: Unknown example "{ex_num}"')
        sys.exit(1)

    name, func = examples[ex_num]
    try:
        func()
    except KeyboardInterrupt:
        print('\n\nInterrupted.')
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
