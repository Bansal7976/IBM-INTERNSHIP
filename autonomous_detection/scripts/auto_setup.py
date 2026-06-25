"""
Auto-setup script — Downloads models, prepares datasets, checks everything.
Run this once after environment setup.

Usage:
    python scripts/auto_setup.py

This script will:
    1. Download pre-trained model weights (COCO)
    2. Check dataset paths
    3. Prepare datasets if they exist
    4. Verify GPU/CUDA
    5. Run a quick inference test
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_gpu():
    """Verify GPU/CUDA is available."""
    print('\n' + '='*70)
    print('  Checking GPU...')
    print('='*70)

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f'✓ GPU Available: {device_name}')
            print(f'  Total GPUs: {device_count}')
            return True
        else:
            print('⚠ GPU NOT available — will use CPU (slow)')
            print('  Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
            return False
    except Exception as e:
        print(f'✗ Error: {e}')
        return False


def download_pretrained_models():
    """Auto-download YOLO pre-trained weights."""
    print('\n' + '='*70)
    print('  Downloading Pre-trained Model Weights...')
    print('='*70)

    try:
        from ultralytics import YOLO

        models_to_download = [
            'yolo11n.pt',  # nano
            'yolo11m.pt',  # medium (recommended)
            'yolo11x.pt',  # extra-large
        ]

        for model_name in models_to_download:
            print(f'\nDownloading {model_name}...')
            try:
                model = YOLO(model_name)
                print(f'✓ {model_name} ready')
            except Exception as e:
                print(f'⚠ Could not download {model_name}: {e}')

        return True
    except ImportError:
        print('✗ ultralytics not installed')
        return False


def check_datasets():
    """Check which datasets are available."""
    print('\n' + '='*70)
    print('  Checking Datasets...')
    print('='*70)

    datasets = {
        'KITTI': PROJECT_ROOT / 'data' / 'kitti',
        'nuScenes': PROJECT_ROOT / 'data' / 'nuscenes',
        'KITTI (YOLO format)': PROJECT_ROOT / 'data' / 'kitti_yolo',
        'nuScenes (YOLO format)': PROJECT_ROOT / 'data' / 'nuscenes_yolo',
    }

    available = []
    missing = []

    for name, path in datasets.items():
        if path.exists():
            available.append(name)
            print(f'✓ {name}: {path}')
        else:
            missing.append(name)
            print(f'✗ {name}: NOT found')

    if missing:
        print(f'\n⚠ {len(missing)} datasets missing. To use them:')
        print('\n1. KITTI:')
        print('   - Register: https://www.cvlibs.net/datasets/kitti/')
        print('   - Download: data_object_image_2.zip, data_object_label_2.zip')
        print('   - Extract to: data/kitti/')
        print('   - Prepare: python main.py prepare --dataset kitti --data_root data/kitti')
        print('\n2. nuScenes:')
        print('   - Register: https://www.nuscenes.org/')
        print('   - Download: v1.0-mini (400MB) or v1.0-trainval (350GB)')
        print('   - Extract to: data/nuscenes/')
        print('   - Prepare: python main.py prepare --dataset nuscenes --data_root data/nuscenes')

    return available, missing


def test_inference():
    """Quick inference test on a sample image."""
    print('\n' + '='*70)
    print('  Testing Inference...')
    print('='*70)

    try:
        from ultralytics import YOLO
        import tempfile
        import urllib.request
        import cv2

        print('Downloading test image...')
        test_url = 'https://ultralytics.com/images/bus.jpg'
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            urllib.request.urlretrieve(test_url, f.name)
            test_image = f.name

        print('Loading YOLOv11m...')
        model = YOLO('yolo11m.pt')

        print('Running inference...')
        results = model.predict(test_image, verbose=False)

        if results and results[0].boxes is not None:
            n_det = len(results[0].boxes)
            print(f'✓ Inference works! Detected {n_det} objects')

            # Show detections
            for box, score, cls in zip(
                results[0].boxes.xyxy,
                results[0].boxes.conf,
                results[0].boxes.cls,
            ):
                class_name = results[0].names[int(cls)]
                print(f'  - {class_name}: {float(score):.2f}')

            # Save result
            output_path = PROJECT_ROOT / 'test_inference_output.jpg'
            annotated = results[0].plot()
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f'✓ Annotated image saved: {output_path}')
        else:
            print('⚠ No detections on test image')

        return True
    except Exception as e:
        print(f'✗ Inference test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def verify_project_structure():
    """Check all required directories exist."""
    print('\n' + '='*70)
    print('  Verifying Project Structure...')
    print('='*70)

    required_dirs = [
        'data',
        'models',
        'training',
        'inference',
        'evaluation',
        'configs',
        'scripts',
        'notebooks',
    ]

    all_ok = True
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f'✓ {dir_name}/')
        else:
            print(f'✗ {dir_name}/ MISSING')
            all_ok = False

    return all_ok


def main():
    print('\n')
    print('╔' + '='*68 + '╗')
    print('║' + ' '*15 + 'AUTONOMOUS DETECTION — AUTO SETUP' + ' '*20 + '║')
    print('╚' + '='*68 + '╝')

    # 1. Check project structure
    structure_ok = verify_project_structure()
    if not structure_ok:
        print('\n✗ Project structure incomplete!')
        return False

    # 2. Check GPU
    gpu_available = check_gpu()

    # 3. Download pre-trained models
    models_ok = download_pretrained_models()

    # 4. Check datasets
    available_datasets, missing_datasets = check_datasets()

    # 5. Test inference
    inference_ok = test_inference()

    # Summary
    print('\n' + '='*70)
    print('  SETUP SUMMARY')
    print('='*70)
    print(f'GPU Available:        {("✓ Yes" if gpu_available else "✗ No (will use CPU)")}')
    print(f'Pre-trained Models:   {"✓ Downloaded" if models_ok else "✗ Failed"}')
    print(f'Datasets Available:   {len(available_datasets)}/{len(available_datasets) + len(missing_datasets)}')
    print(f'Inference Test:       {"✓ Passed" if inference_ok else "✗ Failed"}')

    if inference_ok:
        print('\n' + '='*70)
        print('  ✓ SETUP COMPLETE!')
        print('='*70)
        print('\nNext steps:')
        print('  1. Run examples:')
        print('     python examples.py 1     # Basic detection')
        print('     python examples.py 6     # Full ADAS pipeline')
        print('\n  2. Prepare dataset:')
        print('     python main.py prepare --dataset kitti --data_root data/kitti')
        print('\n  3. Start training:')
        print('     python main.py train --config configs/yolov11/yolov11_kitti.yaml')
        print('\n  Documentation: cat QUICKSTART.md')
        return True
    else:
        print('\n⚠ Setup incomplete — some issues detected')
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print('\n\nSetup cancelled.')
        sys.exit(1)
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
