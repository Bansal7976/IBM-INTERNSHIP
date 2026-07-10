"""
Convert nuScenes dataset → YOLO format.
Requires: pip install nuscenes-devkit
"""

from __future__ import annotations
from pathlib import Path
import shutil


NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(NUSCENES_CLASSES)}


def prepare_nuscenes_yolo(
    data_root: str,
    output_root: str,
    version: str = 'v1.0-mini',
    cameras: list = None,
) -> str:
    """
    Convert nuScenes dataset to YOLO format for 2D detection.

    Args:
        data_root:   path to nuScenes root (contains v1.0-mini/ etc.)
        output_root: where to write YOLO-format dataset
        version:     'v1.0-mini', 'v1.0-trainval', or 'v1.0-test'
        cameras:     list of camera names to use (default: all 6)

    Returns:
        Path to generated nuscenes.yaml
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import view_points
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "pip install nuscenes-devkit\n"
            "Then: python main.py prepare --dataset nuscenes --data_root <path>"
        )

    data_root   = Path(data_root)
    output_root = Path(output_root)

    if cameras is None:
        cameras = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
        ]

    print(f'\nLoading nuScenes {version} from {data_root}...')
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)

    # Use scenes for split
    all_scenes = nusc.scene
    n_val = max(1, int(len(all_scenes) * 0.2))
    val_scenes = {s['token'] for s in all_scenes[:n_val]}

    # Create output dirs
    for split in ('train', 'val'):
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    train_count = val_count = 0

    print(f'Processing {len(nusc.sample)} samples across {len(cameras)} cameras...')

    for sample in nusc.sample:
        scene_token = sample['scene_token']
        split = 'val' if scene_token in val_scenes else 'train'

        for cam in cameras:
            if cam not in sample['data']:
                continue

            cam_token = sample['data'][cam]
            cam_data  = nusc.get('sample_data', cam_token)
            img_path  = data_root / cam_data['filename']

            if not img_path.exists():
                continue

            with PILImage.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

            # Get calibration
            cs_record  = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])

            import pyquaternion
            import numpy as np

            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            cam_rotation  = pyquaternion.Quaternion(cs_record['rotation'])
            cam_translation = np.array(cs_record['translation'])

            yolo_lines = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat = ann['category_name'].split('.')[0]  # e.g. 'vehicle' → skip; 'human.pedestrian' → 'pedestrian'

                # Map nuScenes categories to our classes
                mapped = None
                for cls in NUSCENES_CLASSES:
                    if cls in ann['category_name']:
                        mapped = cls
                        break
                if mapped is None:
                    continue

                cls_id = CLASS_TO_IDX[mapped]

                # Project 3D box corners to 2D
                from nuscenes.utils.data_classes import Box
                box = nusc.get_box(ann_token)
                box.translate(-cam_translation)
                box.rotate(cam_rotation.inverse)

                corners = view_points(box.corners(), cam_intrinsic, normalize=True)[:2]
                x1, y1 = corners.min(axis=1)
                x2, y2 = corners.max(axis=1)

                # Filter out boxes behind camera or outside image
                if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
                    continue

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                if bw <= 0 or bh <= 0:
                    continue

                yolo_lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

            stem = img_path.stem
            dst_img = output_root / 'images' / split / img_path.name
            dst_lbl = output_root / 'labels' / split / f'{stem}.txt'

            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(yolo_lines))

            if split == 'train':
                train_count += 1
            else:
                val_count += 1

    print(f'\nConversion complete:')
    print(f'  Train: {train_count} images')
    print(f'  Val  : {val_count} images')

    yaml_path = output_root / 'nuscenes.yaml'
    yaml_content = (
        f"# nuScenes 2D Detection — YOLO format\n"
        f"# Generated by prepare_nuscenes.py\n"
        f"\n"
        f"path: {output_root.absolute()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(NUSCENES_CLASSES)}\n"
        f"names: {NUSCENES_CLASSES}\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f'✓ nuscenes.yaml written: {yaml_path}')
    return str(yaml_path)
