"""
nuScenes dataset preparation and YOLO-format conversion.

Usage:
    python data/prepare_nuscenes.py \
        --data_root /path/to/nuscenes \
        --output_root ./data/nuscenes_yolo \
        --version v1.0-mini    # use v1.0-trainval for full dataset

Download nuScenes:
    Register at: https://www.nuscenes.org/sign-up
    Mini (4 scenes, ~400MB) is great for quick testing.
    Full trainval (700 scenes, ~350GB) for production.
"""

import argparse
import shutil
import json
from pathlib import Path
from typing import List, Dict


NUSCENES_DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]


def prepare_nuscenes_yolo(
    data_root: str,
    output_root: str,
    version: str = 'v1.0-mini',
    camera: str = 'CAM_FRONT',
    classes: List[str] = None,
):
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        from nuscenes.utils.geometry_utils import view_points, BoxVisibility
    except ImportError:
        raise ImportError("pip install nuscenes-devkit")

    classes = classes or NUSCENES_DETECTION_CLASSES
    class_to_idx = {c: i for i, c in enumerate(classes)}

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    splits = create_splits_scenes()

    split_map = {'train': splits['train'], 'val': splits['val']}
    if version == 'v1.0-mini':
        split_map = {'train': splits['mini_train'], 'val': splits['mini_val']}

    output_root = Path(output_root)

    for split_name, scene_names in split_map.items():
        img_out = output_root / 'images' / split_name
        lbl_out = output_root / 'labels' / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        scene_name_set = set(scene_names)
        samples = [
            s for s in nusc.sample
            if nusc.get('scene', s['scene_token'])['name'] in scene_name_set
        ]

        print(f"{split_name}: processing {len(samples)} samples...")
        converted = 0

        for sample in samples:
            cam_token = sample['data'][camera]
            cam_data = nusc.get('sample_data', cam_token)

            # Get image
            img_src = Path(data_root) / cam_data['filename']
            sample_token = sample['token']
            img_dst = img_out / f'{sample_token}.jpg'
            shutil.copy2(img_src, img_dst)

            # Get 2D boxes
            import cv2
            img = cv2.imread(str(img_src))
            orig_h, orig_w = img.shape[:2]

            _, boxes, cam_intrinsic = nusc.get_sample_data(
                cam_token, box_vis_level=BoxVisibility.ANY
            )

            yolo_lines = []
            for box in boxes:
                cls_name = box.name.split('.')[0]
                if cls_name not in class_to_idx:
                    continue

                corners = view_points(box.corners(), cam_intrinsic, normalize=True)[:2]
                x1, y1 = corners.min(axis=1)
                x2, y2 = corners.max(axis=1)

                x1, x2 = max(0, x1), min(orig_w, x2)
                y1, y2 = max(0, y1), min(orig_h, y2)

                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                cx = ((x1 + x2) / 2) / orig_w
                cy = ((y1 + y2) / 2) / orig_h
                bw = (x2 - x1) / orig_w
                bh = (y2 - y1) / orig_h

                yolo_lines.append(
                    f'{class_to_idx[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'
                )

            lbl_path = lbl_out / f'{sample_token}.txt'
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            converted += 1

        print(f"  {split_name}: {converted} samples written")

    # Dataset YAML
    yaml_content = f"""path: {output_root.absolute()}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    yaml_path = output_root / 'nuscenes.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML: {yaml_path}")
    print(f"Train with: yolo train data={yaml_path} model=yolo11m.pt epochs=50")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--version', default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--camera', default='CAM_FRONT')
    args = parser.parse_args()

    prepare_nuscenes_yolo(
        args.data_root,
        args.output_root,
        version=args.version,
        camera=args.camera,
    )
