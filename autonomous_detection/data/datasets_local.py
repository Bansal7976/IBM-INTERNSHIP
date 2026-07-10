"""
Dataset classes for KITTI and nuScenes.
Used by training, evaluation, and data preparation pipelines.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Callable
import numpy as np

# ── Class definitions ─────────────────────────────────────────────────────────

KITTI_CLASSES = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]


# ── KITTI Dataset ─────────────────────────────────────────────────────────────

class KITTIDetectionDataset:
    """
    KITTI Object Detection dataset loader (YOLO format).

    Expects data converted to YOLO format via prepare_kitti.py:
        data/kitti_yolo/
            images/train/  images/val/
            labels/train/  labels/val/
            kitti.yaml
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 640,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.class_names = KITTI_CLASSES
        self.num_classes = len(KITTI_CLASSES)

        self.img_dir = self.data_root / 'images' / split
        self.lbl_dir = self.data_root / 'labels' / split

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.img_dir}\n"
                f"Run: python main.py prepare --dataset kitti --data_root <path>"
            )

        self.image_paths = sorted(self.img_dir.glob('*.png')) + \
                           sorted(self.img_dir.glob('*.jpg'))

        print(f'[KITTIDataset] {split}: {len(self.image_paths)} images')

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        import cv2
        img_path = self.image_paths[idx]
        lbl_path = self.lbl_dir / f'{img_path.stem}.txt'

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load labels (YOLO format: cls cx cy w h)
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        # Convert to xyxy pixel coords
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        boxes.append([x1, y1, x2, y2, cls_id])

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5)),
            'image_id': img_path.stem,
            'orig_size': (h, w),
        }

    def get_class_counts(self) -> Dict[str, int]:
        """Count annotations per class."""
        counts = {cls: 0 for cls in self.class_names}
        for img_path in self.image_paths:
            lbl_path = self.lbl_dir / f'{img_path.stem}.txt'
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id < len(self.class_names):
                                counts[self.class_names[cls_id]] += 1
        return counts


# ── nuScenes Dataset ──────────────────────────────────────────────────────────

class NuScenesDetectionDataset:
    """
    nuScenes detection dataset loader (YOLO format).
    Requires data converted via prepare_nuscenes.py.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: int = 640,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.class_names = NUSCENES_CLASSES
        self.num_classes = len(NUSCENES_CLASSES)

        self.img_dir = self.data_root / 'images' / split
        self.lbl_dir = self.data_root / 'labels' / split

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.img_dir}\n"
                f"Run: python main.py prepare --dataset nuscenes --data_root <path>"
            )

        self.image_paths = sorted(self.img_dir.glob('*.jpg')) + \
                           sorted(self.img_dir.glob('*.png'))

        print(f'[nuScenesDataset] {split}: {len(self.image_paths)} images')

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        import cv2
        img_path = self.image_paths[idx]
        lbl_path = self.lbl_dir / f'{img_path.stem}.txt'

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        boxes.append([x1, y1, x2, y2, cls_id])

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5)),
            'image_id': img_path.stem,
            'orig_size': (h, w),
        }


# ── Dataset Factory ───────────────────────────────────────────────────────────

def build_dataset(cfg: dict, split: str = 'train'):
    """
    Build dataset from config dict.

    Example cfg:
        {'name': 'kitti', 'yaml': 'data/kitti_yolo/kitti.yaml', 'img_size': 640}
        {'name': 'nuscenes', 'yaml': 'data/nuscenes_yolo/nuscenes.yaml'}
    """
    name = cfg.get('name', 'kitti').lower()
    img_size = cfg.get('img_size', 640)

    # Derive data_root from yaml path
    yaml_path = cfg.get('yaml', '')
    if yaml_path:
        data_root = str(Path(yaml_path).parent)
    else:
        data_root = cfg.get('data_root', f'data/{name}_yolo')

    if name == 'kitti':
        return KITTIDetectionDataset(data_root, split=split, img_size=img_size)
    elif name == 'nuscenes':
        return NuScenesDetectionDataset(data_root, split=split, img_size=img_size)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: kitti, nuscenes")
