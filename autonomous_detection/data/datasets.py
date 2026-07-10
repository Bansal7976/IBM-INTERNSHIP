"""
Unified dataset loaders for KITTI, nuScenes, and Kaggle vehicle datasets.
Wraps standard formats into a common DetectionDataset interface.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import cv2


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class DetectionSample:
    image: np.ndarray          # HxWx3 uint8
    boxes: np.ndarray          # Nx4 xyxy float32
    labels: np.ndarray         # N int64
    image_id: str
    metadata: dict             # sensor info, calibration, etc.


KITTI_CLASSES = {
    'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
    'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7,
}

NUSCENES_CLASSES = {
    'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3,
    'construction_vehicle': 4, 'pedestrian': 5, 'motorcycle': 6,
    'bicycle': 7, 'traffic_cone': 8, 'barrier': 9,
}

COCO_VEHICLE_CLASSES = {
    'car': 2, 'bus': 5, 'truck': 7, 'motorcycle': 3,
    'bicycle': 1, 'person': 0,
}


# ─── KITTI Dataset ──────────────────────────────────────────────────────────

class KITTIDetectionDataset(Dataset):
    """
    KITTI 2D Object Detection dataset.
    Expected structure:
        data_root/
          training/
            image_2/    *.png
            label_2/    *.txt
            calib/      *.txt
          testing/
            image_2/    *.png
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',          # 'train' | 'val' | 'test'
        classes: Optional[List[str]] = None,
        transforms=None,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.img_size = img_size
        self.classes = classes or list(KITTI_CLASSES.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        subset = 'training' if split != 'test' else 'testing'
        self.image_dir = self.data_root / subset / 'image_2'
        self.label_dir = self.data_root / subset / 'label_2'

        all_ids = sorted([p.stem for p in self.image_dir.glob('*.png')])
        if split == 'train':
            self.sample_ids = all_ids[:int(len(all_ids) * 0.8)]
        elif split == 'val':
            self.sample_ids = all_ids[int(len(all_ids) * 0.8):]
        else:
            self.sample_ids = all_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> DetectionSample:
        sample_id = self.sample_ids[idx]
        image = self._load_image(sample_id)
        boxes, labels = self._load_labels(sample_id, image.shape[:2])

        sample = DetectionSample(
            image=image,
            boxes=boxes,
            labels=labels,
            image_id=sample_id,
            metadata={'dataset': 'kitti'},
        )

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, sample_id: str) -> np.ndarray:
        path = self.image_dir / f'{sample_id}.png'
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        return img

    def _load_labels(
        self, sample_id: str, orig_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        label_path = self.label_dir / f'{sample_id}.txt'
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        h, w = orig_shape
        boxes, labels = [], []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                cls_name = parts[0]
                if cls_name not in self.class_to_idx:
                    continue
                # KITTI format: left top right bottom (pixel coords)
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                # Scale to resized image
                x1 = x1 / w * self.img_size[0]
                x2 = x2 / w * self.img_size[0]
                y1 = y1 / h * self.img_size[1]
                y2 = y2 / h * self.img_size[1]
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[cls_name])

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    @staticmethod
    def collate_fn(batch: List[DetectionSample]) -> Dict:
        images = torch.stack([
            torch.from_numpy(s.image).permute(2, 0, 1).float() / 255.0
            for s in batch
        ])
        return {
            'images': images,
            'boxes': [torch.from_numpy(s.boxes) for s in batch],
            'labels': [torch.from_numpy(s.labels) for s in batch],
            'image_ids': [s.image_id for s in batch],
        }


# ─── nuScenes Dataset ────────────────────────────────────────────────────────

class NuScenesDetectionDataset(Dataset):
    """
    nuScenes 2D detection dataset using the front camera (CAM_FRONT).
    Requires: pip install nuscenes-devkit

    Expected structure:
        data_root/
          v1.0-trainval/
          samples/
          sweeps/
    """

    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-trainval',
        split: str = 'train',
        camera: str = 'CAM_FRONT',
        transforms=None,
        img_size: Tuple[int, int] = (640, 640),
    ):
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError:
            raise ImportError("Install nuscenes-devkit: pip install nuscenes-devkit")

        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        self.camera = camera
        self.transforms = transforms
        self.img_size = img_size
        self.classes = list(NUSCENES_CLASSES.keys())
        self.class_to_idx = NUSCENES_CLASSES

        splits = create_splits_scenes()
        split_scenes = set(splits[split])

        self.samples = [
            s for s in self.nusc.sample
            if self.nusc.get('scene', s['scene_token'])['name'] in split_scenes
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> DetectionSample:
        sample = self.samples[idx]
        cam_token = sample['data'][self.camera]
        cam_data = self.nusc.get('sample_data', cam_token)

        img_path = Path(self.nusc.dataroot) / cam_data['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.img_size)

        boxes, labels = self._get_2d_boxes(sample, cam_token, orig_w, orig_h)

        return DetectionSample(
            image=image,
            boxes=boxes,
            labels=labels,
            image_id=sample['token'],
            metadata={'camera': self.camera, 'dataset': 'nuscenes'},
        )

    def _get_2d_boxes(
        self, sample, cam_token: str, orig_w: int, orig_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
        import pyquaternion

        _, boxes, camera_intrinsic = self.nusc.get_sample_data(
            cam_token, box_vis_level=BoxVisibility.ANY
        )

        result_boxes, result_labels = [], []
        for box in boxes:
            cls_name = box.name.split('.')[0] if '.' in box.name else box.name
            if cls_name not in self.class_to_idx:
                continue

            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2]
            x1, y1 = corners.min(axis=1)
            x2, y2 = corners.max(axis=1)

            x1 = np.clip(x1 / orig_w * self.img_size[0], 0, self.img_size[0])
            x2 = np.clip(x2 / orig_w * self.img_size[0], 0, self.img_size[0])
            y1 = np.clip(y1 / orig_h * self.img_size[1], 0, self.img_size[1])
            y2 = np.clip(y2 / orig_h * self.img_size[1], 0, self.img_size[1])

            if x2 - x1 > 2 and y2 - y1 > 2:
                result_boxes.append([x1, y1, x2, y2])
                result_labels.append(self.class_to_idx[cls_name])

        if not result_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(result_boxes, dtype=np.float32), np.array(result_labels, dtype=np.int64)

    @staticmethod
    def collate_fn(batch):
        return KITTIDetectionDataset.collate_fn(batch)


# ─── Generic YOLO-format Dataset ─────────────────────────────────────────────

class YOLOFormatDataset(Dataset):
    """
    Loads any dataset already converted to YOLO format.
    Works for Kaggle datasets (Udacity, aiMotive, etc.) after conversion.

    Structure:
        root/
          images/train/*.jpg
          labels/train/*.txt   (cx cy w h normalized, per line: class cx cy w h)
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (640, 640),
        transforms=None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.transforms = transforms

        self.image_paths = sorted((self.root / 'images' / split).glob('*.[jp][pn]g'))
        self.label_dir = self.root / 'labels' / split

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> DetectionSample:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        label_path = self.label_dir / f'{img_path.stem}.txt'
        boxes, labels = self._load_yolo_labels(label_path)

        return DetectionSample(
            image=image,
            boxes=boxes,
            labels=labels,
            image_id=img_path.stem,
            metadata={'dataset': 'yolo_format'},
        )

    def _load_yolo_labels(
        self, label_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        W, H = self.img_size
        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    @staticmethod
    def collate_fn(batch):
        return KITTIDetectionDataset.collate_fn(batch)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, split: str = 'train') -> Dataset:
    """Build dataset from config dict."""
    name = cfg['name'].lower()

    if name == 'kitti':
        return KITTIDetectionDataset(
            data_root=cfg['data_root'],
            split=split,
            classes=cfg.get('classes'),
            img_size=tuple(cfg.get('img_size', [640, 640])),
        )
    elif name == 'nuscenes':
        return NuScenesDetectionDataset(
            data_root=cfg['data_root'],
            version=cfg.get('version', 'v1.0-trainval'),
            split=split,
            camera=cfg.get('camera', 'CAM_FRONT'),
            img_size=tuple(cfg.get('img_size', [640, 640])),
        )
    elif name == 'yolo':
        return YOLOFormatDataset(
            root=cfg['data_root'],
            split=split,
            img_size=tuple(cfg.get('img_size', [640, 640])),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: kitti, nuscenes, yolo")
