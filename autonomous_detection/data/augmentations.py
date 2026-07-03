"""
Augmentation pipelines using Albumentations for vehicle detection training.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def get_train_transforms(img_size: int = 640):
    """
    Training augmentation pipeline.
    Optimised for vehicle / pedestrian detection in driving scenarios.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("pip install albumentations")

    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomRain(p=0.1),           # adverse weather
        A.RandomFog(p=0.1),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=0,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def get_val_transforms(img_size: int = 640):
    """
    Validation transform — resize + normalize only, no augmentation.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("pip install albumentations")

    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
    ))


def get_inference_transform(img_size: int = 640):
    """
    Inference transform — letterbox resize + normalize.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError("pip install albumentations")

    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=(114, 114, 114),
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
