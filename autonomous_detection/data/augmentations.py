"""
Augmentation pipeline for vehicle detection training.
Implements Albumentations-based transforms tuned for autonomous driving.
"""

import numpy as np
from dataclasses import replace

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def build_train_transforms(img_size: int = 640, mosaic: bool = True):
    """
    Strong augmentation pipeline for training.
    Order: spatial → color → normalize.
    """
    assert HAS_ALBUMENTATIONS, "pip install albumentations"

    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        # Simulate driving conditions
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomRain(blur_value=2, brightness_coefficient=0.8, p=1.0),
            A.RandomSunFlare(p=1.0),
        ], p=0.2),
        # Color/brightness
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))


def build_val_transforms(img_size: int = 640):
    assert HAS_ALBUMENTATIONS, "pip install albumentations"

    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class AlbumentationsWrapper:
    """Wraps Albumentations transform to work with DetectionSample."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        result = self.transform(
            image=sample.image,
            bboxes=sample.boxes.tolist(),
            class_labels=sample.labels.tolist(),
        )
        import numpy as np
        boxes = np.array(result['bboxes'], dtype=np.float32) if result['bboxes'] else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(result['class_labels'], dtype=np.int64) if result['class_labels'] else np.zeros(0, dtype=np.int64)
        return replace(sample, image=result['image'], boxes=boxes, labels=labels)
