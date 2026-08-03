from .datasets import (
    KITTIDetectionDataset,
    NuScenesDetectionDataset,
    YOLOFormatDataset,
    build_dataset,
    KITTI_CLASSES,
)
# BUG FIX: this used to (incorrectly) try `from .datasets import
# NUSCENES_DETECTION_CLASSES`, but that name only ever existed in
# prepare_nuscenes.py (datasets.py has the differently-named NUSCENES_CLASSES
# dict) — so `import data` / `from data import ...` raised ImportError for
# anyone who actually did a package import instead of running scripts
# directly (`python data/prepare_kitti.py ...`), which is why this went
# unnoticed. Import it from where it's actually defined.
from .prepare_nuscenes import NUSCENES_DETECTION_CLASSES
from .augmentations import build_train_transforms, build_val_transforms, AlbumentationsWrapper

__all__ = [
    'KITTIDetectionDataset', 'NuScenesDetectionDataset', 'YOLOFormatDataset',
    'build_dataset', 'KITTI_CLASSES', 'NUSCENES_DETECTION_CLASSES',
    'build_train_transforms', 'build_val_transforms', 'AlbumentationsWrapper',
]
