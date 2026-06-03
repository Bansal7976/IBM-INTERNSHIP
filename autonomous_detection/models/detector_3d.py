"""
3D Object Detection wrappers for BEV (Bird's Eye View) perception.

Models:
  - BEVFormer   — camera-only 3D detection via spatial-temporal transformers (ECCV 2022)
  - BEVFusion   — LiDAR + Camera fusion, ICRA 2023 (MIT HAN Lab)
  - Sparse4D v3 — temporal 3D detection + tracking (2024 SOTA)

These use MMDetection3D as the training framework.

Setup:
    pip install mmcv mmdet mmdet3d
    Or use the official Docker image: openmmlab/mmdetection3d:latest

Reference repos:
  BEVFormer: github.com/fundamentalvision/BEVFormer
  BEVFusion:  github.com/mit-han-lab/bevfusion
  Sparse4D:   github.com/linxuewu/Sparse4D
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Detection3D:
    """3D bounding box in camera/lidar coordinate frame."""
    center: np.ndarray       # [x, y, z]
    size: np.ndarray         # [w, l, h]
    yaw: float               # rotation around vertical axis (radians)
    score: float
    class_id: int
    class_name: str
    velocity: Optional[np.ndarray] = None   # [vx, vy] for nuScenes


# ─── MMDetection3D Base Class ─────────────────────────────────────────────────

class MMDet3DModel:
    """
    Base class for MMDetection3D models.
    Provides train/predict/export interface.
    """

    def __init__(self, config: str, checkpoint: str, device: str = 'cuda'):
        try:
            from mmdet3d.apis import init_model, inference_detector
            self._init_model = init_model
            self._inference = inference_detector
        except ImportError:
            raise ImportError(
                "pip install openmim && mim install mmdet3d\n"
                "Or: pip install mmdet3d"
            )

        self.model = self._init_model(config, checkpoint, device=device)
        self.device = device

    def predict(self, data_sample) -> List[Detection3D]:
        result, _ = self._inference(self.model, data_sample)
        return self._parse_result(result)

    def _parse_result(self, result) -> List[Detection3D]:
        detections = []
        if hasattr(result, 'pred_instances_3d'):
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.cpu().numpy()  # Nx9: x,y,z,w,l,h,yaw,vx,vy
            scores = pred.scores_3d.cpu().numpy()
            labels = pred.labels_3d.cpu().numpy()
            class_names = self.model.dataset_meta.get('classes', [])

            for box, score, label in zip(boxes, scores, labels):
                detections.append(Detection3D(
                    center=box[:3],
                    size=box[3:6],
                    yaw=float(box[6]),
                    score=float(score),
                    class_id=int(label),
                    class_name=class_names[label] if label < len(class_names) else str(label),
                    velocity=box[7:9] if box.shape[0] > 7 else None,
                ))
        return detections


# ─── BEVFormer ────────────────────────────────────────────────────────────────

class BEVFormerDetector(MMDet3DModel):
    """
    BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images
    via Spatiotemporal Transformers. (ECCV 2022)

    Camera-only 3D detection using BEV (Bird's Eye View) space.
    Temporal attention aggregates information across past frames.

    Variants:
      bevformer_base:  ~200M params, 25.2 FPS, 51.7 NDS on nuScenes
      bevformer_small: ~130M params, faster, 47.9 NDS
      bevformer_tiny:  ~28M params, edge deployment

    Weights: github.com/fundamentalvision/BEVFormer (Google Drive links in README)

    Key concepts learned:
      - Deformable attention for multi-scale features
      - Spatial cross-attention: 3D ref points projected to each camera
      - Temporal self-attention: current BEV fused with previous BEV
    """

    PRETRAINED = {
        'bevformer_base':  'bevformer_r101_dcn_24ep.pth',
        'bevformer_small': 'bevformer_small_epoch_24.pth',
        'bevformer_tiny':  'bevformer_tiny_epoch_24.pth',
    }

    def __init__(
        self,
        variant: str = 'bevformer_base',
        config: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = 'cuda',
    ):
        # Use pre-built configs from BEVFormer repo or our local configs
        config = config or f'configs/bevformer/{variant}.py'
        checkpoint = checkpoint or self.PRETRAINED.get(variant, f'{variant}.pth')
        super().__init__(config, checkpoint, device)
        self.variant = variant

    @staticmethod
    def get_train_command(
        config: str = 'configs/bevformer/bevformer_base.py',
        gpus: int = 8,
        launcher: str = 'pytorch',
        work_dir: str = 'work_dirs/bevformer_base',
    ) -> str:
        return (
            f"python -m torch.distributed.launch "
            f"--nproc_per_node={gpus} "
            f"$(which python) tools/train.py "
            f"{config} "
            f"--launcher {launcher} "
            f"--work-dir {work_dir}"
        )


# ─── BEVFusion ───────────────────────────────────────────────────────────────

class BEVFusionDetector:
    """
    BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View
    Representation. (MIT HAN Lab, ICRA 2023)

    Fuses LiDAR point clouds + multi-camera images in unified BEV space.
    Supports: 3D detection + BEV segmentation simultaneously.

    Performance on nuScenes:
      mAP: 70.2, NDS: 72.9 (vs BEVFormer: 56.9 NDS camera-only)

    Repo: github.com/mit-han-lab/bevfusion
    Paper: arxiv.org/abs/2205.13542

    Key concepts learned:
      - Voxel pooling for LiDAR → BEV
      - LSS (Lift-Splat-Shoot) for camera → BEV
      - Convolutional fusion of both BEV maps
      - Multi-task head: detection + segmentation

    Requirements: CUDA 11.3+, PyTorch 1.9+
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = 'cuda',
    ):
        try:
            import mmcv
            from mmdet3d.apis import init_model, inference_detector
            self.model = init_model(config, checkpoint, device=device)
            self._inference = inference_detector
        except ImportError:
            raise ImportError("pip install mmdet3d")

    def predict(self, lidar_path: str, camera_images: Dict[str, str]) -> List[Detection3D]:
        """
        Args:
            lidar_path: path to .pcd / .bin LiDAR file
            camera_images: dict of {'CAM_FRONT': path, 'CAM_BACK': path, ...}
        """
        data = self._build_input(lidar_path, camera_images)
        result, _ = self._inference(self.model, data)
        return self._parse_result(result)

    def _build_input(self, lidar_path: str, camera_images: Dict[str, str]):
        # This mirrors the MMDetection3D input format for BEVFusion
        return {'pts_filename': lidar_path, 'img_filename': list(camera_images.values())}

    def _parse_result(self, result) -> List[Detection3D]:
        detections = []
        if hasattr(result, 'pred_instances_3d'):
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.cpu().numpy()
            scores = pred.scores_3d.cpu().numpy()
            labels = pred.labels_3d.cpu().numpy()
            names = self.model.dataset_meta.get('classes', [])
            for box, score, label in zip(boxes, scores, labels):
                detections.append(Detection3D(
                    center=box[:3], size=box[3:6], yaw=float(box[6]),
                    score=float(score), class_id=int(label),
                    class_name=names[label] if label < len(names) else str(label),
                ))
        return detections

    @staticmethod
    def get_train_command(
        config: str = 'configs/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
        gpus: int = 8,
        work_dir: str = 'work_dirs/bevfusion',
    ) -> str:
        return (
            f"torchrun --nproc_per_node={gpus} "
            f"tools/train.py {config} "
            f"--work-dir {work_dir} "
            f"--launcher pytorch"
        )


# ─── Sparse4D v3 (Temporal 3D Detection + Tracking) ──────────────────────────

class Sparse4DDetector:
    """
    Sparse4D v3: Advancing End-to-End 3D Detection and Tracking (2024)

    Temporal, sparse 4D detection using deformable attention on 3D anchor points.
    Jointly outputs: detection + tracking in one forward pass.

    Performance on nuScenes val:
      mAP: 63.0, NDS: 67.7, AMOTA: 63.6 (SOTA as of 2024)

    Repo: github.com/linxuewu/Sparse4D
    Paper: arxiv.org/abs/2311.11722

    Key concepts learned:
      - 4D anchors: (x, y, z, t) — spatial + temporal
      - Instance denoising during training (like DN-DETR)
      - Quality estimation head for better NMS replacement
      - Decoupled attention: spatial and temporal separately
    """

    def __init__(self, config: str, checkpoint: str, device: str = 'cuda'):
        try:
            from mmdet3d.apis import init_model
            self.model = init_model(config, checkpoint, device=device)
        except ImportError:
            raise ImportError("pip install mmdet3d")

    @staticmethod
    def get_train_command(config: str, gpus: int = 8, work_dir: str = 'work_dirs/sparse4d') -> str:
        return (
            f"torchrun --nproc_per_node={gpus} "
            f"tools/train.py {config} "
            f"--work-dir {work_dir} --launcher pytorch"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_detector_3d(cfg: dict):
    model_type = cfg.get('type', 'bevformer').lower()

    if model_type == 'bevformer':
        return BEVFormerDetector(
            variant=cfg.get('variant', 'bevformer_base'),
            config=cfg.get('config'),
            checkpoint=cfg.get('checkpoint'),
            device=cfg.get('device', 'cuda'),
        )
    elif model_type == 'bevfusion':
        return BEVFusionDetector(
            config=cfg['config'],
            checkpoint=cfg['checkpoint'],
            device=cfg.get('device', 'cuda'),
        )
    elif model_type == 'sparse4d':
        return Sparse4DDetector(
            config=cfg['config'],
            checkpoint=cfg['checkpoint'],
            device=cfg.get('device', 'cuda'),
        )
    else:
        raise ValueError(f"Unknown 3D detector type: {model_type}")
