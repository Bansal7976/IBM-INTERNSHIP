"""
2D Object Detection model wrappers.

Supported models:
  - YOLOv11  (Ultralytics) — fastest, easiest to use, recommended start
  - YOLOv10  (THU-MIG)     — NMS-free, lower latency
  - RT-DETR  (Ultralytics) — transformer-based real-time detector
  - Grounding DINO          — open-vocabulary, language-guided detection

All expose a common interface:
    detector = build_detector_2d(cfg)
    results   = detector.predict(image_or_path)   -> List[Detection]
    detector.train(data_yaml, ...)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import numpy as np


@dataclass
class Detection:
    box: np.ndarray      # [x1, y1, x2, y2] in pixel coords
    score: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


# ─── YOLOv11 / YOLOv10 / RT-DETR (Ultralytics) ───────────────────────────────

class UltralyticsDetector:
    """
    Unified wrapper for any Ultralytics model: YOLO11, YOLOv10, RT-DETR.

    Model weights auto-download on first use from Ultralytics HUB.

    Recommended starting points:
      yolo11n.pt  — fastest (nano), edge devices
      yolo11m.pt  — balanced, good for training
      yolo11x.pt  — highest accuracy
      rtdetr-l.pt — transformer-based, excellent for crowded scenes
    """

    MODEL_URLS = {
        'yolo11n': 'yolo11n.pt',
        'yolo11s': 'yolo11s.pt',
        'yolo11m': 'yolo11m.pt',
        'yolo11l': 'yolo11l.pt',
        'yolo11x': 'yolo11x.pt',
        'yolov10n': 'yolov10n.pt',
        'yolov10m': 'yolov10m.pt',
        'yolov10x': 'yolov10x.pt',
        'rtdetr-l': 'rtdetr-l.pt',
        'rtdetr-x': 'rtdetr-x.pt',
    }

    def __init__(
        self,
        model_name: str = 'yolo11m',
        weights: Optional[str] = None,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        weight_path = weights or self.MODEL_URLS.get(model_name, f'{model_name}.pt')
        self.model = YOLO(weight_path)
        self.model.to(device)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.classes = classes
        self.device = device

    def predict(
        self,
        source: Union[str, np.ndarray, list],
        verbose: bool = False,
    ) -> List[List[Detection]]:
        """
        Run inference.
        Returns a list (one per image) of Detection lists.
        """
        results = self.model.predict(
            source,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=verbose,
            device=self.device,
        )

        all_detections = []
        for r in results:
            detections = []
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names

                for box, score, cid in zip(boxes, scores, class_ids):
                    detections.append(Detection(
                        box=box,
                        score=float(score),
                        class_id=int(cid),
                        class_name=names[cid],
                    ))
            all_detections.append(detections)

        return all_detections

    def train(
        self,
        data: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        device: str = 'cuda',
        workers: int = 8,
        project: str = 'runs/train',
        name: str = 'exp',
        pretrained: bool = True,
        resume: bool = False,
        amp: bool = True,
        multi_scale: bool = False,
        freeze: Optional[int] = None,
        lr0: float = 0.01,
        lrf: float = 0.01,
        warmup_epochs: float = 3.0,
        **kwargs,
    ):
        """
        Fine-tune on custom dataset.

        Args:
            data: path to dataset YAML (kitti.yaml, nuscenes.yaml, etc.)
            freeze: freeze first N layers (useful for fine-tuning)
        """
        return self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=project,
            name=name,
            pretrained=pretrained,
            resume=resume,
            amp=amp,
            multi_scale=multi_scale,
            freeze=freeze,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            **kwargs,
        )

    def export(
        self,
        format: str = 'onnx',
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
    ) -> str:
        """Export to ONNX / TensorRT / CoreML etc."""
        return self.model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
        )

    def validate(self, data: str, **kwargs):
        return self.model.val(data=data, **kwargs)


# ─── Grounding DINO (Open-Vocabulary) ─────────────────────────────────────────

class GroundingDINODetector:
    """
    Grounding DINO 1.5 — open-vocabulary detector.
    Detects objects described by natural language prompts.

    Requires: pip install groundingdino-py
    Weights: auto-downloaded from HuggingFace

    Use case: detect vehicles without retraining by just changing the prompt.
    Example prompt: "car . truck . bus . pedestrian . cyclist ."
    """

    CONFIGS = {
        'swinb': 'GroundingDINO_SwinB_cfg.py',
        'swint': 'GroundingDINO_SwinT_OGC.py',
    }

    WEIGHTS = {
        'swinb': 'groundingdino_swinb_cogcoor.pth',
        'swint': 'groundingdino_swint_ogc.pth',
    }

    def __init__(
        self,
        backbone: str = 'swinb',
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T
        except ImportError:
            raise ImportError(
                "pip install groundingdino-py\n"
                "Or clone: github.com/IDEA-Research/GroundingDINO"
            )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        cfg = config_path or self.CONFIGS[backbone]
        wts = weights_path or self.WEIGHTS[backbone]
        self.model = load_model(cfg, wts, device=device)

        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        image: np.ndarray,
        prompt: str = "car . truck . bus . motorcycle . pedestrian . bicycle .",
    ) -> List[Detection]:
        """
        Detect objects matching the text prompt.
        Prompt format: "class1 . class2 . class3 ."
        """
        from groundingdino.util.inference import predict
        from PIL import Image
        import torch

        pil_image = Image.fromarray(image)
        img_tensor, _ = self._transform(pil_image, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=img_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        h, w = image.shape[:2]
        detections = []
        for box, score, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            detections.append(Detection(
                box=np.array([x1, y1, x2, y2]),
                score=float(score),
                class_id=-1,
                class_name=phrase,
            ))

        return detections


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_detector_2d(cfg: dict) -> Union[UltralyticsDetector, GroundingDINODetector]:
    """
    Build a 2D detector from a config dict.

    Example cfg:
        {'type': 'yolo11m', 'device': 'cuda', 'conf_threshold': 0.25}
        {'type': 'grounding_dino', 'backbone': 'swinb', 'box_threshold': 0.35}
    """
    model_type = cfg.get('type', 'yolo11m')

    if model_type == 'grounding_dino':
        return GroundingDINODetector(
            backbone=cfg.get('backbone', 'swinb'),
            weights_path=cfg.get('weights'),
            device=cfg.get('device', 'cuda'),
            box_threshold=cfg.get('box_threshold', 0.35),
            text_threshold=cfg.get('text_threshold', 0.25),
        )
    else:
        return UltralyticsDetector(
            model_name=model_type,
            weights=cfg.get('weights'),
            device=cfg.get('device', 'cuda'),
            conf_threshold=cfg.get('conf_threshold', 0.25),
            iou_threshold=cfg.get('iou_threshold', 0.45),
            classes=cfg.get('classes'),
        )
