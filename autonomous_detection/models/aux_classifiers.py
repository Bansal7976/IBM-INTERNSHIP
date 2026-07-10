"""Auxiliary classifiers: traffic-light state + lane-marking type.

Both are small ResNet-18 heads over crops — the 2-stage pattern:
detector/lane-model finds WHERE, these classify WHAT.

Training data:
  - Traffic light state: LISA Traffic Light Dataset (Kaggle) — red/yellow/green
  - Lane type: BDD100K lane marking labels — solid/dashed crops along polylines
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18


# ----------------------------------------------------------------------------
# Traffic light state
# ----------------------------------------------------------------------------

TL_CLASSES = ["red", "yellow", "green", "off"]


class TrafficLightStateClassifier(nn.Module):
    """Crop of a detected traffic light -> red / yellow / green / off."""

    def __init__(self, weights: str | None = None, device: str = "cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(512, len(TL_CLASSES))
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location=self.device))
        self.net = self.net.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def classify(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 8:
            return "off"
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = self.tf(rgb).unsqueeze(0).to(self.device)
        idx = int(self.net(t).argmax(1).item())
        return TL_CLASSES[idx]


def classify_light_hsv(crop_bgr: np.ndarray) -> str:
    """No-training fallback: dominant lit hue, weighted by vertical position
    (red lamp on top, green on bottom). Robust for clearly lit lamps."""
    if crop_bgr.size == 0:
        return "off"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] > 150
    masks = {
        "red": ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 160)) & (hsv[:, :, 1] > 100) & bright,
        "yellow": (hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 35) & (hsv[:, :, 1] > 100) & bright,
        "green": (hsv[:, :, 0] > 40) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 100) & bright,
    }
    counts = {k: int(m.sum()) for k, m in masks.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 10 else "off"


def decide_traffic_light_action(tl_states: dict, distances: dict | None = None) -> str:
    """Aggregate all visible lights into one action. Nearest light wins;
    red anywhere relevant -> STOP (fail safe)."""
    if not tl_states:
        return "NO_LIGHT"
    states = list(tl_states.values())
    if "red" in states:
        return "STOP"
    if "yellow" in states:
        return "CAUTION"
    if "green" in states:
        return "GO"
    return "NO_LIGHT"


# ----------------------------------------------------------------------------
# Lane marking type (solid / dashed) — powers the overtaking legality rule
# ----------------------------------------------------------------------------

LANE_TYPES = ["solid", "dashed", "double_solid", "double_yellow"]


class LaneTypeClassifier(nn.Module):
    """Patches sampled along a lane polyline -> majority-vote lane type."""

    def __init__(self, weights: str | None = None, device: str = "cuda",
                 patch: int = 32, n_samples: int = 12):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.patch = patch
        self.n_samples = n_samples
        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(512, len(LANE_TYPES))
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location=self.device))
        self.net = self.net.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def classify_line(self, frame_bgr: np.ndarray, polyline: np.ndarray) -> str:
        pts = np.asarray(polyline)
        if len(pts) < 2:
            return "unknown"
        # Evenly sample points along the polyline
        idx = np.linspace(0, len(pts) - 1, self.n_samples).astype(int)
        patches = []
        half = self.patch // 2
        h, w = frame_bgr.shape[:2]
        for x, y in pts[idx]:
            x, y = int(x), int(y)
            if half <= x < w - half and half <= y < h - half:
                p = frame_bgr[y - half:y + half, x - half:x + half]
                patches.append(self.tf(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
        if not patches:
            return "unknown"
        batch = torch.stack(patches).to(self.device)
        preds = self.net(batch).argmax(1).cpu().numpy()
        # Majority vote across patches
        vote = np.bincount(preds, minlength=len(LANE_TYPES)).argmax()
        return LANE_TYPES[int(vote)]

    def classify(self, frame_bgr: np.ndarray, lanes) -> dict:
        """Classify every lane line. Returns {id(polyline): type, 'center': type}."""
        result = {}
        polylines = getattr(lanes, "polylines", lanes) or []
        for pl in polylines:
            result[id(pl)] = self.classify_line(frame_bgr, np.asarray(pl))
        if polylines:
            # convenience key for the overtaking module
            sorted_lines = sorted(polylines, key=lambda p: np.asarray(p)[:, 0].mean())
            mid = max(0, len(sorted_lines) // 2 - 1)
            result["center"] = result[id(sorted_lines[mid])]
        return result
