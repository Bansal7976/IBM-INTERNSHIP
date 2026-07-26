"""Collision detection via Time-To-Collision (TTC).

TTC = distance / closing_speed, computed per tracked object in the ego path.
Distance comes from monocular metric depth (Depth Anything V2); closing speed
from the object's depth history (NOT pixel velocity — pixel motion is not
physical approach).

Usage:
    detector = CollisionDetector(depth_model, fps=30)
    alerts = detector.update(frame, tracks, lanes)
    for a in alerts:
        print(a.level, a.distance_m, a.ttc_s)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Alert:
    level: str            # "WARNING" or "BRAKE"
    track_id: int
    class_name: str
    distance_m: float
    ttc_s: float
    bbox: tuple           # (x1, y1, x2, y2)


@dataclass
class CollisionDetector:
    depth_model: object                  # must expose .infer(frame) -> HxW meters
    fps: int = 30
    critical_ttc: float = 1.5            # seconds -> BRAKE
    warning_ttc: float = 3.0             # seconds -> WARNING
    history_len: int = 10
    min_closing_speed: float = 0.5       # m/s — below this, not approaching

    dist_history: dict = field(default_factory=dict)
    last_depth: Optional[np.ndarray] = None

    def set_night_mode(self, is_dark_unlit: bool):
        """Longer safety margins when driving in unlit darkness."""
        self.critical_ttc = 2.5 if is_dark_unlit else 1.5
        self.warning_ttc = 4.5 if is_dark_unlit else 3.0

    def update(self, frame: np.ndarray, tracks: list, lanes=None) -> list[Alert]:
        self.last_depth = self.depth_model.infer(frame)
        alerts = []
        active_ids = set()

        for t in tracks:
            active_ids.add(t.track_id)

            dist = self._object_distance(t.bbox)
            if dist is None:
                continue

            # BUG FIX: distance history (and therefore depth_speed_mps) used to
            # only get built for objects inside our own ego-path, because the
            # ego-path check used to `continue` before any of this ran. That
            # meant oncoming-lane vehicles never got a depth_speed_mps, so
            # overtaking.py's oncoming-traffic rule silently treated every
            # oncoming car as stationary (closing speed = ego speed only),
            # which OVERESTIMATES the time until it arrives -> unsafe
            # "OVERTAKE POSSIBLE" calls. Track distance/speed for every object
            # first; only gate the BRAKE/WARNING *alerts* by ego-path below.
            hist = self.dist_history.setdefault(
                t.track_id, deque(maxlen=self.history_len)
            )
            hist.append(dist)

            closing = None
            if len(hist) >= 5:
                elapsed = (len(hist) - 1) / self.fps
                closing = (hist[0] - hist[-1]) / elapsed  # m/s, + = approaching
                # Exposed for overtaking.py's oncoming/lead-vehicle speed rules.
                t.depth_speed_mps = closing

            # BUG FIX: this used to be `if lanes is not None and not
            # self._in_ego_path(...)`, which short-circuited BEFORE ever
            # calling _in_ego_path() when lanes is None — meaning the
            # documented "fall back to the central 40% of the frame" behavior
            # in _in_ego_path() could never run, and EVERY tracked object
            # anywhere in the frame (oncoming lane, sidewalk, parked cars)
            # would raise BRAKE/WARNING alerts whenever lane detection was
            # unavailable for a frame. _in_ego_path() already handles
            # lanes=None internally, so just always call it.
            if not self._in_ego_path(t, lanes, frame.shape):
                continue

            if closing is not None and closing > self.min_closing_speed:
                ttc = dist / closing
                cls_name = getattr(t, "class_name", str(getattr(t, "cls", "?")))
                if ttc < self.critical_ttc:
                    alerts.append(Alert("BRAKE", t.track_id, cls_name, dist, ttc, tuple(t.bbox)))
                elif ttc < self.warning_ttc:
                    alerts.append(Alert("WARNING", t.track_id, cls_name, dist, ttc, tuple(t.bbox)))

        # Drop history of vanished tracks
        for tid in list(self.dist_history):
            if tid not in active_ids:
                del self.dist_history[tid]

        return alerts

    def _object_distance(self, bbox) -> Optional[float]:
        """Median depth over the LOWER HALF of the bbox.

        Lower half = vehicle body/road contact. Upper half often contains
        windows/sky reflections that corrupt depth. Median beats mean for
        robustness against outlier pixels.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = self.last_depth.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y_mid = max(0, min(h, (y1 + y2) // 2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y_mid:
            return None
        region = self.last_depth[y_mid:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.median(region))

    def _in_ego_path(self, track, lanes, frame_shape) -> bool:
        """Is the object inside our lane corridor?

        With lanes: check bbox bottom-center against ego-lane boundaries.
        Without usable lanes: fall back to the central 40% of the frame.
        """
        x1, _, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0

        ego = getattr(lanes, "ego_lane", None) if lanes is not None else None
        if ego and ego.get("left") is not None and ego.get("right") is not None:
            lx = _lane_x_at_y(ego["left"], y2)
            rx = _lane_x_at_y(ego["right"], y2)
            if lx is not None and rx is not None:
                return lx <= cx <= rx

        w = frame_shape[1]
        return 0.30 * w <= cx <= 0.70 * w


def _lane_x_at_y(polyline: np.ndarray, y: float) -> Optional[float]:
    """Interpolate lane x-coordinate at row y from an (N,2) polyline."""
    pts = np.asarray(polyline)
    if len(pts) < 2:
        return None
    order = np.argsort(pts[:, 1])
    ys, xs = pts[order, 1], pts[order, 0]
    if not (ys[0] <= y <= ys[-1]):
        return None
    return float(np.interp(y, ys, xs))


class DepthAnythingV2Metric:
    """Wrapper for Depth Anything V2 metric-depth (outdoor) checkpoint.

    Clone: git clone https://github.com/DepthAnything/Depth-Anything-V2 external/DepthAnythingV2
    Checkpoint: depth_anything_v2_metric_vkitti_vitl.pth (metric, outdoor)
    """

    def __init__(self, weights: str, encoder: str = "vits", max_depth: float = 80.0,
                 device: str = "cuda"):
        import sys
        from pathlib import Path
        import torch

        repo = Path(__file__).parent.parent / "external" / "DepthAnythingV2" / "metric_depth"
        sys.path.insert(0, str(repo))
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E501

        cfg = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }[encoder]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = DepthAnythingV2(**cfg, max_depth=max_depth)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = self.model.to(self.device).eval()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns HxW float32 depth map in METERS."""
        return self.model.infer_image(frame_bgr)  # repo API handles preprocessing
