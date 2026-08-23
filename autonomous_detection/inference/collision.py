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

    # Scale the TTC thresholds by the object's decision group, so a pedestrian
    # gets a longer margin than a car. Without this the taxonomy is only a
    # labelling scheme; with it, the grouping actually changes what the system
    # does -- which is the claim the whole design rests on.
    use_group_margins: bool = True

    dist_history: dict = field(default_factory=dict)
    last_depth: Optional[np.ndarray] = None

    # Telemetry. TTC needs `history_len >= 5` samples of a track's distance
    # before it can report a closing speed, and that history is keyed on
    # track_id -- so every tracker ID switch throws it away and blinds this
    # module on that object for the next five frames.
    #
    # In dense unstructured traffic, where two-wheelers weave and occlude each
    # other constantly, that can happen often enough to matter, and it would be
    # invisible: the module simply raises no alert. These counters make it
    # measurable, so a decision to change tracker parameters rests on a number
    # rather than a hunch. Read them with stats().
    stats_frames: int = 0
    stats_track_observations: int = 0     # (track, frame) pairs seen
    stats_with_closing_speed: int = 0     # ... of which had enough history
    stats_new_tracks: int = 0             # first sighting of an id
    stats_lost_tracks: int = 0            # ids that disappeared
    stats_no_depth: int = 0               # distance unresolvable
    stats_rescued: int = 0                # histories recovered across an ID switch

    # Recovering a distance history across a tracker ID switch.
    #
    # Measured on a synthetic approach: with stable ids, closing speed is
    # available for 93% of sightings. At an ID switch every 5 frames that falls
    # to 20%, and every 3 frames to ZERO -- the collision layer goes completely
    # silent, and silently, because "no alert" is indistinguishable from "road
    # clear". Dense unstructured traffic, where two-wheelers weave and occlude
    # each other constantly, is exactly where that churn happens.
    #
    # So when an unseen track_id appears where a track vanished moments ago and
    # the boxes overlap, its distance history is inherited rather than
    # restarted. This is deliberately geometric and not appearance-based: it
    # costs nothing, needs no ReID model, and the failure mode of a wrong match
    # is a slightly stale distance sample rather than a fabricated object.
    rescue_across_id_switch: bool = True
    rescue_iou_threshold: float = 0.5
    rescue_max_age_frames: int = 5
    _lost: dict = field(default_factory=dict, repr=False)   # tid -> (bbox, frame, hist)
    _last_bbox: dict = field(default_factory=dict, repr=False)

    def set_night_mode(self, is_dark_unlit: bool):
        """Longer safety margins when driving in unlit darkness."""
        self.critical_ttc = 2.5 if is_dark_unlit else 1.5
        self.warning_ttc = 4.5 if is_dark_unlit else 3.0

    def _group_of(self, track) -> Optional[str]:
        """Decision group for a track, or None if the class is unrecognised.

        Accepts a group name the detector already emits (a model trained on the
        decision taxonomy outputs these directly) and otherwise maps the
        fine-grained class name. Returns None rather than guessing, so an
        unknown class falls back to the ungrouped thresholds instead of
        silently receiving a pedestrian's margins or a cone's.
        """
        name = getattr(track, "class_name", None)
        if not name:
            return None
        from models.taxonomy import GROUP_BEHAVIOUR, NAME_TO_GROUP, normalise
        key = normalise(name)
        if name in GROUP_BEHAVIOUR:
            return name
        return NAME_TO_GROUP.get(key)

    def _thresholds_for(self, track) -> tuple:
        """(critical_ttc, warning_ttc) for this object, group-scaled."""
        if not self.use_group_margins:
            return self.critical_ttc, self.warning_ttc
        group = self._group_of(track)
        if group is None:
            return self.critical_ttc, self.warning_ttc
        from models.taxonomy import GROUP_BEHAVIOUR
        scale = GROUP_BEHAVIOUR[group]["ttc_margin_scale"]
        return self.critical_ttc * scale, self.warning_ttc * scale

    def update(self, frame: np.ndarray, tracks: list, lanes=None) -> list[Alert]:
        self.last_depth = self.depth_model.infer(frame)
        alerts = []
        self.stats_frames += 1
        active_ids = {t.track_id for t in tracks}

        # Retire vanished tracks BEFORE processing this frame's tracks, not
        # after. At an ID switch the old id disappears and the new one appears
        # in the SAME frame, so retiring afterwards leaves nothing for the new
        # track to inherit and the rescue only ever catches a history one
        # generation stale. Ordering it this way is what makes the recovery
        # exact rather than accidental.
        self._retire_missing(active_ids)

        for t in tracks:
            self.stats_track_observations += 1
            if t.track_id not in self.dist_history:
                self.stats_new_tracks += 1
                self._try_rescue_history(t)
            self._last_bbox[t.track_id] = tuple(t.bbox)

            dist = self._object_distance(t.bbox)
            if dist is None:
                self.stats_no_depth += 1
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
                self.stats_with_closing_speed += 1

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
                critical, warning = self._thresholds_for(t)
                if ttc < critical:
                    alerts.append(Alert("BRAKE", t.track_id, cls_name, dist, ttc, tuple(t.bbox)))
                elif ttc < warning:
                    alerts.append(Alert("WARNING", t.track_id, cls_name, dist, ttc, tuple(t.bbox)))

        return alerts

    def _retire_missing(self, active_ids: set) -> None:
        """Park the histories of tracks that are no longer present.

        Parked rather than discarded, so a track that reappears under a new id
        can inherit its distance history instead of starting from nothing.
        """
        for tid in list(self.dist_history):
            if tid in active_ids:
                continue
            if self.rescue_across_id_switch and tid in self._last_bbox:
                self._lost[tid] = (self._last_bbox[tid], self.stats_frames,
                                   self.dist_history[tid])
            del self.dist_history[tid]
            self._last_bbox.pop(tid, None)
            self.stats_lost_tracks += 1

        # Expire parked histories nothing claimed. A stale distance is worse
        # than none: it would report a closing speed measured against where the
        # object was several frames ago.
        for tid, (_, seen_at, _) in list(self._lost.items()):
            if self.stats_frames - seen_at > self.rescue_max_age_frames:
                del self._lost[tid]

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _try_rescue_history(self, track) -> bool:
        """Inherit a just-lost track's distance history if the boxes overlap.

        Without this, a tracker ID switch resets the history and the module
        cannot report a closing speed for the next several frames -- during
        which it raises no alert at all, whatever is actually approaching.
        """
        if not self.rescue_across_id_switch or not self._lost:
            return False
        bbox = tuple(track.bbox)
        best_tid, best_iou = None, 0.0
        for tid, (last_bbox, seen_at, _hist) in self._lost.items():
            if self.stats_frames - seen_at > self.rescue_max_age_frames:
                continue
            iou = self._iou(bbox, last_bbox)
            if iou > best_iou:
                best_tid, best_iou = tid, iou
        if best_tid is None or best_iou < self.rescue_iou_threshold:
            return False
        self.dist_history[track.track_id] = self._lost.pop(best_tid)[2]
        self.stats_rescued += 1
        return True

    def stats(self) -> dict:
        """How often TTC was actually computable, and why it was not.

        `closing_speed_coverage` is the number that matters. It is the fraction
        of object sightings for which a closing speed -- and therefore a TTC --
        was available. Anything well below 1.0 means the collision layer is
        frequently silent not because the road is clear but because it could
        not measure, and the usual cause is tracker ID churn discarding the
        distance history.

        `tracks_per_100_frames` is the companion figure: a high rate of new ids
        relative to the objects actually present is what ID churn looks like.
        """
        obs = max(self.stats_track_observations, 1)
        return {
            "frames": self.stats_frames,
            "track_observations": self.stats_track_observations,
            "closing_speed_coverage": round(self.stats_with_closing_speed / obs, 4),
            "new_tracks": self.stats_new_tracks,
            "lost_tracks": self.stats_lost_tracks,
            "tracks_per_100_frames": round(
                self.stats_new_tracks / max(self.stats_frames, 1) * 100, 2),
            "histories_rescued": self.stats_rescued,
            "depth_unresolvable": self.stats_no_depth,
            "history_len_required": self.history_len,
        }

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
