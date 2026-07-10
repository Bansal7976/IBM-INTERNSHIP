"""ByteTrack-style multi-object tracker for the ADAS pipeline.

Self-contained (no external ByteTrack pip package needed): two-stage
association exactly like ByteTrack — match high-confidence detections first,
then rescue low-confidence ones against unmatched tracks — using IoU +
greedy assignment. Tracks carry velocity for the collision/overtaking modules.

Usage:
    tracker = ByteTrackWrapper()
    tracks = tracker.update(detections)   # detections: objects with .bbox/.conf/.cls
    for t in tracks:
        t.track_id, t.bbox, t.velocity, t.class_name
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: tuple                     # (x1, y1, x2, y2)
    conf: float
    cls: int
    class_name: str
    velocity: tuple = (0.0, 0.0)    # pixels/frame (vx, vy) of the centroid
    age: int = 0                    # frames since last matched detection
    hits: int = 1                   # total matched detections
    depth_speed_mps: float = 0.0    # filled by collision module if depth available
    _prev_center: tuple = field(default=None, repr=False)

    @property
    def tlbr(self):
        return self.bbox

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, None, :]  # (N,1,4)
    b = boxes_b[None, :, :]  # (1,M,4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def _greedy_match(iou: np.ndarray, thresh: float):
    """Greedy assignment on the IoU matrix. Returns (matches, un_a, un_b)."""
    matches, used_a, used_b = [], set(), set()
    if iou.size:
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for i, j in order:
            if iou[i, j] < thresh:
                break
            if i in used_a or j in used_b:
                continue
            matches.append((int(i), int(j)))
            used_a.add(int(i))
            used_b.add(int(j))
    un_a = [i for i in range(iou.shape[0]) if i not in used_a]
    un_b = [j for j in range(iou.shape[1]) if j not in used_b]
    return matches, un_a, un_b


class ByteTrackWrapper:
    """Two-stage (high/low confidence) IoU tracker with velocity estimation."""

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 match_thresh: float = 0.3, max_age: int = 30, min_hits: int = 2,
                 velocity_smoothing: float = 0.7):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.alpha = velocity_smoothing
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list) -> list[Track]:
        """detections: objects with .bbox (xyxy), .conf, .cls, optional .class_name"""
        dets = [d for d in detections if getattr(d, "conf", 1.0) >= self.low_thresh]
        high = [d for d in dets if d.conf >= self.high_thresh]
        low = [d for d in dets if d.conf < self.high_thresh]

        # Predict track positions forward by their velocity before matching
        predicted = np.array([
            [t.bbox[0] + t.velocity[0], t.bbox[1] + t.velocity[1],
             t.bbox[2] + t.velocity[0], t.bbox[3] + t.velocity[1]]
            for t in self.tracks]) if self.tracks else np.zeros((0, 4))

        # STAGE 1: high-confidence detections vs all tracks
        high_boxes = np.array([d.bbox for d in high]) if high else np.zeros((0, 4))
        m1, un_tracks, un_high = _greedy_match(
            _iou_matrix(predicted, high_boxes), self.match_thresh)
        for ti, di in m1:
            self._apply(self.tracks[ti], high[di])

        # STAGE 2 (the ByteTrack idea): rescue low-confidence detections
        # against still-unmatched tracks — keeps occluded objects alive.
        rem_tracks = [self.tracks[i] for i in un_tracks]
        rem_pred = predicted[un_tracks] if un_tracks else np.zeros((0, 4))
        low_boxes = np.array([d.bbox for d in low]) if low else np.zeros((0, 4))
        m2, un_tracks2, _ = _greedy_match(
            _iou_matrix(rem_pred, low_boxes), self.match_thresh)
        for ti, di in m2:
            self._apply(rem_tracks[ti], low[di])

        # Unmatched tracks age; expired ones are dropped
        for idx in un_tracks2:
            rem_tracks[idx].age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # New tracks from unmatched HIGH-confidence detections only
        for di in un_high:
            d = high[di]
            self.tracks.append(Track(
                track_id=self._next_id, bbox=tuple(d.bbox), conf=d.conf,
                cls=int(d.cls),
                class_name=getattr(d, "class_name", str(int(d.cls)))))
            self._next_id += 1

        # Return confirmed, currently-visible tracks
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age == 0]

    def _apply(self, track: Track, det):
        new_center = ((det.bbox[0] + det.bbox[2]) / 2.0,
                      (det.bbox[1] + det.bbox[3]) / 2.0)
        if track._prev_center is not None:
            raw_v = (new_center[0] - track._prev_center[0],
                     new_center[1] - track._prev_center[1])
            # Exponential smoothing — kills jitter that ruins TTC estimates
            track.velocity = (
                self.alpha * track.velocity[0] + (1 - self.alpha) * raw_v[0],
                self.alpha * track.velocity[1] + (1 - self.alpha) * raw_v[1])
        track._prev_center = new_center
        track.bbox = tuple(det.bbox)
        track.conf = det.conf
        track.age = 0
        track.hits += 1


if __name__ == "__main__":
    # Smoke test with synthetic moving boxes
    from dataclasses import dataclass as dc

    @dc
    class D:
        bbox: tuple
        conf: float
        cls: int
        class_name: str = "car"

    tr = ByteTrackWrapper()
    for f in range(10):
        dets = [D((100 + f * 5, 200, 160 + f * 5, 260), 0.9, 0),
                D((400, 300 + f * 3, 470, 380 + f * 3), 0.85, 0)]
        tracks = tr.update(dets)
        print(f"frame {f}: " + "  ".join(
            f"#{t.track_id} v=({t.velocity[0]:.1f},{t.velocity[1]:.1f})"
            for t in tracks))
    assert len(tr.update([D((150, 200, 210, 260), 0.9, 0)])) >= 1
    print("OK - tracker works")
