"""
Multi-object tracking integration.

Supported trackers:
  - ByteTrack  — Fast, high-performance, built into Ultralytics
  - BotSort    — ByteTrack + ReID + camera-motion compensation
  - UCMCTrack  — AAAI 2024, uniform camera motion compensation

ByteTrack is the recommended default (built into Ultralytics, zero setup).

Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
       arxiv.org/abs/2110.06864
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .detector_2d import Detection


@dataclass
class TrackedObject:
    track_id: int
    box: np.ndarray         # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: str
    age: int                # frames since first detected
    hits: int               # total frames tracked
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


# ─── ByteTrack via Ultralytics ────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    ByteTrack associates every detection box (high + low confidence)
    using IoU matching. Simple, fast, and very effective.

    Usage via Ultralytics:
        results = model.track(source, tracker='bytetrack.yaml', persist=True)

    This class wraps the track results into TrackedObject list.

    Key hyperparameters:
      track_thresh: 0.25   — detection confidence to enter track
      track_buffer: 30     — frames to keep lost tracks alive
      match_thresh: 0.8    — IoU threshold for association
    """

    def __init__(
        self,
        model,
        tracker_type: str = 'bytetrack',  # 'bytetrack' or 'botsort'
        device: str = 'cuda',
        persist: bool = True,
    ):
        """
        Args:
            model: UltralyticsDetector.model (YOLO object)
            tracker_type: 'bytetrack' or 'botsort'
        """
        self.model = model
        self.tracker_type = tracker_type
        self.device = device
        self.persist = persist

    def track(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[List[TrackedObject]]:
        """
        Run detection + tracking on source (video file, stream, or image list).
        Returns list of frames, each frame a list of TrackedObjects.
        """
        results = self.model.track(
            source,
            conf=conf,
            iou=iou,
            classes=classes,
            tracker=f'{self.tracker_type}.yaml',
            persist=self.persist,
            device=self.device,
            verbose=verbose,
        )

        all_frames = []
        for r in results:
            frame_tracks = []
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for box, score, cid, tid in zip(boxes, scores, class_ids, track_ids):
                    frame_tracks.append(TrackedObject(
                        track_id=int(tid),
                        box=box,
                        score=float(score),
                        class_id=int(cid),
                        class_name=r.names[cid],
                        age=0,
                        hits=1,
                    ))
            all_frames.append(frame_tracks)

        return all_frames


# ─── Simple IoU-based tracker (no dependencies) ───────────────────────────────

class SimpleIOUTracker:
    """
    Minimal IoU tracker for debugging / baseline comparison.
    No deep features, just spatial overlap.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        if not detections:
            # Age out lost tracks
            lost = [tid for tid, t in self._tracks.items() if t['lost'] >= self.max_lost]
            for tid in lost:
                del self._tracks[tid]
            return []

        det_boxes = np.array([d.box for d in detections])
        track_ids = list(self._tracks.keys())

        if not track_ids:
            tracked = []
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {'box': det.box, 'lost': 0, 'hits': 1, 'age': 1}
                tracked.append(TrackedObject(
                    track_id=tid, box=det.box, score=det.score,
                    class_id=det.class_id, class_name=det.class_name,
                    age=1, hits=1,
                ))
            return tracked

        track_boxes = np.array([self._tracks[tid]['box'] for tid in track_ids])
        iou_matrix = self._batch_iou(det_boxes, track_boxes)

        # Greedy matching
        matched_dets = set()
        matched_trks = set()
        for _ in range(min(len(detections), len(track_ids))):
            if iou_matrix.max() < self.iou_threshold:
                break
            di, ti = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_dets.add(di)
            matched_trks.add(ti)
            iou_matrix[di, :] = -1
            iou_matrix[:, ti] = -1

        tracked = []
        for di, det in enumerate(detections):
            if di in matched_dets:
                ti = list(matched_trks)[list(matched_dets).index(di)]
                tid = track_ids[ti]
                self._tracks[tid]['box'] = det.box
                self._tracks[tid]['lost'] = 0
                self._tracks[tid]['hits'] += 1
                self._tracks[tid]['age'] += 1
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {'box': det.box, 'lost': 0, 'hits': 1, 'age': 1}

            tracked.append(TrackedObject(
                track_id=tid, box=det.box, score=det.score,
                class_id=det.class_id, class_name=det.class_name,
                age=self._tracks[tid]['age'],
                hits=self._tracks[tid]['hits'],
            ))

        # Mark unmatched tracks as lost
        for i, tid in enumerate(track_ids):
            if i not in matched_trks:
                self._tracks[tid]['lost'] += 1

        # Remove dead tracks
        dead = [tid for tid, t in self._tracks.items() if t['lost'] > self.max_lost]
        for tid in dead:
            del self._tracks[tid]

        return tracked

    @staticmethod
    def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes (xyxy)."""
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = area_a[:, None] + area_b[None, :] - inter_area
        return inter_area / (union_area + 1e-6)
