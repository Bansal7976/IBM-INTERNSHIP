"""
Advanced Full ADAS Perception Pipeline
Combines: Detection + Lane Detection + Depth Estimation + Tracking + Collision Detection

Usage:
    from inference.advanced_pipeline import ADASPerceptionSystem

    adas = ADASPerceptionSystem(
        detector='yolo11m',
        detector_weights='best.pt',
        enable_lane=True,
        enable_depth=True,
        enable_collision=True,
    )

    results = adas.process_frame(image)
    # results.detections, results.lanes, results.depth, results.ttc_alerts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
import cv2
import time

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


@dataclass
class ADASResult:
    """Single frame output from ADAS system."""
    timestamp: float
    frame: np.ndarray                          # Original image
    detections: List = field(default_factory=list)    # [Detection objects]
    lanes: Optional[np.ndarray] = None        # Lane polygons (Nx4)
    depth_map: Optional[np.ndarray] = None    # HxW metric depth
    collision_alerts: List = field(default_factory=list)  # [{'track_id': ..., 'ttc': ..., 'risk': ...}]
    latency_ms: float = 0.0                   # Total pipeline latency


class ADASPerceptionSystem:
    """
    Complete ADAS perception pipeline combining all modules.

    Modules:
      1. YOLOv11 (2D object detection)
      2. ByteTrack (multi-object tracking + velocity estimation)
      3. Ultra-Fast Lane Detection v2 (lane detection)
      4. Depth Anything V2 (monocular depth)
      5. TTC Calculator (Time-to-Collision)
      6. Visualizer (overlay results)
    """

    def __init__(
        self,
        detector: str = 'yolo11m',
        detector_weights: Optional[str] = None,
        enable_lane: bool = True,
        enable_depth: bool = True,
        enable_collision: bool = True,
        device: str = 'cuda',
        conf_threshold: float = 0.25,
        ttc_threshold: float = 2.0,  # seconds
        safety_margin: float = 3.0,   # meters
    ):
        """
        Args:
            detector: Model name (yolo11n/m/l/x, rtdetr-l)
            detector_weights: Path to fine-tuned weights
            enable_lane: Enable lane detection
            enable_depth: Enable depth estimation
            enable_collision: Enable TTC calculation
            device: cuda / cpu
            conf_threshold: Detection confidence
            ttc_threshold: Alert if TTC < this value
            safety_margin: Safety distance threshold
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.ttc_threshold = ttc_threshold
        self.safety_margin = safety_margin

        # Load detector
        if not HAS_ULTRALYTICS:
            raise ImportError("pip install ultralytics")

        self.detector = YOLO(detector_weights or f'{detector}.pt')
        self.detector.to(device)
        print(f'[ADAS] Loaded detector: {detector}')

        # Lane detector (placeholder — use Ultra-Fast v2 in production)
        self.enable_lane = enable_lane
        self.lane_detector = None
        if enable_lane:
            print(f'[ADAS] Lane detection: enabled (stub)')

        # Depth estimator (placeholder — use Depth Anything V2 in production)
        self.enable_depth = enable_depth
        self.depth_estimator = None
        if enable_depth:
            print(f'[ADAS] Depth estimation: enabled (stub)')

        # Tracking state
        self.enable_collision = enable_collision
        self.tracks = {}          # track_id → {box, center_3d, velocity}
        self.next_track_id = 1

        # Visualization
        self.visualizer = ADASVisualizer()

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> ADASResult:
        """
        Process a single frame through the full ADAS pipeline.

        Returns:
            ADASResult with detections, lanes, depth, and collision alerts
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. Object Detection
        detections = self._detect(frame)

        # 2. Tracking & Velocity Estimation
        tracked_objects = self._track(detections, frame_id)

        # 3. Lane Detection
        lanes = None
        if self.enable_lane:
            lanes = self._detect_lanes(frame)

        # 4. Depth Estimation
        depth_map = None
        if self.enable_depth:
            depth_map = self._estimate_depth(frame)

        # 5. Collision Detection / TTC Calculation
        collision_alerts = []
        if self.enable_collision and depth_map is not None:
            collision_alerts = self._calculate_ttc(tracked_objects, depth_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ADASResult(
            timestamp=t0,
            frame=frame,
            detections=detections,
            lanes=lanes,
            depth_map=depth_map,
            collision_alerts=collision_alerts,
            latency_ms=elapsed_ms,
        )

    def _detect(self, frame: np.ndarray) -> List:
        """Run YOLOv11 detection."""
        results = self.detector.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        detections = []
        if results and results[0].boxes is not None:
            r = results[0]
            for box, score, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
            ):
                detections.append({
                    'box': box,
                    'score': float(score),
                    'class_id': int(cls_id),
                    'class_name': r.names[cls_id],
                })
        return detections

    def _track(self, detections: List, frame_id: int) -> List:
        """
        Simplified tracking: associate detections to existing tracks by IoU.
        Returns tracked objects with track_id and velocity.
        """
        if not detections:
            return []

        # Associate detections to existing tracks
        new_tracks = {}
        for det in detections:
            best_iou = 0.3
            best_track_id = None

            for track_id, track in self.tracks.items():
                iou = self._iou(det['box'], track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                prev_center = track['center']
                curr_center = self._box_center(det['box'])
                velocity = np.array(curr_center) - np.array(prev_center)
                new_tracks[best_track_id] = {
                    'box': det['box'],
                    'center': curr_center,
                    'velocity': velocity,
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'score': det['score'],
                    'age': self.tracks[best_track_id]['age'] + 1,
                }
            else:
                # New track
                center = self._box_center(det['box'])
                new_tracks[self.next_track_id] = {
                    'box': det['box'],
                    'center': center,
                    'velocity': np.array([0., 0.]),
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'score': det['score'],
                    'age': 1,
                }
                self.next_track_id += 1

        self.tracks = new_tracks

        # Return with track IDs
        tracked = []
        for track_id, track in self.tracks.items():
            tracked.append({
                **track,
                'track_id': track_id,
            })
        return tracked

    def _detect_lanes(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder for lane detection.
        In production: use Ultra-Fast Lane Detection v2 or CLRNet
        """
        if not self.enable_lane or self.lane_detector is None:
            return None
        # TODO: Implement actual lane detection
        return None

    def _estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder for monocular depth estimation.
        In production: use Depth Anything V2
        """
        if not self.enable_depth or self.depth_estimator is None:
            return None
        # TODO: Implement actual depth estimation
        return None

    def _calculate_ttc(
        self,
        tracked_objects: List,
        depth_map: np.ndarray,
    ) -> List:
        """
        Calculate Time-to-Collision for each tracked vehicle.
        TTC = distance_to_ego / relative_velocity
        """
        alerts = []
        h, w = depth_map.shape[:2]

        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj['box'])
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            # Clamp to image bounds
            x_center = max(0, min(x_center, w - 1))
            y_center = max(0, min(y_center, h - 1))

            # Get depth at object center
            distance = float(depth_map[y_center, x_center])

            # Relative velocity (simplified — velocity[1] is forward direction in camera coords)
            relative_velocity = abs(obj['velocity'][1]) + 0.1  # avoid divide by zero

            # TTC = distance / velocity
            ttc = distance / (relative_velocity + 1e-6)

            risk_level = 'safe'
            if ttc < self.ttc_threshold:
                if ttc < 1.0:
                    risk_level = 'critical'
                else:
                    risk_level = 'warning'

                alerts.append({
                    'track_id': obj['track_id'],
                    'class': obj['class_name'],
                    'distance_m': distance,
                    'velocity': relative_velocity,
                    'ttc_s': ttc,
                    'risk': risk_level,
                })

        return alerts

    @staticmethod
    def _box_center(box: np.ndarray) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def visualize(self, result: ADASResult) -> np.ndarray:
        """Overlay all detections, lanes, and alerts on frame."""
        return self.visualizer.draw_adas_result(result)


class ADASVisualizer:
    """Visualization for ADAS perception results."""

    def draw_adas_result(self, result: ADASResult) -> np.ndarray:
        """Draw detection boxes, track IDs, lanes, and collision warnings."""
        img = result.frame.copy()
        h, w = img.shape[:2]

        # Draw detections + tracks
        for det in result.detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)

        # Draw collision warnings (red if critical, yellow if warning)
        for alert in result.collision_alerts:
            color = (0, 0, 255) if alert['risk'] == 'critical' else (0, 255, 255)
            text = f"⚠ {alert['class']} TTC={alert['ttc_s']:.1f}s {alert['risk'].upper()}"
            cv2.putText(img, text, (10, 30 + len(result.collision_alerts) * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # FPS
        fps = 1000 / max(result.latency_ms, 1)
        cv2.putText(img, f"FPS: {fps:.1f} | Latency: {result.latency_ms:.1f}ms",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python advanced_pipeline.py <video_path>')
        sys.exit(1)

    video_path = sys.argv[1]

    print('[ADAS] Initializing perception system...')
    adas = ADASPerceptionSystem(
        detector='yolo11m',
        enable_lane=True,
        enable_depth=True,
        enable_collision=True,
        device='cuda',
    )

    print(f'[ADAS] Processing video: {video_path}')
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = adas.process_frame(frame, frame_count)
        annotated = adas.visualize(result)

        print(f'Frame {frame_count}: {result.latency_ms:.1f}ms, '
              f'{len(result.detections)} objects, {len(result.collision_alerts)} alerts')

        cv2.imshow('ADAS Perception', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f'[ADAS] Processed {frame_count} frames')
