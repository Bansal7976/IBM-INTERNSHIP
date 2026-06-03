"""
Visualization utilities for detection and tracking results.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2

# Colormap: 20 distinct colors for class labels / track IDs
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


class Visualizer:
    """Draw bounding boxes, labels, and tracks on images."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.class_names = class_names
        self.thickness = line_thickness
        self.font_scale = font_scale

    def _color(self, idx: int) -> Tuple[int, int, int]:
        return PALETTE[idx % len(PALETTE)]

    def draw_detections(self, image: np.ndarray, detections, alpha: float = 0.0) -> np.ndarray:
        """Draw detections on image. detections: List[Detection]."""
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.box)
            color = self._color(det.class_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)

            label = f'{det.class_name} {det.score:.2f}'
            if hasattr(det, 'track_id') and det.track_id is not None:
                label = f'#{det.track_id} {label}'

            self._draw_label(img, label, (x1, y1), color)

        return img

    def draw_ultralytics_result(
        self, image: np.ndarray, result, show_tracks: bool = False
    ) -> np.ndarray:
        """Draw from a raw Ultralytics Results object."""
        img = image.copy()
        if result.boxes is None:
            return img

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int) if (show_tracks and result.boxes.id is not None) else None

        for i, (box, score, cid) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box
            color = self._color(int(track_ids[i]) if track_ids is not None else cid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)

            cls_name = result.names[cid]
            label = f'{cls_name} {score:.2f}'
            if track_ids is not None:
                label = f'#{track_ids[i]} {label}'

            self._draw_label(img, label, (x1, y1), color)

        return img

    def draw_3d_bev(
        self,
        detections_3d,
        bev_size: Tuple[int, int] = (600, 600),
        range_m: float = 50.0,
    ) -> np.ndarray:
        """Draw Bird's Eye View visualization of 3D detections."""
        bev = np.zeros((*bev_size, 3), dtype=np.uint8)
        bev[:] = (30, 30, 30)  # dark background

        # Draw ego vehicle at center
        cx, cy = bev_size[0] // 2, bev_size[1] // 2
        cv2.rectangle(bev, (cx - 10, cy - 20), (cx + 10, cy + 20), (0, 255, 0), -1)

        # Draw grid
        scale = bev_size[0] / (2 * range_m)  # pixels per meter
        for r in range(10, int(range_m) + 1, 10):
            radius = int(r * scale)
            cv2.circle(bev, (cx, cy), radius, (50, 50, 50), 1)
            cv2.putText(bev, f'{r}m', (cx + radius + 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

        for det in detections_3d:
            # Project 3D center to BEV
            px = int(cx + det.center[0] * scale)
            py = int(cy - det.center[1] * scale)  # y forward in camera coords

            if not (0 <= px < bev_size[0] and 0 <= py < bev_size[1]):
                continue

            color = self._color(det.class_id)
            w_px = max(4, int(det.size[0] * scale))
            l_px = max(6, int(det.size[1] * scale))

            self._draw_rotated_box_bev(bev, px, py, w_px, l_px, det.yaw, color)
            cv2.putText(bev, det.class_name[:3], (px + 5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return bev

    def _draw_rotated_box_bev(
        self, img, cx, cy, w, h, angle, color
    ):
        """Draw a rotated rectangle on BEV image."""
        import math
        corners = np.array([
            [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
        ])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (R @ corners.T).T
        corners += np.array([cx, cy])
        corners = corners.astype(int)
        cv2.polylines(img, [corners], True, color, 2)

    def _draw_label(self, img, text, pos, color):
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
        y_text = max(y - 4, th + 4)
        cv2.rectangle(img, (x, y_text - th - 4), (x + tw + 4, y_text + 2), color, -1)
        cv2.putText(img, text, (x + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1)
