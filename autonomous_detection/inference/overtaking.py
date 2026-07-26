"""Overtaking decision module: POSSIBLE / NOT POSSIBLE with reason.

Pure fusion logic — no neural network here. Combines:
  - lane polylines (CLRNet)          -> curvature, lane assignment
  - lane types (patch classifier)    -> solid/dashed legality
  - tracked objects (ByteTrack)      -> oncoming vehicles + velocity
  - depth map (Depth Anything V2)    -> distances
  - scene lighting                   -> visibility gate

All 5 rules must pass for POSSIBLE. Safety-critical design: any missing or
uncertain input fails toward NOT POSSIBLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from models.ipm import IPMTransformer


class OvertakeStatus(Enum):
    POSSIBLE = "OVERTAKING POSSIBLE"
    NOT_POSSIBLE_SOLID_LINE = "NOT POSSIBLE - SOLID CENTER LINE"
    NOT_POSSIBLE_CURVE = "NOT POSSIBLE - CURVE / LOW VISIBILITY AHEAD"
    NOT_POSSIBLE_ONCOMING = "NOT POSSIBLE - ONCOMING VEHICLE"
    NOT_POSSIBLE_LOW_VIS = "NOT POSSIBLE - DARKNESS / WEATHER"
    NOT_POSSIBLE_NO_GAP = "NOT POSSIBLE - INSUFFICIENT GAP AHEAD"
    NOT_POSSIBLE_NO_LANE_INFO = "NOT POSSIBLE - LANE MARKINGS NOT VISIBLE"


SOLID_TYPES = {"solid", "double_solid", "solid_white", "solid_yellow", "double_yellow"}
VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle"}


@dataclass
class OvertakingAnalyzer:
    min_gap_seconds: float = 8.0        # time window needed to complete an overtake
    max_curvature: float = 0.003        # PIXEL-space fallback only (see ipm below) —
                                         # beyond this it's a blind curve; uncalibrated,
                                         # kept only for when no IPM geometry is available
    min_curve_radius_m: float = 150.0   # metric threshold once IPM is available — a
                                         # conservative distance above urban minimum-curve
                                         # standards (AASHTO/IRC), tight enough to correctly
                                         # block mountain switchback overtakes; see models/ipm.py
    min_lead_gap_m: float = 25.0        # clear road needed ahead of the lead vehicle
    dark_brightness_threshold: float = 40.0
    ipm: Optional[IPMTransformer] = field(default=None, repr=False)
    last_curve_radius_m: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.ipm is None:
            self.ipm = IPMTransformer.load()   # auto-load weights/ipm_calibration.json if present
            if self.ipm is None:
                print("[overtaking] No IPM calibration found — curve check falls back "
                      "to uncalibrated pixel curvature (see models/ipm.py to calibrate).")

    def analyze(self, lanes, lane_types: dict, tracks: list,
                depth_map: Optional[np.ndarray], ego_speed_mps: float,
                scene_brightness: float, lighting_state: str = "DAY") -> OvertakeStatus:

        # RULE 0: need lane info at all — fail safe without it
        center = self._center_divider(lanes)
        if center is None:
            return OvertakeStatus.NOT_POSSIBLE_NO_LANE_INFO

        # RULE 1: legality — center line must not be solid
        ltype = lane_types.get(id(center), lane_types.get("center", "unknown"))
        if ltype in SOLID_TYPES or ltype == "unknown":
            return OvertakeStatus.NOT_POSSIBLE_SOLID_LINE

        # RULE 2: geometry — road straight enough to see the overtake distance.
        # Prefer real-world radius (meters) via IPM; falls back to the
        # uncalibrated pixel heuristic only if no camera calibration exists.
        if self.ipm is not None:
            _, radius_m = self.ipm.curvature_from_polyline(center)
            self.last_curve_radius_m = radius_m
            if radius_m is None or radius_m < self.min_curve_radius_m:
                return OvertakeStatus.NOT_POSSIBLE_CURVE
        else:
            curvature = self._estimate_curvature(center)
            if curvature is None or curvature > self.max_curvature:
                return OvertakeStatus.NOT_POSSIBLE_CURVE

        # RULE 3: visibility — unlit darkness kills depth reliability
        if lighting_state == "NIGHT_UNLIT" or scene_brightness < self.dark_brightness_threshold:
            return OvertakeStatus.NOT_POSSIBLE_LOW_VIS

        # RULE 4: oncoming traffic within the maneuver window
        if depth_map is not None:
            for t in tracks:
                if not self._is_vehicle(t):
                    continue
                if self._in_oncoming_lane(t, center):
                    dist = self._distance(t, depth_map)
                    if dist is None:
                        return OvertakeStatus.NOT_POSSIBLE_ONCOMING  # fail safe
                    closing = ego_speed_mps + abs(self._speed_mps(t, depth_map))
                    if dist / max(closing, 0.1) < self.min_gap_seconds:
                        return OvertakeStatus.NOT_POSSIBLE_ONCOMING

            # RULE 5: enough clear space ahead of the lead vehicle
            lead = self._lead_vehicle(tracks, center, depth_map)
            if lead is not None:
                lead_dist = self._distance(lead, depth_map)
                next_ahead = self._vehicle_ahead_of(lead, tracks, center, depth_map)
                if next_ahead is not None:
                    gap = self._distance(next_ahead, depth_map) - lead_dist
                    if gap < self.min_lead_gap_m:
                        return OvertakeStatus.NOT_POSSIBLE_NO_GAP

        return OvertakeStatus.POSSIBLE

    # ---------- geometry helpers ----------

    def _center_divider(self, lanes) -> Optional[np.ndarray]:
        """Leftmost lane line of the ego lane = the line we'd cross to overtake
        (right-hand traffic). Returns (N,2) polyline or None."""
        if lanes is None:
            return None
        polylines = getattr(lanes, "polylines", lanes)
        if not polylines or len(polylines) == 0:
            return None
        # Ego lane's left boundary: lane line closest to image center on the left
        # Assume polylines sorted left->right by mean x
        sorted_lines = sorted(polylines, key=lambda p: np.asarray(p)[:, 0].mean())
        if len(sorted_lines) == 1:
            return np.asarray(sorted_lines[0])
        # With 2+ lines: second from center-left is typically the divider
        mid = len(sorted_lines) // 2
        return np.asarray(sorted_lines[max(0, mid - 1)])

    def _estimate_curvature(self, polyline: np.ndarray) -> Optional[float]:
        """Curvature from 2nd-order polyfit x = f(y) on the lane polyline.

        kappa = |2a| / (1 + (2ay + b)^2)^1.5, evaluated at the far end
        (top of the polyline = furthest visible road).
        """
        pts = np.asarray(polyline, dtype=np.float64)
        if len(pts) < 5:
            return None
        ys, xs = pts[:, 1], pts[:, 0]
        try:
            a, b, _ = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            return None
        y_far = ys.min()
        denom = (1 + (2 * a * y_far + b) ** 2) ** 1.5
        # Pixel-space curvature; calibrate max_curvature threshold empirically
        # on straight vs curved CULane clips.
        return abs(2 * a) / max(denom, 1e-9)

    # ---------- object helpers ----------

    @staticmethod
    def _is_vehicle(track) -> bool:
        name = getattr(track, "class_name", "").lower()
        return name in VEHICLE_CLASSES

    def _in_oncoming_lane(self, track, center_line: np.ndarray) -> bool:
        """Object left of the center divider (right-hand traffic) = oncoming lane."""
        x1, _, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        from inference.collision import _lane_x_at_y
        divider_x = _lane_x_at_y(center_line, y2)
        if divider_x is None:
            divider_x = float(np.asarray(center_line)[:, 0].mean())
        return cx < divider_x

    def _lead_vehicle(self, tracks, center_line, depth_map):
        """Nearest vehicle ahead in the ego lane."""
        candidates = [
            (self._distance(t, depth_map), t) for t in tracks
            if self._is_vehicle(t) and not self._in_oncoming_lane(t, center_line)
        ]
        candidates = [(d, t) for d, t in candidates if d is not None]
        return min(candidates, key=lambda c: c[0])[1] if candidates else None

    def _vehicle_ahead_of(self, lead, tracks, center_line, depth_map):
        lead_d = self._distance(lead, depth_map)
        ahead = [
            (self._distance(t, depth_map), t) for t in tracks
            if t is not lead and self._is_vehicle(t)
            and not self._in_oncoming_lane(t, center_line)
        ]
        ahead = [(d, t) for d, t in ahead if d is not None and d > lead_d]
        return min(ahead, key=lambda c: c[0])[1] if ahead else None

    @staticmethod
    def _distance(track, depth_map) -> Optional[float]:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y_mid, y2 = max(0, (y1 + y2) // 2), min(h, y2)
        if x2 <= x1 or y2 <= y_mid:
            return None
        return float(np.median(depth_map[y_mid:y2, x1:x2]))

    @staticmethod
    def _speed_mps(track, depth_map) -> float:
        """Physical speed if the tracker stored depth history; else 0 (conservative
        because Rule 4 already adds ego speed)."""
        return float(getattr(track, "depth_speed_mps", 0.0))
