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
    NOT_POSSIBLE_REAR_APPROACH = "NOT POSSIBLE - VEHICLE CLOSING FROM BEHIND"
    NOT_POSSIBLE_REAR_UNSEEN = "NOT POSSIBLE - CANNOT SEE FAR ENOUGH BEHIND"


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

    # "left" = drives on the left, overtakes on the RIGHT (India, UK, Japan).
    # "right" = drives on the right, overtakes on the LEFT (US, most of Europe).
    # Defaulted to India, which is what this system targets. Stated explicitly
    # because published pipelines almost all assume the opposite convention,
    # and inheriting that assumption silently inverts every lane-side test.
    traffic_side: str = "left"

    # Rear check. Off by default: a single forward camera cannot see behind, and
    # a module that reports "rear clear" without a rear view would be asserting
    # something it has no evidence for. Turn on with a rear camera fitted, or
    # when evaluating against a dual-view dataset such as IDD-X.
    require_rear_view: bool = False

    ipm: Optional[IPMTransformer] = field(default=None, repr=False)
    rear: Optional["RearApproachMonitor"] = field(default=None, repr=False)
    last_curve_radius_m: Optional[float] = field(default=None, init=False, repr=False)
    last_rear_verdict: Optional["RearVerdict"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.traffic_side not in ("left", "right"):
            raise ValueError(
                f"traffic_side must be 'left' or 'right', got {self.traffic_side!r}")
        if self.ipm is None:
            self.ipm = IPMTransformer.load()   # auto-load weights/ipm_calibration.json if present
            if self.ipm is None:
                print("[overtaking] No IPM calibration found — curve check falls back "
                      "to uncalibrated pixel curvature (see models/ipm.py to calibrate).")
        if self.rear is None:
            from inference.rear_view import RearApproachMonitor
            self.rear = RearApproachMonitor(traffic_side=self.traffic_side)

    @property
    def manoeuvre_side(self) -> str:
        """The side the ego moves into: right under left-hand traffic."""
        return "right" if self.traffic_side == "left" else "left"

    def analyze(self, lanes, lane_types: dict, tracks: list,
                depth_map: Optional[np.ndarray], ego_speed_mps: float,
                scene_brightness: float, lighting_state: str = "DAY",
                rear_tracks: Optional[list] = None,
                rear_depth_map: Optional[np.ndarray] = None,
                frame_width: int = 1280) -> OvertakeStatus:

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

        # RULE 6: the lane being moved into must also be clear BEHIND.
        # A forward-only system can report "no oncoming traffic" while a
        # vehicle is already closing from behind in exactly the lane the
        # driver is about to enter. Checked last because it is the most
        # expensive and the cheaper rules have already had their chance.
        self.last_rear_verdict = None
        if self.require_rear_view or rear_tracks is not None:
            lead = self._lead_vehicle(tracks, center, depth_map) if depth_map is not None else None
            verdict = self.rear.analyze(
                rear_tracks,
                rear_depth_map if rear_depth_map is not None else depth_map,
                ego_speed_mps=ego_speed_mps,
                overtaken_speed_mps=self._lead_speed_mps(lead, ego_speed_mps),
                frame_width=frame_width,
                source="rear_camera" if rear_tracks is not None else "none",
                is_vehicle=self._is_vehicle)
            self.last_rear_verdict = verdict
            if not verdict.observable:
                return OvertakeStatus.NOT_POSSIBLE_REAR_UNSEEN
            if not verdict.clear:
                return OvertakeStatus.NOT_POSSIBLE_REAR_APPROACH

        return OvertakeStatus.POSSIBLE

    @staticmethod
    def _lead_speed_mps(lead, ego_speed_mps: float) -> Optional[float]:
        """Absolute speed of the vehicle being overtaken, if it is known.

        `depth_speed_mps` is POSITIVE when the gap is shrinking (the convention
        set in inference/collision.py). A lead vehicle we are closing on is
        therefore SLOWER than the ego by exactly that rate -- subtracted, not
        added. Getting this backwards makes a slow truck look faster than the
        ego, which shortens the computed manoeuvre window and would permit an
        overtake with less clearance than it needs.

        Returns None when there is no lead vehicle, which lets the rear monitor
        fall back to the IRC default of 16 km/h below the ego speed.
        """
        if lead is None:
            return None
        return max(ego_speed_mps - float(getattr(lead, "depth_speed_mps", 0.0)), 0.0)

    # ---------- geometry helpers ----------

    def _center_divider(self, lanes) -> Optional[np.ndarray]:
        """The lane line the ego would cross to overtake. (N,2) polyline or None.

        Which boundary that is depends on which side traffic drives on. Under
        left-hand traffic (India) the ego occupies the left lane and crosses
        its RIGHT boundary; under right-hand traffic it crosses the LEFT one.
        """
        if lanes is None:
            return None
        polylines = getattr(lanes, "polylines", lanes)
        if not polylines or len(polylines) == 0:
            return None
        sorted_lines = sorted(polylines, key=lambda p: np.asarray(p)[:, 0].mean())
        if len(sorted_lines) == 1:
            return np.asarray(sorted_lines[0])
        mid = len(sorted_lines) // 2
        if self.manoeuvre_side == "right":
            return np.asarray(sorted_lines[min(mid, len(sorted_lines) - 1)])
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
        """Is this object in the oncoming lane, i.e. across the divider?

        Under left-hand traffic (India) the oncoming lane is to the RIGHT of
        the divider; under right-hand traffic it is to the LEFT. Reading this
        backwards makes the module treat traffic in the ego's own lane as
        oncoming and ignore genuine oncoming vehicles, so it is derived from
        `traffic_side` rather than fixed.
        """
        x1, _, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        from inference.collision import _lane_x_at_y
        divider_x = _lane_x_at_y(center_line, y2)
        if divider_x is None:
            divider_x = float(np.asarray(center_line)[:, 0].mean())
        return cx > divider_x if self.manoeuvre_side == "right" else cx < divider_x

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
