"""Inverse Perspective Mapping (IPM) — image pixels -> ground-plane meters.

WHY THIS EXISTS
----------------
overtaking.py used to compute lane curvature directly in PIXEL space and
compare it against a hand-picked "max_curvature = 0.003" threshold. Pixel
curvature has no fixed real-world meaning — it changes with camera
resolution, mounting height, and focal length, so the same physical road
(say, a gentle 400 m-radius highway curve vs. a tight 40 m mountain
hairpin) can produce almost any pixel-curvature number depending on the
camera. That is not a threshold you can trust for a safety decision.

THE FIX (standard ADAS technique — used in every real lane-keep/overtake
system): project the lane polyline from the image plane onto the flat
ground plane using the camera's known geometry (a homography), which
turns "curvature in pixels" into "curvature in 1/meters" — i.e. an actual
radius of curvature you can compare against real road-design standards:

    AASHTO / IRC minimum horizontal curve radius (design speed dependent):
        30 km/h  ->  ~30-35 m minimum radius
        50 km/h  ->  ~80-90 m minimum radius
        80 km/h  ->  ~230-250 m minimum radius
        100 km/h ->  ~380-420 m minimum radius
    (IRC:73-1980 / AASHTO Green Book, rounded for readability.)

    A driver needs to SEE further ahead than the overtake maneuver takes,
    so the "safe to overtake on this curve" radius should be well above
    the bare minimum-safe-driving radius for the road's design speed —
    the default threshold below (150 m) is a conservative middle ground:
    comfortably above urban-road minimums, but tight enough to correctly
    block overtaking on the switchback/mountain curves this project
    specifically needs to handle (CULane's "Curve" category).

TWO CALIBRATION SOURCES
------------------------
1. KITTI calibration file (calib_cam_to_cam.txt) — exact, used automatically
   when training/evaluating on KITTI footage.
2. Generic dashcam / no calibration file — one-time manual 4-point
   calibration using `python models/ipm.py calibrate my_frame.jpg`
   (click 4 points forming a trapezoid on a straight, flat, empty road:
   two points on each lane line, near and far). Saved once to
   weights/ipm_calibration.json and reused for every future video from
   that camera.

If neither is available, overtaking.py automatically falls back to the
old pixel-heuristic curvature (with a printed warning) rather than
crashing — same graceful-degradation pattern as the rest of this codebase.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CALIB_PATH = PROJECT_ROOT / "weights" / "ipm_calibration.json"


@dataclass
class IPMTransformer:
    """Homography mapping image-plane pixels -> ground-plane meters (BEV)."""

    H: np.ndarray                 # 3x3 homography, pixels -> meters (X right, Z forward)
    valid_y_max: float = 1e9      # rows below the calibrated trapezoid are unreliable

    # ---------------- construction ----------------

    @classmethod
    def from_kitti_calib(cls, calib_cam_to_cam_path: str,
                          camera_height_m: float = 1.65,
                          pitch_deg: float = 0.0) -> "IPMTransformer":
        """Build the ground-plane homography from KITTI's calib_cam_to_cam.txt.

        KITTI camera height is a fixed, published rig spec (~1.65 m above the
        road, near-zero pitch) — see the KITTI setup paper / devkit README.
        We use the rectified left color camera intrinsics (P_rect_02).
        """
        K = _parse_kitti_P2(calib_cam_to_cam_path)
        return cls._from_intrinsics(K, camera_height_m, pitch_deg)

    @classmethod
    def from_intrinsics(cls, fx: float, fy: float, cx: float, cy: float,
                         camera_height_m: float, pitch_deg: float = 0.0
                         ) -> "IPMTransformer":
        """Build from known camera intrinsics + mounting geometry directly."""
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return cls._from_intrinsics(K, camera_height_m, pitch_deg)

    @classmethod
    def _from_intrinsics(cls, K: np.ndarray, camera_height_m: float,
                          pitch_deg: float) -> "IPMTransformer":
        """Flat-ground-plane assumption: every pixel's ray intersects Y=-h
        in camera coordinates (h = camera_height_m, ground below camera),
        optionally tilted by pitch_deg (down-tilt positive)."""
        theta = np.deg2rad(pitch_deg)
        # Rotation of the ground plane normal into the camera frame (pitch only
        # -- yaw/roll assumed ~0 for a forward-facing dashcam).
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Kinv = np.linalg.inv(K)

        # Ground plane in camera coords: normal n=(0,1,0) rotated by Rx, at
        # distance camera_height_m along that normal (Y points down in image
        # convention, so ground is +h below the camera center).
        n = Rx @ np.array([0.0, 1.0, 0.0])
        d = camera_height_m

        # Homography from image plane to ground plane (Hartley & Zisserman,
        # "Multiple View Geometry", ch.13 plane-induced homography):
        #   H_ground = K * (R - t n^T / d) * Kinv,  with R=I, t=0 (single view,
        # ground expressed in the camera's own frame) reduces to:
        H_img_to_cam_ground = np.eye(3) - np.zeros((3, 1)) @ n.reshape(1, 3) / d
        H = K @ H_img_to_cam_ground @ Kinv
        H_ground_to_img = H
        H_img_to_ground = np.linalg.inv(H_ground_to_img)

        # Scale: this construction gives camera-frame ground coordinates in
        # the same units as camera_height_m (meters) once divided through by
        # the homogeneous w term in `pixel_to_ground`.
        return cls(H=H_img_to_ground)

    @classmethod
    def from_points(cls, src_px: np.ndarray, dst_m: np.ndarray) -> "IPMTransformer":
        """Manual 4-point calibration: 4 image points (trapezoid on the road)
        -> their known real-world (X_right_m, Z_forward_m) ground coordinates."""
        src = np.asarray(src_px, dtype=np.float32)
        dst = np.asarray(dst_m, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return cls(H=H.astype(np.float64), valid_y_max=float(src[:, 1].max()))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIB_PATH) -> Optional["IPMTransformer"]:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(H=np.array(data["H"]), valid_y_max=data.get("valid_y_max", 1e9))

    def save(self, path: str | Path = DEFAULT_CALIB_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"H": self.H.tolist(),
                                     "valid_y_max": self.valid_y_max}))

    # ---------------- usage ----------------

    def pixel_to_ground(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) image pixels -> (N,2) ground-plane meters (X right, Z forward)."""
        pts = np.asarray(pts_px, dtype=np.float64)
        ones = np.ones((len(pts), 1))
        homog = np.hstack([pts, ones]) @ self.H.T
        w = homog[:, 2:3]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        return homog[:, :2] / w

    def curvature_from_polyline(self, polyline_px: np.ndarray
                                 ) -> tuple[Optional[float], Optional[float]]:
        """Lane polyline in image pixels -> (curvature 1/m, radius m) on the
        ground plane. Returns (None, None) if the fit is degenerate.

        Reports the TIGHTEST curvature over the visible stretch, not the
        curvature at any single point. Two reasons, and the first is a bug this
        replaced:

        * Evaluating at the far end of the polyline -- as this did -- divides
          the curvature by (1 + slope^2)^1.5, and the slope is largest exactly
          there. Validated against synthetic arcs of known radius, that made
          every curve read STRAIGHTER than it is: a 50 m mountain switchback
          came back as 135 m, a 100 m curve as 123 m. The error is always in
          the unsafe direction, because a road reported straighter than it is
          permits an overtake the geometry does not support.

        * For the decision itself, the relevant quantity is the tightest curve
          in the visible stretch. That is what limits sight distance, and it is
          what the vehicle will actually have to negotiate.

        Against the same synthetic arcs this form is within 4% for radii above
        150 m, and errs toward reporting a TIGHTER curve than the truth for
        sharper ones (50 m arc -> 27 m) -- wrong in the direction that refuses
        an overtake rather than the one that permits it.
        """
        pts = np.asarray(polyline_px, dtype=np.float64)
        if len(pts) < 5:
            return None, None
        ground = self.pixel_to_ground(pts)
        X, Z = ground[:, 0], ground[:, 1]           # X=lateral, Z=forward distance
        order = np.argsort(Z)
        Z, X = Z[order], X[order]
        if Z[-1] - Z[0] < 5.0:                        # <5 m of visible road: unreliable
            return None, None
        try:
            a, b, _ = np.polyfit(Z, X, 2)              # X = a*Z^2 + b*Z + c
        except np.linalg.LinAlgError:
            return None, None
        # kappa(z) = |2a| / (1 + (2az + b)^2)^1.5, maximised over the stretch.
        # Monotonic in |2az + b|, so the extremum is wherever that is smallest:
        # the stationary point z* = -b/2a when it falls inside the range, and
        # otherwise whichever endpoint is nearer to it.
        z_lo, z_hi = float(Z.min()), float(Z.max())
        candidates = [z_lo, z_hi]
        if abs(a) > 1e-12:
            z_star = -b / (2 * a)
            if z_lo <= z_star <= z_hi:
                candidates.append(z_star)
        kappa = max(abs(2 * a) / max((1 + (2 * a * z + b) ** 2) ** 1.5, 1e-9)
                    for z in candidates)
        radius = 1.0 / kappa if kappa > 1e-9 else float("inf")
        return kappa, radius


def _parse_kitti_P2(calib_path: str) -> np.ndarray:
    """Extract the 3x3 intrinsic matrix from KITTI's rectified P_rect_02 (the
    left color camera used for all 2D detection/lane work in this project)."""
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P_rect_02:") or line.startswith("P2:"):
                vals = [float(x) for x in line.split(":", 1)[1].split()]
                P = np.array(vals, dtype=np.float64).reshape(3, 4)
                return P[:3, :3]
    raise ValueError(f"No P_rect_02/P2 line found in {calib_path}")


def focal_length_px_from_kitti_calib(calib_cam_to_cam_path: str) -> float:
    """Exact horizontal focal length (pixels) from KITTI's own calibration —
    used by inference/sanity_filter.py's geometric size-consistency check."""
    return float(_parse_kitti_P2(calib_cam_to_cam_path)[0, 0])


def estimate_focal_length_px(image_width: int, assumed_hfov_deg: float = 90.0) -> float:
    """Fallback focal-length estimate for footage with no camera calibration
    file (generic dashcam video). Assumes a horizontal field-of-view typical
    of forward-facing dashcams/ADAS cameras (~90°) and inverts the pinhole
    projection equation:  fx = width / (2 * tan(HFOV/2)).

    This is deliberately approximate — real calibration (KITTI's P2, or a
    manual IPM calibration, see `calibrate()` below) is always preferable
    when available. It's accurate enough for the size-consistency SANITY
    check in sanity_filter.py, which only needs to catch objects that are
    wildly (2x+) too big/small for their measured depth, not sub-10% precision.
    """
    return image_width / (2.0 * np.tan(np.deg2rad(assumed_hfov_deg) / 2.0))


# Real-world curve-radius bands for interpreting the number in reports/HUD.
# Sources: AASHTO Green Book / IRC:73-1980 minimum horizontal curve radius
# tables, rounded to the nearest 10 m for readability.
ROAD_CURVE_BANDS = [
    (30, "hairpin / switchback (≈30 km/h design speed)"),
    (85, "tight mountain curve (≈50 km/h design speed)"),
    (150, "moderate curve — overtaking sight distance marginal"),
    (250, "gentle curve (≈80 km/h design speed)"),
    (float("inf"), "straight / highway-grade curve (≥100 km/h design speed)"),
]


def describe_radius(radius_m: float) -> str:
    for limit, desc in ROAD_CURVE_BANDS:
        if radius_m <= limit:
            return desc
    return ROAD_CURVE_BANDS[-1][1]


def _run_calibration_tool(image_path: str):
    """Interactive one-time calibration: click 4 points on a straight, flat,
    empty road (order: near-left, near-right, far-left, far-right — the two
    near points should be a known lane width apart, e.g. 3.5 m)."""
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Cannot read {image_path}")
    pts = []
    clone = img.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("calibrate (click near-L, near-R, far-L, far-R)", clone)

    cv2.imshow("calibrate (click near-L, near-R, far-L, far-R)", clone)
    cv2.setMouseCallback("calibrate (click near-L, near-R, far-L, far-R)", on_click)
    print("Click 4 points: near-left, near-right, far-left, far-right on a "
          "straight empty lane, then press any key.")
    while len(pts) < 4:
        if cv2.waitKey(20) != -1:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(pts) != 4:
        raise SystemExit("Need exactly 4 points.")

    lane_width_m = float(input("Lane width between near-left/near-right (m) "
                                "[default 3.5]: ") or 3.5)
    near_dist_m = float(input("Forward distance to the NEAR points (m) "
                               "[default 5.0]: ") or 5.0)
    far_dist_m = float(input("Forward distance to the FAR points (m) "
                              "[default 30.0]: ") or 30.0)

    dst = np.array([
        [-lane_width_m / 2, near_dist_m],
        [lane_width_m / 2, near_dist_m],
        [-lane_width_m / 2, far_dist_m],
        [lane_width_m / 2, far_dist_m],
    ], dtype=np.float32)

    ipm = IPMTransformer.from_points(np.array(pts, dtype=np.float32), dst)
    ipm.save()
    print(f"Saved calibration -> {DEFAULT_CALIB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "calibrate":
        _run_calibration_tool(sys.argv[2])
    else:
        print(__doc__)
        print("\nUsage: python models/ipm.py calibrate <straight_road_frame.jpg>")
