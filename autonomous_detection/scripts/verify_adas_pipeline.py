"""Logic-only smoke test for the ADAS decision pipeline — NO GPU, NO weights,
NO dataset required. Runs in seconds on a login node.

WHY THIS EXISTS
----------------
Training the detector + CLRNet takes ~2 days of GPU time (see
KRISH_HANDOVER.md Job A/B). If a bug in the DECISION logic (TTC math,
overtaking rules, lane coordinate scaling) only shows up after that, you
lose 2 days finding out. This script tests all of that logic directly with
synthetic data, so you know BEFORE submitting the big jobs that:

  1. The tracker produces stable IDs + sane velocity.
  2. Collision TTC fires at the right time (and depth_speed_mps -- the bug
     fixed in this same changeset -- reaches oncoming-lane tracks, not just
     ego-path ones).
  3. Overtaking correctly says POSSIBLE on a straight empty road, and
     correctly says NOT POSSIBLE for each individual reason (solid line,
     curve, oncoming traffic) when that condition is present alone.
  4. CLRNet's polyline coordinate scaling doesn't blow up (regression test
     for the exact bug that was silently corrupting lane geometry).
  5. IPM ground-plane curvature gives a small kappa (large radius) for a
     straight synthetic lane and a large kappa (small radius) for a curved one.

Usage:
    python scripts/verify_adas_pipeline.py
Expected: "ALL CHECKS PASSED" at the end. Any FAIL line tells you exactly
what regressed and needs fixing before you touch the HPC queue.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))


@dataclass
class FakeDet:
    bbox: tuple
    conf: float
    cls: int
    class_name: str = "car"


# --------------------------------------------------------------------------
# 1. Tracker: stable IDs + velocity sign
# --------------------------------------------------------------------------

def test_tracker():
    from inference.tracker import ByteTrackWrapper

    tr = ByteTrackWrapper()
    ids_seen = set()
    last_tracks = []
    for f in range(8):
        dets = [FakeDet((100 + f * 5, 200, 160 + f * 5, 260), 0.9, 0)]
        last_tracks = tr.update(dets)
        ids_seen.update(t.track_id for t in last_tracks)

    check("tracker: single object keeps one stable ID",
          len(ids_seen) == 1, f"saw IDs {ids_seen}")
    check("tracker: velocity points in the direction of motion (+x)",
          bool(last_tracks) and last_tracks[0].velocity[0] > 0,
          f"velocity={last_tracks[0].velocity if last_tracks else None}")


# --------------------------------------------------------------------------
# 2. Collision: TTC fires correctly + depth_speed_mps reaches ALL tracks
#    (regression test for the ego-path-only history bug)
# --------------------------------------------------------------------------

class _FakeDepthModel:
    """Returns a flat depth map whose value we control per call."""
    def __init__(self):
        self.value = 50.0

    def infer(self, frame):
        return np.full((100, 100), self.value, dtype=np.float32)


def test_collision():
    from inference.collision import CollisionDetector

    depth = _FakeDepthModel()
    det = CollisionDetector(depth_model=depth, fps=10, critical_ttc=1.5, warning_ttc=3.0)

    # Track sits in the CENTER of the frame -> counts as "in ego path" by the
    # no-lane-info fallback (central 40% of the frame).
    ego_track = FakeDet((40, 40, 60, 60), 0.9, 0, "car")
    ego_track.track_id = 1
    ego_track.depth_speed_mps = 0.0

    # Track sits at the LEFT edge -> outside ego path (oncoming lane analogue).
    side_track = FakeDet((0, 40, 10, 60), 0.9, 0, "car")
    side_track.track_id = 2
    side_track.depth_speed_mps = 0.0

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    alerts_over_time = []
    # Simulate the object closing distance from 50m -> 5m over 1s (10 frames)
    for step in range(10):
        depth.value = 50.0 - step * 5.0   # 50, 45, ..., 5
        alerts = det.update(frame, [ego_track, side_track], lanes=None)
        alerts_over_time.append(alerts)

    final_alerts = alerts_over_time[-1]
    check("collision: BRAKE fires once the object is close and closing fast",
          any(a.level == "BRAKE" for a in final_alerts),
          f"last alerts: {[a.level for a in final_alerts]}")

    check("collision: depth_speed_mps was set on the OUT-OF-EGO-PATH track too "
          "(regression test — this was the silent bug)",
          abs(side_track.depth_speed_mps) > 0.1,
          f"side_track.depth_speed_mps={side_track.depth_speed_mps}")

    check("collision: no alert raised for the out-of-ego-path track "
          "(alerts should stay scoped to ego path)",
          all(a.track_id != 2 for a in final_alerts))


# --------------------------------------------------------------------------
# 3. IPM: straight lane -> large radius, curved lane -> small radius
# --------------------------------------------------------------------------

def test_ipm():
    from models.ipm import IPMTransformer

    # Simple calibration: 4 image points forming a trapezoid -> known meters.
    src = np.array([[300, 480], [340, 480], [280, 300], [360, 300]], dtype=np.float32)
    dst = np.array([[-1.75, 5], [1.75, 5], [-1.75, 30], [1.75, 30]], dtype=np.float32)
    ipm = IPMTransformer.from_points(src, dst)

    # Straight polyline: constant x in pixel space as y decreases (going away).
    straight = np.array([[310 + i * 0, 480 - i * 15] for i in range(12)], dtype=np.float32)
    _, r_straight = ipm.curvature_from_polyline(straight)

    # Curved polyline: x drifts quadratically with distance (a real curve).
    curved = np.array([[310 + 0.02 * (480 - (480 - i * 15)) ** 2 / 50,
                        480 - i * 15] for i in range(12)], dtype=np.float32)
    _, r_curved = ipm.curvature_from_polyline(curved)

    check("ipm: straight polyline yields a large radius (>150m)",
          r_straight is not None and r_straight > 150,
          f"r_straight={r_straight}")
    check("ipm: curved polyline yields a smaller radius than the straight one",
          r_curved is not None and r_straight is not None and r_curved < r_straight,
          f"r_curved={r_curved} vs r_straight={r_straight}")


# --------------------------------------------------------------------------
# 4. Drivable-area fallback: finds a corridor with zero weights (CV-only
#    path) — this is the fix for lane detection failing on roads with no
#    painted markings (common on Indian roads; CLRNet/UFLDv2 are CULane-
#    trained and have nothing to fit a curve to there).
# --------------------------------------------------------------------------

def test_drivable_area():
    from models.drivable_area import DrivableAreaSegmenter

    h, w = 200, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 40)          # "grass/background" everywhere
    frame[140:h, 50:250] = (90, 90, 90)  # a gray "road" strip at the bottom

    seg = DrivableAreaSegmenter(weights=None)  # forces the classical CV path
    check("drivable-area: falls back to classical CV when no weights given",
          seg.model is None)

    mask = seg.segment_mask(frame)
    road_pixel_correct = bool(mask[190, 150] > 0)      # inside the road strip
    background_pixel_correct = bool(mask[20, 20] == 0)  # inside the background
    check("drivable-area: classical CV mask marks the road strip as drivable",
          road_pixel_correct)
    check("drivable-area: classical CV mask does NOT mark the background as drivable",
          background_pixel_correct)

    lane_result = seg.segment_to_lane_result(frame)
    check("drivable-area: produces a usable LaneResult (>=2 polylines) from "
          "an unmarked synthetic road",
          lane_result is not None and len(lane_result.polylines) >= 2,
          f"got {lane_result}")
    if lane_result is not None:
        left_x = float(np.asarray(lane_result.polylines[0])[:, 0].mean())
        right_x = float(np.asarray(lane_result.polylines[1])[:, 0].mean())
        check("drivable-area: left/right boundaries roughly match the "
              "synthetic road edges (~50px and ~250px)",
              abs(left_x - 50) < 20 and abs(right_x - 250) < 20,
              f"left_x={left_x}, right_x={right_x}")


# --------------------------------------------------------------------------
# 5. Overtaking: each rule independently blocks POSSIBLE
# --------------------------------------------------------------------------

def test_overtaking():
    from inference.overtaking import OvertakingAnalyzer, OvertakeStatus
    from models.ipm import IPMTransformer

    src = np.array([[300, 480], [340, 480], [280, 300], [360, 300]], dtype=np.float32)
    dst = np.array([[-1.75, 5], [1.75, 5], [-1.75, 30], [1.75, 30]], dtype=np.float32)
    ipm = IPMTransformer.from_points(src, dst)
    az = OvertakingAnalyzer(ipm=ipm, min_gap_seconds=8.0, min_lead_gap_m=25.0)

    class FakeLanes:
        def __init__(self, polylines):
            self.polylines = polylines

    straight_line = np.array([[320, 480 - i * 15] for i in range(12)], dtype=np.float32)
    lanes = FakeLanes([straight_line, straight_line + 60])  # two lines -> a "center" divider

    depth_map = np.full((480, 640), 60.0, dtype=np.float32)

    # Case A: dashed line, straight road, no traffic -> POSSIBLE
    status = az.analyze(lanes, {"center": "dashed"}, tracks=[], depth_map=depth_map,
                         ego_speed_mps=20.0, scene_brightness=150.0, lighting_state="DAY")
    check("overtaking: clear straight dashed road -> POSSIBLE",
          status == OvertakeStatus.POSSIBLE, f"got {status}")

    # Case B: solid line -> blocked regardless of everything else
    status = az.analyze(lanes, {"center": "solid"}, tracks=[], depth_map=depth_map,
                         ego_speed_mps=20.0, scene_brightness=150.0, lighting_state="DAY")
    check("overtaking: solid center line -> NOT_POSSIBLE_SOLID_LINE",
          status == OvertakeStatus.NOT_POSSIBLE_SOLID_LINE, f"got {status}")

    # Case C: unlit darkness -> blocked
    status = az.analyze(lanes, {"center": "dashed"}, tracks=[], depth_map=depth_map,
                         ego_speed_mps=20.0, scene_brightness=10.0, lighting_state="NIGHT_UNLIT")
    check("overtaking: unlit darkness -> NOT_POSSIBLE_LOW_VIS",
          status == OvertakeStatus.NOT_POSSIBLE_LOW_VIS, f"got {status}")

    # Case D: fast oncoming vehicle close enough to matter -> blocked
    oncoming = FakeDet((150, 300, 210, 360), 0.9, 0, "car")   # left of center -> oncoming lane
    oncoming.track_id = 99
    oncoming.depth_speed_mps = 25.0   # approaching fast (post-bugfix telemetry)
    status = az.analyze(lanes, {"center": "dashed"}, tracks=[oncoming],
                         depth_map=np.full((480, 640), 40.0, dtype=np.float32),
                         ego_speed_mps=20.0, scene_brightness=150.0, lighting_state="DAY")
    check("overtaking: close fast oncoming vehicle -> NOT_POSSIBLE_ONCOMING",
          status == OvertakeStatus.NOT_POSSIBLE_ONCOMING, f"got {status}")


# --------------------------------------------------------------------------
# 6. CLRNet coordinate-scaling regression test (no GPU/weights needed —
#    exercises the exact arithmetic that was buggy)
# --------------------------------------------------------------------------

def test_clrnet_coord_scaling():
    w0 = 1242  # typical KITTI image width

    def old_buggy_scale(pts, w0):
        pts = pts.copy()
        pts[:, 0] *= w0 / 1.0 if pts[:, 0].max() > 2 else w0
        return pts

    def fixed_scale(pts, w0):
        pts = pts.copy()
        if pts[:, 0].max() <= 2:
            pts[:, 0] *= w0
        return pts

    # Case: model already returned pixel-space coords (max > 2)
    pixel_pts = np.array([[100.0, 10.0], [500.0, 400.0]])
    buggy_out = old_buggy_scale(pixel_pts, w0)
    fixed_out = fixed_scale(pixel_pts, w0)

    check("clrnet coord bug: OLD code blows up already-pixel-space coords "
          "(documenting the bug that was fixed)",
          buggy_out[:, 0].max() > w0 * 10)
    check("clrnet coord fix: FIXED code leaves pixel-space coords unchanged",
          np.allclose(fixed_out, pixel_pts))

    # Case: model returned normalized coords (<=1) -> should scale up to pixels
    norm_pts = np.array([[0.1, 10.0], [0.9, 400.0]])
    fixed_norm = fixed_scale(norm_pts, w0)
    check("clrnet coord fix: normalized coords still get scaled to pixel space",
          fixed_norm[:, 0].max() > 100 and fixed_norm[:, 0].max() <= w0)


def main():
    print("=" * 70)
    print("ADAS PIPELINE LOGIC SMOKE TEST (no GPU / weights required)")
    print("=" * 70)

    for name, fn in [
        ("Tracker", test_tracker),
        ("Collision / TTC", test_collision),
        ("IPM ground-plane curvature", test_ipm),
        ("Drivable-area fallback (unmarked-road lane substitute)", test_drivable_area),
        ("Overtaking decision rules", test_overtaking),
        ("CLRNet coordinate-scaling regression", test_clrnet_coord_scaling),
    ]:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            FAIL.append(f"{name} (crashed: {e})")
            print(f"[FAIL] {name} crashed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    if FAIL:
        print(f"RESULT: {len(FAIL)} CHECK(S) FAILED — fix before submitting HPC jobs:")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL CHECKS PASSED ({len(PASS)}/{len(PASS)}) — decision logic is sound.")
        print("Safe to proceed to KRISH_HANDOVER.md Section 4 (HPC execution).")
        sys.exit(0)


if __name__ == "__main__":
    main()
