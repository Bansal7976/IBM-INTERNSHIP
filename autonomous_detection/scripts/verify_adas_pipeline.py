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


def _raises(fn) -> bool:
    """True if calling fn() raises — used to assert that bad input is rejected."""
    try:
        fn()
    except Exception:
        return True
    return False


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

def test_id_churn_resilience():
    """Collision detection must survive tracker ID switches.

    TTC needs several frames of a track's distance history before it can report
    a closing speed, and that history is keyed on track_id. So every ID switch
    throws it away -- and the module then raises no alert at all, which is
    indistinguishable from "the road is clear". Dense unstructured traffic,
    where two-wheelers weave and occlude each other constantly, is exactly
    where that churn happens.

    Measured below: without recovery, an ID switch every three frames drops
    closing-speed coverage to ZERO.
    """
    from inference.collision import CollisionDetector

    class RampDepth:
        """A scene approaching at a steady rate."""
        def __init__(self):
            self.d = 60.0

        def infer(self, frame):
            self.d -= 0.5
            return np.full(frame.shape[:2], self.d, dtype=np.float32)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def coverage(switch_every: int, rescue: bool) -> dict:
        det = CollisionDetector(RampDepth(), fps=30,
                                rescue_across_id_switch=rescue)
        for i in range(60):
            tid = 1 if switch_every == 0 else 1 + i // switch_every
            x = 280 + i // 6          # drifts as an approaching vehicle would
            t = FakeDet((x, 200, x + 80, 400), 0.9, 0, "car")
            t.track_id = tid
            t.depth_speed_mps = 0.0
            det.update(frame, [t])
        return det.stats()

    stable = coverage(0, False)["closing_speed_coverage"]
    check("id-churn: with stable ids, closing speed is available for most "
          "sightings (the ceiling this mechanism aims at)",
          stable > 0.85, f"{stable}")

    dense_off = coverage(3, False)["closing_speed_coverage"]
    check("id-churn: WITHOUT recovery, an ID switch every 3 frames drops "
          "coverage to zero -- the collision layer goes silent, and silently",
          dense_off == 0.0, f"{dense_off}")

    dense_on = coverage(3, True)
    check("id-churn: WITH recovery, the same churn keeps coverage at the "
          "stable-tracker level",
          dense_on["closing_speed_coverage"] >= stable - 0.01,
          f"{dense_on['closing_speed_coverage']} vs stable {stable}")
    check("id-churn: and it reports how many histories it had to rescue, so "
          "tracker quality stays visible rather than papered over",
          dense_on["histories_rescued"] > 10, str(dense_on))

    worst = coverage(1, True)["closing_speed_coverage"]
    check("id-churn: coverage holds even when the id changes EVERY frame",
          worst >= stable - 0.01, f"{worst}")

    for every in (5, 10, 20):
        off = coverage(every, False)["closing_speed_coverage"]
        on = coverage(every, True)["closing_speed_coverage"]
        check(f"id-churn: recovery improves coverage at a switch every "
              f"{every} frames ({off:.2f} -> {on:.2f})", on > off,
              f"{off} -> {on}")

    # The recovery must not invent continuity between DIFFERENT objects.
    det = CollisionDetector(RampDepth(), fps=30)
    left = FakeDet((50, 200, 130, 400), 0.9, 0, "car")
    left.track_id = 1
    left.depth_speed_mps = 0.0
    for _ in range(8):
        det.update(frame, [left])
    before = det.stats()["histories_rescued"]

    far_away = FakeDet((500, 200, 580, 400), 0.9, 0, "car")   # no overlap at all
    far_away.track_id = 2
    far_away.depth_speed_mps = 0.0
    det.update(frame, [far_away])
    check("id-churn: a genuinely different object elsewhere in the frame does "
          "NOT inherit the vanished track's history",
          det.stats()["histories_rescued"] == before,
          f"rescued {det.stats()['histories_rescued']} (was {before})")

    # A history parked too long is worse than none: it would measure closing
    # speed against where the object was many frames ago.
    det2 = CollisionDetector(RampDepth(), fps=30, rescue_max_age_frames=2)
    a = FakeDet((280, 200, 360, 400), 0.9, 0, "car")
    a.track_id = 1
    a.depth_speed_mps = 0.0
    for _ in range(8):
        det2.update(frame, [a])
    for _ in range(5):                       # object absent for 5 frames
        det2.update(frame, [])
    b = FakeDet((280, 200, 360, 400), 0.9, 0, "car")
    b.track_id = 2
    b.depth_speed_mps = 0.0
    det2.update(frame, [b])
    check("id-churn: a history parked longer than rescue_max_age_frames is "
          "expired rather than reused stale",
          det2.stats()["histories_rescued"] == 0,
          str(det2.stats()))

    check("id-churn: the mechanism can be switched off entirely",
          coverage(3, False)["closing_speed_coverage"] == 0.0)


def test_group_scaled_margins():
    """The decision taxonomy must actually change what the system does.

    A grouping that only relabels boxes is a labelling scheme, not a decision
    layer. These checks confirm the group's ttc_margin_scale reaches the
    collision thresholds -- a pedestrian gets a longer margin than a car
    because the group says so, not because a constant was tuned per class.
    """
    from inference.collision import CollisionDetector
    from models.taxonomy import GROUP_BEHAVIOUR

    class FakeDepth:
        def infer(self, frame):
            return np.full(frame.shape[:2], 30.0, dtype=np.float32)

    det = CollisionDetector(FakeDepth(), fps=30)
    base_c, base_w = det.critical_ttc, det.warning_ttc

    def thresholds(class_name):
        return det._thresholds_for(FakeDet((0, 0, 10, 10), 0.9, 0, class_name))

    ped_c, ped_w = thresholds("pedestrian")
    car_c, car_w = thresholds("car")
    truck_c, _ = thresholds("truck")
    cone_c, _ = thresholds("traffic cone")

    check("margins: a pedestrian gets a LONGER brake margin than a car",
          ped_c > car_c, f"pedestrian={ped_c:.2f} car={car_c:.2f}")
    check("margins: the pedestrian margin equals the base times the group's "
          "documented scale (the taxonomy is the source, not a tuned constant)",
          abs(ped_c - base_c * GROUP_BEHAVIOUR["VULNERABLE"]["ttc_margin_scale"]) < 1e-9,
          f"{ped_c} vs {base_c * GROUP_BEHAVIOUR['VULNERABLE']['ttc_margin_scale']}")
    check("margins: a heavy vehicle gets a longer margin than a car "
          "(longer stopping distance, blocks the view)",
          truck_c > car_c, f"truck={truck_c:.2f} car={car_c:.2f}")
    check("margins: an animal is treated as VULNERABLE, like a pedestrian "
          "(cattle on the carriageway is an Indian-road norm, not an outlier)",
          abs(thresholds("cow")[0] - ped_c) < 1e-9)
    check("margins: every India-specific class reaches a group",
          all(det._group_of(FakeDet((0, 0, 10, 10), 0.9, 0, n)) is not None
              for n in ("autorickshaw", "tempo traveller", "water tanker",
                        "pushcart", "buffalo", "mini bus")),
          str({n: det._group_of(FakeDet((0, 0, 10, 10), 0.9, 0, n))
               for n in ("autorickshaw", "tempo traveller", "water tanker",
                         "pushcart", "buffalo", "mini bus")}))
    check("margins: an unrecognised class falls back to the base thresholds "
          "rather than inheriting a pedestrian's margin or a cone's",
          thresholds("flying saucer") == (base_c, base_w))
    check("margins: a detector that already emits group names is handled "
          "(a model trained on the decision taxonomy outputs these directly)",
          thresholds("VULNERABLE") == (ped_c, ped_w))
    check("margins: warning always sits further out than brake, for every group",
          all(thresholds(n)[1] > thresholds(n)[0]
              for n in ("pedestrian", "car", "truck", "autorickshaw",
                        "motorcycle", "traffic cone")))

    det.use_group_margins = False
    check("margins: the scaling can be switched off, restoring the previous "
          "uniform behaviour exactly",
          thresholds("pedestrian") == (base_c, base_w))

    det.use_group_margins = True
    det.set_night_mode(True)
    night_ped, _ = thresholds("pedestrian")
    check("margins: night mode and group scaling compose, rather than one "
          "overwriting the other",
          night_ped > ped_c, f"night={night_ped:.2f} day={ped_c:.2f}")


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

def test_curvature_metric_accuracy():
    """Does the metric curve radius actually recover a known radius?

    The overtaking rule refuses a manoeuvre when the road's radius falls below
    150 m, so the estimate is load-bearing. Until now it was never checked
    against a known answer -- only that a curved polyline scored tighter than a
    straight one, which any monotone function would satisfy.

    Here arcs of known radius are projected through the ground-plane homography
    into image pixels and recovered. The property that matters is not raw
    accuracy but the DIRECTION of the error: reporting a road straighter than
    it is permits an overtake the geometry does not support.
    """
    from models.ipm import IPMTransformer

    fx = fy = 1280 / (2 * np.tan(np.deg2rad(60) / 2))
    ipm = IPMTransformer.from_intrinsics(fx, fy, 640, 360,
                                         camera_height_m=1.5, pitch_deg=5.0)
    ground_to_image = np.linalg.inv(ipm.H)

    def arc_pixels(radius_m, n=25):
        """Circular arc of the given radius, tangent to the forward axis."""
        # An arc of radius R spans only Z <= R before it turns back, so a
        # tighter curve is visible over a shorter stretch -- as on a real
        # switchback.
        z_far = min(45.0, 0.85 * radius_m)
        Z = np.linspace(min(5.0, 0.2 * z_far), z_far, n)
        X = radius_m - np.sqrt(np.maximum(radius_m ** 2 - Z ** 2, 0.0))
        g = np.stack([X, Z, np.ones_like(Z)], axis=1) @ ground_to_image.T
        return g[:, :2] / g[:, 2:3]

    errors = {}
    for R in (25, 30, 50, 75, 100, 150, 200, 400, 1000):
        _, got = ipm.curvature_from_polyline(arc_pixels(R))
        errors[R] = None if got is None else (got - R) / R

    check("curvature: every known arc from 25 m to 1000 m is recovered",
          all(v is not None for v in errors.values()), str(errors))
    check("curvature: NO radius is over-reported by more than 5% -- an error "
          "toward 'straighter than it is' permits an unsupported overtake",
          all(v is not None and v <= 0.05 for v in errors.values()),
          str({k: f"{v*100:+.0f}%" for k, v in errors.items() if v is not None}))
    check("curvature: accurate to within 5% at and above the 150 m decision "
          "threshold, where the rule actually switches",
          all(abs(errors[R]) <= 0.05 for R in (150, 200, 400, 1000)),
          str({R: f"{errors[R]*100:+.0f}%" for R in (150, 200, 400, 1000)}))
    check("curvature: tighter arcs err toward reporting a SHARPER curve, "
          "which refuses an overtake rather than permitting one",
          errors[50] < 0 and errors[75] < 0 and errors[100] < 0,
          str({R: f"{errors[R]*100:+.0f}%" for R in (50, 75, 100)}))
    check("curvature: monotone -- a tighter true radius never reports a larger "
          "radius than a gentler one",
          all(_recover(ipm, arc_pixels, a) <= _recover(ipm, arc_pixels, b)
              for a, b in zip((50, 75, 100, 150, 200), (75, 100, 150, 200, 400))))

    straight = np.stack([np.zeros(25), np.linspace(5, 45, 25), np.ones(25)],
                        axis=1) @ ground_to_image.T
    _, r_straight = ipm.curvature_from_polyline(straight[:, :2] / straight[:, 2:3])
    check("curvature: a perfectly straight road reads as effectively infinite",
          r_straight is not None and r_straight > 10_000, str(r_straight))

    # Document the defect this replaced: evaluating curvature at the FAR end of
    # the polyline divides it by (1 + slope^2)^1.5, and the slope peaks exactly
    # there. Reintroduced locally so the test fails if the old form returns.
    def far_end_radius(polyline):
        g = ipm.pixel_to_ground(np.asarray(polyline, dtype=np.float64))
        X, Z = g[:, 0], g[:, 1]
        o = np.argsort(Z)
        Z, X = Z[o], X[o]
        a, b, _ = np.polyfit(Z, X, 2)
        z = float(Z.max())
        k = abs(2 * a) / max((1 + (2 * a * z + b) ** 2) ** 1.5, 1e-9)
        return 1.0 / k if k > 1e-9 else float("inf")

    old_150 = far_end_radius(arc_pixels(150))
    check("curvature: the previous far-end formulation over-reported the "
          "radius (documents the defect this fix removes)",
          old_150 > 150 * 1.05, f"old formulation gave {old_150:.0f} m for a 150 m arc")
    check("curvature: the current formulation does NOT share that bias",
          _recover(ipm, arc_pixels, 150) <= 150 * 1.05,
          f"{_recover(ipm, arc_pixels, 150):.0f} m for a 150 m arc")


def _recover(ipm, arc_fn, radius_m) -> float:
    _, got = ipm.curvature_from_polyline(arc_fn(radius_m))
    return float("inf") if got is None else float(got)


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

    # Case D: fast oncoming vehicle close enough to matter -> blocked.
    # Run under BOTH traffic conventions. India drives on the LEFT, so the
    # oncoming lane is to the RIGHT of the divider -- the opposite of the
    # US/Europe convention this module originally hardcoded. A vehicle placed
    # on the wrong side must NOT read as oncoming, or the system would ignore
    # genuine head-on traffic while braking for cars in its own lane.
    near_depth = np.full((480, 640), 40.0, dtype=np.float32)
    for side_name, ego_x, onc_x in (("left (India)", 150, 500),
                                    ("right (US/EU)", 500, 150)):
        az_side = OvertakingAnalyzer(
            ipm=ipm, min_gap_seconds=8.0, min_lead_gap_m=25.0,
            traffic_side="left" if "India" in side_name else "right")

        oncoming = FakeDet((onc_x, 300, onc_x + 60, 360), 0.9, 0, "car")
        oncoming.track_id = 99
        oncoming.depth_speed_mps = 25.0   # approaching fast
        status = az_side.analyze(lanes, {"center": "dashed"}, tracks=[oncoming],
                                 depth_map=near_depth, ego_speed_mps=20.0,
                                 scene_brightness=150.0, lighting_state="DAY")
        check(f"overtaking [{side_name}]: close fast oncoming vehicle across "
              f"the divider -> NOT_POSSIBLE_ONCOMING",
              status == OvertakeStatus.NOT_POSSIBLE_ONCOMING, f"got {status}")

        same_lane = FakeDet((ego_x, 300, ego_x + 60, 360), 0.9, 0, "car")
        same_lane.track_id = 98
        same_lane.depth_speed_mps = 25.0
        status = az_side.analyze(lanes, {"center": "dashed"}, tracks=[same_lane],
                                 depth_map=near_depth, ego_speed_mps=20.0,
                                 scene_brightness=150.0, lighting_state="DAY")
        check(f"overtaking [{side_name}]: a vehicle on the EGO's side of the "
              f"divider is not treated as oncoming",
              status != OvertakeStatus.NOT_POSSIBLE_ONCOMING, f"got {status}")

    check("overtaking: India default overtakes on the right",
          OvertakingAnalyzer(ipm=ipm).manoeuvre_side == "right")
    check("overtaking: an invalid traffic_side is rejected outright rather "
          "than silently defaulting to one convention",
          _raises(lambda: OvertakingAnalyzer(ipm=ipm, traffic_side="LHD")))


# --------------------------------------------------------------------------
# 6. Sanity filter: geometric size-consistency check that catches "detected
#    something where nothing real is there" (billboards/hoardings/
#    reflections a domain-shifted detector can hallucinate onto).
# --------------------------------------------------------------------------

def test_rear_view():
    """Rear-and-side assessment for the overtaking decision.

    Two things are being protected here. The first is the IRC:66 arithmetic --
    the spacing formula takes m/s, and feeding it km/h silently doubles every
    distance. The second, and the more important one, is that the module
    refuses to certify a lane clear when it cannot see far enough back to know:
    "I see nothing" is not "nothing is there".
    """
    from inference.overtaking import OvertakingAnalyzer, OvertakeStatus
    from inference.rear_view import (RearApproachMonitor, manoeuvre_time_s,
                                     overtaken_speed_default_mps,
                                     overtaking_sight_distance_m, spacing_m)
    from models.ipm import IPMTransformer

    # -- IRC:66 arithmetic ---------------------------------------------------
    # Published OSD values for two-lane highways. The implementation should
    # track them; large divergence means a units error, which is the failure
    # mode this exists to catch.
    published = {40: 165, 50: 235, 60: 300, 65: 340, 80: 470, 100: 640}
    devs = {}
    for kmph, want in published.items():
        got = overtaking_sight_distance_m(kmph / 3.6)
        devs[kmph] = (got - want) / want
    check("rear/IRC: overtaking sight distance tracks the published IRC:66 "
          "table across 40-100 km/h (a units error would show as ~2x)",
          all(abs(d) < 0.25 for d in devs.values()),
          str({k: f"{v*100:.0f}%" for k, v in devs.items()}))
    check("rear/IRC: spacing formula takes m/s, not km/h "
          "(s = 0.7*Vb + 6 at 10 m/s is 13 m, not 132 m)",
          abs(spacing_m(10.0) - 13.0) < 1e-6, str(spacing_m(10.0)))
    check("rear/IRC: the overtaken vehicle defaults to 16 km/h below ego, not "
          "to the ego's own speed (passing something equally fast is not a "
          "manoeuvre and inflates the window)",
          abs(overtaken_speed_default_mps(20.0) - (20.0 - 16 / 3.6)) < 1e-6,
          str(overtaken_speed_default_mps(20.0)))
    check("rear/IRC: sight distance grows with speed",
          all(overtaking_sight_distance_m(v / 3.6)
              < overtaking_sight_distance_m((v + 10) / 3.6)
              for v in range(30, 100, 10)))
    check("rear/IRC: manoeuvre time grows with the speed of the vehicle passed",
          manoeuvre_time_s(8.0) < manoeuvre_time_s(20.0))
    check("rear/IRC: a stationary ego still yields a positive manoeuvre time "
          "(no divide-by-zero at rest)",
          manoeuvre_time_s(0.0) > 0)

    # -- which side is the manoeuvre side ------------------------------------
    check("rear: left-hand traffic (India) overtakes on the RIGHT",
          RearApproachMonitor(traffic_side="left").manoeuvre_side == "right")
    check("rear: right-hand traffic (US/EU) overtakes on the LEFT",
          RearApproachMonitor(traffic_side="right").manoeuvre_side == "left")

    india = RearApproachMonitor(traffic_side="left")
    right_side = FakeDet((900, 300, 1000, 400), 0.9, 0, "car")
    left_side = FakeDet((100, 300, 200, 400), 0.9, 0, "car")
    check("rear: a vehicle on the right is on India's manoeuvre side",
          india.on_manoeuvre_side(right_side, frame_width=1280))
    check("rear: a vehicle on the left is NOT on India's manoeuvre side",
          not india.on_manoeuvre_side(left_side, frame_width=1280))

    # -- observability: the check most rule-based systems omit ---------------
    far = np.full((480, 1280), 80.0, dtype=np.float32)
    near = np.full((480, 1280), 12.0, dtype=np.float32)

    v = india.analyze(None, far, ego_speed_mps=20.0)
    check("rear: with no rear view configured the verdict is NOT observable "
          "(silence is not evidence of a clear lane)",
          not v.observable and not v.clear, v.reason)

    v = india.analyze([], near, ego_speed_mps=20.0)
    check("rear: a rear view too short for the manoeuvre is NOT observable, "
          "even with zero vehicles detected in it",
          not v.observable and v.required_sight_m > v.observed_sight_m,
          f"observed={v.observed_sight_m:.0f} required={v.required_sight_m:.0f}")

    v = india.analyze([], far, ego_speed_mps=20.0)
    check("rear: an empty lane within a sufficient rear view IS clear",
          v.observable and v.clear, v.reason)
    check("rear: the verdict carries the numbers behind it, so a refusal can "
          "be explained rather than just asserted",
          v.required_clear_time_s > 0 and v.observed_sight_m > 0
          and "clear" in v.describe(), v.describe())

    # -- a vehicle closing from behind ---------------------------------------
    def rear_car(x, closing_mps, tid):
        d = FakeDet((x, 300, x + 100, 400), 0.9, 0, "car")
        d.track_id = tid
        # Project convention (inference/collision.py): POSITIVE means the gap
        # is shrinking. A negative value here would read as pulling away.
        d.depth_speed_mps = closing_mps
        return d

    closer = rear_car(900, 15.0, 1)          # right side, closing fast
    v = india.analyze([closer], far, ego_speed_mps=20.0)
    check("rear: a vehicle closing fast on the manoeuvre side blocks the "
          "overtake",
          v.observable and not v.clear and v.blocking_track_id == 1,
          v.describe())
    check("rear: the block reports gap and time-to-arrival",
          v.blocking_gap_m is not None and v.blocking_tta_s is not None,
          v.describe())

    other_side = rear_car(100, 15.0, 2)      # left side: not where we are going
    v = india.analyze([other_side], far, ego_speed_mps=20.0)
    check("rear: a vehicle closing on the OPPOSITE side does not block "
          "(this is the whole point of checking the manoeuvre side)",
          v.clear, v.describe())

    keeping_pace = rear_car(900, 0.5, 3)     # right side, barely gaining
    v = india.analyze([keeping_pace], far, ego_speed_mps=20.0)
    check("rear: a vehicle behind that is merely keeping pace does not block",
          v.clear, v.describe())

    pulling_away = rear_car(900, -12.0, 4)   # right side, falling back
    v = india.analyze([pulling_away], far, ego_speed_mps=20.0)
    check("rear: a vehicle behind that is falling back does not block "
          "(sign convention: negative depth_speed_mps means the gap grows)",
          v.clear, v.describe())

    # Under the mirrored convention the same geometry must give the mirrored
    # answer -- otherwise one of the two is hardcoded.
    us = RearApproachMonitor(traffic_side="right")
    check("rear [US/EU]: the SAME closing vehicle on the right does NOT block, "
          "because the manoeuvre side is the left",
          us.analyze([rear_car(900, 15.0, 5)], far, ego_speed_mps=20.0).clear)
    check("rear [US/EU]: a closing vehicle on the left DOES block",
          not us.analyze([rear_car(100, 15.0, 6)], far, ego_speed_mps=20.0).clear)

    # -- integration with the overtaking decision ----------------------------
    src = np.array([[300, 480], [340, 480], [280, 300], [360, 300]], dtype=np.float32)
    dst = np.array([[-1.75, 5], [1.75, 5], [-1.75, 30], [1.75, 30]], dtype=np.float32)
    ipm = IPMTransformer.from_points(src, dst)

    class FakeLanes:
        def __init__(self, polylines):
            self.polylines = polylines

    straight = np.array([[320, 480 - i * 15] for i in range(12)], dtype=np.float32)
    lanes = FakeLanes([straight, straight + 60])
    depth = np.full((480, 1280), 80.0, dtype=np.float32)
    kw = dict(lanes=lanes, lane_types={"center": "dashed"}, tracks=[],
              depth_map=depth, ego_speed_mps=20.0, scene_brightness=150.0,
              lighting_state="DAY")

    az_single = OvertakingAnalyzer(ipm=ipm)
    check("overtaking: a single forward camera behaves exactly as before "
          "(the rear rule does not silently block every frame)",
          az_single.analyze(**kw) == OvertakeStatus.POSSIBLE,
          str(az_single.analyze(**kw)))
    check("overtaking: and it records that the rear was never assessed, rather "
          "than implying it was checked and found clear",
          az_single.last_rear_verdict is None)

    az_rear = OvertakingAnalyzer(ipm=ipm, require_rear_view=True)
    check("overtaking: with the rear check REQUIRED and no rear input, the "
          "verdict fails safe to NOT_POSSIBLE_REAR_UNSEEN",
          az_rear.analyze(**kw) == OvertakeStatus.NOT_POSSIBLE_REAR_UNSEEN,
          str(az_rear.analyze(**kw)))

    status = az_rear.analyze(rear_tracks=[], rear_depth_map=depth, **kw)
    check("overtaking: with a rear view that reaches far enough and nothing "
          "in it -> POSSIBLE", status == OvertakeStatus.POSSIBLE, str(status))

    status = az_rear.analyze(rear_tracks=[rear_car(900, 15.0, 7)],
                             rear_depth_map=depth, **kw)
    check("overtaking: a vehicle closing from behind on the side being turned "
          "into -> NOT_POSSIBLE_REAR_APPROACH",
          status == OvertakeStatus.NOT_POSSIBLE_REAR_APPROACH, str(status))
    check("overtaking: the rear verdict is retained for display/logging",
          az_rear.last_rear_verdict is not None
          and az_rear.last_rear_verdict.blocking_track_id == 7)

    status = az_rear.analyze(rear_tracks=[], rear_depth_map=near, **kw)
    check("overtaking: an empty but too-short rear view is refused, not "
          "treated as clear — the sensor range must cover the decision",
          status == OvertakeStatus.NOT_POSSIBLE_REAR_UNSEEN, str(status))

    # -- lead-vehicle speed sign ---------------------------------------------
    lead = FakeDet((300, 300, 400, 400), 0.9, 0, "truck")
    lead.track_id = 8
    lead.depth_speed_mps = 5.0     # we are closing on it at 5 m/s
    check("overtaking: a lead vehicle we are gaining on is computed as SLOWER "
          "than the ego, not faster (sign error here shortens the window)",
          abs(OvertakingAnalyzer._lead_speed_mps(lead, 20.0) - 15.0) < 1e-6,
          str(OvertakingAnalyzer._lead_speed_mps(lead, 20.0)))
    check("overtaking: with no lead vehicle the speed is unknown, letting the "
          "IRC default apply rather than inventing a number",
          OvertakingAnalyzer._lead_speed_mps(None, 20.0) is None)


def test_sanity_filter():
    from inference.sanity_filter import SizeConsistencyFilter
    from models.ipm import estimate_focal_length_px

    fx = estimate_focal_length_px(image_width=1280)  # ~640px for 90 deg HFOV
    check("sanity_filter: estimate_focal_length_px returns a sane positive value",
          200 < fx < 2000, f"fx={fx}")

    filt = SizeConsistencyFilter(focal_length_px=fx)
    depth_map = np.full((480, 1280), 20.0, dtype=np.float32)  # uniform 20m depth

    # Plausible real car: ~58px wide at 20m depth with fx~640 -> ~1.8m real width
    plausible_car = FakeDet((100, 300, 158, 380), 0.8, 0, "car")
    # Implausibly HUGE "car" (billboard-scale): 400px wide at the same 20m depth
    huge_car = FakeDet((300, 300, 700, 380), 0.8, 0, "car")
    # Implausibly TINY "car": 5px wide at 20m depth
    tiny_car = FakeDet((800, 300, 805, 380), 0.8, 0, "car")
    # Unbounded class ("misc") at the same huge size -- should NOT be rejected
    huge_misc = FakeDet((900, 300, 1270, 380), 0.8, 10, "misc")

    kept = filt.filter([plausible_car, huge_car, tiny_car, huge_misc], depth_map)
    kept_names = [(d.class_name, d.bbox) for d in kept]

    check("sanity_filter: plausible-size car is kept",
          plausible_car in kept, f"kept={kept_names}")
    check("sanity_filter: billboard-scale huge 'car' is rejected",
          huge_car not in kept, f"kept={kept_names}")
    check("sanity_filter: implausibly tiny 'car' is rejected",
          tiny_car not in kept, f"kept={kept_names}")
    check("sanity_filter: unbounded class ('misc') passes through regardless of size",
          huge_misc in kept, f"kept={kept_names}")
    check("sanity_filter: rejected_count tracks rejections by class",
          filt.rejected_count.get("car", 0) == 2, f"rejected_count={filt.rejected_count}")

    # Graceful degradation: no focal length / no depth map -> everything passes
    filt_noop = SizeConsistencyFilter(focal_length_px=None)
    kept_noop = filt_noop.filter([huge_car], depth_map)
    check("sanity_filter: no-ops (passes everything) when focal_length_px is None",
          huge_car in kept_noop)
    kept_noop2 = filt.filter([huge_car], None)
    check("sanity_filter: no-ops (passes everything) when depth_map is None",
          huge_car in kept_noop2)

    # REGRESSION: the first real cluster run over-rejected (3961 detections,
    # incl. 1130 "van" + 1039 "car") because with no calibration the focal
    # length is a ~90-degree-FOV guess, and guessing too WIDE an FOV inflates
    # every implied real-world width. An uncertain focal length must widen the
    # acceptance band, never tighten it.
    est = SizeConsistencyFilter(focal_length_px=fx, focal_length_is_estimated=True)
    cal = SizeConsistencyFilter(focal_length_px=fx, focal_length_is_estimated=False)
    check("sanity_filter: estimated focal length uses a WIDER margin than calibrated",
          est.effective_margin > cal.effective_margin,
          f"estimated={est.effective_margin} vs calibrated={cal.effective_margin}")

    # A real car mis-measured because the FOV guess was wrong (~1.7x too wide
    # a bbox for its depth) should survive under the estimated-fx band...
    borderline = FakeDet((100, 300, 199, 380), 0.8, 0, "car")   # ~99px -> ~3.1m implied
    check("sanity_filter: FOV-guess-inflated real car survives with estimated fx",
          borderline in est.filter([borderline], depth_map))
    # ...while a billboard-scale box stays rejected even with the wider band.
    check("sanity_filter: billboard-scale box still rejected under the wider band",
          huge_car not in est.filter([huge_car], depth_map))

    check("sanity_filter: tracks seen/checked counts for the reject ratio",
          est.seen_count > 0 and est.checked_count > 0,
          f"seen={est.seen_count} checked={est.checked_count}")


# --------------------------------------------------------------------------
# 7. Entry points actually run as scripts.
#
#    REGRESSION TEST for a real bug this suite originally MISSED: every test
#    above imports modules with PROJECT_ROOT already on sys.path (this file
#    inserts it at the top), so `from models...` always resolved here — but
#    `python inference/adas_final.py ...` puts inference/ on sys.path instead
#    of the project root, and crashed instantly with ModuleNotFoundError on
#    a real HPC run. Testing imports is NOT the same as testing the command
#    people actually type, so this launches each entry point as a real
#    subprocess the way a user would.
# --------------------------------------------------------------------------

def test_entry_points_run_as_scripts():
    import subprocess

    # (path, args) — args chosen to exit fast without needing weights/data
    entry_points = [
        ("inference/adas_final.py", ["--help"]),
        ("training/train_drivable_area.py", ["--help"]),
        ("training/train_aux_classifiers.py", ["--help"]),
        ("models/ipm.py", []),
        ("inference/tracker.py", []),
        ("data/prepare_idd.py", ["--help"]),
        ("data/prepare_driveindia.py", ["--help"]),
        ("data/prepare_uvh26.py", ["--help"]),
        ("data/prepare_merged.py", ["--help"]),
    ]

    for rel_path, args in entry_points:
        script = PROJECT_ROOT / rel_path
        if not script.exists():
            check(f"entry point: {rel_path} exists", False, "file missing")
            continue
        proc = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120)
        combined = proc.stdout + proc.stderr
        # We only care that it doesn't die on an import-resolution error.
        # Non-zero exit from e.g. a missing dataset path is fine here.
        broke_on_import = ("ModuleNotFoundError" in combined
                           or "ImportError" in combined)
        check(f"entry point runs as script: python {rel_path}",
              not broke_on_import,
              combined.strip().splitlines()[-1] if broke_on_import else "")


# --------------------------------------------------------------------------
# 8. Optional modules degrade gracefully instead of killing the pipeline.
#
#    REGRESSION TEST for a real HPC failure: CLRNet raised AttributeError
#    (mmcv 2.x dropped mmcv.jit), load_lane_detector only caught
#    (ImportError, FileNotFoundError), so the error escaped and took the
#    entire pipeline down — even though lane detection is explicitly
#    optional and has a drivable-area fallback.
# --------------------------------------------------------------------------

def test_optional_module_degradation():
    import models.lane_detector as ld

    original = ld.CLRNetWrapper
    try:
        # Simulate the exact cluster failure: a non-Import/FileNotFound error
        # raised while constructing the third-party lane model.
        class _Boom:
            def __init__(self, *a, **k):
                raise AttributeError("module 'mmcv' has no attribute 'jit'")

        ld.CLRNetWrapper = _Boom
        result = ld.load_lane_detector("clrnet")
        check("degradation: load_lane_detector returns None (not raises) when "
              "CLRNet blows up with a non-import error",
              result is None, f"got {result!r}")
    except Exception as e:
        check("degradation: load_lane_detector returns None (not raises) when "
              "CLRNet blows up with a non-import error",
              False, f"it RAISED instead: {type(e).__name__}: {e}")
    finally:
        ld.CLRNetWrapper = original

    # models/mmcv_compat.py (Krish's fix) restores the mmcv 1.x symbols CLRNet
    # needs under mmcv 2.x. Verify it exposes them and that its jit shim is a
    # faithful no-op in BOTH decorator forms — CLRNet uses `@mmcv.jit(...)`,
    # which is the form that breaks if a shim only handles the bare version.
    import types
    fake_mmcv = types.ModuleType("mmcv")
    saved_mmcv = sys.modules.get("mmcv")
    saved_compat = sys.modules.pop("models.mmcv_compat", None)
    sys.modules["mmcv"] = fake_mmcv
    try:
        import importlib
        importlib.import_module("models.mmcv_compat")   # applies patches on import

        for attr in ("jit", "load", "dump", "runner", "parallel"):
            check(f"mmcv_compat: restores mmcv.{attr}", hasattr(fake_mmcv, attr))

        # mmcv.runner fp16 decorators — the SECOND cluster failure, after the
        # jit fix: "cannot import name 'auto_fp16' from 'mmcv.runner'".
        # CLRNet decorates forward() with @auto_fp16() / @force_fp32().
        for attr in ("auto_fp16", "force_fp32", "load_checkpoint"):
            check(f"mmcv_compat: mmcv.runner exposes {attr}",
                  hasattr(fake_mmcv.runner, attr))

        @fake_mmcv.runner.auto_fp16(apply_to=("x",))   # CLRNet's actual usage
        def h(x):
            return x * 3
        check("mmcv_compat: @auto_fp16(...) leaves the function working",
              h(14) == 42, f"got {h(14)}")

        @fake_mmcv.runner.auto_fp16                     # bare form
        def k(x):
            return x - 1
        check("mmcv_compat: bare @auto_fp16 leaves the function working",
              k(43) == 42, f"got {k(43)}")

        @fake_mmcv.jit(coderize=True)          # CLRNet's actual usage
        def f(x):
            return x * 2
        check("mmcv_compat: @mmcv.jit(...) leaves the function working",
              f(21) == 42, f"got {f(21)}")

        @fake_mmcv.jit                          # bare-decorator usage
        def g(x):
            return x + 1
        check("mmcv_compat: bare @mmcv.jit leaves the function working",
              g(41) == 42, f"got {g(41)}")
    except ImportError as e:
        check("mmcv_compat: module importable", False, str(e))
    finally:
        sys.modules.pop("models.mmcv_compat", None)
        if saved_compat is not None:
            sys.modules["models.mmcv_compat"] = saved_compat
        if saved_mmcv is not None:
            sys.modules["mmcv"] = saved_mmcv
        else:
            sys.modules.pop("mmcv", None)


# --------------------------------------------------------------------------
# 9. Heuristic lane-type classifier — solid vs dashed with no trained weights.
#
#    Without ANY lane-type source, overtaking.py's legality rule fails safe on
#    every frame, so the verdict is a constant "NOT POSSIBLE - SOLID CENTER
#    LINE" (observed 3604/3604 on the cluster) and the feature never engages.
# --------------------------------------------------------------------------

def test_heuristic_lane_type():
    import cv2
    from models.aux_classifiers import HeuristicLaneTypeClassifier

    h, w = 400, 200
    clf = HeuristicLaneTypeClassifier()

    def frame_with_line(dashed: bool):
        img = np.full((h, w, 3), 60, dtype=np.uint8)     # dark asphalt
        for y in range(20, h - 20):
            if dashed and (y // 20) % 2 == 1:            # periodic gaps
                continue
            cv2.line(img, (100, y), (100, y + 1), (235, 235, 235), 7)
        return img

    polyline = np.array([[100, y] for y in range(25, h - 25, 6)], dtype=np.float32)

    solid_type = clf.classify_line(frame_with_line(dashed=False), polyline)
    dashed_type = clf.classify_line(frame_with_line(dashed=True), polyline)

    check("lane-type heuristic: continuous paint -> 'solid'",
          solid_type == "solid", f"got {solid_type!r}")
    check("lane-type heuristic: periodic gaps -> 'dashed'",
          dashed_type == "dashed", f"got {dashed_type!r}")

    # No paint at all (unmarked road) must stay 'unknown' so overtaking.py
    # keeps failing safe rather than being told the road is clear.
    blank = np.full((h, w, 3), 60, dtype=np.uint8)
    check("lane-type heuristic: unmarked road stays 'unknown' (fail-safe)",
          clf.classify_line(blank, polyline) == "unknown",
          f"got {clf.classify_line(blank, polyline)!r}")

    # classify() must return the same shape the pipeline/overtaking expect.
    class FakeLanes:
        polylines = [polyline, polyline + np.array([40, 0], dtype=np.float32)]
    types = clf.classify(frame_with_line(dashed=True), FakeLanes())
    check("lane-type heuristic: classify() provides the 'center' key "
          "overtaking.py reads", "center" in types, f"got keys {list(types)}")

    # End-to-end: a dashed centre line must let overtaking reach POSSIBLE,
    # which is the whole point of adding this fallback.
    from inference.overtaking import OvertakingAnalyzer, OvertakeStatus
    from models.ipm import IPMTransformer
    src = np.array([[300, 480], [340, 480], [280, 300], [360, 300]], dtype=np.float32)
    dst = np.array([[-1.75, 5], [1.75, 5], [-1.75, 30], [1.75, 30]], dtype=np.float32)
    az = OvertakingAnalyzer(ipm=IPMTransformer.from_points(src, dst))
    straight = np.array([[320, 480 - i * 15] for i in range(12)], dtype=np.float32)

    class Lanes2:
        polylines = [straight, straight + np.array([60, 0], dtype=np.float32)]

    status = az.analyze(Lanes2(), {"center": "dashed"}, tracks=[],
                        depth_map=np.full((480, 640), 60.0, dtype=np.float32),
                        ego_speed_mps=20.0, scene_brightness=150.0,
                        lighting_state="DAY")
    check("lane-type heuristic: a 'dashed' verdict unblocks overtaking "
          "(no longer a constant NOT_POSSIBLE)",
          status == OvertakeStatus.POSSIBLE, f"got {status}")


# --------------------------------------------------------------------------
# 11. Decision-grouped taxonomy and the safety-weighted misclassification cost.
#
#     Classes are grouped by what the decision layer must do differently, not
#     by visual similarity. The cost of a misclassification is deliberately
#     ASYMMETRIC: predicting something that demands less caution than the truth
#     is a hazard, while predicting something more cautious is only a nuisance.
# --------------------------------------------------------------------------

def test_taxonomy():
    from models.taxonomy import (DecisionTaxonomy, DECISION_CLASSES,
                                 misclassification_cost, miss_cost,
                                 cost_matrix,
                                 GROUP_BEHAVIOUR)

    tx = DecisionTaxonomy()

    # Name normalisation: separators and case must not matter, or a dataset
    # writing "Auto-Rickshaw" silently loses every one of those labels.
    for variant in ("autorickshaw", "Auto-Rickshaw", "AUTO RICKSHAW",
                    "auto_rickshaw", "three wheeler"):
        got = tx.map_name(variant)
        check(f"taxonomy: {variant!r} maps to THREE_WHEELER",
              got == "THREE_WHEELER", f"got {got!r}")

    # The confusion mode actually observed on the cluster (truck/bus/tanker/
    # tractor/LCV competing) must collapse into one class.
    heavy = ["truck", "bus", "tanker", "tractor", "LCV", "mini bus",
             "construction vehicle", "trailer"]
    mapped = {tx.map_name(n) for n in heavy}
    check("taxonomy: all heavy-vehicle variants collapse to one class "
          "(removes the observed truck/bus/tanker confusion by construction)",
          mapped == {"HEAVY_VEHICLE"}, f"got {mapped}")

    check("taxonomy: animals are VULNERABLE, not obstacles",
          tx.map_name("cow") == "VULNERABLE", tx.map_name("cow"))
    check("taxonomy: a cart is a STATIC_OBSTACLE, not a vehicle",
          tx.map_name("pushcart") == "STATIC_OBSTACLE", tx.map_name("pushcart"))

    # Unknown names must return None, never a guess -- silently bucketing an
    # unrecognised class would corrupt the labels with no way to notice.
    check("taxonomy: unknown class returns None rather than guessing",
          tx.map_name("flying saucer") is None)
    check("taxonomy: unknown names are recorded for reporting",
          "flying saucer" in tx.unmapped)

    # --- the asymmetry, which is the whole point of the metric ---
    danger = misclassification_cost("VULNERABLE", "STATIC_OBSTACLE")
    nuisance = misclassification_cost("STATIC_OBSTACLE", "VULNERABLE")
    check("SWMC: calling a person a barrier is near the top of the scale",
          0.8 <= danger < 1.0, f"got {danger}")
    check("SWMC: the maximum cost (1.0) is a vulnerable road user going "
          "entirely UNDETECTED -- strictly worse than any mislabelling of one",
          abs(miss_cost("VULNERABLE") - 1.0) < 1e-6
          and miss_cost("VULNERABLE") > danger,
          f"miss={miss_cost('VULNERABLE')} vs mislabel={danger}")
    check("SWMC: calling a barrier a person is cheap (over-caution)",
          nuisance <= 0.15, f"got {nuisance}")
    check("SWMC: the cost is ASYMMETRIC (this is what a plain confusion "
          "matrix cannot express)",
          danger > nuisance * 5, f"{danger} vs {nuisance}")

    check("SWMC: same-group confusion is free (bus vs truck)",
          misclassification_cost("HEAVY_VEHICLE", "HEAVY_VEHICLE") == 0.0)

    # Falling further down the caution ordering must cost more.
    near = misclassification_cost("VULNERABLE", "TWO_WHEELER")
    far = misclassification_cost("VULNERABLE", "STATIC_OBSTACLE")
    check("SWMC: cost grows with how far the prediction falls in caution",
          far > near, f"far={far} near={near}")

    m = cost_matrix()
    check("SWMC: matrix is square over the decision classes",
          len(m) == len(DECISION_CLASSES) and len(m[0]) == len(DECISION_CLASSES))
    check("SWMC: diagonal is zero", all(m[i][i] == 0.0 for i in range(len(m))))

    # Behaviour parameters must exist for every class, or a downstream module
    # will KeyError at inference time on whichever class was forgotten.
    missing = [c for c in DECISION_CLASSES if c not in GROUP_BEHAVIOUR]
    check("taxonomy: every class has decision parameters", not missing, str(missing))
    check("taxonomy: vulnerable road users get the largest TTC margin",
          GROUP_BEHAVIOUR["VULNERABLE"]["ttc_margin_scale"] ==
          max(b["ttc_margin_scale"] for b in GROUP_BEHAVIOUR.values()))
    check("taxonomy: only heavy vehicles are flagged as view-blocking",
          [c for c in DECISION_CLASSES if GROUP_BEHAVIOUR[c]["blocks_view"]]
          == ["HEAVY_VEHICLE"])
    check("taxonomy: only static obstacles are marked as unable to move",
          [c for c in DECISION_CLASSES if not GROUP_BEHAVIOUR[c]["can_move"]]
          == ["STATIC_OBSTACLE"])

    # id remap from a source dataset's own name list
    src = {0: "car", 1: "truck", 2: "autorickshaw", 3: "person", 4: "unknown thing"}
    remap = tx.map_id_from_names(src)
    check("taxonomy: id remap covers known classes and drops unknown ones",
          set(remap) == {0, 1, 2, 3}, f"got {remap}")
    check("taxonomy: remapped ids point at the right groups",
          remap[2] == DECISION_CLASSES.index("THREE_WHEELER")
          and remap[3] == DECISION_CLASSES.index("VULNERABLE"), str(remap))


# --------------------------------------------------------------------------
# 10. CLRNet coordinate-scaling regression test (no GPU/weights needed —
#     exercises the exact arithmetic that was buggy)
# --------------------------------------------------------------------------

def test_swmc_and_granularity():
    """SWMC scoring, and the experiment-validity guarantees around it.

    The checks that matter most here are not the arithmetic ones -- they are
    the two that protect the granularity experiment from producing a number
    that looks like a result but is an artefact:
      * the two groupings must cover the same source vocabulary, and
      * the three prepared datasets must contain identical boxes.
    Without those, a taxonomy that merely *drops more hard objects* would post
    the best score.
    """
    import subprocess
    import tempfile

    from models.taxonomy import (CAUTION_RANK, DECISION_CLASSES,
                                 IDD_LEVEL3_GROUPS, NAME_TO_GROUP,
                                 false_positive_cost, misclassification_cost,
                                 miss_cost)
    from evaluation.evaluate_swmc import (UNMAPPED, SWMCAccumulator,
                                          label_path_for,
                                          match_class_agnostic,
                                          split_weights_data)

    I = {c: i for i, c in enumerate(DECISION_CLASSES)}
    BOX = [10.0, 10.0, 60.0, 120.0]
    BOX2 = [200.0, 10.0, 250.0, 120.0]

    # -- vocabulary parity: the guard that keeps the comparison honest -------
    check("swmc: decision and semantic groupings cover the same vocabulary "
          "(else the three datasets differ in content, not just labels)",
          set(NAME_TO_GROUP) == set(IDD_LEVEL3_GROUPS),
          f"decision-only={sorted(set(NAME_TO_GROUP) - set(IDD_LEVEL3_GROUPS))[:5]} "
          f"semantic-only={sorted(set(IDD_LEVEL3_GROUPS) - set(NAME_TO_GROUP))[:5]}")

    # -- miss / phantom costs -----------------------------------------------
    check("swmc: missing a pedestrian costs more than missing a static obstacle",
          miss_cost("VULNERABLE") > miss_cost("STATIC_OBSTACLE"),
          f"{miss_cost('VULNERABLE')} vs {miss_cost('STATIC_OBSTACLE')}")
    check("swmc: a miss is never cheaper than any misclassification of the "
          "same object (the object is absent from the world model entirely)",
          all(miss_cost(t) >= misclassification_cost(t, p) - 1e-9
              for t in DECISION_CLASSES for p in DECISION_CLASSES),
          str([(t, p, miss_cost(t), misclassification_cost(t, p))
               for t in DECISION_CLASSES for p in DECISION_CLASSES
               if miss_cost(t) < misclassification_cost(t, p) - 1e-9][:3]))
    check("swmc: a phantom costs strictly less than a miss of the same group "
          "(over-caution is a nuisance, absence is a hazard)",
          all(false_positive_cost(g) < miss_cost(g) for g in DECISION_CLASSES))
    check("swmc: phantom cost is non-zero (false alarms erode trust in alerts)",
          all(false_positive_cost(g) > 0 for g in DECISION_CLASSES))
    check("swmc: miss/phantom costs are monotone in caution rank",
          all(miss_cost(a) > miss_cost(b)
              for a in DECISION_CLASSES for b in DECISION_CLASSES
              if CAUTION_RANK[a] > CAUTION_RANK[b]))

    # -- accumulator accounting ---------------------------------------------
    acc = SWMCAccumulator()
    acc.add_frame([I["VULNERABLE"]], [BOX], [I["VULNERABLE"]], [BOX], [0.9])
    acc.add_frame([I["VULNERABLE"]], [BOX], [I["STATIC_OBSTACLE"]], [BOX], [0.9])
    acc.add_frame([I["HEAVY_VEHICLE"]], [BOX], [I["HEAVY_VEHICLE"]], [BOX], [0.9])
    acc.add_frame([I["VULNERABLE"]], [BOX], [], [], [])
    acc.add_frame([], [], [I["LIGHT_VEHICLE"]], [BOX], [0.9])
    s = acc.summary()

    check("swmc: ground-truth count ignores phantom-only frames",
          s["n_ground_truth"] == 4, str(s["n_ground_truth"]))
    check("swmc: correct prediction adds zero cost",
          abs(acc.cost_cls - misclassification_cost("VULNERABLE",
                                                    "STATIC_OBSTACLE")) < 1e-9,
          f"cost_cls={acc.cost_cls}")
    check("swmc: missed pedestrian charged at miss_cost",
          abs(acc.cost_miss - miss_cost("VULNERABLE")) < 1e-9,
          f"cost_miss={acc.cost_miss}")
    check("swmc: phantom car charged at false_positive_cost",
          abs(acc.cost_fp - false_positive_cost("LIGHT_VEHICLE")) < 1e-9,
          f"cost_fp={acc.cost_fp}")
    check("swmc: total is normalised per ground-truth object",
          abs(s["swmc"] - round(acc.total_cost / 4, 4)) < 1e-9, str(s["swmc"]))
    check("swmc: critical-error rate counts the miss and the less-cautious "
          "call, but not the benign same-group one",
          abs(s["critical_error_rate"] - 0.5) < 1e-9,
          str(s["critical_error_rate"]))
    check("swmc: bus-called-truck (same decision group) is NOT a critical error",
          s["per_group"]["HEAVY_VEHICLE"]["recall"] == 1.0)

    # -- the asymmetry, which is the whole point ----------------------------
    a, b = SWMCAccumulator(), SWMCAccumulator()
    a.add_frame([I["VULNERABLE"]], [BOX], [I["STATIC_OBSTACLE"]], [BOX], [0.9])
    b.add_frame([I["STATIC_OBSTACLE"]], [BOX], [I["VULNERABLE"]], [BOX], [0.9])
    check("swmc: pedestrian-called-obstacle costs far more than the reverse "
          "(a symmetric metric cannot express this)",
          a.summary()["swmc"] >= 5 * b.summary()["swmc"],
          f"{a.summary()['swmc']} vs {b.summary()['swmc']}")

    # -- matching semantics --------------------------------------------------
    m, mg, mp = match_class_agnostic([BOX], [BOX], [0.9])
    check("swmc: a correctly-located but mislabelled box is ONE match, not a "
          "miss plus a phantom (else the cost asymmetry never fires)",
          (len(m), len(mg), len(mp)) == (1, 0, 0), f"{len(m)},{len(mg)},{len(mp)}")
    m, mg, mp = match_class_agnostic([BOX, BOX2], [BOX], [0.9])
    check("swmc: two objects with one detection -> one match, one miss",
          (len(m), len(mg), len(mp)) == (1, 1, 0), f"{len(m)},{len(mg)},{len(mp)}")
    m, mg, mp = match_class_agnostic([BOX], [BOX, BOX2], [0.9, 0.8])
    check("swmc: one object with two detections -> one match, one phantom",
          (len(m), len(mg), len(mp)) == (1, 0, 1), f"{len(m)},{len(mg)},{len(mp)}")
    m, _, _ = match_class_agnostic([BOX], [BOX2], [0.9])
    check("swmc: a box far from any object never matches",
          len(m) == 0)

    u = SWMCAccumulator()
    u.add_frame([UNMAPPED], [BOX], [UNMAPPED], [BOX], [0.9])
    check("swmc: unmapped classes are skipped and counted, never scored",
          u.summary()["n_ground_truth"] == 0
          and u.summary()["unmapped_gt_boxes"] == 1
          and u.summary()["unmapped_pred_boxes"] == 1)

    # -- path plumbing -------------------------------------------------------
    check("swmc: images->labels rewrite touches only the last path component "
          "(a dataset root named *_images must survive)",
          label_path_for(Path("d/indian_images/val/images/a.jpg"))
          == Path("d/indian_images/val/labels/a.txt"),
          str(label_path_for(Path("d/indian_images/val/images/a.jpg"))))
    check("swmc: --compare spec splits correctly with Windows drive letters",
          split_weights_data(r"C:\runs\best.pt:C:\data\decision.yaml")
          == (r"C:\runs\best.pt", r"C:\data\decision.yaml"),
          str(split_weights_data(r"C:\runs\best.pt:C:\data\decision.yaml")))
    check("swmc: --compare spec splits correctly with absolute POSIX paths",
          split_weights_data("/h/u/best.pt:/h/u/d.yaml")
          == ("/h/u/best.pt", "/h/u/d.yaml"),
          str(split_weights_data("/h/u/best.pt:/h/u/d.yaml")))

    # -- the experiment-validity test ---------------------------------------
    # Build a miniature dataset and run the real remapper at all three levels.
    # If they disagree on box count, the granularity comparison is measuring a
    # difference in DATA, and any conclusion drawn from it is worthless.
    names = ["car", "truck", "bus", "tanker", "autorickshaw", "motorcycle",
             "person", "cow", "traffic cone", "unmappable widget"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir(parents=True)
        (src / "data.yaml").write_text(
            "train: train/images\nval: val/images\nnames:\n"
            + "\n".join(f"  {i}: {n}" for i, n in enumerate(names)),
            encoding="utf-8")
        for split in ("train", "val"):
            (src / split / "images").mkdir(parents=True)
            (src / split / "labels").mkdir(parents=True)
            for k in range(4):
                (src / split / "images" / f"{k:03d}.jpg").write_bytes(b"\xff\xd8\xff")
                rows = [f"{(k * 3 + j) % len(names)} 0.5 0.5 0.2 0.3"
                        for j in range(3)]
                (src / split / "labels" / f"{k:03d}.txt").write_text(
                    "\n".join(rows), encoding="utf-8")

        counts, class_counts, ok = {}, {}, True
        for level in ("fine", "semantic", "decision"):
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "data" / "prepare_taxonomy.py"),
                 "--src", str(src), "--level", level, "--out", str(tmp / level)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=180)
            if r.returncode != 0:
                ok = False
                counts[level] = f"crashed: {r.stderr.strip()[-200:]}"
                continue
            total = 0
            for split in ("train", "val"):
                for f in (tmp / level / split / "labels").glob("*.txt"):
                    total += len(f.read_text(encoding="utf-8").strip().splitlines())
            counts[level] = total
            for line in r.stdout.splitlines():
                if line.startswith("source classes"):
                    class_counts[level] = line.split("->")[-1].strip()

        check("granularity: all three levels build without error", ok, str(counts))
        check("granularity: the three prepared datasets contain IDENTICAL box "
              "counts -- same boxes, only the labels differ",
              len(set(counts.values())) == 1, str(counts))
        n = {k: int(v.rsplit(":", 1)[1]) for k, v in class_counts.items()}
        check("granularity: each level really does collapse the label space "
              "(fine > semantic > decision in class count)",
              len(n) == 3 and n["fine"] > n["semantic"] > n["decision"] == 6,
              str(n))
        check("granularity: the unmappable class was dropped, not silently "
              "bucketed into a real group",
              all(isinstance(v, int) and v < 24 for v in counts.values()),
              str(counts))


def test_clrnet_coord_scaling():
    # Calls the REAL remap_to_frame. The previous version of this test
    # reimplemented the arithmetic locally and asserted against its own copy —
    # which is why it passed green while the actual wrapper stayed broken on
    # the cluster. Test the shipped function, not a paraphrase of it.
    from models.lane_detector import CLRNetWrapper
    remap = CLRNetWrapper.remap_to_frame

    ORI_W, ORI_H, CUT = 1640, 590, 270      # CULane space CLRNet reports in
    FW, FH = 848, 480                        # the actual cluster video

    # A lane as CLRNet actually returns it: CULane pixels, y within the crop.
    culane_pts = np.array([[820.0, 590.0], [820.0, 430.0], [820.0, 270.0],
                           [1640.0, 590.0], [0.0, 270.0]])
    out = remap(culane_pts, FW, FH, ORI_W, ORI_H, CUT)

    check("clrnet remap: CULane-space points land INSIDE the video frame "
          "(this is the bug that made lane-type 'unknown' on 3604/3604 frames)",
          out[:, 0].max() <= FW and out[:, 1].max() <= FH,
          f"max x={out[:, 0].max():.1f} (frame {FW}), max y={out[:, 1].max():.1f} (frame {FH})")

    # Horizontal centre must stay the horizontal centre.
    check("clrnet remap: x maps proportionally (CULane centre -> frame centre)",
          abs(out[0, 0] - FW / 2) < 1.0, f"got x={out[0, 0]:.2f}, expected {FW/2}")

    # y must map THROUGH the crop, not the full height: the cut row stays put
    # and the bottom row stays the bottom row.
    check("clrnet remap: y at the cut line is unchanged",
          abs(out[2, 1] - CUT) < 1.0, f"got y={out[2, 1]:.2f}, expected {CUT}")
    check("clrnet remap: y at the image bottom maps to the frame bottom",
          abs(out[0, 1] - FH) < 1.0, f"got y={out[0, 1]:.2f}, expected {FH}")
    check("clrnet remap: y midway through the crop stays midway",
          abs(out[1, 1] - (CUT + (FH - CUT) / 2)) < 1.0,
          f"got y={out[1, 1]:.2f}, expected {CUT + (FH - CUT) / 2}")

    # Documents the old failure: unscaled CULane coords are far outside frame.
    check("clrnet remap: WITHOUT remapping the raw coords overflow the frame "
          "(documents the original defect)",
          culane_pts[:, 0].max() > FW * 1.5)

    # Normalized output (other CLRNet configs/versions) must still work.
    norm_pts = np.array([[0.5, 0.0], [0.5, 1.0]])
    out_norm = remap(norm_pts, FW, FH, ORI_W, ORI_H, CUT)
    check("clrnet remap: normalized coords still map into the frame",
          abs(out_norm[0, 0] - FW / 2) < 1.0
          and abs(out_norm[0, 1] - CUT) < 1.0
          and abs(out_norm[1, 1] - FH) < 1.0,
          f"got {out_norm.tolist()}")


def main():
    print("=" * 70)
    print("ADAS PIPELINE LOGIC SMOKE TEST (no GPU / weights required)")
    print("=" * 70)

    for name, fn in [
        ("Tracker", test_tracker),
        ("Collision / TTC", test_collision),
        ("Collision survives tracker ID churn", test_id_churn_resilience),
        ("Decision-group scaled collision margins",
         test_group_scaled_margins),
        ("IPM ground-plane curvature", test_ipm),
        ("Metric curve radius vs known-radius arcs",
         test_curvature_metric_accuracy),
        ("Drivable-area fallback (unmarked-road lane substitute)", test_drivable_area),
        ("Overtaking decision rules", test_overtaking),
        ("Rear-view overtaking (IRC:66 window + sensor-range honesty)",
         test_rear_view),
        ("Sanity filter (phantom-detection geometric check)", test_sanity_filter),
        ("Entry points run as real scripts (import-path regression)",
         test_entry_points_run_as_scripts),
        ("Optional-module graceful degradation + mmcv shim",
         test_optional_module_degradation),
        ("Heuristic lane-type classifier (no-weights fallback)",
         test_heuristic_lane_type),
        ("Decision taxonomy + safety-weighted cost", test_taxonomy),
        ("SWMC metric + granularity experiment validity",
         test_swmc_and_granularity),
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
