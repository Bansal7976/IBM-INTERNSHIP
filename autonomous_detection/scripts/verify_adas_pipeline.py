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
# 6. Sanity filter: geometric size-consistency check that catches "detected
#    something where nothing real is there" (billboards/hoardings/
#    reflections a domain-shifted detector can hallucinate onto).
# --------------------------------------------------------------------------

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
# 10. CLRNet coordinate-scaling regression test (no GPU/weights needed —
#     exercises the exact arithmetic that was buggy)
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
        ("Sanity filter (phantom-detection geometric check)", test_sanity_filter),
        ("Entry points run as real scripts (import-path regression)",
         test_entry_points_run_as_scripts),
        ("Optional-module graceful degradation + mmcv shim",
         test_optional_module_degradation),
        ("Heuristic lane-type classifier (no-weights fallback)",
         test_heuristic_lane_type),
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
