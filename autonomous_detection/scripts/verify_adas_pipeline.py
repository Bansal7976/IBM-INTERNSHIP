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

def test_planning():
    """Reference path, corridor and Frenet planning.

    Three defects are pinned here, all found by testing against known geometry
    and all wrong in the unsafe direction:

      * a corridor clipped by the edge of the observed area produced a
        centreline that bent back INTO the grid -- steering the vehicle off a
        road that was actually leaving the field of view;
      * an obstacle shorter than the station spacing constrained NOTHING,
        so a pedestrian could be invisible to the corridor builder;
      * only static objects produced hard bounds, so a traffic cone diverted
        the vehicle while a pedestrian did not.
    """
    from planning.corridor import Obstacle, build_corridor
    from planning.frenet import FrenetPlanner, PlannerConfig, quartic, quintic
    from planning.reference_path import (GroundGrid,
                                         reference_path_from_free_space)

    RES = 0.2

    def road(centre_fn, half_width_fn, n_fwd=250, n_lat=200):
        g = np.zeros((n_fwd, n_lat), dtype=bool)
        for r in range(n_fwd):
            f = r * RES
            c = n_lat // 2 + centre_fn(f) / RES
            hw = half_width_fn(f) / RES
            lo, hi = int(round(c - hw)), int(round(c + hw))
            g[r, max(0, lo):min(n_lat, hi)] = True
        return GroundGrid(g, resolution_m=RES, ego_row=0, ego_col=n_lat // 2)

    straight = road(lambda f: 0.0, lambda f: 4.0)
    path = reference_path_from_free_space(straight)

    # -- reference path ----------------------------------------------------
    check("planner: a straight corridor yields a straight reference path",
          path is not None and np.abs(path.x).max() < 0.3,
          f"max lateral {np.abs(path.x).max():.2f} m" if path else "no path")
    check("planner: the reference sits at the centre of the drivable width",
          abs(path.half_width.mean() - 4.0) < 0.2, f"{path.half_width.mean():.2f}")

    def arc(R):
        return road(lambda f: R - np.sqrt(max(R * R - f * f, 0.0)) if f < R else R,
                    lambda f: 3.0, n_lat=400)

    radii = {}
    for R in (50, 100, 150, 400):
        p = reference_path_from_free_space(arc(float(R)))
        radii[R] = p.min_radius_m() if p else None
    check("planner: curved corridors recover their true radius within 10%",
          all(v is not None and abs(v - R) / R < 0.10 for R, v in radii.items()),
          str({R: f"{v:.0f}" for R, v in radii.items() if v}))
    check("planner: recovered radius never OVER-reports (a road read straighter "
          "than it is invites a manoeuvre the geometry does not support)",
          all(v <= R * 1.05 for R, v in radii.items() if v),
          str({R: f"{v:.0f}" for R, v in radii.items() if v}))

    # A road leaving the field of view must truncate, not bend back inward.
    narrow_view = road(lambda f: (100.0 - np.sqrt(max(10000.0 - f * f, 0.0))
                                  if f < 100 else 100.0),
                       lambda f: 3.0, n_lat=100)
    p_edge = reference_path_from_free_space(narrow_view)
    check("planner: a road running out of the observed area truncates the path "
          "rather than curving it back inside",
          p_edge is not None and "field of view" in p_edge.truncated_reason,
          p_edge.truncated_reason if p_edge else "no path")

    ends = road(lambda f: 0.0, lambda f: 4.0 if f < 20.0 else 0.0)
    p_end = reference_path_from_free_space(ends)
    check("planner: the path stops where the road stops, and says so",
          p_end is not None and 19.0 < p_end.observed_length_m < 21.0
          and p_end.truncated_reason != "",
          f"{p_end.observed_length_m:.1f} m — {p_end.truncated_reason}" if p_end else "none")

    # -- Frenet projection -------------------------------------------------
    s, d = path.to_frenet([0.0, 2.0, -2.0], [20.0, 20.0, 20.0])
    check("planner: Frenet projection puts an on-path point at d = 0",
          abs(d[0]) < 0.3, f"{d[0]:.2f}")
    check("planner: d is positive to the right of travel, negative to the left",
          d[1] > 1.5 and d[2] < -1.5, f"right {d[1]:.2f}, left {d[2]:.2f}")
    check("planner: s tracks forward distance along the path",
          abs(s[0] - 20.0) < 1.0, f"{s[0]:.2f}")

    # -- corridor: the taxonomy must set the berth -------------------------
    def width_beside(group, x=1.5, y=20.0, **kw):
        ob = Obstacle(x=x, y=y, half_width=0.3, half_length=0.3,
                      group=group, track_id=1, **kw)
        c = build_corridor(path, [ob])
        return c, c.width()[np.argmin(np.abs(c.s - y))]

    widths = {g: width_beside(g)[1] for g in
              ("STATIC_OBSTACLE", "LIGHT_VEHICLE", "HEAVY_VEHICLE",
               "TWO_WHEELER", "VULNERABLE")}
    check("planner: a pedestrian narrows the corridor MORE than a traffic cone "
          "does, because the taxonomy says so",
          widths["VULNERABLE"] < widths["STATIC_OBSTACLE"],
          str({k: round(v, 2) for k, v in widths.items()}))
    check("planner: corridor width falls monotonically with the group's "
          "lateral clearance",
          widths["STATIC_OBSTACLE"] >= widths["HEAVY_VEHICLE"]
          >= widths["TWO_WHEELER"] >= widths["VULNERABLE"],
          str({k: round(v, 2) for k, v in widths.items()}))

    clear_width = build_corridor(path, []).width().mean()
    check("planner: EVERY group narrows the corridor — a mover must produce a "
          "hard bound too, or the plan drives through a standing pedestrian",
          all(w < clear_width - 0.1 for w in widths.values()),
          f"clear {clear_width:.2f} vs {({k: round(v,2) for k,v in widths.items()})}")

    # An obstacle shorter than the station spacing must never slip between them.
    missed = 0
    for y in np.arange(10.0, 35.0, 0.17):
        c = build_corridor(path, [Obstacle(x=0.0, y=float(y), half_width=0.25,
                                           half_length=0.25, group="VULNERABLE",
                                           track_id=9)])
        if c.width()[np.argmin(np.abs(c.s - y))] >= clear_width - 0.05:
            missed += 1
    check("planner: a small obstacle is never missed between corridor stations",
          missed == 0, f"{missed} of {len(np.arange(10.0, 35.0, 0.17))} missed")

    # Predicted motion is a forecast: it should raise cost, not forbid.
    c_still, w_still = width_beside("VULNERABLE", x=3.0, y=25.0)
    c_step, w_step = width_beside("VULNERABLE", x=3.0, y=25.0, vx=-1.0)
    i = np.argmin(np.abs(c_step.s - 25.0))
    check("planner: a pedestrian stepping into the road narrows the SOFT "
          "corridor (predicted motion is a forecast, not an observation)",
          c_step.soft_width()[i] < c_still.soft_width()[i] - 0.5,
          f"still {c_still.soft_width()[i]:.2f} -> stepping {c_step.soft_width()[i]:.2f}")
    check("planner: but the hard corridor is unchanged, so a crowded street "
          "stays plannable instead of becoming infeasible",
          abs(w_step - w_still) < 0.01, f"{w_still:.2f} vs {w_step:.2f}")
    check("planner: soft bounds are never laxer than hard ones",
          np.all(c_step.soft_min >= c_step.d_min - 1e-9)
          and np.all(c_step.soft_max <= c_step.d_max + 1e-9))

    # -- polynomials -------------------------------------------------------
    from planning.frenet import poly_eval
    c5 = quintic(0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 4.0)
    check("planner: the quintic meets its boundary conditions",
          abs(poly_eval(c5, 4.0) - 3.0) < 1e-6
          and abs(poly_eval(c5, 4.0, 1)) < 1e-6
          and abs(poly_eval(c5, 0.0) - 0.0) < 1e-6,
          f"end {poly_eval(c5, 4.0):.4f}, end-vel {poly_eval(c5, 4.0, 1):.4f}")
    c4 = quartic(0.0, 10.0, 0.0, 14.0, 0.0, 3.0)
    check("planner: the velocity-keeping quartic reaches the target speed",
          abs(poly_eval(c4, 3.0, 1) - 14.0) < 1e-6, f"{poly_eval(c4, 3.0, 1):.4f}")

    # -- planning ----------------------------------------------------------
    planner = FrenetPlanner()
    corridor = build_corridor(path, [])
    traj, diag = planner.plan(path, corridor, ego_speed_mps=10.0,
                              target_speed_mps=12.0)
    check("planner: a clear straight road yields a trajectory",
          traj is not None, str(diag.get("failure")))
    check("planner: on a clear road it stays near the reference",
          abs(traj.target_d) < 0.6, f"{traj.target_d:+.2f} m")
    check("planner: and accelerates toward the target speed",
          traj.end_speed > 10.0, f"{traj.end_speed:.1f} m/s")

    ob_right = Obstacle(x=1.5, y=20.0, half_width=0.3, half_length=0.3,
                        group="VULNERABLE", track_id=1)
    traj_r, _ = planner.plan(path, build_corridor(path, [ob_right]), 10.0, 12.0)
    check("planner: it steers AWAY from an obstacle on the right",
          traj_r is not None and traj_r.target_d < -0.2,
          f"{traj_r.target_d:+.2f} m" if traj_r else "no plan")

    ob_left = Obstacle(x=-1.5, y=20.0, half_width=0.3, half_length=0.3,
                       group="VULNERABLE", track_id=2)
    traj_l, _ = planner.plan(path, build_corridor(path, [ob_left]), 10.0, 12.0)
    check("planner: and the other way for an obstacle on the left",
          traj_l is not None and traj_l.target_d > 0.2,
          f"{traj_l.target_d:+.2f} m" if traj_l else "no plan")

    check("planner: the chosen trajectory keeps a real clearance margin",
          traj_r.min_clearance_m >= PlannerConfig().min_clearance_m,
          f"{traj_r.min_clearance_m:.2f} m")

    wall = [Obstacle(x=float(x), y=20.0, half_width=0.6, half_length=0.6,
                     group="STATIC_OBSTACLE", track_id=i)
            for i, x in enumerate(np.arange(-4.0, 4.1, 0.8))]
    blocked_corr = build_corridor(path, wall)
    traj_b, diag_b = planner.plan(path, blocked_corr, 10.0, 12.0)
    check("planner: a fully blocked road yields NO trajectory, with the reason",
          traj_b is None and diag_b.get("failure") == "corridor blocked",
          str(diag_b.get("failure")))
    check("planner: and it reports where the blockage starts",
          diag_b.get("first_blocked_s") is not None,
          str(diag_b.get("first_blocked_s")))

    # -- the competence gate ----------------------------------------------
    short = reference_path_from_free_space(
        road(lambda f: 0.0, lambda f: 4.0 if f < 15.0 else 0.0))
    traj_s, diag_s = planner.plan(short, build_corridor(short, []),
                                  ego_speed_mps=12.0, target_speed_mps=14.0)
    check("planner: a plan is TRUNCATED to the observed road rather than "
          "extrapolated past it",
          traj_s is not None and traj_s.s[-1] <= short.observed_length_m + 0.5,
          f"plan reaches {traj_s.s[-1]:.1f} m, observed "
          f"{short.observed_length_m:.1f} m" if traj_s else "no plan")
    check("planner: and the truncation carries its reason, so a short plan is "
          "distinguishable from a short road",
          traj_s is not None and traj_s.truncated
          and "not observed" in traj_s.truncated_reason,
          traj_s.truncated_reason if traj_s else "")

    long_road = reference_path_from_free_space(
        road(lambda f: 0.0, lambda f: 4.0, n_fwd=600))
    unlimited, _ = planner.plan(long_road, build_corridor(long_road, []),
                                12.0, 14.0)
    check("planner: given road observed beyond the planning horizon, nothing "
          "is truncated",
          unlimited is not None and not unlimited.truncated,
          f"plan {unlimited.s[-1]:.0f} m of {long_road.observed_length_m:.0f} m "
          f"observed" if unlimited else "no plan")

    # -- feasibility -------------------------------------------------------
    cfg = PlannerConfig()
    check("planner: the steering limit comes from the bicycle model, not a "
          "tuned constant",
          abs(cfg.max_curvature - np.tan(cfg.max_steer_rad) / cfg.wheelbase_m) < 1e-9)
    check("planner: no chosen trajectory exceeds the steering limit",
          traj_r.max_curvature <= cfg.max_curvature + 1e-9,
          f"{traj_r.max_curvature:.4f} vs {cfg.max_curvature:.4f}")

    tight = FrenetPlanner(PlannerConfig(max_steer_rad=0.02, wheelbase_m=2.7))
    _, diag_t = tight.plan(path, build_corridor(path, [ob_right]), 14.0, 16.0)
    check("planner: a vehicle that cannot steer sharply enough is told so, "
          "rather than being handed a trajectory it cannot execute",
          diag_t.get("n_feasible", 1) == 0 or "steer" in str(diag_t.get("failure", "")),
          str(diag_t.get("failure")))


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

def test_multi_source_granularity():
    """Combining several Indian datasets into one granularity experiment.

    The fine arm has to be genuinely fine-grained for the comparison to mean
    anything, and the project's merged 15-class taxonomy has already collapsed
    the distinctions under test -- measuring what collapsing costs on labels
    that are already collapsed is circular. So the arms are built from IDD and
    UVH-26 with their OWN vocabularies preserved and unioned.

    That union is where the correctness risk sits, and these checks target it:
    UVH-26 carries no pedestrians, so a class list derived per-source omits
    "person" from the semantic arm, and each source's ids then index a
    different list. Labels would point at the wrong classes with nothing
    reporting an error.
    """
    import subprocess
    import tempfile

    import yaml

    vocabularies = {
        # IDD-like: has the vulnerable road users UVH-26 lacks.
        "idd": ["car", "bus", "truck", "person", "rider", "motorcycle",
                "bicycle", "autorickshaw", "animal", "vehicle fallback",
                "traffic sign", "caravan"],
        # UVH-26-like: body-type granularity, vehicles only.
        "uvh": ["Hatchback", "Sedan", "SUV", "MUV", "Van", "LCV",
                "Tempo-Traveller", "Mini-Bus", "Bus", "Truck", "Three-Wheeler",
                "Two-Wheeler", "Bicycle", "Other"],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        srcs = []
        for tag, names in vocabularies.items():
            root = tmp / tag
            root.mkdir(parents=True)
            (root / f"{tag}.yaml").write_text(
                "train: train/images\nval: val/images\nnames:\n"
                + "\n".join(f"  {i}: {n}" for i, n in enumerate(names)),
                encoding="utf-8")
            for split, n_img in (("train", 6), ("val", 3)):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)
                for k in range(n_img):
                    # Deliberately identical filenames across both sources.
                    (root / split / "images" / f"{k:03d}.jpg").write_bytes(b"\xff\xd8\xff")
                    rows = [f"{j} 0.5 0.5 0.2 0.3" for j in range(len(names))]
                    (root / split / "labels" / f"{k:03d}.txt").write_text(
                        "\n".join(rows), encoding="utf-8")
            srcs.append(str(root))

        results = {}
        for level in ("fine", "semantic", "decision"):
            out = tmp / f"g_{level}"
            r = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "data" / "prepare_taxonomy.py"),
                 "--src", *srcs, "--level", level, "--out", str(out)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300)
            if r.returncode != 0:
                results[level] = {"error": r.stderr.strip()[-300:]}
                continue

            cfg = yaml.safe_load((out / f"{level}.yaml").read_text(encoding="utf-8"))
            names = cfg["names"]
            names = ({int(k): v for k, v in names.items()} if isinstance(names, dict)
                     else dict(enumerate(names)))
            boxes, frames, max_id = 0, 0, -1
            for split in ("train", "val"):
                for f in (out / split / "labels").glob("*.txt"):
                    frames += 1
                    for line in f.read_text(encoding="utf-8").strip().splitlines():
                        boxes += 1
                        max_id = max(max_id, int(line.split()[0]))
            results[level] = {"names": names, "n_classes": len(names),
                              "boxes": boxes, "frames": frames, "max_id": max_id}

        errors = {k: v.get("error") for k, v in results.items() if "error" in v}
        check("multi-source: all three levels build from two sources at once",
              not errors, str(errors))
        if errors:
            return

        check("multi-source: the three arms hold IDENTICAL box counts, so the "
              "comparison measures taxonomy and not data",
              len({r["boxes"] for r in results.values()}) == 1,
              str({k: v["boxes"] for k, v in results.items()}))

        # The check that catches the real bug: an id outside the declared class
        # list means the labels and the config disagree, and nothing else in
        # the pipeline would report it.
        for level, r in results.items():
            check(f"multi-source [{level}]: every label id is within the "
                  f"declared class list (a per-source list would put them out "
                  f"of range)",
                  r["max_id"] < r["n_classes"],
                  f"max id {r['max_id']} vs {r['n_classes']} classes")

        sem = {v.lower() for v in results["semantic"]["names"].values()}
        check("multi-source: the semantic arm keeps classes present in only ONE "
              "source -- UVH-26 has no pedestrians, and building its class list "
              "per-source silently dropped 'person'",
              {"person", "rider", "animal"} <= sem, str(sorted(sem)))

        fine = {v.lower() for v in results["fine"]["names"].values()}
        check("multi-source: the fine arm keeps body-type granularity, which is "
              "the distinction the decision taxonomy argues is unnecessary",
              {"hatchback", "sedan", "suv"} <= fine, str(sorted(fine)))
        check("multi-source: the fine arm merges names differing only in case "
              "('Truck' and 'truck' are one class)",
              sum(1 for n in fine if n == "truck") == 1
              and results["fine"]["n_classes"] < sum(
                  len(v) for v in vocabularies.values()),
              str(results["fine"]["n_classes"]))

        n = {k: v["n_classes"] for k, v in results.items()}
        check("multi-source: the arms are a genuine spread, not three near-"
              "identical class counts",
              n["fine"] > 2 * n["semantic"] > n["decision"] == 6, str(n))

        expected_frames = {"train": 12, "val": 6}
        check("multi-source: identically-named images from different sources do "
              "not overwrite each other",
              results["decision"]["frames"] == sum(expected_frames.values()),
              f"{results['decision']['frames']} frames, expected "
              f"{sum(expected_frames.values())}")


def test_indian_dataset_ingestion():
    """Any of the named Indian datasets must convert with one command.

    The problem statement points at IDD plus "Mendeley traffic data", and the
    Mendeley candidates ship in three different formats -- VOC XML, YOLO text
    and COCO JSON. One converter handles all three, and the property that
    matters most is not that it parses them but that it says loudly which class
    names the taxonomy does NOT recognise: an unrecognised name is data about
    to be discarded silently.
    """
    import json
    import subprocess
    import tempfile
    import xml.etree.ElementTree as ET

    try:
        from PIL import Image
    except ImportError:
        check("ingestion: Pillow available for the format round-trip", False,
              "pip install pillow")
        return

    from data.prepare_indian import detect_format
    from models.taxonomy import EXCLUDED_CLASSES, DecisionTaxonomy, normalise

    # Names drawn from the actual Mendeley Indian datasets, plus one the
    # taxonomy has never heard of and one it excludes on purpose.
    NAMES = ["car", "bike", "rickshaw", "bullock cart", "cow",
             "signboard", "jugaad"]

    def build(root: Path, fmt: str, n_img: int = 6):
        def img(p):
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640, 480), (128, 128, 128)).save(p)

        if fmt == "voc":
            for k in range(n_img):
                img(root / "JPEGImages" / f"{k:03d}.jpg")
                ann = ET.Element("annotation")
                sz = ET.SubElement(ann, "size")
                ET.SubElement(sz, "width").text = "640"
                ET.SubElement(sz, "height").text = "480"
                for j, n in enumerate(NAMES):
                    o = ET.SubElement(ann, "object")
                    ET.SubElement(o, "name").text = n
                    bb = ET.SubElement(o, "bndbox")
                    for t, v in (("xmin", 40 + j * 30), ("ymin", 100),
                                 ("xmax", 100 + j * 30), ("ymax", 300)):
                        ET.SubElement(bb, t).text = str(v)
                d = root / "Annotations"
                d.mkdir(parents=True, exist_ok=True)
                ET.ElementTree(ann).write(d / f"{k:03d}.xml")

        elif fmt == "yolo":
            for k in range(n_img):
                img(root / "images" / f"{k:03d}.jpg")
                d = root / "labels"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{k:03d}.txt").write_text(
                    "\n".join(f"{j} 0.5 0.5 0.2 0.3" for j in range(len(NAMES))))
            (root / "classes.txt").write_text("\n".join(NAMES))

        else:  # coco
            imgs, anns = [], []
            for k in range(n_img):
                img(root / "images" / f"{k:03d}.jpg")
                imgs.append({"id": k, "file_name": f"{k:03d}.jpg",
                             "width": 640, "height": 480})
                for j in range(len(NAMES)):
                    anns.append({"id": len(anns), "image_id": k,
                                 "category_id": j,
                                 "bbox": [40 + j * 30, 100, 60, 200]})
            (root / "ann.json").write_text(json.dumps({
                "images": imgs, "annotations": anns,
                "categories": [{"id": i, "name": n}
                               for i, n in enumerate(NAMES)]}))

    script = PROJECT_ROOT / "data" / "prepare_indian.py"
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for fmt in ("voc", "yolo", "coco"):
            src = tmp / fmt
            src.mkdir(parents=True)
            build(src, fmt)

            check(f"ingestion [{fmt}]: the format is detected from the files "
                  f"on disk, not from a flag",
                  detect_format(src) == fmt, f"detected {detect_format(src)!r}")

            r = subprocess.run(
                [sys.executable, str(script), "--src", str(src),
                 "--out", str(tmp / f"{fmt}_out"), "--copy"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300)
            results[fmt] = r
            check(f"ingestion [{fmt}]: converts without error",
                  r.returncode == 0, r.stderr.strip()[-300:])
            if r.returncode != 0:
                continue

            out = tmp / f"{fmt}_out"
            check(f"ingestion [{fmt}]: emits a dataset config with the SOURCE "
                  f"class names preserved (the granularity experiment needs "
                  f"to know which original class each box came from)",
                  (out / "data.yaml").exists()
                  and "rickshaw" in (out / "data.yaml").read_text(encoding="utf-8"))

            boxes = sum(
                len(f.read_text(encoding="utf-8").strip().splitlines())
                for split in ("train", "val")
                for f in (out / split / "labels").glob("*.txt"))
            check(f"ingestion [{fmt}]: every annotation survives the round trip",
                  boxes == 6 * len(NAMES), f"{boxes} of {6 * len(NAMES)}")

            # A converted dataset must feed straight into the granularity tool.
            rep = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "data" / "prepare_taxonomy.py"),
                 "--src", str(out), "--report-only"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300)
            check(f"ingestion [{fmt}]: the output feeds prepare_taxonomy.py "
                  f"directly",
                  "UNRECOGNISED" in rep.stdout,
                  rep.stderr.strip()[-200:] or rep.stdout[-200:])

    # The reporting distinction is the point of the whole script.
    out = results["voc"].stdout
    check("ingestion: an unrecognised class name is flagged as data about to "
          "be discarded, not passed over in silence",
          "UNRECOGNISED" in out and "jugaad" in out, out[-300:])
    check("ingestion: a deliberately excluded class is reported as a decision, "
          "not as a gap",
          "excluded by design" in out, out[-300:])
    check("ingestion: the two are distinguished, so a real hole in the mapping "
          "is not lost among things we meant to drop",
          out.count("UNRECOGNISED") >= 1 and "excluded by design" in out)

    # Vocabulary coverage for the datasets the problem statement names.
    tx = DecisionTaxonomy()
    indian_vocab = [
        "autorickshaw", "rickshaw", "e-rickshaw", "toto", "cycle rickshaw",
        "bike", "scooty", "motorbike", "bullock cart", "animal drawn cart",
        "cow", "buffalo", "goat", "elephant", "pillion", "hawker", "vendor",
        "tempo traveller", "mini truck", "pickup", "trolley", "jcb",
        "water tanker", "tractor", "pushcart", "vehicle fallback",
    ]
    unmapped = [n for n in indian_vocab
                if tx.map_name(n) is None and normalise(n) not in EXCLUDED_CLASSES]
    check("ingestion: the taxonomy covers the vocabulary of IDD and the "
          "Mendeley Indian datasets",
          not unmapped, f"unmapped: {unmapped}")

    check("ingestion: an animal-drawn cart is VULNERABLE — the hazard is the "
          "animal that can bolt, not the cart",
          tx.map_name("bullock cart") == "VULNERABLE",
          str(tx.map_name("bullock cart")))
    check("ingestion: a speed breaker is excluded, not treated as an obstacle "
          "to steer around (it is driven over)",
          normalise("speed breaker") in EXCLUDED_CLASSES)


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
                if line.startswith("target classes ("):
                    class_counts[level] = line.split("(")[1].split(")")[0]

        check("granularity: all three levels build without error", ok, str(counts))
        check("granularity: the three prepared datasets contain IDENTICAL box "
              "counts -- same boxes, only the labels differ",
              len(set(counts.values())) == 1, str(counts))
        n = {k: int(v) for k, v in class_counts.items()}
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
        ("Path planning: reference, corridor, Frenet, competence gate",
         test_planning),
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
        ("Multi-source granularity build (IDD + UVH-26 union)",
         test_multi_source_granularity),
        ("Indian dataset ingestion (IDD / Mendeley formats)",
         test_indian_dataset_ingestion),
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
