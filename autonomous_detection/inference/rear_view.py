"""Rear and side-rear assessment for the overtaking decision.

WHAT THIS ANSWERS
-----------------
Before pulling out to overtake, the lane being moved into must be clear
*behind* the ego vehicle, not only ahead of it. A forward-only system can
report "no oncoming traffic" while a vehicle is already closing from behind in
exactly the lane the driver is about to enter.

The rule implemented here is the one a driver actually applies: look back along
the side you are moving into, as far back as you can see, and only pull out if
nothing there will reach you during the manoeuvre.

TWO THINGS MOST RULE-BASED SYSTEMS GET WRONG, AND WHAT IS DONE INSTEAD
-----------------------------------------------------------------------
1. **The manoeuvre time is guessed.** A hard-coded "8 seconds" has no defence.
   Here it is derived from overtaking sight distance as specified by IRC:66 for
   Indian two-lane highways, from the ego speed, the speed of the vehicle being
   passed, and a standard acceleration. It changes with the situation, as it
   should: overtaking a slow autorickshaw at 40 km/h is not the same manoeuvre
   as passing a truck at 80.

2. **Sensor range is not checked against the decision.** A system that can see
   30 m behind cannot honestly certify a lane clear when a vehicle 60 m back
   would arrive inside the manoeuvre window. "I see nothing" is not "nothing is
   there" when the required sight distance exceeds what the sensor covers.
   `RearApproachMonitor` computes both numbers and refuses the permission when
   observation falls short -- which is the honest reading of "look as far back
   as the frame allows".

INPUTS
------
Works from either source, and reports which one it used:
  * a rear camera (IDD-X provides paired front/rear views), or
  * the side-rear region of the forward camera, which catches a vehicle already
    drawing level.
With neither, the verdict is `observable=False` and the caller must fail safe.

TRAFFIC SIDE
------------
India drives on the LEFT, so overtaking happens on the RIGHT and the oncoming
lane is to the RIGHT. This is the opposite of the US/Europe convention that
most published pipelines assume, and getting it backwards inverts every
lane-side test in the system. `traffic_side` makes it explicit rather than
implied by unstated convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# IRC:66-1976 overtaking sight distance for two-lane highways.
#
# Everything below is in SI. Two details are easy to get wrong and change the
# answer by a factor of two, so they are spelled out:
#
#   * The spacing s = 0.7*Vb + 6 takes Vb in METRES PER SECOND, not km/h.
#     Feeding km/h into it roughly doubles every downstream distance.
#   * Vb is the speed of the vehicle being OVERTAKEN, which the code takes as
#     16 km/h below the design speed when it is not measured. Setting Vb equal
#     to the ego speed describes overtaking something moving just as fast --
#     not a manoeuvre anyone performs, and it inflates the window badly.
#
# Computed values track the OSD figures tabulated in the standard across the
# whole speed range -- 139 m vs 165 m at 40 km/h, 201 m vs 235 m at 50, 477 m
# vs 470 m at 80, 755 m vs 640 m at 100 (deviation -16% to +18%, closest where
# it matters most for highway overtaking). The residual is the usual variation
# between published parameter tables; the decision layer applies a safety
# factor on top rather than treating the number as exact.
# --------------------------------------------------------------------------

REACTION_TIME_S = 2.0        # driver reaction before committing to the manoeuvre
SPEED_DIFFERENTIAL_KMPH = 16.0   # IRC: the overtaken vehicle is this much slower
KMPH_PER_MPS = 3.6

# IRC acceleration for the overtaking manoeuvre, m/s^2, against design speed in
# km/h. Slower roads permit brisker acceleration, so the value is not constant.
_ACCEL_TABLE_MPS2 = [(25, 1.41), (30, 1.30), (40, 1.24), (50, 1.11),
                     (65, 0.92), (80, 0.72), (100, 0.53)]


def overtake_accel_mps2(design_speed_mps: float) -> float:
    """IRC manoeuvre acceleration for a design speed, linearly interpolated."""
    v = max(design_speed_mps, 0.0) * KMPH_PER_MPS
    if v <= _ACCEL_TABLE_MPS2[0][0]:
        return _ACCEL_TABLE_MPS2[0][1]
    if v >= _ACCEL_TABLE_MPS2[-1][0]:
        return _ACCEL_TABLE_MPS2[-1][1]
    for (v0, a0), (v1, a1) in zip(_ACCEL_TABLE_MPS2, _ACCEL_TABLE_MPS2[1:]):
        if v0 <= v <= v1:
            return a0 + (a1 - a0) * (v - v0) / (v1 - v0)
    return _ACCEL_TABLE_MPS2[-1][1]


def overtaken_speed_default_mps(ego_speed_mps: float) -> float:
    """Speed of the vehicle being passed, when it has not been measured."""
    return max(ego_speed_mps - SPEED_DIFFERENTIAL_KMPH / KMPH_PER_MPS, 0.0)


def spacing_m(overtaken_speed_mps: float) -> float:
    """IRC spacing between the overtaking and overtaken vehicle, in metres.

    s = 0.7*Vb + 6, with Vb in m/s.
    """
    return 0.7 * max(overtaken_speed_mps, 0.0) + 6.0


def manoeuvre_time_s(overtaken_speed_mps: float,
                     design_speed_mps: Optional[float] = None) -> float:
    """Time the ego vehicle spends in the adjacent lane, in seconds.

    T = sqrt(4s/a), the IRC:66 expression. This is the window during which the
    adjacent lane must stay clear -- both of oncoming traffic ahead and of
    anything closing from behind.
    """
    s = spacing_m(overtaken_speed_mps)
    ref = design_speed_mps if design_speed_mps is not None else overtaken_speed_mps
    return math.sqrt(4.0 * s / max(overtake_accel_mps2(ref), 0.05))


def overtaking_sight_distance_m(ego_speed_mps: float,
                                overtaken_speed_mps: Optional[float] = None,
                                oncoming_speed_mps: Optional[float] = None) -> float:
    """Full IRC:66 overtaking sight distance, OSD = d1 + d2 + d3, in metres.

    d1  ego closes on the slower vehicle during the reaction time
    d2  the overtaking manoeuvre itself, including both spacings
    d3  ground covered by an oncoming vehicle during the same window

    Reported alongside the decision so a verdict can be explained in the units
    road engineers already use, rather than in pixels or arbitrary scores.
    """
    vb = (overtaken_speed_mps if overtaken_speed_mps is not None
          else overtaken_speed_default_mps(ego_speed_mps))
    v = oncoming_speed_mps if oncoming_speed_mps is not None else ego_speed_mps
    s = spacing_m(vb)
    t_man = manoeuvre_time_s(vb, design_speed_mps=ego_speed_mps)
    d1 = vb * REACTION_TIME_S
    d2 = vb * t_man + 2.0 * s
    d3 = max(v, 0.0) * t_man
    return d1 + d2 + d3


@dataclass
class RearVerdict:
    """Outcome of the rear-and-side check, with the numbers behind it.

    Every field exists so the decision can be explained. A verdict a driver or
    an examiner cannot interrogate is not usable in a safety context, and
    "NOT POSSIBLE" with no reason is indistinguishable from a broken module.
    """
    clear: bool
    observable: bool
    reason: str
    source: str = "none"                       # "rear_camera" | "forward_side" | "none"
    required_clear_time_s: float = 0.0
    required_sight_m: float = 0.0
    observed_sight_m: float = 0.0
    blocking_gap_m: Optional[float] = None
    blocking_tta_s: Optional[float] = None
    blocking_track_id: Optional[int] = None
    n_candidates: int = 0

    def describe(self) -> str:
        if not self.observable:
            return (f"rear not assessable ({self.reason}); "
                    f"needs {self.required_sight_m:.0f} m of rear view")
        if not self.clear:
            gap = f"{self.blocking_gap_m:.0f} m" if self.blocking_gap_m is not None else "?"
            tta = f"{self.blocking_tta_s:.1f} s" if self.blocking_tta_s is not None else "?"
            return (f"vehicle closing from behind on the manoeuvre side: "
                    f"{gap} away, arrives in {tta} "
                    f"(need {self.required_clear_time_s:.1f} s)")
        return (f"rear clear over {self.observed_sight_m:.0f} m "
                f"({self.source}, {self.n_candidates} tracked)")


@dataclass
class RearApproachMonitor:
    """Decides whether the lane being entered is clear behind the ego vehicle."""

    traffic_side: str = "left"          # "left" = India/UK/Japan; "right" = US/EU
    safety_factor: float = 1.3          # margin on the manoeuvre window
    min_closing_speed_mps: float = 2.8  # ~10 km/h; below this, a vehicle behind
                                        # is not meaningfully catching up
    design_speed_mps: float = 22.2      # 80 km/h, the speed a vehicle could be
                                        # approaching at when none is detected
    min_required_sight_m: float = 30.0  # floor, so a stationary ego does not
                                        # certify the lane on a 2 m rear view

    # Set once from the deployment's geometry; None means unknown, which is
    # itself a reason to refuse rather than a reason to assume.
    max_rear_range_m: Optional[float] = field(default=None)

    # ---------------------------------------------------------------- side

    @property
    def manoeuvre_side(self) -> str:
        """The side the ego moves into to overtake.

        Left-hand traffic (India) overtakes on the RIGHT. Stated explicitly
        because assuming the US/Europe convention silently inverts every
        lane-side test in the pipeline.
        """
        return "right" if self.traffic_side == "left" else "left"

    def on_manoeuvre_side(self, track, frame_width: int,
                          ego_centre_x: Optional[float] = None) -> bool:
        """Is this track in the lane the ego is about to move into?"""
        cx = (track.bbox[0] + track.bbox[2]) / 2.0
        centre = ego_centre_x if ego_centre_x is not None else frame_width / 2.0
        return cx > centre if self.manoeuvre_side == "right" else cx < centre

    # ------------------------------------------------------------ distance

    @staticmethod
    def _distance_m(track, depth_map) -> Optional[float]:
        """Median depth over the lower half of the box, as elsewhere in the
        pipeline -- the upper half often contains background above the roof."""
        if depth_map is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y_mid, y2 = max(0, (y1 + y2) // 2), min(h, y2)
        if x2 <= x1 or y2 <= y_mid:
            return None
        return float(np.median(depth_map[y_mid:y2, x1:x2]))

    def _observed_range_m(self, depth_map) -> float:
        """How far back the sensor actually sees.

        The 95th percentile rather than the maximum: a single far pixel is
        usually sky or a depth artefact, and certifying a lane clear on the
        strength of one outlier pixel is exactly the failure this guards.
        """
        if self.max_rear_range_m is not None:
            return float(self.max_rear_range_m)
        if depth_map is None:
            return 0.0
        finite = np.asarray(depth_map)[np.isfinite(depth_map)]
        if finite.size == 0:
            return 0.0
        return float(np.percentile(finite, 95))

    # ------------------------------------------------------------- verdict

    def analyze(self, tracks, depth_map, ego_speed_mps: float,
                overtaken_speed_mps: Optional[float] = None,
                frame_width: int = 1280,
                source: str = "rear_camera",
                is_vehicle=None) -> RearVerdict:
        """Assess the manoeuvre-side lane behind the ego vehicle.

        `tracks` are from the rear camera, or the side-rear region of the
        forward camera. Passing None (no rear input at all) yields
        observable=False -- the caller must then refuse the overtake rather
        than treat silence as safety.
        """
        # How long the adjacent lane must stay clear, and therefore how far
        # back we need to be able to see.
        v_overtaken = (overtaken_speed_mps if overtaken_speed_mps is not None
                       else overtaken_speed_default_mps(ego_speed_mps))
        t_clear = manoeuvre_time_s(
            v_overtaken, design_speed_mps=ego_speed_mps) * self.safety_factor
        closing_worst = max(self.design_speed_mps - ego_speed_mps,
                            self.min_closing_speed_mps)
        required_sight = max(closing_worst * t_clear, self.min_required_sight_m)

        base = dict(required_clear_time_s=t_clear, required_sight_m=required_sight)

        if tracks is None:
            return RearVerdict(clear=False, observable=False, source="none",
                               reason="no rear view configured", **base)

        observed = self._observed_range_m(depth_map)
        if observed < required_sight:
            # The decisive check. Seeing nothing within 30 m says nothing about
            # a vehicle 70 m back that would arrive inside the window.
            return RearVerdict(
                clear=False, observable=False, source=source,
                reason=(f"rear view reaches {observed:.0f} m but the manoeuvre "
                        f"needs {required_sight:.0f} m"),
                observed_sight_m=observed, n_candidates=len(tracks), **base)

        if is_vehicle is None:
            def is_vehicle(t):
                return True

        worst = None
        n_candidates = 0
        for t in tracks:
            if not is_vehicle(t):
                continue
            if not self.on_manoeuvre_side(t, frame_width):
                continue
            n_candidates += 1
            gap = self._distance_m(t, depth_map)
            if gap is None:
                # A vehicle on the manoeuvre side whose distance cannot be
                # resolved is not evidence of a clear lane.
                return RearVerdict(
                    clear=False, observable=True, source=source,
                    reason="vehicle on the manoeuvre side at unresolvable distance",
                    observed_sight_m=observed,
                    blocking_track_id=getattr(t, "track_id", None),
                    n_candidates=n_candidates, **base)

            # Project-wide convention, set in inference/collision.py:
            # depth_speed_mps is POSITIVE when the gap is shrinking. A vehicle
            # behind that is merely keeping pace does not block the manoeuvre.
            closing = float(getattr(t, "depth_speed_mps", 0.0))
            if closing < self.min_closing_speed_mps:
                continue
            tta = gap / max(closing, 1e-3)
            if worst is None or tta < worst[0]:
                worst = (tta, gap, getattr(t, "track_id", None))

        if worst is not None and worst[0] < t_clear:
            tta, gap, tid = worst
            return RearVerdict(
                clear=False, observable=True, source=source,
                reason="vehicle closing from behind on the manoeuvre side",
                observed_sight_m=observed, blocking_gap_m=gap,
                blocking_tta_s=tta, blocking_track_id=tid,
                n_candidates=n_candidates, **base)

        return RearVerdict(clear=True, observable=True, source=source,
                           reason="rear clear", observed_sight_m=observed,
                           n_candidates=n_candidates, **base)
