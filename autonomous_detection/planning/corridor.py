"""Drivable corridor: lateral bounds along the reference path.

WHAT THIS IS FOR
----------------
The reference path says where the road goes. The corridor says how far either
side of it the vehicle may be at each point. Together they turn planning into a
one-dimensional problem at each longitudinal station, which is what makes
Frenet trajectory generation fast enough to replan every cycle.

WHERE THE TAXONOMY BECOMES CONSEQUENTIAL
-----------------------------------------
This is the module where classifying an object stops being a label and starts
changing the vehicle's behaviour. Each decision group carries a
`lateral_clearance_m`, and that number is exactly the amount by which the
corridor narrows beside an obstacle of that group:

    pedestrian / rider / animal   1.5 m
    motorcycle / bicycle          1.2 m
    auto-rickshaw                 1.0 m
    heavy vehicle                 1.0 m
    car / van                     0.8 m
    static obstacle               0.8 m

A pedestrian therefore gets nearly twice the berth of a parked cart, because
the group says so -- not because a constant was tuned per class in this file.
The same numbers become capsule radii in the MATLAB port.

HARD AND SOFT BOUNDS
--------------------
The split is by WHAT IS KNOWN, not by what kind of object it is.

**Where an obstacle is now** is a HARD bound, for everything -- a pedestrian
standing in the road is as impassable as a boulder. An earlier version of this
module made only static objects hard and left movers to soft bounds alone, on
the theory that movers should not be able to close the corridor. The effect was
the exact inversion of the intent: a traffic cone diverted the vehicle while a
pedestrian did not, because the pedestrian never touched the hard bound. The
most dangerous class had the weakest effect on the plan.

**Where an obstacle might BE** -- the region swept by its predicted motion
beyond its current footprint -- is a SOFT bound. That part is a forecast, not an
observation, and forecasts should raise cost rather than forbid. It is also what
keeps a crowded Indian street plannable: if every predicted sweep were hard, the
corridor would close completely and the planner would report failure, when what
a driver actually does is squeeze past with reduced clearance and reduced speed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import GROUP_BEHAVIOUR, NAME_TO_GROUP, normalise  # noqa: E402

# Half-width of the ego vehicle, metres. A typical Indian hatchback is about
# 1.7 m wide; the corridor must account for the vehicle's own body, not just
# for the gap between obstacles.
EGO_HALF_WIDTH_M = 0.9


@dataclass
class Obstacle:
    """A tracked object in ground-plane metres, ready for corridor building."""
    x: float                       # lateral position, m (+ right of ego)
    y: float                       # forward position, m
    half_width: float = 0.9        # own physical half-width, m
    half_length: float = 2.0       # own physical half-length, m
    group: str = "LIGHT_VEHICLE"   # decision group
    vx: float = 0.0                # lateral velocity, m/s
    vy: float = 0.0                # forward velocity relative to ground, m/s
    track_id: Optional[int] = None

    @classmethod
    def from_class_name(cls, name: str, **kwargs) -> "Obstacle":
        """Build from a raw detector class name, mapping it to its group.

        Falls back to LIGHT_VEHICLE -- the baseline parameters -- when the name
        is unrecognised, rather than to the most or least cautious group. An
        unknown object should not silently receive a pedestrian's berth, nor a
        cone's.
        """
        group = NAME_TO_GROUP.get(normalise(name), "LIGHT_VEHICLE")
        return cls(group=group, **kwargs)

    @property
    def clearance_m(self) -> float:
        """Lateral berth this object's group demands."""
        return GROUP_BEHAVIOUR.get(
            self.group, GROUP_BEHAVIOUR["LIGHT_VEHICLE"])["lateral_clearance_m"]

    @property
    def can_move(self) -> bool:
        return GROUP_BEHAVIOUR.get(
            self.group, GROUP_BEHAVIOUR["LIGHT_VEHICLE"])["can_move"]

    @property
    def blocks_view(self) -> bool:
        return GROUP_BEHAVIOUR.get(
            self.group, GROUP_BEHAVIOUR["LIGHT_VEHICLE"])["blocks_view"]


@dataclass
class Corridor:
    """Lateral bounds along the reference path, plus why they are where they are."""
    s: np.ndarray                  # stations, m
    d_min: np.ndarray              # hard right-hand bound (negative = left)
    d_max: np.ndarray              # hard left-hand bound
    soft_min: np.ndarray           # preferred bounds including movers
    soft_max: np.ndarray
    blocked: np.ndarray            # bool: hard bounds cross -> impassable
    limiting: dict = field(default_factory=dict)   # station index -> track_id

    def width(self) -> np.ndarray:
        return self.d_max - self.d_min

    def soft_width(self) -> np.ndarray:
        return self.soft_max - self.soft_min

    def is_passable(self) -> bool:
        return not bool(self.blocked.any())

    def first_blocked_s(self) -> Optional[float]:
        idx = np.flatnonzero(self.blocked)
        return float(self.s[idx[0]]) if len(idx) else None

    def contains(self, s, d, soft: bool = False) -> np.ndarray:
        """Is (s, d) inside the corridor?"""
        lo = np.interp(s, self.s, self.soft_min if soft else self.d_min)
        hi = np.interp(s, self.s, self.soft_max if soft else self.d_max)
        return (d >= lo) & (d <= hi)

    def clearance_at(self, s, d) -> np.ndarray:
        """Distance from (s, d) to the nearer hard wall, metres."""
        lo = np.interp(s, self.s, self.d_min)
        hi = np.interp(s, self.s, self.d_max)
        return np.minimum(d - lo, hi - d)


def build_corridor(path, obstacles, horizon_s: float = 3.0,
                   ego_half_width_m: float = EGO_HALF_WIDTH_M,
                   n_stations: int = 60) -> Corridor:
    """Lateral bounds along `path`, narrowed by each obstacle's group clearance.

    `horizon_s` is how far ahead in TIME a moving obstacle's footprint is swept.
    A vehicle that will occupy space during the manoeuvre constrains the
    corridor even if it is not there yet -- planning against only present
    positions is how a planner drives into the space something is moving into.
    """
    s_grid = np.linspace(float(path.s[0]), float(path.s[-1]), n_stations)

    # Start from the road itself, less the ego vehicle's own half-width.
    road_half = np.maximum(path.half_width_at(s_grid) - ego_half_width_m, 0.0)
    d_min, d_max = -road_half.copy(), road_half.copy()
    soft_min, soft_max = d_min.copy(), d_max.copy()
    limiting: dict = {}

    for ob in obstacles:
        s_ob, d_ob = path.to_frenet(ob.x, ob.y)
        s_ob, d_ob = float(s_ob[0]), float(d_ob[0])

        # Where it IS: a hard bound, whatever kind of object it is.
        now_s_lo, now_s_hi = s_ob - ob.half_length, s_ob + ob.half_length
        now_d_lo, now_d_hi = d_ob - ob.half_width, d_ob + ob.half_width

        # Where it MIGHT BE: the swept region, a soft bound. A pedestrian
        # stepping off a kerb will occupy, over the next few seconds, far more
        # of the road than its body covers now -- but that is a forecast, so it
        # raises cost rather than forbidding.
        pred_s_lo, pred_s_hi = now_s_lo, now_s_hi
        pred_d_lo, pred_d_hi = now_d_lo, now_d_hi
        if ob.can_move and (abs(ob.vx) > 1e-3 or abs(ob.vy) > 1e-3):
            pred_s_hi += max(ob.vy * horizon_s, 0.0)
            pred_s_lo += min(ob.vy * horizon_s, 0.0)
            sweep = ob.vx * horizon_s
            pred_d_hi += max(sweep, 0.0)
            pred_d_lo += min(sweep, 0.0)

        # The berth: the obstacle's own body, plus its group's clearance, plus
        # the ego vehicle's half-width.
        berth = ob.clearance_m + ego_half_width_m

        def stations(s_lo: float, s_hi: float) -> np.ndarray:
            """Stations an extent constrains, including the bracketing ones.

            A pedestrian is far shorter than the station spacing, so an
            interior-only test lets a small obstacle fall between two stations
            and constrain NOTHING. The ego vehicle also has length, so it is
            beside the obstacle for longer than the obstacle's own extent --
            being conservative here is both safer and more accurate.
            """
            mask = (s_grid >= s_lo) & (s_grid <= s_hi)
            lo_i = int(np.searchsorted(s_grid, s_lo, side="right") - 1)
            hi_i = int(np.searchsorted(s_grid, s_hi, side="left"))
            for edge in (lo_i, hi_i):
                if 0 <= edge < len(s_grid):
                    mask[edge] = True
            return np.flatnonzero(mask)

        def narrow(indices, lo_edge, hi_edge, bounds_min, bounds_max):
            """Cut the corridor beside an obstacle, from the tighter side.

            Whichever side leaves more room is the side the vehicle would
            actually pass on, so the other side is what gets cut away.
            """
            block_lo, block_hi = lo_edge - berth, hi_edge + berth
            for i in indices:
                room_left = block_lo - d_min[i]
                room_right = d_max[i] - block_hi
                if room_right >= room_left:
                    bounds_min[i] = max(bounds_min[i], block_hi)
                else:
                    bounds_max[i] = min(bounds_max[i], block_lo)
                limiting[int(i)] = ob.track_id

        now_idx = stations(now_s_lo, now_s_hi)
        if len(now_idx):
            narrow(now_idx, now_d_lo, now_d_hi, d_min, d_max)

        pred_idx = stations(pred_s_lo, pred_s_hi)
        if len(pred_idx):
            narrow(pred_idx, pred_d_lo, pred_d_hi, soft_min, soft_max)

    # Soft bounds can never be laxer than hard ones -- a wall is a wall,
    # whether or not anything is predicted to move through it.
    soft_min = np.maximum(soft_min, d_min)
    soft_max = np.minimum(soft_max, d_max)

    blocked = d_max <= d_min
    return Corridor(s=s_grid, d_min=d_min, d_max=d_max,
                    soft_min=np.minimum(soft_min, soft_max),
                    soft_max=np.maximum(soft_min, soft_max),
                    blocked=blocked, limiting=limiting)
