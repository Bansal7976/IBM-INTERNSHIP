"""Frenet trajectory generation, scoring, and the competence gate.

THE METHOD
----------
Werling, Ziegler, Kammel and Thrun (ICRA 2010). Planning decouples into two
one-dimensional problems against the reference path: a quintic polynomial for
lateral offset d(t), and a quartic for longitudinal motion s(t) when the goal is
to hold a speed rather than reach a fixed point. Sampling over end states gives
a family of candidates; the cheapest feasible one wins.

This is the same formulation behind MATLAB's `trajectoryGeneratorFrenet`, which
is why the port is a translation rather than a rewrite.

WHAT IS ADDED HERE
------------------
Two things, and both come from the same principle.

**Clearance is scored, not just checked.** A trajectory that scrapes past a
pedestrian at the minimum legal berth is feasible but not good. Cost rises as
clearance falls, so among feasible options the planner prefers the one that
leaves room -- and in a corridor too tight for anyone's comfort it still returns
something rather than failing.

**The competence gate.** Before a trajectory is emitted it must lie entirely
within space the sensors actually observed. If it extends past the observed
range it is TRUNCATED and the speed reduced, never extrapolated. A planner that
plans into unobserved space is not planning; it is guessing, and the guess is
invisible in the output because a confident trajectory through unseen road
looks exactly like a confident trajectory through seen road.

That gate is also where an aerial view would enter: a drone extends the observed
range, so the same gate stops truncating and the vehicle commits to a manoeuvre
it would otherwise have refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# polynomials
# --------------------------------------------------------------------------

def quintic(p0, v0, a0, p1, v1, a1, T):
    """Coefficients of the quintic through the given boundary conditions."""
    c0, c1, c2 = p0, v0, a0 / 2.0
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    A = np.array([[T3, T4, T5],
                  [3 * T2, 4 * T3, 5 * T4],
                  [6 * T, 12 * T2, 20 * T3]], dtype=float)
    b = np.array([p1 - (c0 + c1 * T + c2 * T2),
                  v1 - (c1 + 2 * c2 * T),
                  a1 - 2 * c2], dtype=float)
    try:
        c3, c4, c5 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([c0, c1, c2, c3, c4, c5])


def quartic(p0, v0, a0, v1, a1, T):
    """Quartic for velocity-keeping: the end POSITION is left free."""
    c0, c1, c2 = p0, v0, a0 / 2.0
    T2, T3 = T * T, T ** 3
    A = np.array([[3 * T2, 4 * T3],
                  [6 * T, 12 * T2]], dtype=float)
    b = np.array([v1 - (c1 + 2 * c2 * T), a1 - 2 * c2], dtype=float)
    try:
        c3, c4 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([c0, c1, c2, c3, c4])


def poly_eval(c, t, order: int = 0):
    """Evaluate a polynomial (or its derivative) given coefficients c."""
    c = np.asarray(c, dtype=float)
    for _ in range(order):
        if len(c) <= 1:
            return np.zeros_like(np.asarray(t, dtype=float))
        c = c[1:] * np.arange(1, len(c))
    t = np.asarray(t, dtype=float)
    return sum(c[i] * t ** i for i in range(len(c)))


# --------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One candidate, in both Frenet and Cartesian form."""
    t: np.ndarray
    s: np.ndarray
    d: np.ndarray
    s_dot: np.ndarray
    d_dot: np.ndarray
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    cost: float = float("inf")
    feasible: bool = True
    reject_reason: str = ""
    truncated: bool = False
    truncated_reason: str = ""
    min_clearance_m: float = float("inf")
    max_curvature: float = 0.0

    @property
    def target_d(self) -> float:
        return float(self.d[-1]) if len(self.d) else 0.0

    @property
    def end_speed(self) -> float:
        return float(self.s_dot[-1]) if len(self.s_dot) else 0.0


@dataclass
class PlannerConfig:
    # sampling
    lateral_offsets_m: tuple = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    horizons_s: tuple = (2.0, 3.0, 4.0)
    speed_offsets_mps: tuple = (-2.0, -1.0, 0.0)
    dt: float = 0.1

    # vehicle limits
    max_speed_mps: float = 16.7        # 60 km/h
    max_accel_mps2: float = 2.5
    max_decel_mps2: float = 4.0
    wheelbase_m: float = 2.7
    max_steer_rad: float = 0.5         # ~29 deg
    max_lateral_accel_mps2: float = 3.0

    # cost weights
    w_jerk: float = 0.1
    w_time: float = 1.0
    w_deviation: float = 2.0
    w_speed: float = 4.0
    w_clearance: float = 20.0
    w_soft_violation: float = 30.0
    # Penalty on ending in a lateral position from which the corridor further
    # ahead is not reachable. See _score for why this is load-bearing.
    w_lookahead: float = 25.0
    lookahead_m: float = 20.0

    # safety
    min_clearance_m: float = 0.2
    comfortable_clearance_m: float = 1.0

    @property
    def max_curvature(self) -> float:
        """Kinematic bicycle limit: kappa = tan(delta_max) / L."""
        return np.tan(self.max_steer_rad) / self.wheelbase_m


# --------------------------------------------------------------------------

class FrenetPlanner:
    """Generates, scores and gates trajectories against a corridor."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.cfg = config or PlannerConfig()

    # -- generation -------------------------------------------------------

    def _corridor_offsets(self, corridor, d0: float) -> list:
        """End offsets read off the corridor itself, not from a fixed grid.

        A fixed grid misses narrow gaps. Squeezing past a stationary
        auto-rickshaw on a 7 m road leaves a corridor of
        [-2.50, -2.25] m -- real, passable, 0.25 m of room for the vehicle's
        centre line -- and not one of (-2, -1, -0.5, 0, 0.5, 1, 2) falls
        inside it. Every candidate was then rejected for leaving the corridor
        and the vehicle waited behind the obstacle indefinitely, which is the
        one thing an overtake-capable planner must not do.

        So the corridor contributes its own candidates: the midpoint of the
        admissible band at several stations, and points just inside each
        bound. The fixed grid is kept alongside -- it is what produces smooth,
        centred driving when there is room.
        """
        if corridor is None or len(corridor.s) == 0:
            return []
        picks = []
        n = len(corridor.s)
        # The NARROWEST passable band must always be offered. A localised
        # constriction -- a stopped auto-rickshaw occupies about three metres
        # of road -- falls between evenly spaced probes, so sampling fixed
        # fractions of the corridor can miss the one band the vehicle actually
        # has to fit through, and the planner then never sees the gap it is
        # supposed to take.
        widths = corridor.d_max - corridor.d_min
        passable = np.flatnonzero(widths > 0)
        stations = {0, n // 4, n // 2, (3 * n) // 4, n - 1}
        if len(passable):
            stations.add(int(passable[np.argmin(widths[passable])]))
        for i in stations:
            lo, hi = float(corridor.d_min[i]), float(corridor.d_max[i])
            if hi <= lo:
                continue                       # blocked here; nothing to offer
            inset = min(0.05, (hi - lo) / 4.0)
            picks.extend([(lo + hi) / 2.0, lo + inset, hi - inset,
                          float(np.clip(d0, lo, hi))])
        # Deduplicate to the nearest centimetre; near-identical end states cost
        # a full trajectory evaluation each and buy nothing.
        return sorted({round(d, 2) for d in picks})

    def _candidates(self, s0, s0_dot, d0, d0_dot, target_speed, corridor=None):
        cfg = self.cfg
        out = []
        offsets = sorted(set(cfg.lateral_offsets_m)
                         | set(self._corridor_offsets(corridor, d0)))
        for T in cfg.horizons_s:
            t = np.arange(0.0, T + cfg.dt, cfg.dt)
            for d1 in offsets:
                lat = quintic(d0, d0_dot, 0.0, d1, 0.0, 0.0, T)
                if lat is None:
                    continue
                d = poly_eval(lat, t)
                d_dot = poly_eval(lat, t, 1)
                d_jerk = poly_eval(lat, t, 3)
                # Reachable end speeds only. Sampling the target speed
                # regardless of the current one means that from a standstill
                # every candidate demands more acceleration than the vehicle
                # has -- 0 to 11 m/s inside a 4 s horizon needs 2.75 m/s^2
                # against a 2.5 limit -- so all of them are rejected on
                # acceleration and the vehicle can never pull away again.
                # Observed in closed loop as a planner that stopped correctly
                # for a pedestrian and then stayed stopped for the rest of the
                # episode, long after the road had cleared.
                # The quartic starts and ends at zero acceleration, so its
                # acceleration is not constant: a(t) = 6*dv/T * [t/T - (t/T)^2],
                # which peaks at t = T/2 with a_max = 1.5*dv/T. The reachable
                # speed change in one horizon is therefore (2/3)*a_limit*T, not
                # a_limit*T -- using the latter puts every candidate 50% over
                # the limit and all of them get rejected.
                REACHABLE = 2.0 / 3.0
                v_reachable = s0_dot + REACHABLE * cfg.max_accel_mps2 * T
                v_floor = max(s0_dot - REACHABLE * cfg.max_decel_mps2 * T, 0.0)
                for dv in cfg.speed_offsets_mps:
                    v1 = float(np.clip(target_speed + dv, v_floor,
                                       min(cfg.max_speed_mps, v_reachable)))
                    lon = quartic(s0, s0_dot, 0.0, v1, 0.0, T)
                    if lon is None:
                        continue
                    s = poly_eval(lon, t)
                    s_dot = poly_eval(lon, t, 1)
                    s_ddot = poly_eval(lon, t, 2)
                    s_jerk = poly_eval(lon, t, 3)
                    traj = Trajectory(t=t, s=s, d=d, s_dot=s_dot, d_dot=d_dot)
                    traj._jerk = float(np.sum(d_jerk ** 2) + np.sum(s_jerk ** 2))
                    traj._accel = s_ddot
                    traj._target_speed = target_speed
                    traj._T = T
                    out.append(traj)
        return out

    # -- feasibility ------------------------------------------------------

    def _check(self, traj: Trajectory, path, corridor) -> bool:
        cfg = self.cfg

        if np.any(traj.s_dot < -1e-6):
            traj.feasible, traj.reject_reason = False, "reverses along the path"
            return False
        if np.any(traj.s_dot > cfg.max_speed_mps + 1e-6):
            traj.feasible, traj.reject_reason = False, "exceeds speed limit"
            return False
        a = getattr(traj, "_accel", np.zeros_like(traj.t))
        if np.any(a > cfg.max_accel_mps2) or np.any(a < -cfg.max_decel_mps2):
            traj.feasible, traj.reject_reason = False, "exceeds acceleration limits"
            return False

        # Path curvature plus the trajectory's own lateral motion. A trajectory
        # the vehicle cannot physically steer is not a plan.
        kappa_path = np.interp(traj.s, path.s, path.curvature())

        # Lateral curvature taken with respect to ARC LENGTH, not time.
        # Converting a time derivative with d''(t)/v^2 blows up as v falls --
        # at 0.6 m/s the v^2 divisor is 0.36 and every candidate reports
        # impossible curvature. That rejected 99 of 108 candidates for
        # "exceeds steering limit" whenever the vehicle was crawling, so a
        # vehicle that had stopped for an obstacle could never steer around it
        # and stayed stopped for good. Differentiating against s has no such
        # divisor and is the quantity the steering limit is actually about:
        # curvature is dtheta/ds, a property of the path, not of how fast it
        # is driven.
        if len(traj.s) > 2 and (traj.s[-1] - traj.s[0]) > 1e-6:
            d_prime = np.gradient(traj.d, traj.s)
            d_double = np.gradient(d_prime, traj.s)
            kappa_lat = d_double / np.power(1.0 + d_prime ** 2, 1.5)
        else:
            kappa_lat = np.zeros_like(traj.s)
        kappa = np.abs(kappa_path + kappa_lat)
        traj.max_curvature = float(np.nanmax(kappa)) if len(kappa) else 0.0
        if traj.max_curvature > cfg.max_curvature:
            traj.feasible, traj.reject_reason = False, "exceeds steering limit"
            return False

        lat_accel = np.abs(kappa) * traj.s_dot ** 2
        if np.any(lat_accel > cfg.max_lateral_accel_mps2):
            traj.feasible, traj.reject_reason = False, "exceeds lateral acceleration"
            return False

        inside = corridor.contains(traj.s, traj.d, soft=False)
        if not np.all(inside):
            traj.feasible, traj.reject_reason = False, "leaves the drivable corridor"
            return False

        clearance = corridor.clearance_at(traj.s, traj.d)
        traj.min_clearance_m = float(np.min(clearance)) if len(clearance) else 0.0
        if traj.min_clearance_m < cfg.min_clearance_m:
            traj.feasible, traj.reject_reason = False, "insufficient clearance"
            return False

        return True

    # -- scoring ----------------------------------------------------------

    def _score(self, traj: Trajectory, corridor) -> float:
        cfg = self.cfg
        speed_err = (getattr(traj, "_target_speed", 0.0) - traj.end_speed) ** 2

        # Clearance is scored, not merely checked: among feasible options,
        # prefer the one that leaves room.
        deficit = max(cfg.comfortable_clearance_m - traj.min_clearance_m, 0.0)

        # Straying outside the soft bounds is allowed but paid for -- that is
        # what keeps a crowded street plannable instead of infeasible.
        #
        # Graded by DISTANCE outside, not by the fraction of points outside.
        # A binary count stops discriminating the moment the soft corridor
        # closes completely: every candidate is then equally "outside", the
        # term becomes a constant added to all of them, and the deviation cost
        # takes over and parks the vehicle on the centreline. Observed in
        # closed loop as a planner that oscillated between centre and offset
        # while a pedestrian walked in, deferring the manoeuvre until the hard
        # corridor forced it -- by which point the lateral move no longer fit
        # inside the horizon and the plan failed outright.
        #
        # With a graded penalty, a candidate hugging the edge of the swept
        # region always scores better than one sitting deep inside it, so the
        # vehicle starts moving over while it still can.
        soft_lo = np.interp(traj.s, corridor.s, corridor.soft_min)
        soft_hi = np.interp(traj.s, corridor.s, corridor.soft_max)
        outside = np.maximum(soft_lo - traj.d, 0.0) + np.maximum(traj.d - soft_hi, 0.0)
        soft_violation = float(np.mean(outside)) if len(outside) else 0.0

        # Where this end state leaves us for the road we can ALREADY see
        # beyond the horizon.
        #
        # Without this the planner systematically refuses to look far enough
        # ahead. Time is penalised, so the cheapest candidate is always the
        # shortest horizon; a 2 s horizon at 8 m/s reaches 15 m, and a
        # constriction at 25 m is simply invisible to it. The vehicle holds the
        # centreline until the constriction enters the horizon, by which point
        # the lateral displacement no longer fits and every candidate is
        # rejected. Observed as a planner that approached a stopped
        # auto-rickshaw, refused from 19 m out, and waited behind it forever --
        # with a passable 0.25 m gap beside it the whole time.
        #
        # Charging the end state against the corridor one look-ahead further on
        # makes it start drifting toward the gap while the move is still cheap.
        # Scanned over the whole look-ahead window for the TIGHTEST band, not
        # sampled at a single point. A stopped auto-rickshaw constrains only
        # the three metres of road it occupies; probing one station 20 m out
        # lands in open road either side of it and reports no constraint at
        # all, which is how this term came to do nothing on the first attempt.
        s_end = float(traj.s[-1]) if len(traj.s) else 0.0
        d_end = traj.target_d
        window = (corridor.s >= s_end) & (corridor.s <= s_end + cfg.lookahead_m)
        lookahead_miss = 0.0
        if window.any():
            los, his = corridor.d_min[window], corridor.d_max[window]
            misses = np.where(his > los,
                              np.maximum(los - d_end, 0.0) + np.maximum(d_end - his, 0.0),
                              0.0)
            lookahead_miss = float(misses.max())

        return (cfg.w_jerk * getattr(traj, "_jerk", 0.0)
                + cfg.w_time * getattr(traj, "_T", 0.0)
                + cfg.w_deviation * traj.target_d ** 2
                + cfg.w_speed * speed_err
                + cfg.w_clearance * deficit ** 2
                + cfg.w_soft_violation * soft_violation
                + cfg.w_lookahead * lookahead_miss ** 2)

    # -- the competence gate ----------------------------------------------

    def apply_competence_gate(self, traj: Trajectory, path,
                              observed_s_max: Optional[float] = None) -> Trajectory:
        """Truncate a trajectory to the space the sensors actually observed.

        A trajectory reaching past the observed road is not a plan through that
        road; it is a plan through an assumption. Truncating and slowing is the
        honest response, and it is recoverable -- the next cycle, having seen
        further, may extend it.
        """
        limit = observed_s_max if observed_s_max is not None else float(path.s[-1])
        if len(traj.s) == 0 or traj.s[-1] <= limit:
            return traj

        keep = traj.s <= limit
        if not keep.any():
            traj.feasible = False
            traj.reject_reason = "starts beyond the observed road"
            return traj

        n = int(np.count_nonzero(keep))
        traj.t, traj.s, traj.d = traj.t[:n], traj.s[:n], traj.d[:n]
        traj.s_dot, traj.d_dot = traj.s_dot[:n], traj.d_dot[:n]
        if len(traj.x):
            traj.x, traj.y = traj.x[:n], traj.y[:n]
        traj.truncated = True
        traj.truncated_reason = (
            f"truncated at {limit:.1f} m: the road beyond this point was not "
            f"observed" + (f" ({path.truncated_reason})" if path.truncated_reason else ""))
        return traj

    # -- to Cartesian -----------------------------------------------------

    @staticmethod
    def to_cartesian(traj: Trajectory, path) -> Trajectory:
        """Frenet (s, d) back to ground-plane (lateral, forward) metres."""
        px = np.interp(traj.s, path.s, path.x)
        py = np.interp(traj.s, path.s, path.y)
        heading = np.interp(traj.s, path.s, path.heading())
        # Normal to the path, pointing right of travel.
        nx, ny = np.cos(heading), -np.sin(heading)
        traj.x = px + traj.d * nx
        traj.y = py + traj.d * ny
        return traj

    # -- the entry point --------------------------------------------------

    def plan(self, path, corridor, ego_speed_mps: float, target_speed_mps: float,
             d0: float = 0.0, d0_dot: float = 0.0,
             observed_s_max: Optional[float] = None):
        """Best feasible trajectory, or None with the reasons recorded.

        Returns (trajectory, diagnostics). The diagnostics matter as much as
        the trajectory: when nothing is feasible, the caller needs to know
        whether the corridor was blocked, the speed too high for the curve, or
        the road simply unobserved -- each calls for a different response.
        """
        candidates = self._candidates(0.0, ego_speed_mps, d0, d0_dot,
                                      target_speed_mps, corridor)
        reasons: dict = {}
        best, best_cost = None, float("inf")

        for traj in candidates:
            if not self._check(traj, path, corridor):
                reasons[traj.reject_reason] = reasons.get(traj.reject_reason, 0) + 1
                continue
            traj.cost = self._score(traj, corridor)
            if traj.cost < best_cost:
                best, best_cost = traj, traj.cost

        diag = {
            "n_candidates": len(candidates),
            "n_feasible": sum(1 for c in candidates if c.feasible),
            "rejections": reasons,
            "corridor_passable": corridor.is_passable(),
            "first_blocked_s": corridor.first_blocked_s(),
            "observed_length_m": float(path.observed_length_m),
        }

        if best is None:
            diag["failure"] = (
                "corridor blocked" if not corridor.is_passable() else
                max(reasons, key=reasons.get) if reasons else "no candidates")
            return None, diag

        best = self.apply_competence_gate(best, path, observed_s_max)
        best = self.to_cartesian(best, path)
        diag["chosen_cost"] = best.cost
        diag["chosen_target_d"] = best.target_d
        diag["min_clearance_m"] = best.min_clearance_m
        diag["truncated"] = best.truncated
        if best.truncated:
            diag["truncated_reason"] = best.truncated_reason
        return best, diag
