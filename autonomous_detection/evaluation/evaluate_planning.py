"""Closed-loop planner evaluation: does the taxonomy actually change outcomes?

WHAT THIS ANSWERS
-----------------
The detection experiment (evaluate_swmc.py) shows that decision-grouped labels
score better on a safety-weighted metric. A reviewer's next question is the
obvious one: *so what?* A better label is only worth having if it changes what
the vehicle does.

This harness answers that by running the planner in closed loop over scenarios
and measuring what actually happens -- how close the vehicle passes a
pedestrian, how often it refuses, whether it completes the route. Then it runs
the SAME scenarios with the taxonomy switched off, replaced by one uniform
clearance for every obstacle. The difference between those two runs is the
contribution, stated in metres and collisions rather than in mAP.

THE ABLATIONS
-------------
    grouped     each obstacle's berth comes from its decision group (ours)
    uniform     every obstacle gets the same berth (the control)
    no_gate     grouped, but the competence gate is disabled -- the planner
                extrapolates past the observed road instead of truncating

The third is there because "refuse when you cannot see" sounds obviously right
and is therefore worth testing rather than asserting. If disabling it changes
nothing, the gate is decoration and should be reported as such.

WHY SYNTHETIC SCENARIOS
-----------------------
Because the outcome that matters -- how close the vehicle came to a pedestrian
-- has no ground truth in recorded footage. We can measure a detector against
labelled boxes, but nobody has annotated "the correct trajectory" for a dashcam
clip, and a planner cannot be scored against a video in which it never drove.

Closed-loop simulation gives exactly what a real clip cannot: the same
situation repeated with one variable changed, and the true position of every
agent throughout. The scenarios below are built from the problem statement's
own list -- village road, pedestrian crossing, an auto-rickshaw stopping
without warning, an occluding truck, a mountain bend.

Usage:
    python evaluation/evaluate_planning.py --out results/planning.json
    python evaluation/evaluate_planning.py --seeds 20 --out results/planning.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import GROUP_BEHAVIOUR                    # noqa: E402
from planning.corridor import EGO_HALF_WIDTH_M, Obstacle, build_corridor  # noqa: E402
from planning.frenet import FrenetPlanner, PlannerConfig       # noqa: E402
from planning.reference_path import (GroundGrid,               # noqa: E402
                                     reference_path_from_free_space)

RESOLUTION_M = 0.2

# The uniform-clearance control. Set to the mid-point of the group range, so
# the comparison is not rigged by making the control obviously too small --
# with this value the control is MORE generous than the taxonomy for cars and
# carts, and less generous only for pedestrians and two-wheelers.
UNIFORM_CLEARANCE_M = 1.0


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    """A road, a set of agents, and how far the ego is meant to get.

    AGENT VELOCITY CONVENTION -- easy to get backwards, and expensive when you
    do. `vy` is the agent's own velocity along the road in the SAME frame as
    the ego's forward motion:

        vy > 0   travelling the same way as the ego. A value below the ego's
                 speed is a slow vehicle ahead; the gap closes at the
                 difference.
        vy = 0   stationary in the carriageway.
        vy < 0   ONCOMING, closing at ego speed plus |vy|.

    Writing vy < 0 for "a slow truck ahead" turns the scenario into an
    unavoidable head-on with a non-reactive agent -- the planner then stops
    dead, correctly, and the truck drives through it, which the harness scores
    as our collision.
    """
    name: str
    description: str
    road_half_width_m: float = 3.5
    road_length_m: float = 60.0
    curvature_radius_m: float = 0.0          # 0 = straight
    observed_range_m: float = 45.0           # how far the sensors resolve
    agents: list = field(default_factory=list)   # dicts describing each agent
    target_speed_mps: float = 11.0           # ~40 km/h

    def road_centre_at(self, forward_m: float) -> float:
        if self.curvature_radius_m <= 0:
            return 0.0
        r = self.curvature_radius_m
        return r - np.sqrt(max(r * r - forward_m * forward_m, 0.0)) \
            if forward_m < r else r


def build_scenarios() -> list:
    """The five situations the problem statement names, as closed-loop tests."""
    return [
        Scenario(
            name="village_road_pedestrian",
            description="Unmarked village road; a pedestrian stands near the "
                        "carriageway edge and steps in as the vehicle nears.",
            road_half_width_m=3.0,
            agents=[dict(group="VULNERABLE", x=2.4, y=28.0, vx=-0.6, vy=0.0,
                         half_width=0.35, half_length=0.35)],
        ),
        Scenario(
            name="autorickshaw_stops",
            description="An auto-rickshaw ahead in the same lane stops without "
                        "warning -- the most common Indian overtake trigger.",
            agents=[dict(group="THREE_WHEELER", x=0.3, y=25.0, vx=0.0, vy=0.0,
                         half_width=0.75, half_length=1.5)],
        ),
        Scenario(
            name="occluding_truck",
            description="A slow truck ahead blocks the forward view; a "
                        "motorcycle filters up the near side.",
            agents=[dict(group="HEAVY_VEHICLE", x=-0.5, y=22.0, vx=0.0, vy=6.0,
                         half_width=1.3, half_length=5.0),
                    dict(group="TWO_WHEELER", x=2.6, y=16.0, vx=-0.3, vy=9.0,
                         half_width=0.4, half_length=1.0)],
        ),
        Scenario(
            name="cattle_and_cart",
            description="Cattle on the carriageway beside a parked handcart -- "
                        "a VULNERABLE agent and a STATIC_OBSTACLE that look "
                        "similar to a detector but demand different berths.",
            agents=[dict(group="VULNERABLE", x=-1.8, y=30.0, vx=0.2, vy=0.0,
                         half_width=0.5, half_length=1.2),
                    dict(group="STATIC_OBSTACLE", x=2.6, y=24.0, vx=0.0, vy=0.0,
                         half_width=0.6, half_length=0.8)],
        ),
        Scenario(
            name="mountain_bend",
            description="A 90 m bend with limited sight distance and an "
                        "oncoming heavy vehicle appearing around it.",
            curvature_radius_m=90.0,
            road_half_width_m=3.0,
            observed_range_m=28.0,
            target_speed_mps=8.0,
            agents=[dict(group="HEAVY_VEHICLE", x=-2.2, y=26.0, vx=0.0, vy=-11.0,
                         half_width=1.3, half_length=5.0)],
        ),
    ]


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

def scenario_grid(scn: Scenario, ego_forward_m: float,
                  ego_lateral_m: float = 0.0) -> GroundGrid:
    """The drivable grid as seen from the ego's current position.

    Rebuilt each step because the sensors see a limited range: as the vehicle
    advances, road that was beyond the horizon comes into view. Handing the
    planner the whole road at once would test a planner nobody can build.
    """
    n_lat = int(2 * (scn.road_half_width_m + 6.0) / RESOLUTION_M)
    n_fwd = int(scn.observed_range_m / RESOLUTION_M)
    grid = np.zeros((n_fwd, n_lat), dtype=bool)
    centre_col = n_lat // 2

    for row in range(n_fwd):
        absolute = ego_forward_m + row * RESOLUTION_M
        if absolute > scn.road_length_m:
            break
        # Road centre expressed relative to where the ego actually is. Using
        # the ego's forward position alone would glue the road to the vehicle,
        # so steering sideways would move the road with it and the corridor
        # would never register that the vehicle had left the centre.
        offset = scn.road_centre_at(absolute) - ego_lateral_m
        c = centre_col + offset / RESOLUTION_M
        hw = scn.road_half_width_m / RESOLUTION_M
        lo, hi = int(round(c - hw)), int(round(c + hw))
        grid[row, max(0, lo):min(n_lat, hi)] = True

    return GroundGrid(drivable=grid, resolution_m=RESOLUTION_M,
                      ego_row=0, ego_col=centre_col)


def agents_at(scn: Scenario, t: float, ego_forward_m: float,
              ego_lateral_m: float, uniform: bool) -> list:
    """Agent positions at time t, relative to the ego, as planner Obstacles."""
    out = []
    for i, a in enumerate(scn.agents):
        # Absolute agent motion, then expressed relative to the ego.
        ax = a["x"] + a["vx"] * t
        ay = a["y"] + a["vy"] * t
        rel_y = ay - (ego_forward_m - scn_start_forward(scn))
        if rel_y < -5.0 or rel_y > scn.observed_range_m:
            continue                      # behind us, or beyond what we can see
        group = a["group"]
        ob = Obstacle(x=ax - ego_lateral_m, y=rel_y,
                      half_width=a["half_width"], half_length=a["half_length"],
                      group=group, vx=a["vx"], vy=a["vy"], track_id=i)
        if uniform:
            # The control: strip the taxonomy's per-group berth and give every
            # obstacle the same one. Implemented by overriding the property so
            # nothing else in the pipeline has to know.
            ob = _UniformObstacle(**{k: getattr(ob, k) for k in
                                     ("x", "y", "half_width", "half_length",
                                      "group", "vx", "vy", "track_id")})
        out.append(ob)
    return out


def scn_start_forward(scn: Scenario) -> float:
    return 0.0


class _UniformObstacle(Obstacle):
    """An obstacle whose berth ignores its group -- the ablation control."""

    @property
    def clearance_m(self) -> float:
        return UNIFORM_CLEARANCE_M


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

@dataclass
class RolloutResult:
    scenario: str
    arm: str
    completed: bool = False
    distance_m: float = 0.0
    steps: int = 0
    refusals: int = 0
    truncations: int = 0
    steps_stopped: int = 0
    min_clearance_by_group: dict = field(default_factory=dict)
    latencies_ms: list = field(default_factory=list)
    lateral_jerk: float = 0.0
    collided_with: list = field(default_factory=list)

    def summary(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {
            "scenario": self.scenario,
            "arm": self.arm,
            "completed": self.completed,
            "distance_m": round(self.distance_m, 2),
            "steps": self.steps,
            "refusals": self.refusals,
            "truncations": self.truncations,
            "steps_stopped": self.steps_stopped,
            "collisions": len(self.collided_with),
            "collided_with": sorted(set(self.collided_with)),
            "min_clearance_by_group": {k: round(v, 3) for k, v in
                                       sorted(self.min_clearance_by_group.items())},
            "latency_ms_mean": round(float(lat.mean()), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "lateral_jerk": round(self.lateral_jerk, 4),
        }


def true_clearance(ego_x: float, ego_forward: float, scn: Scenario,
                   t: float) -> dict:
    """Actual body-to-body gap to each agent, per group. Ground truth.

    Deliberately computed from the scenario's own agent positions, NOT from
    what the planner believed -- otherwise the ablation would only measure
    whether the planner obeyed its own corridor, which it does by construction.
    """
    gaps: dict = {}
    for a in scn.agents:
        ax = a["x"] + a["vx"] * t
        ay = a["y"] + a["vy"] * t
        longitudinal = abs(ay - ego_forward)
        # Only count the lateral gap while the vehicle is actually alongside.
        if longitudinal > a["half_length"] + 2.5:
            continue
        gap = abs(ax - ego_x) - a["half_width"] - EGO_HALF_WIDTH_M
        g = a["group"]
        gaps[g] = min(gaps.get(g, float("inf")), gap)
    return gaps


def run_rollout(scn: Scenario, arm: str, planner: FrenetPlanner,
                dt: float = 0.2, max_steps: int = 150,
                jitter: float = 0.0, rng=None) -> RolloutResult:
    """Drive the scenario once, replanning every step."""
    res = RolloutResult(scenario=scn.name, arm=arm)
    uniform = (arm == "uniform")
    gate = (arm != "no_gate")

    ego_forward, ego_lateral, speed = 0.0, 0.0, scn.target_speed_mps * 0.7
    lateral_history = []

    # Lateral offset from the road centre, CARRIED BETWEEN STEPS. Without
    # this the planner is told d0 = 0 every cycle -- it believes it starts on
    # the centreline no matter where it actually is, so a chosen offset never
    # persists and every plan has to swing back out from the centre, straight
    # through whatever the corridor was avoiding.
    d_offset, d_offset_rate = 0.0, 0.0

    for step in range(max_steps):
        grid = scenario_grid(scn, ego_forward, ego_lateral)
        path = reference_path_from_free_space(grid)
        if path is None:
            # Refusing is not failing. Slowing to a stop and waiting for a
            # pedestrian to finish crossing is what a careful driver does, and
            # ending the episode at that moment would score correct caution as
            # an incomplete route. The vehicle holds and keeps replanning; the
            # situation it refused is usually transient.
            res.refusals += 1
            speed = max(speed - 3.0 * dt, 0.0)
            if speed <= 0.05:
                res.steps_stopped += 1
                speed = 0.0
            ego_forward += speed * dt
            continue

        obstacles = agents_at(scn, step * dt, ego_forward, ego_lateral, uniform)

        # A little noise on measured obstacle position, so the comparison is
        # not decided by perfect information the real system never has.
        if jitter > 0 and rng is not None:
            for ob in obstacles:
                ob.x += rng.normal(0.0, jitter)
                ob.y += rng.normal(0.0, jitter)

        corridor = build_corridor(path, obstacles)

        t0 = time.perf_counter()
        observed = None if gate else float(path.s[-1]) + 1e6
        traj, diag = planner.plan(path, corridor, ego_speed_mps=speed,
                                  target_speed_mps=scn.target_speed_mps,
                                  d0=d_offset, d0_dot=d_offset_rate,
                                  observed_s_max=observed)
        res.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if traj is None:
            # Refusing is not failing. Slowing to a stop and waiting for a
            # pedestrian to finish crossing is what a careful driver does, and
            # ending the episode at that moment would score correct caution as
            # an incomplete route. The vehicle holds and keeps replanning; the
            # situation it refused is usually transient.
            res.refusals += 1
            speed = max(speed - 3.0 * dt, 0.0)
            if speed <= 0.05:
                res.steps_stopped += 1
                speed = 0.0
            ego_forward += speed * dt
            continue

        if traj.truncated:
            res.truncations += 1

        # Advance along the chosen trajectory by one control interval.
        # traj.d is the offset from the reference path, which IS the road
        # centreline -- so the ego's absolute lateral position is the road
        # centre at the new station plus that offset, not the previous
        # position plus it.
        idx = min(int(dt / planner.cfg.dt), len(traj.s) - 1)
        ego_forward += float(traj.s[idx])
        d_offset = float(traj.d[idx])
        d_offset_rate = float(traj.d_dot[idx])
        ego_lateral = scn.road_centre_at(ego_forward) + d_offset
        speed = float(traj.s_dot[idx])
        lateral_history.append(ego_lateral)

        gaps = true_clearance(ego_lateral, ego_forward, scn, step * dt)
        for g, gap in gaps.items():
            res.min_clearance_by_group[g] = min(
                res.min_clearance_by_group.get(g, float("inf")), gap)
            if gap < 0:
                res.collided_with.append(g)

        res.steps = step + 1
        res.distance_m = ego_forward
        if ego_forward >= scn.road_length_m - 2.0:
            res.completed = True
            break

    if len(lateral_history) > 3:
        res.lateral_jerk = float(np.mean(np.abs(np.diff(lateral_history, n=3))))
    return res


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="repeats per scenario per arm, with measurement noise")
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="std-dev of obstacle position noise, metres")
    ap.add_argument("--out", type=Path, default=Path("results/planning.json"))
    args = ap.parse_args()

    planner = FrenetPlanner(PlannerConfig())
    scenarios = build_scenarios()
    arms = ("grouped", "uniform", "no_gate")

    print(f"{len(scenarios)} scenarios x {len(arms)} arms x {args.seeds} seeds "
          f"= {len(scenarios) * len(arms) * args.seeds} rollouts\n")

    rows = []
    for scn in scenarios:
        print(f"--- {scn.name} ---")
        print(f"    {scn.description}")
        for arm in arms:
            runs = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                runs.append(run_rollout(scn, arm, planner, jitter=args.jitter,
                                        rng=rng))
            agg = aggregate(runs)
            agg.update(scenario=scn.name, arm=arm, seeds=args.seeds)
            rows.append(agg)
            print(f"    {arm:<9} complete {agg['completion_rate']:.0%}  "
                  f"collisions {agg['collision_rate']:.0%}  "
                  f"min VULNERABLE gap "
                  f"{agg.get('min_clearance_VULNERABLE', float('nan')):.2f} m  "
                  f"refusals {agg['refusals_mean']:.1f}  "
                  f"p95 {agg['latency_ms_p95']:.1f} ms")
        print()

    report_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


def aggregate(runs: list) -> dict:
    out = {
        "completion_rate": float(np.mean([r.completed for r in runs])),
        "collision_rate": float(np.mean([bool(r.collided_with) for r in runs])),
        "distance_m_mean": float(np.mean([r.distance_m for r in runs])),
        "refusals_mean": float(np.mean([r.refusals for r in runs])),
        "truncations_mean": float(np.mean([r.truncations for r in runs])),
        "steps_stopped_mean": float(np.mean([r.steps_stopped for r in runs])),
        "latency_ms_mean": float(np.mean([np.mean(r.latencies_ms or [0])
                                          for r in runs])),
        "latency_ms_p95": float(np.mean([np.percentile(r.latencies_ms or [0], 95)
                                         for r in runs])),
        "lateral_jerk_mean": float(np.mean([r.lateral_jerk for r in runs])),
    }
    for group in GROUP_BEHAVIOUR:
        vals = [r.min_clearance_by_group[group] for r in runs
                if group in r.min_clearance_by_group]
        if vals:
            out[f"min_clearance_{group}"] = float(np.min(vals))
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in out.items()}


def report_table(rows: list) -> None:
    print("=" * 78)
    print("CLOSED-LOOP PLANNING RESULTS")
    print("=" * 78)
    print(f"  {'scenario':<26}{'arm':<10}{'compl':>7}{'collis':>8}"
          f"{'VULN gap':>10}{'refuse':>8}{'p95 ms':>8}")
    for r in rows:
        gap = r.get("min_clearance_VULNERABLE")
        gap_s = f"{gap:.2f}" if gap is not None else "-"
        print(f"  {r['scenario']:<26}{r['arm']:<10}"
              f"{r['completion_rate']:>6.0%}{r['collision_rate']:>8.0%}"
              f"{gap_s:>10}{r['refusals_mean']:>8.1f}{r['latency_ms_p95']:>8.1f}")

    print("\n  Read the VULNERABLE gap column first: it is the body-to-body")
    print("  distance the vehicle actually left a pedestrian or animal,")
    print("  measured from the scenario's own agent positions rather than from")
    print("  what the planner believed. If 'grouped' does not beat 'uniform'")
    print("  there, the taxonomy is not earning its place and the paper should")
    print("  say so.")


if __name__ == "__main__":
    main()
