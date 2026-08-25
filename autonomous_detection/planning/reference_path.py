"""Synthesise a reference path from drivable free space.

WHY THIS EXISTS
---------------
Frenet-frame planning -- Werling et al., ICRA 2010, and MATLAB's own
`referencePathFrenet` -- needs a reference path as its INPUT. The road
centreline becomes the s axis and everything else is expressed relative to it.

On a marked highway a map or a lane detector supplies that centreline. On an
unmarked Indian village road, nothing does. That is not a tooling limitation;
it is the state of the field, and it is the specific gap this module fills.

The input here is a drivable-space mask in the GROUND PLANE (bird's-eye,
metres), which the existing IPM homography already produces from the
segmentation output. The output is an ordered, smooth polyline that Frenet
planning can use as its s axis.

WHY A SCANLINE MEDIAL AXIS, NOT MORPHOLOGICAL SKELETONISATION
--------------------------------------------------------------
The obvious approach is to skeletonise the mask and call the skeleton the
centreline. It behaves badly here for three reasons:

  * A skeleton is an unordered set of pixels. Frenet needs an ordered path
    starting at the ego vehicle, so the ordering has to be recovered anyway.
  * Skeletons sprout spurious branches from every boundary irregularity, and
    unstructured road edges -- shoulder, dirt, shopfront -- are nothing but
    boundary irregularity.
  * At a junction the skeleton forks, and nothing in it says which fork the
    vehicle is on.

Scanning forward instead gives an ordered path by construction: at each
longitudinal station, take the drivable interval the vehicle is actually in and
use its midpoint. Continuity is then enforceable directly -- the interval at
each station must connect to the one before it -- which resolves the junction
case by simply staying in the branch the vehicle already occupies.

WHERE THE PATH STOPS
--------------------
Deliberately, at the first station with no connected drivable interval. That
truncation is not a failure to report; it is the observed extent of the road,
and the planner must not commit beyond it. `ReferencePath.truncated_reason`
records why it ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.interpolate import splev, splprep
    SCIPY = True
except ImportError:                                    # pragma: no cover
    SCIPY = False


@dataclass
class GroundGrid:
    """A bird's-eye occupancy grid in metres, with the ego at a known cell.

    Rows index forward distance, columns index lateral offset. This is the
    frame the IPM homography already delivers, so nothing upstream changes.
    """
    drivable: np.ndarray          # (n_forward, n_lateral) bool
    resolution_m: float = 0.2     # metres per cell, both axes
    ego_row: int = 0              # row of the ego vehicle (0 = nearest)
    ego_col: Optional[int] = None  # defaults to the grid's centre column

    def __post_init__(self):
        self.drivable = np.asarray(self.drivable, dtype=bool)
        if self.drivable.ndim != 2:
            raise ValueError("drivable must be a 2-D grid")
        if self.ego_col is None:
            self.ego_col = self.drivable.shape[1] // 2

    @property
    def n_forward(self) -> int:
        return self.drivable.shape[0]

    @property
    def n_lateral(self) -> int:
        return self.drivable.shape[1]

    def forward_m(self, row: int) -> float:
        return (row - self.ego_row) * self.resolution_m

    def lateral_m(self, col: float) -> float:
        return (col - self.ego_col) * self.resolution_m


@dataclass
class ReferencePath:
    """An ordered reference path in ego-centred metres, plus its provenance.

    `source` and `truncated_reason` exist because a planner that cannot say
    where its reference came from, or why it stops, cannot be audited -- and an
    unaudited reference is exactly how a vehicle commits to a path through
    space nothing ever observed.
    """
    s: np.ndarray                       # arc length along the path, metres
    x: np.ndarray                       # lateral offset from ego, metres
    y: np.ndarray                       # forward distance from ego, metres
    half_width: np.ndarray              # drivable half-width at each point, m
    source: str = "free_space"          # "free_space" | "lane" | "fused"
    truncated_reason: str = ""
    observed_length_m: float = 0.0

    def __len__(self) -> int:
        return len(self.s)

    @property
    def points(self) -> np.ndarray:
        """(N, 2) array of (lateral, forward) points in metres."""
        return np.column_stack([self.x, self.y])

    def heading(self) -> np.ndarray:
        """Path heading at each point, radians, measured from the forward axis."""
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        return np.arctan2(dx, dy)

    def curvature(self) -> np.ndarray:
        """Signed curvature at each point, 1/m."""
        dx, dy = np.gradient(self.x), np.gradient(self.y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = (dx ** 2 + dy ** 2) ** 1.5
        return np.where(denom > 1e-9, (dx * ddy - dy * ddx) / np.maximum(denom, 1e-9), 0.0)

    def min_radius_m(self) -> float:
        """Tightest radius along the path. Large means effectively straight."""
        k = np.abs(self.curvature())
        peak = float(k.max()) if len(k) else 0.0
        return 1.0 / peak if peak > 1e-6 else float("inf")

    def to_frenet(self, x, y):
        """Cartesian (lateral, forward) in metres -> Frenet (s, d).

        s is arc length along the reference; d is signed lateral offset, with
        positive to the right of the direction of travel. Obstacles have to
        live in this frame before the corridor can be expressed as simple
        lateral bounds at each station.

        Nearest-point projection onto the polyline. The path is smooth and
        densely sampled, so a nearest-vertex search followed by a segment
        projection is both accurate and cheap -- no iterative solve.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        px, py = self.x, self.y

        d2 = (px[None, :] - x[:, None]) ** 2 + (py[None, :] - y[:, None]) ** 2
        idx = np.argmin(d2, axis=1)

        s_out = np.empty(len(x))
        d_out = np.empty(len(x))
        for k, i in enumerate(idx):
            j = min(max(i, 1), len(px) - 1)
            # Tangent from the segment ending at the nearest vertex.
            tx, ty = px[j] - px[j - 1], py[j] - py[j - 1]
            norm = np.hypot(tx, ty)
            if norm < 1e-9:
                s_out[k], d_out[k] = self.s[i], 0.0
                continue
            tx, ty = tx / norm, ty / norm
            vx, vy = x[k] - px[j - 1], y[k] - py[j - 1]
            along = vx * tx + vy * ty
            s_out[k] = self.s[j - 1] + np.clip(along, 0.0, norm)
            # Signed lateral: cross product of tangent with the offset vector.
            d_out[k] = -(tx * vy - ty * vx)
        return s_out, d_out

    def half_width_at(self, s):
        """Drivable half-width at arc length s, interpolated."""
        return np.interp(s, self.s, self.half_width)


# --------------------------------------------------------------------------
# scanline extraction
# --------------------------------------------------------------------------

def _intervals(row: np.ndarray) -> list:
    """Contiguous runs of True in a 1-D boolean row, as (start, end_exclusive)."""
    if not row.any():
        return []
    padded = np.concatenate([[False], row, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _pick_interval(intervals: list, prev_centre: float,
                   max_jump_cells: float) -> Optional[tuple]:
    """Choose the interval continuous with the previous station's centre.

    Preference order: the interval CONTAINING the previous centre, then the
    nearest one within `max_jump_cells`. Beyond that the road has not
    continued -- it has been replaced by a different piece of drivable space,
    and following it would mean planning across a gap the vehicle cannot cross.

    This is what resolves a junction: the branch the vehicle already occupies
    contains the previous centre, so it wins without any junction-specific
    logic.
    """
    if not intervals:
        return None
    for a, b in intervals:
        if a <= prev_centre < b:
            return (a, b)
    best, best_gap = None, float("inf")
    for a, b in intervals:
        centre = (a + b - 1) / 2.0
        gap = abs(centre - prev_centre)
        if gap < best_gap:
            best, best_gap = (a, b), gap
    return best if best_gap <= max_jump_cells else None


def extract_centreline(grid: GroundGrid, max_lateral_jump_m: float = 1.5,
                       min_width_m: float = 2.0):
    """Scan forward, taking the midpoint of the connected drivable interval.

    Returns (rows, centre_cols, half_widths_m, reason). `reason` is empty if the
    scan reached the top of the grid, otherwise it says what stopped it.
    """
    min_width_cells = max(min_width_m / grid.resolution_m, 1.0)
    max_jump_cells = max_lateral_jump_m / grid.resolution_m

    rows, centres, half_widths = [], [], []
    prev_centre = float(grid.ego_col)
    reason = ""

    for row in range(grid.ego_row, grid.n_forward):
        chosen = _pick_interval(_intervals(grid.drivable[row]),
                                prev_centre, max_jump_cells)
        if chosen is None:
            reason = (f"no connected drivable space at "
                      f"{grid.forward_m(row):.1f} m ahead")
            break
        a, b = chosen
        # An interval touching the grid edge has been CUT by the field of view,
        # not by the road. Its midpoint is therefore biased toward the grid
        # centre -- and following that midpoint would steer the vehicle back
        # across a road that is actually leaving the view. The error is in the
        # unsafe direction, so the scan stops here: the honest statement is
        # "the road left my field of view", not a guess at where it went.
        if a == 0 or b == grid.n_lateral:
            reason = (f"drivable space reaches the edge of the observed area at "
                      f"{grid.forward_m(row):.1f} m ahead; the road continues "
                      f"outside the field of view")
            break

        width_cells = b - a
        if width_cells < min_width_cells:
            reason = (f"drivable corridor narrows to "
                      f"{width_cells * grid.resolution_m:.1f} m at "
                      f"{grid.forward_m(row):.1f} m ahead")
            break
        centre = (a + b - 1) / 2.0
        rows.append(row)
        centres.append(centre)
        half_widths.append(width_cells * grid.resolution_m / 2.0)
        prev_centre = centre

    return (np.asarray(rows, dtype=int), np.asarray(centres, dtype=float),
            np.asarray(half_widths, dtype=float), reason)


def _smooth(x: np.ndarray, y: np.ndarray, smoothing: float, n_out: int):
    """Fit a smoothing spline through the raw centreline.

    The raw scanline centres are quantised to the grid and step laterally by
    whole cells, which would show up as curvature spikes and make the path
    unusable for a curvature-limited planner. Falls back to a moving average
    when SciPy is unavailable or the spline fit degenerates, so the module
    still returns a usable path rather than failing.
    """
    if len(x) < 4:
        return x, y
    if SCIPY:
        try:
            tck, _ = splprep([x, y], s=smoothing * len(x), k=min(3, len(x) - 1))
            u = np.linspace(0, 1, n_out)
            xs, ys = splev(u, tck)
            return np.asarray(xs), np.asarray(ys)
        except (ValueError, TypeError):
            pass
    k = min(9, len(x) // 2 * 2 + 1)
    kernel = np.ones(k) / k
    pad = k // 2
    xs = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")
    ys = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")
    return xs, ys


def reference_path_from_free_space(grid: GroundGrid,
                                   max_lateral_jump_m: float = 1.5,
                                   min_width_m: float = 2.0,
                                   smoothing: float = 0.05,
                                   n_points: int = 100) -> Optional[ReferencePath]:
    """Drivable-space mask -> the reference path Frenet planning needs.

    Returns None when no usable stretch of road is found ahead, which is a
    legitimate outcome the caller must handle by stopping rather than by
    planning into space it cannot see.
    """
    rows, centres, half_widths, reason = extract_centreline(
        grid, max_lateral_jump_m, min_width_m)

    if len(rows) < 2:
        return None

    x_raw = np.array([grid.lateral_m(c) for c in centres])
    y_raw = np.array([grid.forward_m(r) for r in rows])
    observed = float(y_raw[-1] - y_raw[0])

    n_out = max(int(n_points), len(x_raw))
    x, y = _smooth(x_raw, y_raw, smoothing, n_out)

    # Half-width is resampled onto the smoothed path by forward distance, so
    # the corridor builder can ask "how much room is there here?" at any station.
    hw = np.interp(y, y_raw, half_widths) if len(half_widths) else np.zeros_like(y)

    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    return ReferencePath(s=s, x=x, y=y, half_width=hw, source="free_space",
                         truncated_reason=reason, observed_length_m=observed)


# --------------------------------------------------------------------------
# using lane markings when they exist
# --------------------------------------------------------------------------

def reference_path_from_lanes(left_px, right_px, ipm,
                              n_points: int = 100) -> Optional[ReferencePath]:
    """Reference path from a detected lane pair, projected to the ground plane.

    Where paint exists this is the more reliable source, and the free-space
    centreline becomes a cross-check rather than the primary. Both are kept
    because a road on which the two disagree is a road to be cautious on --
    see `agreement_m`.
    """
    if left_px is None or right_px is None:
        return None
    left = ipm.pixel_to_ground(np.asarray(left_px, dtype=float))
    right = ipm.pixel_to_ground(np.asarray(right_px, dtype=float))
    if len(left) < 2 or len(right) < 2:
        return None

    y_lo = max(left[:, 1].min(), right[:, 1].min())
    y_hi = min(left[:, 1].max(), right[:, 1].max())
    if y_hi - y_lo < 2.0:
        return None

    y = np.linspace(y_lo, y_hi, n_points)
    lx = np.interp(y, left[:, 1], left[:, 0])
    rx = np.interp(y, right[:, 1], right[:, 0])
    x = (lx + rx) / 2.0
    hw = np.abs(rx - lx) / 2.0

    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return ReferencePath(s=s, x=x, y=y, half_width=hw, source="lane",
                         observed_length_m=float(y_hi - y_lo))


def agreement_m(a: ReferencePath, b: ReferencePath) -> Optional[float]:
    """Mean lateral disagreement between two references over their overlap.

    Two sources that disagree are evidence about the road, not a nuisance to
    be averaged away: it usually means the paint no longer describes where
    vehicles can actually drive. The caller should widen its margins, and the
    number is what lets it decide by how much.
    """
    lo = max(a.y.min(), b.y.min())
    hi = min(a.y.max(), b.y.max())
    if hi - lo < 1.0:
        return None
    y = np.linspace(lo, hi, 50)
    return float(np.mean(np.abs(np.interp(y, a.y, a.x) - np.interp(y, b.y, b.x))))
