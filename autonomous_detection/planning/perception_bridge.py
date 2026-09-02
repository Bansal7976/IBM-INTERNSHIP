"""The join between perception and planning.

WHAT THIS SOLVES
----------------
Perception works in IMAGE space: bounding boxes in pixels, a drivable mask in
pixels, a depth map in pixels. Planning works in the GROUND PLANE: metres
forward and metres laterally, because a corridor width of 1.5 m means something
and a corridor width of 200 pixels does not.

Everything needed to cross that gap already existed separately -- the IPM
homography, the drivable-area segmenter, the tracker, the decision taxonomy --
but nothing joined them, so the planner could only be run on synthetic grids.
This module is that join, kept separate from the pipeline so it can be tested
on its own.

TWO CONVERSIONS, AND WHY EACH IS DONE THE WAY IT IS
----------------------------------------------------
**Drivable mask -> bird's-eye grid.** Backward warp: for each ground cell, work
out which pixel it corresponds to and sample there. Warping forward instead --
pixel by pixel into the grid -- leaves holes at range, because distant pixels
spread far apart on the ground and the gaps between them read as "not
drivable". A hole in the middle of the road would truncate the reference path
and stop the vehicle for nothing.

**Box -> ground position.** From the bottom-centre of the box, not from the
depth map. The bottom edge of a box is where the object meets the road, and the
homography maps that point exactly. Depth gives distance but not lateral
offset, and its error grows with range; the ground-contact point does not have
that problem. Depth is still used to sanity-check the result, because a box
whose bottom edge is above the horizon projects to nonsense.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taxonomy import NAME_TO_GROUP, normalise           # noqa: E402
from planning.corridor import Obstacle                          # noqa: E402
from planning.reference_path import GroundGrid                  # noqa: E402

# Physical half-extents per decision group, metres. Used when the detector
# gives a class but no metric size. Deliberately coarse -- the corridor adds
# the group's clearance on top, and that is the number doing the real work.
GROUP_SIZE_M = {
    "VULNERABLE":      (0.35, 0.35),
    "TWO_WHEELER":     (0.40, 1.00),
    "THREE_WHEELER":   (0.75, 1.50),
    "LIGHT_VEHICLE":   (0.90, 2.20),
    "HEAVY_VEHICLE":   (1.30, 5.00),
    "STATIC_OBSTACLE": (0.60, 0.60),
}


def ground_grid_from_mask(mask, ipm, forward_range_m=(2.0, 50.0),
                          lateral_range_m=(-12.0, 12.0),
                          resolution_m: float = 0.2) -> Optional[GroundGrid]:
    """Image-space drivable mask -> bird's-eye occupancy grid in metres.

    Returns None when the homography is unavailable, which the caller must
    treat as "no planning this frame" rather than as an empty road.
    """
    if mask is None or ipm is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    h, w = mask.shape[:2]

    y_vals = np.arange(forward_range_m[0], forward_range_m[1], resolution_m)
    x_vals = np.arange(lateral_range_m[0], lateral_range_m[1], resolution_m)
    if len(y_vals) < 2 or len(x_vals) < 2:
        return None

    xx, yy = np.meshgrid(x_vals, y_vals)          # (n_forward, n_lateral)
    ground = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)

    try:
        px = ground @ np.linalg.inv(ipm.H).T
    except np.linalg.LinAlgError:
        return None
    wcoord = px[:, 2:3]
    # A near-zero homogeneous coordinate means the ground point maps to the
    # horizon or behind the camera; those cells are not observable.
    valid = np.abs(wcoord[:, 0]) > 1e-9
    uv = np.zeros((len(px), 2))
    uv[valid] = px[valid, :2] / wcoord[valid]

    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    drivable = np.zeros(len(px), dtype=bool)
    drivable[inside] = mask[v[inside], u[inside]] > 0
    drivable = drivable.reshape(xx.shape)

    ego_col = int(np.argmin(np.abs(x_vals)))
    grid = GroundGrid(drivable=drivable, resolution_m=resolution_m,
                      ego_row=0, ego_col=ego_col)
    # Forward offset is baked into the grid: row 0 is forward_range_m[0], not
    # the vehicle itself. Recorded so the caller can report absolute distances.
    grid.forward_offset_m = float(forward_range_m[0])
    return grid


def obstacles_from_tracks(tracks, ipm, depth_map=None, frame_shape=None,
                          max_range_m: float = 60.0) -> list:
    """Tracked boxes -> ground-plane obstacles carrying their decision group.

    Tracks whose ground position cannot be resolved are DROPPED and the caller
    is told how many, rather than being placed at a guessed position. A phantom
    obstacle in the corridor stops the vehicle for nothing; an obstacle at the
    wrong place is worse than one the planner never saw, because the collision
    layer is still watching the real one.
    """
    out, dropped = [], 0
    if not tracks or ipm is None:
        return out

    for t in tracks:
        bbox = getattr(t, "bbox", None)
        if bbox is None:
            dropped += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        contact = np.array([[(x1 + x2) / 2.0, y2]], dtype=float)

        try:
            ground = ipm.pixel_to_ground(contact)
        except (np.linalg.LinAlgError, ValueError):
            dropped += 1
            continue
        gx, gy = float(ground[0, 0]), float(ground[0, 1])

        # Behind the camera, on the horizon, or implausibly far: the projection
        # has failed, not found a distant object.
        if not np.isfinite(gx) or not np.isfinite(gy) or gy <= 0 or gy > max_range_m:
            dropped += 1
            continue

        name = getattr(t, "class_name", "") or ""
        group = (name if name in GROUP_SIZE_M
                 else NAME_TO_GROUP.get(normalise(name), "LIGHT_VEHICLE"))
        half_w, half_l = GROUP_SIZE_M.get(group, GROUP_SIZE_M["LIGHT_VEHICLE"])

        # Closing speed, where the collision layer has measured one. Positive
        # depth_speed_mps means the gap is shrinking, so relative to the ground
        # the object is moving toward us: negative forward velocity.
        closing = float(getattr(t, "depth_speed_mps", 0.0))

        out.append(Obstacle(x=gx, y=gy, half_width=half_w, half_length=half_l,
                            group=group, vx=0.0, vy=-closing,
                            track_id=getattr(t, "track_id", None)))

    obstacles_from_tracks.last_dropped = dropped
    return out


obstacles_from_tracks.last_dropped = 0


def lane_mask_from_polylines(lanes, frame_shape, thickness: int = 3):
    """Fill between the two innermost lane polylines to get a drivable mask.

    Used when lane detection succeeds but no segmentation model is loaded, so
    the planner still has free space to work from. Narrower than the true
    drivable area -- it is the marked lane, not everything the vehicle could
    use -- which is the conservative direction.
    """
    import cv2

    polylines = getattr(lanes, "polylines", lanes)
    if polylines is None or len(polylines) < 2:
        return None
    h, w = frame_shape[:2]
    ordered = sorted(polylines, key=lambda p: np.asarray(p)[:, 0].mean())
    mid = len(ordered) // 2
    left = np.asarray(ordered[max(0, mid - 1)], dtype=np.int32)
    right = np.asarray(ordered[min(mid, len(ordered) - 1)], dtype=np.int32)
    if len(left) < 2 or len(right) < 2:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.vstack([left, right[::-1]]).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], 255)
    return mask
