"""Geometric size-consistency filter — catches detections on things that
aren't actually there ("phantom"/hallucinated boxes), as opposed to
detections that ARE a real object but the wrong class (that's the domain-
gap problem covered in KRISH_HANDOVER.md PLAN C).

WHY THIS EXISTS
----------------
A detector shown out-of-distribution scenes (which Indian street footage is,
relative to KITTI/COCO training data — see PLAN C) doesn't just misclassify
real objects, it sometimes fires on things that aren't objects at all: a
car-shaped billboard/hoarding, a shop-front poster, a reflection in a wet
road or a glass storefront, glare. This is a well-known failure mode in the
autonomous-driving safety literature — sometimes literally called "hallucin-
ations" or "phantom objects" (see e.g. the PhantomPerception hallucination-
injection safety-evaluation work, arxiv.org/html/2510.07749v1) — and the
two standard mitigations are (1) stricter confidence calibration and (2)
geometric consistency checks.

THIS MODULE implements (2): a real car and a car-shaped billboard sitting at
the SAME measured depth produce very different apparent sizes, because the
billboard's "car" is printed art on a large flat surface, not an actual
car-sized 3D object. Monocular 3D detection research calls this "geometric
consistency" — detectors are known to be vulnerable exactly when apparent
size and estimated depth don't agree (see "Exploring Geometric Consistency
for Monocular 3D Object Detection", arxiv.org/abs/2104.05858). Given the
depth map this pipeline already computes for collision/TTC, the check is
just the pinhole camera equation:

    real_world_width_m = bbox_width_px * depth_m / focal_length_px

...compared against a generously wide plausible range for the detected
class. This ONLY rejects detections that are wildly (multiple times) too
big or too small for their measured distance — it is not a precise size
estimator, it's a sanity check to catch the clearly-impossible cases
(flat printed imagery, distant background structures, reflections).

USAGE
------
    from inference.sanity_filter import SizeConsistencyFilter
    filt = SizeConsistencyFilter(focal_length_px=721.5)  # from KITTI calib,
                                                          # or estimate_focal_length_px()
    kept = filt.filter(detections, depth_map)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Plausible real-world WIDTH range (meters) per class in the unified 15-class
# taxonomy (see data/prepare_merged.py). Deliberately wide — this only
# rejects detections outside an even-more-generous margin (see `margin`
# below), so it targets "impossible", not "unusual".
CLASS_WIDTH_RANGE_M = {
    "car": (1.3, 2.3),
    "van": (1.5, 2.6),
    "truck": (1.8, 3.2),
    "bus": (2.0, 3.2),
    "tram": (2.2, 3.4),
    "motorcycle": (0.5, 1.1),
    "cyclist": (0.4, 1.0),
    "bicycle": (0.4, 1.0),
    "pedestrian": (0.25, 0.9),
    "autorickshaw": (1.1, 1.9),
    "rider": (0.4, 1.0),
    # animal / vehicle_fallback / traffic_light / traffic_sign / misc are
    # deliberately NOT bounded here — too variable (a dog vs. a buffalo; a
    # hand-cart vs. a tractor) to set a meaningful range without rejecting
    # real objects. They pass through unchecked.
}


@dataclass
class SizeConsistencyFilter:
    focal_length_px: Optional[float] = None
    margin: float = 1.6   # extra multiplicative slack beyond the table's own range
    min_depth_m: float = 1.0   # ignore very-close objects (depth noisy, box often clipped)
    focal_length_is_estimated: bool = True
    # Extra slack applied ON TOP of `margin` when the focal length is a
    # field-of-view GUESS rather than real calibration.
    #
    # Why this exists: on the first real cluster run this filter rejected
    # 3961 detections including 1130 "van" and 1039 "car" — clearly real
    # objects, not phantoms. Cause: with no calibration it assumes ~90° HFOV,
    # but a typical forward/dashcam camera is narrower (KITTI is ~81°, many
    # dashcams ~70°). Assuming too WIDE an FOV underestimates focal length,
    # which inflates every implied real-world width and over-rejects.
    #
    # The principled response to an uncertain input is a WIDER acceptance
    # band, not a tighter one — and the asymmetry matters: dropping a real
    # vehicle is worse than letting a phantom through, because phantoms are
    # still gated by the ego-path check before they can raise a collision
    # alert. Pass --kitti_calib (or any real calibration) to get the tight
    # band; this slack then disappears automatically.
    estimated_focal_extra_margin: float = 1.8
    rejected_count: dict = field(default_factory=dict, repr=False)
    seen_count: int = 0          # total detections examined (for the reject ratio)
    checked_count: int = 0       # detections actually size-checked

    @property
    def effective_margin(self) -> float:
        return (self.margin * self.estimated_focal_extra_margin
                if self.focal_length_is_estimated else self.margin)

    def filter(self, detections: list, depth_map: Optional[np.ndarray]) -> list:
        """Returns the subset of `detections` that pass the size-consistency
        check. Detections in unbounded classes, or when focal length/depth
        aren't available, pass through untouched (graceful degradation —
        matches the rest of this codebase's philosophy: missing info means
        "can't check", not "reject")."""
        self.seen_count += len(detections)
        if self.focal_length_px is None or depth_map is None:
            return detections

        margin = self.effective_margin
        kept = []
        for d in detections:
            name = getattr(d, "class_name", "")
            rng = CLASS_WIDTH_RANGE_M.get(name)
            if rng is None:
                kept.append(d)
                continue

            depth_m = self._object_depth(d.bbox, depth_map)
            if depth_m is None or depth_m < self.min_depth_m:
                kept.append(d)   # no reliable depth read -- don't reject on missing info
                continue

            x1, _, x2, _ = d.bbox
            bbox_w_px = max(x2 - x1, 1e-6)
            implied_width_m = bbox_w_px * depth_m / self.focal_length_px
            self.checked_count += 1

            lo, hi = rng[0] / margin, rng[1] * margin
            if lo <= implied_width_m <= hi:
                kept.append(d)
            else:
                self.rejected_count[name] = self.rejected_count.get(name, 0) + 1

        return kept

    @staticmethod
    def _object_depth(bbox, depth_map: np.ndarray) -> Optional[float]:
        """Median depth over the lower half of the bbox — same convention as
        inference/collision.py's _object_distance (lower half = where the
        object actually meets the road; median resists outlier pixels)."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y_mid = max(0, min(h, (y1 + y2) // 2))
        y2c = max(0, min(h, y2))
        if x2 <= x1 or y2c <= y_mid:
            return None
        region = depth_map[y_mid:y2c, x1:x2]
        return float(np.median(region)) if region.size else None
