"""Drivable-area segmentation — fallback (or primary) signal for roads where
lane-LINE detection fundamentally doesn't apply.

WHY THIS EXISTS
----------------
CLRNet and UFLDv2 (models/lane_detector.py) are trained on CULane: Chinese
urban/highway roads with continuous, clearly painted lane markings. Their
entire output representation — polylines fit through painted-line pixels —
assumes those lines exist and are visible. On Indian roads this assumption
frequently fails: markings are faded, absent on rural/semi-urban stretches,
or simply not followed (vehicles use the full road width). No amount of
retraining CLRNet fixes this, because the ground truth it's built around
(a continuous painted curve) often doesn't exist in the scene to detect.

Every independent group that has tackled Indian/unstructured roads converges
on the same reframing: stop asking "where are the lane lines" and ask
"which part of the image is drivable road surface" instead — a strictly
easier, better-defined question that degrades gracefully when markings are
absent. This is exactly why the IDD dataset's segmentation labels include
"drivable fallback" / "non-drivable fallback" classes instead of only lane
lines (arxiv.org/abs/1811.10200), and why BDD100K ships a dedicated
"drivable area" segmentation task. Several Indian-roads GitHub projects use
the same reframing directly:
    github.com/moatifbutt/Drivable-Road-Region-Detection-and-Steering-Angle-Estimation-Method
    github.com/AbhayVAshokan/Semantic-Segmentation-of-Road-Surface  (IDD-based)
    github.com/balnarendrasapa/road-detection

HOW IT PLUGS IN
-----------------
segment_to_lane_result() returns a `LaneResult` — the SAME type
models/lane_detector.py already produces — so inference/collision.py and
inference/overtaking.py need zero changes to consume it: adas_final.py just
falls back to this when CLRNet/UFLDv2 found fewer than 2 confident lane
lines (the common case on unmarked Indian roads), and every downstream
consumer (`_in_ego_path`, `_center_divider`, IPM curvature) works exactly as
if real lane lines had been found — because a drivable-area boundary IS a
lane boundary, just extracted from surface segmentation instead of painted
paint.

NOTE ON OVERTAKING LEGALITY: this deliberately does NOT make the overtaking
module more permissive. Without real lane-type info (solid/dashed), Rule 1
in overtaking.py still fails safe to NOT_POSSIBLE_SOLID_LINE/NO_LANE_INFO —
crossing into oncoming traffic on an unmarked road is a genuinely higher-risk
judgment call this project intentionally does not make. What this DOES fix
is collision-alert ego-path gating (`_in_ego_path`), which today falls back
to a crude "central 40% of the frame" guess when no lane info exists — a
guess that's frequently wrong on Indian roads' wide, undivided, or curved
carriageways and is a real contributor to the "fake" collision alerts firing
on objects that aren't actually in the vehicle's path.

TRAINING (optional — a CV-only fallback works with zero weights)
--------------------------------------------------------------------
Fine-tune DrivableNet on IDD's segmentation "drivable fallback" mask class:
    python training/train_drivable_area.py \
        --data data/IDD_Segmentation --out weights/drivable_area.pth
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models.lane_detector import LaneResult

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DrivableNet(nn.Module if TORCH_AVAILABLE else object):
    """Small encoder-decoder for binary drivable-surface segmentation.
    Deliberately lightweight (same complexity class as night_enhance.py's
    _DCENet) — this only needs to separate "road" from "not road", not
    perform fine-grained multi-class segmentation."""

    def __init__(self, base_ch: int = 24):
        super().__init__()
        c = base_ch

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.enc1 = block(3, c)
        self.enc2 = block(c, c * 2, stride=2)
        self.enc3 = block(c * 2, c * 4, stride=2)
        self.enc4 = block(c * 4, c * 8, stride=2)
        self.dec3 = block(c * 8 + c * 4, c * 4)
        self.dec2 = block(c * 4 + c * 2, c * 2)
        self.dec1 = block(c * 2 + c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        u3 = nn.functional.interpolate(e4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = nn.functional.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = nn.functional.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return self.out(d1)   # logits, upsample to full res by caller


@dataclass
class DrivableAreaSegmenter:
    """Binary drivable-surface segmentation with a classical-CV fallback
    (same graceful-degradation philosophy as NightEnhancer / lane_detector)."""

    weights: Optional[str] = None
    device: str = "cuda"
    input_size: tuple = (512, 256)   # (w, h) — small, this runs every frame

    def __post_init__(self):
        self.model = None
        if TORCH_AVAILABLE:
            self.device = self.device if torch.cuda.is_available() else "cpu"
            if self.weights and Path(self.weights).exists():
                self.model = DrivableNet().to(self.device).eval()
                try:
                    state = torch.load(self.weights, map_location=self.device)
                    self.model.load_state_dict(state)
                    print(f"[DrivableArea] Loaded trained weights: {self.weights}")
                except (RuntimeError, KeyError) as e:
                    print(f"[DrivableArea] Weights failed to load ({e}); "
                          "using classical CV fallback")
                    self.model = None
            else:
                print("[DrivableArea] No trained weights — using classical "
                      "CV road-color fallback (works, just less accurate on "
                      "shadows/glare). Train with training/train_drivable_area.py "
                      "when IDD segmentation data is available.")

    # ---------------- mask extraction ----------------

    def segment_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns an HxW uint8 binary mask (255 = drivable surface)."""
        if self.model is not None:
            return self._segment_dl(frame_bgr)
        return self._segment_classical(frame_bgr)

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def _segment_dl(self, frame_bgr: np.ndarray) -> np.ndarray:
        h0, w0 = frame_bgr.shape[:2]
        w, h = self.input_size
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
        t = (torch.from_numpy(rgb.astype(np.float32) / 255.0)
             .permute(2, 0, 1).unsqueeze(0).to(self.device))
        logits = self.model(t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (cv2.resize(prob, (w0, h0)) > 0.5).astype(np.uint8) * 255
        return mask

    def _segment_classical(self, frame_bgr: np.ndarray) -> np.ndarray:
        """No-training fallback: model the road surface's own color from a
        seed strip directly in front of the vehicle (bottom-center of frame
        — almost always road, regardless of country/markings), then grow a
        similarity mask from that seed. Robust to "no lane paint at all"
        since it never looks for lines in the first place."""
        h, w = frame_bgr.shape[:2]
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        seed_y0, seed_y1 = int(h * 0.85), h
        seed_x0, seed_x1 = int(w * 0.40), int(w * 0.60)
        seed = lab[seed_y0:seed_y1, seed_x0:seed_x1].reshape(-1, 3)
        if seed.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        mean, std = seed.mean(axis=0), seed.std(axis=0) + 1e-3

        # Only search the lower ~65% of the frame (road never appears in sky)
        # for pixels within an adaptive color distance of the seed.
        search_y0 = int(h * 0.35)
        region = lab[search_y0:, :, :].astype(np.float32)
        dist = np.sqrt((((region - mean) / (std * 3.0)) ** 2).sum(axis=2))
        raw_mask = (dist < 1.0).astype(np.uint8) * 255

        # Keep only the connected component touching the seed strip — avoids
        # merging with unrelated same-colored regions elsewhere in frame
        # (e.g. a distant building or sidewalk).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        n_labels, labels = cv2.connectedComponents(raw_mask)
        seed_label = labels[int((seed_y0 + seed_y1) / 2) - search_y0,
                             int((seed_x0 + seed_x1) / 2)]
        component = (labels == seed_label).astype(np.uint8) * 255 if seed_label != 0 \
            else np.zeros_like(raw_mask)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[search_y0:, :] = component
        return mask

    # ---------------- corridor extraction (-> reuse LaneResult machinery) ----------------

    def segment_to_lane_result(self, frame_bgr: np.ndarray, num_rows: int = 16
                                ) -> Optional[LaneResult]:
        """Drivable mask -> synthetic left/right boundary polylines, packaged
        as the SAME LaneResult type models/lane_detector.py produces. This is
        what lets collision.py/overtaking.py consume drivable-area output
        with no code changes — see module docstring."""
        h, w = frame_bgr.shape[:2]
        mask = self.segment_mask(frame_bgr)
        seed_x = w // 2

        rows = np.linspace(h - 1, int(h * 0.35), num_rows).astype(int)
        left_pts, right_pts = [], []
        for y in rows:
            row = mask[y] > 0
            if not row[seed_x]:
                continue   # seed column itself isn't drivable at this row -> skip
            # Scan outward from the seed column to find the corridor edges.
            lx = seed_x
            while lx > 0 and row[lx - 1]:
                lx -= 1
            rx = seed_x
            while rx < w - 1 and row[rx + 1]:
                rx += 1
            if rx - lx < w * 0.05:      # too narrow to be a real corridor reading
                continue
            left_pts.append((float(lx), float(y)))
            right_pts.append((float(rx), float(y)))

        if len(left_pts) < 3 or len(right_pts) < 3:
            return None   # couldn't find a usable drivable corridor this frame

        left_arr = np.asarray(left_pts, dtype=np.float32)
        right_arr = np.asarray(right_pts, dtype=np.float32)
        result = LaneResult(polylines=[left_arr, right_arr],
                             ego_lane={"left": left_arr, "right": right_arr})
        return result


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read {path}")

    seg = DrivableAreaSegmenter()
    mask = seg.segment_mask(img)
    lanes = seg.segment_to_lane_result(img)

    overlay = img.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    if lanes is not None:
        for pl, color in zip(lanes.polylines, [(0, 0, 255), (255, 0, 0)]):
            pts = pl.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, color, 3)
    cv2.imwrite("drivable_area_output.jpg", overlay)
    print(f"Corridor found: {lanes is not None}")
    print("Saved: drivable_area_output.jpg")
