"""Lane detection wrappers: CLRNet (accuracy, curves) + UFLDv2 (speed).

CLRNet — best CULane F1 on the Curve category (mountain roads) and Night.
    git clone https://github.com/Turoad/CLRNet.git external/CLRNet
UFLDv2 — 300+ FPS fallback when the latency budget is tight.
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 external/UFLDv2

Both wrappers return a LaneResult with:
    .polylines  - list of (N,2) float arrays [(x,y), ...] in image coords
    .ego_lane   - {"left": polyline, "right": polyline} boundaries of our lane
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class LaneResult:
    polylines: list = field(default_factory=list)   # list of (N,2) arrays
    ego_lane: dict = field(default_factory=dict)    # {"left": ..., "right": ...}

    def assign_ego_lane(self, frame_width: int):
        """Ego lane = pair of lines straddling the image center at the bottom."""
        if len(self.polylines) < 2:
            return
        cx = frame_width / 2.0

        def bottom_x(pl):
            pts = np.asarray(pl)
            return pts[pts[:, 1].argmax(), 0]

        left = [p for p in self.polylines if bottom_x(p) < cx]
        right = [p for p in self.polylines if bottom_x(p) >= cx]
        if left:
            self.ego_lane["left"] = max(left, key=bottom_x)      # closest from left
        if right:
            self.ego_lane["right"] = min(right, key=bottom_x)    # closest from right


class CLRNetWrapper:
    """CLRNet inference wrapper (primary lane detector)."""

    def __init__(self, weights: str, config: str | None = None, device: str = "cuda"):
        repo = PROJECT_ROOT / "external" / "CLRNet"
        sys.path.insert(0, str(repo))
        import torch
        # Patch mmcv 2.x to expose the 1.x API that CLRNet expects
        from models import mmcv_compat  # noqa: F401  (side-effects only)
        from clrnet.models.registry import build_net
        from clrnet.utils.config import Config

        config = config or str(repo / "configs/clrnet/clr_resnet101_culane.py")
        self.cfg = Config.fromfile(config)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = build_net(self.cfg).to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        raw = state["net"] if "net" in state else state
        # Strip DataParallel 'module.' prefix saved by multi-GPU training
        if any(k.startswith("module.") for k in raw):
            raw = {k[len("module."):]: v for k, v in raw.items()}
        self.net.load_state_dict(raw, strict=False)
        self.input_w = self.cfg.img_w
        self.input_h = self.cfg.img_h
        self.cut_height = getattr(self.cfg, "cut_height", 270)
        # The coordinate space CLRNet reports lanes in — see remap_to_frame().
        self.ori_img_w = getattr(self.cfg, "ori_img_w", 1640)
        self.ori_img_h = getattr(self.cfg, "ori_img_h", 590)

    @staticmethod
    def remap_to_frame(pts: np.ndarray, frame_w: int, frame_h: int,
                       ori_w: int, ori_h: int, cut_height: int) -> np.ndarray:
        """Map CLRNet lane points into the coordinate space of OUR frame.

        THE BUG THIS FIXES
        -------------------
        CLRNet's `Lane.to_array(cfg)` does NOT return normalized coordinates.
        It returns pixels in the ORIGINAL CULane image space — `cfg.ori_img_w`
        x `cfg.ori_img_h`, i.e. 1640x590 — because it multiplies its
        normalized predictions back up by those config values internally.

        The previous code only rescaled when `pts[:,0].max() <= 2` (i.e. only
        if the values looked normalized). With real CULane-space values of up
        to 1640 that branch never fired, so NO rescaling happened at all and
        1640x590 coordinates were drawn straight onto an 848x480 video. The
        polylines landed nowhere near the actual paint, which is why the
        lane-type heuristic reported "unknown" on 3604/3604 cluster frames
        while CLRNet was happily reporting 4 lanes per frame.

        THE MAPPING
        -----------
        Inference feeds `frame[cut_height:]` resized to (img_w, img_h), which
        mirrors what CLRNet does to a CULane image: crop `cut_height` off the
        top, resize the remaining `ori_h - cut_height` rows. So a model-space
        row corresponds to the same FRACTION of the cropped region in both,
        and the y mapping has to go through the crop, not the full height:

            x_frame = x_ori * frame_w / ori_w
            y_frame = cut + (y_ori - cut) * (frame_h - cut) / (ori_h - cut)

        Normalized input (max <= 2.0) is still handled, so this stays correct
        if a config or CLRNet version reports in [0,1] instead.
        """
        pts = np.asarray(pts, dtype=np.float64).copy()
        if len(pts) == 0:
            return pts.astype(np.float32)

        # --- x ---
        if pts[:, 0].max() <= 2.0:            # normalized [0,1]
            pts[:, 0] *= frame_w
        else:                                  # CULane pixel space
            pts[:, 0] *= frame_w / float(ori_w)

        # --- y ---
        cut = float(cut_height)
        if pts[:, 1].max() <= 2.0:            # normalized within the CROPPED region
            pts[:, 1] = cut + pts[:, 1] * (frame_h - cut)
        else:                                  # CULane pixel space
            denom = max(float(ori_h) - cut, 1e-6)
            pts[:, 1] = cut + (pts[:, 1] - cut) * (frame_h - cut) / denom

        return pts.astype(np.float32)

    def infer(self, frame_bgr: np.ndarray) -> LaneResult:
        import torch

        h0, w0 = frame_bgr.shape[:2]
        img = frame_bgr[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.net(t)
            lanes = self.net.heads.get_lanes(output)[0]   # list of Lane objects

        result = LaneResult()
        for lane in lanes:
            pts = lane.to_array(self.cfg)      # CULane-space pixels (see remap_to_frame)
            pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
            if len(pts) < 2:
                continue
            pts = self.remap_to_frame(pts, w0, h0, self.ori_img_w,
                                      self.ori_img_h, self.cut_height)
            # Drop anything that still falls outside the frame after remapping
            # rather than letting off-screen points skew the ego-lane pick.
            inside = ((pts[:, 0] >= 0) & (pts[:, 0] < w0)
                      & (pts[:, 1] >= 0) & (pts[:, 1] < h0))
            pts = pts[inside]
            if len(pts) < 2:
                continue
            result.polylines.append(pts)
        result.assign_ego_lane(w0)
        return result


class UFLDv2Wrapper:
    """Ultra-Fast Lane Detection v2 wrapper (300+ FPS fallback)."""

    def __init__(self, weights: str, device: str = "cuda",
                 input_size: tuple = (1600, 320), num_row: int = 72,
                 num_col: int = 81):
        repo = PROJECT_ROOT / "external" / "UFLDv2"
        sys.path.insert(0, str(repo))
        import torch
        from utils.common import get_model  # UFLDv2 repo util

        self.device = device if torch.cuda.is_available() else "cpu"

        class _Cfg:  # minimal config the repo model factory expects
            backbone = "34"
            num_lanes = 4
            train_width, train_height = input_size
            num_row_, num_col_ = num_row, num_col
            use_aux = False
            fc_norm = True

        self.cfg = _Cfg()
        self.net = get_model(self.cfg).to(self.device).eval()
        state = torch.load(weights, map_location=self.device)["model"]
        self.net.load_state_dict(
            {k.replace("module.", ""): v for k, v in state.items()})
        self.input_size = input_size

    def infer(self, frame_bgr: np.ndarray) -> LaneResult:
        import torch

        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.net(t)

        result = LaneResult()
        result.polylines = self._decode(pred, w0, h0)
        result.assign_ego_lane(w0)
        return result

    def _decode(self, pred, w0, h0) -> list:
        """Row-anchor decoding: argmax over column bins per row anchor."""
        import torch

        loc = pred["loc_row"][0]          # (num_grid, num_row, num_lanes)
        exist = pred["exist_row"][0]      # (2, num_row, num_lanes)
        num_grid, num_row, num_lanes = loc.shape
        polylines = []
        row_anchors = np.linspace(0.42, 1.0, num_row) * h0
        for lane_idx in range(num_lanes):
            pts = []
            for r in range(num_row):
                if int(exist[:, r, lane_idx].argmax()) != 1:
                    continue
                col_probs = torch.softmax(loc[:, r, lane_idx], dim=0)
                col = float((col_probs * torch.arange(
                    num_grid, device=col_probs.device)).sum())
                x = col / (num_grid - 1) * w0
                pts.append((x, row_anchors[r]))
            if len(pts) >= 2:
                polylines.append(np.asarray(pts, dtype=np.float32))
        return polylines


def load_lane_detector(prefer: str = "clrnet", **kwargs):
    """Factory: CLRNet if available, else UFLDv2, else None with a warning."""
    weights_dir = PROJECT_ROOT / "weights"
    try:
        if prefer == "clrnet":
            return CLRNetWrapper(str(weights_dir / "clrnet_r101_culane.pth"), **kwargs)
        return UFLDv2Wrapper(str(weights_dir / "ufldv2_culane_res34.pth"), **kwargs)
    except Exception as e:
        # BUG FIX: this used to catch only (ImportError, FileNotFoundError).
        # On the HPC cluster CLRNet raised AttributeError (mmcv 2.x removed
        # mmcv.jit — see models/mmcv_compat.py), which escaped
        # this handler and crashed the WHOLE pipeline instead of degrading
        # to the fallback. A third-party research repo can fail at import or
        # construction time in many ways (missing CUDA ops, version skew,
        # config drift), so catch broadly here: an unavailable OPTIONAL lane
        # model must never take the pipeline down with it.
        print(f"[lane_detector] {prefer} unavailable "
              f"({type(e).__name__}: {e}); trying fallback")
        if prefer == "clrnet":
            return load_lane_detector("ufldv2", **kwargs)
        print("[lane_detector] No lane model available — lane features "
              "disabled (adas_final.py will use the drivable-area fallback)")
        return None
