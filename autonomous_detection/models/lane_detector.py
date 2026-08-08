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


def _install_mmcv1_compat_shim():
    """Let CLRNet import under mmcv 2.x.

    CLRNet was written against mmcv-full 1.x and decorates functions with
    `@mmcv.jit(...)` / `@mmcv.skip_no_elena`. Both were removed in mmcv 2.x,
    so importing CLRNet on a modern environment dies with
    `AttributeError: module 'mmcv' has no attribute 'jit'` (hit on the HPC
    cluster, Aug 2026).

    Re-adding them as identity decorators is not a hack — it restores the
    exact behavior of mmcv 1.x on standard PyTorch. In mmcv 1.x these were
    only functional under the Parrots backend (an internal SenseTime
    framework); on regular PyTorch builds they returned the undecorated
    function unchanged. So on this cluster the shim is a faithful no-op,
    identical to what CLRNet ran with originally.

    Does nothing if mmcv is absent or already provides these attributes.
    """
    try:
        import mmcv
    except ImportError:
        return   # caller's import of clrnet will fail with a clear error

    def _identity_decorator(func=None, **_kwargs):
        if func is None:                      # used as @mmcv.jit(...)
            return lambda f: f
        return func                           # used as @mmcv.jit

    for attr in ("jit", "skip_no_elena"):
        if not hasattr(mmcv, attr):
            setattr(mmcv, attr, _identity_decorator)
            print(f"[lane_detector] mmcv.{attr} missing (mmcv 2.x) — "
                  f"installed no-op compat shim for CLRNet")


class CLRNetWrapper:
    """CLRNet inference wrapper (primary lane detector)."""

    def __init__(self, weights: str, config: str | None = None, device: str = "cuda"):
        repo = PROJECT_ROOT / "external" / "CLRNet"
        sys.path.insert(0, str(repo))
        import torch

        _install_mmcv1_compat_shim()

        from clrnet.models.registry import build_net
        from clrnet.utils.config import Config

        config = config or str(repo / "configs/clrnet/clr_resnet101_culane.py")
        self.cfg = Config.fromfile(config)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = build_net(self.cfg).to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        self.net.load_state_dict(state["net"] if "net" in state else state)
        self.input_w = self.cfg.img_w
        self.input_h = self.cfg.img_h
        self.cut_height = getattr(self.cfg, "cut_height", 270)

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
        scale_y = (h0 - self.cut_height)
        for lane in lanes:
            pts = lane.to_array(self.cfg)                  # normalized (x, y)
            pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
            if len(pts) < 2:
                continue
            # to_array() may return pixel coords already (CLRNet's own convention
            # varies by config) or normalized [0,1] coords depending on cfg.ori_img_h/w.
            # BUG FIX: the old line always multiplied by w0 regardless of the
            # ternary branch (`w0/1.0 if cond else w0` both equal w0), so pixel-
            # scale output got re-multiplied by w0 and blew up to nonsense
            # coordinates -> polylines silently unusable downstream.
            if pts[:, 0].max() <= 2:
                pts[:, 0] *= w0
            pts[:, 1] = pts[:, 1] * scale_y + self.cut_height \
                if pts[:, 1].max() <= 2 else pts[:, 1]
            result.polylines.append(pts.astype(np.float32))
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
        # mmcv.jit — see _install_mmcv1_compat_shim above), which escaped
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
