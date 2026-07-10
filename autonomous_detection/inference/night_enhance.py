"""Scene lighting analysis + low-light enhancement (Zero-DCE++).

Layer that runs BEFORE detection. In daytime it's a no-op (zero cost).
At night it brightens the frame so the detector sees more.

Usage:
    enhancer = NightEnhancer()
    lighting = scene_lighting(gray_frame)   # "DAY" / "NIGHT_LIT" / "NIGHT_UNLIT"
    frame = enhancer.maybe_enhance(frame, lighting)
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def scene_lighting(frame_gray: np.ndarray) -> str:
    """Classify scene lighting from a grayscale frame.

    Returns:
        "DAY"         - bright scene, no enhancement needed
        "NIGHT_LIT"   - dark but street lights present (urban night)
        "NIGHT_UNLIT" - dark with no artificial lighting (rural/mountain)
    """
    brightness = float(frame_gray.mean())
    # Bright blobs in top third of the frame at night = street lamps / lit signs
    top_third = frame_gray[: frame_gray.shape[0] // 3]
    lit_fraction = float((top_third > 200).mean())

    if brightness > 90:
        return "DAY"
    if lit_fraction > 0.01:
        return "NIGHT_LIT"
    return "NIGHT_UNLIT"


class _DCENet(nn.Module):
    """Zero-DCE++ curve estimation network (7 conv layers, depthwise-separable)."""

    def __init__(self, scale_factor: int = 12):
        super().__init__()
        self.scale_factor = scale_factor
        n = 32

        def dsconv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, 1, 1, groups=cin),
                nn.Conv2d(cin, cout, 1),
            )

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = dsconv(3, n)
        self.conv2 = dsconv(n, n)
        self.conv3 = dsconv(n, n)
        self.conv4 = dsconv(n, n)
        self.conv5 = dsconv(n * 2, n)
        self.conv6 = dsconv(n * 2, n)
        self.conv7 = dsconv(n * 2, 3)

    def forward(self, x):
        down = nn.functional.interpolate(
            x, scale_factor=1.0 / self.scale_factor, mode="bilinear", align_corners=False
        )
        x1 = self.relu(self.conv1(down))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        alpha = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
        alpha = nn.functional.interpolate(
            alpha, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        # Apply the light-enhancement curve 8 times (Zero-DCE formulation)
        for _ in range(8):
            x = x + alpha * (torch.pow(x, 2) - x)
        return x


class NightEnhancer:
    """Low-light frame enhancement. Falls back to CLAHE if no torch/weights."""

    def __init__(self, weights: str | None = None, device: str = "cuda"):
        self.model = None
        if TORCH_AVAILABLE:
            self.device = device if torch.cuda.is_available() else "cpu"
            self.model = _DCENet().to(self.device).eval()
            if weights:
                try:
                    state = torch.load(weights, map_location=self.device)
                    self.model.load_state_dict(state)
                    print(f"[NightEnhancer] Loaded Zero-DCE++ weights: {weights}")
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"[NightEnhancer] Weights not loaded ({e}); using CLAHE fallback")
                    self.model = None
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def maybe_enhance(self, frame_bgr: np.ndarray, lighting_state: str) -> np.ndarray:
        """Enhance only when the scene is dark. Day frames pass through untouched."""
        if lighting_state == "DAY":
            return frame_bgr
        if self.model is not None:
            return self._enhance_dce(frame_bgr)
        return self._enhance_clahe(frame_bgr)

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def _enhance_dce(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(t).clamp(0, 1)
        out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def _enhance_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """No-training fallback: CLAHE on the luminance channel."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    state = scene_lighting(gray)
    print(f"Scene lighting: {state} (mean brightness {gray.mean():.1f})")
    enhancer = NightEnhancer()
    out = enhancer.maybe_enhance(img, state)
    cv2.imwrite("enhanced_output.jpg", out)
    print("Saved: enhanced_output.jpg")
