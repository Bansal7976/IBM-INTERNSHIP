"""Write weights/ipm_calibration.json without the GUI click tool.

WHY THIS EXISTS
---------------
The path planner works in metres, and the only thing that turns image pixels
into metres is the ground homography in weights/ipm_calibration.json. Without
that file `inference/adas_final.py` reports plan_status "no camera-to-ground
calibration" on every frame and produces no trajectory at all -- correctly, since
planning in pixels would be worse than not planning.

The existing way to make the file (models/ipm.py, interactive) needs a window and
mouse clicks on a lane. That cannot run on the cluster.

WHAT THE NUMBERS ARE
--------------------
This builds the homography from ASSUMED camera parameters, not measured ones:
horizontal field of view, mounting height and downward pitch. Every distance the
planner then reports is only as good as those three guesses. That is fine for a
demonstration video and is NOT a substitute for calibrating the actual camera.
Say so wherever the output is shown. If you know the real values, pass them.

Typical dashcam: ~60-90 degree HFOV, 1.2-1.5 m high, 0-5 degrees nose-down.

    python scripts/make_ipm_calibration.py --video demo/clip.mp4
    python scripts/make_ipm_calibration.py --width 1920 --height 1080 --hfov 70
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                   # noqa: E402

from models.ipm import DEFAULT_CALIB_PATH, IPMTransformer   # noqa: E402


def frame_size(video: Path):
    import cv2
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {video}")
    return frame.shape[1], frame.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, help="read the frame size from this clip")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--hfov", type=float, default=60.0, help="horizontal FOV, degrees")
    ap.add_argument("--camera-height", type=float, default=1.4, help="metres above the road")
    ap.add_argument("--pitch", type=float, default=3.0, help="degrees nose-down")
    ap.add_argument("--out", type=Path, default=DEFAULT_CALIB_PATH)
    args = ap.parse_args()

    if args.video:
        w, h = frame_size(args.video)
    elif args.width and args.height:
        w, h = args.width, args.height
    else:
        raise SystemExit("give --video, or both --width and --height")

    focal = w / (2.0 * math.tan(math.radians(args.hfov) / 2.0))
    ipm = IPMTransformer.from_intrinsics(focal, focal, w / 2.0, h / 2.0,
                                         args.camera_height, args.pitch)
    if np.allclose(ipm.H, np.eye(3)):
        raise SystemExit("refusing to write an identity homography; that would "
                         "make every 'metre' a pixel")
    ipm.save(args.out)

    horizon = h / 2.0 - focal * math.tan(math.radians(args.pitch))
    nearest = ipm.pixel_to_ground(np.array([[w / 2.0, h - 1.0]]))[0][1]
    mid = ipm.pixel_to_ground(np.array([[w / 2.0, 0.75 * h]]))[0][1]
    print(f"wrote {args.out}")
    print(f"  frame {w}x{h}, hfov {args.hfov} deg, height {args.camera_height} m, "
          f"pitch {args.pitch} deg  (ASSUMED, not measured)")
    print(f"  horizon at row {horizon:.0f}; nearest visible road {nearest:.1f} m; "
          f"row 75% down the frame is {mid:.1f} m ahead")
    print("  Treat every distance as approximate until the real camera is calibrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
