"""ADAS Final Pipeline — the complete end product.

One frame in -> all perception + all decisions out:
    detection (11 classes incl. lights/signs) + tracking + lanes + lane types
    + depth + collision TTC + overtaking decision + traffic-light action
    + automatic night enhancement.

Usage:
    python inference/adas_final.py --source video.mp4 --save out.mp4
    python inference/adas_final.py --source 0 --show            # webcam
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from models.ipm import describe_radius

PROJECT_ROOT = Path(__file__).parent.parent

CLASS_NAMES = {
    0: "car", 1: "truck", 2: "bus", 3: "van", 4: "pedestrian", 5: "cyclist",
    6: "motorcycle", 7: "tram", 8: "traffic_light", 9: "traffic_sign", 10: "misc",
}
TRAFFIC_LIGHT_CLS = 8


@dataclass
class FrameResult:
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    lanes: object = None
    lanes_source: str = "model"    # "model" | "drivable_area_fallback"
    lane_types: dict = field(default_factory=dict)
    lighting: str = "DAY"
    collision_alerts: list = field(default_factory=list)
    overtaking: object = None
    traffic_light: str = "NO_LIGHT"
    latency_ms: float = 0.0


class ADASFinalPipeline:
    """Loads every module it can find; degrades gracefully when weights missing."""

    def __init__(self,
                 detector_weights: str = "runs/final/yolo11x_merged/weights/best.pt",
                 imgsz: int = 1280,
                 depth_every_n: int = 2,
                 conf_threshold: float = 0.35,
                 kitti_calib_path: str | None = None):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold

        w = Path(detector_weights)
        if not w.exists():
            print(f"[pipeline] {w} missing — falling back to pretrained yolo11x.pt")
            detector_weights = "yolo11x.pt"
        self.detector = YOLO(detector_weights)
        self.imgsz = imgsz
        wdir = PROJECT_ROOT / "weights"

        # Lane detection (CLRNet -> UFLDv2 -> None), with a drivable-area
        # segmentation fallback for roads with no usable painted markings
        # (very common on Indian roads — CLRNet/UFLDv2 are CULane-trained
        # and simply have nothing to fit a curve to there). See
        # models/drivable_area.py for the full rationale.
        from models.lane_detector import load_lane_detector
        from models.drivable_area import DrivableAreaSegmenter
        self.lanes_model = load_lane_detector("clrnet")
        da_weights = wdir / "drivable_area.pth"
        self.drivable_area = self._try(
            lambda: DrivableAreaSegmenter(str(da_weights) if da_weights.exists() else None))

        # Auxiliary classifiers (optional weights)
        from models.aux_classifiers import (
            TrafficLightStateClassifier, LaneTypeClassifier,
            classify_light_hsv, decide_traffic_light_action)
        self._decide_tl = decide_traffic_light_action
        self._hsv_fallback = classify_light_hsv
        self.tl_classifier = self._try(
            lambda: TrafficLightStateClassifier(str(wdir / "tl_state.pt")))
        self.lane_type_classifier = self._try(
            lambda: LaneTypeClassifier(str(wdir / "lane_type.pt")))

        # Depth (metric)
        from inference.collision import CollisionDetector, DepthAnythingV2Metric
        self.depth_model = self._try(lambda: DepthAnythingV2Metric(
            str(wdir / "depth_anything_v2_metric_vkitti_vits.pth")))
        self.collision = (CollisionDetector(self.depth_model)
                          if self.depth_model else None)
        self.depth_every_n = depth_every_n

        # Geometric size-consistency filter — rejects detections on things
        # that AREN'T actually there (billboards/hoardings/reflections a
        # domain-shifted detector can hallucinate a "car"-shaped box onto),
        # using the depth map we already compute for collision detection.
        # See inference/sanity_filter.py for the full rationale. Focal
        # length: exact from KITTI calib if you have it, else an approximate
        # dashcam-HFOV estimate computed lazily from the first frame's width.
        from inference.sanity_filter import SizeConsistencyFilter
        from models.ipm import focal_length_px_from_kitti_calib, estimate_focal_length_px
        self._estimate_focal_length_px = estimate_focal_length_px
        focal_length_px = None
        if kitti_calib_path:
            focal_length_px = self._try(
                lambda: focal_length_px_from_kitti_calib(kitti_calib_path))
        self.sanity_filter = SizeConsistencyFilter(focal_length_px=focal_length_px)

        # Tracking
        from inference.tracker import ByteTrackWrapper
        self.tracker = self._try(ByteTrackWrapper)

        # Night enhancement + overtaking
        from inference.night_enhance import NightEnhancer, scene_lighting
        from inference.overtaking import OvertakingAnalyzer
        self._scene_lighting = scene_lighting
        self.enhancer = NightEnhancer(str(wdir / "zero_dce_plus.pth"))
        self.overtaking = OvertakingAnalyzer()

        self._frame_idx = 0
        self._last_depth = None
        print("[pipeline] Modules loaded:",
              f"lanes={'Y' if self.lanes_model else 'N (drivable-area fallback only)'}",
              f"drivable_area={'trained' if self.drivable_area and self.drivable_area.model else 'CV-fallback'}",
              f"depth={'Y' if self.depth_model else 'N'}",
              f"tracker={'Y' if self.tracker else 'N'}",
              f"tl_state={'Y' if self.tl_classifier else 'HSV-fallback'}",
              f"size_check={'KITTI-calibrated' if self.sanity_filter.focal_length_px and kitti_calib_path else ('estimated-fx (pending)' if self.depth_model else 'N (no depth)')}",
              f"conf_threshold={self.conf_threshold}")

    @staticmethod
    def _try(factory):
        try:
            return factory()
        except Exception as e:  # missing weights/repo — degrade, don't crash
            print(f"[pipeline] Optional module unavailable: {e}")
            return None

    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, ego_speed_mps: float = 15.0
                      ) -> FrameResult:
        t0 = time.perf_counter()
        result = FrameResult()

        # 0. Lighting + optional enhancement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result.lighting = self._scene_lighting(gray)
        proc = self.enhancer.maybe_enhance(frame, result.lighting)

        # 1. Detection
        yolo_out = self.detector.predict(proc, imgsz=self.imgsz,
                                         conf=self.conf_threshold,
                                         verbose=False)[0]
        result.detections = self._to_detections(yolo_out)

        # 1.5 Geometric size-consistency filter — catches "detected something
        # where nothing real is there" (billboards/hoardings/reflections),
        # using the depth map from the previous depth-computation cycle
        # (self._last_depth; see inference/sanity_filter.py for why this is a
        # cheap, no-extra-inference check). No-ops gracefully if depth/focal
        # length aren't available yet (e.g. very first frame).
        if self.sanity_filter.focal_length_px is None:
            self.sanity_filter.focal_length_px = self._estimate_focal_length_px(frame.shape[1])
        result.detections = self.sanity_filter.filter(result.detections, self._last_depth)

        # 2. Lanes + lane types
        if self.lanes_model:
            result.lanes = self.lanes_model.infer(proc)
            if self.lane_type_classifier and result.lanes.polylines:
                result.lane_types = self.lane_type_classifier.classify(
                    proc, result.lanes)

        # Drivable-area fallback: CLRNet/UFLDv2 need painted lines to fit a
        # curve through, so `len(polylines) < 2` is the normal outcome on
        # roads with faded/absent/unfollowed markings (common on Indian
        # roads — see models/drivable_area.py). Substitute a
        # segmentation-derived corridor so collision.py's ego-path gating
        # still has something better than the crude "central 40%" guess.
        # lane_types stays empty here on purpose: overtaking.py's Rule 1
        # already fails safe to NOT_POSSIBLE_SOLID_LINE without real
        # lane-type info, which is the correct conservative behavior for a
        # road we have no painted-marking legality info for.
        if (result.lanes is None or len(result.lanes.polylines) < 2) and self.drivable_area:
            fallback = self.drivable_area.segment_to_lane_result(proc)
            if fallback is not None:
                result.lanes = fallback
                result.lanes_source = "drivable_area_fallback"

        # 3. Tracking
        if self.tracker:
            result.tracks = self.tracker.update(result.detections) or []
            for t in result.tracks:
                if not hasattr(t, "bbox"):
                    t.bbox = t.tlbr  # ByteTrack STrack compatibility
                if not hasattr(t, "class_name"):
                    t.class_name = "car"

        # 4. Traffic light states
        tl_states = {}
        for d in result.detections:
            if d.cls == TRAFFIC_LIGHT_CLS:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                crop = proc[max(0, y1):y2, max(0, x1):x2]
                tl_states[id(d)] = (self.tl_classifier.classify(crop)
                                    if self.tl_classifier
                                    else self._hsv_fallback(crop))
        result.traffic_light = self._decide_tl(tl_states)

        # 5. Collision TTC (depth every Nth frame to save latency)
        if self.collision and result.tracks:
            if self._frame_idx % self.depth_every_n == 0 or self._last_depth is None:
                result.collision_alerts = self.collision.update(
                    proc, result.tracks, result.lanes)
                self._last_depth = self.collision.last_depth
            self.collision.set_night_mode(result.lighting == "NIGHT_UNLIT")

        # 6. Overtaking decision
        if result.lanes is not None:
            result.overtaking = self.overtaking.analyze(
                result.lanes, result.lane_types, result.tracks,
                self._last_depth, ego_speed_mps,
                scene_brightness=float(gray.mean()),
                lighting_state=result.lighting)

        self._frame_idx += 1
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    @staticmethod
    def _to_detections(yolo_result):
        @dataclass
        class Det:
            bbox: tuple
            conf: float
            cls: int
            class_name: str

        dets = []
        if yolo_result.boxes is None:
            return dets
        for box, conf, cls in zip(yolo_result.boxes.xyxy.cpu().numpy(),
                                  yolo_result.boxes.conf.cpu().numpy(),
                                  yolo_result.boxes.cls.cpu().numpy()):
            c = int(cls)
            dets.append(Det(tuple(box), float(conf), c,
                            CLASS_NAMES.get(c, str(c))))
        return dets

    # ------------------------------------------------------------------ #

    def draw(self, frame: np.ndarray, r: FrameResult) -> np.ndarray:
        out = frame.copy()

        # Lanes: green = dashed (overtake OK), red = solid, yellow = drivable-
        # area fallback (no real lane-type info -> overtaking stays conservative)
        if r.lanes is not None:
            for pl in r.lanes.polylines:
                if r.lanes_source == "drivable_area_fallback":
                    color = (0, 220, 255)
                else:
                    ltype = r.lane_types.get(id(pl), "unknown")
                    color = (0, 0, 255) if "solid" in ltype else (0, 255, 0)
                pts = np.asarray(pl, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], False, color, 3)

        # Tracks / detections
        items = r.tracks if r.tracks else r.detections
        for obj in items:
            x1, y1, x2, y2 = (int(v) for v in obj.bbox)
            tid = getattr(obj, "track_id", None)
            label = (f"#{tid} " if tid is not None else "") + \
                getattr(obj, "class_name", "")
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(out, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 255), 2)

        # Collision alerts
        for a in r.collision_alerts:
            x1, y1, x2, y2 = (int(v) for v in a.bbox)
            color = (0, 0, 255) if a.level == "BRAKE" else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, f"{a.level} TTC {a.ttc_s:.1f}s {a.distance_m:.0f}m",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD banner
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 78), (20, 20, 20), -1)
        overtake_txt = r.overtaking.value if r.overtaking else "OVERTAKE: N/A"
        radius_m = getattr(self.overtaking, "last_curve_radius_m", None)
        if radius_m is not None:
            overtake_txt += f"  (curve radius ~{radius_m:.0f}m)"
        ok = r.overtaking is not None and "POSSIBLE" == r.overtaking.name
        cv2.putText(out, overtake_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if ok else (0, 100, 255), 2)
        tl_color = {"STOP": (0, 0, 255), "CAUTION": (0, 255, 255),
                    "GO": (0, 255, 0)}.get(r.traffic_light, (160, 160, 160))
        cv2.putText(out, f"LIGHT: {r.traffic_light}   SCENE: {r.lighting}   "
                    f"{1000 / max(r.latency_ms, 1):.0f} FPS",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_color, 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path / image / webcam id")
    ap.add_argument("--save", default=None, help="output video path")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--weights", default="runs/final/yolo11x_merged/weights/best.pt")
    ap.add_argument("--log", default="decisions.jsonl")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Detection confidence threshold. Raise this (e.g. 0.45-0.5) "
                         "if you're seeing phantom/false-positive detections on "
                         "out-of-distribution footage — see KRISH_HANDOVER.md status table.")
    ap.add_argument("--kitti_calib", default=None,
                    help="Path to a KITTI calib_cam_to_cam.txt for exact focal length "
                         "in the size-consistency filter. Omit to use an approximate "
                         "estimate from the video's own resolution.")
    args = ap.parse_args()

    pipe = ADASFinalPipeline(detector_weights=args.weights, conf_threshold=args.conf,
                             kitti_calib_path=args.kitti_calib)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    writer, log = None, open(args.log, "w")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = pipe.process_frame(frame)
        vis = pipe.draw(frame, r)

        radius_m = getattr(pipe.overtaking, "last_curve_radius_m", None)
        log.write(json.dumps({
            "frame": pipe._frame_idx, "lighting": r.lighting,
            "lanes_source": r.lanes_source if r.lanes is not None else None,
            "traffic_light": r.traffic_light,
            "overtaking": r.overtaking.name if r.overtaking else None,
            "curve_radius_m": round(radius_m, 1) if radius_m else None,
            "curve_desc": describe_radius(radius_m) if radius_m else None,
            "alerts": [{"level": a.level, "ttc": round(a.ttc_s, 2),
                        "dist": round(a.distance_m, 1)}
                       for a in r.collision_alerts],
            "n_objects": len(r.tracks or r.detections),
            "sanity_rejected_total": sum(pipe.sanity_filter.rejected_count.values()),
            "latency_ms": round(r.latency_ms, 1),
        }) + "\n")

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                    (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if args.show:
            cv2.imshow("ADAS", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    log.close()
    if writer:
        writer.release()
        print(f"Saved: {args.save}")
    print(f"Decision log: {args.log}")
    if pipe.sanity_filter.rejected_count:
        total = sum(pipe.sanity_filter.rejected_count.values())
        print(f"\n[sanity_filter] Rejected {total} phantom/implausible-size "
              f"detections this run: {dict(pipe.sanity_filter.rejected_count)}")
        print("  If this number seems too high (rejecting real objects) or too "
              "low (still seeing phantoms), tune --conf and/or check that "
              "--kitti_calib is set for accurate focal length.")


if __name__ == "__main__":
    main()
