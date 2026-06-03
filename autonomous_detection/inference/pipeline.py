"""
Full inference pipeline: detect → track → visualize.

Supports:
  - Single image
  - Video file / webcam / RTSP stream
  - Batch of images

Quick demo:
    python inference/pipeline.py \
        --source path/to/video.mp4 \
        --model yolo11m \
        --weights runs/train/exp/weights/best.pt \
        --track \
        --save
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser(description='Run vehicle detection pipeline')
    p.add_argument('--source', required=True, help='Image, video, dir, URL, or webcam index')
    p.add_argument('--model', default='yolo11m', help='Model type (yolo11n/m/x, rtdetr-l, grounding_dino)')
    p.add_argument('--weights', default=None, help='Path to fine-tuned weights')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--track', action='store_true', help='Enable multi-object tracking')
    p.add_argument('--tracker', default='bytetrack', choices=['bytetrack', 'botsort'])
    p.add_argument('--save', action='store_true', help='Save annotated output')
    p.add_argument('--save_dir', default='runs/inference', help='Output directory')
    p.add_argument('--device', default='cuda', help='Device: cuda / cpu / mps')
    p.add_argument('--prompt', default=None, help='Text prompt for Grounding DINO')
    p.add_argument('--show', action='store_true', help='Show live preview')
    return p.parse_args()


class DetectionPipeline:
    """
    End-to-end detection + tracking pipeline.
    Wraps detector and optional tracker into a single callable.
    """

    def __init__(
        self,
        model_type: str = 'yolo11m',
        weights: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = 'cuda',
        enable_tracking: bool = False,
        tracker_type: str = 'bytetrack',
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.detector_2d import build_detector_2d

        self.detector = build_detector_2d({
            'type': model_type,
            'weights': weights,
            'device': device,
            'conf_threshold': conf,
            'iou_threshold': iou,
        })
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type
        self.device = device

    def run_image(self, image: Union[str, np.ndarray], prompt: Optional[str] = None):
        """Run on a single image. Returns list of Detection objects."""
        if hasattr(self.detector, 'predict'):
            if prompt and hasattr(self.detector, 'predict'):
                # GroundingDINO path
                if isinstance(image, str):
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self.detector.predict(image, prompt)
            else:
                results = self.detector.predict(image)
                return results[0] if results else []
        return []

    def run_video(
        self,
        source: Union[str, int],
        save: bool = False,
        save_dir: str = 'runs/inference',
        show: bool = False,
        prompt: Optional[str] = None,
    ):
        """Run on video file, webcam, or stream."""
        from inference.visualizer import Visualizer

        source_int = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(source_int)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            src_name = Path(str(source)).stem if isinstance(source, str) else 'webcam'
            out_path = save_dir / f'{src_name}_detected.mp4'
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height),
            )

        visualizer = Visualizer()
        frame_count = 0
        total_time = 0.0

        # Use Ultralytics track() for tracking mode
        if self.enable_tracking:
            model = self.detector.model
            results_gen = model.track(
                source=source_int,
                conf=self.detector.conf,
                iou=self.detector.iou,
                tracker=f'{self.tracker_type}.yaml',
                persist=True,
                stream=True,
                device=self.device,
            )
        else:
            results_gen = self.detector.model.predict(
                source=source_int,
                conf=self.detector.conf,
                iou=self.detector.iou,
                stream=True,
                device=self.device,
            )

        for result in results_gen:
            t0 = time.perf_counter()
            frame = result.orig_img.copy()
            annotated = visualizer.draw_ultralytics_result(frame, result, show_tracks=self.enable_tracking)

            elapsed = time.perf_counter() - t0
            total_time += elapsed
            frame_count += 1

            fps_display = frame_count / total_time
            cv2.putText(annotated, f'FPS: {fps_display:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow('Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"Saved to: {out_path}")
        if show:
            cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames at {frame_count/total_time:.1f} FPS avg")


def main():
    args = parse_args()

    pipeline = DetectionPipeline(
        model_type=args.model,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        enable_tracking=args.track,
        tracker_type=args.tracker,
    )

    source = args.source
    # Check if it's an image file
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if Path(source).suffix.lower() in img_exts:
        detections = pipeline.run_image(source, prompt=args.prompt)
        print(f"Detected {len(detections)} objects:")
        for d in detections:
            print(f"  {d.class_name}: {d.score:.2f} @ {d.box.tolist()}")

        if args.save:
            from inference.visualizer import Visualizer
            img = cv2.imread(source)
            viz = Visualizer()
            annotated = viz.draw_detections(img, detections)
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{Path(source).stem}_detected.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved: {out_path}")
    else:
        pipeline.run_video(
            source=source,
            save=args.save,
            save_dir=args.save_dir,
            show=args.show,
            prompt=args.prompt,
        )


if __name__ == '__main__':
    main()
