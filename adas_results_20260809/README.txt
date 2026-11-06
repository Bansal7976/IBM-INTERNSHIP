ADAS PERCEPTION PIPELINE — RESULTS BUNDLE
IBM Internship 2026 · Vishal Bansal, Krish

HEADLINE (all measured, not projected)
  Detector      : YOLOv11x, 56.8M params, 194.5 GFLOPs
  mAP@0.5       : 95.4%
  mAP@0.5:0.95  : 79.9%
  Precision     : 95.4%      Recall : 92.5%
  Validation    : 1,496 images / 8,104 instances
  Pipeline      : 3,604 frames @ 23.1 FPS (43 ms/frame) on NVIDIA H100 80GB
  Collision     : 69 alerts — 20 BRAKE, 49 WARNING
  Phantom filter: 979 of 7,077 size-checked detections rejected (13.8%)

CONTENTS
  videos/       DEMO_highlight.mp4 = 30s clip for the presentation
                adas_FINAL_v11x.mp4 = full annotated run, 3,604 frames
  frames/       Stills at real BRAKE / WARNING / dense-traffic moments.
                Every overlay was drawn by the pipeline, not added after.
  plots/        Confusion matrix, PR / F1 curves, training curves, class balance
  metrics/      Raw per-epoch CSVs for all three training runs, plus the exact
                hyperparameters and dataset config used
  decision_log/ Per-frame JSONL: every alert, lane source, overtaking verdict
                and latency. This is what makes each result auditable.
  logs/         Pipeline run log (shows which modules loaded) + training tail
  weights_info/ Where the checkpoints live (too large to bundle)

TRAINING PROGRESSION
  YOLOv11m  50 ep  -> mAP@0.5 91.7%  |  mAP@0.5:0.95 69.3%
  YOLOv11m 500 ep  -> mAP@0.5 94.2%  |  mAP@0.5:0.95 77.7%
  YOLOv11x 300 ep  -> mAP@0.5 95.4%  |  mAP@0.5:0.95 79.9%   <- final

KNOWN LIMITATIONS (stated deliberately)
  - Only 7 of the 11 declared classes are trained. The merge that produced this
    checkpoint contained KITTI only, which has no traffic lights, signs, buses
    or motorcycles — so those four classes had nothing to learn from.
  - Overtaking returns "not possible" whenever lane-marking type is unavailable.
    That is the intended fail-safe, not a failure.
  - Indian-specific classes (autorickshaw, animal, cart) are not trained yet.
    Converters for IDD, DriveIndia and UVH-26 are written and tested.
