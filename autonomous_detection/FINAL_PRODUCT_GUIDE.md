# FINAL PRODUCT IMPLEMENTATION GUIDE
## Complete ADAS Perception System — SOTA Accuracy, All Features, HPC Training
### Phase 3: From Working Model → Production End Product

**Current State:** YOLOv11m @ 91.71% mAP@0.5 on KITTI (detection only)
**Target State:** Full ADAS system — detection + lane + traffic lights + overtaking decision + collision + night mode

---

# TABLE OF CONTENTS

1. [System Overview — What the End Product Does](#1-system-overview)
2. [Accuracy Improvements for Detection (SOTA push)](#2-accuracy-improvements)
3. [Feature 1: Lane Detection (incl. mountain/curved roads)](#3-lane-detection)
4. [Feature 2: Overtaking Decision (Possible / Not Possible)](#4-overtaking-decision)
5. [Feature 3: Traffic Light & Sign Detection](#5-traffic-lights)
6. [Feature 4: Night & Adverse Condition Operation](#6-night-mode)
7. [Feature 5: Collision Detection (TTC)](#7-collision-detection)
8. [All Datasets — Download Instructions](#8-datasets)
9. [HPC Training Plan (SLURM jobs, order of execution)](#9-hpc-plan)
10. [End-to-End Pipeline Integration](#10-integration)
11. [Expected Final Results](#11-expected-results)
12. [3-Week Execution Schedule](#12-schedule)

---

# 1. SYSTEM OVERVIEW

## What the Final Product Does (per video frame)

```
INPUT: Camera frame (dashcam / vehicle camera)
   │
   ├─▶ [A] OBJECT DETECTION (YOLOv11x, fine-tuned)
   │      → cars, trucks, pedestrians, cyclists, traffic lights, signs
   │
   ├─▶ [B] LANE DETECTION (CLRNet / UFLDv2)
   │      → lane polylines + lane TYPE (solid / dashed / double)
   │      → works on curves & mountain roads (CULane curve category)
   │
   ├─▶ [C] DEPTH ESTIMATION (Depth Anything V2)
   │      → per-pixel metric distance
   │
   ├─▶ [D] TRACKING (ByteTrack)
   │      → persistent IDs + velocity per object
   │
   └─▶ [E] LOW-LIGHT ENHANCEMENT (Zero-DCE++, only at night)
          → brightens frame BEFORE detection when scene is dark

DECISION LAYER (fuses A+B+C+D):
   ├─ COLLISION WARNING:  TTC = distance / closing-speed → SAFE / WARN / BRAKE
   ├─ OVERTAKING:         lane-type + oncoming traffic + gap → POSSIBLE / NOT POSSIBLE
   ├─ TRAFFIC LIGHT:      red/yellow/green state → STOP / CAUTION / GO
   └─ LANE DEPARTURE:     ego position vs lane center → drift warning

OUTPUT: Annotated video + JSON decision log + alerts
```

## Why these specific models (research-backed)

| Task | Model | Why (SOTA evidence) |
|---|---|---|
| 2D Detection | **YOLOv11x** (upgrade from m) | +3-4 mAP over 11m; still 30+ FPS on A100 |
| Lane Detection | **CLRNet** (ResNet-101) | 80.47 F1 on CULane — best on CURVES, critical for mountain roads |
| Lane (fast fallback) | **UFLDv2** | 300+ FPS if CLRNet too slow for real-time |
| Traffic Lights | Same YOLOv11x, trained on **BDD100K** | BDD100K has 10 classes incl. traffic light/sign, 100K images, day+night |
| Depth | **Depth Anything V2** (metric, outdoor) | Foundation model, zero-shot metric depth, no LiDAR needed |
| Tracking | **ByteTrack** | 80.3 MOTA MOT17; velocity comes free for TTC |
| Night Enhancement | **Zero-DCE++** | Unsupervised, 500 FPS, no paired data needed |
| 3D (optional stretch) | **BEVFormer** | 56.9 NDS camera-only if time permits |

---

# 2. ACCURACY IMPROVEMENTS

Current: YOLOv11m, 640px, 500 epochs → 91.71% mAP@0.5.
Push to **94-95% mAP@0.5** with these, in priority order:

## 2.1 Upgrade model size: YOLOv11m → YOLOv11x  (+2-3 mAP)

```bash
# On HPC — same config, bigger model, higher resolution
yolo detect train \
  model=yolo11x.pt \
  data=data/kitti_yolo/kitti.yaml \
  imgsz=1280 \            # KITTI images are 1242px wide — 640 destroys small objects
  epochs=300 \
  batch=16 \              # per GPU; use 8x GPUs → effective 128
  device=0,1,2,3,4,5,6,7 \
  cos_lr=True \
  patience=50 \
  optimizer=AdamW lr0=0.0005
```

**Why imgsz=1280 matters:** your confusion matrix shows 35% of missed objects are small (<20px). At 640px input, a distant car in a 1242px KITTI image becomes ~10px. At 1280px it stays ~20px — detectable.

## 2.2 Merge datasets: KITTI + BDD100K + nuScenes 2D  (+1-2 mAP, huge robustness)

Your model only saw German daytime roads (KITTI). Merge:
- **KITTI**: 7,481 images (daytime, Germany)
- **BDD100K**: 100,000 images (day/night/rain/fog, USA) ← fixes night blindness
- **nuScenes 2D projections**: 28K frames (Boston/Singapore, rain, night)

Unify class map (script below in §8.5). Training on merged ~135K images is exactly what HPC is for.

## 2.3 Better augmentation for driving-specific failure modes

Add to training config (`configs/yolov11/yolov11_final.yaml`):

```yaml
# Fixes your documented failure modes:
mosaic: 1.0            # small objects (35% of FNs)
mixup: 0.15
copy_paste: 0.3        # occlusion (40% of FNs) — pastes objects over others
hsv_v: 0.6             # aggressive brightness — night/shadow robustness
degrees: 5.0           # mountain road banking angles
perspective: 0.0005    # slope/gradient viewpoint changes
close_mosaic: 20       # disable mosaic last 20 epochs (cleaner convergence)
```

Plus Albumentations weather layer (already in `data/augmentations.py`) — enable fog/rain/sun-flare at p=0.2.

## 2.4 Test-Time Augmentation + tuned NMS (+0.5-1 mAP, free at eval)

```python
model.predict(source, augment=True, conf=0.25, iou=0.5)  # TTA: multi-scale + flip
```

## 2.5 Optional: ensemble YOLOv11x + RT-DETR-X (weighted box fusion)

```bash
pip install ensemble-boxes
```
```python
from ensemble_boxes import weighted_boxes_fusion
boxes, scores, labels = weighted_boxes_fusion(
    [yolo_boxes, rtdetr_boxes], [yolo_scores, rtdetr_scores],
    [yolo_labels, rtdetr_labels], weights=[2, 1], iou_thr=0.55)
```
Use only for offline evaluation/reports (2x inference cost). Report both numbers.

---

# 3. LANE DETECTION (incl. MOUNTAIN / CURVED ROADS)

## 3.1 Model choice: CLRNet (primary) + UFLDv2 (real-time fallback)

**CLRNet** — Cross Layer Refinement Network (CVPR 2022)
- Repo: `https://github.com/Turoad/CLRNet`
- CULane F1: 80.47 (ResNet-101), **best-in-class on the CURVE category** — this is exactly the mountain-road requirement
- CULane has 9 scenario categories including **Curve**, **Night**, **Shadow**, **No-line** — evaluate per-category to prove mountain/night performance

**UFLDv2** — Ultra-Fast Lane Detection v2 (TPAMI 2022)
- Repo: `https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2`
- 300+ FPS — use when total pipeline latency budget is tight

## 3.2 Setup & training (CLRNet on CULane)

```bash
# Clone + install
git clone https://github.com/Turoad/CLRNet.git external/CLRNet
cd external/CLRNet
pip install -r requirements.txt
python setup.py build develop

# Train on HPC (dataset download in §8.2)
python main.py configs/clrnet/clr_resnet101_culane.py --gpus 0 1 2 3

# Evaluate — REPORT PER-CATEGORY (this is your mountain-road evidence)
python main.py configs/clrnet/clr_resnet101_culane.py --validate \
  --load_from culane_r101.pth
# Output includes: Normal / Crowded / Night / No-line / Shadow / Arrow / 
#                  Dazzle-light / CURVE / Crossroad F1 scores
```

**Expected results table (CLRNet ResNet-101 published numbers — your target):**

| Category | F1 Score | Relevance |
|---|---|---|
| Normal | 93.7 | baseline |
| **Curve** | **75.6** | ← mountain roads |
| **Night** | **75.0** | ← night driving |
| Shadow | 82.1 | tree-lined mountain roads |
| Dazzle light | 73.7 | oncoming headlights |
| Crowded | 78.8 | traffic |

## 3.3 Lane TYPE classification (solid vs dashed) — needed for overtaking

CULane gives lane *position*, not *type*. For solid/dashed classification (legal-to-overtake signal), fine-tune on **BDD100K lane marking labels** which include lane type attributes (solid/dashed, single/double, color):

```
BDD100K lane labels: bdd100k.com → Download → "Lane Marking"
Categories: single white, single yellow, double white, double yellow,
            solid, dashed  ← exactly what overtaking logic needs
```

Simple, robust approach that works well in practice:
1. CLRNet gives lane polyline (set of points)
2. Sample image patches (32×32) along the polyline
3. Tiny CNN classifier (ResNet-18) on patches → solid / dashed / double
4. Majority vote along the line → lane type

This 2-stage approach is easier to train and debug than end-to-end, and the patch classifier trains in <1 hour on 1 GPU with BDD100K crops.

---

# 4. OVERTAKING DECISION (POSSIBLE / NOT POSSIBLE)

This is a **fusion rule module**, not a new neural network. It combines outputs you already have. This is a genuine research contribution (Gap G1/G3 from your literature review — nobody fuses these).

## 4.1 Decision logic

```python
# inference/overtaking.py
from dataclasses import dataclass
from enum import Enum

class OvertakeStatus(Enum):
    POSSIBLE = "OVERTAKING POSSIBLE"
    NOT_POSSIBLE_SOLID_LINE = "NOT POSSIBLE - SOLID LINE"
    NOT_POSSIBLE_ONCOMING = "NOT POSSIBLE - ONCOMING VEHICLE"
    NOT_POSSIBLE_NO_GAP = "NOT POSSIBLE - INSUFFICIENT GAP"
    NOT_POSSIBLE_CURVE = "NOT POSSIBLE - CURVE/LOW VISIBILITY"
    NOT_POSSIBLE_LOW_VIS = "NOT POSSIBLE - NIGHT/WEATHER"

@dataclass
class OvertakingAnalyzer:
    min_gap_seconds: float = 8.0      # time needed to complete overtake
    max_lane_curvature: float = 0.003 # 1/m — beyond this, blind curve

    def analyze(self, lanes, lane_types, tracked_objects, depth_map, ego_speed_mps,
                scene_brightness):
        # RULE 1: Lane marking must legally permit it
        center_line = self._get_center_divider(lanes)
        if lane_types.get(center_line) in ('solid', 'double_solid', 'solid_yellow'):
            return OvertakeStatus.NOT_POSSIBLE_SOLID_LINE

        # RULE 2: Road must be straight enough to see ahead (mountain roads!)
        curvature = self._estimate_curvature(lanes)   # fit polynomial, take 2nd deriv
        if curvature > self.max_lane_curvature:
            return OvertakeStatus.NOT_POSSIBLE_CURVE

        # RULE 3: No oncoming vehicle within the overtake window
        for obj in tracked_objects:
            if self._in_oncoming_lane(obj, lanes):
                dist = depth_map.at(obj.center)               # meters
                closing = ego_speed_mps + abs(obj.velocity_mps)
                time_to_meet = dist / max(closing, 0.1)
                if time_to_meet < self.min_gap_seconds:
                    return OvertakeStatus.NOT_POSSIBLE_ONCOMING

        # RULE 4: Enough visibility (night/fog degrades depth confidence)
        if scene_brightness < 40 and not self._has_streetlights(tracked_objects):
            return OvertakeStatus.NOT_POSSIBLE_LOW_VIS

        # RULE 5: Gap ahead of lead vehicle must fit ego + margin
        lead = self._lead_vehicle(tracked_objects, lanes)
        if lead and not self._gap_sufficient(lead, tracked_objects, depth_map):
            return OvertakeStatus.NOT_POSSIBLE_NO_GAP

        return OvertakeStatus.POSSIBLE

    def _estimate_curvature(self, lanes):
        # Fit 2nd-order polynomial x = ay² + by + c to ego-lane points
        # (in BEV-warped coordinates); curvature = |2a| / (1+ (2ay+b)²)^1.5
        ...
```

## 4.2 How each input feeds the decision

| Signal | Source module | Rule it powers |
|---|---|---|
| Lane polyline | CLRNet | curvature (Rule 2), lane assignment (Rule 3) |
| Lane type (solid/dashed) | patch classifier (§3.3) | legality (Rule 1) |
| Oncoming vehicle + distance | YOLOv11x + Depth Anything V2 | gap timing (Rule 3) |
| Vehicle velocity | ByteTrack | closing speed (Rule 3) |
| Scene brightness | mean pixel luminance | visibility (Rule 4) |

## 4.3 Validation method (for the report)

No public "overtaking" ground-truth dataset exists — so create an evaluation protocol (this is a contribution):
1. Take 50 clips from BDD100K with visible center lines + oncoming traffic
2. Manually label each 1-second window: overtaking possible yes/no (2-3 hours of work)
3. Report accuracy / precision / recall of the decision module
4. Target: >90% agreement with human labels; **zero false "POSSIBLE" when oncoming vehicle <8s away** (safety-critical metric)

---

# 5. TRAFFIC LIGHT & SIGN DETECTION

## 5.1 Approach: extend YOLOv11x classes via BDD100K

BDD100K detection labels already include **traffic light** and **traffic sign** among its 10 classes — plus every image tagged day/night/dawn. One training run gives you vehicles + lights + signs + night robustness simultaneously.

**BDD100K 10 classes:** pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, **traffic light, traffic sign**

## 5.2 Traffic light STATE (red/yellow/green)

BDD100K boxes don't include color state. Two-stage approach (same pattern as lane type):

1. YOLOv11x detects traffic light bbox
2. Crop bbox → tiny classifier for state

**Dataset for state classifier: LISA Traffic Light Dataset**
- Kaggle: `https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset`
- 43,000 frames, 113,000 annotated lights WITH states (go/stop/warning), day+night
- `kaggle datasets download -d mbornoe/lisa-traffic-light-dataset`

```python
# models/light_classifier.py — trains in ~30 min on 1 GPU
import torch.nn as nn
from torchvision.models import resnet18

class TrafficLightStateClassifier(nn.Module):
    """Crop of detected traffic light -> red / yellow / green / off."""
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights='IMAGENET1K_V1')
        self.net.fc = nn.Linear(512, 4)
    def forward(self, crop_batch):   # (N,3,64,64)
        return self.net(crop_batch)
```

HSV-threshold fallback (no training, surprisingly robust for lit lamps): mask the crop for red/amber/green hue ranges, pick the dominant lit region weighted by vertical position (red on top). Ship both; report classifier as primary.

## 5.3 Street lights (context signal)

Street lights are not a KITTI/BDD class. You don't need to *detect* them individually — you need the *signal* "is the road artificially lit at night" for the overtaking/visibility rules. Compute it directly:

```python
def scene_lighting(frame_gray):
    brightness = frame_gray.mean()
    # bright blobs in top third of frame at night = street lighting
    top = frame_gray[: frame_gray.shape[0]//3]
    lit_pixels = (top > 200).mean()
    if brightness > 90:   return "DAY"
    if lit_pixels > 0.01: return "NIGHT_LIT"      # street lights present
    return "NIGHT_UNLIT"                            # rural/mountain darkness
```

This drives: (a) whether Zero-DCE enhancement activates, (b) Rule 4 of overtaking, (c) TTC alert thresholds (longer margins when dark).

---

# 6. NIGHT & ADVERSE CONDITION OPERATION

## 6.1 Three-layer strategy

**Layer 1 — Train on night data (the biggest win)**
BDD100K is ~40% night/dawn/dusk images. Merged training (§2.2) makes the detector natively night-capable. This alone typically recovers most of the night mAP gap.

**Layer 2 — Low-light enhancement preprocessing (Zero-DCE++)**
- Repo: `https://github.com/Li-Chongyi/Zero-DCE_extension`
- Unsupervised (no paired data), ~500 FPS at 640px, one small CNN
- Activate only when `scene_lighting() != "DAY"` — zero cost in daytime

```python
# inference/night_enhance.py
class NightEnhancer:
    def __init__(self, weights='weights/zero_dce_plus.pth'):
        self.model = load_zero_dce(weights).eval().cuda()
    def maybe_enhance(self, frame, lighting_state):
        if lighting_state == "DAY":
            return frame
        return self.model.enhance(frame)   # brightened frame -> detector
```

**Layer 3 — Evaluate on dedicated night benchmark (proof for the report)**
**ExDark dataset** — 7,363 genuinely dark images, 12 object classes:
- `https://github.com/cs-chan/Exclusively-Dark-Image-Dataset`
- Evaluate detector with/without Zero-DCE → ablation table for the report

**Expected ablation (typical published behavior):**

| Configuration | Night mAP@0.5 (relative) |
|---|---|
| KITTI-only model, raw night images | ~45-55% (poor — current state) |
| + BDD100K merged training | ~75-80% |
| + Zero-DCE enhancement | ~80-85% |
| Day performance (reference) | ~93-95% |

## 6.2 Adverse weather (rain/fog) — free from BDD100K

BDD100K weather tags: clear / rainy / snowy / foggy / overcast. Evaluate per-condition using tags — one more robustness table for the report with zero extra labeling.

---

# 7. COLLISION DETECTION (TTC)

## 7.1 Components (all already planned — this wires them together)

```python
# inference/collision.py
class CollisionDetector:
    """TTC = distance / closing_speed, per tracked object in ego path."""
    CRITICAL_TTC = 1.5   # seconds -> BRAKE alert
    WARNING_TTC  = 3.0   # seconds -> WARN alert

    def __init__(self, depth_model, fps=30):
        self.depth = depth_model     # Depth Anything V2 (metric, outdoor)
        self.fps = fps
        self.dist_history = {}       # track_id -> deque of distances

    def update(self, frame, tracks, lanes):
        depth_map = self.depth.infer(frame)          # metric meters
        alerts = []
        for t in tracks:
            if not self._in_ego_path(t, lanes):      # only objects in our lane
                continue
            d = float(np.median(self._bbox_depth(depth_map, t.bbox)))  # robust
            hist = self.dist_history.setdefault(t.track_id, deque(maxlen=10))
            hist.append(d)
            if len(hist) >= 5:
                # closing speed from depth history (m/s), smoothed
                closing = (hist[0] - hist[-1]) / (len(hist)/self.fps)
                if closing > 0.5:                    # actually approaching
                    ttc = d / closing
                    if ttc < self.CRITICAL_TTC:
                        alerts.append(Alert("BRAKE", t, d, ttc))
                    elif ttc < self.WARNING_TTC:
                        alerts.append(Alert("WARNING", t, d, ttc))
        return alerts
```

Key implementation details that make it actually work:
- **Median depth over bbox lower-half** (not center pixel) — robust to windows/reflections
- **Closing speed from depth history**, not pixel velocity — pixel motion ≠ physical approach
- **Ego-path filter using lanes** — a parked car in the next lane must not trigger BRAKE
- **Night mode: raise thresholds** (CRITICAL 1.5→2.5s) when `NIGHT_UNLIT`

## 7.2 Depth Anything V2 setup

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/DepthAnythingV2
cd external/DepthAnythingV2
pip install -r requirements.txt
# Use the METRIC outdoor checkpoint (trained on virtual KITTI):
# depth_anything_v2_metric_vkitti_vitl.pth  (from repo's model zoo)
```

Validate metric accuracy against KITTI depth ground truth (KITTI provides LiDAR-projected depth maps) — report AbsRel error. Target: AbsRel < 0.12 outdoors.

## 7.3 Validation

KITTI provides no collision labels. Protocol:
1. Compare predicted distance vs KITTI LiDAR ground-truth distance per detected object → distance error % (target <10% within 40m)
2. Synthetic scenario tests: clips of approaching vehicles → does TTC alert fire before the 3s mark? Measure alert lead time.

---

# 8. ALL DATASETS — DOWNLOAD INSTRUCTIONS

## 8.1 BDD100K (THE key addition — night, weather, lights, signs, lane types)

```bash
# Registration: http://bdd-data.berkeley.edu/  (free, academic)
# After login, download these packages:
#   - "100K Images"          (5.3 GB)  bdd100k_images_100k.zip
#   - "Detection 2020 Labels" (~110 MB) bdd100k_det_20_labels_trainval.zip
#   - "Lane Marking Labels"   (~40 MB)  bdd100k_lane_labels_trainval.zip

mkdir -p data/bdd100k && cd data/bdd100k
unzip bdd100k_images_100k.zip
unzip bdd100k_det_20_labels_trainval.zip
unzip bdd100k_lane_labels_trainval.zip

# Convert to YOLO format (script provided in §8.5)
python data/prepare_bdd100k.py --data_root data/bdd100k
```

## 8.2 CULane (lane detection — has Curve + Night categories)

```bash
# Official: https://xingangpan.github.io/projects/CULane.html
# Google Drive links on that page. Total ~40 GB (133,235 frames)
# Download: driver_*_frame.tar.gz (6 parts), annotations, list.tar.gz

mkdir -p data/culane && cd data/culane
for f in driver_23_30frame driver_161_90frame driver_182_30frame \
         driver_193_90frame driver_100_30frame driver_37_30frame; do
  tar xzf ${f}.tar.gz
done
tar xzf annotations_new.tar.gz && tar xzf list.tar.gz
```

## 8.3 LISA Traffic Light (state classifier)

```bash
pip install kaggle    # needs ~/.kaggle/kaggle.json API token
kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data/lisa_tl
cd data/lisa_tl && unzip lisa-traffic-light-dataset.zip
```

## 8.4 ExDark (night evaluation benchmark)

```bash
git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset data/exdark
# Images + annotations included in repo releases (1.5 GB)
```

## 8.5 Merge script — KITTI + BDD100K + nuScenes → one YOLO dataset

Create `data/prepare_merged.py`:

```python
"""Merge KITTI + BDD100K + nuScenes into unified YOLO dataset.

Unified 11-class map:
  0 car  1 truck  2 bus  3 van  4 pedestrian  5 cyclist
  6 motorcycle  7 tram/train  8 traffic_light  9 traffic_sign  10 misc
"""
CLASS_MAP = {
    # KITTI -> unified
    'kitti': {'Car':0,'Truck':1,'Van':3,'Pedestrian':4,'Cyclist':5,
              'Tram':7,'Misc':10,'Person_sitting':4},
    # BDD100K -> unified
    'bdd':   {'car':0,'truck':1,'bus':2,'pedestrian':4,'rider':5,
              'bicycle':5,'motorcycle':6,'train':7,
              'traffic light':8,'traffic sign':9},
    # nuScenes -> unified
    'nusc':  {'car':0,'truck':1,'bus':2,'pedestrian':4,'bicycle':5,
              'motorcycle':6,'construction_vehicle':10,'trailer':1},
}
# ... walks each dataset, rewrites label txt files with unified ids,
#     symlinks images into data/merged_yolo/{train,val}/images
#     writes data/merged_yolo/merged.yaml
```

Result: `data/merged_yolo/merged.yaml` with ~135K train images — the single dataset for the final detector run.

## 8.6 Dataset summary table

| Dataset | Size | Purpose | Registration |
|---|---|---|---|
| KITTI (have it) | 12 GB | detection baseline | cvlibs.net ✅ done |
| **BDD100K** | 5.4 GB | night/weather/lights/signs/lane types | bdd-data.berkeley.edu |
| **CULane** | 40 GB | lane detection (curve+night F1) | direct GDrive links |
| **LISA TL** | 4 GB | light state red/yellow/green | Kaggle |
| **ExDark** | 1.5 GB | night eval benchmark | GitHub, none |
| nuScenes (optional) | 350 GB full / 4 GB mini | 3D + extra 2D | nuscenes.org |

Total new download: **~51 GB** (without full nuScenes) — download directly on HPC login node with `wget/curl`, not through your laptop.

---

# 9. HPC TRAINING PLAN

## 9.1 Job list (run in this order; A & B can run in parallel)

| # | Job | GPUs | Time | Output |
|---|---|---|---|---|
| A | YOLOv11x on merged 135K @ 1280px | 8 | ~36h | final detector |
| B | CLRNet ResNet-101 on CULane | 4 | ~24h | lane model |
| C | Lane-type patch classifier (BDD) | 1 | ~1h | solid/dashed |
| D | Traffic light state classifier (LISA) | 1 | ~30min | red/yellow/green |
| E | Zero-DCE++ (or use published weights) | 1 | ~2h | night enhancer |
| F | Eval suite: KITTI + BDD-night + ExDark + CULane categories | 1 | ~3h | all report tables |

## 9.2 SLURM script for Job A (the big one)

```bash
#!/bin/bash
#SBATCH --job-name=yolo11x_merged
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/yolo11x_merged_%j.log

module load cuda/11.8
source activate autodet

cd $SLURM_SUBMIT_DIR
yolo detect train \
  model=yolo11x.pt \
  data=data/merged_yolo/merged.yaml \
  imgsz=1280 epochs=300 batch=128 device=0,1,2,3,4,5,6,7 \
  optimizer=AdamW lr0=0.0005 cos_lr=True warmup_epochs=5 \
  mosaic=1.0 mixup=0.15 copy_paste=0.3 hsv_v=0.6 close_mosaic=20 \
  patience=50 save_period=25 \
  project=runs/final name=yolo11x_merged
```

```bash
# Submit everything:
sbatch training/slurm/yolo11x_merged.sh          # Job A
sbatch training/slurm/clrnet_culane.sh           # Job B (parallel)
# After A+B finish:
sbatch training/slurm/classifiers.sh             # C+D (1 GPU, quick)
sbatch training/slurm/eval_all.sh                # F
```

## 9.3 Monitoring

```bash
squeue -u $USER                                   # job status
tail -f logs/yolo11x_merged_*.log                 # live loss
tensorboard --logdir runs/final --port 6006       # curves (ssh tunnel)
```

---

# 10. END-TO-END PIPELINE INTEGRATION

## 10.1 Final pipeline class

```python
# inference/adas_final.py
class ADASFinalPipeline:
    """Complete production pipeline. One frame in -> decisions out."""

    def __init__(self, cfg):
        self.detector   = YOLO('runs/final/yolo11x_merged/weights/best.pt')
        self.lanes      = CLRNetWrapper('weights/clrnet_r101_culane.pth')
        self.lane_type  = LaneTypeClassifier('weights/lane_type.pt')
        self.tl_state   = TrafficLightStateClassifier('weights/tl_state.pt')
        self.depth      = DepthAnythingV2Metric('weights/dav2_metric_outdoor.pth')
        self.tracker    = ByteTrackWrapper()
        self.enhancer   = NightEnhancer('weights/zero_dce_plus.pth')
        self.collision  = CollisionDetector(self.depth)
        self.overtaking = OvertakingAnalyzer()

    def process_frame(self, frame, ego_speed_mps=None):
        # 0. Scene analysis + optional night enhancement
        lighting = scene_lighting(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        proc = self.enhancer.maybe_enhance(frame, lighting)

        # 1. Perception (detector + lanes can run on parallel CUDA streams)
        dets   = self.detector.predict(proc, imgsz=1280, conf=0.3, verbose=False)
        lanes  = self.lanes.infer(proc)
        tracks = self.tracker.update(dets)

        # 2. Semantics
        lane_types = self.lane_type.classify(proc, lanes)
        tl_states  = {d.id: self.tl_state.classify(crop(proc, d.bbox))
                      for d in dets if d.cls == TRAFFIC_LIGHT}

        # 3. Decisions
        collision_alerts = self.collision.update(proc, tracks, lanes)
        overtake = self.overtaking.analyze(
            lanes, lane_types, tracks, self.collision.last_depth,
            ego_speed_mps or estimate_ego_speed(tracks), 
            scene_brightness=proc.mean())
        light_action = decide_traffic_light_action(tl_states, self.collision.last_depth)

        return FrameResult(
            detections=dets, tracks=tracks, lanes=lanes, lane_types=lane_types,
            lighting=lighting, collision_alerts=collision_alerts,
            overtaking=overtake, traffic_light=light_action)
```

## 10.2 Latency budget (A100, 1280px input)

| Stage | Latency | Notes |
|---|---|---|
| Night enhance (night only) | 3 ms | skipped in day |
| YOLOv11x detection | 25 ms | TensorRT FP16: ~12 ms |
| CLRNet lanes | 18 ms | UFLDv2 fallback: 3 ms |
| Depth Anything V2 (ViT-S) | 15 ms | run every 2nd frame, interpolate |
| ByteTrack + classifiers | 3 ms | CPU-light |
| Decision layer | <1 ms | pure logic |
| **Total** | **~62 ms → 16 FPS** | **TensorRT + frame-skip depth: ~35 ms → 28 FPS** ✅ |

Export path for deployment: `yolo export model=best.pt format=engine half=True imgsz=1280`

## 10.3 Output artifacts per run

- Annotated MP4: boxes+IDs, lane overlay colored by type (green=dashed, red=solid), TTC countdown, overtaking banner, traffic-light state chip, day/night indicator
- `decisions.jsonl`: one JSON per frame (all detections, distances, TTC values, decisions) — for quantitative evaluation
- `summary.csv`: per-frame latency + FPS

---

# 11. EXPECTED FINAL RESULTS (report targets)

| Capability | Metric | Current | Target | Benchmark |
|---|---|---|---|---|
| 2D Detection (day) | mAP@0.5 | 91.7% | **94%+** | KITTI val |
| 2D Detection (night) | mAP@0.5 | untested | **80%+** | BDD100K night / ExDark |
| Lane detection overall | F1 | — | **79%+** | CULane |
| Lane — **Curve** (mountain) | F1 | — | **74%+** | CULane curve split |
| Lane — **Night** | F1 | — | **73%+** | CULane night split |
| Lane type (solid/dashed) | Accuracy | — | **95%+** | BDD100K crops |
| Traffic light state | Accuracy | — | **97%+** | LISA test |
| Tracking | MOTA | — | **75%+** | KITTI tracking |
| Distance estimation | AbsRel | — | **<0.12** | KITTI LiDAR GT |
| Collision alerts | Lead time | — | **>3s before TTC=0** | scenario clips |
| Overtaking decision | Human agreement | — | **>90%**, 0 unsafe-POSSIBLE | 50 labeled clips |
| Full pipeline | FPS | — | **25+ (TensorRT)** | A100 |

---

# 12. 3-WEEK EXECUTION SCHEDULE

## Week A: Data + Big Training Launch
```
Day 1-2 (Vishal): Download BDD100K, CULane, LISA, ExDark ON HPC (§8)
Day 2-3 (Vishal): Write + run prepare_bdd100k.py, prepare_merged.py; verify
                  merged.yaml with a 1-epoch smoke test on 1 GPU
Day 3   (Vishal): Submit Job A (YOLOv11x merged, 8 GPU) — runs ~36h unattended
Day 3   (Krish):  Clone CLRNet, verify on CULane sample, submit Job B (4 GPU)
Day 4-5 (Krish):  Build lane-type patch dataset from BDD100K; train classifier (Job C)
Day 5   (Vishal): Train traffic-light state classifier on LISA (Job D)
Day 6-7 (both):   Zero-DCE integration + scene_lighting(); Jobs A/B finish → 
                  collect metrics
SUNDAY REVIEW: merged detector results + CLRNet per-category table
```

## Week B: Decision Modules + Integration
```
Day 1-2 (Vishal): collision.py — depth validation vs KITTI LiDAR, TTC on clips
Day 2-3 (Vishal): overtaking.py — all 5 rules, unit tests per rule
Day 3-4 (Krish):  adas_final.py — wire all modules, run on BDD100K videos
Day 4-5 (Krish):  Label 50 overtaking clips (validation protocol §4.3)
Day 5-6 (both):   Night ablation on ExDark (raw vs enhanced vs merged-trained)
Day 7:            Bug fixes; demo videos: day, night, mountain curve, overtake
SUNDAY REVIEW: live demo of full pipeline + ablation tables
```

## Week C: Optimization + Final Results
```
Day 1-2 (Krish):  TensorRT export (detector + depth), measure real FPS
Day 2-3 (Vishal): Full evaluation suite (Job F) — every table in §11
Day 3-4 (both):   Failure analysis with example frames; fix worst 2 issues
Day 5-6 (Vishal): Final report: methodology + all results tables + demo videos
Day 7:            Final presentation + GitHub release (tag v1.0)
FINAL REVIEW: complete end product, all metrics, demos
```

---

# QUICK-START CHECKLIST (do these first, in order)

```bash
# 1. On HPC login node — start dataset downloads TODAY (they're the bottleneck)
#    Register at bdd-data.berkeley.edu NOW (approval can take a day)

# 2. While waiting: smoke-test upgraded detector on existing KITTI data
yolo detect train model=yolo11x.pt data=data/kitti_yolo/kitti.yaml \
  imgsz=1280 epochs=5 batch=8 device=0        # verify 1280px fits in memory

# 3. Clone all external repos
git clone https://github.com/Turoad/CLRNet.git external/CLRNet
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/DepthAnythingV2
git clone https://github.com/Li-Chongyi/Zero-DCE_extension external/ZeroDCE
git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset data/exdark

# 4. Verify each model loads + runs on one test image before writing any
#    integration code. Integration order: detector → lanes → depth → tracker
#    → classifiers → decision layer. One module at a time, test after each.
```

---

**Prepared by:** Vishal Bansal & Krish — IBM Internship 2026
**Document version:** Final Product Guide 
