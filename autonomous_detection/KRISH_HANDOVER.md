# KRISH — COMPLETE EXECUTION HANDOVER
## Everything to do, in order, with exact commands and links
### Read this top to bottom, then execute. Nothing else needed.

**What's already done:** All code is written and in this repo. Your job = download datasets, run scripts, train on HPC, collect results, push.

**Two paths in this doc:**
- **PLAN A** — with BDD100K (best results; needs registration at Berkeley, can take 1-2 days for approval)
- **PLAN B (FALLBACK)** — KITTI + nuScenes + Kaggle datasets only, **zero registration walls except nuScenes (instant approval)**. Use this if BDD100K approval doesn't come or download fails.

Both plans produce the complete end product. Plan B loses ~nothing on features, slightly less night-training data.

**Changelog since your Phase 3 push (8e4bcf9):** three real bugs found by deep-review + one accuracy gap closed, all in the decision layer (lane math / collision / overtaking), not the detector. Your 95.42% mAP detector training is untouched and still valid. Details + a 2-minute test to prove it all works: **Section 1.5** right below.

| # | File | What was wrong | Fixed |
|---|---|---|---|
| 1 | `models/lane_detector.py` (CLRNet) | Coordinate-scaling line always multiplied by frame width regardless of whether coords were already pixel-scale — could blow lane polylines up to nonsense numbers | Only rescale when coords are actually normalized |
| 2 | `inference/collision.py` | Distance/speed history was only ever built for objects inside our own lane, so `depth_speed_mps` (the real closing speed) was **never set** on oncoming-lane vehicles — `overtaking.py`'s oncoming-traffic rule silently assumed oncoming cars were stationary, overestimating the safe window | History + speed now tracked for every object; alerts still scoped to ego-path only |
| 3 | `inference/collision.py` | The ego-path check used `if lanes is not None and not _in_ego_path(...)`, which skipped calling `_in_ego_path()` entirely whenever lanes were `None` — so its documented "fall back to center 40% of frame" behavior could never run, and every object anywhere in frame would raise alerts when lane detection was down for a frame | Always call `_in_ego_path()`; it already handles `lanes=None` |
| 4 (gap, not a bug) | `inference/overtaking.py` | Curve/visibility rule compared **pixel-space** curvature against a hand-picked number with no real-world meaning — unreliable exactly on the mountain/switchback roads this project needs to handle | New `models/ipm.py`: converts lane polylines to real ground-plane meters (standard ADAS Inverse Perspective Mapping) and compares against actual road-design curve-radius standards (AASHTO/IRC) |

---

# 0. WHAT THE FINAL PRODUCT IS

One command runs the full ADAS system on any video:

```bash
python inference/adas_final.py --source dashcam.mp4 --save output.mp4
```

Output: annotated video + `decisions.jsonl` with per-frame:
- Object detection (11 classes incl. traffic lights & signs) + tracking IDs
- Lane lines colored by type (green=dashed → overtake legal, red=solid)
- **OVERTAKING POSSIBLE / NOT POSSIBLE** banner with reason
- **Collision TTC warnings** (WARNING / BRAKE with distance + seconds)
- **Traffic light action** (STOP / CAUTION / GO)
- **Day/Night detection** with automatic low-light enhancement

---

# 1. SETUP (30 minutes, do once)

```bash
# Clone the repo (branch has everything)
git clone -b feature/autonomous-detection-complete \
  https://github.com/Bansal7976/IBM-INTERNSHIP.git
cd IBM-INTERNSHIP/autonomous_detection

# Environment (on HPC login node)
conda create -n autodet python=3.10 -y
conda activate autodet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install kaggle tqdm ensemble-boxes

# Verify the already-tested tracker works on your machine:
python inference/tracker.py
# Expected last line: "OK - tracker works"

# Verify detector loads (auto-downloads yolo11x.pt):
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); print('OK')"
```

**Kaggle API setup (needed for LISA dataset):**
1. kaggle.com → Account → Settings → "Create New Token" → downloads `kaggle.json`
2. `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

---

# 1.5 VALIDATE THE DECISION LOGIC FIRST (2 minutes — do this before any GPU job)

Job A (detector) takes ~36h, Job B (lanes) takes ~24h. If a bug in the
**decision logic** (TTC math, overtaking rules, lane coordinate handling —
not the detector itself) breaks something, you don't want to find that out
after burning 2 days of GPU time. This runs every decision rule against
synthetic data — no GPU, no weights, no dataset, seconds to run:

```bash
python scripts/verify_adas_pipeline.py
```

Expected output: `ALL CHECKS PASSED (14/14)`. If anything says `FAIL`, stop
and fix it (or ping Vishal) before touching `qsub`/`sbatch` — it tells you
exactly which rule broke and why. This script is also how the 3 bugs in the
changelog above were caught, so it's a real regression net, not a formality.

**One more one-time step — calibrate real-world curve detection:**
The overtaking module's curve/blind-corner check needs to know how camera
pixels map to real road meters (this is what turns "curvature" from a
meaningless pixel number into an actual curve radius you can trust — see
`models/ipm.py` docstring for the full explanation). Two options:

```bash
# Option A — training on KITTI footage: exact, automatic, nothing to do.
# IPMTransformer.from_kitti_calib() reads KITTI's calib_cam_to_cam.txt directly.

# Option B — any other dashcam footage: one-time manual calibration.
# Pick one clear frame of a STRAIGHT, FLAT, EMPTY road from your video and run:
python models/ipm.py calibrate path/to/a_straight_road_frame.jpg
# Click 4 points (near-left, near-right, far-left, far-right lane-line points),
# enter the lane width and the two distances when prompted (defaults are fine
# for a standard ~3.5 m lane). Saves to weights/ipm_calibration.json — reused
# automatically by every future run of inference/adas_final.py from that camera.
```

If you skip this, `overtaking.py` auto-detects the missing calibration and
falls back to the old pixel-based heuristic (prints a warning, doesn't
crash) — so nothing breaks, but the curve/mountain-road detection you asked
about is meaningfully more reliable once calibrated.

---

# 2. FILE MAP — WHAT EXISTS AND WHAT IT DOES

```
autonomous_detection/
├── inference/
│   ├── adas_final.py        ★ THE END PRODUCT — full pipeline, run this
│   ├── tracker.py           ✅ TESTED — ByteTrack-style tracking + velocity
│   ├── collision.py         TTC collision detection + Depth Anything V2 wrapper
│   ├── overtaking.py        5-rule overtaking decision (possible/not possible)
│   └── night_enhance.py     Day/night classifier + Zero-DCE++ (CLAHE fallback)
├── models/
│   ├── lane_detector.py     CLRNet + UFLDv2 wrappers (lane polylines)
│   ├── aux_classifiers.py   Traffic-light state + lane-type (solid/dashed)
│   └── ipm.py               NEW — pixel→real-meters curvature (see §1.5)
├── scripts/
│   └── verify_adas_pipeline.py  NEW — 2-min logic smoke test, run before HPC (see §1.5)
├── data/
│   ├── prepare_kitti.py     KITTI → YOLO (already used in Week 1)
│   ├── prepare_nuscenes.py  nuScenes → YOLO (multi-camera 2D projection)
│   ├── prepare_bdd100k.py   BDD100K → YOLO (PLAN A)
│   ├── prepare_lisa_det.py  LISA → YOLO traffic-light boxes (PLAN B)
│   └── prepare_merged.py    Merges everything → data/merged_yolo/merged.yaml
├── training/
│   ├── slurm/yolo11x_merged.sh   JOB A: main detector (8 GPU, ~36h)
│   ├── slurm/clrnet_culane.sh    JOB B: lane model (4 GPU, ~24h)
│   └── train_aux_classifiers.py  JOBS C+D: tiny classifiers (1 GPU, <1h each)
├── evaluation/
│   └── evaluate_final.py    JOB F: produces ALL report tables in one run
├── FINAL_PRODUCT_GUIDE.md   Deep research doc (why each model/dataset)
└── KRISH_HANDOVER.md        This file
```

---

# 3. DATASETS

## 3.1 Decision table — which plan are you on?

| Question | Answer |
|---|---|
| BDD100K approval came within 2 days? | → **PLAN A** (skip §3.3) |
| No approval / download broken / in a hurry? | → **PLAN B** (skip §3.2) |

Common to BOTH plans (download these regardless, start today):

| Dataset | Size | For | Link | Registration? |
|---|---|---|---|---|
| KITTI | done ✅ | detection base | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | already have |
| CULane | 40 GB | lanes (curve+night) | https://xingangpan.github.io/projects/CULane.html | **NO** (GDrive links) |
| LISA Traffic Light | 4 GB | light states | https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset | Kaggle account only |
| ExDark | 1.5 GB | night eval | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset | **NO** |

```bash
# CULane — download the GDrive files from the project page above into data/culane/, then:
cd data/culane
for f in driver_*.tar.gz annotations_new.tar.gz list.tar.gz; do tar xzf "$f"; done
cd ../..

# LISA
kaggle datasets download -d mbornoe/lisa-traffic-light-dataset -p data/lisa_tl
cd data/lisa_tl && unzip -q lisa-traffic-light-dataset.zip && cd ../..

# ExDark
git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset data/exdark
```

## 3.2 PLAN A — BDD100K (preferred)

1. Register: http://bdd-data.berkeley.edu (academic email helps; approval usually <2 days)
2. Download from the portal:
   - `bdd100k_images_100k.zip` (5.3 GB)
   - `bdd100k_det_20_labels_trainval.zip` (~110 MB)
   - `bdd100k_lane_labels_trainval.zip` (~40 MB)
3. Extract into `data/bdd100k/` so you have `data/bdd100k/images/100k/{train,val}` and `data/bdd100k/labels/`
4. Convert + merge:

```bash
python data/prepare_bdd100k.py --data_root data/bdd100k --out data/bdd100k_yolo
python data/prepare_merged.py --out data/merged_yolo
# → prints per-source counts; expect ~85K train images total
```

What Plan A gives you: night/rain/fog training images (40% of BDD100K), traffic light + sign boxes, lane TYPE labels (solid/dashed) for the overtaking legality classifier.

## 3.3 PLAN B — FALLBACK: KITTI + nuScenes + LISA (no Berkeley needed)

Every Plan-A capability has a replacement:

| Capability | Plan A source | Plan B replacement |
|---|---|---|
| Extra detection data | BDD100K 100K imgs | **nuScenes** 28K frames (6 cameras, Boston+Singapore) |
| Night training images | BDD100K night | **nuScenes night scenes** (Singapore night clips) + synthetic darkening |
| Traffic light BOXES | BDD100K class | **LISA** boxes (43K frames, day+night) via `prepare_lisa_det.py` |
| Traffic light STATE | LISA (same in both) | LISA (unchanged) |
| Lane type solid/dashed | BDD100K lane labels | **Heuristic classifier** (below) — no training needed |
| Night evaluation | ExDark (same) | ExDark (unchanged) |
| Rain/fog robustness | BDD100K weather | **Albumentations synthetic weather** (already in data/augmentations.py) |

### Step B1 — nuScenes (instant registration)

```bash
# Register: https://www.nuscenes.org/sign-up  (approval is instant, unlike Berkeley)
# Option 1 (RECOMMENDED START): mini split — 4 GB, 404 scenes sample
#   Download "v1.0-mini" from https://www.nuscenes.org/nuscenes#download
# Option 2 (full power): v1.0-trainval Metadata + "Keyframe blobs only"
#   (keyframes-only camera data ≈ 40 GB instead of 350 GB — enough for 2D detection)

mkdir -p data/nuscenes && cd data/nuscenes
tar xzf v1.0-mini.tgz          # or the trainval parts
cd ../..
pip install nuscenes-devkit

# Convert 6-camera 3D boxes -> 2D YOLO (script already in repo):
python data/prepare_nuscenes.py --data_root data/nuscenes --version v1.0-mini
# For full: --version v1.0-trainval  → ~168K camera images (28K frames × 6 cams)
```

### Step B2 — LISA as traffic-light detection source

```bash
python data/prepare_lisa_det.py --lisa data/lisa_tl --out data/lisa_yolo
```

### Step B3 — synthetic night augmentation for KITTI (compensates missing BDD night)

Add to the training command (already supported flags): `hsv_v=0.7` — aggressive
brightness jitter. Plus enable the Albumentations weather pipeline
(`data/augmentations.py` — RandomFog/RandomRain/RandomSunFlare, set p=0.25).
nuScenes Singapore night scenes provide REAL night images on top of this.

### Step B4 — merge everything

```bash
python data/prepare_merged.py --out data/merged_yolo
# Auto-detects which sources exist: kitti + nuscenes + lisa (bdd skipped)
# Expect: ~7.5K kitti + up to 168K nuscenes + ~30K lisa train images
```

### Step B5 — lane type WITHOUT BDD100K (heuristic, no training)

The trained classifier needs BDD crops. Fallback: gap analysis along the
polyline — dashed lines have periodic gaps, solid lines don't. Drop this into
`models/aux_classifiers.py` usage — `LaneTypeClassifier` already returns
"unknown" without weights, and `overtaking.py` fails safe on "unknown". To get
real decisions in Plan B, use this heuristic instead:

```python
def classify_lane_type_heuristic(frame_gray, polyline, patch=9):
    """Solid vs dashed by white-pixel continuity along the line."""
    import numpy as np
    pts = np.asarray(polyline).astype(int)
    hits = []
    h, w = frame_gray.shape
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            region = frame_gray[max(0,y-patch):y+patch, max(0,x-patch):x+patch]
            hits.append(1 if (region > 160).mean() > 0.08 else 0)
    if len(hits) < 6:
        return "unknown"
    coverage = np.mean(hits)             # fraction of samples on paint
    transitions = np.abs(np.diff(hits)).sum()  # paint<->gap switches
    if coverage > 0.85 and transitions <= 2:
        return "solid"
    if transitions >= 4:
        return "dashed"
    return "solid" if coverage > 0.6 else "unknown"
```
(Already fail-safe: "unknown" → overtaking says NOT POSSIBLE.)

**Plan B honest cost:** night detection mAP will be a few points lower than
Plan A (less real night data), and lane-type accuracy ~90% (heuristic) vs ~95%
(trained). Everything else identical.

---

# 4. HPC EXECUTION — JOB BY JOB

## 4.0 SLURM ya PBS? Pehle check karo

```bash
which sbatch && echo "SLURM cluster"     # -> use training/slurm/*.sh  (sbatch)
which qsub   && echo "PBS cluster"       # -> use training/pbs/*.pbs   (qsub)
```

Both versions of every job script exist — same training, different scheduler:

| Job | SLURM | PBS |
|---|---|---|
| A (detector) | `sbatch training/slurm/yolo11x_merged.sh` | `qsub training/pbs/yolo11x_merged.pbs` |
| B (lanes) | `sbatch training/slurm/clrnet_culane.sh` | `qsub training/pbs/clrnet_culane.pbs` |
| C+D+F (classifiers+eval) | run interactively (§4 below) | `qsub training/pbs/aux_and_eval.pbs` |

PBS monitoring: `qstat -u $USER` (status), `qdel <jobid>` (cancel),
`tail -f logs/yolo11x_merged.log` (live output). If your cluster's PBS needs a
queue name, add `#PBS -q <queue>` (find queues with `qstat -Q`).

## Job order (A and B run in parallel; C/D/E after; F last)

```
   ┌── JOB A: detector (8 GPU, ~36h) ──┐
   │                                    ├──▶ JOB C+D: classifiers (1 GPU, 1.5h)
   └── JOB B: lanes    (4 GPU, ~24h) ──┘         │
                                                  ▼
                                          JOB F: evaluation (1 GPU, ~3h)
```

## JOB A — Main detector (YOLOv11x @ 1280px on merged data)

```bash
# Smoke test FIRST (5 min) — catches dataset/memory problems before burning 36h:
yolo detect train model=yolo11x.pt data=data/merged_yolo/merged.yaml \
  imgsz=1280 epochs=1 batch=8 device=0 name=smoke_test

# If smoke test passes:
sbatch training/slurm/yolo11x_merged.sh
# Monitor:
squeue -u $USER
tail -f logs/yolo11x_merged_*.log
# Output: runs/final/yolo11x_merged/weights/best.pt
```

If GPU memory errors at 1280px: edit the script, `batch=64` (or `imgsz=960`).

## JOB B — Lane detection (CLRNet on CULane)

```bash
sbatch training/slurm/clrnet_culane.sh
# Output: weights/clrnet_r101_culane.pth
#         logs/clrnet_per_category_eval.txt  ← Curve + Night F1 numbers for report
```

**Shortcut if training fails or time is short:** use the author's pretrained
CULane ResNet-101 weights from the CLRNet repo releases
(https://github.com/Turoad/CLRNet — README "Trained models" table) →
save as `weights/clrnet_r101_culane.pth`. Identical published numbers; note in
report that weights are from the original authors.

## JOBS C + D — Tiny classifiers (interactive 1-GPU node, <2h total)

```bash
# C: traffic light state (both plans)
python training/train_aux_classifiers.py prepare-lisa --lisa data/lisa_tl
python training/train_aux_classifiers.py train-tl
# → weights/tl_state.pt   (expect val_acc > 0.97)

# D: lane type — PLAN A ONLY (Plan B uses the heuristic from §3.3 B5)
python training/train_aux_classifiers.py prepare-lanes --bdd data/bdd100k
python training/train_aux_classifiers.py train-lane
# → weights/lane_type.pt  (expect val_acc > 0.93)
```

## JOB E — Depth + night weights (downloads, no training)

```bash
# Depth Anything V2 metric (outdoor) — small encoder is enough:
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/DepthAnythingV2
# Download checkpoint from that repo's README "Pre-trained Models" (metric depth,
# vKITTI/outdoor, ViT-S) → save as:
#   weights/depth_anything_v2_metric_vkitti_vits.pth

# Zero-DCE++ weights (optional — CLAHE fallback works without it):
git clone https://github.com/Li-Chongyi/Zero-DCE_extension external/ZeroDCE
# Copy their released .pth → weights/zero_dce_plus.pth
```

## JOB F — Full evaluation (produces every report table)

```bash
python evaluation/evaluate_final.py \
  --weights runs/final/yolo11x_merged/weights/best.pt \
  --video data/demo/test_drive.mp4
# → runs/final_evaluation.json + console tables:
#   detection mAP / per-condition / ExDark night ablation / depth AbsRel / FPS
```

---

# 5. RUN THE END PRODUCT + MAKE DEMO VIDEOS

```bash
# Full pipeline on a video:
python inference/adas_final.py \
  --source demo_drive.mp4 \
  --weights runs/final/yolo11x_merged/weights/best.pt \
  --save demo_output.mp4

# Make these 4 demo clips for the presentation:
#   1. Day highway    — detection + tracking + TTC
#   2. Night clip     — enhancement on + detection still working
#   3. Curved road    — lane detection following the curve
#   4. Two-lane road  — overtaking POSSIBLE→NOT POSSIBLE transitions
# Good free source clips: nuScenes mini scenes, or record/YouTube dashcam
# (for internal demo only).
```

The pipeline degrades gracefully: any missing weights just disable that
feature and print which module is off — so you can demo detection+tracking+TTC
even before lanes finish training.

---

# 6. GIT WORKFLOW — WHAT AND HOW TO PUSH

```bash
git checkout feature/autonomous-detection-complete
git pull

# After each milestone:
git add -A
git commit -m "Phase 3: <what you did — e.g. merged dataset prep + Job A submitted>"
git push origin feature/autonomous-detection-complete
```

**Push:** code changes, configs, small result files (results.csv, evaluation JSON, confusion matrices, per-category lane eval txt, this doc updated with actual numbers).
**NEVER push:** datasets (`data/`), weights (`weights/`, `runs/*/weights/`), videos. Check `.gitignore` covers them; if not:

```bash
cat >> .gitignore << 'EOF'
data/
weights/
runs/**/weights/
*.mp4
*.pth
*.pt
external/
EOF
```

---

# 7. RESULTS TO COLLECT (fill this table, it goes in the final report)

| Metric | Where it comes from | Target | Actual |
|---|---|---|---|
| Detection mAP@0.5 (merged val) | Job F console | ≥93% | ___ |
| Detection mAP@0.5:0.95 | Job F | ≥72% | ___ |
| Night ablation gain (ExDark) | Job F | +15% dets recovered | ___ |
| Lane F1 overall (CULane) | Job B eval log | ≥79 | ___ |
| Lane F1 **Curve** | Job B eval log | ≥74 | ___ |
| Lane F1 **Night** | Job B eval log | ≥73 | ___ |
| TL state accuracy | Job C output | ≥97% | ___ |
| Lane type accuracy | Job D output (Plan A) | ≥93% | ___ |
| Depth AbsRel | Job F | <0.12 | ___ |
| Pipeline FPS | Job F | ≥15 (≥25 w/ TensorRT) | ___ |
| Overtaking human-agreement | manual: 50 clips | ≥90%, 0 unsafe-POSSIBLE | ___ |

For the overtaking validation: pick 50 one-second windows from nuScenes/dashcam
clips with visible center lines, label each yourself possible/not-possible
(2-3 hours), then compare against `decisions.jsonl`.

---

# 8. TROUBLESHOOTING (the 6 things most likely to break)

| Problem | Fix |
|---|---|
| CUDA OOM at imgsz=1280 | batch=64 → 32; or imgsz=960 (still much better than 640) |
| CLRNet build fails (`python setup.py build develop`) | needs gcc + matching CUDA; `module load gcc/9 cuda/11.8`; last resort: use author pretrained weights (§4 Job B shortcut) |
| Depth Anything import error | its repo API moved — check `external/DepthAnythingV2/metric_depth/README`; the wrapper in `inference/collision.py` has the import path at top, adjust one line |
| nuScenes devkit "table not found" | version string mismatch — pass exactly `--version v1.0-mini` or `v1.0-trainval` matching what you extracted |
| Overtaking always says NOT POSSIBLE - NO LANE INFO | lanes model not loaded → check `weights/clrnet_r101_culane.pth` exists; pipeline prints module status at startup |
| SLURM job pending forever | `sinfo` to see free partitions; add `#SBATCH --partition=<gpu_partition_name>` |

Every module has a `__main__` smoke test or prints its load status — test modules **one at a time** before running the full pipeline: tracker → detector → lanes → depth → classifiers → adas_final.

---

# 9. DAY-BY-DAY CHECKLIST

```
DAY 1: Setup (§1) + start ALL dataset downloads (§3.1 + Plan A or B)
       + register nuScenes (instant) + BDD100K attempt (Plan A)
DAY 2: Convert datasets (prepare_*.py) + merge + SMOKE TEST Job A (1 epoch)
DAY 3: Submit Job A + Job B → both run unattended 1-2 days
       Meanwhile: Jobs C, D (classifiers) + Job E (weight downloads)
DAY 4: Monitor training; test pipeline modules one-by-one with pretrained weights
DAY 5: Job A/B finish → copy weights → run Job F evaluation
DAY 6: Demo videos (4 clips, §5) + overtaking validation (50 clips, §7)
DAY 7: Fill results table (§7) + commit + push + update METHODOLOGY_REPORT
```

---

**Sab kuch is doc me hai. Order me follow karo, har step ke baad commit karo. Kuch atke to har module ka smoke test hai — usse pehle wo chalao.**

*Prepared by Vishal — IBM Internship 2026, Phase 3 handover*
