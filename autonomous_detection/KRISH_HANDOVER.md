# KRISH — COMPLETE EXECUTION HANDOVER
## Everything to do, in order, with exact commands and links
### Read this top to bottom, then execute. Nothing else needed.

**What's already done:** All code is written and in this repo. Your job = download datasets, run scripts, train on HPC, collect results, push.

**Two paths in this doc:**
- **PLAN A** — with BDD100K (best results; needs registration at Berkeley, can take 1-2 days for approval)
- **PLAN B (FALLBACK)** — KITTI + nuScenes + Kaggle datasets only, **zero registration walls except nuScenes (instant approval)**. Use this if BDD100K approval doesn't come or download fails.

Both plans produce the complete end product. Plan B loses ~nothing on features, slightly less night-training data.

---

## STATUS — READ THIS FIRST (what you reported, what it was, what's true now)

You reported two things after Phase 3: **too many fake detections** and
**lane detection basically not working on Indian videos**. Here's exactly
what was wrong and what's actually fixed vs. what still needs you to run a
command.

| # | You reported / we found | Root cause (plain language) | Is it fixed? |
|---|---|---|---|
| 1 | CLRNet lane lines looked like garbage / nonsense coordinates | A copy-paste math bug always re-scaled lane points by the frame width even when they were already in pixel units — blowing the numbers up | ✅ **Fixed in code, nothing to do.** Just re-run the pipeline. |
| 2 | (found during review, not reported, but safety-relevant) Overtaking sometimes said POSSIBLE when an oncoming car was actually close and fast | The code never measured how fast oncoming-lane vehicles were approaching — it silently assumed they were standing still | ✅ **Fixed in code, nothing to do.** |
| 3 | (found during review) Collision alerts sometimes fired for things nowhere near the car | A guard condition skipped the "which objects are actually in my lane" check whenever lane detection had no output for a frame | ✅ **Fixed in code, nothing to do.** |
| 4 | Overtake "possible on this curve" felt arbitrary, especially on mountain/curvy roads | The curve check compared raw pixel numbers with no real-world meaning — a camera-resolution-dependent guess, not an actual road-curve radius | ✅ **Fixed in code.** ⚠️ Works better if you do the 1-time camera calibration in §1.5 (skippable — falls back safely without it). |
| 5 | (found during review) `import data` crashed for some workflows | Unrelated pre-existing typo in `data/__init__.py`, from before Phase 3 | ✅ **Fixed in code, nothing to do.** |
| 6 | **Real objects misclassified on Indian videos** (an autorickshaw/cow/hand-cart gets called "car" or "misc") | Our detector was only ever trained on KITTI (Germany) + COCO — it has **never seen** an autorickshaw, a cow on the road, or a hand-cart, so it force-fits them into the nearest class it knows. Confirmed via research — this is a well-known, published domain-gap problem (IDD paper, WACV 2019), not a bug in our pipeline. | ⚠️ **Code is ready (15-class taxonomy + 2 India dataset converters), but NOT fixed yet on its own** — you need to download IDD and/or DriveIndia and re-run Job A (§3.4, Part 1) so the detector actually learns these classes. Nothing will change until that training runs. |
| 7 | **"Lane detection doesn't work at all" on Indian videos** (main complaint) | CLRNet/UFLDv2 are trained on CULane, which assumes a continuous painted lane line to fit a curve to. Indian roads frequently have faded/absent/ignored markings — there's often nothing there for the model to find. No amount of retraining CLRNet fixes this. | ✅ **Fixed and ACTIVE automatically, nothing to do.** Added a "drivable-area" fallback (`models/drivable_area.py`) that segments the road surface instead of hunting for paint — it kicks in automatically whenever CLRNet finds fewer than 2 lines, using a built-in no-training CV method. Training it further (§3.4, Part 2) makes it more accurate but isn't required for it to work. |
| 8 | **"Something is detected where there's literally nothing"** — a different failure mode from #6: not a misclassified real object, a box on EMPTY road/background (billboard, hoarding, reflection, glare, shopfront poster) | This is called a "phantom"/"hallucinated" detection in the safety literature. A detector shown out-of-distribution scenes gets less reliable at knowing when it's actually confident — a car-shaped billboard/poster can trigger a genuine-looking box even though there's no real 3D car there. | ✅ **Fixed and ACTIVE automatically, nothing to do**, plus one dial you can tune. Added a geometric size-consistency check (`inference/sanity_filter.py`): using the depth map we already compute, it works out the REAL-WORLD size implied by each box's pixel-width + distance (pinhole camera math), and throws out anything wildly too big/small for its class (a "car" box that would have to be 15m wide to be real, at its measured distance, isn't a real car). Also made the confidence threshold tunable: `--conf 0.45` (up from default 0.35) makes detection stricter overall if you're still seeing too many phantoms — see §3.4 Part 3 below. |

**In one line:** items 1-5, 7, and 8 are done — pull the branch and they
just work (item 8 has an optional `--conf` dial to tune further). Item 6
(misclassified Indian-specific objects) needs YOU to run the dataset
downloads + a re-train in **Section 3.4 (PLAN C)** — the code can't fix a
missing-training-data problem by itself, only real training data can.

**Prove it to yourself in 2 minutes, no GPU needed:**
```bash
python scripts/verify_adas_pipeline.py
# Expect: ALL CHECKS PASSED (27/27)
```

Full technical detail on every row above (which file, exact diff reasoning,
paper citations) is in **Section 3.4 (PLAN C)** and inline code comments —
this table is the fast version.

---

## THE COMPLETE FLOW — YOUR ROADMAP (read this second, then follow it in order)

Everything below is one continuous path from zero to a finished, demoable
product. Each phase says roughly how long it takes, whether you need to sit
and watch it or can walk away, and which section has the full detail.

```
PHASE 0   Check if this has already been done                    (5 min, YOU WATCH)
   │      -> §PHASE 0 below. Skip Phase 1-4 entirely if you find a checkpoint.
   ▼
PHASE 1   Get the code + environment on the HPC cluster           (30 min, YOU WATCH)
   │      -> Section 1
   ▼
PHASE 2   Validate the decision logic (no GPU needed)              (2 min, YOU WATCH)
   │      -> Section 1.5   |   expect: ALL CHECKS PASSED (27/27)
   ▼
PHASE 3   Decide Plan A/B, download + convert datasets          (hours-1 day, YOU WATCH)
   │      -> Section 3 (3.1 decision table, then 3.2 or 3.3)
   │      -> ALSO do Plan C now if targeting Indian-road video (Section 3.4 Part 1)
   ▼
PHASE 4   Smoke test, then submit the two big training jobs        (36-48h, WALK AWAY)
   │      -> Section 4, Job A (detector) + Job B (lanes)
   │      -> submitted in parallel, run unattended, check back with qstat/squeue
   ▼
PHASE 5   Small jobs: classifiers + drivable-area + weight downloads (~2-6h, YOU WATCH)
   │      -> Section 4, Jobs C+D+G+E. Simplest: wait for Job A, then
   │         `qsub training/pbs/aux_and_eval.pbs` (does C+D+G+F together).
   │         Faster: run C+D+G on an interactive node WHILE Job A trains
   │         (they don't need its output, only Job F does — see Section 4.0).
   ▼
PHASE 6   Run full evaluation (needs Job A's best.pt)                  (~3h, YOU WATCH)
   │      -> Section 4, Job F -> runs/final_evaluation.json + report tables
   ▼
PHASE 7   Run the end product on real video, make demo clips          (~1h, YOU WATCH)
   │      -> Section 5
   ▼
PHASE 8   Fill the results table, commit, push                        (~1h, YOU WATCH)
          -> Section 6 (git) + Section 7 (results table)
```

### PHASE 0 — Check if this has already been done (do this FIRST, always)

Training Job A takes 36-48h. Before you spend that time, check whether a
trained checkpoint already exists somewhere on the cluster (Krish or anyone
else may have already run this):

```bash
find / -name "best.pt" -o -name "clrnet_r101_culane.pth" 2>/dev/null
find / -iname "*yolo11x_merged*" -o -iname "autonomous_detection" 2>/dev/null
ls /scratch/ /project/ /data/ 2>/dev/null   # common shared-storage locations
```

If you find one, verify it before trusting it — don't assume it's current:

```bash
python -c "from ultralytics import YOLO; m = YOLO('<found_path>/best.pt'); print(len(m.names), m.names)"
# 15 classes (car...vehicle_fallback) -> Plan C (India fix) already trained, skip to PHASE 6/7
# 11 classes (car...misc)             -> only base training done, autorickshaw/animal etc.
#                                         still won't detect -> you still need PHASE 3 (Plan C)
#                                         + a re-run of Job A, but can otherwise skip to PHASE 6
```

The fastest way to find an existing checkpoint is still just asking whoever
ran it for the absolute path — filesystem archaeology is the fallback, not
the first move.

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

Expected output: `ALL CHECKS PASSED (27/27)`. If anything says `FAIL`, stop
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
│   ├── night_enhance.py     Day/night classifier + Zero-DCE++ (CLAHE fallback)
│   └── sanity_filter.py     NEW — phantom-detection geometric size check (PLAN C, §3.4 Part 3)
├── models/
│   ├── lane_detector.py     CLRNet + UFLDv2 wrappers (lane polylines)
│   ├── aux_classifiers.py   Traffic-light state + lane-type (solid/dashed)
│   ├── ipm.py               NEW — pixel→real-meters curvature (see §1.5)
│   └── drivable_area.py     NEW — unmarked-road corridor fallback (PLAN C, §3.4)
├── scripts/
│   └── verify_adas_pipeline.py  NEW — 2-min logic smoke test, run before HPC (see §1.5)
├── data/
│   ├── prepare_kitti.py       KITTI → YOLO (already used in Week 1)
│   ├── prepare_nuscenes.py    nuScenes → YOLO (multi-camera 2D projection)
│   ├── prepare_bdd100k.py     BDD100K → YOLO (PLAN A)
│   ├── prepare_lisa_det.py    LISA → YOLO traffic-light boxes (PLAN B)
│   ├── prepare_idd.py         NEW — IDD (India) VOC-XML → YOLO (PLAN C, see §3.4)
│   ├── prepare_driveindia.py  NEW — DriveIndia YOLO → unified taxonomy (PLAN C)
│   ├── prepare_uvh26.py       NEW — UVH-26 (IISc) COCO-JSON → YOLO (PLAN C, §3.3.5)
│   └── prepare_merged.py      Merges everything → data/merged_yolo/merged.yaml
├── training/
│   ├── slurm/yolo11x_merged.sh   JOB A: main detector (8 GPU, ~36h)
│   ├── slurm/clrnet_culane.sh    JOB B: lane model (4 GPU, ~24h)
│   ├── train_aux_classifiers.py  JOBS C+D: tiny classifiers (1 GPU, <1h each)
│   └── train_drivable_area.py    JOB G: NEW — drivable-area segmentation (PLAN C, §3.4)
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

## 3.3.5 HOW WE COMPARE TO OTHER PUBLIC INDIAN-ROAD PROJECTS (researched on GitHub/arXiv)

Honest answer to "kisi aur ne kaise kiya, humse better kaise hai": on
**architecture/scope**, nobody found is doing more than this project — full
detection + tracking + lanes + depth/TTC + overtaking decision + night
enhancement in one pipeline is broader than every comparable public repo.
On **actually having trained on Indian data and measured the result**, they
are ahead of us right now, because they've run the training and we (as of
this writing) haven't yet. Specifics:

| Project | What it does | How it's ahead of us (today) |
|---|---|---|
| [UVH-26 (IISc, Nov 2025)](https://arxiv.org/abs/2511.02563) | 26,646 real Bengaluru traffic-camera images, 1.8M boxes, 14 India-specific classes with body-type granularity, fine-tuned YOLOv11/DAMO-YOLO/RT-DETRv2 | **Measured the fine-tuning gain**: up to 31.5% mAP@50:95 improvement over COCO-trained baselines. We built the same capability (Plan C, now including this dataset — see 1c above) but haven't run the training yet, so we don't have our own number to compare. |
| ["Fine-Tuning Without Forgetting" (arXiv 2505.01016)](https://arxiv.org/abs/2505.01016) + related error-analysis work | Quantifies exactly the failure mode we diagnosed: COCO-baseline YOLO/RT-DETR models "did not recognize certain vehicle classes unique to India... such as 3-wheelers and LCVs" | Independent third-party confirmation (beyond the IDD paper itself) that our root-cause diagnosis was correct. RT-DETR-X fine-tuned on Indian data reached 0.67 mAP@50:95 vs. 0.40 for the COCO-trained baseline on shared classes (Car/Bus/Truck) — again, a number we don't have yet ourselves. |
| [AdroitAnandAI/ADAS-Car-using-Raspberry-Pi](https://github.com/AdroitAnandAI/ADAS-Car-using-Raspberry-Pi) | Real hardware ADAS on Indian roads: Raspberry Pi + actual LiDAR + camera, low-level sensor fusion for collision avoidance | Uses REAL LiDAR for depth, which is inherently more accurate/robust than our monocular depth estimation (Depth Anything V2) — the right tradeoff if you're building physical hardware. We're a software/video-analysis pipeline (matches this project's actual scope — nobody asked for a physical sensor rig), so monocular depth is the correct choice here, just worth knowing the ceiling if this ever becomes a hardware project. |
| IDD (WACV 2019) + DriveIndia (2025) | The two India-specific datasets we already integrated (Section 3.4 Part 1) | We're not behind here — both are already wired into `prepare_merged.py`. |

**The one real, actionable gap this research surfaced:** we hadn't added
UVH-26 as a data source. It's now integrated (`data/prepare_uvh26.py`,
Section 3.4 Part 1c) — it's the freshest and largest of the three India
sources, so combining all three (IDD + DriveIndia + UVH-26) should give
more training signal per class than any of them alone.

**The one thing we're already doing right that's worth knowing:** the
"Fine-Tuning Without Forgetting" paper's whole premise is that naively
fine-tuning on new data risks catastrophically forgetting the original
(COCO/KITTI) classes. `prepare_merged.py` already avoids this by construction
— it trains on KITTI + India sources TOGETHER in one merged dataset (joint
training), not as a sequential "train on KITTI, then fine-tune on India"
step, which is the pattern that actually risks forgetting.

---

## 3.4 PLAN C — FIX FOR "FAKE DETECTIONS" + "LANE DETECTION DOESN'T WORK" ON INDIAN VIDEOS

**Do this if you're testing/deploying on Indian road footage** (in addition
to Plan A or B above, not instead of).

### Why this happens — it's not a bug in our pipeline

KITTI (Karlsruhe, Germany) and CULane (Chinese highways) were both captured
in structured, rule-following traffic with continuous painted lane markings
and a small, homogeneous set of vehicle types. This is a well-documented,
specifically-named research problem, not something specific to our code:

> "Datasets like KITTI, Cityscapes, Argoverse, and nuScenes are captured in
> developed countries where infrastructure is well-developed and road
> activity is structured... results obtained from these datasets are often
> not directly applicable in unstructured road situations prevalent in
> large parts of the world."
> — Varma et al., *IDD: A Dataset for Exploring Problems of Autonomous
> Navigation in Unconstrained Environments*, WACV 2019
> ([arxiv.org/abs/1811.10200](https://arxiv.org/abs/1811.10200))

That paper's own experiments show segmentation models trained on
Cityscapes-style data score much lower on Indian roads than on their native
distribution — i.e. the exact domain gap producing the two symptoms
reported:

- **"Fake detections"** — a detector that has never seen an autorickshaw,
  a cow on the road, or a hand-cart either misses it or force-fits it into
  the nearest class it does know (car/van/misc). That's not random noise,
  it's a systematic classification error from a missing class.
- **"Lane detection doesn't work at all"** — CLRNet/UFLDv2 (`models/lane_detector.py`)
  are trained on CULane, which assumes a continuous, clearly painted lane
  line to fit a curve through. On roads where markings are faded, absent,
  or simply not followed (very common outside Indian highways), there is
  often nothing there for the model to find — no amount of retraining
  CLRNet itself fixes this, because the assumption behind its whole output
  representation doesn't hold.

### The fix — two parts, both already wired into the code in this branch

**Part 1 — Indian-specific detection classes (fixes fake detections).**
The unified taxonomy in `data/prepare_merged.py` grew from 11 to 15
classes: `autorickshaw`, `animal`, `rider`, `vehicle_fallback` (IDD's
open-world bucket for street cart / tractor / water tanker / excavator —
IDD's own paper describes exactly this expansion for the same reason).
Fine-tune on real Indian data so the detector actually learns these:

```bash
# 1a. IDD (India Driving Dataset) — 10K-47K images, 15-class detection subset,
#     PASCAL-VOC XML format, IIIT Hyderabad + Intel, WACV 2019.
#     Register (free, ~1 day approval): https://idd.insaan.iiit.ac.in/
#     Download "IDD Detection", extract to data/IDD_Detection/, then:
python data/prepare_idd.py --data_root data/IDD_Detection --out data/idd_yolo

# 1b. DriveIndia — newer (2025), LARGER (66,986 images, 24 classes), and
#     already in YOLO format (no bbox math needed, just a class remap).
#     TiHAN-IIT Hyderabad: https://tihan.iith.ac.in/tiand-datasets/
#     Paper: https://arxiv.org/abs/2507.19912
#     Extract to data/DriveIndia/ (needs the dataset's own data.yaml — this
#     script reads its real class names rather than guessing IDs), then:
python data/prepare_driveindia.py --data_root data/DriveIndia --out data/driveindia_yolo

# 1c. UVH-26 (IISc, Nov 2025) — found while researching how other projects
#     solved this exact problem (see comparison below). 26,646 REAL Bengaluru
#     traffic-camera images, 1.8M boxes, COCO JSON format, no registration
#     wall (HF account only). Body-type granularity we don't need (Hatchback/
#     Sedan/SUV/MUV) gets collapsed into car/van by this script.
#     https://huggingface.co/datasets/iisc-aim/UVH-26
python data/prepare_uvh26.py --data_root data/UVH26 --out data/uvh26_yolo

# 1d. Re-merge (adds to whatever Plan A/B sources you already have):
python data/prepare_merged.py --out data/merged_yolo \
  --idd data/idd_yolo --driveindia data/driveindia_yolo --uvh26 data/uvh26_yolo
# You'll see a WARNING in the output if idd/driveindia/uvh26 are ALL missing
# — that means autorickshaw/animal/rider/vehicle_fallback get ZERO training
# examples and will never be detected. Don't skip this if you're testing on
# Indian footage. Any ONE of the three is enough to silence the warning, but
# more sources = better coverage — no reason not to use all three if you can.

# 1e. Re-train (same Job A as before, just with the enlarged merged dataset —
# see KRISH_HANDOVER.md Section 4 Job A). Even a partial-epoch fine-tune from
# your existing best.pt checkpoint on just the new classes helps a lot more
# than training from yolo11x.pt COCO weights again.
```

**Part 2 — drivable-area segmentation (fixes lane detection on unmarked
roads).** Every independent project working on this problem reaches the
same conclusion: stop asking "where are the lane lines" and ask "which part
of the image is drivable road surface" instead — see e.g.
[moatifbutt/Drivable-Road-Region-Detection](https://github.com/moatifbutt/Drivable-Road-Region-Detection-and-Steering-Angle-Estimation-Method),
[AbhayVAshokan/Semantic-Segmentation-of-Road-Surface](https://github.com/AbhayVAshokan/Semantic-Segmentation-of-Road-Surface)
(IDD-based). `models/drivable_area.py` implements exactly this, and
`inference/adas_final.py` already calls it automatically whenever CLRNet/
UFLDv2 return fewer than 2 usable lane lines — **you don't need to change
any pipeline code**, just optionally train it for better accuracy than the
built-in classical-CV fallback (which works with zero weights, just less
precisely around shadows/glare):

```bash
# One-time: convert IDD Segmentation's multi-class label PNGs into simple
# binary drivable/not-drivable masks. You need the correct pixel-ID(s) for
# "road"/"drivable fallback" in YOUR IDD release — get these from the
# AutoNUE devkit (github.com/AutoNUE/public-code, helpers/anue_labels.py),
# which ships with the dataset. Don't guess a number here — a wrong id
# silently trains the model on the wrong thing.
python training/train_drivable_area.py prepare \
  --idd_seg data/IDD_Segmentation --drivable_ids <see anue_labels.py> \
  --out data/drivable_binary

# Train (small model, ~1-2h on a single GPU):
python training/train_drivable_area.py train --data data/drivable_binary \
  --epochs 40 --batch 32
# -> weights/drivable_area.pth (auto-loaded by adas_final.py next run)
```

This is now folded into Job "C+D+G+F" — `training/pbs/aux_and_eval.pbs`
runs it automatically if `data/drivable_binary/` exists, skips with a
message otherwise.

**What this does NOT change:** overtaking legality (`inference/overtaking.py`
Rule 1) still fails safe to "NOT POSSIBLE" without real painted-lane-type
info — crossing into oncoming traffic on an unmarked road is a genuinely
higher-risk judgment call this project intentionally doesn't make. The
drivable-area fallback only improves collision-alert accuracy (knowing
what's actually in your path) and gives the HUD something to draw instead
of nothing.

**Verify it worked:** `python scripts/verify_adas_pipeline.py` includes
regression tests for the drivable-area fallback (finds a corridor on a
synthetic unmarked-road image, both with and without trained weights).

**Part 3 — phantom/hallucinated detections (fixes "something is detected
where there's literally nothing").** This is a DIFFERENT problem from Part
1 — it's not a real object getting the wrong class label, it's a box drawn
on empty road, a billboard, a reflection, or glare. This is a documented
failure mode in AV safety research, sometimes literally called a
"hallucination" (see the PhantomPerception hallucination-injection safety
framework, [arxiv.org/html/2510.07749v1](https://arxiv.org/html/2510.07749v1)) —
out-of-distribution scenes make a detector's confidence scores less
trustworthy, so it fires on things that resemble a class without actually
being a 3D instance of it.

The fix (`inference/sanity_filter.py`) is a geometric consistency check —
a standard technique from monocular 3D detection research (see "Exploring
Geometric Consistency for Monocular 3D Object Detection",
[arxiv.org/abs/2104.05858](https://arxiv.org/abs/2104.05858)): using the
depth map already computed for collision detection, work out how wide the
detected object would have to be in the REAL world to produce that
pixel-width box at that measured distance (basic pinhole camera math), and
reject anything wildly outside a generous plausible range for its class. A
billboard "car" or a reflection doesn't have the depth profile of an actual
3.5m-wide 3D object at its apparent size, so this catches it.

This runs **automatically, nothing to configure** — but two dials help if
you're still seeing too many (or too few) rejections:

```bash
# Stricter overall confidence (fewer detections attempted in the first place):
python inference/adas_final.py --source video.mp4 --conf 0.45   # default is 0.35

# More ACCURATE size-consistency filtering (uses real camera geometry
# instead of the ~90° HFOV guess it falls back to):
python inference/adas_final.py --source video.mp4 \
  --kitti_calib data/kitti/training/calib/000000.txt   # any KITTI calib_cam_to_cam.txt-style file
```

At the end of every run, `adas_final.py` prints how many detections got
rejected and for which classes — if that number looks way too high (you
suspect it's rejecting real objects), raise `margin` in
`inference/sanity_filter.py`'s `SizeConsistencyFilter`; if you're still
seeing obvious phantoms, tighten it or raise `--conf`.

**Verify it worked:** `scripts/verify_adas_pipeline.py` includes tests
proving a billboard-scale "car" gets rejected while a normal-sized one and
unbounded classes (animal/misc/etc.) pass through untouched.

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
| C+D+G+F (classifiers+drivable-area+eval) | run interactively (Jobs C+D+G+F below) | `qsub training/pbs/aux_and_eval.pbs` |

Job E (depth/night weight downloads) is separate on both schedulers — it's
just `git clone` + manually grabbing a checkpoint file, not a training job,
so it doesn't need `sbatch`/`qsub` at all. Do it anytime, no GPU needed
(see Job E below).

PBS monitoring: `qstat -u $USER` (status), `qdel <jobid>` (cancel),
`tail -f logs/yolo11x_merged.log` (live output). If your cluster's PBS needs a
queue name, add `#PBS -q <queue>` (find queues with `qstat -Q`).

## Job order (A and B run in parallel; C/D/G need A's output; F is last)

```
   ┌── JOB A: detector (8 GPU, ~36h) ──┐
   │                                    ├──▶ JOB C+D: classifiers (1 GPU, 1.5h)
   └── JOB B: lanes    (4 GPU, ~24h) ──┘    JOB G: drivable-area (1 GPU, ~1-2h, PLAN C)
                                                  │
   JOB E: weight downloads — no GPU, do anytime, doesn't block anything
                                                  ▼
                                          JOB F: evaluation (1 GPU, ~3h,
                                                  needs Job A's best.pt)
```

On PBS, Jobs C+D+G+F are bundled as one script
(`qsub training/pbs/aux_and_eval.pbs`) for convenience — it does each one in
sequence and skips any whose input data isn't there yet (e.g. Job G skips
itself if you haven't done the Plan C drivable-area `prepare` step from
Section 3.4 Part 2). Only Job F actually needs Job A's `best.pt`, so you
have two options:
- **Simple (recommended):** submit the bundle only after Job A finishes.
  You lose a bit of parallelism (C/D/G could've run earlier) but it's one
  command and always correct.
- **Faster:** run Jobs C+D (see "JOBS C + D" below) and Job G's `train`
  step manually on an interactive node WHILE Job A is still training, then
  submit just Job F once `best.pt` exists.

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

This is the same flow as "THE COMPLETE FLOW" roadmap near the top of this
doc, spread across a realistic week. Start with **PHASE 0** — check for an
existing trained checkpoint before you commit to any of this.

```
DAY 0: PHASE 0 — search for/ask about an existing trained checkpoint (5 min).
       Found one with 15 classes? Skip to DAY 5. Found one with 11 classes?
       You still need DAY 1-3 (Plan C data) + a re-run of Job A, but can
       skip straight to DAY 4's monitoring once that's submitted.

DAY 1: Setup (§1) + validate logic (§1.5, expect 27/27) + start ALL dataset
       downloads (§3.1 + Plan A or B) + register nuScenes (instant) and
       BDD100K (Plan A, ~2 day approval wait — start this early)
       + if targeting Indian-road video: register IDD (§3.4 Part 1, ~1 day
       approval) and start the DriveIndia/UVH-26 downloads (no wait)

DAY 2: Convert datasets (prepare_*.py, including prepare_idd.py /
       prepare_driveindia.py / prepare_uvh26.py if doing Plan C) + merge
       (prepare_merged.py — check the console for the "no India data" warning)
       + SMOKE TEST Job A (1 epoch)

DAY 3: Submit Job A + Job B → both run unattended 1-2 days
       Meanwhile: Jobs C, D, G (classifiers + drivable-area, see §4.0's
       "faster" option) + Job E (weight downloads, no GPU needed)

DAY 4: Monitor training (qstat/squeue); test pipeline modules one-by-one
       with pretrained weights while you wait

DAY 5: Job A/B finish → run Job F evaluation (§4, needs Job A's best.pt)

DAY 6: Demo videos (4 clips, §5) + overtaking validation (50 clips, §7)
       + if Indian-road footage is in scope, specifically test a phantom-
       detection clip and an unmarked-road clip to confirm §3.4 Parts 2-3
       are behaving (drivable-area corridor drawn, sanity_filter rejection
       count printed at the end of the run)

DAY 7: Fill results table (§7) + commit + push + update METHODOLOGY_REPORT
```

---

**Sab kuch is doc me hai. Order me follow karo, har step ke baad commit karo. Kuch atke to har module ka smoke test hai — usse pehle wo chalao.**

*Prepared by Vishal — IBM Internship 2026, Phase 3 handover*
