# ADAS PERCEPTION PIPELINE — PRESENTATION SOURCE CONTENT
### IBM Internship 2026 · Vishal Bansal, Krish
### Every number below is measured, not projected. Sources noted inline.

---

## 0. VISUAL ASSETS — ASK ME FOR THESE BEFORE BUILDING SLIDES

**To whoever is building this deck: request each asset below from me before you
lay out its slide.** I have all of them. Do not substitute stock imagery,
generated illustrations, or placeholder graphics — every visual in this deck must
be genuine output from the system, because the audience will ask how it was
produced.

If I have not supplied an asset when you reach its slide, leave a clearly marked
empty frame with the caption in place rather than filling the space.

| # | Asset | Filename I will send | Slide it belongs on | What it proves |
|---|---|---|---|---|
| A1 | Collision alert — BRAKE | `BRAKE_*.jpg` | Results → collision detection | A real frame where the system raised a brake alert, with the box, track ID and TTC drawn by the pipeline |
| A2 | Collision alert — WARNING | `WARNING_*.jpg` | Results → collision detection | The lower-severity tier of the same mechanism |
| A3 | Dense traffic frame | `BUSY_*.jpg` | Results → detection quality | Many simultaneous objects tracked at once |
| A4 | Demo video clip | `DEMO_highlight.mp4` (~30 s) | Live demo slide | The full pipeline running end to end |
| A5 | Confusion matrix | `confusion_matrix_normalized.png` | Results → per-class analysis | Where classes are confused with each other |
| A6 | Training curves | `results.png` | Results → training progression | Loss convergence and mAP climb over epochs |
| A7 | Precision–recall curve | `PR_curve.png` | Results → detection performance | Operating-point behaviour per class |
| A8 | Class distribution | `labels.jpg` | Data pipeline | The long-tailed class balance we trained against |
| A9 | Architecture diagram | *(build from Section 3 — no image needed)* | Methodology | The six-stage flow |

**Ask me for A1–A8 explicitly.** A9 you should draw yourself from the table in
Section 3.

---

## 1. PROBLEM STATEMENT

Driver-assistance systems that work reliably in Europe, the US, or China degrade
sharply on Indian roads. The cause is not weaker hardware — it is that the
datasets these systems are built on encode a road environment that does not
match ours.

**Three concrete failures this creates:**

1. **Missing vehicle categories.** A detector trained on KITTI has never seen an
   autorickshaw, a cow on the carriageway, or a hand-cart. It does not skip them;
   it forces them into the nearest class it knows — a "car" or "misc" box on a
   three-wheeler. A wrong label is more dangerous than no label, because
   downstream logic acts on it.

2. **Lane detection has nothing to detect.** Lane-line models assume a continuous
   painted marking to fit a curve through. On roads where markings are faded,
   absent, or simply not followed, that assumption fails outright — and no amount
   of retraining the lane model repairs it, because the thing it is built to find
   is not in the scene.

3. **Phantom detections.** Out-of-distribution scenes make a detector's confidence
   less trustworthy. It fires boxes on car-shaped hoardings, shopfront posters,
   and reflections in wet asphalt — objects that are not physically there at all.

**What this project builds:** a single-pass monocular camera pipeline that
detects and tracks road users, reads the drivable corridor, estimates metric
depth, and converts all of it into two decisions a driver can act on —
*is a collision imminent*, and *is it safe to overtake* — with every stage
degrading to a weaker method rather than failing outright.

---

## 2. LITERATURE REVIEW

> **Builder's note — 10-slide deck:** this section gets **one main slide**, not
> four. Put the quotation in 2.0 and the summary table in 2.5 on it — those two
> together show the survey drove real design decisions, which is the point.
> Move the full tables (2.1–2.4) to **backup slides after the last numbered
> slide**, so they can be jumped to if a question demands the detail.

### 2.0 The one quotation to put on a slide

The entire premise of this project rests on a finding that is already published
and independently replicated, which is worth stating in the authors' own words:

> *"Datasets like KITTI, Cityscapes, Argoverse, and nuScenes are captured in
> developed countries where infrastructure is well-developed and road activity
> is structured... results obtained from these datasets are often not directly
> applicable in unstructured road situations prevalent in large parts of the
> world."*
>
> — Varma et al., **IDD: A Dataset for Exploring Problems of Autonomous
> Navigation in Unconstrained Environments**, WACV 2019

This is why the system's Indian-road behaviour was treated as a *diagnosed
research problem* rather than an implementation bug — and why the fix is a data
strategy, not a patch.

### 2.1 The domain-gap problem (the foundation of this work)

| Work | Venue / ID | Contribution | How we used it |
|---|---|---|---|
| Varma et al., **IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments** | WACV 2019 · [arXiv:1811.10200](https://arxiv.org/abs/1811.10200) | Establishes that KITTI/Cityscapes/nuScenes-trained models transfer poorly to unstructured roads. Introduces India-specific classes and a "drivable fallback" label. | Diagnosed our root cause; motivated our 15-class taxonomy extension and the drivable-area reframing |
| **DriveIndia** | 2025 · [arXiv:2507.19912](https://arxiv.org/abs/2507.19912) · TiHAN-IIT Hyderabad | 66,986 images, 24 Indian traffic classes, YOLO format, 3,400+ km of driving | Integrated as a data source (`data/prepare_driveindia.py`) |
| **UVH-26** (Urban Vision Hackathon) | Nov 2025 · [arXiv:2511.02563](https://arxiv.org/abs/2511.02563) · AIM@IISc | 26,646 Bengaluru traffic-camera images, 1.8M boxes, 14 India-specific classes. Reports **up to 31.5% mAP@50:95 improvement** over COCO baselines | Integrated as a data source (`data/prepare_uvh26.py`); its measured gain is our target |
| **Fine-Tuning Without Forgetting** | 2025 · [arXiv:2505.01016](https://arxiv.org/abs/2505.01016) | Shows COCO-trained YOLO/RT-DETR "did not recognize vehicle classes unique to India such as 3-wheelers and LCVs"; RT-DETR-X reached 0.67 mAP@50:95 fine-tuned vs 0.40 COCO-trained | Independent confirmation of our diagnosis; motivated joint (not sequential) training |

### 2.2 Component models adopted

| Model | Venue | Why chosen |
|---|---|---|
| **YOLOv11** (Ultralytics, 2024) | — | Anchor-free, strong speed/accuracy trade-off; the x variant gives headroom on an H100 |
| **CLRNet** (Zheng et al.) | CVPR 2022 | Cross-layer refinement; strongest published CULane F1 on the Curve and Night categories — the two that matter for mountain and night driving |
| **ByteTrack** (Zhang et al.) | ECCV 2022 | Associates low-confidence detections too, which keeps briefly-occluded vehicles alive — critical for TTC continuity |
| **Depth Anything V2** (Yang et al.) | NeurIPS 2024 | Metric monocular depth; removes the need for LiDAR to get distances in real metres |
| **Zero-DCE++** (Li et al.) | — | Zero-reference low-light enhancement; no paired night data needed |

### 2.3 Techniques adopted from the literature

| Technique | Source | Applied to |
|---|---|---|
| Geometric consistency for rejecting implausible detections | [arXiv:2104.05858](https://arxiv.org/abs/2104.05858) — *Exploring Geometric Consistency for Monocular 3D Object Detection* | Our phantom-detection filter |
| Detector "hallucination" as a named AV safety failure mode | [arXiv:2510.07749](https://arxiv.org/html/2510.07749v1) — PhantomPerception | Framing and mitigation of false-positive boxes |
| Drivable-region segmentation instead of lane lines on unmarked roads | IDD "drivable fallback" class; [moatifbutt/Drivable-Road-Region-Detection](https://github.com/moatifbutt/Drivable-Road-Region-Detection-and-Steering-Angle-Estimation-Method) | Our lane fallback path |
| Inverse Perspective Mapping for ground-plane geometry | Standard ADAS technique (Hartley & Zisserman, plane-induced homography) | Curvature in real metres, compared against AASHTO/IRC road-design radii |

### 2.4 Comparable systems

| System | What it does | Honest comparison |
|---|---|---|
| [AdroitAnandAI/ADAS-Car-using-Raspberry-Pi](https://github.com/AdroitAnandAI/ADAS-Car-using-Raspberry-Pi) | Physical ADAS rig on Indian roads: Raspberry Pi + RP LIDAR + camera, low-level sensor fusion | **Ahead of us on depth** — real LiDAR beats monocular estimation. Different scope: hardware build vs. our software pipeline |
| UVH-26 baseline models | YOLOv11 / DAMO-YOLO / RT-DETRv2 fine-tuned on Indian data | **Ahead of us on Indian-data training** — they have run it and measured 31.5%; we have built the capability but not yet retrained |
| This project | Six-stage pipeline: detection + tracking + lanes + depth/TTC + overtaking + night, in one pass | **Ahead on scope** — no comparable public repo covers all six stages with decision output |

### 2.5 Summary of the survey — what the literature told us to do

| Finding in the literature | What we changed because of it |
|---|---|
| Western/Chinese benchmarks do not transfer to unstructured roads (IDD, WACV 2019) | Extended the taxonomy 11 → 15 classes; built converters for three Indian datasets |
| COCO-trained detectors miss 3-wheelers and LCVs entirely (arXiv:2505.01016) | Confirmed our own failure mode was this, not a bug — stopped debugging code and started fixing data |
| Sequential fine-tuning risks forgetting the original classes (arXiv:2505.01016) | Merged all sources into one dataset and trained jointly instead |
| Fine-tuning on Indian data yields up to +31.5% mAP@50:95 (UVH-26, IISc 2025) | Set that as the measurable target for the retraining step |
| Detectors hallucinate objects on out-of-distribution scenes (PhantomPerception) | Added a geometric plausibility filter rather than only raising the confidence threshold |
| Apparent size and estimated depth must agree for a detection to be physical (arXiv:2104.05858) | That agreement check *is* the filter's mechanism |
| Unstructured roads need drivable-region segmentation, not lane lines (IDD "drivable fallback") | Built the segmentation fallback that activates automatically |

---

## 3. METHODOLOGY — SIX-STAGE PIPELINE

All six stages run in order on every frame. Each consumes the frame plus the
outputs above it; no second pass over the video is needed.

| Stage | Function | Model / method | Output |
|---|---|---|---|
| **0** | Scene lighting and low-light enhancement | Zero-DCE++, CLAHE fallback | Enhanced frame, lighting state (day / lit night / unlit night) |
| **1** | Object detection | YOLOv11x @ 640px | Boxes, class, confidence |
| **1b** | Geometric plausibility filter | Pinhole size check against depth | Filtered detections (phantoms removed) |
| **2** | Lane and drivable corridor | CLRNet R101, drivable-surface segmentation fallback | Lane polylines, ego-lane boundaries |
| **3** | Multi-object tracking | Two-stage IoU association with smoothed velocity | Persistent track IDs, velocity |
| **4** | Metric depth and time-to-collision | Depth Anything V2 (metric) | Distance (m), closing speed (m/s), TTC (s), BRAKE / WARNING |
| **5** | Overtaking decision | Five-rule safety fusion (no network) | POSSIBLE / NOT POSSIBLE with blocking reason |

### 3.1 Design decisions worth defending

**Closing speed from depth, not pixel motion.** An object growing in the frame is
not the same as an object approaching. Distance is read from the depth map at
each object's road-contact region (the lower half of the box — the upper half
catches sky reflections in windscreens), and closing speed is the rate that
distance changes. Pixel velocity would give a plausible-looking but physically
wrong TTC.

**Curvature in metres, not pixels.** Pixel curvature shifts with camera
resolution and mounting, so the same road yields different numbers on different
cameras. The pipeline projects lane geometry onto the ground plane via IPM and
compares the resulting radius against published road-design standards
(AASHTO / IRC minimum horizontal curve radius). That is what makes a
"too tight to overtake" verdict defensible rather than arbitrary.

**Every stage fails safe.** Any missing model, weight file, or uncertain input
resolves toward the conservative answer — and never aborts the pipeline. This
was validated in practice: on the cluster, the lane model failed to load and the
pipeline completed all 3,604 frames using the fallback path.

**Joint training, not sequential fine-tuning.** All data sources are merged into
one dataset and trained together, rather than training on KITTI and then
fine-tuning on Indian data. The latter is the pattern that risks catastrophic
forgetting of the original classes (arXiv:2505.01016).

---

## 4. DATA PIPELINE

### 4.1 Preprocessing

| Step | Detail |
|---|---|
| Format conversion | KITTI label format → YOLO normalized xywh |
| Validation | Bounding boxes clipped to image bounds; boxes under 2 px in either dimension discarded |
| Unified taxonomy | Source-specific class names remapped to one shared ID space so datasets can be merged |
| Split | Train / validation, stratified; **5,985 train / 1,496 validation images** |
| Augmentation | Mosaic 1.0, MixUp 0.15, Copy-Paste 0.3, HSV value 0.6, rotation ±5°, translate 0.1, scale 0.5, horizontal flip 0.5 |

### 4.2 Unified class taxonomy

Extended from 11 to **15 classes** to cover Indian road users:

```
0 car          4 pedestrian    8  traffic_light   12 animal
1 truck        5 cyclist       9  traffic_sign    13 rider
2 bus          6 motorcycle    10 misc            14 vehicle_fallback
3 van          7 tram                                (cart/tractor/tanker)
```

Classes 11–14 (`autorickshaw`, `animal`, `rider`, `vehicle_fallback`) were added
specifically to close the domain gap identified in the IDD paper.

### 4.3 Data sources integrated

| Source | Status | Purpose |
|---|---|---|
| KITTI | ✅ Trained on | Detection baseline |
| IDD (IIIT Hyderabad) | Converter ready | Indian classes, PASCAL-VOC XML → YOLO |
| DriveIndia (TiHAN-IIT-H) | Converter ready | 66,986 Indian images, YOLO native |
| UVH-26 (AIM@IISc) | Converter ready | 26,646 Bengaluru images, COCO JSON → YOLO |
| CULane | Used (pretrained) | Lane detection weights |

> **[ASK ME FOR ASSET A8 — `labels.jpg`]**
> Caption: "Class distribution in the training set — the long tail the model had
> to handle without collapsing on rare classes."

---

## 5. RESULTS

### 5.1 Headline

| Metric | Value |
|---|---|
| **mAP@0.5** | **95.4%** |
| **mAP@0.5:0.95** | **79.9%** |
| Precision | 95.4% |
| Recall | 92.5% |
| Detector inference | 0.8 ms / image |
| **Full six-stage pipeline** | **23.1 FPS** (43 ms mean latency) |
| Validation set | 1,496 images · 8,104 instances |
| Model | YOLOv11x · 56.8M parameters · 194.5 GFLOPs · 191 layers |
| Hardware | NVIDIA H100 80GB, PBS cluster |

### 5.2 Per-class detection performance

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Car | 5,798 | 95.7% | 96.6% | 97.9% | 87.6% |
| Pedestrian | 884 | 95.0% | 81.3% | 89.0% | 55.9% |
| Van | 587 | 94.9% | 97.3% | 97.5% | 86.0% |
| Cyclist | 313 | 94.3% | 88.2% | 92.7% | 71.0% |
| Truck | 219 | 97.5% | 98.6% | 99.4% | 91.5% |
| Misc | 194 | 95.1% | 92.3% | 94.3% | 80.8% |
| Tram | 109 | 95.3% | 93.1% | 96.8% | 86.7% |
| **All** | **8,104** | **95.4%** | **92.5%** | **95.4%** | **79.9%** |

**Two things to point out from this table:**
- **Rare classes did not collapse.** Truck (219 instances) and tram (109) score at
  or above the fleet average despite being ~25× rarer than car. This is normally
  where long-tailed driving datasets fail.
- **Pedestrian localisation is the weakest column** (55.9% at the strict metric).
  The class is *found* reliably — 95% precision — but boxed less tightly than
  vehicles. This is the honest limitation of the current model.

> **[ASK ME FOR ASSET A5 — `confusion_matrix_normalized.png`]**
> Place beside this table. Caption: "Normalised confusion matrix — the diagonal
> shows correct classification; off-diagonal cells show which classes get
> mistaken for each other."

> **[ASK ME FOR ASSET A7 — `PR_curve.png`]**
> Caption: "Precision–recall behaviour per class across confidence thresholds."

### 5.3 Training progression

| Run | Model | Epochs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|
| Baseline | YOLOv11m | 50 | 91.7% | 69.3% |
| Extended | YOLOv11m | 500 | 94.2% | 77.7% |
| **Final** | **YOLOv11x, merged** | **300** | **95.4%** | **79.9%** |

Worth noting: the 50 → 500 epoch step bought more than the medium → extra-large
architecture step. Useful to know before spending further GPU time.

> **[ASK ME FOR ASSET A6 — `results.png`]**
> Caption: "Training curves — box, classification and DFL loss converging, with
> mAP@0.5 and mAP@0.5:0.95 climbing over 300 epochs."

### 5.4 End-to-end pipeline run (measured)

| Measurement | Value |
|---|---|
| Frames processed | 3,604 |
| Throughput | 23.1 FPS (43 ms mean per frame) |
| **Collision alerts raised** | **69 — 20 BRAKE, 49 WARNING** |
| Lanes detected | 4 lane lines on 3,603 of 3,604 frames |
| Phantom detections rejected | 979 of 7,077 size-checked (13.8%) |
| Outputs | Annotated MP4 + per-frame JSONL decision record |

**Speed finding worth reporting:** the first measured run showed 0.9 FPS. Root
cause was that the depth model had fallen back to CPU, not a pipeline design
problem. On GPU the same code runs **25× faster**, which is the difference
between unusable and real-time.

> **[ASK ME FOR ASSETS A1, A2, A3 — `BRAKE_*.jpg`, `WARNING_*.jpg`, `BUSY_*.jpg`]**
> These are the centrepiece result visuals. Lay them out as a three-panel row.
> Captions:
> - A1 — "BRAKE alert: object closing fast inside the ego corridor. Box, track ID
>   and time-to-collision are drawn by the pipeline itself."
> - A2 — "WARNING tier: same mechanism, longer time margin."
> - A3 — "Dense traffic: multiple road users detected and tracked simultaneously."
>
> Note for the builder: these frames come straight out of the annotated video —
> every overlay on them was produced by the system, not added afterwards.

> **[ASK ME FOR ASSET A4 — `DEMO_highlight.mp4`]**
> Give this its own slide. Embed the file itself, do not link to it, or it will
> not play on another machine.

### 5.5 Engineering validation

A 62-check logic suite runs in seconds with no GPU, weights, or dataset, and is
executed before any training job is queued. It covers TTC mathematics, all five
overtaking rules, lane coordinate mapping, phantom filtering, and graceful
degradation.

**It caught real defects, including several that had already reached the
cluster** — among them a lane-coordinate mapping error where CLRNet's output
(in 1640×590 CULane space) was being drawn directly onto an 848×480 video.

---

## 6. RESEARCH GAPS IDENTIFIED

| # | Gap | Evidence | Our response |
|---|---|---|---|
| 1 | Benchmark datasets encode structured Western/Chinese roads; models transfer poorly to unstructured traffic | IDD paper (WACV 2019); independently confirmed by arXiv:2505.01016 | 15-class taxonomy + three Indian dataset converters built |
| 2 | Lane-line detection is undefined where markings are absent — the ground truth itself does not exist | CULane-trained models have nothing to fit; IDD adds a "drivable fallback" label for this reason | Drivable-surface segmentation as an automatic fallback |
| 3 | Detectors hallucinate objects on out-of-distribution scenes | PhantomPerception (arXiv:2510.07749) | Geometric size-consistency filter using existing depth |
| 4 | Lane curvature reported in pixels has no physical meaning, yet is used for safety decisions | Pixel curvature varies with camera geometry | IPM to ground plane; thresholds from AASHTO/IRC road-design standards |
| 5 | Monocular depth is less reliable than LiDAR for collision distance | Hardware ADAS systems use LiDAR | Acknowledged limitation; monocular chosen to match the software-only scope |
| 6 | Traffic-light, sign, bus and motorcycle classes have **zero training examples** in the current merge | KITTI, the only source trained on, contains none of them — validation therefore reports 7 classes, not 11 | Identified as a **data** step, not a code change; converters for sources that carry them are ready |

---

## 7. TECHNOLOGY STACK

| Layer | Technology |
|---|---|
| Detection | YOLOv11x (Ultralytics), 640px, AdamW, cosine LR |
| Lane detection | CLRNet ResNet-101 (CULane pretrained) |
| Depth | Depth Anything V2, metric outdoor checkpoint |
| Tracking | Self-contained two-stage IoU tracker with exponentially smoothed velocity |
| Low-light | Zero-DCE++ with CLAHE fallback |
| Geometry | Inverse Perspective Mapping (plane-induced homography) |
| Framework | PyTorch 2.8 + CUDA 12.8, Python 3.9 |
| Compute | NVIDIA H100 80GB, PBS-scheduled HPC cluster |
| Deployment | ONNX export at 640px |
| Verification | 62-check logic suite, GPU-free |
| Reproducibility | Per-frame JSONL decision record for every run |

---

## 8. CURRENT STATUS — HONEST

| Capability | State | Detail |
|---|---|---|
| Object detection | ✅ Live | 95.4% mAP@0.5 across 7 trained classes |
| Multi-object tracking | ✅ Live | Stable IDs, smoothed velocity |
| Depth + collision TTC | ✅ Live | 69 alerts raised across the measured run |
| Phantom-detection filter | ✅ Live | 13.8% rejection rate; tightens once calibrated |
| Lane detection | ✅ Live | CLRNet loading, 4 lanes on 3,603/3,604 frames |
| Night enhancement | ⚠️ Fallback | CLAHE active; Zero-DCE++ weights not downloaded |
| Overtaking verdict | ⚠️ Fail-safe | Returns "not possible" without reliable marking-type input — correct conservative behaviour |
| Traffic-light state | ❌ Data-blocked | Class has zero training examples in the current merge |
| Indian-specific classes | ❌ Not trained | Converters and taxonomy ready; retraining not yet run |

---

## 9. FUTURE WORK

1. **Retrain on the merged Indian datasets** (IDD + DriveIndia + UVH-26). All
   three converters are written and tested. Expected gain, based on UVH-26's
   published result: up to 31.5% mAP@50:95 on Indian scenes.
2. **Add a source carrying traffic lights and signs** so those four classes stop
   being untrainable.
3. **Camera calibration** to tighten the phantom filter from its current
   uncertainty-widened band and give real-metre curvature.
4. **Train the drivable-area segmenter** on IDD segmentation labels, replacing the
   current classical-CV fallback.
5. **TensorRT export** for a further 3–5× inference speedup.

---

## 10. KEY TAKEAWAYS (closing slide material)

1. **95.4% mAP@0.5** detection, with rare classes holding up — no long-tail collapse.
2. **Full six-stage perception pipeline running at 23 FPS** on real dashcam footage
   with real collision alerts.
3. **The Indian-road failure was diagnosed, not guessed** — traced to a published
   domain-gap problem, with the fix built and ready to train.
4. **Graceful degradation is real, not aspirational** — proven when the lane model
   failed to load on the cluster and the pipeline still completed all 3,604 frames.
5. **Engineering rigour**: a 62-check suite that caught defects already running on
   the cluster, and a per-frame decision log that makes every verdict auditable.
