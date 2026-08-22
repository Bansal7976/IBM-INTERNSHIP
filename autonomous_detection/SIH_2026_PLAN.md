# SIH 2026 — INDIAN-ROAD ADAS: RESEARCH FINDINGS AND EXECUTION PLAN

**Scope:** everything on Indian datasets. Detection, lane estimation, night and
adverse weather, mountain-road curvature, and the overtaking decision including
rear-view.

**Two outputs from one body of work:** a working prototype for SIH, and a paper.
The design below is chosen so those two do not pull in different directions —
every prototype decision is one a reviewer can be shown a reason for.

---

## PART 1 — WHAT THE RESEARCH FOUND

The single most useful finding: **almost every capability we need already has a
dedicated Indian dataset**, and they come from one group (IIIT Hyderabad /
AutoNUE), so they share conventions and can be combined.

| Requirement | Indian dataset that covers it | Venue |
|---|---|---|
| Detection, India-specific classes | **IDD Detection** | WACV 2019 |
| Detection at scale, 24 classes | **DriveIndia** (66,986 imgs) | IEEE ITSC 2025 |
| Detection, fine-grained vehicle types | **UVH-26** (26,646 imgs, 14 classes) | IISc TR, Nov 2025 |
| **Night, rain, fog, snow** | **IDD-AW** (5,000 imgs + paired NIR) | WACV 2024 |
| **Rear-view + overtaking behaviour** | **IDD-X** (697K boxes, 9K tracks, dual-view) | **ICRA 2024** |
| Lane detection, unstructured roads | **IDD lane subset** (~6,149 labelled imgs) | derived from IDD |
| Tracking / temporal consistency | **IDD Temporal** (±15 frames) | IDD release |
| 3D / LiDAR | **IDD-3D** (12K LiDAR frames) | WACV 2023 |

We do not need a single non-Indian corpus to build this.

---

## PART 2 — THE CLASS-COLLAPSING IDEA IS RIGHT, AND THE DATASET ALREADY SUPPORTS IT

Your instinct to merge classes (heavy / small / car-type / pedestrian) is
sound, and the research turns it from a shortcut into a defensible contribution.

**IDD ships its own 4-level label hierarchy**: 7 labels at level 1, 16 at
level 2, 26 at level 3, 30 at level 4. Each level is the union of labels in the
level below it, and the authors state the levels are constructed so that
**higher levels are less ambiguous**. That is precisely the property we want.

This matters for two reasons:

1. **It is not an ad-hoc merge.** Using the dataset authors' own hierarchy is
   defensible; inventing our own grouping and calling it a contribution is not.
2. **The level becomes an experimental variable.** We can train at several
   levels and *measure* the accuracy-versus-usefulness trade-off rather than
   asserting that merging helps.

### But we should not simply use IDD's levels either

IDD's hierarchy is organised by **visual and semantic similarity**. What our
pipeline actually needs is a grouping by **decision consequence** — because the
collision and overtaking layers do not care whether an object is a bus or a
truck. They care that it is large, slow to stop, and blocks the view ahead.

That reframing is the intellectual contribution: **a taxonomy whose classes are
defined by what the decision layer must do differently for each.**

### Proposed taxonomy (6 classes)

| Class | Members | Why grouped | What the decision layer does differently |
|---|---|---|---|
| VULNERABLE | pedestrian, rider, animal | unprotected, unpredictable, small | largest TTC margin; never overtake at close lateral distance |
| TWO_WHEELER | motorcycle, bicycle | lane-splits, sudden lateral movement | wide lateral clearance; expect entry from either side |
| THREE_WHEELER | autorickshaw | slow, stops without warning, India-specific | most common overtake target; anticipate abrupt stop |
| LIGHT_VEHICLE | car, van, jeep, SUV, hatchback, sedan | standard road behaviour | baseline parameters |
| HEAVY_VEHICLE | truck, bus, tractor, tanker, construction, mini-bus, LCV | blocks forward view, long stopping distance, slow | much larger overtake clearance; **sets a view-obstructed flag** |
| STATIC_OBSTACLE | cart, barrier, cone, debris, parked obstruction | does not move | path obstruction only, no TTC from relative motion |

Note what this buys us on the specific failure you reported: truck, bus,
tanker, tractor and LCV were separate classes competing for the same visual
evidence, and the detector was confusing them. Collapsed into HEAVY_VEHICLE,
that entire confusion mode **disappears by construction** — and nothing
downstream loses information, because every member of the group gets the same
treatment anyway.

### The experiment that makes this a paper contribution

Train the same architecture at three granularities on the same Indian data,
evaluate on the same held-out split:

| Setting | Classes | What it tests |
|---|---|---|
| Fine | ~24 (DriveIndia native) | the baseline everyone reports |
| IDD level-3 | ~14 | the dataset authors' own hierarchy |
| **Decision-grouped** | **6 (above)** | **ours — grouped by decision consequence** |

Report per-setting mAP **and** a safety-weighted error measure (next section).

The claim is not "fewer classes scores higher" — that is trivially true and a
reviewer will say so. The claim is: *at the granularity the decision layer
actually consumes, accuracy rises and dangerous confusions fall, at no cost to
downstream behaviour.*

---

## PART 3 — NOT ALL MISCLASSIFICATIONS ARE EQUAL

**IDD-AW (WACV 2024) introduced "Safe mIoU"**, a metric that uses the label
hierarchy to *penalise dangerous mispredictions* that ordinary mIoU treats the
same as harmless ones. Confusing two kinds of truck is benign. Confusing a
pedestrian with a roadside pole is not.

IDD-AW proposed this **for segmentation**. Applying the same principle to
**detection**, with a cost derived from decision consequence, is a genuine and
modest novelty — and it directly serves your goal of reducing the
misclassifications that actually matter.

**Proposed: Safety-Weighted Misclassification Cost (SWMC).**
Order the six groups by how much caution each demands, then price every
outcome by how far the prediction falls short of the truth in that ordering:

| Cost | Situation | Example |
|---|---|---|
| 0.0 | same decision group | bus predicted as truck |
| 0.10 | different group, **more** cautious | car predicted as vulnerable |
| 0.30 | different group, similar consequence | two-wheeler ↔ three-wheeler |
| 0.44 – 0.86 | different group, **less** cautious, scaled by distance fallen | pedestrian predicted as static obstacle = 0.86 |
| **1.00** | **not detected at all** | **a pedestrian the system never saw** |

Two properties make this coherent, and both were arrived at by finding the
first version wrong:

1. **A miss dominates every mislabelling of the same object.** A missed object
   is scored as a prediction of a virtual class at caution rank 0 — below every
   real group. An early version put the ceiling on the worst *misclassification*
   instead, which made "two-wheeler called a barrier" (0.825) cost more than not
   seeing the two-wheeler at all (0.80). That inverts the real hazard: a wrong
   label still places an obstacle in the world model; a miss places nothing.
2. **The scale's ceiling is the genuinely worst event**, not an arbitrary
   maximum: 1.0 is exactly a vulnerable road user going undetected.

A **false positive costs 0.02 – 0.10**, scaled the same way. Small, but
deliberately not zero — the phantom detections observed on Indian footage are
what teach a driver to ignore the alerts, and a driver who ignores them has
lost the protection entirely.

The asymmetry is the point throughout: an error toward more caution is a
nuisance, an error toward less caution is a hazard, and a symmetric confusion
matrix cannot distinguish them.

**Implementation:** `models/taxonomy.py` (costs), `evaluation/evaluate_swmc.py`
(scoring), 32 checks in `scripts/verify_adas_pipeline.py`.

### Making the three-way comparison fair

The three granularities have different label spaces, so their mAP numbers are
**not comparable** — fewer classes is an easier problem and the 6-class model
would win by construction. Two safeguards:

- **One common evaluation space.** All three models are projected onto the six
  decision groups before scoring. A fine model's "tanker" becomes
  HEAVY_VEHICLE, and every model answers the same question.
- **Identical boxes across the three datasets.** All three are restricted to
  the same shared vocabulary, so they contain the *same* objects and differ
  only in labelling. Enforced by an import-time check in `taxonomy.py` and
  re-verified in the prepare job before any GPU time is spent — otherwise a
  taxonomy that merely drops more hard objects would post the best score.

---

## PART 4 — REMAINING CAPABILITIES, EACH ON INDIAN DATA

### 4.1 Night and adverse weather — IDD-AW

IDD-AW gives 1,500 rain + 1,500 fog + 1,000 low-light + 1,000 snow images,
**each with a paired near-infrared frame**. The NIR pairing is unusual and
useful: it supervises what a scene should look like when the visible channel is
degraded.

Plan, in increasing effort:

1. **Evaluate** the current detector on IDD-AW low-light to quantify the drop.
   That number alone is a paper-worthy result.
2. **Train** on IDD-AW low-light and normal jointly.
3. Optional: use the NIR pairs to supervise the enhancement stage instead of
   relying on the zero-reference method.

Our pipeline already classifies DAY / NIGHT_LIT / NIGHT_UNLIT, which maps onto
"under lights" versus "dark" in the requirement — IDD-AW lets us *validate*
that split instead of assuming it.

**A per-condition evaluation that needs no new data is already implemented**
(`evaluation/evaluate_conditions.py`, `training/pbs/india_night_eval.pbs`). It
buckets the validation set with the pipeline's *own* `scene_lighting()`, so the
table validates the classifier that ships and gates the overtaking rule, not a
separate offline one. Per bucket it reports SWMC and the critical-error rate,
because night failures are overwhelmingly misses and mAP averages a missed
pedestrian together with a missed parked cart.

It also runs the **enhancement ablation** — raw versus Zero-DCE++/CLAHE, over
the same images. This is worth doing rather than assuming: enhancement can
amplify sensor noise into texture that a domain-shifted detector reads as an
object, which is one of the routes to the phantom detections observed on Indian
footage. The comparison is controlled and the answer is allowed to come out
negative. Buckets smaller than `--min-bucket` are flagged rather than reported
as a confident number; unlit-night frames are rare, and a metric computed over
30 images is noise dressed as a result.

### 4.2 Lane detection on unstructured roads — IDD lane subset

An Indian lane dataset of roughly 6,149 labelled images derived from IDD exists
for exactly this evaluation. Two paths, and we should do both:

- **Where markings exist:** fine-tune CLRNet on the IDD lane subset instead of
  relying on CULane weights. This is the single highest-value fix for the lane
  problem observed on Indian footage.
- **Where markings do not exist:** the drivable-corridor fallback, trained on
  IDD segmentation's *drivable fallback* class — which IDD created precisely
  because "road" and "drivable but not road" are different things here.

### 4.3 Mountain roads and curvature

Ground-plane projection gives the curve radius in metres, compared against
road-design minimum radii. Validating it turned up a defect worth reporting.

**The estimate used to read every road straighter than it is.** Projecting arcs
of *known* radius through the homography and recovering them showed a
consistent bias: a 50 m switchback came back as 135 m, a 100 m curve as 123 m,
a 150 m curve as 164 m. The cause was evaluating curvature at the far end of
the lane polyline, where κ = |2a|/(1+(2az+b)²)^1.5 divides by the largest slope
term. The error was always in the unsafe direction — a road reported straighter
than it is permits an overtake the geometry does not support.

It now reports the **tightest curvature over the visible stretch**, which is
also the quantity the decision needs, since that is what limits sight distance.
Against the same arcs it is within 4% at and above the 150 m decision
threshold, and errs toward reporting a *sharper* curve for tighter ones (50 m
arc → 27 m) — wrong in the direction that refuses rather than permits.

| true radius | before | now |
|---|---|---|
| 50 m | 135 m (**+171%**) | 27 m (−46%) |
| 100 m | 123 m (+23%) | 90 m (−10%) |
| 150 m | 164 m (+9%) | 143 m (−4%) |
| 400 m | 405 m (+1%) | 398 m (−1%) |

Validation is synthetic by construction, and that is the point: it tests the
*estimator* against a known answer, which no amount of real footage without
surveyed radii can do. Real curved IDD sequences remain useful as a
qualitative check that the lane input feeding it is sane.

Implementation: `models/ipm.py`; 8 checks in the verification suite.

### 4.4 Overtaking with rear-view — IDD-X

This is the finding that most changes the plan. **IDD-X (ICRA 2024) is a
dual-view dataset that explicitly includes rearview information**, with 697K
bounding boxes, 9K *important-object* tracks, and 19 explanation categories
describing why the ego vehicle behaved as it did.

Your requirement — *"jis side mudna, us taraf jitna frame me aa raha hai, utna
piche dekhkar"* — is exactly ego-relative important-object reasoning, and IDD-X
is built for it.

This upgrades the overtaking module from five hand-written rules to:

| Current | With IDD-X |
|---|---|
| forward view only | forward **and rear** view |
| "is there an oncoming vehicle" | "is a vehicle **closing from behind on the side I intend to move into**" |
| all objects weighted equally | ego-relative **importance**, as annotated |
| verdict with a reason string | verdict with an **explanation category**, comparable to the dataset's own labels |

**The rear-view rule is implemented** (`inference/rear_view.py`, RULE 6 in
`inference/overtaking.py`, 27 checks). Two design choices depart from how this
is usually done, and both are defensible in the paper:

**The manoeuvre window is derived, not guessed.** A hard-coded "8 seconds" has
no defence. The window now comes from **IRC:66 overtaking sight distance**,
computed from the ego speed, the *measured* speed of the vehicle being passed,
and the standard's speed-dependent acceleration table. Values track the
published OSD figures within −16%/+18% across 40–100 km/h, closest at highway
speeds. Overtaking an autorickshaw at 40 km/h and passing a truck at 80 are no
longer the same manoeuvre.

**Sensor range is checked against the decision.** A system that sees 30 m back
cannot honestly certify a lane clear when a vehicle 70 m back would arrive
inside the window. When observable range falls short, the verdict is
`NOT_POSSIBLE_REAR_UNSEEN` — *"I see nothing" is not "nothing is there"*. This
is the honest reading of the requirement to look as far back as the frame
allows, and most rule-based pipelines simply omit the check.

The rule is **off by default**, so a single-camera deployment behaves exactly
as before and never implies a rear check it did not perform.

### 4.5 A bug this uncovered: India drives on the LEFT

The overtaking module hardcoded the **US/Europe convention** — that the
oncoming lane is to the *left* of the divider. India drives on the left and
overtakes on the **right**, so the module was treating vehicles in the ego's
own lane as oncoming and ignoring genuine head-on traffic, and picking the
wrong lane boundary as the line to cross.

This is worth a sentence in the paper. Nearly every published pipeline is built
on KITTI/BDD100K/Cityscapes and inherits the right-hand-traffic assumption
silently, so any of them ported to India carries the same inversion. It is now
an explicit `traffic_side` parameter defaulting to India, with both conventions
covered by tests.

---

## PART 5 — EXECUTION ORDER

Sequenced so each step produces a result usable in the paper even if the next
step does not finish.

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Register and download IDD, IDD-AW, IDD-X | data | pending approval |
| 2 | Decision-grouped taxonomy + SWMC metric | code | **done** |
| 3 | Dataset remapper, all three granularities | code | **done** |
| 4 | Rear-view overtaking rule (IRC:66 window) | code | **done** |
| 5 | Metric curvature, validated against known arcs | code | **done** |
| 6 | Taxonomy drives collision margins | code | **done** |
| 7 | Train at 3 granularities | **the taxonomy result** | 3 × ~8 h GPU |
| 8 | Compare all three under SWMC | **the safety result** | 1 h GPU |
| 9 | Per-condition (night) evaluation + enhancement ablation | **the night result** | 2 h GPU |
| 10 | Fine-tune CLRNet on IDD lane subset | **the lane result** | 8 h GPU, needs step 1 |
| 11 | Evaluate the rear-view rule against IDD-X | **the overtaking result** | 1 day, needs step 1 |

**Steps 7 to 9 alone constitute a complete paper**, and every input they need
already exists. Steps 10 and 11 strengthen it but are blocked on dataset
access, so nothing on the critical path waits for an approval email.

### Submission order

```bash
qsub training/pbs/granularity_step1_prepare.pbs          # CPU, ~1 h
# then the three trainings — independent, may queue in parallel
qsub -v LEVEL=fine     training/pbs/granularity_step2_train.pbs
qsub -v LEVEL=semantic training/pbs/granularity_step2_train.pbs
qsub -v LEVEL=decision training/pbs/granularity_step2_train.pbs
# once all three finish
qsub training/pbs/granularity_step3_compare.pbs          # the paper's main table
qsub -v LEVEL=decision training/pbs/india_night_eval.pbs # the night table
```

Run `python scripts/verify_adas_pipeline.py` before submitting anything. It is
GPU-free, takes about a minute, and has now caught six defects that would
otherwise have surfaced only after hours of cluster time — including two that
were already running.

---

## PART 6 — WHAT MAKES THIS DEFENSIBLE RATHER THAN "JUST MORE TRAINING"

A reviewer, or an SIH judge, will ask what is new. Three answers, strongest
first:

1. **A taxonomy derived from decision consequence rather than visual
   similarity**, with the accuracy and safety trade-off measured across three
   granularities rather than asserted — and with the grouping actually driving
   behaviour (collision margins scale by group), not merely relabelling boxes.
2. **A safety-weighted misclassification cost for detection**, extending to
   detection the principle IDD-AW established for segmentation, with an
   explicitly asymmetric cost for errors toward less caution and a scale
   anchored at the genuinely worst event — a vulnerable road user undetected.
3. **A decision layer that respects its own competence boundary.** This is the
   thread running through the whole system, and it is what most distinguishes
   it from "we trained a detector on Indian data":
   - the overtaking rule refuses when the rear view does not reach as far as
     the manoeuvre requires, instead of reading silence as safety;
   - the curvature estimate errs toward reporting a sharper curve, so its
     failures refuse rather than permit;
   - the geometric sanity filter rejects detections that are the wrong
     physical size for their distance;
   - optional modules degrade explicitly rather than silently.

   In each case the system distinguishes *"I checked and it is clear"* from
   *"I could not check"*, and reports which.

What we should **not** claim: that we beat published detectors. We will not, and
it is not the point. The point is that the granularity, the cost function and
the refusal conditions are all chosen for the decision the system has to make.

### Honest notes for the write-up

- The three-granularity comparison must state that native mAP is **not**
  comparable across rows. Claiming otherwise is the error the design exists to
  avoid, and a reviewer will find it immediately.
- The IRC:66 figures deviate from the published table by −16% to +18%. Say so.
- Curvature validation is **synthetic**. That is a strength — it tests the
  estimator against a known answer — but it must not be presented as validation
  on real curved roads, which would require surveyed radii we do not have.
- If the low-light enhancement ablation comes out neutral or negative, report
  it as the finding. It is a more interesting result than a small improvement,
  and the pipeline currently applies enhancement on every dark frame.

---

## REFERENCES USED IN THIS PLAN

Verified against publisher listings, not preprint servers.

- G. Varma, A. Subramanian, A. Namboodiri, M. Chandraker, C. V. Jawahar, "IDD: A
  dataset for exploring problems of autonomous navigation in unconstrained
  environments," **WACV 2019**, pp. 1743–1751.
- F. Shaik et al., "IDD-AW: A benchmark for safe and robust segmentation of
  drive scenes in unstructured traffic and adverse weather," **WACV 2024**.
- C. Parikh, R. Saluja, C. V. Jawahar, R. K. Sarvadevabhatla, "IDD-X: A
  multi-view dataset for ego-relative important object localization and
  explanation in dense and unstructured traffic," **ICRA 2024**, Yokohama.
- S. Dokania et al., "IDD-3D: Indian driving dataset for 3D unstructured road
  scenes," **WACV 2023**.
- R. Kumar, D. S. Reddy, P. Rajalakshmi, "DriveIndia: An object detection
  dataset for diverse Indian traffic scenes," **IEEE ITSC 2025**.
- AIM, Indian Institute of Science, "The Urban Vision Hackathon dataset and
  models," Tech. Rep. UVH-26-v1.0, Nov 2025.
- Indian Roads Congress, **IRC:66-1976**, "Recommended practice for sight
  distance on rural highways," New Delhi. *Source of the overtaking sight
  distance and manoeuvre-time formulation used by the rear-view rule. Verify
  the edition and clause numbers against a copy of the standard before the
  paper goes out — this citation has been taken from secondary sources.*
- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
  2nd ed., Cambridge University Press, 2004, ch. 13. *Plane-induced homography
  used for the ground-plane projection.*
