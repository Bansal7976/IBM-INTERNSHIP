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
"under lights" versus "dark" in your requirement — IDD-AW lets us *validate*
that split instead of assuming it.

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

Already implemented: ground-plane projection giving curve radius in metres,
compared against road-design minimum radii. What is missing is **validation on
real curved Indian roads**. IDD sequences include hill and outskirt driving;
selecting a curved subset and reporting radius estimates against the road's
known geometry would substantiate the claim.

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

The rear-view rule to add: before permitting an overtake, check the rear and
side region on the manoeuvre side for an approaching vehicle, using the same
depth-differenced closing speed already implemented for the forward direction.
Fail safe if that region is not observable.

---

## PART 5 — EXECUTION ORDER

Sequenced so each step produces a result usable in the paper even if the next
step does not finish.

| # | Step | Output | Effort |
|---|---|---|---|
| 1 | Register and download IDD, IDD-AW, IDD-X | data | 1 day (approval) |
| 2 | Build the decision-grouped taxonomy mapper | code | **done, this commit** |
| 3 | Baseline: current detector on IDD test split | **the domain-gap number** | 1 h GPU |
| 4 | Train at 3 granularities (fine / IDD-L3 / decision-6) | **the taxonomy result** | 3 × 6 h GPU |
| 5 | Evaluate all three with mAP **and** SWMC | **the safety result** | 1 h |
| 6 | Baseline and train on IDD-AW low-light | **the night result** | 4 h |
| 7 | Fine-tune CLRNet on IDD lane subset | **the lane result** | 8 h |
| 8 | Add rear-view rule, evaluate against IDD-X | **the overtaking result** | 1 day |
| 9 | Curved-subset curvature validation | mountain-road result | 4 h |

**Steps 3 to 5 alone constitute a complete paper.** Everything after
strengthens it.

---

## PART 6 — WHAT MAKES THIS DEFENSIBLE RATHER THAN "JUST MORE TRAINING"

A reviewer, or an SIH judge, will ask what is new. Three answers, strongest
first:

1. **A taxonomy derived from decision consequence rather than visual
   similarity**, with the accuracy and safety trade-off measured across three
   granularities rather than asserted.
2. **A safety-weighted misclassification cost for detection**, extending to
   detection the principle IDD-AW established for segmentation, with an
   explicitly asymmetric cost for errors toward less caution.
3. **An integrated decision layer consuming that taxonomy** — forward and rear
   view, metric curvature, a degradation contract — evaluated end to end on
   Indian data rather than per-task in isolation.

What we should **not** claim: that we beat published detectors. We will not, and
it is not the point. The point is that the granularity and the cost function are
chosen for the decision the system has to make.

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
