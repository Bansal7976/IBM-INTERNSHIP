# EXECUTION PLAN — what to retrain, what to change, in what order

Everything here runs on **Indian data** (UVH-26 + IDD, merged), with Indian
road standards (IRC:66) and left-hand traffic.

The problem statement is the whole chain working together:
**detection → tracking → collision → overtaking → lane.** This document walks
that chain and states, for each link, whether it needs retraining, what changed
in the code, and what is still open.

Run `python scripts/verify_adas_pipeline.py` before submitting anything (186
checks). It is GPU-free, takes about a minute, and has now caught nine defects
that would otherwise have surfaced only after hours of cluster time — two of
them while jobs were already running.

---

## THE CHAIN, LINK BY LINK

### 1. Detection — **RETRAIN. Ready to submit.**

Three runs, identical in every hyperparameter including seed, differing only in
the label space. That identity is not tidiness; it is what makes the comparison
attributable to the taxonomy.

| run | classes | what it tests |
|---|---|---|
| `fine` | ~21 | IDD and UVH-26's own vocabularies, unioned (control) |
| `semantic` | ~10 | grouped by appearance, IDD level-3 style (second control) |
| `decision` | **6** | grouped by decision consequence (ours) |

The second control is what stops a reviewer concluding "fewer classes scores
higher", which is trivially true. With it, the question becomes whether
grouping by *decision consequence* beats grouping by *appearance* at a
comparable class count.

**The arms are built from the sources' NATIVE vocabularies, not from the
project's merged 15-class set.** The merged taxonomy has already collapsed
exactly the distinctions under test — Hatchback, Sedan, SUV and MUV are one
"car" in it — so measuring what collapsing costs on already-collapsed labels
would be circular. Built from the merged set the arms come out 12 / 11 / 6, and
12 against 11 tests nothing.

**Both sources are wanted.** UVH-26 supplies the body-type granularity that
makes the fine arm meaningful; IDD supplies pedestrians, riders and animals.
Without IDD the VULNERABLE group never appears in any arm, and the
safety-weighted result would not test the case it exists for. Step 1 says so in
its log if IDD is missing.

```bash
qsub training/pbs/granularity_step1_prepare.pbs            # CPU, ~1 h
qsub -v LEVEL=fine     training/pbs/granularity_step2_train.pbs
qsub -v LEVEL=semantic training/pbs/granularity_step2_train.pbs   # parallel
qsub -v LEVEL=decision training/pbs/granularity_step2_train.pbs
qsub training/pbs/granularity_step3_compare.pbs            # after all three
```

Step 1 runs a **mapping report first** and refuses to continue if more than 5%
of boxes belong to classes the taxonomy does not recognise. Read that report —
it takes seconds and tells you exactly what survives the mapping. To see it
without submitting anything:

```bash
python data/prepare_taxonomy.py --src data/merged_india_yolo --report-only
```

Step 3 scores all three models in **one common six-group space**, whatever
space each was trained in. Native mAP across different class counts is not a
comparison at all, and the job prints that warning next to the table.

### 2. Tracking — **no retraining. Measure before changing.**

ByteTrack needs no training. But it feeds everything downstream, and its
failure mode is quiet.

**What was wrong:** TTC needs several frames of a track's distance history, and
that history is keyed on `track_id`. Every ID switch threw it away, after which
the collision module raised no alert at all — which from the outside is
identical to "the road is clear".

Measured on a synthetic steady approach:

| ID switches | closing-speed coverage (before) | after |
|---|---|---|
| none | 0.93 | 0.93 |
| every 10 frames | 0.60 | 0.93 |
| every 5 frames | 0.20 | 0.93 |
| every 3 frames | **0.00** | 0.93 |
| every frame | **0.00** | 0.93 |

Dense unstructured traffic — two-wheelers weaving and occluding each other — is
exactly where that churn happens, so the collision layer was most likely to be
silent precisely where it is most needed.

**Fixed:** a track appearing under a new id where another vanished a moment
earlier inherits its distance history when the boxes overlap. Geometric, not
appearance-based: no ReID model, and a wrong match costs a slightly stale
distance sample rather than a fabricated object.

**Still to decide, with data:** run the pipeline on real Indian footage and read
the new `[collision]` block in the summary. If `histories rescued` is a large
share of `new track ids`, the recovery is holding the module up but the tracker
itself is unstable — then switch to `botsort.yaml` (ReID) or lengthen
`track_buffer`. Do not change it on a hunch; the number is now printed.

### 3. Collision — **no retraining. Changes done.**

Depth Anything V2 is pretrained; nothing to train here.

- TTC thresholds now scale by decision group. A pedestrian gets 2.70 s to a
  car's 1.50 s; a heavy vehicle 1.95 s for its longer stopping distance. Cattle
  map to VULNERABLE alongside pedestrians, which on Indian roads is the norm
  rather than an edge case.
- The scale comes from the taxonomy, not from constants tuned in this file — so
  the grouping is what changes behaviour, which is the claim the design rests
  on.
- History recovery across ID switches (above).
- Telemetry printed on every run, so a degraded pipeline announces itself
  instead of looking healthy.

### 4. Overtaking — **no retraining. One open hardware question.**

Rule-based, so nothing to train. Three things changed:

**Rear-view rule added.** A forward-only system can report "no oncoming
traffic" while a vehicle is already closing from behind in the lane the driver
is about to enter. Rule 6 looks back along the manoeuvre side.

**The manoeuvre window is derived, not guessed.** A hard-coded "8 seconds" has
no defence. It now comes from IRC:66 overtaking sight distance — ego speed, the
*measured* speed of the vehicle being passed, and the standard's speed-dependent
acceleration. Values track the published OSD table within −16%/+18% across
40–100 km/h, closest at highway speeds.

**Sensor range is checked against the decision.** A system that sees 30 m back
cannot certify a lane clear when a vehicle 70 m back would arrive inside the
window. Short range yields `NOT_POSSIBLE_REAR_UNSEEN`, not `POSSIBLE`.

**A bug this uncovered:** the module hardcoded the US/Europe convention that the
oncoming lane is to the *left*. India drives on the left and overtakes on the
**right**, so it was treating vehicles in the ego's own lane as oncoming and
ignoring genuine head-on traffic. Now an explicit `traffic_side`, defaulting to
India, with both conventions under test. Worth noting that nearly every
published pipeline built on KITTI/BDD100K/Cityscapes inherits the same
inversion.

**Open:** the rear rule is **off by default**, because a single forward camera
cannot see behind and a module reporting "rear clear" without a rear view would
be asserting something it has no evidence for. Two ways forward, and this is a
decision for you:

- **Fit a rear camera** for the prototype. Then set `require_rear_view=True` and
  the rule is fully live — this is what the SIH demo would show.
- **Ship single-camera.** The verdict then carries "rear not assessed" rather
  than implying a check that did not happen. Honest, but the capability is not
  demonstrated.

There is a partial middle option worth knowing about: a forward camera does see
a vehicle *already drawing level* at the frame edge. That is a blind-spot
warning, not a pull-out decision — useful, but it does not answer "is anything
coming up behind me".

### 5. Lane detection — **RETRAIN NEEDED. This is the bottleneck.**

The weakest link, and the one the rest of the chain leans on: without a lane
divider the overtaking module returns `NOT_POSSIBLE_NO_LANE_INFO` and the
collision module falls back to the central 40% of the frame for its ego-path
test.

Two defects already fixed:

- **CLRNet coordinate remap.** `to_array()` returns CULane-space pixels
  (1640×590), not normalised coordinates, so no rescaling was happening and
  lane points landed outside the video frame. This is what made lane type read
  `unknown` on 3604 of 3604 frames.
- **Metric curve radius read every road straighter than it is** — a 50 m
  switchback came back as 135 m, a 150 m curve as 164 m. Always in the unsafe
  direction, because a road reported straighter than it is permits an overtake
  the geometry does not support. Now reports the tightest curvature over the
  visible stretch: within 4% at and above the 150 m decision threshold, and
  erring toward a *sharper* curve for tighter ones.

**What remains is training data.** CLRNet is running on CULane weights —
Chinese structured highways — against Indian unstructured roads. Fine-tuning on
the IDD lane subset (~6,149 labelled images) is the single highest-value
remaining item, and it is **blocked on dataset access**, not on code.

Until then the drivable-corridor fallback carries unmarked roads, which is the
right architecture for India but is itself untrained on Indian data.

---

## ORDER OF WORK

Nothing on the critical path waits for a dataset approval.

| # | do this | needs | why now |
|---|---|---|---|
| 1 | Submit the three granularity trainings | ready | everything downstream improves with detection; this is also the paper's main result |
| 2 | Run the pipeline on real Indian video, read the `[collision]` block | ready | tells you whether tracking is starving collision, with a number |
| 3 | Submit the comparison + night evaluation | after 1 | the two result tables |
| 4 | Decide the rear camera question | — | gates whether the overtaking capability can be demonstrated |
| 5 | Fine-tune CLRNet on the IDD lane subset | IDD access | the biggest remaining quality gap |
| 6 | Evaluate the rear-view rule against IDD-X | IDD-X access | validates rule 6 against annotated behaviour |

Steps 1–3 alone produce a complete result set.

---

## WHAT TO WATCH IN THE OUTPUT

Three numbers will tell you where the pipeline actually stands:

- **`closing-speed coverage`** (pipeline summary). Below 0.6 means the collision
  layer was frequently unable to measure, so quiet stretches do **not** mean the
  road was clear.
- **`critical-error rate`** (SWMC evaluation). The fraction of objects either
  missed or called something less cautious than they are. This is the single
  number to quote.
- **`sanity_filter` rejection rate**. Above 25% with an *estimated* focal length
  means real objects are being rejected as phantoms — pass `--kitti_calib` for
  the true focal length rather than accepting the loss.

---

## FOUR CLAIMS THAT MUST BE QUALIFIED

Not caveats for their own sake — each is somewhere a careful reader will push.

1. **Native mAP is not comparable across granularities.** Fewer classes is an
   easier problem. Only the common-space numbers compare.
2. **The IRC:66 figures deviate from the published table** by −16% to +18%, and
   the citation itself is currently from secondary sources — check it against
   the standard before the paper goes out.
3. **Curvature validation is synthetic.** That is a strength — it tests the
   estimator against a known answer, which real footage without surveyed radii
   cannot — but it is not validation on real curved roads.
4. **If the low-light enhancement ablation comes out neutral or negative, that
   is the finding.** The pipeline currently enhances every dark frame;
   enhancement can amplify noise into texture a domain-shifted detector reads as
   an object, which is one route to the phantom detections seen on Indian
   footage.
