# SIH260037 — Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

**Theme:** Robotics and Drones · **Category:** Software · **Organisation:** Government of India

---

## 1. WHAT THE PROBLEM STATEMENT ACTUALLY ASKS FOR

The title carries three constraints, and each one rules out a large part of the
standard autonomous-driving toolbox.

**"Unstructured Indian Roads"** — no reliable lane markings, no lane discipline,
mixed traffic that shares the same carriageway (autorickshaws, handcarts,
cattle, two-wheelers filtering between vehicles), and road edges that are often
not edges at all. Every planner in mainstream use assumes a **lane centreline**
to plan relative to. On these roads that centreline does not exist.

**"Adaptive"** — the behaviour must change with the situation, not run on fixed
thresholds. Adaptive to what the obstacle *is* (a pedestrian and a parked cart
demand different clearance), to the road geometry (a mountain switchback is not
a highway), to the light, and — the part most systems skip — to **how much the
sensors can currently see**.

**"Collision Avoidance"** — not just detection. A trajectory that is provably
clear of every tracked object for the duration it is executed.

### Honest position: what is built, and what is not

We have spent this project building the perception and decision layers. Being
straight about the gap is more useful than claiming completeness:

| Layer | State |
|---|---|
| Detection, India-specific classes | **built**, retraining now |
| Multi-object tracking | **built**, with ID-churn recovery |
| Metric depth and distance | **built** |
| Collision detection (TTC) | **built**, margins scale by obstacle type |
| Lane / drivable-corridor estimation | **built**, fallback for unmarked roads |
| Metric road curvature | **built**, validated against known-radius arcs |
| Overtake decision (forward + rear) | **built**, IRC:66 manoeuvre window |
| **Adaptive path planning** | **NOT BUILT — this is the gap** |

Everything above produces a *verdict*: brake, warn, overtake possible or not.
None of it produces a **trajectory**. That is the half of the problem statement
still open, and section 4 is how we close it.

---

## 2. THE RESEARCH LANDSCAPE, AND THE GAP IN IT

### 2.1 How local planning is normally done

The dominant approach is **Frenet-frame trajectory generation**, introduced by
Werling, Ziegler, Kammel and Thrun (ICRA 2010). The road centreline becomes the
*s* axis, lateral offset becomes *d*, and the planning problem decouples into a
quintic polynomial laterally and a quartic longitudinally. Sample many
end-states, score them for jerk, speed and safety, discard the ones that
collide, execute the best. It is the backbone of Apollo, Autoware and most
production stacks.

It has been refined steadily. The **Frenet Corridor Planner** (Honda Research
Institute, 2025) replaces sampling with a constrained optimisation inside a
drivable corridor — static obstacles become hard corridor boundaries, dynamic
ones become soft convex penalties — and reports a 0.035 s solve time against
0.17–0.63 s for A\*, RRT\* and B-RRT\* on the same scenarios, with smoother
paths (max yaw change 0.053 rad).

For genuinely unstructured space, the reference is **Hybrid A\*** (Dolgov,
Thrun, Montemerlo and Diebel, IJRR 2010) — a kinematically-feasible A\* variant
over the vehicle's 3-D state space followed by non-linear smoothing, built for
the DARPA Urban Challenge parking lots.

### 2.2 The gap

**Every member of the Frenet family requires a reference centreline.** The
Honda 2025 paper is explicit about it: corridor boundaries are computed relative
to the reference, with *d = 0* being that path. On a marked highway the map or
the lane detector supplies it. On an unmarked Indian road, nothing does.

Hybrid A\* does not need a centreline, but it plans in a grid over free space
and pays for it in runtime — the very comparison the Honda paper reports.
Neither end of the spectrum fits a vehicle moving at 40–60 km/h down a road
that has a clear direction of travel but no marked lanes.

### 2.3 What the Indian-roads literature has established

The Indian unstructured-driving problem has a dedicated research line, almost
all of it from IIIT-Hyderabad:

- **IDD** (Varma, Subramanian, Namboodiri, Chandraker and Jawahar, WACV 2019)
  established that Cityscapes/KITTI-trained models transfer poorly here, and
  introduced a label set with a **drivable-fallback** class precisely because
  "road" and "drivable but not road" are different things on these roads.
- **IDD-AW** (WACV 2024) added night, rain, fog and snow with paired NIR, and
  introduced **Safe mIoU** — the idea that the label hierarchy should be used to
  penalise *dangerous* mispredictions more than harmless ones.
- **IDD-X** (ICRA 2024) added dual-view (including **rear**) driving sequences
  with ego-relative important-object annotations.
- **Diffusion-FS** (Gupta, Stanley, Paul, Singh et al., 2025) predicts
  **multimodal free space** on IDD — several plausible drivable configurations
  rather than one, because the answer on these roads is genuinely ambiguous.

The pattern is clear: the Indian-roads community has largely converged on
**free space** as the representation, because lanes are not available. What is
missing is the bridge from free space to a *committed, kinematically feasible
trajectory* — which is what the problem statement asks for.

---

## 3. OUR APPROACH IN ONE SENTENCE

> Synthesise the reference path from the drivable free space instead of from
> lane markings, plan inside a corridor whose width is set by **what each
> obstacle is**, and refuse to commit where the sensors cannot see far enough to
> justify the manoeuvre.

Three contributions follow from that, and each is already partly in place.

### 3.1 A taxonomy defined by decision consequence, not appearance

The detector was confusing truck / bus / tanker / tractor / LCV — five classes
competing for the same visual evidence. But the planner treats all five
identically: large, slow to stop, blocks the forward view. So classes are
grouped by **what the decision layer must do differently**:

| Group | Members | TTC margin | Lateral clearance | Blocks view |
|---|---|---|---|---|
| VULNERABLE | pedestrian, rider, animal | ×1.8 | 1.5 m | no |
| TWO_WHEELER | motorcycle, bicycle | ×1.4 | 1.2 m | no |
| THREE_WHEELER | autorickshaw | ×1.2 | 1.0 m | no |
| LIGHT_VEHICLE | car, van, jeep, SUV | ×1.0 | 0.8 m | no |
| HEAVY_VEHICLE | truck, bus, tractor, tanker | ×1.3 | 1.0 m | **yes** |
| STATIC_OBSTACLE | cart, barrier, cone, debris | ×1.0 | 0.8 m | no |

This is not an ad-hoc merge: IDD ships its own 4-level hierarchy (7/16/26/30
labels) built so that higher levels are less ambiguous. Ours differs by
grouping on *consequence* rather than *similarity*, and we measure the
difference rather than assert it — three granularities trained identically and
compared in one common space.

**The link to planning is the point.** `lateral_clearance_m` is exactly the
corridor half-width the planner needs. Classifying an object as VULNERABLE
narrows the corridor by 1.5 m; a car by 0.8 m. The taxonomy is not a labelling
scheme that happens to sit upstream — it *parameterises the planner*.

### 3.2 A safety-weighted error measure (SWMC)

mAP scores "bus called a truck" and "pedestrian called a barrier" identically.
Only the second removes a protection. Extending IDD-AW's Safe-mIoU principle
from segmentation to detection, each outcome is priced by how far the
prediction falls short in a caution ordering:

| Cost | Outcome |
|---|---|
| 0.00 | same decision group (bus ↔ truck) |
| 0.10 | more cautious than truth (car called vulnerable) |
| 0.30 | comparable consequence |
| 0.44 – 0.86 | less cautious, scaled by distance fallen |
| **1.00** | **not detected at all — a pedestrian the system never saw** |

The scale's ceiling is the genuinely worst event, not an arbitrary maximum. A
miss is modelled as a prediction of a virtual class at caution rank zero, which
makes it dominate every mislabelling of the same object by construction.

### 3.3 A decision layer that knows what it cannot know

This is the thread that runs through the whole system and most distinguishes it
from "we trained a detector on Indian data":

- The **overtake rule refuses** when the rear view does not reach as far back as
  the manoeuvre requires. *"I see nothing" is not "nothing is there."*
- The **curvature estimate errs toward reporting a sharper curve**, so its
  failures refuse rather than permit.
- The **geometric sanity filter** rejects detections that are the wrong physical
  size for their measured distance — the phantom-detection fix.
- The **collision layer reports its own coverage**, so silence caused by a
  blinded tracker is distinguishable from silence caused by an empty road.
- **Optional modules degrade explicitly** rather than silently.

The planner inherits this: it will not commit a trajectory into space it has not
observed, and it reports which constraint bound the solution.

---

## 4. THE PLANNER — WHAT WE BUILD NEXT

Four stages, each independently testable.

### Stage 1 — Drivable corridor → reference path

Take the drivable-area mask (already produced, trained on IDD's drivable-fallback
class where markings are absent), project it to the ground plane through the
calibrated homography, and extract its **medial axis** — the locus equidistant
from both edges. Fit a smoothing spline. That spline is the reference the Frenet
frame needs, synthesised from free space rather than read off lane paint.

Where lane markings *do* exist, CLRNet supplies the reference directly and the
medial axis becomes a cross-check. The system uses whichever is available and
records which — a road that produces two disagreeing references is a road to be
cautious on.

### Stage 2 — Corridor bounds from the obstacle taxonomy

For each longitudinal station *s* along the reference, compute the lateral
bounds *[d_min(s), d_max(s)]*:

- start from the projected free-space edges;
- for each tracked obstacle, subtract its footprint **inflated by its group's
  `lateral_clearance_m`**;
- for obstacles that can move, inflate along their predicted motion over the
  planning horizon rather than at their current position alone.

Following the Honda formulation, static obstacles become **hard** bounds and
dynamic ones **soft** penalties, so a crowded scene degrades into a
tighter-but-feasible corridor instead of becoming infeasible.

### Stage 3 — Trajectory generation and selection

Werling's formulation on the synthesised reference: quintic lateral, quartic
longitudinal, sampled over end-states. Each candidate is scored on

```
J = w_jerk·J_comfort + w_dev·J_deviation + w_clear·J_clearance + w_prog·J_progress
```

and rejected outright if it leaves the corridor, violates the kinematic bicycle
model's curvature limit, or intersects any obstacle's swept volume over the
horizon.

### Stage 4 — The competence gate

Before the chosen trajectory is emitted, it must satisfy: *the entire path lies
within observed space, for the whole horizon.* If the horizon extends past what
depth and free-space estimation actually resolve, the trajectory is **truncated
to the observed range and the speed reduced accordingly**, rather than
extrapolated. The output carries the reason.

This is the planning-layer expression of the same principle as the rear-view
rule, and on a mountain road with a blind curve it is the difference between a
system that slows down and one that does not.

---

## 5. TECH STACK

Chosen for defensibility and for what runs on a single GPU — not for novelty
per component.

### Perception

| Component | Choice | Why this one |
|---|---|---|
| Object detection | **YOLOv11x** (Ultralytics) | Real-time on one GPU; anchor-free; the accuracy/latency point that leaves budget for depth and planning in the same frame |
| Label space | **Decision-grouped, 6 classes** | Removes the truck/bus/tanker confusion by construction; supplies the planner's clearance parameters |
| Tracking | **ByteTrack**, with distance-history recovery across ID switches | Associates low-confidence boxes too, which matters in dense traffic; the recovery keeps TTC alive through ID churn |
| Monocular depth | **Depth Anything V2 (metric)** | Metric depth from one camera; no stereo rig or LiDAR needed for a student build |
| Lane detection | **CLRNet** | Row-anchor + refinement; strongest of the classical lane detectors where paint exists |
| Unmarked roads | **Drivable-area segmentation** (IDD drivable-fallback) | The IDD authors created that class because "road" and "drivable" differ here |
| Low light | **Zero-DCE++**, CLAHE fallback | Zero-reference — needs no paired day/night training data |

### Geometry and decision

| Component | Choice | Why |
|---|---|---|
| Ground projection | **Plane-induced homography** (Hartley & Zisserman, ch. 13) | Converts pixels to metres so curvature and clearance are physical, not pixel counts |
| Curvature | Tightest radius over the visible stretch | The quantity that limits sight distance; errs toward reporting sharper |
| Collision | **TTC from depth differencing** | Pixel motion is not physical approach; depth history is |
| Overtake window | **IRC:66 overtaking sight distance** | The Indian standard for these roads; our figures track its table within −16%/+18% |
| Error measure | **SWMC** (ours, after IDD-AW) | mAP cannot express that some confusions are dangerous |

### Planning

| Component | Choice | Why |
|---|---|---|
| Reference path | **Medial axis of the drivable corridor** | The centreline the Frenet frame needs, where no lane paint exists |
| Local planner | **Frenet corridor optimisation** (Werling; Honda 2025 corridor form) | Decouples lateral/longitudinal; the corridor form is ~5–18× faster than A\*/RRT\* in the published comparison |
| Feasibility | **Kinematic bicycle model** | Rejects trajectories the vehicle cannot physically execute |
| Fallback | **Hybrid A\*** | For genuinely unstructured stretches where no coherent corridor exists |

### Infrastructure

PyTorch · Ultralytics · OpenCV · NumPy/SciPy · PBS/`qsub` on the institute HPC ·
conda · Git. Target: **≥ 20 FPS on a single GPU**, which the perception stack
already meets at 23–25 FPS.

---

## 6. HOW WE WILL SHOW IT WORKS

| Claim | Evidence | Status |
|---|---|---|
| Decision-grouped labels beat both fine and appearance-based grouping | Three models, identical hyperparameters and seed, identical boxes, compared in one common space by SWMC | jobs ready to submit |
| Grouping actually changes behaviour | Collision margins scale by group; 10 checks | done |
| Curvature is metric and trustworthy | Recovery of known-radius arcs; error never in the unsafe direction | done, 8 checks |
| Rear-view refusal is principled | IRC:66 window; refuses when sensor range < required sight distance | done, 27 checks |
| Collision survives dense traffic | Coverage held at 0.93 through ID switches on every frame (0.00 without recovery) | done, 11 checks |
| Night degradation is quantified, enhancement is not assumed to help | Per-condition SWMC + controlled ablation | job ready |
| Planned trajectories are collision-free and feasible | Swept-volume check against tracked obstacles; curvature limit | **to build** |

The whole system is guarded by a **GPU-free verification suite — 186 checks**
that runs in about a minute. It has caught nine real defects so far, including
two that were already running on the cluster, and three that were wrong in the
unsafe direction.

---

## 7. RISKS, STATED PLAINLY

- **Path planning is not yet built.** It is the largest remaining piece and the
  centre of the problem statement. Sections 4 is a design, not a result.
- **IDD access is pending.** Without it the current data (UVH-26) is
  vehicles-only, so pedestrians, riders and animals — the VULNERABLE group the
  safety argument rests on — are absent. Everything else runs; this one result
  waits.
- **No rear camera.** The rear-view rule is implemented and tested but reports
  "not assessable" on a single-camera rig, which is the honest output. A second
  camera, or IDD-X for evaluation, closes it.
- **Monocular depth is not LiDAR.** Distances carry error that grows with range.
  The competence gate is the mitigation: we truncate the horizon to where depth
  is trustworthy rather than planning on optimistic numbers.
- **Two citations are preprints.** The Frenet Corridor Planner and Diffusion-FS
  have no confirmed venue yet and must be cited as arXiv, or replaced if they
  appear at a conference before submission.

---

## 8. REFERENCES

Publication details verified against publisher listings.

1. M. Werling, J. Ziegler, S. Kammel and S. Thrun, "Optimal trajectory
   generation for dynamic street scenarios in a Frenét frame," *IEEE ICRA*,
   2010, pp. 987–993.
2. M. Werling, S. Kammel, J. Ziegler and L. Gröll, "Optimal trajectories for
   time-critical street scenarios using discretized terminal manifolds,"
   *International Journal of Robotics Research*, vol. 31, no. 3, 2012.
3. D. Dolgov, S. Thrun, M. Montemerlo and J. Diebel, "Path planning for
   autonomous vehicles in unknown semi-structured environments," *International
   Journal of Robotics Research*, vol. 29, no. 5, 2010, pp. 485–501.
4. G. Varma, A. Subramanian, A. Namboodiri, M. Chandraker and C. V. Jawahar,
   "IDD: A dataset for exploring problems of autonomous navigation in
   unconstrained environments," *IEEE WACV*, 2019, pp. 1743–1751.
5. F. Shaik et al., "IDD-AW: A benchmark for safe and robust segmentation of
   drive scenes in unstructured traffic and adverse weather," *IEEE WACV*, 2024.
6. C. Parikh, R. Saluja, C. V. Jawahar and R. K. Sarvadevabhatla, "IDD-X: A
   multi-view dataset for ego-relative important object localization and
   explanation in dense and unstructured traffic," *IEEE ICRA*, 2024, Yokohama.
7. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
   2nd ed., Cambridge University Press, 2004, ch. 13.
8. Indian Roads Congress, **IRC:66-1976**, "Recommended practice for sight
   distance on rural highways," New Delhi.
   *Taken from secondary sources — verify clause numbers against the standard
   before submission.*

**Preprints — cite as such, or update if a venue appears:**

9. F. M. Tariq, Z.-H. Yeh, A. Singh, D. Isele and S. Bae, "Frenet corridor
   planner: An optimal local path planning framework for autonomous driving,"
   arXiv:2505.03695, May 2025. (Honda Research Institute)
10. K. Gupta, T. S. Stanley, P. Paul, A. K. Singh et al., "Diffusion-FS:
    Multimodal free-space prediction via diffusion for autonomous driving,"
    arXiv:2507.18763, July 2025.
