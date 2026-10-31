# HANDOVER — everything needed to take this project over

**Prepared 31 Oct 2026 for Krish Manwani.** Repo: `github.com/Bansal7976/IBM-INTERNSHIP`, branch **`krish-implementation`**.
The other branches (`main`, `first-execution`, `feature/autonomous-detection-complete`) are older and behind. Do not use them.

> **I have no visibility into the cluster after 2 Oct 2026.** Everything about the cluster below is as of that date.
> Your very first action is `bash scripts/hpc_status.sh` (section 5, step 2). It reads the disk and tells you what has actually happened since.

---

## CONTENTS

0. [Read this first](#0-read-this-first)
1. [The project in one page](#1-the-project-in-one-page)
2. [Where we stand](#2-where-we-stand)
3. [The cluster: facts, not assumptions](#3-the-cluster-facts-not-assumptions)
4. [Data](#4-data)
5. [Running it, step by step](#5-running-it-step-by-step)
6. [Reading the results honestly](#6-reading-the-results-honestly)
7. [The paper](#7-the-paper)
8. [Known limitations](#8-known-limitations)
9. [When something fails](#9-when-something-fails)
10. [Not built](#10-not-built)
11. [File map](#11-file-map)
12. [Decisions that are not yours to make alone](#12-decisions-that-are-not-yours-to-make-alone)

---

## 0. READ THIS FIRST

**What is missing, in one sentence:** the paper's main result does not exist yet. It needs three trained detectors (queued on 2 Oct), then one comparison job, and until those finish there is no number to put in the paper.

**Four rules that protect the result:**

1. **Never touch `data/gran_*` while any training job is queued or running.** The jobs read those folders when they start and while they train. Changing them underneath a job silently changes what it trains on.
2. **All three arms train identically.** Same model, same epochs, same everything. The label space is the only thing allowed to differ, otherwise the comparison measures something else.
3. **Paste commands one at a time.** A multi-line paste into the cluster terminal broke a command earlier (stray `^[[200~` characters). Every command in this file is a single line for that reason.
4. **Every number you report is one you measured.** Several figures earlier in this project were quoted from memory or from an older run and turned out wrong. Quote only what a result file says.

**First three commands on every login** (a new ssh session starts in `~`, not in the project):

```
cd ~/IBM-INTERNSHIP/autonomous_detection
bash scripts/hpc_status.sh
python scripts/verify_adas_pipeline.py 2>&1 | tail -3
```

The last line must say `ALL CHECKS PASSED (272/272)`. If it does not, stop and tell Vishal.

---

## 1. THE PROJECT IN ONE PAGE

**Problem.** Standard trajectory planners (Frenet, including MATLAB's) take a lane centreline as *input*. Unmarked Indian and South Asian roads have none. The detectors also confuse visually similar vehicles (truck / bus / tanker / tractor) that a planner would treat identically anyway.

**Approach, three ideas:**

1. **Classify by consequence, not appearance.** Six decision groups: `VULNERABLE`, `TWO_WHEELER`, `THREE_WHEELER`, `LIGHT_VEHICLE`, `HEAVY_VEHICLE`, `STATIC_OBSTACLE`. Each carries a lateral clearance in metres and a time-to-collision margin, so the label *changes what the vehicle does*. Defined in `models/taxonomy.py`.
2. **Price errors by consequence (SWMC).** mAP scores "bus called a truck" and "pedestrian called a barrier" the same. Our Safety-Weighted Misclassification Cost does not: an undetected vulnerable road user costs 1.0, a harmless same-group confusion costs 0. Defined in `models/taxonomy.py`, scored by `evaluation/evaluate_swmc.py`.
3. **Plan from free space, and refuse to plan past what was seen.** The reference path is synthesised from the drivable-area mask (no lane paint needed); the corridor width beside each obstacle comes from its group; a "competence gate" truncates any plan that runs beyond the observed road.

**The experiment the paper rests on.** Train the same detector three times on the *same images and same boxes*, labelled at three granularities, then score all three in one common six-group space:

| Arm | Labels | Role |
|---|---|---|
| `fine` | the source datasets' own class names | control |
| `semantic` | grouped by appearance | second control |
| `decision` | our six groups | ours |

The second control matters. Without it, a reviewer can only conclude "fewer classes scores higher", which is trivially true. With it, the question becomes whether grouping by *consequence* beats grouping by *appearance*.

**What a reviewer will ask first:** "Native mAP is higher with fewer classes, so how is this a fair comparison?" Answer: it is not compared on native mAP. All three models are projected to the six-group space before scoring (`evaluate_swmc.py`). Native mAP is printed but explicitly marked non-comparable.

---

## 2. WHERE WE STAND

Cluster rows are as of 2 Oct 2026.

| Area | State | Evidence |
|---|---|---|
| Code: taxonomy, SWMC, planner, bridge, pipeline | **built, tested** | 272/272 automated checks |
| Data: DATS_2022 + HeteroTraffic downloaded, converted | **done** | data build log |
| Three label spaces built, identical boxes | **done** | `OK: all three label spaces cover identical boxes`, **88,613 boxes in each** |
| Three detector trainings | **submitted 2 Oct, last seen queued** | jobs `40037` (decision), `40038` (fine), `40039` (semantic) |
| Comparison job (the main table) | **not started** | needs all three trainings |
| Night / low-light evaluation | **not started** | needs the decision model |
| Closed-loop planner, 5 seeds | **done** | `results/planning.json` |
| Closed-loop planner, 20 seeds | **not started** | `qsub training/pbs/planning_eval.pbs` |
| Annotated demo video | **not started** | needs a trained model, a dashcam clip, a calibration file |
| IEEE paper | **not updated for any of this** | section 7 |
| MATLAB/Simulink port (SIH deliverable) | **not started** | section 10 |

**Submission deadline:** `________` (Vishal to fill in; it is not recorded anywhere I can see).

**Minimum evidence for a defensible paper:** Table 1 (the three-arm comparison) plus Table 3 (the planner). That needs the three trainings, one compare job, and one CPU job. Everything else strengthens it but is not required.

---

## 3. THE CLUSTER: FACTS, NOT ASSUMPTIONS

Each of these was observed, not assumed. Several cost a failed job to learn.

**Python environment. This is the one that cost the most time.**

- The conda env `auto_det` is an **empty shell**: `~/.conda/envs/auto_det/` contains only `conda-meta/` and `etc/`. There is no `bin/python`. Activating it changes your prompt and nothing else.
- Every job that did `conda activate auto_det` aborted with `python resolved to /bin/python`. Three earlier "fixes" (`#PBS -V`, `set -u`, a fallback path) treated symptoms.
- The real stack is **system Python 3.9.21 with packages under `~/.local`**: torch 2.8.0+cu128, ultralytics 8.4.86, and a `yolo` command in `~/.local/bin`. That is what everything has actually run on.
- The six chain PBS scripts now use it directly and check at the *start* of the job that `torch`, `ultralytics`, `cv2` import. A broken environment shows up in one line within seconds.
- You do **not** need `conda activate` for anything here.

**Two copies of the project exist on the cluster.**

| Path | What it is |
|---|---|
| `~/IBM-INTERNSHIP/autonomous_detection` | **the one we use** |
| `~/autonomous_detection` | an older copy; it holds the original weights `yolo11x.pt` and `runs/final/yolo11x_merged*` |

`yolo11x.pt` was copied into the project folder. Keep it there: the training script refuses to start unless its weights file is in the working directory, so a job never tries to download one.

**Queues.** `workq` is CPU, `gpu` is GPU. On 2 Oct the GPU queue had about 22 jobs waiting and 13 running, and some users had held GPUs for 35 to 3000+ hours. Expect to wait.

**Compute nodes: assume no internet.** Not tested, assumed. Download on the login node (`mgmt01`) and keep model weights in the project folder.

**Getting code onto the cluster.** We use scp, not git, on the cluster. The cluster copy was updated by scp over an older clone, so a `git pull` there may conflict. GitHub is the record and the backup; scp is the delivery mechanism.

From a Windows PowerShell window, **code only**, run from inside the `autonomous_detection` folder (do not send `node_modules`, it is thousands of tiny files and makes scp crawl):

```
scp -r data models planning inference evaluation scripts training krish.123449@<CLUSTER_HOST>:~/IBM-INTERNSHIP/autonomous_detection/
```

Replace `<CLUSTER_HOST>` with the login host. Then **always**, on the cluster:

```
sed -i 's/\r$//' training/pbs/*.pbs scripts/*.sh
```

Windows writes CRLF line endings, and without that line every job dies instantly with `bad interpreter: /bin/bash^M`.

**Other things PBS will not do for you:**

- `logs/` must exist before you submit, or the job is rejected with no log to explain why: `mkdir -p logs results`
- A job whose log is two lines long and says ERROR within a minute is almost always the environment, not your data.
- `qstat -u $USER` printing nothing means your job has *finished* (or died), not that it is stuck. See section 9.

---

## 4. DATA

### 4.1 Why not IDD

The plan was built around IDD (IIIT Hyderabad). On 1 Oct its portal was unreachable. Two Mendeley datasets that looked like replacements (**Indistreet2K25**, **IndiaScene365**) are **under embargo until 2030**. What is open and downloadable without registration:

| Dataset | Images | License | Format | Used? |
|---|---|---|---|---|
| **HeteroTraffic** | 17,310 annotated (we measured; see caveat) | CC BY 4.0 | YOLO txt, already | **yes** |
| **DATS_2022** | 1,587 annotated (we measured) | CC BY 4.0 | VOC XML (+ txt, json) | **yes** |
| UVH-26 | 26,646 | CC BY 4.0 | COCO JSON | **no**, see 4.4 |

### 4.2 What each one is

**HeteroTraffic** (Mendeley id `tvkfk56s2k`, version 2, published 7 Nov 2025, DOI `10.17632/tvkfk56s2k.2`).

- Authors as the page lists them: Md Jahidul Alam Sagar and Zannatul Mawa Koli (the page prints the second surname twice; check the spelling on the page before you cite it). Institution: Daffodil International University.
- 17 classes: Motorbike, MPV, Pedestrian, Pickup, PowerTiller, Rickshaw, Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, Leguna, ShoppingVan, Truck, Van, Wheelbarrow. We measured **81,510 boxes** over **17,310 images**.
- **The file we downloaded is `roadvision_blur.zip`**: the privacy version with faces and licence plates blurred. It is the only file the Mendeley API listed for version 2. State this in the paper.
- The page describes the images as "highway and roadside environments", DSLR and smartphone, "different lighting, weather, and traffic densities". **It does not name a country.** The class names (CNG, Easybike, Leguna, Bhotbhoti) and the institution (which is in Dhaka, from my own knowledge, not from the page) suggest Bangladeshi roads. Treat that as a suggestion, not a fact. Until you confirm it with the authors, call the data **"South Asian"**, never "Indian".
- The page text says 16,289 images in one place and 17,310 in another. Our converter found 17,310 annotated images. Report the number you measured.

**DATS_2022** (Mendeley id `nfc34n8svj`, version 2, published 10 May 2022, contributor Bhakti Paranjape).

- Indian roads, captured in photo *and* video modes. There is an associated data article (PubMed Central `PMC9309657`, "DATS_2022: A versatile Indian dataset for object detection in unstructured traffic conditions"). Verify its journal, volume and pages, and copy the dataset DOI from the Mendeley page, before citing. I have not confirmed either.
- 6.6 GB, 5,518 files in folders `Images with Annotations/{Images,XML files,JSON files,TXT files}` and `Images and XML files/{...}`.
- We measured **1,587 annotated images, 10,773 boxes, 61 distinct label strings**.
- **It annotates the whole scene, not just road users.** Trees, lamp posts, buildings, road dividers, the road itself and signage are labelled. About a third of its boxes are scenery and are **excluded by design** (`EXCLUDED_CLASSES` in `models/taxonomy.py`), because none is a path obstacle a detector should be trained to report.
- Label noise: one box is labelled with a string of repeated `W` characters (the only unrecognised class, 0.0% of boxes), and class names appear in both cases (`Person` 366 and `person` 833 are the same class).

### 4.3 What this means for the groups

| Group | Source | Boxes measured |
|---|---|---|
| `VULNERABLE` (people) | HeteroTraffic Pedestrian | 10,578 |
| | DATS Person + person | 1,199 |
| | DATS Traffic Police, Hawker | 30 |
| `VULNERABLE` (animals, animal carts) | DATS Cattle, Goat, Dog, Camel, Horse, Bullock Cart | **about 250** |
| `STATIC_OBSTACLE` | HeteroTraffic Wheelbarrow | 1,072 |
| | DATS Cart, Barricade, Garbage Bin | about 31 |

Two consequences you must carry into the paper:

- **The animal evidence is thin.** About 250 boxes. `VULNERABLE` is dominated by pedestrians, so any claim about *animals* is statistically weak and should be worded that way.
- `STATIC_OBSTACLE` has around 1,100 boxes. Small, but enough to score.

### 4.4 UVH-26 is not used, deliberately

Hugging Face allows **1000 API requests per 5 minutes** and UVH-26 is about 26,650 separate files. Our download stopped at 1,000 files and 3.3 GB with a `429 Too Many Requests`. Extrapolating, the full set is very roughly 85 to 90 GB, and at the rate limit it needs at least about two hours if each file costs one request.

The half-download is parked at `data/UVH26_partial` on the cluster. It is not used. **Two safeguards now exist in code** so it cannot leak into training by accident:

- `download_open_datasets.py` no longer downloads UVH-26 unless you ask for it (`--only uvh26`).
- The data-build job only converts UVH-26 if a `data/UVH26/.complete` marker exists, and that marker is written only when the download finishes.

If you ever want it: make a free Hugging Face account, create a read token, `export HF_TOKEN=<token>` (keep it private), run `python data/download_open_datasets.py --only uvh26`, then re-run the data build **and retrain all three arms**, because the dataset has changed. Do this before training starts or not at all.

### 4.5 Downloading from scratch

On the **login node**:

```
cd ~/IBM-INTERNSHIP/autonomous_detection
python -m pip install --user -q pillow huggingface_hub
nohup python data/download_open_datasets.py > logs/download.log 2>&1 &
du -sh data/DATS_2022 data/HeteroTraffic
```

About an hour for both. The log stays empty while it runs (Python buffers output); watch the folder sizes instead. It is resumable: if it is interrupted, run it again and it skips files that are already complete. HeteroTraffic's zip is checked against Mendeley's published SHA-256 before unpacking.

Then **look before converting** (writes nothing):

```
python data/prepare_indian.py --src data/DATS_2022 --out data/dats_native --report-only
python data/prepare_indian.py --src data/HeteroTraffic --out data/hetero_native --report-only
```

Read the `UNRECOGNISED` lines. For both datasets the expected result is: every HeteroTraffic class maps, and DATS shows exactly one unrecognised box. Anything else is a hole in the mapping and must be fixed in `models/taxonomy.py` *before* any GPU time is spent.

### 4.6 Caveats the paper must carry

- **Random image-level split.** `prepare_indian.py` shuffles images and holds out 20% (seed 0), per source. DATS was captured in photo *and video* modes, and neighbouring video frames are near-duplicates, so a random split can put almost-identical frames in both train and val and inflate every score. HeteroTraffic's capture method is not stated. This is a real threat to validity. Mitigation if time allows: split DATS by capture session instead of by image. At minimum, say it in the limitations.
- **No official benchmark split exists for this combined corpus.** It is a corpus we assembled.
- **Night data may be too thin to report.** IDD-AW was the intended night source. With the open data the lighting buckets (day, lit night, unlit night) may be tiny or empty. The night job flags buckets below 100 images instead of reporting a confident number. If they are empty, the honest result is "insufficient data", and the night claim should be dropped or reported as such.

### 4.7 Not verified, worth a look if time allows

DriveIndia (66,986 images, IEEE ITSC 2025) was in the original plan. I never checked whether it is downloadable. IDD may come back; if it does, `data/prepare_idd.py` exists and the data build picks up `data/idd_native` automatically, but that again means retraining.

---

## 5. RUNNING IT, STEP BY STEP

### Step 1. Log in and enter the project

```
cd ~/IBM-INTERNSHIP/autonomous_detection
```

### Step 2. Find out what has actually happened

```
bash scripts/hpc_status.sh
qstat -u $USER
```

`hpc_status.sh` prints DONE / PARTIAL / TODO for every stage and ends with the next command. `qstat -u $USER` shows your own jobs: `Q` queued, `R` running, `H` held, nothing means finished.

### Step 3. If the three trainings are still queued: why?

```
qstat -f 40037.mgmt01 | grep -E "job_state|comment"
```

The `comment` line states the scheduler's reason. **Do not `qdel` a queued job to "fix" it.** You lose your place in the queue.

The three jobs were submitted as:

| Job | Arm | Resources | Walltime |
|---|---|---|---|
| `40037` | decision | 1 GPU, 4 CPU, 20 GB | 24 h |
| `40038` | fine | same | 24 h |
| `40039` | semantic | same | 24 h |

All use `yolo11x`, 100 epochs, patience 25, seed 0. They were submitted with the script as it was on 2 Oct. Later edits to the script do not affect jobs that were already submitted.

### Step 4. When a training job reaches `R`: check it within two minutes

```
tail -20 logs/gran_decision.log
```

The first lines must show `python : ... (Python 3.9.x)` and `environment: OK`. Most failures happen in the first 30 seconds, and there is no point holding a GPU for a day on a job that has already died.

Then find how fast it trains:

```
tail -c 800 logs/gran_decision.log
```

The progress line shows the epoch and iteration speed.

**The walltime risk.** Each job has 24 hours for 100 epochs. The time per epoch on this cluster's GPU is unknown to me. If one epoch takes more than about 14 minutes (24 h divided by 100, before validation time), 100 epochs will not fit and the job will be killed at hour 24. Decide in the first hour, not at hour 24.

**If it will not fit,** cancel and resubmit **all three** with a smaller, identical setting. The script now accepts `MODEL` and `EPOCHS` for exactly this. Put the smaller weights in the project folder first (login node), for example:

```
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

Then, once per arm, changing only `LEVEL`:

```
qsub -N gran_decision -o logs/gran_decision.log -v LEVEL=decision,MODEL=yolo11m.pt,EPOCHS=50 training/pbs/granularity_step2_train.pbs
qsub -N gran_fine -o logs/gran_fine.log -v LEVEL=fine,MODEL=yolo11m.pt,EPOCHS=50 training/pbs/granularity_step2_train.pbs
qsub -N gran_semantic -o logs/gran_semantic.log -v LEVEL=semantic,MODEL=yolo11m.pt,EPOCHS=50 training/pbs/granularity_step2_train.pbs
```

`MODEL` and `EPOCHS` must be the same on all three. A smaller model scores lower in absolute terms; that is acceptable because the experiment compares the *arms*, but **write down what you used** and report it in the paper. Upload the current `training/pbs/granularity_step2_train.pbs` first (it carries the `MODEL`/`EPOCHS` knob), then run the `sed` line from section 3.

If a job is killed at the wall anyway, `runs/granularity/<arm>/weights/best.pt` and `last.pt` exist. Using them is legitimate only if you report the epochs each arm actually completed. Unequal epochs between arms weaken the comparison.

### Step 5. When all three have finished: the comparison (the main table)

```
ls runs/granularity/fine/weights/best.pt runs/granularity/semantic/weights/best.pt runs/granularity/decision/weights/best.pt
qsub training/pbs/granularity_step3_compare.pbs
tail -f logs/gran_step3.log
```

Outputs: `results/granularity_comparison.json` (conf 0.25) and `results/granularity_comparison_conf010.json` (conf 0.10, to show the conclusion is not an artefact of one threshold).

### Step 6. Night and low light (needs only the decision model)

```
qsub -v LEVEL=decision training/pbs/india_night_eval.pbs
```

Output: `results/conditions_decision.json`. See the caveat in 4.6: the buckets may be too small to report.

### Step 7. The closed-loop planner at 20 seeds (CPU only, runs any time)

This needs **no GPU and no dataset**. You can submit it right now, while everything else waits:

```
qsub training/pbs/planning_eval.pbs
tail -f logs/planning_eval.log
```

Output: `results/planning_20seed.json`. The 5-seed run in the repo is too noisy to quote (section 6).

### Step 8. Annotated video (optional, GPU, interactive)

You need a dashcam clip (ask Vishal; some earlier clips are on his laptop, not in the repo) and a camera calibration file. Without the calibration the pipeline produces **no trajectory** on any frame and says so in `plan_status`. The old way to make one needs a window and mouse clicks, which does not work on the cluster. Use the new script (assumed camera values, not measured):

```
python scripts/make_ipm_calibration.py --video demo/clip.mp4
qsub -I -l select=1:ncpus=4:ngpus=1:mem=20gb -q gpu -l walltime=02:00:00
python inference/adas_final.py --source demo/clip.mp4 --weights runs/granularity/decision/weights/best.pt --save out_annotated.mp4 --log decisions.jsonl
```

Say on the video that distances come from **assumed** camera parameters. Read the `[collision]` block printed at the end: closing-speed coverage below 0.60 means the collision layer often could not measure, so quiet stretches do not mean the road was clear.

### Step 9. Bring results back

Run from your own machine, not the cluster:

```
scp -r krish.123449@<CLUSTER_HOST>:~/IBM-INTERNSHIP/autonomous_detection/results ./results_from_hpc
scp -r krish.123449@<CLUSTER_HOST>:~/IBM-INTERNSHIP/autonomous_detection/logs ./logs_from_hpc
```

The JSON files are small and belong in the repo, so commit them from the laptop. The weights are large; do not commit them. To send a status snapshot in chat: `bash scripts/hpc_status.sh > status.txt` (colour is stripped automatically).

---

## 6. READING THE RESULTS HONESTLY

**Main table, from `granularity_comparison.json`.** Report **SWMC** (lower is better), **critical-error rate** (fraction of objects missed or called less cautious than they are), **group accuracy** and **detection recall**. All four are measured in the same six-group space and are comparable across arms. For each arm also report `native_num_classes`, so a reader can see how many classes each arm really trained on (read the exact counts from `data/gran_*/*.yaml`; I do not have them).

**Native mAP is printed but is not comparable across rows.** Fewer classes is an easier problem. Put it in the table only with an explicit label saying so. Claiming otherwise is the first thing a reviewer will catch.

**Closed-loop planner.** Last measured, 5 seeds, `results/planning.json`:

| Scenario | Grouped: complete | Uniform: complete | Pedestrian gap, grouped / uniform |
|---|---|---|---|
| village road + pedestrian | 80% | 100% | **4.58 m / 2.70 m** |
| occluding truck | 80% | 100% | n/a |
| mountain bend | 80% | 80% | n/a |
| stopped auto-rickshaw | 0% | 0% | n/a |
| cattle + handcart | 0% | 0% | n/a |

Collision rate: **0% in every scenario and arm.**

**Read that table as a trade-off, not a win.** The decision taxonomy leaves a pedestrian about 1.7 times the clearance and also completes the route *less* often. It buys margin and pays in progress. Present it exactly that way. An earlier version of this project reported "better safety *and* better progress"; that came from a run with a bug in it and was wrong.

**These numbers move between runs.** One scenario scored 0% at 3 seeds and 80% at 5. Five seeds cannot separate 80% from 100%. Use the 20-seed job and quote only that.

**If a result comes out unfavourable, report it.** If the decision arm does not beat the fine arm under SWMC, that is the finding and the paper says so. If low-light enhancement is neutral or harmful, that is a finding too (enhancement can amplify noise into texture a domain-shifted detector reads as an object).

---

## 7. THE PAPER

Files, in the repo-root `paper/` folder (one level above `autonomous_detection/`): `adas_paper.tex` (6-page IEEE conference format), `project_report.tex` (long report), `CITATIONS_VERIFIED.md`, `README.md`.

**The paper has not been updated for any of this project's later work.** I checked: it contains no mention of SWMC, the three-granularity experiment, the Frenet planner, HeteroTraffic or DATS. It was written before them. So the contribution section, the experiments and the abstract all need rewriting around the results above, not just filling in blanks.

**Placeholders still in `adas_paper.tex`:**

| Where | What |
|---|---|
| lines 37 to 55 | author blocks: `TODO: Institution`, city, email for all four people |
| lines 579 to 649 | Section VI-E and Table IV: seven `XX.X` cells and `TODO(authors)` notes |

**Section VI-E cannot be filled as written.** It describes an adaptation experiment (a KITTI-trained detector against one adapted on UVH-26, via `india_step1_prepare.pbs` and `india_step2_compare.pbs`). UVH-26 is no longer used, and those two scripts download it from Hugging Face, which rate-limits; expect them to fail. Either rewrite that section around the granularity experiment or delete it. Do not run those two scripts.

**Suggested results to show**, each backed by one result file:

| Table | Content | Source |
|---|---|---|
| 1 | three arms: SWMC, critical-error rate, group accuracy, recall; native mAP flagged non-comparable | `granularity_comparison.json` |
| 2 | per-condition SWMC, raw vs enhanced (only if buckets are large enough) | `conditions_decision.json` |
| 3 | closed-loop: completion, collisions, pedestrian gap, grouped vs uniform vs gate-off | `planning_20seed.json` |

**Citations to settle before submission:**

- IRC:66-1976 (overtaking sight distance) was taken from secondary sources. Check the clause numbers against the standard.
- Two works cited in `MASTER_DOCUMENT.md` are preprints with no confirmed venue: the Frenet Corridor Planner (arXiv 2505.03695) and Diffusion-FS (arXiv 2507.18763). Cite as arXiv or replace.
- DriveIndia needs page numbers from IEEE Xplore.
- Dataset citations: HeteroTraffic by its DOI (above); DATS_2022 by its Mendeley DOI plus the data article, after verifying both.
- Supervisor names: "Pranshu Chander" and "Bhushan Singh Negi" is an interpretation of one run-together string. Confirm.

**Wording rules that follow from section 4:** say "South Asian" for HeteroTraffic; say the HeteroTraffic images are the face-and-plate-blurred version; describe the animal evidence as limited (about 250 boxes); state the random split and its leakage risk.

---

## 8. KNOWN LIMITATIONS

**The planner completes three of five scenarios, not five.**

| Scenario | Why it stalls |
|---|---|
| stopped auto-rickshaw | The gap beside it is 0.25 m wide, real and passable, but the planner's fixed lateral offsets (-2, -1, -0.5, 0, 0.5, 1, 2 m) never land inside it |
| cattle + handcart | The planner holds the centreline until the corridor forces a move, and by then the lateral shift no longer fits inside the horizon |

Every failure is a refusal, not a collision. That is the safe direction, but it is still a defect and the paper should say so.

**Low-speed steering check is unreliable below about 1 m/s.** The lateral curvature is computed with a `1/v²` divisor, which blows up as speed falls, so a vehicle that has stopped is slow to steer around an obstacle. Two attempted replacements both made things worse and were reverted (documented in `planning/frenet.py`). A proper fix computes curvature from the Cartesian trajectory after the Frenet-to-world transform, where it is well-conditioned. That is a larger change than a patch.

**Two fixes that were written, measured and removed** (so nobody re-adds them): deriving extra lateral candidates from the corridor walls (it pinned the vehicle against a boundary; completion fell from 100% to 0%), and arc-length curvature (failed five checks).

**The planner is evaluated on synthetic scenarios.** There is no ground truth for "the correct trajectory" in recorded footage, so closed-loop simulation is the honest way to measure it, but it is simulation. The paper must not describe it as validation on real roads.

**Curvature validation is synthetic too.** It tests the estimator against arcs of known radius. It is not validation on real curved roads.

**The IRC:66 sight-distance figures deviate from the published table by -16% to +18%** across 40 to 100 km/h. State it.

**Camera calibration is assumed**, not measured, for any demo video.

---

## 9. WHEN SOMETHING FAILS

| In the log or `qstat` | Cause | Fix |
|---|---|---|
| `python resolved to /bin/python` | the old conda-based scripts | you have a stale script; re-upload `training/pbs/*.pbs` |
| `ERROR: this python cannot import torch / ultralytics / cv2` | that node's Python cannot load the packages | `python -m pip install --user ultralytics` on the login node, resubmit |
| `bad interpreter: /bin/bash^M` | CRLF line endings from Windows | `sed -i 's/\r$//' training/pbs/*.pbs` |
| rejected instantly, no log | `logs/` does not exist | `mkdir -p logs` |
| `ERROR: yolo11x.pt not found` | weights not in the project folder | copy it there on the login node |
| `ModuleNotFoundError: No module named 'models'` | project root not on `sys.path` | already fixed in every entry point; run from the project root |
| `429 Too Many Requests` (Hugging Face) | UVH-26 rate limit | not needed; see 4.4 |
| `UNRECOGNISED` boxes in the data build | a class name is missing from `models/taxonomy.py` | send Vishal the output; one-line fix |
| box counts differ across `data/gran_*` | taxonomy vocabularies out of sync | do not train; send Vishal the log |
| `qstat -u $USER` prints nothing | your job finished or died | run `qstat -xf <id>`, read `job_state` and `exit_status`, then read the log |
| a long command shows an empty log | Python buffers output when not on a terminal | watch `du -sh` or `ls`, not the log |

---

## 10. NOT BUILT

Stated plainly so nobody plans around something that does not exist.

- **The MATLAB/Simulink port.** The SIH problem statement asks for a closed-loop MATLAB/Simulink simulation (Automated Driving Toolbox, RoadRunner scenes, camera + LiDAR + radar, agent motion prediction). None of that exists. The algorithms are implemented and verified in Python so the port can be a translation of working logic. See `MASTER_DOCUMENT.md` and `SIH_2026_PROPOSAL.md`.
- **Motion prediction** for other agents. The corridor sweeps each obstacle along its current velocity, a constant-velocity model, not a learned prediction.
- **LiDAR and radar.** Camera only.
- **Real camera calibration.**

---

## 11. FILE MAP

| Concern | File |
|---|---|
| Six groups, clearances, SWMC costs, name mapping | `models/taxonomy.py` |
| Camera to ground homography | `models/ipm.py` |
| Write a calibration file without the GUI | `scripts/make_ipm_calibration.py` |
| Drivable mask to reference path | `planning/reference_path.py` |
| Obstacle corridor | `planning/corridor.py` |
| Trajectory generation, competence gate | `planning/frenet.py` |
| Pixels to metres (perception to planning) | `planning/perception_bridge.py` |
| Whole pipeline, one frame in | `inference/adas_final.py` |
| Collision (TTC) | `inference/collision.py` |
| Overtaking (forward, rear, IRC:66) | `inference/overtaking.py`, `inference/rear_view.py` |
| Convert any dataset to YOLO | `data/prepare_indian.py` |
| Collapse to the three label spaces | `data/prepare_taxonomy.py` |
| Download the open datasets | `data/download_open_datasets.py` |
| Detection metric (SWMC) | `evaluation/evaluate_swmc.py` |
| Night / condition metric | `evaluation/evaluate_conditions.py` |
| Closed-loop planner metric | `evaluation/evaluate_planning.py` |
| 272 automated checks (GPU-free, about a minute) | `scripts/verify_adas_pipeline.py` |
| Where am I, what next | `scripts/hpc_status.sh` |
| PBS jobs in order | `training/pbs/granularity_step1_prepare.pbs`, `granularity_step2_train.pbs`, `granularity_step3_compare.pbs`, `india_night_eval.pbs`, `planning_eval.pbs` |
| Architecture and data flow, in depth | `KRISH_IMPLEMENTATION_GUIDE.md` (background only; this file wins on any conflict) |

`GPU_RUN_SHEET.md`, `HPC_RUNBOOK.md` and `KRISH_IMPLEMENTATION_GUIDE.md` are older and marked superseded. `india_step1_prepare.pbs`, `india_step2_compare.pbs`, `aux_and_eval.pbs`, `clrnet_culane.pbs`, `yolo11x_merged.pbs` are not part of this chain; do not run them.

---

## 12. DECISIONS THAT ARE NOT YOURS TO MAKE ALONE

Bring these to Vishal rather than settling them quietly:

1. **Training size.** If the GPU queue will not allow `yolo11x` for 100 epochs, whether to drop to `yolo11m` with fewer epochs. It changes absolute scores, so it must be decided and recorded once for all three arms.
2. **Dropping the night claim** if the lighting buckets turn out too small.
3. **Reframing Section VI-E** of the paper, or deleting it.
4. **How to word the dataset origin** once the HeteroTraffic country is confirmed.
5. **Whether to split DATS by capture session** (more rigorous, more work) or disclose the random-split limitation.
6. **The submission deadline.** It is not written down anywhere I can see.
