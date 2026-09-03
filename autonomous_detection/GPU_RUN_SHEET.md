# GPU Run Sheet — what is done, what to run next

**One command answers "how far have I got?"** — it reads the actual disk state
rather than anyone's memory, and ends by printing the single next command:

```bash
cd ~/IBM_Internship/autonomous_detection
bash scripts/hpc_status.sh
```

Run it any time. It only reads; it never submits anything.

---

## THE SHORT VERSION

| # | Job | Where | Time | Needs |
|---|---|---|---|---|
| 0 | Download IDD | login node | ~1 h | registration |
| 1 | Convert to YOLO | login node | ~10 min | job 0 |
| 2 | Build 3 label spaces | **CPU queue** | ~1 h | job 1 |
| 3 | Train 3 models | **GPU × 3** | 8–24 h each | job 2 |
| 4 | Compare them | **GPU** | ~2 h | job 3 (all three) |
| 5 | Night / low light | **GPU** | ~3 h | job 3 |
| 6 | Closed-loop planner | **CPU** | ~20 min | nothing — run it today |
| 7 | Annotated video | GPU interactive | ~30 min | job 3 |

**Only jobs 3, 4 and 5 need a GPU.** Job 6 needs nothing at all and produces a
real result — start there if you are waiting on the IDD download.

---

## WHAT IS ALREADY DONE (no GPU was needed)

These are finished and verified. Nothing to re-run.

| Result | Number |
|---|---|
| Automated checks passing | **260 / 260** |
| Curve radius recovered from known arcs | within **4%** at the 150 m decision threshold |
| IRC:66 overtaking sight distance vs published table | **−16% to +18%** across 40–100 km/h |
| Ground homography round-trip accuracy | **3e-14 m** (machine precision) |
| Collision coverage under ID churn every 3 frames | **0.00 → 0.93** after the history-recovery fix |
| TTC margin, pedestrian vs car | **2.70 s vs 1.50 s** (set by the taxonomy) |
| Closed-loop pedestrian clearance, grouped vs uniform | **4.20 m vs 1.73 m** |
| Closed-loop route completion, grouped vs uniform | **100% vs 33%** |

That last pair is the headline: on the same road with the same pedestrian, the
decision taxonomy left **two and a half times the clearance** *and* completed
the route **three times as often**. Better safety and better progress at once,
which is the argument the whole design rests on.

Those two closed-loop figures come from a run made *before* the most recent
planner fixes (the low-speed curvature and look-ahead changes). They should
hold or improve — every fix removed a case where the planner stalled — but
**job 6 re-derives them in twenty minutes on a CPU, so confirm rather than
quote these.** Everything above them in the table is re-derived by the
verification suite on every run.

---

## STEP BY STEP

### Before anything

```bash
cd ~/IBM_Internship/autonomous_detection
git pull                                    # branch: krish-implementation
mkdir -p logs results                       # PBS rejects jobs without logs/
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate auto_det
python scripts/verify_adas_pipeline.py      # expect: ALL CHECKS PASSED (260/260)
```

**If the check count is not 260, stop and tell Vishal.** The failure is real
and every job below will hit it.

---

### Job 0 — get the data (login node, no GPU)

The problem statement names **IDD** (https://idd.insaan.iiit.ac.in/) — needs
registration — and Mendeley Indian traffic datasets.

Download on the **login node**, not a compute node: compute nodes usually have
no outbound internet.

Unpack into `data/IDD_Detection/`.

---

### Job 1 — convert to YOLO (login node, ~10 min)

**Look before you convert:**

```bash
python data/prepare_indian.py --src data/IDD_Detection --out data/idd_yolo --report-only
```

This writes nothing. It prints every class name, its box count, and which
decision group it maps to.

**Read the UNRECOGNISED line.** An unrecognised class is data about to be
silently discarded. It is far cheaper to see that here than to notice a missing
class after 24 GPU-hours. If anything is unrecognised, send Vishal the output —
it is a one-line fix in `models/taxonomy.py`.

If everything maps:

```bash
python data/prepare_indian.py --src data/IDD_Detection --out data/idd_yolo
```

Auto-detects VOC XML, COCO JSON and YOLO txt, so the same command works for
DATS_2022, HeteroTraffic, IndiaScene365 and Indistreet2K25.

---

### Job 2 — build the three label spaces (CPU queue, ~1 h)

```bash
qsub training/pbs/granularity_step1_prepare.pbs
qstat -u $USER
tail -f logs/gran_step1.log
```

Produces three copies of the dataset with the **same images and the same
boxes**, differing only in labelling:

| Level | Classes | Role |
|---|---|---|
| `fine` | the source dataset's own | control |
| `semantic` | grouped by appearance | second control |
| `decision` | our six groups | ours |

The second control is what makes the experiment mean anything — without it a
reviewer can only conclude "fewer classes scores higher", which is trivially
true and proves nothing.

**Wait for this line:**

```
OK: all three label spaces cover identical boxes.
```

If the three box counts differ, **do not train**. The comparison would be
measuring a difference in data rather than in taxonomy, and every number after
it would be meaningless.

---

### Job 3 — train the three models (GPU × 3, 8–24 h each)

Independent — they can queue together. **Give each its own name and log** or
they overwrite each other:

```bash
qsub -N gran_fine     -o logs/gran_fine.log     -v LEVEL=fine \
     training/pbs/granularity_step2_train.pbs

qsub -N gran_semantic -o logs/gran_semantic.log -v LEVEL=semantic \
     training/pbs/granularity_step2_train.pbs

qsub -N gran_decision -o logs/gran_decision.log -v LEVEL=decision \
     training/pbs/granularity_step2_train.pbs
```

One GPU at a time is fine — submit them one after another, order does not
matter.

**Every hyperparameter, including the random seed, is identical across the
three. Do not tune one arm and not the others** — that identity is the entire
basis of the comparison.

**Watch the first minute.** Most failures happen in the first thirty seconds
and there is no point holding a GPU for a day on a job that already died:

```bash
tail -f logs/gran_decision.log
```

The first lines must read:

```
python : /home/soft/anaconda3/envs/auto_det/bin/python
conda environment: OK
```

If you see `python resolved to /bin/python`, the job aborts by design — the
environment did not activate.

**Check when done:** `ls runs/granularity/*/weights/best.pt` — all three.

---

### Job 4 — the comparison (GPU, ~2 h) ← the main results table

```bash
qsub training/pbs/granularity_step3_compare.pbs
tail -f logs/gran_step3.log
cat results/granularity_comparison.json
```

Scores all three models in **one common six-group space**, whatever space each
was trained in, at two confidence thresholds so the conclusion is not an
artefact of a single operating point.

**Report these columns:** SWMC, critical-error rate, group accuracy, recall.
All measured in the same space, all comparable.

**Native mAP is printed but is NOT comparable across rows** — fewer classes is
an easier problem. The job prints that warning next to the table. Keep it in
whatever you send on; claiming otherwise is the first thing a reviewer will
catch.

---

### Job 5 — night and low light (GPU, ~3 h)

```bash
qsub -v LEVEL=decision training/pbs/india_night_eval.pbs
cat results/conditions_decision.json
```

Buckets the validation set with the pipeline's own `scene_lighting()` and
reports SWMC per condition, plus the enhancement ablation.

Two things to read honestly:

- A bucket flagged **"too few images"** — report the count, not the metric. A
  confident number over 30 frames is noise dressed as a result.
- If enhancement comes out **neutral or negative, that is the finding.**
  Enhancement can amplify sensor noise into texture a domain-shifted detector
  reads as an object. Do not bury it.

---

### Job 6 — closed-loop planner (CPU, ~20 min) ← run this today

**No GPU, no dataset, no waiting.** Runs right now:

```bash
python evaluation/evaluate_planning.py --seeds 20 --out results/planning.json
```

Drives the planner through five Indian scenarios — village road with a
pedestrian stepping in, a stopped auto-rickshaw, an occluding truck, cattle
beside a handcart, a mountain bend — with three ablations: our grouped
clearances, a uniform clearance for everything, and the competence gate
disabled.

**Read the VULNERABLE-gap column first.** It is the body-to-body distance the
vehicle actually left a pedestrian or animal, measured from the scenario's true
agent positions rather than from what the planner believed. If `grouped` does
not beat `uniform` there, the taxonomy is not earning its place and we say so.

---

### Job 7 — annotated video (GPU interactive)

```bash
qsub -I -l select=1:ncpus=8:ngpus=1:mem=40gb -q gpu -l walltime=02:00:00

cd ~/IBM_Internship/autonomous_detection
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate auto_det

python inference/adas_final.py \
    --source demo/india_clip.mp4 \
    --weights runs/granularity/decision/weights/best.pt \
    --save out_annotated.mp4 \
    --log decisions.jsonl
```

(`--source`, not `--video`.)

**Read the `[collision]` block at the end** — it says whether tracking is
starving the collision layer:

```
[collision] 18432 object sightings over 3604 frames
  closing-speed coverage : 0.87
  new track ids          : 412 (11.4 per 100 frames)
  histories rescued      : 96
```

- **coverage below 0.60** — the collision layer frequently could not measure,
  so quiet stretches in the video do **not** mean the road was clear.
- **rescued above ~30% of new ids** — the recovery is holding the module up but
  the tracker is unstable. *Then*, and only then, try `botsort.yaml` or a
  longer `track_buffer`. The number decides it, not a hunch.

---

## WATCHING JOBS

```bash
qstat -u $USER              # your queued and running jobs
qstat -f <jobid>            # why one is stuck, or its exit status
tail -f logs/<name>.log     # live output
qdel <jobid>                # kill one
bash scripts/hpc_status.sh  # where am I, what next
```

---

## WHEN A JOB FAILS

Every failure this cluster has produced, and what it looks like:

| In the log | Cause | Fix |
|---|---|---|
| `python: /bin/python`, then import errors | `#PBS -V` exported CONDA_SHLVL, so `conda activate` silently did nothing | already removed from every script |
| `PYTHONPATH: unbound variable` | `set -u` with an unset variable | already removed |
| `bad interpreter: /bin/bash^M` | CRLF from a Windows copy | `sed -i 's/\r$//' training/pbs/*.pbs` |
| rejected instantly, no log at all | `logs/` does not exist | `mkdir -p logs` |
| `pip: command not found` | console scripts not on PATH in a batch shell | already handled via `python -m pip` |
| `ModuleNotFoundError: No module named 'models'` | project root not on `sys.path` | already fixed in every entry point |
| `ERROR: missing runs/granularity/.../best.pt` | a previous job has not finished | wait; the job checks its inputs on purpose |

**A job that dies within thirty seconds is almost always the environment, not
your data.**

---

## SENDING RESULTS BACK

```powershell
scp -r USER@CLUSTER:~/IBM_Internship/autonomous_detection/results ./results_from_hpc
scp -r USER@CLUSTER:~/IBM_Internship/autonomous_detection/logs ./logs_from_hpc
scp USER@CLUSTER:~/IBM_Internship/autonomous_detection/out_annotated.mp4 .
```

Weights are large — `results/` and `logs/` are enough for the numbers.

A quick status snapshot to paste into chat:

```bash
bash scripts/hpc_status.sh > status.txt
```

(Colour is dropped automatically when the output is not a terminal, so the file
stays readable.)

---

## WHAT IS NOT BUILT

Stated plainly so nobody plans around something that does not exist:

- **Motion prediction** for other agents. The corridor sweeps obstacles along
  their current velocity — a constant-velocity model, not a learned prediction.
- **The MATLAB/Simulink port.** The algorithms are verified in Python
  deliberately, so the port becomes a translation of working logic rather than
  fresh untested code.
- **LiDAR and radar.** Camera only.
- **RoadRunner scenes.**

---

## IF YOU GET STUCK

```bash
bash scripts/hpc_status.sh          # where am I, what next
python scripts/verify_adas_pipeline.py   # is the code sound — expect 260/260
```

The verification suite is the fastest diagnostic — GPU-free, about a minute,
260 checks. It caught nine defects that would otherwise have surfaced only
after hours of cluster time, two of them in code already running on the
cluster.

The closed-loop harness (job 6) caught a further five that the unit checks
could not, because they only appear once the vehicle is actually driving: it
could not pull away from a standstill, it deferred lateral manoeuvres until
they no longer fitted, its fixed offset grid missed narrow gaps beside parked
obstacles, its look-ahead probe sampled a single point and missed localised
constrictions, and its curvature check divided by v² and so rejected almost
every candidate at crawling speed. Run both.
