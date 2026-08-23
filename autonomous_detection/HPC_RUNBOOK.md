# HPC RUNBOOK — upload, then run the jobs one at a time

Every command below is copy-paste. Replace `USER` with your cluster username
and `CLUSTER` with the login host.

Two rules that save the most time:

- **Never submit without running the verification suite first.** It is GPU-free,
  takes a minute, and has caught nine defects that would otherwise have cost
  hours of queue time.
- **Read each job's log before submitting the next one.** These jobs are
  chained: step 2 cannot succeed if step 1 half-finished, and a job that fails
  in its first thirty seconds still occupies a queue slot until you notice.

---

## 0. UPLOAD THE CODE

### First time — clone on the cluster

The repository is on GitHub, so the cleanest route is to pull it there rather
than copy from Windows. Line endings stay correct automatically (`.gitattributes`
forces LF on `.pbs` and `.sh`, which matters — a CRLF checkout makes Linux
report `bad interpreter: /bin/bash^M`, which reads as a missing shell).

```bash
ssh USER@CLUSTER
cd ~                          # or wherever your project lives
git clone <repo-url> IBM_Internship
cd IBM_Internship/autonomous_detection
```

### Every time after — pull the latest

```bash
ssh USER@CLUSTER
cd ~/IBM_Internship
git fetch origin
git checkout feature/autonomous-detection-complete
git pull
```

If Krish is working on `first-execution`, use that branch instead — both carry
the same commits.

### If you cannot use git on the cluster

From Windows PowerShell, in the project folder:

```powershell
scp -r autonomous_detection USER@CLUSTER:~/IBM_Internship/
```

**Then fix the line endings**, because Windows will have written CRLF:

```bash
ssh USER@CLUSTER
cd ~/IBM_Internship/autonomous_detection
sed -i 's/\r$//' training/pbs/*.pbs scripts/*.sh
```

Skip this and every job dies instantly with `bad interpreter: /bin/bash^M`.

### One-time setup on the cluster

```bash
cd ~/IBM_Internship/autonomous_detection
mkdir -p logs results          # PBS cannot create the log directory itself;
                               # without it the job is rejected before it starts
ls training/pbs/               # sanity: the scripts arrived
```

---

## 1. VERIFY BEFORE SUBMITTING ANYTHING

```bash
cd ~/IBM_Internship/autonomous_detection
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate auto_det
python scripts/verify_adas_pipeline.py
```

Expect `ALL CHECKS PASSED (186/186)`. **If anything fails, stop and fix it** —
the failure is real and the jobs will hit it several hours in.

This runs on the login node and needs no GPU.

---

## 2. THE JOBS, IN ORDER

### Job 1 — prepare the datasets  (CPU, ~1 h)

```bash
qsub training/pbs/granularity_step1_prepare.pbs
```

Builds the three label spaces (~21 / ~10 / 6 classes) over identical boxes.

**Before checking anything else, read the mapping report at the top of the
log.** It runs in seconds, prints what each class maps to, and stops the job if
more than 5% of boxes belong to classes the taxonomy does not recognise. That
is the single most useful thing in this whole pipeline: it catches a mapping
hole in seconds instead of after twenty-four GPU-hours.

```bash
qstat -u $USER                 # R = running, Q = queued
tail -f logs/gran_step1.log    # Ctrl-C to stop watching; the job keeps going
```

**Done when the log ends with:**

```
OK: all three label spaces cover identical boxes.
```

**Check before moving on:**

```bash
ls data/gran_fine data/gran_semantic data/gran_decision
head -20 data/gran_decision/decision.yaml
```

If the three box counts differ, **do not continue** — the comparison would
measure a difference in data rather than in taxonomy, and the result would be
worthless. The log says so explicitly if it happens.

If the log says *"no IDD source"*, the experiment will run but VULNERABLE
(pedestrians, riders, animals) will not appear in any arm, because UVH-26 is
vehicles only. Usable, but the safety result will not test the case it exists
for. Add IDD when access comes through and re-run.

---

### Job 2 — train the three granularities  (GPU, ~8–24 h each)

These three are independent and may queue at the same time. Give each its own
job name and log, or they overwrite each other's output:

```bash
qsub -N gran_fine     -o logs/gran_fine.log     -v LEVEL=fine \
     training/pbs/granularity_step2_train.pbs

qsub -N gran_semantic -o logs/gran_semantic.log -v LEVEL=semantic \
     training/pbs/granularity_step2_train.pbs

qsub -N gran_decision -o logs/gran_decision.log -v LEVEL=decision \
     training/pbs/granularity_step2_train.pbs
```

If the queue only gives you one GPU at a time, submit them one after another —
the order does not matter.

Every hyperparameter, including the random seed, is identical across the three.
That is not tidiness: it is the entire basis of the claim. **Do not tune one arm
and not the others**, or the measured difference stops being attributable to the
taxonomy.

**Watch for the first few minutes.** Most failures happen in the first thirty
seconds, and there is no point holding a GPU for a day on a job that already
died:

```bash
qstat -u $USER
tail -f logs/gran_decision.log
```

The first lines should read:

```
python : /home/soft/anaconda3/envs/auto_det/bin/python
conda environment: OK
```

If instead you see `python resolved to /bin/python`, the job aborts by design —
the environment did not activate and nothing downstream would have worked.

**Done when each log ends with** `DONE [<level>]`.

**Check:**

```bash
ls runs/granularity/*/weights/best.pt
```

All three must exist before job 3.

---

### Job 3 — the comparison  (GPU, ~2 h)

```bash
qsub training/pbs/granularity_step3_compare.pbs
```

Scores all three models in **one common six-group space**, whatever space each
was trained in, and repeats it at a second confidence threshold so the
conclusion is not an artefact of one operating point.

```bash
tail -f logs/gran_step3.log
```

**This produces the paper's main table.** Read it from:

```bash
cat results/granularity_comparison.json
```

The columns that matter are **SWMC**, **critical-error rate**, **group accuracy**
and **recall** — all measured in the same space, all comparable. Native mAP is
printed too but is **not** comparable across rows, because fewer classes is an
easier problem. The job prints that warning next to the table; keep it in the
paper.

---

### Job 4 — night and low light  (GPU, ~3 h)

```bash
qsub -v LEVEL=decision training/pbs/india_night_eval.pbs
```

Buckets the validation set with the pipeline's own `scene_lighting()` and
reports SWMC per condition, plus the enhancement ablation.

```bash
tail -f logs/india_night.log
cat results/conditions_decision.json
```

Two things to read honestly:

- A bucket flagged **"too few images"** — report the count, not the metric. A
  confident number over 30 frames is noise dressed as a result.
- If enhancement comes out **neutral or negative**, that is the finding, not a
  failure. The pipeline enhances every dark frame today, and enhancement can
  amplify noise into texture a domain-shifted detector reads as an object —
  which is one route to the phantom detections seen on Indian footage.

---

### Job 5 — run the pipeline on real video  (GPU, interactive)

Not a batch job. Grab an interactive GPU node:

```bash
qsub -I -l select=1:ncpus=8:ngpus=1:mem=40gb -q gpu -l walltime=02:00:00
```

Then:

```bash
cd ~/IBM_Internship/autonomous_detection
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate auto_det

python inference/adas_final.py \
    --source demo/india_clip.mp4 \
    --weights runs/granularity/decision/weights/best.pt \
    --save out_annotated.mp4 \
    --log decisions.jsonl
```

(`--source`, not `--video`. It also takes an image path or a webcam index.)

If phantom detections show up on unfamiliar footage, raise the detection
threshold — `--conf 0.45` is the usual first move. And if you have a KITTI
calibration file for the camera, pass `--kitti_calib <calib_cam_to_cam.txt>`:
without it the size-consistency filter guesses the focal length from the frame
width, and a wrong guess makes it reject real objects as phantoms.

**Read the `[collision]` block at the end.** It is what tells you whether
tracking is starving the collision layer:

```
[collision] 18432 object sightings over 3604 frames
  closing-speed coverage : 0.87   (1.00 = TTC computable for every sighting)
  new track ids          : 412 (11.4 per 100 frames)
  histories rescued      : 96 (recovered across an ID switch)
```

- **coverage below 0.60** — the collision layer frequently could not measure, so
  quiet stretches in the video do **not** mean the road was clear.
- **rescued > ~30% of new ids** — the recovery is holding the module up, but the
  tracker itself is unstable. Then, and only then, switch to `botsort.yaml`
  (ReID) or lengthen `track_buffer`. The number decides it, not a hunch.

Also check the `[sanity_filter]` block: a rejection rate above 25% with an
*estimated* focal length means real objects are being thrown away as phantoms —
pass `--kitti_calib <calib.txt>` for the true focal length.

---

## 3. WHEN A JOB FAILS

```bash
qstat -u $USER              # queued / running jobs
qstat -f <jobid>            # why a job is stuck, or its exit status
tail -50 logs/<name>.log    # what actually happened
qdel <jobid>                # kill a job
```

The failures already seen on this cluster, and what they look like:

| symptom in the log | cause | fix |
|---|---|---|
| `python: /bin/python` then import errors | `#PBS -V` exported CONDA_SHLVL, so `conda activate` silently did nothing | already removed from every script — if you see it, a stale script came back |
| `PYTHONPATH: unbound variable` | `set -u` with an unset variable | already removed |
| `bad interpreter: /bin/bash^M` | CRLF line endings from a Windows copy | `sed -i 's/\r$//' training/pbs/*.pbs` |
| job rejected immediately, no log | `logs/` does not exist | `mkdir -p logs` |
| `pip: command not found` | console scripts not on PATH in a batch shell | already handled with `python -m pip` |
| `ModuleNotFoundError: No module named 'models'` | entry point run without the project root on `sys.path` | already fixed in the entry points |

A job that exits within thirty seconds is almost always the environment, not
your data. The scripts check for that explicitly and abort with a clear message
rather than failing confusingly an hour later.

---

## 4. GETTING RESULTS BACK

```powershell
# from Windows PowerShell, in the project folder
scp -r USER@CLUSTER:~/IBM_Internship/autonomous_detection/results ./results_from_hpc
scp USER@CLUSTER:~/IBM_Internship/autonomous_detection/out_annotated.mp4 .
scp -r USER@CLUSTER:~/IBM_Internship/autonomous_detection/runs/granularity ./runs_granularity
```

The weights are large; if you only need the numbers, `results/` and `logs/` are
enough.

---

## QUICK REFERENCE

```bash
# once
mkdir -p logs results
python scripts/verify_adas_pipeline.py          # expect 186/186

# then, one at a time, reading each log before the next
qsub training/pbs/granularity_step1_prepare.pbs
qsub -N gran_fine     -o logs/gran_fine.log     -v LEVEL=fine     training/pbs/granularity_step2_train.pbs
qsub -N gran_semantic -o logs/gran_semantic.log -v LEVEL=semantic training/pbs/granularity_step2_train.pbs
qsub -N gran_decision -o logs/gran_decision.log -v LEVEL=decision training/pbs/granularity_step2_train.pbs
qsub training/pbs/granularity_step3_compare.pbs
qsub -v LEVEL=decision training/pbs/india_night_eval.pbs
```

**Not part of this chain** — `india_step1_prepare.pbs` and
`india_step2_compare.pbs` are the earlier KITTI-versus-India adaptation
experiment, and `yolo11x_merged.pbs`, `clrnet_culane.pbs` and `aux_and_eval.pbs`
are the original training jobs. Run `india_step1_prepare.pbs` only if
`data/UVH26` is not on the cluster yet — job 1 above needs it.
