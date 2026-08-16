# IEEE PAPER — BUILD AND SUBMISSION PACKAGE

**Title:** A Fail-Safe Monocular Perception Pipeline for Collision and
Overtaking Decisions in Unstructured Traffic

**Authors:** Krish Manwani, Vishal Bansal
**Supervisors:** Pranshu Chander, Bhushan Singh Negi *(spelling needs your
confirmation — see `CITATIONS_VERIFIED.md`)*

---

## Files

Two documents are provided. They share the same research content and the same
verified reference list, but serve different purposes.

| File | What it is | Pages |
|---|---|---|
| **`project_report.pdf`** | Full project report — title page, contents, list of figures and tables, abstract, six chapters, references. Plain black on white, no coloured panels. **This is the submission document.** | 22 |
| **`adas_paper.pdf`** | The same work condensed into IEEE conference format, for submission to a venue | 6 |
| `project_report.tex` / `adas_paper.tex` | Sources for both, if you need to edit | — |
| `CITATIONS_VERIFIED.md` | Verification record for all 14 references, plus three items needing your action | — |
| `figures/` | The five result images both documents embed | — |

Both PDFs are already compiled and included — you do not need LaTeX to read or
submit them. The sections below are only needed if you edit the sources.

### What the report contains

```
Title page
Contents · List of Figures · List of Tables
Abstract
1  Introduction          background, problem statement, objectives, scope
2  Literature Review     domain shift, component architectures, synthesis,
                         research gaps
3  System Architecture   the six-stage pipeline diagram, degradation contract
4  Methodology           collision estimation, plausibility filtering,
                         curvature in physical units, overtaking verdict
5  Data Pipeline         preprocessing, label space, training configuration,
                         verification procedure
6  Results               per-class accuracy, training progression, end-to-end
                         pipeline, plausibility filter
7  Limitations           four items, each traced to its cause
8  Conclusion and Future Work
References                14 entries
```

## How to compile

**Overleaf (easiest — nothing to install):**
1. Go to overleaf.com → New Project → Upload Project
2. Upload `adas_paper.tex`
3. Menu → Compiler → **pdfLaTeX**
4. Recompile

The IEEEtran class and TikZ are both preinstalled on Overleaf. The architecture
diagram is drawn in TikZ inside the `.tex`, so there is no external image
dependency for it.

**Locally**, if you have TeX Live or MiKTeX:
```
pdflatex adas_paper.tex
pdflatex adas_paper.tex     # run twice so references resolve
```

---

## ⚠ THE IEEE PAPER CONTAINS PLACEHOLDER NUMBERS — READ THIS FIRST

This applies to **`adas_paper.pdf` / `adas_paper.tex` only**. The project
report (`project_report.pdf`) contains no placeholders and is ready as-is.

Section VI-E of the IEEE paper (*Adaptation to Unstructured Traffic*) and its
Table IV are **written but not yet measured** — every value is `XX.X`, and the
section carries a `TODO(authors)` comment in the source.

**Do not submit the IEEE paper while those placeholders are present.** Either
run the experiment below and fill them in, or delete the subsection, Table IV,
and the sentences in Limitations and Conclusion that reference
`\ref{sec:india}`.

### Running the adaptation experiment

Two cluster jobs, in order:

```bash
cd ~/IBM-INTERNSHIP/autonomous_detection
git pull

# Step 1 — download UVH-26, convert, build both dataset configs (CPU, ~1h)
qsub training/pbs/india_step1_prepare.pbs

# ...wait for it to finish, check the log:
tail -40 ~/logs/india_step1.log

# Step 2 — the controlled comparison (GPU, ~4-8h)
qsub training/pbs/india_step2_compare.pbs
tail -80 ~/logs/india_step2.log
```

**If step 1 fails at the download**, the compute node has no outbound internet.
Run just the download on the login node, then skip it on resubmit:

```bash
pip install huggingface_hub
huggingface-cli download iisc-aim/UVH-26 --repo-type dataset --local-dir data/UVH26
qsub -v SKIP_DOWNLOAD=1 training/pbs/india_step1_prepare.pbs
```

**If step 1 fails at conversion**, the extracted folder layout differs from what
`data/prepare_uvh26.py` expects. The log prints both the expected layout and
what actually landed on disk — compare them and either rename the folders or
adjust the paths in that script's `main()`.

### Why the experiment is designed this way

Both models are scored on **one identical held-out split** of Indian imagery, so
the difference between rows (A) and (B) of Table IV is attributable to training
data and nothing else. Simply reporting "we trained on more data and got X"
would not support the paper's claim; a common test set does.

Row (C) re-scores the adapted model on the *original* validation split. This is
the check for catastrophic forgetting — the paper claims joint training avoids
it, and row (C) is what substantiates that claim rather than asserting it. If
row (C) drops far below 95.4 %, the claim is wrong and the text must change.

### Filling in Table IV

The job prints three labelled blocks — `(A) BASELINE`, `(B) ADAPTED`,
`(C) ADAPTED model on the ORIGINAL test set`. Copy the mAP figures into the
corresponding rows, then:

- Delete the `TODO(authors)` comment block above the subsection.
- Update the Limitations paragraph if you ran more than one corpus.
- If the (A)→(B) gain is small or negative, **say so** rather than dropping the
  table. A negative result honestly reported is defensible; a missing experiment
  that the text still references is not.

---

## Before you submit — four things

1. **Fix the author block.** Four `TODO` fields (institution, city, email) per
   author, and confirm the supervisor names. See `CITATIONS_VERIFIED.md`.

2. **Add the DriveIndia page numbers** from IEEE Xplore — the one bibliographic
   field that could not be verified from open sources.

3. **Add result figures.** The paper currently carries the architecture diagram
   plus three tables. To add the annotated frames and training curves from your
   results bundle, drop them in `figures/` and insert, for example:

   ```latex
   \begin{figure}[t]
     \centering
     \includegraphics[width=\columnwidth]{figures/BRAKE_01021.jpg}
     \caption{A brake-tier alert. The bounding box, track identifier and
              time-to-collision annotation are produced by the pipeline.}
     \label{fig:brake}
   \end{figure}
   ```

   Good candidates, in order of value to a reviewer:
   - a brake-alert frame (shows the collision layer working)
   - the normalised confusion matrix (supports Table I)
   - the training curves (supports Table II)

4. **Read the Limitations section yourself.** It states plainly that four
   declared classes have no training data and that the India-specific classes
   are untrained. This is deliberate — a reviewer will find these facts anyway,
   and stating them with their diagnosed cause is stronger than omitting them.
   But make sure you are comfortable defending each one, because you will be
   asked.

---

## Honest note on scope

The reported accuracy (95.4 % mAP@0.5) is measured on a structured-road
validation split, **not** on unstructured Indian traffic. The paper says this
explicitly in Section VII. The Indian-road adaptation is presented as
implemented-but-not-yet-trained infrastructure, which is what it is. Do not let
the abstract be read as a claim of validated performance on Indian roads — the
current wording avoids that, and it should stay that way.
