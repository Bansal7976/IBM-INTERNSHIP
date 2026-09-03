#!/bin/bash
# Where am I, and what do I run next?
#
# Answers both from what is actually on disk rather than from memory. Run it
# any time on the cluster:
#
#     bash scripts/hpc_status.sh
#
# Every stage prints DONE, PARTIAL or TODO, and the script ends by printing the
# single next command to run. Safe to run at any point -- it only reads.

cd "$(dirname "$0")/.." || exit 1

# Colour only when writing to a terminal. Piping this into a file or a log --
# which is exactly what you do when sending status to someone -- would
# otherwise fill it with raw escape codes.
if [ -t 1 ]; then
    green()  { printf '[32m%s[0m' "$1"; }
    yellow() { printf '[33m%s[0m' "$1"; }
    red()    { printf '[31m%s[0m' "$1"; }
else
    green()  { printf '%s' "$1"; }
    yellow() { printf '%s' "$1"; }
    red()    { printf '%s' "$1"; }
fi

status_line() {
    # $1 = state, $2 = label, $3 = detail
    case "$1" in
        DONE)    printf '  [%s] %-42s %s\n' "$(green ' OK ')" "$2" "$3" ;;
        PARTIAL) printf '  [%s] %-42s %s\n' "$(yellow 'PART')" "$2" "$3" ;;
        *)       printf '  [%s] %-42s %s\n' "$(red 'TODO')" "$2" "$3" ;;
    esac
}

count_images() {
    [ -d "$1" ] && find "$1" -maxdepth 1 -type f \
        \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) 2>/dev/null | wc -l || echo 0
}

echo "============================================================"
echo "  PIPELINE STATUS   $(date)"
echo "  $(pwd)"
echo "============================================================"
echo ""

NEXT=""

# ---------------------------------------------------------------- 0. env
echo "ENVIRONMENT"
if python -c "import ultralytics" 2>/dev/null; then
    status_line DONE "conda env auto_det + ultralytics" "$(command -v python)"
else
    status_line TODO "conda env auto_det + ultralytics" "run: conda activate auto_det"
    NEXT="source /home/soft/anaconda3/etc/profile.d/conda.sh && conda activate auto_det"
fi

if [ -d logs ]; then
    status_line DONE "logs/ directory exists" ""
else
    status_line TODO "logs/ directory exists" "PBS rejects jobs without it"
    [ -z "$NEXT" ] && NEXT="mkdir -p logs results"
fi
echo ""

# --------------------------------------------------------- 1. raw datasets
echo "1. RAW DATA"
RAW_FOUND=0
for d in data/IDD_Detection data/idd data/DATS_2022 data/HeteroTraffic \
         data/IndiaScene365 data/Indistreet2K25 data/UVH26; do
    if [ -d "$d" ]; then
        status_line DONE "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
        RAW_FOUND=1
    fi
done
if [ "$RAW_FOUND" -eq 0 ]; then
    status_line TODO "any Indian dataset downloaded" "IDD: https://idd.insaan.iiit.ac.in/"
    [ -z "$NEXT" ] && NEXT="# Download IDD (registration required), unpack into data/IDD_Detection"
fi
echo ""

# ------------------------------------------------------ 2. converted to YOLO
echo "2. CONVERTED TO YOLO"
CONV_FOUND=0
for d in data/*_yolo; do
    [ -d "$d" ] || continue
    n=$(( $(count_images "$d/train/images") + $(count_images "$d/val/images") ))
    if [ "$n" -gt 0 ]; then
        status_line DONE "$d" "$n images"
        CONV_FOUND=1
    fi
done
if [ "$CONV_FOUND" -eq 0 ]; then
    status_line TODO "converted dataset" "one command handles VOC/COCO/YOLO"
    [ -z "$NEXT" ] && [ "$RAW_FOUND" -eq 1 ] && \
        NEXT="python data/prepare_indian.py --src data/IDD_Detection --out data/idd_yolo --report-only"
fi
echo ""

# ------------------------------------------------------- 3. three label sets
echo "3. THREE LABEL SPACES (job 1)"
GRAN_OK=0
declare -a GRAN_COUNTS
for L in fine semantic decision; do
    d="data/gran_$L"
    if [ -d "$d" ]; then
        boxes=$(find "$d" -name '*.txt' -path '*labels*' -exec cat {} + 2>/dev/null | wc -l)
        status_line DONE "data/gran_$L" "$boxes boxes"
        GRAN_COUNTS+=("$boxes")
        GRAN_OK=$((GRAN_OK + 1))
    else
        status_line TODO "data/gran_$L" ""
    fi
done
if [ "$GRAN_OK" -eq 3 ]; then
    uniq_count=$(printf '%s\n' "${GRAN_COUNTS[@]}" | sort -u | wc -l)
    if [ "$uniq_count" -eq 1 ]; then
        echo "       all three hold identical boxes -- the comparison is valid"
    else
        echo "       WARNING: box counts differ (${GRAN_COUNTS[*]})."
        echo "       The comparison would measure a difference in DATA, not taxonomy."
        echo "       Do not train on these. Re-run job 1 and read its mapping report."
    fi
elif [ "$CONV_FOUND" -eq 1 ] && [ -z "$NEXT" ]; then
    NEXT="qsub training/pbs/granularity_step1_prepare.pbs"
fi
echo ""

# ------------------------------------------------------------ 4. the models
echo "4. TRAINED MODELS (job 2)"
TRAINED=0
for L in fine semantic decision; do
    w="runs/granularity/$L/weights/best.pt"
    if [ -f "$w" ]; then
        status_line DONE "runs/granularity/$L" "$(du -h "$w" 2>/dev/null | cut -f1)"
        TRAINED=$((TRAINED + 1))
    elif [ -d "runs/granularity/$L" ]; then
        status_line PARTIAL "runs/granularity/$L" "started, no best.pt yet"
    else
        status_line TODO "runs/granularity/$L" ""
    fi
done
if [ "$GRAN_OK" -eq 3 ] && [ "$TRAINED" -lt 3 ] && [ -z "$NEXT" ]; then
    for L in fine semantic decision; do
        [ -f "runs/granularity/$L/weights/best.pt" ] && continue
        NEXT="qsub -N gran_$L -o logs/gran_$L.log -v LEVEL=$L training/pbs/granularity_step2_train.pbs"
        break
    done
fi
echo ""

# --------------------------------------------------------------- 5. results
echo "5. RESULTS"
for f in results/granularity_comparison.json:"job 3 -- the main table" \
         results/granularity_comparison_conf010.json:"job 3 -- threshold check" \
         results/conditions_decision.json:"job 4 -- night and low light" \
         results/planning.json:"job 5 -- closed-loop planner"; do
    path="${f%%:*}"; label="${f##*:}"
    if [ -f "$path" ]; then
        status_line DONE "$label" "$path"
    else
        status_line TODO "$label" "$path"
    fi
done
if [ "$TRAINED" -eq 3 ] && [ ! -f results/granularity_comparison.json ] && [ -z "$NEXT" ]; then
    NEXT="qsub training/pbs/granularity_step3_compare.pbs"
elif [ -f results/granularity_comparison.json ] && [ ! -f results/conditions_decision.json ] && [ -z "$NEXT" ]; then
    NEXT="qsub -v LEVEL=decision training/pbs/india_night_eval.pbs"
elif [ ! -f results/planning.json ] && [ -z "$NEXT" ]; then
    NEXT="python evaluation/evaluate_planning.py --seeds 20 --out results/planning.json"
fi
echo ""

# --------------------------------------------------------------- 6. the queue
echo "6. QUEUE"
if command -v qstat >/dev/null 2>&1; then
    running=$(qstat -u "$USER" 2>/dev/null | tail -n +6)
    if [ -n "$running" ]; then
        echo "$running" | sed 's/^/  /'
        echo ""
        echo "  Jobs are still in the queue. Let them finish before submitting more."
        NEXT=""
    else
        status_line DONE "nothing queued or running" ""
    fi
else
    echo "  (qstat not available -- not on the cluster)"
fi
echo ""

echo "============================================================"
if [ -n "$NEXT" ]; then
    echo "  NEXT COMMAND:"
    echo ""
    echo "    $NEXT"
else
    echo "  Nothing to submit right now."
fi
echo ""
echo "  Always run this before submitting anything:"
echo "    python scripts/verify_adas_pipeline.py     # expect 260/260"
echo "============================================================"
