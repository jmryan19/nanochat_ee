#!/bin/bash
# Submit a grid of self-distillation experiments as independent SLURM jobs.
# Mirrors heir_distill/submit_chunked_train_eval.sh: edit the arrays at the top,
# run the script once, and N independent chunked-resume jobs are queued.
#
# Each job runs runs/run_chunked_pretrain.slurm, which trains for ~1 hour on a
# single A100, self-requeues until done.flag is written, and saves checkpoints
# under run_artifacts/base_checkpoints/<MODEL_TAG>/.
#
# Usage:
#   bash runs/submit_distill_grid.sh                # actually submit
#   DRY_RUN=1 bash runs/submit_distill_grid.sh      # print sbatch invocations only

set -e
set -o pipefail
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"
mkdir -p run_artifacts/chunked_logs

# === Edit this section to change the grid ====================================
DEPTH="${DEPTH:-24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"

# Initial grid: baseline + 3 distill_layer values (4 runs total).
# distill_layer = -1 means "no distillation" -> baseline run.
# For d24, backout_layer = 12; layer=6 is mid-early, layer=11 is just-before-backout,
# layer=18 is after-backout (mirror branch fires). This is the fastest informative
# ablation: tests (a) early-vs-late, and (b) the backout boundary.
DISTILL_LAYERS=(-1 6 11 18)
DISTILL_WEIGHTS=(0.1)
DISTILL_KL_DIRECTIONS=(forward)
DISTILL_TOP_KS=(-1)              # -1 = full vocab; e.g. 100 for top-100

SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LAST="${KEEP_LAST:-1}"
# =============================================================================

DRY_RUN="${DRY_RUN:-0}"
SLURM_SCRIPT="$REPO/runs/run_chunked_pretrain.slurm"
[ -f "$SLURM_SCRIPT" ] || { echo "ERROR: $SLURM_SCRIPT not found"; exit 1; }

n_submitted=0
for DL in "${DISTILL_LAYERS[@]}"; do
    for DW in "${DISTILL_WEIGHTS[@]}"; do
        for DD in "${DISTILL_KL_DIRECTIONS[@]}"; do
            for DK in "${DISTILL_TOP_KS[@]}"; do
                # Build MODEL_TAG (must match the derivation in run_chunked_pretrain.slurm).
                if [ "$DL" -lt 0 ] 2>/dev/null; then
                    MODEL_TAG="d${DEPTH}_baseline"
                else
                    MODEL_TAG="d${DEPTH}_dL${DL}_w${DW}"
                    [ "$DD" = "reverse" ] && MODEL_TAG="${MODEL_TAG}_rev"
                    [ "$DK" -ge 0 ] 2>/dev/null && MODEL_TAG="${MODEL_TAG}_topk${DK}"
                fi

                LOG="$REPO/run_artifacts/chunked_logs/${MODEL_TAG}_%A.out"
                echo "Submitting: MODEL_TAG=$MODEL_TAG  (DL=$DL DW=$DW DD=$DD DK=$DK)"

                if [ "$DRY_RUN" = "1" ]; then
                    echo "  DRY_RUN: would sbatch --job-name=$MODEL_TAG --output=$LOG $SLURM_SCRIPT"
                    echo "    env: DEPTH=$DEPTH DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE \\"
                    echo "         TARGET_PARAM_DATA_RATIO=$TARGET_PARAM_DATA_RATIO \\"
                    echo "         SAVE_EVERY=$SAVE_EVERY KEEP_LAST=$KEEP_LAST \\"
                    echo "         DISTILL_LAYER=$DL DISTILL_WEIGHT=$DW \\"
                    echo "         DISTILL_KL_DIRECTION=$DD DISTILL_TOP_K=$DK \\"
                    echo "         MODEL_TAG=$MODEL_TAG"
                else
                    DEPTH="$DEPTH" \
                    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
                    TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO" \
                    SAVE_EVERY="$SAVE_EVERY" \
                    KEEP_LAST="$KEEP_LAST" \
                    DISTILL_LAYER="$DL" \
                    DISTILL_WEIGHT="$DW" \
                    DISTILL_KL_DIRECTION="$DD" \
                    DISTILL_TOP_K="$DK" \
                    MODEL_TAG="$MODEL_TAG" \
                    sbatch \
                        --job-name="$MODEL_TAG" \
                        --output="$LOG" \
                        "$SLURM_SCRIPT"
                fi
                n_submitted=$((n_submitted + 1))
            done
        done
    done
done

echo
echo "Done. $n_submitted job(s) ${DRY_RUN:+would have been} submitted."
[ "$DRY_RUN" = "1" ] && echo "Re-run without DRY_RUN=1 to actually submit."
