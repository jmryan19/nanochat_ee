#!/bin/bash
#SBATCH --account=bbjr-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=run_artifacts/chunked_logs/chunked_%A.out

# Per-experiment chunked pretraining runner.
# Mirrors the proven self-requeue / done-marker / progress-guard pattern from
# heir_distill/run_chunked_train_eval.slurm, adapted for nanochat's base_train.
#
# Inputs (env vars exported by the caller, typically runs/submit_distill_grid.sh):
#   DEPTH, DEVICE_BATCH_SIZE, TARGET_PARAM_DATA_RATIO  -- model + data sizing
#   SAVE_EVERY, KEEP_LAST                              -- checkpoint cadence + retention
#   DISTILL_LAYER, DISTILL_WEIGHT,
#   DISTILL_KL_DIRECTION, DISTILL_TOP_K                -- distillation knobs (default off)
#   MODEL_TAG                                          -- override default tag derivation
#
# Behavior:
#   1. Resolve REPO via SLURM_SUBMIT_DIR / BASH_SOURCE / hardcoded fallback.
#   2. Build MODEL_TAG from distill knobs if not given.
#   3. If $CHECKPOINT_DIR/done.flag exists, exit 0 (training complete).
#   4. Persist a wandb run-id per MODEL_TAG so chained jobs continue the same run.
#   5. Progress guard: read .resubmit_state, abort the chain if no checkpoint
#      progress for MAX_NO_PROGRESS consecutive runs.
#   6. Queue the next chunk (begin in 55 min) BEFORE training, so a crash still
#      re-queues.
#   7. source .venv/bin/activate (symlinked to /u/jryan1/.venvs/nanochat_ee).
#   8. python -m scripts.base_train --resume-from-latest ...

set -e
set -o pipefail
set -u

# --- 1. Resolve REPO ---------------------------------------------------------
if [ -n "${REPO:-}" ]; then
    :
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    REPO="$SLURM_SUBMIT_DIR"
elif [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/../pyproject.toml" ]; then
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
else
    REPO="/work/hdd/bbjr/jryan1/exit_gates/nanochat_ee"
fi
if [ ! -f "$REPO/pyproject.toml" ]; then
    echo "ERROR: REPO=$REPO does not look like the nanochat_ee root (no pyproject.toml)"
    exit 1
fi
cd "$REPO"
mkdir -p run_artifacts/chunked_logs

# --- 2. Defaults + MODEL_TAG derivation --------------------------------------
DEPTH="${DEPTH:-24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LAST="${KEEP_LAST:-1}"
DISTILL_LAYER="${DISTILL_LAYER:--1}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
DISTILL_KL_DIRECTION="${DISTILL_KL_DIRECTION:-forward}"
DISTILL_TOP_K="${DISTILL_TOP_K:--1}"

# Derive MODEL_TAG if caller didn't set one. Distill-on tags encode the knobs
# so each grid combo gets its own checkpoint dir.
if [ -z "${MODEL_TAG:-}" ]; then
    if [ "$DISTILL_LAYER" -lt 0 ] 2>/dev/null; then
        MODEL_TAG="d${DEPTH}_baseline"
    else
        MODEL_TAG="d${DEPTH}_dL${DISTILL_LAYER}_w${DISTILL_WEIGHT}"
        [ "$DISTILL_KL_DIRECTION" = "reverse" ] && MODEL_TAG="${MODEL_TAG}_rev"
        [ "$DISTILL_TOP_K" -ge 0 ] 2>/dev/null && MODEL_TAG="${MODEL_TAG}_topk${DISTILL_TOP_K}"
    fi
fi
export MODEL_TAG

# Pin every artifact to <repo>/run_artifacts (HDD; the venv is on /u/'s SSD).
export NANOCHAT_BASE_DIR="$REPO/run_artifacts"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
DONE_FLAG="$CHECKPOINT_DIR/done.flag"
STATE_FILE="$CHECKPOINT_DIR/.resubmit_state"
WANDB_ID_FILE="$CHECKPOINT_DIR/.wandb_run_id"
mkdir -p "$CHECKPOINT_DIR"

echo "=================================================================="
echo "REPO            = $REPO"
echo "MODEL_TAG       = $MODEL_TAG"
echo "CHECKPOINT_DIR  = $CHECKPOINT_DIR"
echo "DEPTH           = $DEPTH"
echo "DISTILL         = layer=$DISTILL_LAYER weight=$DISTILL_WEIGHT dir=$DISTILL_KL_DIRECTION topk=$DISTILL_TOP_K"
echo "SAVE_EVERY      = $SAVE_EVERY  KEEP_LAST=$KEEP_LAST"
echo "JOB_ID          = ${SLURM_JOB_ID:-unset}  JOB_NAME=${SLURM_JOB_NAME:-unset}"
echo "=================================================================="

# --- 3. Done check (happy path) ----------------------------------------------
if [ -f "$DONE_FLAG" ]; then
    echo "Training complete (done.flag exists) -- not resubmitting."
    rm -f "$STATE_FILE"
    exit 0
fi

# --- 4. WandB run-id persistence ---------------------------------------------
# Mint once per MODEL_TAG; chained jobs reuse it so wandb stitches them into one
# logical run. Tying to the dir's lifetime (not the path string) means renaming
# the dir starts a new wandb run rather than continuing the old one.
if [ ! -f "$WANDB_ID_FILE" ]; then
    printf 'nanochat_%s_%s\n' "$(date +%s)" "$(openssl rand -hex 3)" > "$WANDB_ID_FILE"
fi
export WANDB_RUN_ID="$(cat "$WANDB_ID_FILE")"
export WANDB_RESUME="allow"

# --- 5. Progress guard -------------------------------------------------------
MAX_NO_PROGRESS=1
current_step=0
latest_model="$(ls -v "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | tail -1)"
if [ -n "$latest_model" ]; then
    bn="$(basename "$latest_model")"
    s="${bn#model_}"; s="${s%.pt}"
    current_step=$((10#$s))   # strip leading zeros
fi
last_step=0
fail_streak=0
has_prior_state=false
if [ -f "$STATE_FILE" ]; then
    has_prior_state=true
    read -r last_step fail_streak < "$STATE_FILE" || true
fi
if [ "$has_prior_state" = true ]; then
    if [ "$current_step" -gt "$last_step" ]; then
        fail_streak=0
    else
        fail_streak=$((fail_streak + 1))
    fi
fi
echo "$current_step $fail_streak" > "$STATE_FILE"
echo "Progress: latest checkpoint step=$current_step, no-progress streak=$fail_streak"

if [ "$fail_streak" -ge "$MAX_NO_PROGRESS" ]; then
    echo "ERROR: $fail_streak consecutive runs with no checkpoint progress -- aborting chain."
    echo "After fixing the issue, delete $STATE_FILE to re-arm."
    exit 1
fi

# --- 6. Self-requeue BEFORE training (so a crash still chains) ---------------
SLURM_SCRIPT="$REPO/runs/run_chunked_pretrain.slurm"
echo "Queuing next chunk (begin in 55 min): job-name=${SLURM_JOB_NAME:-$MODEL_TAG}"
sbatch \
    --begin=now+55minutes \
    --export=ALL \
    --job-name="${SLURM_JOB_NAME:-$MODEL_TAG}" \
    --output="$REPO/run_artifacts/chunked_logs/${MODEL_TAG}_%A.out" \
    "$SLURM_SCRIPT"

# --- 7. Activate venv --------------------------------------------------------
if [ ! -e ".venv" ]; then
    echo "ERROR: no .venv (symlink or dir) found at $REPO/.venv"
    exit 1
fi
source .venv/bin/activate
which python
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# --- 8. Build DISTILL_ARGS + run training ------------------------------------
DISTILL_ARGS=""
if [ "$DISTILL_LAYER" -ge 0 ] 2>/dev/null && [ "$(echo "$DISTILL_WEIGHT > 0" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
    DISTILL_ARGS="--distill-layer=$DISTILL_LAYER --distill-weight=$DISTILL_WEIGHT"
    [ "$DISTILL_KL_DIRECTION" != "forward" ] && DISTILL_ARGS="$DISTILL_ARGS --distill-kl-direction=$DISTILL_KL_DIRECTION"
    [ "$DISTILL_TOP_K" -ge 0 ] 2>/dev/null && DISTILL_ARGS="$DISTILL_ARGS --distill-top-k-logits=$DISTILL_TOP_K"
fi
echo "DISTILL_ARGS: ${DISTILL_ARGS:-<empty>}"

# Single-GPU launch (no torchrun needed). compute_init detects no-DDP cleanly.
python -m scripts.base_train \
    --depth="$DEPTH" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --model-tag="$MODEL_TAG" \
    --save-every="$SAVE_EVERY" \
    --keep-last="$KEEP_LAST" \
    --resume-from-latest \
    --run="$WANDB_RUN_ID" \
    $DISTILL_ARGS

# --- 9. Post-run summary -----------------------------------------------------
if [ -f "$DONE_FLAG" ]; then
    echo "done.flag written -- training complete; chained follow-up will exit 0."
else
    echo "Training paused without done.flag; chain will continue in 55 min."
fi
