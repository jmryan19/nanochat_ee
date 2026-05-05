#!/bin/bash

# Pretraining-only variant of runs/speedrun.sh.
# Stages: tokenizer training -> base pretraining -> base evaluation.
# The SFT/chat block from speedrun.sh is intentionally omitted.
# Designed to run on an 8xH100 node; smaller hardware works with reduced --depth and --device-batch-size.
#
# Launch examples:
#   bash runs/pretrain_only.sh
#   screen -L -Logfile runs/pretrain_only.log -S pretrain bash runs/pretrain_only.sh
#   WANDB_RUN=pretrain_only bash runs/pretrain_only.sh
#
# Self-distillation flags (default off; populated in stage 1+):
#   DISTILL_LAYER=4 DISTILL_WEIGHT=0.1 bash runs/pretrain_only.sh

export OMP_NUM_THREADS=1

# Keep all artifacts inside the repo (not ~/.cache) so they're easy to locate and clean up.
# This directory is .gitignore'd.
export NANOCHAT_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_artifacts"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# Set WANDB_RUN=<name> to log to wandb (run `wandb login` first). Default: dummy (no wandb).
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Self-distillation flags (added in stage 1; pass through to base_train).
# Defaults preserve the no-distill behavior.
DISTILL_LAYER="${DISTILL_LAYER:--1}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
DISTILL_ARGS=""
if [ "$DISTILL_LAYER" -ge 0 ] 2>/dev/null && [ "$(echo "$DISTILL_WEIGHT > 0" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
    DISTILL_ARGS="--distill-layer=$DISTILL_LAYER --distill-weight=$DISTILL_WEIGHT"
    echo "Self-distillation enabled: $DISTILL_ARGS"
else
    echo "Self-distillation disabled (DISTILL_LAYER=$DISTILL_LAYER, DISTILL_WEIGHT=$DISTILL_WEIGHT)"
fi

# -----------------------------------------------------------------------------
# Reset the markdown report (writes a header with system info + start timestamp).
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download ~2B chars (8 shards) for tokenizer training.
python -m nanochat.dataset -n 8
# Background-download the rest (170 shards total) while tokenizer trains.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# Train the BPE tokenizer (vocab size 2**15 = 32768).
python -m scripts.tok_train
# Evaluate the tokenizer (compression ratio, etc.).
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model, slightly undertrained (data:params ratio 8 vs. compute-optimal ~10.5).
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN $DISTILL_ARGS
# Base evaluation: CORE metric, BPB on train/val splits, sample generations.
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# Generate the markdown report (concatenates per-stage sections written above).
python -m nanochat.report generate
