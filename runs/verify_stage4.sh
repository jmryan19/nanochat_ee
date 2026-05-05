#!/bin/bash
#
# Stage-4 integration smoke run for the self-distillation + base-task-eval branch.
# Exercises every code path that needed real GPU + tokenizer + dataset to verify.
#
# Designed for ONE GPU and ~10-20 minutes of wall-clock time. Skips long-running
# pieces (full CORE eval, full pretraining) -- those wait until full pretraining.
#
# What runs (in order):
#   1. Tokenizer training on 1 dataset shard (~250M chars, vocab=32768)
#   2. base_train --depth=4 --num-iterations=20 -- BASELINE (no distillation)
#   3. base_train --depth=4 --num-iterations=20 --distill-layer=1 --distill-weight=0.1 -- DISTILL
#   4. base_eval --eval bpb on the DISTILL checkpoint (verifies BPB path with the
#      distillation knobs set in the saved meta)
#   5. base_task_eval -a MMLU --max-problems 32 -k 1 (categorical scoring path)
#   6. base_task_eval -a HumanEval --max-problems 4 -k 2 -t 0.7 (generative pass@k)
#   7. REPL-style assertion on tokenizer.render_for_base_completion (no chat
#      special-token IDs in output)
#   8. Grep report.md for the "Base task evaluation" section
#
# All artifacts go to <repo>/run_artifacts (set via NANOCHAT_BASE_DIR).
#
# ---------------------------------------------------------------------------
# HOW TO RUN
#
# Login node, one-time setup (creates a dedicated nanochat .venv via uv).
# IMPORTANT: the venv MUST live on /u/ (fast SSD) -- not /work/hdd/ (slow HDD).
# torch+cu128 import is ~2.5s from /u/ vs 50+s from /work/hdd/.
#   command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
#   uv venv /u/jryan1/.venvs/nanochat_ee --python 3.10
#   cd /work/hdd/bbjr/jryan1/exit_gates/nanochat_ee
#   ln -s /u/jryan1/.venvs/nanochat_ee .venv
#   VIRTUAL_ENV=/u/jryan1/.venvs/nanochat_ee uv sync --extra gpu --active
#
# Compute node (one GPU is enough). Examples:
#   # interactive on Delta (NCSA), one A100 for 30 min:
#   srun -A bbjr-delta-gpu -p gpuA100x4-interactive --gpus-per-task=1 --time=00:30:00 \
#        --pty bash runs/verify_stage4.sh
#
#   # or batch:
#   sbatch --account=bbjr-delta-gpu --partition=gpuA100x4 --gpus-per-task=1 \
#          --time=00:30:00 --output=runs/verify_stage4.log \
#          runs/verify_stage4.sh
#
# Note: this script does NOT call sbatch on its own; you submit it.
# ---------------------------------------------------------------------------

set -e
set -o pipefail
set -u

# SLURM gotcha: with `sbatch`, the script is copied to /var/spool/slurmd and BASH_SOURCE
# points at that copy -- so we can't derive REPO from BASH_SOURCE under sbatch. Prefer
# SLURM_SUBMIT_DIR (the directory you submitted from); fall back to BASH_SOURCE for
# `srun --pty bash runs/verify_stage4.sh` and to the hardcoded repo path as last resort.
if [ -n "${REPO:-}" ]; then
    :  # explicit override
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/pyproject.toml" ]; then
    REPO="$SLURM_SUBMIT_DIR"
elif [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/../pyproject.toml" ]; then
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
else
    REPO="/work/hdd/bbjr/jryan1/exit_gates/nanochat_ee"
fi
if [ ! -f "$REPO/pyproject.toml" ]; then
    echo "ERROR: REPO=$REPO does not look like the nanochat_ee root (no pyproject.toml)"
    echo "  Set REPO=<absolute path> in the env, or sbatch from the repo directory."
    exit 1
fi
cd "$REPO"
echo "REPO = $REPO"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$REPO/run_artifacts"
mkdir -p "$NANOCHAT_BASE_DIR"
LOGDIR="$NANOCHAT_BASE_DIR/verify_logs"
mkdir -p "$LOGDIR"
echo "NANOCHAT_BASE_DIR = $NANOCHAT_BASE_DIR"
echo "LOGDIR            = $LOGDIR"

# Activate the project venv. Per the user's setup convention, the actual venv lives
# under /u/jryan1/.venvs/nanochat_ee (fast SSD); the repo has .venv as a symlink to it.
# /work/hdd is too slow an HDD for the ~50 CUDA SOs that torch+cu128 has to load.
if [ ! -e ".venv" ]; then
    echo "ERROR: no .venv (symlink or dir) found. Login-node setup:"
    echo "  uv venv /u/jryan1/.venvs/nanochat_ee --python 3.10"
    echo "  ln -s /u/jryan1/.venvs/nanochat_ee \$(pwd)/.venv"
    echo "  VIRTUAL_ENV=/u/jryan1/.venvs/nanochat_ee uv sync --extra gpu --active"
    exit 1
fi
source .venv/bin/activate
echo "python  = $(which python)"
python -c "import rustbpe; print('rustbpe import OK')"
python -c "import kernels; print(f'kernels {kernels.__version__} OK')"
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

banner() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
banner "1) Download 1 dataset shard (+ val shard) for tokenizer training"
# 1 train shard (~250M chars) is enough for the tokenizer; the val shard is
# always downloaded too. Skipped if shards are already present.
python -m nanochat.dataset -n 1 2>&1 | tee "$LOGDIR/01_dataset.log"
ls "$NANOCHAT_BASE_DIR/base_data_climbmix" | head -5

# ---------------------------------------------------------------------------
banner "2) Train tokenizer (BPE vocab=32768)"
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train 2>&1 | tee "$LOGDIR/02_tok_train.log"
else
    echo "Tokenizer already exists at $NANOCHAT_BASE_DIR/tokenizer -- skipping retrain."
fi

# ---------------------------------------------------------------------------
banner "3a) base_train BASELINE: --depth=4 --num-iterations=20 (no distill)"
python -m scripts.base_train \
    --depth=4 --device-batch-size=2 --num-iterations=20 \
    --eval-every=10000 --sample-every=10000 --core-metric-every=10000 \
    --model-tag=d4_baseline \
    2>&1 | tee "$LOGDIR/03a_train_baseline.log"

banner "3b) base_train DISTILL: --depth=4 --num-iterations=20 --distill-layer=1 --distill-weight=0.1"
python -m scripts.base_train \
    --depth=4 --device-batch-size=2 --num-iterations=20 \
    --eval-every=10000 --sample-every=10000 --core-metric-every=10000 \
    --distill-layer=1 --distill-weight=0.1 \
    --model-tag=d4_distill \
    2>&1 | tee "$LOGDIR/03b_train_distill.log"

ls "$NANOCHAT_BASE_DIR/base_checkpoints/"

# ---------------------------------------------------------------------------
banner "4) base_eval --eval bpb on d4_distill (verifies loss_reduction='none' path with distill knobs in meta)"
python -m scripts.base_eval \
    --model-tag=d4_distill --eval=bpb \
    --device-batch-size=2 --split-tokens=4096 \
    2>&1 | tee "$LOGDIR/04_bpb_eval.log"

# ---------------------------------------------------------------------------
banner "5) base_task_eval MMLU --max-problems 32 -k 1 (categorical / log-prob scoring)"
python -m scripts.base_task_eval \
    -g d4_distill -a MMLU --max-problems 32 -k 1 \
    2>&1 | tee "$LOGDIR/05_base_task_eval_mmlu.log"

# ---------------------------------------------------------------------------
banner "6) base_task_eval HumanEval --max-problems 4 -k 2 -t 0.7 (generative pass@k)"
# kept --max-problems small because HumanEval generation is slower than MC scoring.
python -m scripts.base_task_eval \
    -g d4_distill -a HumanEval --max-problems 4 -k 2 -t 0.7 \
    2>&1 | tee "$LOGDIR/06_base_task_eval_humaneval.log"

# ---------------------------------------------------------------------------
banner "7) REPL: render_for_base_completion must NOT contain any chat special-token IDs"
python - <<'PY' 2>&1 | tee "$LOGDIR/07_render_repl.log"
from nanochat.tokenizer import get_tokenizer, SPECIAL_TOKENS
tok = get_tokenizer()
chat_only_ids = set()
for t in SPECIAL_TOKENS:
    if t == "<|bos|>":
        continue
    chat_only_ids.add(tok.encode_special(t))
print("chat-only special-token IDs:", sorted(chat_only_ids))

conv = {"messages": [
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]}
ids = tok.render_for_base_completion(conv)
print(f"len(ids)={len(ids)}, ids[:5]={ids[:5]}, ids[-5:]={ids[-5:]}")
print(f"first id == bos? {ids[0] == tok.get_bos_token_id()}")
leaks = [i for i in ids if i in chat_only_ids]
assert not leaks, f"chat special tokens leaked into base prompt: {leaks}"
print("OK: no chat special tokens in render_for_base_completion output.")

# Also visualize the decoded text so we can eyeball it.
text = tok.decode(ids[1:])
print(f"decoded text after bos: {text!r}")
PY

# ---------------------------------------------------------------------------
banner "8) report.md: confirm 'Base task evaluation' section was written"
REPORT="$NANOCHAT_BASE_DIR/report.md"
if [ -f "$REPORT" ]; then
    echo "Report found at: $REPORT"
    if grep -q "Base task evaluation" "$REPORT"; then
        echo "OK: 'Base task evaluation' section present."
        grep -n "Base task evaluation" "$REPORT" | head -3
    else
        echo "WARN: 'Base task evaluation' section NOT in report.md."
        echo "(could be expected if report.reset wasn't run; check $REPORT manually)"
    fi
else
    echo "WARN: $REPORT not found. (expected if no nanochat.report reset/generate was run)"
fi

# ---------------------------------------------------------------------------
banner "DONE. Logs in: $LOGDIR"
ls -la "$LOGDIR"
echo
echo "Stage-4 integration smoke complete. Review the per-step logs above for:"
echo "  - any tracebacks"
echo "  - 3a vs 3b loss curves (distill should make total loss differ -- they will not match)"
echo "  - BPB number printed in step 4 (should be a finite number)"
echo "  - MMLU/HumanEval accuracy in steps 5-6 (small/random is OK; we just need no crashes)"
echo "  - 'OK' lines in step 7 + 8"
