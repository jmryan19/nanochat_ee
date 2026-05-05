"""
Evaluate a BASE (pretrained-only) model on the tasks/ suite using
prompt-completion semantics (no chat special tokens).

Mirror of scripts.chat_eval, with three differences:
  1) Loads via load_model("base", ...) instead of "sft".
  2) Renders prompts via tokenizer.render_for_base_completion (no
     <|user_start|>/<|assistant_start|> wrapping) instead of
     tokenizer.render_for_completion.
  3) Categorical (MC) tasks are scored by summed log-probability over
     each candidate's *full* tokenization, not by reading a single
     letter-position logit -- because for a base BPE tokenizer the
     letters "A"/"B"/etc. may not tokenize as single tokens.

Generative tasks support pass@k via the existing engine.generate_batch
num_samples interface; the per-problem `passed = any(outcomes)` already
gives correct pass@k semantics.

Example runs:
    # zero-shot MC, single-process
    python -m scripts.base_task_eval -a MMLU -k 1 --max-problems 32

    # pass@5 on HumanEval, distributed across 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_task_eval -- \
        -a HumanEval -k 5 -t 0.7 --max-problems 8

    # all tasks, full eval
    torchrun --nproc_per_node=8 -m scripts.base_task_eval -- -k 1
"""

import argparse
from functools import partial
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


# -----------------------------------------------------------------------------
# Generative evaluation: pass@k via num_samples
def run_generative_eval_passk(
    task_object, tokenizer, model, engine,
    k, max_new_tokens, temperature, top_k,
    max_problems=None, separator="\n", few_shot_examples=None,
):
    """
    For each problem, draw k completions from the base model. A problem passes if
    ANY of the k completions evaluates to a passing outcome -- standard pass@k.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        encoded_prompt = tokenizer.render_for_base_completion(
            conversation, separator=separator, few_shot_examples=few_shot_examples,
        )
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=k,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(r[prefix_length:]) for r in results]
        outcomes = [task_object.evaluate(conversation, c) for c in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / max(1, total):.2f}%)",
              end="", flush=True)
    print()

    if ddp:
        np_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        tot_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(np_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_t, op=dist.ReduceOp.SUM)
        num_passed = np_t.item()
        total = tot_t.item()

    print0("=" * 50)
    pct = 100 * num_passed / max(1, total)
    print0(f"pass@{k}: {num_passed}/{total} ({pct:.2f}%)")
    return num_passed / max(1, total)


# -----------------------------------------------------------------------------
# Categorical evaluation: summed log-probability scoring of each candidate's full
# tokenization. Robust to BPE quirks (single-letter tokens, leading-space tokens).
def run_categorical_eval_logprob(
    task_object, tokenizer, model,
    max_problems=None, separator="\n", few_shot_examples=None,
):
    """
    For each problem and each candidate letter L:
      - Build the prompt P with render_for_base_completion (which appends `separator`).
      - Encode just L: cand_ids = tokenizer.encode(L). May be 1+ tokens.
      - Stack [P + cand_ids] for all candidates as a padded batch, run a single
        forward pass, sum log-softmax over the candidate-token positions for
        each candidate.
      - argmax candidate is the predicted letter.

    Why: chat_eval uses a single-letter logit at the answer position, which
    requires "A"/"B"/etc. to be exactly one BPE token. For a base model trained
    with vanilla BPE that may not hold, so we score full continuations instead.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    # Cache letter encodings -- many problems share the same letter set (e.g. A,B,C,D).
    letter_cache = {}
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        letters = conversation["letters"]

        prompt_ids = tokenizer.render_for_base_completion(
            conversation, separator=separator, few_shot_examples=few_shot_examples,
        )
        P = len(prompt_ids)

        cand_ids_list = []
        for L in letters:
            if L not in letter_cache:
                letter_cache[L] = tokenizer.encode(L)
            cand_ids_list.append(letter_cache[L])
        cand_lens = [len(c) for c in cand_ids_list]
        max_full_len = P + max(cand_lens)

        # Pad with bos -- those positions are not used for scoring, just for valid forward.
        batch = torch.full((len(letters), max_full_len), bos, dtype=torch.long, device=device)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        for j, cand_ids in enumerate(cand_ids_list):
            batch[j, :P] = prompt_t
            batch[j, P:P + len(cand_ids)] = torch.tensor(cand_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(batch)  # (num_letters, max_full_len, V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Score: log_probs at position P-1+t predicts the token at position P+t.
        candidate_log_probs = []
        for j, cand_ids in enumerate(cand_ids_list):
            K = len(cand_ids)
            row_lp = sum(log_probs[j, P - 1 + t, cand_ids[t]].item() for t in range(K))
            candidate_log_probs.append(row_lp)

        pred_idx = max(range(len(letters)), key=lambda jj: candidate_log_probs[jj])
        predicted_letter = letters[pred_idx]
        outcome = task_object.evaluate(conversation, predicted_letter)

        num_passed += int(outcome)
        total += 1
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / max(1, total):.2f}%)",
              end="", flush=True)
    print()

    if ddp:
        np_t = torch.tensor([num_passed], dtype=torch.long, device=device)
        tot_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(np_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tot_t, op=dist.ReduceOp.SUM)
        num_passed = np_t.item()
        total = tot_t.item()

    average = num_passed / max(1, total)
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------
# Per-task dispatcher
def run_base_task_eval(
    task_name, model, tokenizer, engine,
    k=1, max_new_tokens=512, temperature=0.0, top_k=50,
    max_problems=None,
    generative_separator="\n", categorical_separator="\n",
    few_shot_examples=None,
):
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()

    if task_object.eval_type == "generative":
        # If user asked for k>1 with t=0, bump temperature to 0.7 so samples actually differ.
        eff_temp = temperature
        if k > 1 and eff_temp == 0.0:
            eff_temp = 0.7
            print0(f"  (auto-set temperature=0.7 for k={k} sampling; passed -t to override)")
        return run_generative_eval_passk(
            task_object, tokenizer, model, engine,
            k=k, max_new_tokens=max_new_tokens, temperature=eff_temp, top_k=top_k,
            max_problems=max_problems, separator=generative_separator,
            few_shot_examples=few_shot_examples,
        )
    elif task_object.eval_type == "categorical":
        return run_categorical_eval_logprob(
            task_object, tokenizer, model,
            max_problems=max_problems, separator=categorical_separator,
            few_shot_examples=few_shot_examples,
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base (pretrained-only) models on tasks/")
    parser.add_argument("-a", "--task-name", type=str, default=None,
                        help="task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument("-k", "--k", type=int, default=1,
                        help="pass@k count for generative tasks (k=1 = greedy if temperature=0).")
    parser.add_argument("-t", "--temperature", type=float, default=0.0,
                        help="sampling temperature. Auto-bumped to 0.7 if k>1 and t=0.")
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50, help="top-k for sampling")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="model tag to load (e.g. d24)")
    parser.add_argument("-s", "--step", type=int, default=None, help="step to load (default = last)")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="max problems to evaluate per task")
    parser.add_argument("--generative-separator", type=str, default="\n",
                        help="separator appended to the prompt before generation (default '\\n')")
    parser.add_argument("--categorical-separator", type=str, default="\n",
                        help="separator appended to the prompt before MC scoring (default '\\n'). "
                             "Try '\\nAnswer: ' if MC accuracy is poor.")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""],
                        help="empty = autodetect")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step,
    )
    engine = Engine(model, tokenizer)

    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    baseline_accuracies = {
        "ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25,
        "GSM8K": 0.0, "HumanEval": 0.0, "SpellingBee": 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    results = {}
    for task_name in task_names:
        print0(f"\n----- {task_name} -----")
        acc = run_base_task_eval(
            task_name, model, tokenizer, engine,
            k=args.k, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, top_k=args.top_k,
            max_problems=args.max_problems,
            generative_separator=args.generative_separator,
            categorical_separator=args.categorical_separator,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy (k={args.k}): {100 * acc:.2f}%")

    # Aggregate BaseCORE if all tasks were evaluated.
    from nanochat.report import get_report
    all_evaluated = all(t in results for t in all_tasks)
    base_core_dict = {}
    if all_evaluated:
        centered_mean = 0.0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        base_core_dict = {"BaseCORE metric": centered_mean / len(results)}

    get_report().log(
        section="Base task evaluation",
        data=[vars(args), results, base_core_dict],
    )

    compute_cleanup()
