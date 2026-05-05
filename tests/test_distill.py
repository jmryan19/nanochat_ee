"""
Tests for the self-distillation KL loss in nanochat.gpt.

Covers:
  * direct unit tests of the module-level helper `distill_kl_loss`
    (forward/reverse direction, top-k truncation, label masking, gradient flow)
  * integration tests on GPT.forward (distill_weight=0 disables, end-to-end
    forward+backward with each (direction, top_k) combination)
  * config validation (bad direction, bad top_k)

Example run:
    python -m pytest tests/test_distill.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, distill_kl_loss


# -----------------------------------------------------------------------------
# Helpers


def _random_logits(N=20, V=32, seed=0, sharper_teacher=True):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(N, V, generator=g, requires_grad=True)
    teacher = torch.randn(N, V, generator=g)
    if sharper_teacher:
        teacher = teacher * 2.0
    return student, teacher


def _build_tiny_gpt(distill_layer=1, distill_weight=0.5,
                    distill_kl_direction='forward', distill_top_k_logits=None,
                    n_layer=4, vocab_size=128, sequence_len=32):
    config = GPTConfig(
        sequence_len=sequence_len, vocab_size=vocab_size,
        n_layer=n_layer, n_head=4, n_kv_head=4, n_embd=64,
        window_pattern="L",
        distill_layer=distill_layer,
        distill_weight=distill_weight,
        distill_kl_direction=distill_kl_direction,
        distill_top_k_logits=distill_top_k_logits,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    # init_weights zeros attn.c_proj and mlp.c_proj (so the residual stream is
    # untouched at step 0 by design) and sets lm_head to normal(std=0.001) (so logits
    # are near-uniform). For tests we want a non-trivial residual evolution AND a
    # meaningful logit scale, so each layer's outputs are distinguishable and the
    # aux KL has a measurable non-zero contribution.
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        model.lm_head.weight.mul_(50.0)
        for block in model.transformer.h:
            block.attn.c_proj.weight.copy_(
                0.05 * torch.randn(block.attn.c_proj.weight.shape, generator=g)
            )
            block.mlp.c_proj.weight.copy_(
                0.05 * torch.randn(block.mlp.c_proj.weight.shape, generator=g)
            )
    return model


def _manual_kl(student_logits, teacher_logits, mask, direction, top_k=None,
               topk_source=None):
    """Reference implementation — independent of distill_kl_loss.

    `topk_source`: 'teacher' or 'student' — overrides the default source rule,
    so tests can verify which side the source actually came from.
    """
    s = student_logits[mask]
    t = teacher_logits[mask]
    s_lp = F.log_softmax(s, dim=-1)
    t_lp = F.log_softmax(t, dim=-1)
    if top_k is not None:
        if topk_source is None:
            topk_source = 'teacher' if direction == 'forward' else 'student'
        src = t if topk_source == 'teacher' else s
        _, idx = torch.topk(src, k=top_k, dim=-1)
        s_lp = torch.gather(s_lp, -1, idx)
        t_lp = torch.gather(t_lp, -1, idx)
    if direction == 'forward':
        # KL(teacher || student) = sum_v p_t * (log p_t - log p_s)
        per = torch.exp(t_lp) * (t_lp - s_lp)
    else:
        # KL(student || teacher) = sum_v p_s * (log p_s - log p_t)
        per = torch.exp(s_lp) * (s_lp - t_lp)
    return per.sum() / mask.sum().clamp_min(1)


# -----------------------------------------------------------------------------
# Unit tests on distill_kl_loss helper


def test_default_matches_legacy_forward_kl():
    """direction='forward' + top_k=None must equal the legacy F.kl_div(batchmean) call."""
    torch.manual_seed(0)
    student, teacher = _random_logits(N=24, V=40, seed=0)
    mask = torch.ones(24, dtype=torch.bool)

    new = distill_kl_loss(student, teacher.detach(), mask, direction='forward', top_k=None)

    # Legacy expression copied from the pre-change code.
    legacy = F.kl_div(
        F.log_softmax(student[mask], dim=-1),
        F.log_softmax(teacher.detach()[mask], dim=-1),
        reduction='batchmean', log_target=True,
    )
    assert torch.allclose(new, legacy, atol=1e-6), f"new={new.item()} legacy={legacy.item()}"


def test_forward_kl_math():
    student, teacher = _random_logits(N=10, V=16, seed=1)
    mask = torch.ones(10, dtype=torch.bool)
    got = distill_kl_loss(student, teacher, mask, direction='forward', top_k=None)
    ref = _manual_kl(student, teacher, mask, direction='forward', top_k=None)
    assert torch.allclose(got, ref, atol=1e-6), f"got={got.item()} ref={ref.item()}"


def test_reverse_kl_math():
    student, teacher = _random_logits(N=10, V=16, seed=2)
    mask = torch.ones(10, dtype=torch.bool)
    got = distill_kl_loss(student, teacher, mask, direction='reverse', top_k=None)
    ref = _manual_kl(student, teacher, mask, direction='reverse', top_k=None)
    assert torch.allclose(got, ref, atol=1e-6), f"got={got.item()} ref={ref.item()}"


def test_topk_eq_full_when_k_is_V():
    """top_k=V is the entire vocab; loss must equal the no-topk loss."""
    V = 24
    for direction in ('forward', 'reverse'):
        student, teacher = _random_logits(N=8, V=V, seed=hash(direction) & 0xFFFF)
        mask = torch.ones(8, dtype=torch.bool)
        full = distill_kl_loss(student, teacher, mask, direction=direction, top_k=None)
        topk = distill_kl_loss(student, teacher, mask, direction=direction, top_k=V)
        assert torch.allclose(full, topk, atol=1e-5), \
            f"direction={direction}: full={full.item()} topk(V)={topk.item()}"


def test_topk_source_forward_uses_teacher():
    """Forward KL: indices must come from teacher, not student."""
    V, k = 32, 5
    student, teacher = _random_logits(N=6, V=V, seed=7)
    # Asymmetric setup so swapping topk source materially changes the loss:
    # teacher's high-mass region (first k) carries most of the KL signal.
    student = student.detach()
    teacher = teacher.detach().clone()
    teacher.zero_()
    teacher[:, :k] = torch.linspace(8.0, 4.0, k)  # teacher peaks on first k
    student.zero_()
    student[:, k:2 * k] = torch.linspace(8.0, 4.0, k)  # student peaks on second k chunk
    mask = torch.ones(6, dtype=torch.bool)

    got = distill_kl_loss(student, teacher, mask, direction='forward', top_k=k)
    ref_teacher_idx = _manual_kl(student, teacher, mask, direction='forward',
                                 top_k=k, topk_source='teacher')
    ref_student_idx = _manual_kl(student, teacher, mask, direction='forward',
                                 top_k=k, topk_source='student')

    assert torch.allclose(got, ref_teacher_idx, atol=1e-6), \
        f"forward KL must use teacher topk: got={got.item()} expected={ref_teacher_idx.item()}"
    assert not torch.allclose(ref_teacher_idx, ref_student_idx, atol=1e-3), \
        "test setup is degenerate — teacher-idx and student-idx should differ"


def test_topk_source_reverse_uses_student():
    """Reverse KL: indices must come from student, not teacher."""
    V, k = 32, 5
    student, teacher = _random_logits(N=6, V=V, seed=11)
    student = student.detach()
    teacher = teacher.detach().clone()
    teacher.zero_()
    teacher[:, :k] = torch.linspace(8.0, 4.0, k)
    student.zero_()
    student[:, k:2 * k] = torch.linspace(8.0, 4.0, k)
    mask = torch.ones(6, dtype=torch.bool)

    got = distill_kl_loss(student, teacher, mask, direction='reverse', top_k=k)
    ref_student_idx = _manual_kl(student, teacher, mask, direction='reverse',
                                 top_k=k, topk_source='student')
    ref_teacher_idx = _manual_kl(student, teacher, mask, direction='reverse',
                                 top_k=k, topk_source='teacher')

    assert torch.allclose(got, ref_student_idx, atol=1e-6), \
        f"reverse KL must use student topk: got={got.item()} expected={ref_student_idx.item()}"
    assert not torch.allclose(ref_student_idx, ref_teacher_idx, atol=1e-3), \
        "test setup is degenerate — teacher-idx and student-idx should differ"


def test_topk_full_softmax_keeps_nonzero_grad_off_topk():
    """
    Confirms the full-softmax-then-gather design: gradients on non-topk vocab
    positions of the student must be NON-ZERO (they flow through log_softmax's
    log-sum-exp denominator). This is the key property that distinguishes our
    approach from "renormalize over topk" which would zero them out.
    """
    V, k, N = 50, 5, 12
    for direction in ('forward', 'reverse'):
        student_data, teacher_data = _random_logits(N=N, V=V, seed=42)
        student = student_data.detach().clone().requires_grad_(True)
        teacher = teacher_data.detach()

        # Compute topk indices the same way the helper does, to identify which
        # student positions are "in the topk subset".
        src = teacher if direction == 'forward' else student.detach()
        _, idx = torch.topk(src, k=k, dim=-1)
        topk_mask = torch.zeros_like(student, dtype=torch.bool)
        topk_mask.scatter_(-1, idx, True)

        mask = torch.ones(N, dtype=torch.bool)
        loss = distill_kl_loss(student, teacher, mask, direction=direction, top_k=k)
        loss.backward()

        non_topk_grad = student.grad[~topk_mask]
        topk_grad = student.grad[topk_mask]
        assert non_topk_grad.abs().max().item() > 0.0, \
            f"direction={direction}: non-topk grads should be non-zero (full-softmax design)"
        assert topk_grad.abs().max().item() > 0.0, \
            f"direction={direction}: topk grads should be non-zero"


def test_label_masking():
    """Masking out positions must give the same answer as filtering them out by hand."""
    student, teacher = _random_logits(N=10, V=20, seed=3)
    mask = torch.tensor([True, False, True, True, False, True, True, True, False, True])
    got = distill_kl_loss(student, teacher, mask, direction='forward', top_k=None)
    # Hand-filtered reference.
    student_kept = student[mask]
    teacher_kept = teacher[mask]
    ref_mask = torch.ones(student_kept.shape[0], dtype=torch.bool)
    ref = distill_kl_loss(student_kept, teacher_kept, ref_mask, direction='forward', top_k=None)
    assert torch.allclose(got, ref, atol=1e-6), f"got={got.item()} ref={ref.item()}"


# -----------------------------------------------------------------------------
# Integration tests on GPT.forward


def test_distill_weight_zero_disables():
    """With distill_weight=0, the aux loss is gated off — total loss must equal CE alone."""
    torch.manual_seed(0)
    model = _build_tiny_gpt(distill_layer=1, distill_weight=0.0)
    model.eval()
    B, T = 2, 8
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    targets = torch.randint(0, model.config.vocab_size, (B, T))

    with torch.no_grad():
        logits = model(idx)  # (B, T, V)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction='mean',
        )
        loss = model(idx, targets)
    assert torch.allclose(loss, ce, atol=1e-5), f"loss={loss.item()} ce={ce.item()}"


@pytest.mark.parametrize("direction,top_k", [
    ('forward', None),
    ('forward', 16),
    ('reverse', None),
    ('reverse', 16),
])
def test_end_to_end_forward_reverse_topk(direction, top_k):
    """End-to-end: tiny GPT with each (direction, top_k) — finite loss + aux contributes."""
    torch.manual_seed(0)
    B, T = 2, 8
    idx = torch.randint(0, 128, (B, T))
    targets = torch.randint(0, 128, (B, T))
    targets[0, 0] = -1
    targets[1, -1] = -1

    # Reference: same model with distill_weight=0 (aux disabled).
    torch.manual_seed(0)
    model_ref = _build_tiny_gpt(
        distill_layer=1, distill_weight=0.0,
        distill_kl_direction=direction, distill_top_k_logits=top_k,
    )
    with torch.no_grad():
        ref_loss = model_ref(idx, targets)

    # Same init, with aux loss enabled.
    torch.manual_seed(0)
    model = _build_tiny_gpt(
        distill_layer=1, distill_weight=0.5,
        distill_kl_direction=direction, distill_top_k_logits=top_k,
    )
    loss = model(idx, targets)
    assert loss.dim() == 0, "loss should be a scalar"
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    # The aux path must actually contribute — otherwise the wiring is broken.
    assert not torch.allclose(loss, ref_loss, atol=1e-6), \
        f"aux loss not contributing: with-aux={loss.item()} no-aux={ref_loss.item()}"

    loss.backward()
    # c_proj receives gradient via the residual stream even at zero-init (q/k/v don't,
    # because the init scheme zeros c_proj, blocking grad to q/k/v on the first step).
    early_grad = model.transformer.h[1].attn.c_proj.weight.grad
    assert early_grad is not None, "early-layer attn.c_proj must have a gradient"
    assert torch.isfinite(early_grad).all(), "early-layer grads must be finite"
    assert early_grad.abs().sum().item() > 0.0, "early-layer grads must be non-zero"


# -----------------------------------------------------------------------------
# Config validation


def test_invalid_direction_raises():
    config = GPTConfig(
        sequence_len=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        window_pattern="L",
        distill_kl_direction='sideways',
    )
    with pytest.raises(AssertionError, match="distill_kl_direction"):
        with torch.device("meta"):
            GPT(config)


def test_invalid_top_k_raises():
    base_kwargs = dict(
        sequence_len=32, vocab_size=128, n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
        window_pattern="L",
    )
    for bad in (0, 129):
        config = GPTConfig(**base_kwargs, distill_top_k_logits=bad)
        with pytest.raises(AssertionError, match="distill_top_k_logits"):
            with torch.device("meta"):
                GPT(config)
