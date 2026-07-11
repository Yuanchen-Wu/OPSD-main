"""AdamW-aware feature transform tests (offline; no LM, no dataset, no GPU).

Covers spec sections 18.1 (AdamW transform correctness vs a real optimizer step,
weight-decay exclusion), 18.2 (raw vs preconditioned ranking reversal), 18.3 (fixed
preconditioner), and optimizer-state handling errors (spec section 12).
"""

import copy
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from less_opsd_selector import (  # noqa: E402
    OptimizerStateError,
    adamw_fixed_preconditioner,
    build_adamw_state_view_from_optimizer,
    hypothetical_adamw_candidate_update,
    normalize_feature,
)

BETA1, BETA2, EPS, LR = 0.9, 0.999, 1e-8, 1e-2


def _warmed_up_model_and_optimizer(num_warmup_steps=3, weight_decay=0.0, seed=0):
    """Tiny model + real torch.optim.AdamW with initialized moments."""
    torch.manual_seed(seed)
    model = nn.Linear(6, 3, bias=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=weight_decay
    )
    x = torch.randn(4, 6)
    for _ in range(num_warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _candidate_grads(model, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(4, 6)
    model.zero_grad(set_to_none=True)
    loss = (model(x) * torch.randn(4, 3)).sum()
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


# --------------------------------------------------------------------------------------
# 18.1 AdamW candidate-update correctness against a real optimizer step
# --------------------------------------------------------------------------------------
def test_adamw_candidate_update_matches_real_step_wd_zero():
    model, optimizer = _warmed_up_model_and_optimizer(weight_decay=0.0)
    view = build_adamw_state_view_from_optimizer(optimizer, model)
    grads = _candidate_grads(model)

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    predicted = {}
    for name, p in model.named_parameters():
        state = view.state_for(name)
        predicted[name] = hypothetical_adamw_candidate_update(
            grads[name], state["exp_avg"], state["exp_avg_sq"], state["step"],
            view.beta1, view.beta2, view.eps,
        )

    # Real AdamW step on the same gradient.
    for n, p in model.named_parameters():
        p.grad = grads[n].clone()
    optimizer.step()

    for name, p in model.named_parameters():
        actual_displacement = p.detach() - params_before[name]
        # With weight_decay=0 the AdamW displacement is exactly -lr * Gamma.
        assert torch.allclose(
            actual_displacement, -LR * predicted[name], rtol=1e-4, atol=1e-7
        ), f"mismatch for {name}"


def test_adamw_candidate_update_excludes_decoupled_weight_decay():
    wd = 0.1
    model, optimizer = _warmed_up_model_and_optimizer(weight_decay=wd)
    view = build_adamw_state_view_from_optimizer(optimizer, model)
    grads = _candidate_grads(model)

    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    predicted = {}
    for name, _ in model.named_parameters():
        state = view.state_for(name)
        predicted[name] = hypothetical_adamw_candidate_update(
            grads[name], state["exp_avg"], state["exp_avg_sq"], state["step"],
            view.beta1, view.beta2, view.eps,
        )

    for n, p in model.named_parameters():
        p.grad = grads[n].clone()
    optimizer.step()

    for name, p in model.named_parameters():
        actual_displacement = p.detach() - params_before[name]
        # Actual = -lr*wd*theta (decoupled decay) - lr*Gamma. The selector feature is
        # only Gamma: the candidate-independent decay term is intentionally excluded.
        decay_term = -LR * wd * params_before[name]
        assert torch.allclose(
            actual_displacement - decay_term, -LR * predicted[name], rtol=1e-4, atol=1e-7
        ), f"weight-decay decomposition failed for {name}"
        # Sanity: the prediction itself must NOT contain the decay term.
        assert not torch.allclose(actual_displacement, -LR * predicted[name], atol=1e-9)


def test_adamw_candidate_update_does_not_mutate_state():
    model, optimizer = _warmed_up_model_and_optimizer()
    view = build_adamw_state_view_from_optimizer(optimizer, model)
    grads = _candidate_grads(model)

    name = next(iter(grads))
    state = view.state_for(name)
    m_before = state["exp_avg"].clone()
    v_before = state["exp_avg_sq"].clone()
    g_before = grads[name].clone()

    hypothetical_adamw_candidate_update(
        grads[name], state["exp_avg"], state["exp_avg_sq"], state["step"],
        view.beta1, view.beta2, view.eps,
    )

    assert torch.equal(state["exp_avg"], m_before)
    assert torch.equal(state["exp_avg_sq"], v_before)
    assert torch.equal(grads[name], g_before)


# --------------------------------------------------------------------------------------
# 18.3 Fixed preconditioner
# --------------------------------------------------------------------------------------
def test_fixed_preconditioner_formula_shapes_dtype_finite():
    g = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
    v = torch.tensor([[0.04, 0.01], [1.0, 0.25]])
    step, beta2, eps = 10, 0.999, 1e-8

    out = adamw_fixed_preconditioner(g, v, step, beta2, eps)

    v_hat = v / (1 - beta2**step)
    expected = g / (v_hat.sqrt() + eps)
    assert out.shape == g.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, expected, rtol=1e-6)


def test_fixed_preconditioner_is_linear_in_grad():
    g = torch.randn(5)
    v = torch.rand(5) + 0.1
    a = adamw_fixed_preconditioner(2.0 * g, v, 5, 0.999, 1e-8)
    b = adamw_fixed_preconditioner(g, v, 5, 0.999, 1e-8)
    assert torch.allclose(a, 2.0 * b, rtol=1e-6)


def test_fixed_preconditioner_rejects_step_zero():
    with pytest.raises(OptimizerStateError):
        adamw_fixed_preconditioner(torch.randn(3), torch.rand(3), 0, 0.999, 1e-8)


# --------------------------------------------------------------------------------------
# 18.2 Raw versus preconditioned ranking reversal
# --------------------------------------------------------------------------------------
def test_raw_vs_adamw_preconditioner_ranking_reversal():
    """2D toy: raw cosine ranks B above A; Adam preconditioning ranks A above B.

    Target direction is [1, 1]/sqrt(2). B's raw gradient is perfectly aligned; A points
    mostly along dim 1. The checkpoint's second moment is huge in dim 1, so the
    preconditioner suppresses dim 1 — after preconditioning A ends up closer to the
    target direction than B.
    """
    target = normalize_feature(torch.tensor([1.0, 1.0]))
    g_a = torch.tensor([0.1, 2.0])
    g_b = torch.tensor([1.0, 1.0])

    raw_score_a = float(normalize_feature(g_a) @ target)
    raw_score_b = float(normalize_feature(g_b) @ target)
    assert raw_score_b > raw_score_a  # raw ranks B first

    # exp_avg_sq chosen so bias-corrected v_hat = [1, 1e4] -> sqrt(v_hat) = [1, 100].
    step, beta2, eps = 10, 0.999, 1e-8
    v_hat = torch.tensor([1.0, 1.0e4])
    exp_avg_sq = v_hat * (1 - beta2**step)

    pre_a = adamw_fixed_preconditioner(g_a, exp_avg_sq, step, beta2, eps)
    pre_b = adamw_fixed_preconditioner(g_b, exp_avg_sq, step, beta2, eps)
    adamw_score_a = float(normalize_feature(pre_a) @ target)
    adamw_score_b = float(normalize_feature(pre_b) @ target)
    assert adamw_score_a > adamw_score_b  # preconditioning ranks A first


# --------------------------------------------------------------------------------------
# Optimizer-state handling (spec section 12)
# --------------------------------------------------------------------------------------
def test_state_view_metadata_and_coverage():
    model, optimizer = _warmed_up_model_and_optimizer(num_warmup_steps=2)
    view = build_adamw_state_view_from_optimizer(optimizer, model)

    assert view.beta1 == BETA1 and view.beta2 == BETA2 and view.eps == EPS
    assert view.learning_rate == LR
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(view.state_by_name) == names
    for state in view.state_by_name.values():
        assert state["step"] == 2
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg"].device.type == "cpu"

    meta = view.metadata()
    assert meta["optimizer_class"] == "AdamW"
    assert meta["optimizer_steps"] == [2]


def test_state_view_rejects_non_adamw_optimizer():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(OptimizerStateError, match="SGD"):
        build_adamw_state_view_from_optimizer(optimizer, model)


def test_state_view_rejects_missing_state():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)  # no step taken
    with pytest.raises(OptimizerStateError, match="Missing optimizer state"):
        build_adamw_state_view_from_optimizer(optimizer, model)


def test_state_view_rejects_foreign_parameters():
    model, optimizer = _warmed_up_model_and_optimizer()
    other_model = copy.deepcopy(model)
    with pytest.raises(OptimizerStateError, match="does not belong to the model"):
        build_adamw_state_view_from_optimizer(optimizer, other_model)


def test_state_view_missing_param_lookup_fails_clearly():
    model, optimizer = _warmed_up_model_and_optimizer()
    view = build_adamw_state_view_from_optimizer(optimizer, model)
    with pytest.raises(OptimizerStateError, match="No optimizer state found"):
        view.state_for("nonexistent.param")
