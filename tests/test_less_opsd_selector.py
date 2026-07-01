"""Unit tests for the pure utilities in less_opsd_selector.

These intentionally avoid loading any model / dataset so they run fast and offline.
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

# Make the repo root importable when running `pytest` from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from less_opsd_selector import (  # noqa: E402
    count_sketch_project_trainable_grads,
    load_selected_indices,
    normalize_feature,
    select_topk_by_score,
)


def _make_model_with_grads(seed: int = 0):
    """Tiny model with deterministic gradients on trainable params."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    return model


# --------------------------------------------------------------------------------------
# normalize_feature
# --------------------------------------------------------------------------------------
def test_normalize_feature_unit_norm():
    z = torch.tensor([3.0, 4.0])
    out = normalize_feature(z)
    assert pytest.approx(out.norm().item(), abs=1e-6) == 1.0
    # direction preserved
    assert torch.allclose(out, torch.tensor([0.6, 0.8]), atol=1e-6)


def test_normalize_feature_zero_safe():
    z = torch.zeros(5)
    out = normalize_feature(z)
    assert torch.isfinite(out).all()
    assert out.norm().item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# CountSketch projection
# --------------------------------------------------------------------------------------
def test_count_sketch_shape_and_dtype():
    model = _make_model_with_grads()
    proj = count_sketch_project_trainable_grads(model, projection_dim=64, seed=1)
    assert proj.shape == (64,)
    assert proj.dtype == torch.float32
    assert proj.device.type == "cpu"


def test_count_sketch_deterministic_same_seed():
    model = _make_model_with_grads()
    a = count_sketch_project_trainable_grads(model, projection_dim=128, seed=123)
    b = count_sketch_project_trainable_grads(model, projection_dim=128, seed=123)
    assert torch.allclose(a, b)


def test_count_sketch_changes_with_seed():
    model = _make_model_with_grads()
    a = count_sketch_project_trainable_grads(model, projection_dim=128, seed=1)
    b = count_sketch_project_trainable_grads(model, projection_dim=128, seed=2)
    # Different hashing seed -> different projection (not identical).
    assert not torch.allclose(a, b)


def test_count_sketch_ignores_non_trainable_and_gradless():
    model = _make_model_with_grads()
    # Freeze the second linear layer: its grads must not contribute.
    second = model[1]
    second.weight.requires_grad_(False)
    second.bias.requires_grad_(False)

    proj_with_frozen_grads = count_sketch_project_trainable_grads(model, projection_dim=64, seed=7)

    # Now also clear the frozen layer's .grad; result should be identical because frozen
    # params are skipped regardless of whether .grad is present.
    second.weight.grad = None
    second.bias.grad = None
    proj_without_frozen_grads = count_sketch_project_trainable_grads(model, projection_dim=64, seed=7)

    assert torch.allclose(proj_with_frozen_grads, proj_without_frozen_grads)


def test_count_sketch_preserves_squared_norm_in_expectation():
    # CountSketch is norm-preserving in expectation; with a large dim the sketch norm
    # should be in a sane ballpark of the true grad norm (sanity, not exact).
    model = _make_model_with_grads()
    true_sq = sum((p.grad.float() ** 2).sum().item() for p in model.parameters() if p.grad is not None)
    proj = count_sketch_project_trainable_grads(model, projection_dim=4096, seed=11)
    assert proj.pow(2).sum().item() == pytest.approx(true_sq, rel=0.5)


# --------------------------------------------------------------------------------------
# load_selected_indices
# --------------------------------------------------------------------------------------
def test_load_selected_indices_list_format(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 5, 10]))
    assert load_selected_indices(str(path)) == [1, 5, 10]


def test_load_selected_indices_dict_format(tmp_path):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps({"selected_indices": [2, 4, 6], "metadata": {"method": "x"}}))
    assert load_selected_indices(str(path)) == [2, 4, 6]


def test_load_selected_indices_bad_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": [1, 2]}))
    with pytest.raises(ValueError):
        load_selected_indices(str(path))


# --------------------------------------------------------------------------------------
# top-k selection
# --------------------------------------------------------------------------------------
def test_select_topk_returns_exactly_k_by_num():
    candidate_indices = [10, 11, 12, 13, 14]
    scores = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.7])
    selected, topk_local, k = select_topk_by_score(
        scores, candidate_indices, selection_num_examples=2
    )
    assert k == 2
    assert len(selected) == 2
    # Highest two scores are at positions 1 (0.9) and 4 (0.7) -> indices 11 and 14.
    assert set(selected) == {11, 14}


def test_select_topk_by_fraction():
    candidate_indices = list(range(100, 110))  # 10 candidates
    scores = torch.arange(10, dtype=torch.float32)
    selected, topk_local, k = select_topk_by_score(
        scores, candidate_indices, selection_fraction=0.3
    )
    assert k == 3
    assert len(selected) == 3
    # Highest scores are the last positions -> indices 109, 108, 107.
    assert set(selected) == {107, 108, 109}


def test_select_topk_clamps_to_at_least_one():
    candidate_indices = [1, 2, 3]
    scores = torch.tensor([0.5, 0.4, 0.3])
    selected, _, k = select_topk_by_score(scores, candidate_indices, selection_fraction=0.0)
    assert k == 1
    assert selected == [1]


def test_select_topk_requires_a_criterion():
    with pytest.raises(ValueError):
        select_topk_by_score(torch.tensor([1.0, 2.0]), [0, 1])
