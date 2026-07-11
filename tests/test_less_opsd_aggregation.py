"""Rollout aggregation, consistency, checkpoint weighting/aggregation, projection,
configuration validation, and resume-fingerprint tests (all offline).

Covers spec sections 18.4-18.9.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from less_opsd_selector import (  # noqa: E402
    FeatureStore,
    LESSOPSDSelectionConfig,
    ResumeMismatchError,
    SelectionConfigError,
    aggregate_checkpoint_scores,
    aggregate_rollout_projections,
    config_fingerprint,
    count_sketch_project_named_tensors,
    count_sketch_project_trainable_grads,
    derive_rollout_seed,
    iter_transformed_grad_tensors,
    normalize_checkpoint_weights,
    normalize_feature,
    rollout_consistency,
    validate_selection_config,
)


# --------------------------------------------------------------------------------------
# 18.4 Multiple rollout aggregation: average BEFORE normalize
# --------------------------------------------------------------------------------------
def test_rollouts_averaged_before_normalization():
    # Two rollouts with very different magnitudes: the big one must dominate.
    z1 = torch.tensor([10.0, 0.0])
    z2 = torch.tensor([0.0, 0.1])

    u, rho, invalid = aggregate_rollout_projections([z1, z2])
    assert not invalid
    assert pytest.approx(u.norm().item(), abs=1e-5) == 1.0

    correct = normalize_feature((z1 + z2) / 2)  # average of unnormalized vectors
    wrong = normalize_feature((normalize_feature(z1) + normalize_feature(z2)) / 2)

    assert torch.allclose(u, correct, atol=1e-6)
    assert not torch.allclose(u, wrong, atol=1e-3)  # provably a different estimator


def test_single_rollout_aggregation_is_just_normalization():
    z = torch.tensor([3.0, 4.0])
    u, rho, invalid = aggregate_rollout_projections([z])
    assert torch.allclose(u, normalize_feature(z), atol=1e-6)
    assert rho == pytest.approx(1.0, abs=1e-5)
    assert not invalid


def test_aggregation_marks_zero_norm_invalid():
    u, rho, invalid = aggregate_rollout_projections([torch.zeros(4), torch.zeros(4)])
    assert invalid
    assert torch.equal(u, torch.zeros(4))


def test_aggregation_drops_non_finite_rollouts():
    good = torch.tensor([1.0, 2.0])
    bad = torch.tensor([float("nan"), 1.0])
    u, rho, invalid = aggregate_rollout_projections([good, bad])
    assert not invalid
    assert torch.allclose(u, normalize_feature(good), atol=1e-6)

    u2, _, invalid2 = aggregate_rollout_projections([bad, bad.clone()])
    assert invalid2


# --------------------------------------------------------------------------------------
# 18.5 Rollout consistency
# --------------------------------------------------------------------------------------
def test_consistency_near_one_for_identical_directions():
    z = torch.tensor([1.0, 2.0, 3.0])
    rho = rollout_consistency([z, 2.0 * z, 0.5 * z])
    assert rho == pytest.approx(1.0, abs=1e-5)


def test_consistency_near_zero_for_opposing_directions():
    z = torch.tensor([1.0, -2.0, 0.5])
    rho = rollout_consistency([z, -z])
    assert rho == pytest.approx(0.0, abs=1e-5)


def test_consistency_intermediate_for_orthogonal_directions():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    rho = rollout_consistency([a, b])
    # mean of two orthogonal unit vectors has norm 1/sqrt(2).
    assert rho == pytest.approx(2**-0.5, abs=1e-5)


# --------------------------------------------------------------------------------------
# 18.6 Multi-checkpoint aggregation
# --------------------------------------------------------------------------------------
def test_checkpoint_weight_normalization_uniform_and_explicit():
    assert normalize_checkpoint_weights("uniform", 4) == [0.25] * 4
    w = normalize_checkpoint_weights("explicit", 3, [0.2, 0.3, 0.5])
    assert w == pytest.approx([0.2, 0.3, 0.5])
    w2 = normalize_checkpoint_weights("explicit", 2, [2.0, 6.0])  # normalized to sum 1
    assert w2 == pytest.approx([0.25, 0.75])


def test_checkpoint_weight_learning_rate():
    w = normalize_checkpoint_weights("learning_rate", 2, learning_rates=[1e-4, 3e-4])
    assert w == pytest.approx([0.25, 0.75])


def test_checkpoint_weight_learning_rate_fails_without_lrs():
    with pytest.raises(SelectionConfigError):
        normalize_checkpoint_weights("learning_rate", 2, learning_rates=None)
    with pytest.raises(SelectionConfigError):
        normalize_checkpoint_weights("learning_rate", 2, learning_rates=[1e-4, 0.0])


def test_checkpoint_weight_explicit_mismatch_fails():
    with pytest.raises(SelectionConfigError):
        normalize_checkpoint_weights("explicit", 3, [0.5, 0.5])


def test_aggregate_checkpoint_scores_weighted_sum():
    scores = torch.tensor([[0.37, 0.10], [0.45, 0.20]])  # [K=2, N=2]
    final, valid = aggregate_checkpoint_scores(scores, [0.25, 0.75])
    assert valid.all()
    assert final[0].item() == pytest.approx(0.25 * 0.37 + 0.75 * 0.45, abs=1e-6)
    assert final[1].item() == pytest.approx(0.25 * 0.10 + 0.75 * 0.20, abs=1e-6)


def test_aggregate_checkpoint_scores_invalid_candidates_get_neg_inf():
    scores = torch.tensor([[0.5, float("nan")], [0.5, 0.9]])
    final, valid = aggregate_checkpoint_scores(scores, [0.5, 0.5])
    assert valid.tolist() == [True, False]
    assert final[1].item() == float("-inf")


# --------------------------------------------------------------------------------------
# 18.7 Projection (named tensors)
# --------------------------------------------------------------------------------------
def _model_with_grads(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    return model


def test_named_tensor_projection_matches_raw_grad_wrapper():
    model = _model_with_grads()
    named = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    a = count_sketch_project_named_tensors(named, projection_dim=128, seed=9)
    b = count_sketch_project_trainable_grads(model, projection_dim=128, seed=9)
    assert torch.allclose(a, b)


def test_named_tensor_projection_deterministic_for_transformed_tensors():
    model = _model_with_grads()
    transformed = [
        (n, p.grad / (p.grad.abs().sqrt() + 1e-8)) for n, p in model.named_parameters()
    ]
    a = count_sketch_project_named_tensors(transformed, projection_dim=128, seed=3)
    b = count_sketch_project_named_tensors(
        [(n, t.clone()) for n, t in transformed], projection_dim=128, seed=3
    )
    assert torch.allclose(a, b)
    c = count_sketch_project_named_tensors(transformed, projection_dim=128, seed=4)
    assert not torch.allclose(a, c)


def test_iter_transformed_raw_skips_frozen_and_gradless():
    model = _model_with_grads()
    model[1].weight.requires_grad_(False)  # frozen but has .grad
    model[0].bias.grad = None  # trainable but no grad
    names = [n for n, _ in iter_transformed_grad_tensors(model, "raw_gradient", None)]
    assert "1.weight" not in names
    assert "0.bias" not in names
    assert "0.weight" in names


def test_iter_transformed_adamw_requires_state_view():
    model = _model_with_grads()
    with pytest.raises(Exception, match="optimizer state"):
        list(iter_transformed_grad_tensors(model, "adamw_candidate_update", None))


# --------------------------------------------------------------------------------------
# Seed derivation (spec section 14)
# --------------------------------------------------------------------------------------
def test_rollout_seed_derivation_stable_and_distinct():
    s = derive_rollout_seed(7, "checkpoint-10", 123, 0, "candidate")
    assert s == derive_rollout_seed(7, "checkpoint-10", 123, 0, "candidate")
    assert s != derive_rollout_seed(7, "checkpoint-10", 123, 1, "candidate")
    assert s != derive_rollout_seed(7, "checkpoint-20", 123, 0, "candidate")
    assert s != derive_rollout_seed(7, "checkpoint-10", 124, 0, "candidate")
    assert s != derive_rollout_seed(7, "checkpoint-10", 123, 0, "target")
    assert s != derive_rollout_seed(8, "checkpoint-10", 123, 0, "candidate")
    assert 0 <= s < 2**63 - 1


# --------------------------------------------------------------------------------------
# 18.8 Configuration validation
# --------------------------------------------------------------------------------------
def _valid_config(**overrides):
    base = dict(
        feature_type="raw_gradient",
        target_objective="opsd",
        num_candidate_rollouts=1,
        num_target_rollouts=1,
        selection_fraction=0.05,
        selection_num_examples=None,
        output_dir="outputs/test",
    )
    base.update(overrides)
    return LESSOPSDSelectionConfig(**base)


def test_valid_default_config_passes():
    validate_selection_config(_valid_config())


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(feature_type="nope"), "feature_type"),
        (dict(target_objective="nope"), "target_objective"),
        (dict(num_candidate_rollouts=0), "num_candidate_rollouts"),
        (dict(num_target_rollouts=-1), "num_target_rollouts"),
        (dict(selection_fraction=None, selection_num_examples=None), "Exactly one"),
        (dict(selection_fraction=0.5, selection_num_examples=10), "Exactly one"),
        (dict(selection_fraction=1.5), "selection_fraction"),
        (dict(selection_fraction=None, selection_num_examples=0), "selection_num_examples"),
        (dict(checkpoint_weighting="nope"), "checkpoint_weighting"),
        (dict(checkpoint_weighting="explicit", checkpoint_weights=None), "explicit"),
        (
            dict(
                checkpoint_paths=["a", "b"],
                checkpoint_weighting="explicit",
                checkpoint_weights=[1.0],
            ),
            "must match",
        ),
        (dict(feature_type="adamw_candidate_update"), "optimizer state"),
        (dict(feature_type="adamw_fixed_preconditioner"), "optimizer state"),
        (dict(save_every=0), "save_every"),
    ],
)
def test_invalid_configs_rejected(overrides, match):
    with pytest.raises(SelectionConfigError, match=match):
        validate_selection_config(_valid_config(**overrides))


def test_adamw_feature_type_with_checkpoint_passes():
    validate_selection_config(
        _valid_config(feature_type="adamw_candidate_update", checkpoint_paths=["ckpt-1"])
    )


def test_legacy_seed_defaults_propagate():
    cfg = _valid_config()
    assert cfg.projection_seed == cfg.seed
    assert cfg.rollout_seed == cfg.seed + 1


# --------------------------------------------------------------------------------------
# 18.9 Resume fingerprint + feature store
# --------------------------------------------------------------------------------------
def test_fingerprint_changes_with_each_critical_field():
    base = _valid_config()
    fp = config_fingerprint(base)
    assert fp == config_fingerprint(_valid_config())  # deterministic

    variants = [
        _valid_config(projection_seed=999),
        _valid_config(projection_dim=8192),
        _valid_config(rollout_seed=999),
        _valid_config(num_candidate_rollouts=2),
        _valid_config(num_target_rollouts=2),
        _valid_config(target_objective="reference_ce"),
        _valid_config(checkpoint_paths=["ckpt-1"]),
        _valid_config(
            feature_type="adamw_candidate_update", checkpoint_paths=["ckpt-1"]
        ),
    ]
    for variant in variants:
        assert config_fingerprint(variant) != fp

    # extra metadata (e.g. optimizer identity, index digests) also changes it.
    assert config_fingerprint(base, {"model": "a"}) != config_fingerprint(base, {"model": "b"})


def test_feature_store_resume_roundtrip(tmp_path):
    store = FeatureStore(str(tmp_path), fingerprint="abc123", resume=False)
    feats = {
        0: {"u": torch.randn(8), "rho": 0.9, "invalid": False},
        3: {"u": torch.randn(8), "rho": 0.7, "invalid": False},
    }
    store.save_shard("candidate", "checkpoint-10", feats)

    # Compatible resume: completed indices and tensors round-trip.
    store2 = FeatureStore(str(tmp_path), fingerprint="abc123", resume=True)
    assert store2.completed_indices("candidate", "checkpoint-10") == {0, 3}
    loaded = store2.load_shard("candidate", "checkpoint-10")
    assert torch.allclose(loaded[0]["u"], feats[0]["u"])
    assert loaded[3]["rho"] == 0.7


def test_feature_store_rejects_mismatched_fingerprint(tmp_path):
    FeatureStore(str(tmp_path), fingerprint="abc123", resume=False)
    with pytest.raises(ResumeMismatchError, match="fingerprint"):
        FeatureStore(str(tmp_path), fingerprint="different", resume=True)


def test_feature_store_fresh_run_clears_stale_artifacts(tmp_path):
    store = FeatureStore(str(tmp_path), fingerprint="abc123", resume=False)
    store.save_shard("candidate", "ckpt", {0: {"u": torch.zeros(2), "rho": 1.0, "invalid": False}})

    # resume=False with a different fingerprint: no error, stale artifacts removed.
    store2 = FeatureStore(str(tmp_path), fingerprint="other", resume=False)
    assert store2.completed_indices("candidate", "ckpt") == set()
