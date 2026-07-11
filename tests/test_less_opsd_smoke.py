"""Integration smoke test for the v2 selection pipeline (spec sections 18.10 and 19).

Uses a tiny PyTorch model and a mock ``DistillationGradientBackend`` — no GPU, no
network, no Hugging Face model or dataset. Exercises candidate feature extraction,
multiple rollouts, optimizer transformation, projection, target features, scoring,
top-k selection, artifact saving, and resume behavior end-to-end.
"""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from less_opsd_selector import (  # noqa: E402
    LESSOPSDSelectionConfig,
    build_adamw_state_view_from_optimizer,
    count_sketch_project_trainable_grads,
    normalize_feature,
    run_multicheckpoint_selection,
)


class TinyBackend:
    """Mock self-distillation backend over a tiny linear model.

    Rollout stochasticity is simulated by perturbing the input with noise drawn from the
    per-rollout generation seed, mimicking on-policy sampling variability.
    """

    def __init__(self, model, rollout_noise: float = 0.05, deterministic: bool = False):
        self.model = model
        self.rollout_noise = rollout_noise
        self.deterministic = deterministic
        self.prepare_calls = 0

    @property
    def student_model(self):
        return self.model

    def _make_input(self, raw_example, generation_seed):
        self.prepare_calls += 1
        x = torch.tensor(raw_example["x"], dtype=torch.float32)
        if not self.deterministic:
            gen = torch.Generator().manual_seed(int(generation_seed) % (2**31))
            x = x + self.rollout_noise * torch.randn(x.shape, generator=gen)
        return x

    def prepare_candidate_example(self, raw_example, generation_seed):
        return self._make_input(raw_example, generation_seed)

    def compute_candidate_loss(self, prepared_example):
        return (self.model(prepared_example) ** 2).sum()

    def prepare_target_example(self, raw_example, objective, generation_seed):
        return self._make_input(raw_example, generation_seed)

    def compute_target_loss(self, prepared_example, objective):
        if objective == "reference_ce":
            return (self.model(prepared_example) - 1.0).pow(2).sum()
        return (self.model(prepared_example) ** 2).sum()

    def target_requires_rollout(self, objective):
        return objective == "opsd"

    def metadata(self):
        return {"backend": "tiny_mock", "teacher_mode": "mock"}


def _make_dataset(n, dim=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [{"x": torch.randn(dim, generator=gen).tolist()} for _ in range(n)]


def _make_model(seed=0, dim=6):
    torch.manual_seed(seed)
    return nn.Linear(dim, 3)


def _run(config, checkpoints=None, backend=None, weights=None, dataset=None):
    dataset = dataset if dataset is not None else _make_dataset(12)
    model = _make_model()
    backend = backend or TinyBackend(model)
    checkpoints = checkpoints or [("base", None)]
    weights = weights or [1.0 / len(checkpoints)] * len(checkpoints)

    def loader(ckpt_id, ckpt_path):
        return backend, None

    return run_multicheckpoint_selection(
        backend_loader=loader,
        checkpoints=checkpoints,
        checkpoint_weights=weights,
        candidate_dataset=dataset,
        candidate_indices=list(range(8)),
        target_dataset=dataset,
        target_indices=[8, 9, 10, 11],
        config=config,
    ), backend


def _config(tmp_path, **overrides):
    base = dict(
        feature_type="raw_gradient",
        target_objective="opsd",
        num_candidate_rollouts=2,
        num_target_rollouts=2,
        projection_dim=64,
        seed=7,
        selection_fraction=None,
        selection_num_examples=3,
        save_every=2,
        output_dir=str(tmp_path / "sel"),
    )
    base.update(overrides)
    return LESSOPSDSelectionConfig(**base)


# --------------------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------------------
def test_smoke_end_to_end_artifacts(tmp_path):
    config = _config(tmp_path)
    result, backend = _run(config)

    assert len(result["selected_indices"]) == 3
    assert result["valid_mask"].all()
    # 8 candidates * 2 rollouts + 4 targets * 2 rollouts = 24 prepared examples.
    assert backend.prepare_calls == 24

    # selected_indices.json is loadable and carries v2 metadata.
    sel_path = result["paths"]["selected_indices"]
    with open(sel_path) as f:
        payload = json.load(f)
    assert payload["selected_indices"] == [int(i) for i in result["selected_indices"]]
    meta = payload["metadata"]
    assert meta["method"] == "less_opsd_raw_static"
    assert meta["method_version"] == "2.0"
    assert meta["feature_type"] == "raw_gradient"
    assert meta["num_candidate_rollouts"] == 2
    assert meta["config_fingerprint"] == result["fingerprint"]
    assert meta["backend"]["backend"] == "tiny_mock"

    # scores.jsonl has per-candidate checkpoint scores + rollout consistency.
    with open(result["paths"]["scores"]) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 8
    for row in rows:
        assert len(row["checkpoint_scores"]) == 1
        assert len(row["rollout_consistency"]) == 1
        assert 0.0 <= row["rollout_consistency"][0] <= 1.0 + 1e-6
    assert sum(r["selected"] for r in rows) == 3
    assert [r["rank"] for r in rows] == list(range(1, 9))

    # target feature saved and normalized.
    tf = torch.load(result["paths"]["target_feature"], weights_only=False)
    assert tf.shape == (64,)
    assert tf.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_smoke_deterministic_across_runs(tmp_path):
    r1, _ = _run(_config(tmp_path / "a"))
    r2, _ = _run(_config(tmp_path / "b"))
    assert torch.allclose(r1["final_scores"], r2["final_scores"], atol=1e-6)
    assert r1["selected_indices"] == r2["selected_indices"]


# --------------------------------------------------------------------------------------
# Resume behavior
# --------------------------------------------------------------------------------------
def test_smoke_resume_skips_completed_examples(tmp_path):
    config = _config(tmp_path)
    r1, backend1 = _run(config)
    calls_first = backend1.prepare_calls

    # Second run with resume=True: everything already complete, so zero extraction calls.
    config2 = _config(tmp_path, resume=True)
    r2, backend2 = _run(config2)
    assert backend2.prepare_calls == 0
    assert torch.allclose(r1["final_scores"], r2["final_scores"], atol=1e-6)
    assert calls_first > 0


def test_smoke_resume_rejects_incompatible_config(tmp_path):
    from less_opsd_selector import ResumeMismatchError

    _run(_config(tmp_path))
    incompatible = _config(tmp_path, resume=True, projection_dim=128)
    with pytest.raises(ResumeMismatchError):
        _run(incompatible)


# --------------------------------------------------------------------------------------
# Multi-checkpoint + AdamW path
# --------------------------------------------------------------------------------------
def test_smoke_multicheckpoint_scores_aggregate(tmp_path):
    dataset = _make_dataset(12)
    config = _config(
        tmp_path,
        num_candidate_rollouts=1,
        num_target_rollouts=1,
        checkpoint_paths=["ckpt-a", "ckpt-b"],
    )

    model = _make_model()
    backend = TinyBackend(model, deterministic=True)

    def loader(ckpt_id, ckpt_path):
        # Different weights per checkpoint -> different per-checkpoint scores.
        torch.manual_seed(0 if ckpt_id == "ckpt-a" else 1)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p))
        return backend, None

    result = run_multicheckpoint_selection(
        backend_loader=loader,
        checkpoints=[("ckpt-a", "ckpt-a"), ("ckpt-b", "ckpt-b")],
        checkpoint_weights=[0.25, 0.75],
        candidate_dataset=dataset,
        candidate_indices=list(range(8)),
        target_dataset=dataset,
        target_indices=[8, 9, 10, 11],
        config=config,
    )
    per_ckpt = result["per_checkpoint_scores"]
    assert per_ckpt.shape == (2, 8)
    # Different checkpoints must yield different score rows (different geometry) ...
    assert not torch.allclose(per_ckpt[0], per_ckpt[1], atol=1e-4)
    # ... and the final score is the weighted sum of per-checkpoint cosines.
    expected = 0.25 * per_ckpt[0] + 0.75 * per_ckpt[1]
    assert torch.allclose(result["final_scores"], expected, atol=1e-6)


def test_smoke_adamw_candidate_update_pipeline(tmp_path):
    """Full pipeline with feature_type=adamw_candidate_update and a real AdamW state."""
    dataset = _make_dataset(12)
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(4, 6)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(x) ** 2).sum().backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    view = build_adamw_state_view_from_optimizer(optimizer, model)

    backend = TinyBackend(model)
    config = _config(tmp_path, feature_type="adamw_candidate_update",
                     checkpoint_paths=["ckpt-1"])

    def loader(ckpt_id, ckpt_path):
        return backend, view

    result = run_multicheckpoint_selection(
        backend_loader=loader,
        checkpoints=[("ckpt-1", "ckpt-1")],
        checkpoint_weights=[1.0],
        candidate_dataset=dataset,
        candidate_indices=list(range(8)),
        target_dataset=dataset,
        target_indices=[8, 9, 10, 11],
        config=config,
    )
    assert result["valid_mask"].all()
    assert torch.isfinite(result["final_scores"]).all()
    with open(result["paths"]["selected_indices"]) as f:
        meta = json.load(f)["metadata"]
    assert meta["method"] == "less_opsd_adamw_static"
    assert meta["optimizer"]["ckpt-1"]["optimizer_class"] == "AdamW"


def test_smoke_reference_ce_target_objective(tmp_path):
    """reference_ce affects only the target gradient; candidate loss stays OPSD."""
    r_opsd, _ = _run(_config(tmp_path / "opsd", num_candidate_rollouts=1,
                             num_target_rollouts=1))
    r_ce, _ = _run(_config(tmp_path / "ce", num_candidate_rollouts=1,
                           num_target_rollouts=1, target_objective="reference_ce"))
    # Same candidates, different target objective -> different scores in general.
    assert not torch.allclose(r_opsd["final_scores"], r_ce["final_scores"], atol=1e-5)


# --------------------------------------------------------------------------------------
# 18.10 Backward compatibility with the v1 scoring behavior
# --------------------------------------------------------------------------------------
def test_v2_single_rollout_raw_reproduces_v1_scoring(tmp_path):
    """feature_type=raw_gradient, 1 rollout, 1 checkpoint == v1 pipeline numerics."""
    dataset = _make_dataset(12)
    model = _make_model()
    backend = TinyBackend(model, deterministic=True)  # v1 had no rollout seeding hook

    config = _config(tmp_path, num_candidate_rollouts=1, num_target_rollouts=1)
    result, _ = _run(config, backend=backend, dataset=dataset)

    # Reimplement the v1 pipeline by hand: per-example normalized projected raw grad,
    # normalized mean target feature, dot-product scores.
    def v1_feature(example):
        model.zero_grad(set_to_none=True)
        x = torch.tensor(example["x"], dtype=torch.float32)
        (model(x) ** 2).sum().backward()
        z = count_sketch_project_trainable_grads(
            model, config.projection_dim, config.projection_seed
        )
        model.zero_grad(set_to_none=True)
        return normalize_feature(z)

    target_feat = normalize_feature(
        torch.stack([v1_feature(dataset[j]) for j in [8, 9, 10, 11]]).mean(dim=0)
    )
    v1_scores = torch.stack(
        [v1_feature(dataset[i]) @ target_feat for i in range(8)]
    )

    assert torch.allclose(result["final_scores"], v1_scores, atol=1e-5)
