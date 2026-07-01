"""LESS-inspired gradient-based data selection adapted to OPSD self-distillation.

This module implements the *static cached-rollout* MVP of LESS-OPSD:

  1. For each candidate problem, generate one on-policy student rollout, build the OPSD
     distillation batch, run the existing ``OPSDTrainer.compute_loss``, backprop once, and
     project the trainable-parameter gradient to a low dimension via CountSketch.
  2. Do the same for a small target/validation set and average the projected gradients.
  3. Score candidates by cosine alignment with the target feature and select top-k/top-p.
  4. Save selected indices + scores so ``opsd_train.py`` can train on the subset.

Reference (methodology only, not vendored): https://github.com/princeton-nlp/LESS
See ``docs/less_opsd_methodology.md`` for the conceptual adaptation and limitations.

The gradient feature deliberately comes from the OPSD loss path
(``OPSDTrainer.prepare_on_policy_distillation_batch`` -> ``OPSDTrainer.compute_loss``),
*not* from a plain supervised cross-entropy loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

try:  # tqdm is nice-to-have; degrade gracefully if missing.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - trivial fallback

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# Stable identifier for the static cached-rollout variant, stored in artifact metadata.
LESS_OPSD_METHOD_NAME = "less_opsd_static_cached_rollout"
RANDOM_METHOD_NAME = "random"
REFERENCE_REPO = "https://github.com/princeton-nlp/LESS"


@dataclass
class LESSOPSDSelectionConfig:
    """Configuration for a LESS-OPSD selection run."""

    projection_dim: int = 4096
    seed: int = 42
    candidate_limit: int | None = None
    target_limit: int | None = None
    selection_fraction: float | None = 0.05
    selection_num_examples: int | None = None
    gradient_batch_size: int = 1
    normalize_per_example: bool = True
    score_metric: str = "dot"
    output_dir: str = "outputs/less_opsd_selection"
    save_candidate_features: bool = False
    method: str = "less_opsd"


# --------------------------------------------------------------------------------------
# Pure utilities (unit-tested in tests/test_less_opsd_selector.py)
# --------------------------------------------------------------------------------------
def normalize_feature(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a feature vector (safe against zero vectors)."""
    return z / (z.norm() + eps)


def _stable_param_seed(seed: int, name: str) -> int:
    """Deterministic per-parameter seed independent of PYTHONHASHSEED."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    name_hash = int(digest, 16)
    return (int(seed) * 1_000_003 + name_hash) % (2**63 - 1)


def count_sketch_project_trainable_grads(
    model,
    projection_dim: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Project the model's trainable-parameter gradients to ``projection_dim`` via CountSketch.

    CountSketch / feature hashing avoids ever materializing a dense
    ``[num_params, projection_dim]`` projection matrix: each gradient coordinate is hashed
    into one of ``projection_dim`` buckets with a random sign in ``{-1, +1}``, and
    ``sign * grad`` is accumulated into that bucket. The hashing is fully deterministic for
    a fixed ``seed`` (per-parameter CPU generators seeded by a stable hash of the parameter
    name), so repeated runs with the same seed yield identical projections, while a
    different seed yields a different projection.

    Only parameters with ``requires_grad=True`` *and* a populated ``.grad`` contribute (for
    a PEFT/LoRA model this is exactly the LoRA adapter weights). Gradients are cast to
    float32 before projection.

    Returns:
        A CPU float32 tensor of shape ``[projection_dim]``.
    """
    sketch = torch.zeros(projection_dim, dtype=torch.float32, device=device)

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        grad = param.grad.detach().reshape(-1).to(torch.float32)
        numel = grad.numel()
        if numel == 0:
            continue

        # Deterministic CPU generator for this parameter (reproducible across machines).
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_param_seed(seed, name))

        buckets = torch.randint(0, projection_dim, (numel,), generator=generator)
        sign_bits = torch.randint(0, 2, (numel,), generator=generator, dtype=torch.int8)
        signs = sign_bits.to(torch.float32) * 2.0 - 1.0  # {0,1} -> {-1,+1}

        contrib = grad.detach().to("cpu") * signs
        sketch.index_add_(0, buckets.to(device), contrib.to(device))

    return sketch.detach().to("cpu", torch.float32)


def select_topk_by_score(
    scores: torch.Tensor,
    candidate_indices: list[int],
    selection_num_examples: int | None = None,
    selection_fraction: float | None = None,
) -> tuple[list[int], list[int], int]:
    """Select the top examples by score.

    Returns ``(selected_indices, topk_local_positions, k)`` where ``selected_indices`` are
    the original dataset indices and ``topk_local_positions`` index into ``candidate_indices``.
    """
    n = len(candidate_indices)
    if n == 0:
        return [], [], 0

    if selection_num_examples is not None:
        k = int(selection_num_examples)
    elif selection_fraction is not None:
        k = int(selection_fraction * n)
    else:
        raise ValueError("Provide either selection_num_examples or selection_fraction.")

    k = max(1, min(k, n))
    topk_local = torch.topk(scores, k=k).indices.tolist()
    selected_indices = [candidate_indices[j] for j in topk_local]
    return selected_indices, topk_local, k


def load_selected_indices(path: str) -> list[int]:
    """Load selected indices from a JSON file.

    Accepts either a bare list ``[1, 5, 10]`` or a dict
    ``{"selected_indices": [1, 5, 10], "metadata": {...}}``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [int(i) for i in data]
    if isinstance(data, dict) and "selected_indices" in data:
        return [int(i) for i in data["selected_indices"]]
    raise ValueError(
        f"Unrecognized selected-indices file format at {path!r}: expected a list or a dict "
        "with a 'selected_indices' key."
    )


# --------------------------------------------------------------------------------------
# Gradient extractor
# --------------------------------------------------------------------------------------
def _move_inputs_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    """Move only the tensor entries of a collator output to ``device`` (ints stay as-is)."""
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


class LESSOPSDGradientExtractor:
    """Extracts projected, normalized OPSD gradient features for dataset examples."""

    def __init__(self, trainer, projection_dim: int, seed: int):
        self.trainer = trainer
        self.model = trainer.model
        self.projection_dim = projection_dim
        self.seed = seed

    @torch.no_grad()
    def _collate(self, raw_example: dict) -> dict[str, Any]:
        # The collator itself does not need grad tracking.
        return self.trainer.data_collator([raw_example])

    def extract_one_example_feature(self, raw_example: dict, example_index: int) -> torch.Tensor:
        """Compute the projected, normalized OPSD gradient feature for one example.

        Uses the *same* generation + distillation-batch path as training
        (``prepare_on_policy_distillation_batch``) and the *same* loss
        (``compute_loss``), so the selection gradient matches the training gradient.
        """
        trainer = self.trainer
        model = self.model

        model.train()
        device = trainer.accelerator.device

        inputs = self._collate(raw_example)
        inputs = _move_inputs_to_device(inputs, device)

        # Generate on-policy rollout + assemble student/teacher tensors + labels.
        prepared_inputs, _, _ = trainer.prepare_on_policy_distillation_batch(model, inputs)

        # IMPORTANT: this is the OPSD distillation loss, not a supervised CE loss.
        loss = trainer.compute_loss(model, prepared_inputs)
        if isinstance(loss, tuple):
            loss = loss[0]

        model.zero_grad(set_to_none=True)
        loss.backward()

        feature = count_sketch_project_trainable_grads(
            model, self.projection_dim, self.seed, device="cpu"
        )
        feature = normalize_feature(feature)

        model.zero_grad(set_to_none=True)
        return feature.detach().cpu()

    def extract_dataset_features(
        self, dataset, indices: list[int], split_name: str
    ) -> torch.Tensor:
        """Extract per-example features for ``indices`` and stack into ``[len(indices), proj_dim]``."""
        features = []
        for local_pos, idx in enumerate(
            tqdm(indices, desc=f"LESS-OPSD grads [{split_name}]")
        ):
            raw_example = dataset[int(idx)]
            feature = self.extract_one_example_feature(raw_example, int(idx))
            features.append(feature)
        if not features:
            return torch.empty(0, self.projection_dim, dtype=torch.float32)
        return torch.stack(features, dim=0)


# --------------------------------------------------------------------------------------
# Target feature + scoring
# --------------------------------------------------------------------------------------
def compute_target_feature(extractor: LESSOPSDGradientExtractor, target_dataset, target_indices):
    """Average the per-example target features, then normalize."""
    target_features = extractor.extract_dataset_features(target_dataset, target_indices, "target")
    return normalize_feature(target_features.mean(dim=0))


def score_candidates(candidate_features: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
    """Cosine similarity assuming inputs are normalized (dot product)."""
    return candidate_features @ target_feature


# --------------------------------------------------------------------------------------
# Artifact saving
# --------------------------------------------------------------------------------------
def save_selection_artifacts(
    output_dir: str,
    config: LESSOPSDSelectionConfig,
    method_name: str,
    candidate_indices: list[int],
    scores: torch.Tensor,
    selected_indices: list[int],
    target_feature: Optional[torch.Tensor] = None,
    candidate_features: Optional[torch.Tensor] = None,
) -> dict[str, str]:
    """Persist selection artifacts. Returns a mapping of artifact name -> path.

    Always writes ``selected_indices.json``, ``scores.jsonl`` and ``selection_config.json``.
    Writes ``target_feature.pt`` when ``target_feature`` is provided, and
    ``candidate_features.pt`` only when ``config.save_candidate_features`` is True.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    selected_set = set(int(i) for i in selected_indices)

    # selected_indices.json
    selected_path = os.path.join(output_dir, "selected_indices.json")
    selected_payload = {
        "selected_indices": [int(i) for i in selected_indices],
        "metadata": {
            "method": method_name,
            "reference_repo": REFERENCE_REPO,
            "projection_dim": config.projection_dim,
            "selection_fraction": config.selection_fraction,
            "selection_num_examples": config.selection_num_examples,
            "num_candidates": len(candidate_indices),
            "num_selected": len(selected_indices),
            "seed": config.seed,
        },
    }
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, indent=2)
    paths["selected_indices"] = selected_path

    # scores.jsonl (ranked by descending score)
    scores_list = scores.tolist()
    order = sorted(range(len(candidate_indices)), key=lambda j: scores_list[j], reverse=True)
    scores_path = os.path.join(output_dir, "scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for rank, local_pos in enumerate(order, start=1):
            original_index = int(candidate_indices[local_pos])
            row = {
                "original_index": original_index,
                "score": float(scores_list[local_pos]),
                "rank": rank,
                "selected": original_index in selected_set,
            }
            f.write(json.dumps(row) + "\n")
    paths["scores"] = scores_path

    # selection_config.json
    config_path = os.path.join(output_dir, "selection_config.json")
    config_payload = dict(asdict(config))
    config_payload["method_name"] = method_name
    config_payload["reference_repo"] = REFERENCE_REPO
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    paths["selection_config"] = config_path

    # target_feature.pt
    if target_feature is not None:
        target_path = os.path.join(output_dir, "target_feature.pt")
        torch.save(target_feature.detach().cpu(), target_path)
        paths["target_feature"] = target_path

    # candidate_features.pt (optional; can be large)
    if config.save_candidate_features and candidate_features is not None:
        cand_path = os.path.join(output_dir, "candidate_features.pt")
        torch.save(candidate_features.detach().cpu(), cand_path)
        paths["candidate_features"] = cand_path

    return paths


# --------------------------------------------------------------------------------------
# High-level drivers
# --------------------------------------------------------------------------------------
def run_less_opsd_selection(
    trainer,
    candidate_dataset,
    candidate_indices: list[int],
    target_dataset,
    target_indices: list[int],
    config: LESSOPSDSelectionConfig,
) -> dict[str, Any]:
    """Run the full static cached-rollout LESS-OPSD selection and save artifacts."""
    extractor = LESSOPSDGradientExtractor(trainer, config.projection_dim, config.seed)

    target_feature = compute_target_feature(extractor, target_dataset, target_indices)

    candidate_features = extractor.extract_dataset_features(
        candidate_dataset, candidate_indices, "candidate"
    )
    scores = score_candidates(candidate_features, target_feature)

    selected_indices, _, k = select_topk_by_score(
        scores,
        candidate_indices,
        selection_num_examples=config.selection_num_examples,
        selection_fraction=config.selection_fraction,
    )

    paths = save_selection_artifacts(
        output_dir=config.output_dir,
        config=config,
        method_name=LESS_OPSD_METHOD_NAME,
        candidate_indices=candidate_indices,
        scores=scores,
        selected_indices=selected_indices,
        target_feature=target_feature,
        candidate_features=candidate_features,
    )

    return {
        "selected_indices": selected_indices,
        "scores": scores,
        "k": k,
        "paths": paths,
    }


def run_random_selection(
    candidate_indices: list[int],
    config: LESSOPSDSelectionConfig,
) -> dict[str, Any]:
    """Random-selection baseline that emits the exact same artifact format as LESS-OPSD.

    Each candidate is assigned a deterministic pseudo-random score (seeded by
    ``config.seed``); selection then reuses the same top-k path so the outputs are
    directly comparable to a LESS-OPSD run.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    scores = torch.rand(len(candidate_indices), generator=generator)

    selected_indices, _, k = select_topk_by_score(
        scores,
        candidate_indices,
        selection_num_examples=config.selection_num_examples,
        selection_fraction=config.selection_fraction,
    )

    paths = save_selection_artifacts(
        output_dir=config.output_dir,
        config=config,
        method_name=RANDOM_METHOD_NAME,
        candidate_indices=candidate_indices,
        scores=scores,
        selected_indices=selected_indices,
        target_feature=None,
        candidate_features=None,
    )

    return {
        "selected_indices": selected_indices,
        "scores": scores,
        "k": k,
        "paths": paths,
    }
