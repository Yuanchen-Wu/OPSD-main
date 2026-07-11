"""LESS-inspired gradient-based data selection adapted to OPSD self-distillation.

Version 2 of the selector adds, on top of the static raw-gradient MVP:

  * AdamW-aware candidate features (``adamw_candidate_update`` approximates the
    hypothetical next AdamW update direction using real saved optimizer moments;
    ``adamw_fixed_preconditioner`` divides the raw gradient by ``sqrt(v_hat)+eps``).
  * Multiple on-policy rollouts per example, averaged *before* normalization, plus a
    rollout-consistency diagnostic.
  * Optional multi-checkpoint score aggregation (cosine per checkpoint, then a weighted
    sum of per-checkpoint scores).
  * A generic backend abstraction (``DistillationGradientBackend``) so the selection
    machinery does not depend on the teacher being the same model.
  * Resumable feature extraction with a configuration fingerprint.

Candidate features use the configured optimizer-aware transform; target features are
always **raw** loss gradients (LESS's candidate/target asymmetry).

Reference (methodology only, not vendored): https://github.com/princeton-nlp/LESS
See ``docs/less_opsd_methodology.md``.

Only ``torch`` + stdlib are imported at module level so all pure utilities remain
unit-testable offline. The OPSD-specific backend lives in ``less_opsd_backend.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Protocol

import torch

try:  # tqdm is nice-to-have; degrade gracefully if missing.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - trivial fallback

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# --------------------------------------------------------------------------------------
# Method naming / constants
# --------------------------------------------------------------------------------------
# Legacy identifier written by the v1 static single-rollout raw-gradient implementation.
LESS_OPSD_METHOD_NAME = "less_opsd_static_cached_rollout"
RANDOM_METHOD_NAME = "random"
WARMUP_RANDOM_METHOD_NAME = "warmup_random"
REFERENCE_REPO = "https://github.com/princeton-nlp/LESS"

# v2 method-version marker stored in all new artifacts so old/new outputs are
# distinguishable.
METHOD_VERSION = "2.0"

FEATURE_TYPES = ("raw_gradient", "adamw_candidate_update", "adamw_fixed_preconditioner")
TARGET_OBJECTIVES = ("opsd", "reference_ce")
CHECKPOINT_WEIGHTINGS = ("uniform", "learning_rate", "explicit")

# Feature types that require saved AdamW optimizer state.
OPTIMIZER_AWARE_FEATURE_TYPES = ("adamw_candidate_update", "adamw_fixed_preconditioner")


def method_name_for(feature_type: str, num_checkpoints: int) -> str:
    """Precise method name for artifacts (see docs: method naming)."""
    base = {
        "raw_gradient": "raw",
        "adamw_candidate_update": "adamw",
        "adamw_fixed_preconditioner": "adamw_fixed",
    }[feature_type]
    suffix = "multicheckpoint" if num_checkpoints > 1 else "static"
    return f"less_opsd_{base}_{suffix}"


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class SelectionConfigError(ValueError):
    """Invalid or inconsistent selection configuration."""


class OptimizerStateError(RuntimeError):
    """Missing, unsupported, or inconsistent optimizer state for optimizer-aware features."""


class ResumeMismatchError(RuntimeError):
    """Partial artifacts on disk were produced with an incompatible configuration."""


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class LESSOPSDSelectionConfig:
    """Configuration for a LESS-OPSD selection run.

    Backward compatibility: ``seed`` is the legacy field. When ``projection_seed`` /
    ``rollout_seed`` are left as ``None`` they default to ``seed`` (projection) and
    ``seed + 1`` (rollouts) so old configs keep working while the two concerns stay
    separable.
    """

    # Feature construction
    feature_type: str = "raw_gradient"
    target_objective: str = "opsd"
    num_candidate_rollouts: int = 1
    num_target_rollouts: int = 1

    # Checkpoints
    checkpoint_paths: list[str] = field(default_factory=list)
    checkpoint_weighting: str = "uniform"
    checkpoint_weights: list[float] | None = None

    # Projection / seeds
    projection_dim: int = 4096
    seed: int = 42  # legacy umbrella seed
    projection_seed: int | None = None
    rollout_seed: int | None = None
    candidate_subset_seed: int | None = None
    target_subset_seed: int | None = None
    warmup_subset_seed: int | None = None

    # Pools
    candidate_limit: int | None = None
    target_limit: int | None = None

    # Selection size (exactly one of the two must be active)
    selection_fraction: float | None = 0.05
    selection_num_examples: int | None = None

    # Extraction / resume
    gradient_batch_size: int = 1
    save_every: int = 25
    resume: bool = False

    # Misc (legacy fields preserved)
    normalize_per_example: bool = True
    score_metric: str = "dot"
    output_dir: str = "outputs/less_opsd_selection"
    save_candidate_features: bool = False
    method: str = "less_opsd"

    def __post_init__(self):
        if self.projection_seed is None:
            self.projection_seed = int(self.seed)
        if self.rollout_seed is None:
            self.rollout_seed = int(self.seed) + 1
        if self.candidate_subset_seed is None:
            self.candidate_subset_seed = int(self.seed)
        if self.target_subset_seed is None:
            self.target_subset_seed = int(self.seed)
        if self.warmup_subset_seed is None:
            self.warmup_subset_seed = int(self.seed)


def validate_selection_config(config: LESSOPSDSelectionConfig) -> None:
    """Raise ``SelectionConfigError`` on invalid or inconsistent configurations."""
    if config.feature_type not in FEATURE_TYPES:
        raise SelectionConfigError(
            f"Unknown feature_type={config.feature_type!r}; expected one of {FEATURE_TYPES}."
        )
    if config.target_objective not in TARGET_OBJECTIVES:
        raise SelectionConfigError(
            f"Unknown target_objective={config.target_objective!r}; "
            f"expected one of {TARGET_OBJECTIVES}."
        )
    if config.num_candidate_rollouts < 1:
        raise SelectionConfigError(
            f"num_candidate_rollouts must be >= 1, got {config.num_candidate_rollouts}."
        )
    if config.num_target_rollouts < 1:
        raise SelectionConfigError(
            f"num_target_rollouts must be >= 1, got {config.num_target_rollouts}."
        )
    if config.projection_dim < 1:
        raise SelectionConfigError(f"projection_dim must be >= 1, got {config.projection_dim}.")
    if config.save_every < 1:
        raise SelectionConfigError(f"save_every must be >= 1, got {config.save_every}.")

    has_fraction = config.selection_fraction is not None
    has_num = config.selection_num_examples is not None
    if has_fraction == has_num:  # both set or both unset
        raise SelectionConfigError(
            "Exactly one of selection_fraction and selection_num_examples must be set "
            f"(got selection_fraction={config.selection_fraction}, "
            f"selection_num_examples={config.selection_num_examples})."
        )
    if has_fraction and not (0.0 < config.selection_fraction <= 1.0):
        raise SelectionConfigError(
            f"selection_fraction must be in (0, 1], got {config.selection_fraction}."
        )
    if has_num and config.selection_num_examples < 1:
        raise SelectionConfigError(
            f"selection_num_examples must be >= 1, got {config.selection_num_examples}."
        )

    if config.checkpoint_weighting not in CHECKPOINT_WEIGHTINGS:
        raise SelectionConfigError(
            f"Unknown checkpoint_weighting={config.checkpoint_weighting!r}; "
            f"expected one of {CHECKPOINT_WEIGHTINGS}."
        )
    if config.checkpoint_weighting == "explicit":
        if not config.checkpoint_weights:
            raise SelectionConfigError(
                "checkpoint_weighting='explicit' requires checkpoint_weights."
            )
        n_ckpts = max(1, len(config.checkpoint_paths))
        if len(config.checkpoint_weights) != n_ckpts:
            raise SelectionConfigError(
                f"Number of explicit checkpoint_weights ({len(config.checkpoint_weights)}) "
                f"must match number of checkpoints ({n_ckpts})."
            )
        if any(w < 0 for w in config.checkpoint_weights):
            raise SelectionConfigError("checkpoint_weights must be non-negative.")
        if sum(config.checkpoint_weights) <= 0:
            raise SelectionConfigError("checkpoint_weights must sum to a positive value.")

    if config.feature_type in OPTIMIZER_AWARE_FEATURE_TYPES and not config.checkpoint_paths:
        raise SelectionConfigError(
            f"feature_type={config.feature_type!r} requires at least one checkpoint with "
            "saved AdamW optimizer state (--checkpoint_paths). The base initialization has "
            "no optimizer moments, and silently falling back to raw gradients is not allowed."
        )


def config_fingerprint(config: LESSOPSDSelectionConfig, extra: dict | None = None) -> str:
    """Stable short hash of every setting that makes partial artifacts (in)compatible.

    Covers projection seed/dim, rollout seed/counts, feature type, target objective,
    checkpoint identities, and whatever the caller adds via ``extra`` (model id, dataset
    id, candidate/target index hashes, optimizer metadata, ...).
    """
    payload = {
        "method_version": METHOD_VERSION,
        "feature_type": config.feature_type,
        "target_objective": config.target_objective,
        "num_candidate_rollouts": config.num_candidate_rollouts,
        "num_target_rollouts": config.num_target_rollouts,
        "projection_dim": config.projection_dim,
        "projection_seed": config.projection_seed,
        "rollout_seed": config.rollout_seed,
        "checkpoint_paths": list(config.checkpoint_paths),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def indices_digest(indices: list[int]) -> str:
    """Short stable hash of an index list (used inside the config fingerprint)."""
    blob = json.dumps([int(i) for i in indices]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------
def derive_rollout_seed(
    base_rollout_seed: int,
    checkpoint_id: str,
    example_index: int,
    rollout_index: int,
    split: str,
) -> int:
    """Deterministic per-rollout seed from a stable (non process-dependent) hash."""
    payload = f"{int(base_rollout_seed)}|{checkpoint_id}|{int(example_index)}|{int(rollout_index)}|{split}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


# --------------------------------------------------------------------------------------
# Pure feature utilities
# --------------------------------------------------------------------------------------
def normalize_feature(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a feature vector (safe against zero vectors)."""
    return z / (z.norm() + eps)


def _stable_param_seed(seed: int, name: str) -> int:
    """Deterministic per-parameter-name seed independent of PYTHONHASHSEED."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    name_hash = int(digest, 16)
    return (int(seed) * 1_000_003 + name_hash) % (2**63 - 1)


def count_sketch_project_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    projection_dim: int,
    seed: int,
) -> torch.Tensor:
    """Project a stream of named tensors to ``projection_dim`` via CountSketch.

    Each tensor coordinate is hashed into one of ``projection_dim`` buckets with a random
    sign in ``{-1, +1}``; ``sign * value`` is accumulated into the bucket. The bucket/sign
    mapping depends only on the tensor *name* and ``seed``, so the same parameter name
    always maps identically, and different seeds produce different mappings. No dense
    ``[num_params, projection_dim]`` matrix is ever materialized.

    Values are accumulated in float32 on CPU; returns a CPU float32 ``[projection_dim]``
    vector.
    """
    sketch = torch.zeros(projection_dim, dtype=torch.float32, device="cpu")

    for name, tensor in named_tensors:
        values = tensor.detach().reshape(-1).to("cpu", torch.float32)
        numel = values.numel()
        if numel == 0:
            continue

        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_param_seed(seed, name))

        buckets = torch.randint(0, projection_dim, (numel,), generator=generator)
        sign_bits = torch.randint(0, 2, (numel,), generator=generator, dtype=torch.int8)
        signs = sign_bits.to(torch.float32) * 2.0 - 1.0  # {0,1} -> {-1,+1}

        sketch.index_add_(0, buckets, values * signs)
        del values, buckets, sign_bits, signs

    return sketch


def iter_trainable_grad_tensors(model) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, grad)`` for trainable parameters with a populated ``.grad``."""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            yield name, param.grad


def count_sketch_project_trainable_grads(
    model,
    projection_dim: int,
    seed: int,
    device: str | torch.device = "cpu",  # kept for backward compatibility; output is CPU
) -> torch.Tensor:
    """Convenience wrapper: project the model's raw trainable-parameter gradients.

    Behaviorally identical to the v1 implementation (same per-name hashing); now
    implemented on top of :func:`count_sketch_project_named_tensors`.
    """
    return count_sketch_project_named_tensors(
        iter_trainable_grad_tensors(model), projection_dim, seed
    )


# --------------------------------------------------------------------------------------
# AdamW-aware feature transforms
# --------------------------------------------------------------------------------------
def hypothetical_adamw_candidate_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Candidate-dependent AdamW direction if ``grad`` were processed next.

    Computes (without mutating any input)::

        m' = beta1 * m + (1 - beta1) * g
        v' = beta2 * v + (1 - beta2) * g^2
        m_hat = m' / (1 - beta1^(t+1))
        v_hat = v' / (1 - beta2^(t+1))
        Gamma = m_hat / (sqrt(v_hat) + eps)

    This equals (actual AdamW parameter displacement) / (-lr) for weight_decay=0; the
    decoupled weight-decay term is deliberately excluded because at a fixed checkpoint it
    is identical for every candidate and therefore uninformative for ranking.
    """
    g = grad.detach().to(torch.float32)
    m = exp_avg.detach().to(torch.float32)
    v = exp_avg_sq.detach().to(torch.float32)

    t_next = int(step) + 1
    m_next = beta1 * m + (1.0 - beta1) * g
    v_next = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m_next / (1.0 - beta1**t_next)
    v_hat = v_next / (1.0 - beta2**t_next)
    return m_hat / (v_hat.sqrt() + eps)


def adamw_fixed_preconditioner(
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Fixed Adam preconditioner: ``g / (sqrt(v_hat_k) + eps)``.

    Uses the checkpoint's bias-corrected second moment without hypothetically updating the
    moments with the candidate gradient. Linear in ``grad``.
    """
    if int(step) < 1:
        raise OptimizerStateError(
            f"adamw_fixed_preconditioner requires optimizer step >= 1, got step={step}."
        )
    g = grad.detach().to(torch.float32)
    v = exp_avg_sq.detach().to(torch.float32)
    v_hat = v / (1.0 - beta2 ** int(step))
    return g / (v_hat.sqrt() + eps)


# --------------------------------------------------------------------------------------
# Optimizer state handling
# --------------------------------------------------------------------------------------
@dataclass
class AdamWStateView:
    """Name-keyed, CPU-resident view of an AdamW optimizer state.

    ``state_by_name[name] = {"exp_avg": Tensor, "exp_avg_sq": Tensor, "step": int}``
    with tensors on CPU in float32. Never mutated by the feature transforms.
    """

    state_by_name: dict[str, dict[str, Any]]
    beta1: float
    beta2: float
    eps: float
    optimizer_class: str
    learning_rate: float | None = None

    def state_for(self, name: str) -> dict[str, Any]:
        if name not in self.state_by_name:
            raise OptimizerStateError(
                f"No optimizer state found for trainable parameter {name!r}. "
                "The checkpoint's optimizer state does not cover the current trainable "
                "parameter set."
            )
        return self.state_by_name[name]

    def metadata(self) -> dict:
        steps = sorted({int(s["step"]) for s in self.state_by_name.values()})
        return {
            "optimizer_class": self.optimizer_class,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "learning_rate": self.learning_rate,
            "num_state_tensors": len(self.state_by_name),
            "optimizer_steps": steps[:8],
        }


def build_adamw_state_view_from_optimizer(optimizer, model) -> AdamWStateView:
    """Build a validated, name-keyed CPU view from a *loaded* optimizer instance.

    The optimizer must already contain the checkpoint's state (i.e.
    ``optimizer.load_state_dict`` was called), which is the only safe way to map
    serialized integer parameter IDs onto parameter objects: PyTorch resolves them
    through the parameter-group structure.

    Validates: optimizer type (AdamW only), presence of ``exp_avg``/``exp_avg_sq``/step,
    shape agreement with the model parameters, coverage of every trainable parameter, and
    consistent betas/eps across groups.
    """
    cls_name = type(optimizer).__name__
    if "adamw" not in cls_name.lower():
        raise OptimizerStateError(
            f"Unsupported optimizer type {cls_name!r}: only AdamW-family optimizers are "
            "supported for optimizer-aware selection features. Paged/fused/8-bit "
            "optimizer-state formats are not supported."
        )

    id_to_name = {id(p): n for n, p in model.named_parameters()}

    betas_seen: set[tuple[float, float]] = set()
    eps_seen: set[float] = set()
    lr_seen: set[float] = set()
    state_by_name: dict[str, dict[str, Any]] = {}

    for group in optimizer.param_groups:
        betas = tuple(float(b) for b in group.get("betas", ()))
        if len(betas) != 2:
            raise OptimizerStateError(
                f"Optimizer group is missing AdamW 'betas' (got {group.get('betas')!r})."
            )
        eps = float(group.get("eps"))
        betas_seen.add(betas)
        eps_seen.add(eps)
        if "lr" in group:
            lr_seen.add(float(group["lr"]))

        for param in group["params"]:
            name = id_to_name.get(id(param))
            if name is None:
                raise OptimizerStateError(
                    "Optimizer contains a parameter that does not belong to the model; "
                    "the checkpoint does not match the current model structure."
                )
            if not param.requires_grad:
                continue
            state = optimizer.state.get(param)
            if not state:
                raise OptimizerStateError(
                    f"Missing optimizer state for trainable parameter {name!r}. "
                    "Was the checkpoint saved after at least one optimizer step?"
                )
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if key not in state:
                    raise OptimizerStateError(
                        f"Optimizer state for {name!r} lacks {key!r}; AdamW state with "
                        "exp_avg/exp_avg_sq/step is required."
                    )
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if exp_avg.shape != param.shape or exp_avg_sq.shape != param.shape:
                raise OptimizerStateError(
                    f"Optimizer moment shape mismatch for {name!r}: param {tuple(param.shape)} "
                    f"vs exp_avg {tuple(exp_avg.shape)} / exp_avg_sq {tuple(exp_avg_sq.shape)}."
                )
            step_val = state["step"]
            step = int(step_val.item()) if torch.is_tensor(step_val) else int(step_val)
            if step < 1:
                raise OptimizerStateError(
                    f"Optimizer state for {name!r} has step={step}; expected >= 1 "
                    "(checkpoint saved before any optimizer step?)."
                )
            state_by_name[name] = {
                "exp_avg": exp_avg.detach().to("cpu", torch.float32),
                "exp_avg_sq": exp_avg_sq.detach().to("cpu", torch.float32),
                "step": step,
            }

    if len(betas_seen) != 1 or len(eps_seen) != 1:
        raise OptimizerStateError(
            f"Inconsistent AdamW hyperparameters across param groups: betas={betas_seen}, "
            f"eps={eps_seen}. Optimizer-aware selection requires a single (betas, eps)."
        )

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n in trainable_names if n not in state_by_name]
    if missing:
        raise OptimizerStateError(
            f"Optimizer state covers {len(state_by_name)} parameters but the model has "
            f"{len(trainable_names)} trainable parameters; missing e.g. {missing[:3]}."
        )

    (beta1, beta2), eps = next(iter(betas_seen)), next(iter(eps_seen))
    return AdamWStateView(
        state_by_name=state_by_name,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        optimizer_class=cls_name,
        learning_rate=(next(iter(lr_seen)) if len(lr_seen) == 1 else None),
    )


def load_adamw_state_view_from_checkpoint(
    model,
    optimizer,
    checkpoint_path: str,
) -> AdamWStateView:
    """Load ``optimizer.pt`` from a HF Trainer checkpoint into ``optimizer`` and view it.

    ``optimizer`` must be a freshly constructed AdamW with the *same parameter-group
    structure* used during training (e.g. built via ``trainer.create_optimizer()``); this
    is what makes the serialized parameter IDs resolve safely. State tensors are loaded
    to CPU.
    """
    opt_path = os.path.join(checkpoint_path, "optimizer.pt")
    if not os.path.exists(opt_path):
        raise OptimizerStateError(
            f"Checkpoint {checkpoint_path!r} has no optimizer.pt; optimizer-aware feature "
            "types require a checkpoint saved with optimizer state (HF Trainer save "
            "checkpoints include it by default)."
        )
    state_dict = torch.load(opt_path, map_location="cpu", weights_only=False)
    try:
        optimizer.load_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001 - re-raise with context
        raise OptimizerStateError(
            f"Failed to load optimizer state from {opt_path!r} into a freshly constructed "
            f"{type(optimizer).__name__}: {exc}. This usually means the current model's "
            "trainable-parameter structure (LoRA config) does not match the checkpoint."
        ) from exc
    return build_adamw_state_view_from_optimizer(optimizer, model)


def iter_transformed_grad_tensors(
    model,
    feature_type: str,
    opt_view: AdamWStateView | None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, transformed_gradient)`` for trainable params, one tensor at a time.

    The transform is applied per parameter on CPU float32 and fed directly into
    CountSketch by the caller — no full flattened gradient vector is materialized.
    """
    if feature_type == "raw_gradient":
        yield from iter_trainable_grad_tensors(model)
        return

    if feature_type not in OPTIMIZER_AWARE_FEATURE_TYPES:
        raise SelectionConfigError(f"Unknown feature_type={feature_type!r}.")
    if opt_view is None:
        raise OptimizerStateError(
            f"feature_type={feature_type!r} requires a loaded AdamW optimizer state, but "
            "none was provided. Raw-gradient fallback is intentionally not performed."
        )

    for name, grad in iter_trainable_grad_tensors(model):
        state = opt_view.state_for(name)
        g_cpu = grad.detach().to("cpu", torch.float32)
        if feature_type == "adamw_candidate_update":
            transformed = hypothetical_adamw_candidate_update(
                g_cpu,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                opt_view.beta1,
                opt_view.beta2,
                opt_view.eps,
            )
        else:  # adamw_fixed_preconditioner
            transformed = adamw_fixed_preconditioner(
                g_cpu,
                state["exp_avg_sq"],
                state["step"],
                opt_view.beta2,
                opt_view.eps,
            )
        yield name, transformed
        del g_cpu, transformed


# --------------------------------------------------------------------------------------
# Rollout aggregation
# --------------------------------------------------------------------------------------
def rollout_consistency(z_list: list[torch.Tensor], eps: float = 1e-12) -> float:
    """Directional consistency of rollout update vectors.

    ``rho = || mean_m z_m / (||z_m|| + eps) ||``: 1.0 when all rollouts agree in
    direction, ~0.0 when they cancel. For a single rollout this is 1.0 by construction.
    """
    if not z_list:
        return 0.0
    unit = torch.stack([z / (z.norm() + eps) for z in z_list], dim=0)
    return float(unit.mean(dim=0).norm().item())


def aggregate_rollout_projections(
    z_list: list[torch.Tensor], eps: float = 1e-12
) -> tuple[torch.Tensor, float, bool]:
    """Average *unnormalized* projected rollout vectors, then normalize once.

    Returns ``(u, rho, invalid)``. Estimates the direction of the expected candidate
    update — deliberately not the average of unit directions. Non-finite rollout vectors
    are dropped; the example is marked invalid when nothing finite remains or the
    averaged vector has (near-)zero norm.
    """
    finite = [z for z in z_list if bool(torch.isfinite(z).all())]
    if not finite:
        dim = z_list[0].numel() if z_list else 0
        return torch.zeros(dim, dtype=torch.float32), 0.0, True

    z_bar = torch.stack(finite, dim=0).mean(dim=0)
    norm = z_bar.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= eps:
        return torch.zeros_like(z_bar), rollout_consistency(finite, eps), True

    rho = rollout_consistency(finite, eps)
    return z_bar / (norm + eps), rho, False


# --------------------------------------------------------------------------------------
# Checkpoint weighting and aggregation
# --------------------------------------------------------------------------------------
def normalize_checkpoint_weights(
    weighting: str,
    num_checkpoints: int,
    explicit_weights: list[float] | None = None,
    learning_rates: list[float] | None = None,
) -> list[float]:
    """Resolve per-checkpoint weights, normalized to sum to one."""
    if num_checkpoints < 1:
        raise SelectionConfigError("At least one checkpoint is required.")

    if weighting == "uniform":
        raw = [1.0] * num_checkpoints
    elif weighting == "explicit":
        if not explicit_weights or len(explicit_weights) != num_checkpoints:
            raise SelectionConfigError(
                f"Explicit weighting requires exactly {num_checkpoints} weights, got "
                f"{explicit_weights!r}."
            )
        raw = [float(w) for w in explicit_weights]
    elif weighting == "learning_rate":
        if not learning_rates or len(learning_rates) != num_checkpoints:
            raise SelectionConfigError(
                "learning_rate weighting requires a recovered learning rate for every "
                f"checkpoint; got {learning_rates!r} for {num_checkpoints} checkpoints."
            )
        if any(lr is None or lr <= 0 for lr in learning_rates):
            raise SelectionConfigError(
                f"learning_rate weighting requires positive learning rates, got "
                f"{learning_rates!r}. Uniform weights are NOT silently substituted."
            )
        raw = [float(lr) for lr in learning_rates]
    else:
        raise SelectionConfigError(f"Unknown checkpoint_weighting={weighting!r}.")

    total = sum(raw)
    if total <= 0:
        raise SelectionConfigError(f"Checkpoint weights must sum to a positive value: {raw}.")
    return [w / total for w in raw]


def aggregate_checkpoint_scores(
    per_checkpoint_scores: torch.Tensor,  # [K, N], NaN marks invalid entries
    weights: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted sum of per-checkpoint cosine scores.

    Returns ``(final_scores [N], valid_mask [N])``. A candidate is valid only when its
    score is finite at *every* checkpoint (documented rule; invalid candidates receive
    ``-inf`` and are never selected).
    """
    if per_checkpoint_scores.ndim != 2:
        raise ValueError("per_checkpoint_scores must be [K, N].")
    k = per_checkpoint_scores.shape[0]
    if k != len(weights):
        raise SelectionConfigError(
            f"Got {k} checkpoint score rows but {len(weights)} weights."
        )
    w = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
    valid_mask = torch.isfinite(per_checkpoint_scores).all(dim=0)
    final = (per_checkpoint_scores * w).sum(dim=0)
    final = torch.where(valid_mask, final, torch.full_like(final, float("-inf")))
    return final, valid_mask


# --------------------------------------------------------------------------------------
# Selection + legacy loaders
# --------------------------------------------------------------------------------------
def select_topk_by_score(
    scores: torch.Tensor,
    candidate_indices: list[int],
    selection_num_examples: int | None = None,
    selection_fraction: float | None = None,
) -> tuple[list[int], list[int], int]:
    """Select the top examples by score.

    Returns ``(selected_indices, topk_local_positions, k)`` where ``selected_indices`` are
    the original dataset indices and ``topk_local_positions`` index into
    ``candidate_indices``. ``-inf`` scores (invalid candidates) are never selected.
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
    topk = torch.topk(scores, k=k)
    topk_local = [
        int(pos) for pos, val in zip(topk.indices.tolist(), topk.values.tolist())
        if val != float("-inf")
    ]
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
# Backend protocol
# --------------------------------------------------------------------------------------
class DistillationGradientBackend(Protocol):
    """What the generic selector needs from a distillation setup.

    The backend owns prompt construction, on-policy rollout generation, teacher scoring,
    and loss computation. The generic selector never reads dataset fields (e.g. the
    reference solution) directly — only the backend does.
    """

    @property
    def student_model(self): ...

    def prepare_candidate_example(self, raw_example: dict, generation_seed: int) -> Any: ...

    def compute_candidate_loss(self, prepared_example: Any) -> torch.Tensor: ...

    def prepare_target_example(
        self, raw_example: dict, objective: str, generation_seed: int
    ) -> Any: ...

    def compute_target_loss(self, prepared_example: Any, objective: str) -> torch.Tensor: ...

    def target_requires_rollout(self, objective: str) -> bool: ...

    def metadata(self) -> dict: ...


# --------------------------------------------------------------------------------------
# Resumable feature store
# --------------------------------------------------------------------------------------
class FeatureStore:
    """Resumable on-disk store for per-example projected features.

    Layout (under ``<output_dir>/resume/``)::

        manifest.json                       # fingerprint + completed indices per shard
        features_<split>_<ckpt_id>.pt       # {index: {"u": Tensor, "rho": float, "invalid": bool}}

    A shard is one (split, checkpoint) pair. The manifest carries the configuration
    fingerprint; resuming with a different fingerprint raises ``ResumeMismatchError`` so
    artifacts from different seeds/checkpoints/feature types/rollout counts/target
    objectives/optimizer states are never silently mixed.
    """

    def __init__(self, output_dir: str, fingerprint: str, resume: bool):
        self.dir = os.path.join(output_dir, "resume")
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.fingerprint = fingerprint
        os.makedirs(self.dir, exist_ok=True)

        self.manifest: dict[str, Any] = {"fingerprint": fingerprint, "completed": {}}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                existing = json.load(f)
            if resume:
                if existing.get("fingerprint") != fingerprint:
                    raise ResumeMismatchError(
                        "Partial selection artifacts in "
                        f"{self.dir!r} were produced with fingerprint "
                        f"{existing.get('fingerprint')!r}, but the current configuration has "
                        f"fingerprint {fingerprint!r}. Refusing to mix incompatible artifacts "
                        "(different projection seed, checkpoints, feature type, rollout "
                        "counts, target objective, or optimizer state). Use a fresh "
                        "output_dir or matching configuration."
                    )
                self.manifest = existing
            else:
                # Fresh run requested: discard stale partial artifacts.
                for fn in os.listdir(self.dir):
                    if fn.startswith("features_") or fn == "manifest.json":
                        os.remove(os.path.join(self.dir, fn))
        self._save_manifest()

    # -- internals -------------------------------------------------------------------
    def _shard_key(self, split: str, checkpoint_id: str) -> str:
        return f"{split}::{checkpoint_id}"

    def _shard_path(self, split: str, checkpoint_id: str) -> str:
        safe_ckpt = checkpoint_id.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.dir, f"features_{split}_{safe_ckpt}.pt")

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    # -- public API ------------------------------------------------------------------
    def completed_indices(self, split: str, checkpoint_id: str) -> set[int]:
        return set(self.manifest["completed"].get(self._shard_key(split, checkpoint_id), []))

    def load_shard(self, split: str, checkpoint_id: str) -> dict[int, dict]:
        path = self._shard_path(split, checkpoint_id)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
        return {}

    def save_shard(self, split: str, checkpoint_id: str, features: dict[int, dict]):
        torch.save(features, self._shard_path(split, checkpoint_id))
        key = self._shard_key(split, checkpoint_id)
        self.manifest["completed"][key] = sorted(int(i) for i in features.keys())
        self._save_manifest()


# --------------------------------------------------------------------------------------
# Generic gradient feature extractor
# --------------------------------------------------------------------------------------
class GradientFeatureExtractor:
    """Backend-driven extractor: rollouts -> loss -> grads -> transform -> project -> average.

    Per example:
      1. Generate ``M`` on-policy rollouts (seeded deterministically).
      2. Backprop the backend loss once per rollout.
      3. Transform each raw gradient per the feature type (candidates only; targets stay
         raw), streaming parameter-by-parameter into CountSketch.
      4. Average the unnormalized projected vectors, then normalize once.
    """

    def __init__(
        self,
        backend: DistillationGradientBackend,
        config: LESSOPSDSelectionConfig,
        checkpoint_id: str,
        opt_view: AdamWStateView | None = None,
    ):
        self.backend = backend
        self.config = config
        self.checkpoint_id = checkpoint_id
        self.opt_view = opt_view

        if config.feature_type in OPTIMIZER_AWARE_FEATURE_TYPES and opt_view is None:
            raise OptimizerStateError(
                f"feature_type={config.feature_type!r} requires AdamW optimizer state for "
                f"checkpoint {checkpoint_id!r}, but none was loaded."
            )

    def _extract_one(self, raw_example: dict, example_index: int, split: str) -> dict:
        cfg = self.config
        model = self.backend.student_model
        model.train()

        if split == "candidate":
            num_rollouts = cfg.num_candidate_rollouts
        else:
            num_rollouts = cfg.num_target_rollouts
            if not self.backend.target_requires_rollout(cfg.target_objective):
                num_rollouts = 1  # deterministic objective: extra rollouts are identical

        z_list: list[torch.Tensor] = []
        for m in range(num_rollouts):
            seed = derive_rollout_seed(
                cfg.rollout_seed, self.checkpoint_id, example_index, m, split
            )
            if split == "candidate":
                prepared = self.backend.prepare_candidate_example(raw_example, seed)
                loss = self.backend.compute_candidate_loss(prepared)
            else:
                prepared = self.backend.prepare_target_example(
                    raw_example, cfg.target_objective, seed
                )
                loss = self.backend.compute_target_loss(prepared, cfg.target_objective)
            if isinstance(loss, tuple):
                loss = loss[0]

            model.zero_grad(set_to_none=True)
            loss.backward()

            if split == "candidate":
                named = iter_transformed_grad_tensors(model, cfg.feature_type, self.opt_view)
            else:
                # LESS asymmetry: target features are always raw loss gradients.
                named = iter_trainable_grad_tensors(model)
            z = count_sketch_project_named_tensors(named, cfg.projection_dim, cfg.projection_seed)
            z_list.append(z)

            model.zero_grad(set_to_none=True)
            del prepared, loss

        u, rho, invalid = aggregate_rollout_projections(z_list)
        del z_list
        return {"u": u, "rho": rho, "invalid": invalid}

    def extract_split_features(
        self,
        dataset,
        indices: list[int],
        split: str,
        store: FeatureStore | None = None,
    ) -> dict[int, dict]:
        """Extract features for ``indices``; resumable when ``store`` is provided."""
        cfg = self.config
        features: dict[int, dict] = {}
        done: set[int] = set()
        if store is not None:
            features = store.load_shard(split, self.checkpoint_id)
            done = store.completed_indices(split, self.checkpoint_id)
            if done:
                print(
                    f"[less-opsd] resume: {len(done)} {split} examples already complete for "
                    f"checkpoint {self.checkpoint_id}"
                )

        pending = [i for i in indices if int(i) not in done]
        n_invalid = sum(1 for rec in features.values() if rec.get("invalid"))
        since_save = 0
        for idx in tqdm(pending, desc=f"[{self.checkpoint_id}] {split} grads"):
            rec = self._extract_one(dataset[int(idx)], int(idx), split)
            if rec["invalid"]:
                n_invalid += 1
            features[int(idx)] = rec
            since_save += 1
            if store is not None and since_save >= cfg.save_every:
                store.save_shard(split, self.checkpoint_id, features)
                since_save = 0
                print(
                    f"[less-opsd] saved partial {split} features "
                    f"({len(features)}/{len(indices)}) for checkpoint {self.checkpoint_id}"
                )
        if store is not None:
            store.save_shard(split, self.checkpoint_id, features)
        if n_invalid:
            print(
                f"[less-opsd] WARNING: {n_invalid} {split} examples produced invalid "
                f"(non-finite or zero-norm) features at checkpoint {self.checkpoint_id}"
            )
        return features


def compute_group_target_feature(target_features: dict[int, dict]) -> torch.Tensor:
    """Normalize the mean of valid per-example target features (single target group)."""
    valid = [rec["u"] for rec in target_features.values() if not rec.get("invalid")]
    if not valid:
        raise RuntimeError(
            "All target examples produced invalid (non-finite or zero-norm) gradient "
            "features; cannot build a target feature."
        )
    return normalize_feature(torch.stack(valid, dim=0).mean(dim=0))


# --------------------------------------------------------------------------------------
# Multi-checkpoint driver
# --------------------------------------------------------------------------------------
def run_multicheckpoint_selection(
    backend_loader,  # (checkpoint_id, checkpoint_path|None) -> (backend, AdamWStateView|None)
    checkpoints: list[tuple[str, str | None]],
    checkpoint_weights: list[float],
    candidate_dataset,
    candidate_indices: list[int],
    target_dataset,
    target_indices: list[int],
    config: LESSOPSDSelectionConfig,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    """Full v2 selection: per-checkpoint features + cosines, then weighted aggregation.

    ``backend_loader`` is called once per checkpoint and must return a backend whose
    student model has that checkpoint's weights loaded (plus the optimizer view when the
    feature type needs one). Checkpoints are processed one at a time; only one model is
    ever resident.
    """
    validate_selection_config(config)
    if not candidate_indices:
        raise SelectionConfigError("Candidate set is empty.")
    if not target_indices:
        raise SelectionConfigError(
            "Target set is empty; LESS-OPSD requires target examples to score against."
        )

    extra = dict(extra_metadata or {})
    extra.setdefault("candidate_indices_digest", indices_digest(candidate_indices))
    extra.setdefault("target_indices_digest", indices_digest(target_indices))
    fingerprint = config_fingerprint(config, extra)
    store = FeatureStore(config.output_dir, fingerprint, resume=config.resume)

    n = len(candidate_indices)
    k_ckpts = len(checkpoints)
    per_ckpt_scores = torch.full((k_ckpts, n), float("nan"), dtype=torch.float32)
    per_ckpt_rho = torch.full((k_ckpts, n), float("nan"), dtype=torch.float32)
    target_feature_paths: dict[str, str] = {}
    candidate_features_last: dict[int, dict] | None = None
    backend_meta: dict = {}
    optimizer_meta: dict[str, dict] = {}

    for k, (ckpt_id, ckpt_path) in enumerate(checkpoints):
        print(
            f"[less-opsd] checkpoint {k + 1}/{k_ckpts}: {ckpt_id} "
            f"(feature_type={config.feature_type}, target_objective={config.target_objective}, "
            f"rollouts: cand={config.num_candidate_rollouts} tgt={config.num_target_rollouts})"
        )
        backend, opt_view = backend_loader(ckpt_id, ckpt_path)
        backend_meta = backend.metadata()
        if opt_view is not None:
            optimizer_meta[ckpt_id] = opt_view.metadata()
        extractor = GradientFeatureExtractor(backend, config, ckpt_id, opt_view)

        print(f"[less-opsd] [{ckpt_id}] extracting target features ({len(target_indices)})")
        target_features = extractor.extract_split_features(
            target_dataset, target_indices, "target", store
        )
        u_target = compute_group_target_feature(target_features)

        os.makedirs(config.output_dir, exist_ok=True)
        safe_ckpt = ckpt_id.replace(os.sep, "_").replace("/", "_")
        tpath = os.path.join(config.output_dir, f"target_feature_{safe_ckpt}.pt")
        torch.save(u_target, tpath)
        target_feature_paths[ckpt_id] = tpath

        print(f"[less-opsd] [{ckpt_id}] extracting candidate features ({len(candidate_indices)})")
        candidate_features = extractor.extract_split_features(
            candidate_dataset, candidate_indices, "candidate", store
        )
        candidate_features_last = candidate_features

        for j, idx in enumerate(candidate_indices):
            rec = candidate_features[int(idx)]
            per_ckpt_rho[k, j] = rec["rho"]
            if not rec["invalid"]:
                per_ckpt_scores[k, j] = float(rec["u"] @ u_target)

        if config.save_candidate_features:
            cpath = os.path.join(config.output_dir, f"candidate_features_{safe_ckpt}.pt")
            torch.save(candidate_features, cpath)

        if opt_view is not None:
            opt_view.state_by_name.clear()

    final_scores, valid_mask = aggregate_checkpoint_scores(per_ckpt_scores, checkpoint_weights)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        print(f"[less-opsd] {n_invalid}/{n} candidates invalid at >=1 checkpoint (never selected)")

    selected_indices, _, k_sel = select_topk_by_score(
        final_scores,
        candidate_indices,
        selection_num_examples=config.selection_num_examples,
        selection_fraction=config.selection_fraction,
    )

    method_name = method_name_for(config.feature_type, k_ckpts)
    paths = save_selection_artifacts_v2(
        config=config,
        method_name=method_name,
        candidate_indices=candidate_indices,
        final_scores=final_scores,
        per_ckpt_scores=per_ckpt_scores,
        per_ckpt_rho=per_ckpt_rho,
        valid_mask=valid_mask,
        selected_indices=selected_indices,
        checkpoints=checkpoints,
        checkpoint_weights=checkpoint_weights,
        fingerprint=fingerprint,
        backend_metadata=backend_meta,
        optimizer_metadata=optimizer_meta or None,
        extra_metadata=extra,
        target_feature_paths=target_feature_paths,
    )
    print(f"[less-opsd] selected {len(selected_indices)}/{n} -> {paths['selected_indices']}")

    return {
        "selected_indices": selected_indices,
        "final_scores": final_scores,
        "per_checkpoint_scores": per_ckpt_scores,
        "per_checkpoint_rho": per_ckpt_rho,
        "valid_mask": valid_mask,
        "k": k_sel,
        "paths": paths,
        "fingerprint": fingerprint,
        "candidate_features": candidate_features_last,
    }


# --------------------------------------------------------------------------------------
# Artifact saving
# --------------------------------------------------------------------------------------
def save_selection_artifacts_v2(
    config: LESSOPSDSelectionConfig,
    method_name: str,
    candidate_indices: list[int],
    final_scores: torch.Tensor,
    per_ckpt_scores: torch.Tensor,
    per_ckpt_rho: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_indices: list[int],
    checkpoints: list[tuple[str, str | None]],
    checkpoint_weights: list[float],
    fingerprint: str,
    backend_metadata: dict | None = None,
    optimizer_metadata: dict | None = None,
    extra_metadata: dict | None = None,
    target_feature_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Persist v2 selection artifacts (selected indices, ranked scores, full config)."""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}
    selected_set = set(int(i) for i in selected_indices)

    metadata = {
        "method": method_name,
        "method_version": METHOD_VERSION,
        "reference_repo": REFERENCE_REPO,
        "feature_type": config.feature_type,
        "target_objective": config.target_objective,
        "num_candidate_rollouts": config.num_candidate_rollouts,
        "num_target_rollouts": config.num_target_rollouts,
        "checkpoints": [{"id": cid, "path": cpath} for cid, cpath in checkpoints],
        "checkpoint_weighting": config.checkpoint_weighting,
        "checkpoint_weights": list(checkpoint_weights),
        "projection_dim": config.projection_dim,
        "projection_seed": config.projection_seed,
        "rollout_seed": config.rollout_seed,
        "candidate_subset_seed": config.candidate_subset_seed,
        "target_subset_seed": config.target_subset_seed,
        "selection_fraction": config.selection_fraction,
        "selection_num_examples": config.selection_num_examples,
        "num_candidates": len(candidate_indices),
        "num_selected": len(selected_indices),
        "config_fingerprint": fingerprint,
        "backend": backend_metadata or {},
        "optimizer": optimizer_metadata or {},
    }
    metadata.update(extra_metadata or {})

    # selected_indices.json (same shape as v1; consumable by opsd_train.py)
    selected_path = os.path.join(output_dir, "selected_indices.json")
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(
            {"selected_indices": [int(i) for i in selected_indices], "metadata": metadata},
            f,
            indent=2,
        )
    paths["selected_indices"] = selected_path

    # scores.jsonl: ranked (invalid candidates last)
    finals = final_scores.tolist()
    order = sorted(
        range(len(candidate_indices)),
        key=lambda j: (finals[j] == float("-inf"), -finals[j] if finals[j] != float("-inf") else 0.0),
    )
    scores_path = os.path.join(output_dir, "scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for rank, j in enumerate(order, start=1):
            idx = int(candidate_indices[j])
            invalid = not bool(valid_mask[j])
            ckpt_scores = [
                (None if not torch.isfinite(per_ckpt_scores[k, j]) else float(per_ckpt_scores[k, j]))
                for k in range(per_ckpt_scores.shape[0])
            ]
            rho = [
                (None if not torch.isfinite(per_ckpt_rho[k, j]) else float(per_ckpt_rho[k, j]))
                for k in range(per_ckpt_rho.shape[0])
            ]
            row = {
                "original_index": idx,
                "final_score": (None if invalid else float(finals[j])),
                "checkpoint_scores": ckpt_scores,
                "rollout_consistency": rho,
                "selected": idx in selected_set,
                "rank": rank,
                "invalid": invalid,
            }
            f.write(json.dumps(row) + "\n")
    paths["scores"] = scores_path

    # selection_config.json
    config_path = os.path.join(output_dir, "selection_config.json")
    payload = dict(asdict(config))
    payload["method_name"] = method_name
    payload["method_version"] = METHOD_VERSION
    payload["reference_repo"] = REFERENCE_REPO
    payload["config_fingerprint"] = fingerprint
    payload["checkpoints"] = [{"id": cid, "path": cpath} for cid, cpath in checkpoints]
    payload["resolved_checkpoint_weights"] = list(checkpoint_weights)
    payload["backend_metadata"] = backend_metadata or {}
    payload["optimizer_metadata"] = optimizer_metadata or {}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    paths["selection_config"] = config_path

    if target_feature_paths:
        paths.update({f"target_feature[{k}]": v for k, v in target_feature_paths.items()})
        # Single-checkpoint convenience alias matching the v1 artifact name.
        if len(target_feature_paths) == 1:
            only = next(iter(target_feature_paths.values()))
            alias = os.path.join(output_dir, "target_feature.pt")
            torch.save(torch.load(only, map_location="cpu", weights_only=False), alias)
            paths["target_feature"] = alias

    return paths


# --------------------------------------------------------------------------------------
# Legacy v1 API (preserved verbatim in behavior)
# --------------------------------------------------------------------------------------
def _move_inputs_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    """Move only the tensor entries of a collator output to ``device`` (ints stay as-is)."""
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


class LESSOPSDGradientExtractor:
    """(v1, kept for backward compatibility) single-rollout raw-gradient extractor."""

    def __init__(self, trainer, projection_dim: int, seed: int):
        self.trainer = trainer
        self.model = trainer.model
        self.projection_dim = projection_dim
        self.seed = seed

    @torch.no_grad()
    def _collate(self, raw_example: dict) -> dict[str, Any]:
        return self.trainer.data_collator([raw_example])

    def extract_one_example_feature(self, raw_example: dict, example_index: int) -> torch.Tensor:
        trainer = self.trainer
        model = self.model

        model.train()
        device = trainer.accelerator.device

        inputs = self._collate(raw_example)
        inputs = _move_inputs_to_device(inputs, device)

        prepared_inputs, _, _ = trainer.prepare_on_policy_distillation_batch(model, inputs)

        loss = trainer.compute_loss(model, prepared_inputs)
        if isinstance(loss, tuple):
            loss = loss[0]

        model.zero_grad(set_to_none=True)
        loss.backward()

        feature = count_sketch_project_trainable_grads(model, self.projection_dim, self.seed)
        feature = normalize_feature(feature)

        model.zero_grad(set_to_none=True)
        return feature.detach().cpu()

    def extract_dataset_features(self, dataset, indices: list[int], split_name: str) -> torch.Tensor:
        features = []
        for idx in tqdm(indices, desc=f"LESS-OPSD grads [{split_name}]"):
            features.append(self.extract_one_example_feature(dataset[int(idx)], int(idx)))
        if not features:
            return torch.empty(0, self.projection_dim, dtype=torch.float32)
        return torch.stack(features, dim=0)


def compute_target_feature(extractor: LESSOPSDGradientExtractor, target_dataset, target_indices):
    """(v1) Average the per-example target features, then normalize."""
    target_features = extractor.extract_dataset_features(target_dataset, target_indices, "target")
    return normalize_feature(target_features.mean(dim=0))


def score_candidates(candidate_features: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
    """Cosine similarity assuming inputs are normalized (dot product)."""
    return candidate_features @ target_feature


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
    """(v1) Persist selection artifacts. Returns a mapping of artifact name -> path."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    selected_set = set(int(i) for i in selected_indices)

    selected_path = os.path.join(output_dir, "selected_indices.json")
    selected_payload = {
        "selected_indices": [int(i) for i in selected_indices],
        "metadata": {
            "method": method_name,
            "method_version": "1.0",
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

    config_path = os.path.join(output_dir, "selection_config.json")
    config_payload = dict(asdict(config))
    config_payload["method_name"] = method_name
    config_payload["reference_repo"] = REFERENCE_REPO
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    paths["selection_config"] = config_path

    if target_feature is not None:
        target_path = os.path.join(output_dir, "target_feature.pt")
        torch.save(target_feature.detach().cpu(), target_path)
        paths["target_feature"] = target_path

    if config.save_candidate_features and candidate_features is not None:
        cand_path = os.path.join(output_dir, "candidate_features.pt")
        torch.save(candidate_features.detach().cpu(), cand_path)
        paths["candidate_features"] = cand_path

    return paths


def run_less_opsd_selection(
    trainer,
    candidate_dataset,
    candidate_indices: list[int],
    target_dataset,
    target_indices: list[int],
    config: LESSOPSDSelectionConfig,
) -> dict[str, Any]:
    """(v1) Static single-rollout raw-gradient selection. Kept for backward compatibility."""
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
    method_name: str = RANDOM_METHOD_NAME,
    seed: int | None = None,
) -> dict[str, Any]:
    """Random-selection baseline emitting the same artifact format as LESS-OPSD.

    Also used (with ``method_name="warmup_random"`` and ``seed=warmup_subset_seed``) to
    create reproducible random warmup subsets whose artifacts are compatible with
    ``load_selected_indices``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed if seed is not None else config.seed))
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
        method_name=method_name,
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
