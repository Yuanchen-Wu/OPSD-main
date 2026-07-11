"""CLI for LESS-OPSD gradient-based data selection (v2: AdamW-aware, multi-rollout,
optionally multi-checkpoint).

Examples
--------
Cheap raw-gradient smoke test (base model, one implicit checkpoint)::

    python less_opsd_select.py \
      --model_name_or_path Qwen/Qwen3-0.6B \
      --feature_type raw_gradient \
      --num_candidate_rollouts 1 \
      --candidate_limit 32 --target_limit 8 \
      --selection_fraction 0.25 \
      --projection_dim 4096 \
      --output_dir outputs/less_opsd_selection/raw_smoke

AdamW-aware selection from a warmup checkpoint::

    python less_opsd_select.py \
      --model_name_or_path Qwen/Qwen3-0.6B \
      --feature_type adamw_candidate_update \
      --checkpoint_paths outputs/warmup/checkpoint-50 \
      --num_candidate_rollouts 2 --num_target_rollouts 2 \
      --candidate_limit 2000 --target_limit 128 \
      --selection_fraction 0.05 \
      --output_dir outputs/less_opsd_selection/adamw_run \
      --resume

Reproducible random warmup subset (no model needed; artifact is compatible with
``opsd_train.py --selected_indices_path``)::

    python less_opsd_select.py \
      --selection_method warmup_subset \
      --selection_num_examples 500 \
      --warmup_subset_seed 7 \
      --output_dir outputs/warmup_subset

Random baseline::

    python less_opsd_select.py --selection_method random \
      --candidate_limit 2000 --selection_fraction 0.05 \
      --output_dir outputs/less_opsd_selection/random_smoke

This script never calls ``trainer.train()`` and disables WandB by default. The candidate
gradient is the OPSD distillation gradient (via ``OPSDTrainer.compute_loss``), not a
supervised cross-entropy gradient. See ``docs/less_opsd_methodology.md`` and
``docs/less_opsd_usage.md``.
"""

from __future__ import annotations

import argparse
import os
import random

from less_opsd_selector import (
    LESSOPSDSelectionConfig,
    RANDOM_METHOD_NAME,
    WARMUP_RANDOM_METHOD_NAME,
    normalize_checkpoint_weights,
    run_multicheckpoint_selection,
    run_random_selection,
    validate_selection_config,
)


def parse_args():
    p = argparse.ArgumentParser(description="LESS-OPSD gradient-based data selection.")

    # Method
    p.add_argument(
        "--selection_method",
        type=str,
        default="less_opsd",
        choices=["less_opsd", "random", "warmup_subset"],
        help="'less_opsd' = gradient alignment selection; 'random' = random baseline; "
        "'warmup_subset' = save a reproducible random warmup subset (no model needed).",
    )

    # Data
    p.add_argument("--dataset_name", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument(
        "--target_dataset_name",
        type=str,
        default=None,
        help="Optional separate dataset for the target/validation set. If unset, a "
        "deterministic, non-overlapping subset of the candidate dataset is used.",
    )
    p.add_argument("--target_dataset_split", type=str, default="train")
    p.add_argument("--candidate_limit", type=int, default=None)
    p.add_argument("--target_limit", type=int, default=None)
    p.add_argument(
        "--allow_candidate_target_overlap",
        action="store_true",
        default=False,
        help="Allow candidate and target indices to overlap (default: disjoint).",
    )

    # Feature construction (v2)
    p.add_argument(
        "--feature_type",
        type=str,
        default="raw_gradient",
        choices=["raw_gradient", "adamw_candidate_update", "adamw_fixed_preconditioner"],
        help="Candidate feature: raw OPSD gradient (v1 behavior), hypothetical AdamW "
        "candidate update, or fixed Adam preconditioner. AdamW modes require "
        "--checkpoint_paths with saved optimizer state.",
    )
    p.add_argument(
        "--target_objective",
        type=str,
        default="opsd",
        choices=["opsd", "reference_ce"],
        help="Target gradient objective: 'opsd' (default, OPSD loss with fresh target "
        "rollouts) or 'reference_ce' (supervised CE on the reference solution).",
    )
    p.add_argument("--num_candidate_rollouts", type=int, default=1)
    p.add_argument("--num_target_rollouts", type=int, default=1)

    # Checkpoints (v2)
    p.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more HF Trainer checkpoint dirs (LoRA adapter + optimizer.pt). "
        "When omitted, the base initialization is used (raw_gradient only).",
    )
    p.add_argument(
        "--checkpoint_weighting",
        type=str,
        default="uniform",
        choices=["uniform", "learning_rate", "explicit"],
    )
    p.add_argument("--checkpoint_weights", type=float, nargs="+", default=None)

    # Selection
    p.add_argument("--selection_fraction", type=float, default=None)
    p.add_argument("--selection_num_examples", type=int, default=None)
    p.add_argument("--projection_dim", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42, help="Legacy umbrella seed.")
    p.add_argument("--projection_seed", type=int, default=None)
    p.add_argument("--rollout_seed", type=int, default=None)
    p.add_argument("--candidate_subset_seed", type=int, default=None)
    p.add_argument("--target_subset_seed", type=int, default=None)
    p.add_argument("--warmup_subset_seed", type=int, default=None)
    p.add_argument("--gradient_batch_size", type=int, default=1)
    p.add_argument("--save_candidate_features", action="store_true", default=False)
    p.add_argument("--save_every", type=int, default=25,
                   help="Save partial (resumable) features every N examples.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from partial artifacts in output_dir (fingerprint-checked).")
    p.add_argument("--output_dir", type=str, default="outputs/less_opsd_selection")

    # Model (only used for selection_method=less_opsd)
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--model_revision", type=str, default=None)
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--trust_remote_code", action="store_true", default=False)

    # LoRA (gradients are taken over trainable params; LoRA keeps this cheap)
    p.add_argument("--use_peft", action="store_true", default=True)
    p.add_argument("--no_peft", dest="use_peft", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # OPSD loss / generation knobs (mirror opsd_train.py defaults so selection gradients
    # match training gradients).
    p.add_argument("--fixed_teacher", action="store_true", default=False)
    p.add_argument("--use_ema_teacher", action="store_true", default=False)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--reason_first", action="store_true", default=False)
    p.add_argument("--use_tinker_loss", action="store_true", default=False)
    p.add_argument("--top_k_loss", type=int, default=0)
    p.add_argument("--jsd_token_clip", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--lmbda", type=float, default=1.0)
    p.add_argument("--student_thinking", action="store_true", default=False)
    p.add_argument("--teacher_thinking", action="store_true", default=True)
    p.add_argument("--no_teacher_thinking", dest="teacher_thinking", action="store_false")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--max_length", type=int, default=2048)

    args = p.parse_args()
    # Legacy default: fraction 0.05 when neither size argument is given.
    if args.selection_fraction is None and args.selection_num_examples is None:
        args.selection_fraction = 0.05
    return args


def build_indices(num_candidates_available, num_targets_available, args, same_dataset):
    """Build deterministic candidate/target index lists.

    When candidate and target come from the same dataset, the two index sets are disjoint
    by default (unless --allow_candidate_target_overlap is set).
    """
    subset_seed = args.candidate_subset_seed if args.candidate_subset_seed is not None else args.seed
    rng = random.Random(subset_seed)

    target_limit = args.target_limit
    if target_limit is None and args.selection_method == "less_opsd":
        # Need a target set for gradient alignment; pick a sane default.
        target_limit = min(128, num_targets_available)

    if same_dataset:
        perm = list(range(num_candidates_available))
        rng.shuffle(perm)

        target_indices = sorted(perm[: (target_limit or 0)])
        if args.allow_candidate_target_overlap:
            remaining = perm
        else:
            remaining = perm[(target_limit or 0):]

        candidate_pool = remaining if args.candidate_limit is None else remaining[: args.candidate_limit]
        candidate_indices = sorted(candidate_pool)
    else:
        cand_perm = list(range(num_candidates_available))
        rng.shuffle(cand_perm)
        candidate_indices = sorted(
            cand_perm if args.candidate_limit is None else cand_perm[: args.candidate_limit]
        )

        tgt_seed = args.target_subset_seed if args.target_subset_seed is not None else args.seed
        tgt_rng = random.Random(tgt_seed)
        tgt_perm = list(range(num_targets_available))
        tgt_rng.shuffle(tgt_perm)
        target_indices = sorted(tgt_perm[: (target_limit or 0)])

    return candidate_indices, target_indices


def make_config(args) -> LESSOPSDSelectionConfig:
    return LESSOPSDSelectionConfig(
        feature_type=args.feature_type,
        target_objective=args.target_objective,
        num_candidate_rollouts=args.num_candidate_rollouts,
        num_target_rollouts=args.num_target_rollouts,
        checkpoint_paths=list(args.checkpoint_paths or []),
        checkpoint_weighting=args.checkpoint_weighting,
        checkpoint_weights=args.checkpoint_weights,
        projection_dim=args.projection_dim,
        seed=args.seed,
        projection_seed=args.projection_seed,
        rollout_seed=args.rollout_seed,
        candidate_subset_seed=args.candidate_subset_seed,
        target_subset_seed=args.target_subset_seed,
        warmup_subset_seed=args.warmup_subset_seed,
        candidate_limit=args.candidate_limit,
        target_limit=args.target_limit,
        selection_fraction=args.selection_fraction,
        selection_num_examples=args.selection_num_examples,
        gradient_batch_size=args.gradient_batch_size,
        save_every=args.save_every,
        resume=args.resume,
        normalize_per_example=True,
        score_metric="dot",
        output_dir=args.output_dir,
        save_candidate_features=args.save_candidate_features,
        method=args.selection_method,
    )


def run_random(args, method_name=RANDOM_METHOD_NAME, seed=None):
    """Random baseline / warmup subset: no model, no dataset gradients required."""
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    n = len(dataset)
    candidate_indices, _ = build_indices(n, n, args, same_dataset=True)

    config = make_config(args)
    result = run_random_selection(candidate_indices, config, method_name=method_name, seed=seed)
    print(
        f"[{method_name}] selected {len(result['selected_indices'])}/{len(candidate_indices)} "
        f"-> {result['paths']['selected_indices']}"
    )
    return result


def build_trainer(args, train_dataset):
    """Construct an OPSDTrainer suitable for gradient extraction (no training)."""
    import torch
    from transformers import AutoTokenizer
    from trl.experimental.gold import GOLDConfig

    from opsd_trainer import OPSDTrainer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Minimal training config; we never call trainer.train(). use_vllm=False so generation
    # goes through model.generate (no vLLM engine needed for selection).
    training_args = GOLDConfig(
        output_dir=os.path.join(args.output_dir, "_trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        max_length=args.max_length,
        beta=args.beta,
        lmbda=args.lmbda,
        use_vllm=False,
        report_to=[],
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
    )
    training_args.model_init_kwargs = dict(
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype,
        use_cache=True,
    )
    training_args.presence_penalty = 0.0

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

    trainer = OPSDTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
        use_thinking_machines_loss=args.use_tinker_loss,
        fixed_teacher=args.fixed_teacher,
        reason_first=args.reason_first,
        top_k_loss=args.top_k_loss if args.top_k_loss > 0 else None,
        jsd_token_clip=args.jsd_token_clip if args.jsd_token_clip > 0 else None,
        use_ema_teacher=args.use_ema_teacher,
        ema_decay=args.ema_decay,
        student_thinking=args.student_thinking,
        teacher_thinking=args.teacher_thinking,
    )
    return trainer


def run_less_opsd(args):
    from datasets import load_dataset

    from less_opsd_backend import (
        OPSDTrainerBackend,
        build_optimizer_view_for_checkpoint,
        load_lora_checkpoint_into_trainer,
        recover_checkpoint_learning_rate,
    )
    from less_opsd_selector import OPTIMIZER_AWARE_FEATURE_TYPES

    config = make_config(args)
    validate_selection_config(config)

    candidate_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    n_candidates = len(candidate_dataset)

    if args.target_dataset_name is not None:
        target_dataset = load_dataset(args.target_dataset_name, split=args.target_dataset_split)
        same_dataset = False
        n_targets = len(target_dataset)
    else:
        target_dataset = candidate_dataset
        same_dataset = True
        n_targets = n_candidates

    candidate_indices, target_indices = build_indices(
        n_candidates, n_targets, args, same_dataset=same_dataset
    )

    if len(target_indices) == 0:
        raise ValueError(
            "LESS-OPSD selection requires a non-empty target set. Provide --target_limit "
            "(and a dataset large enough) or --target_dataset_name."
        )

    # Checkpoint list: explicit paths, or the base initialization as a single pseudo
    # checkpoint (raw_gradient only; validate_selection_config rejects AdamW modes here).
    if config.checkpoint_paths:
        checkpoints = [(os.path.basename(os.path.normpath(p)) or p, p)
                       for p in config.checkpoint_paths]
    else:
        checkpoints = [("base", None)]

    learning_rates = None
    if config.checkpoint_weighting == "learning_rate":
        learning_rates = [recover_checkpoint_learning_rate(p) for _, p in checkpoints
                          if p is not None]
        if len(learning_rates) != len(checkpoints):
            raise ValueError(
                "checkpoint_weighting='learning_rate' requires real checkpoint paths "
                "(the base initialization has no recorded learning rate)."
            )
    weights = normalize_checkpoint_weights(
        config.checkpoint_weighting, len(checkpoints), config.checkpoint_weights, learning_rates
    )

    print(
        f"[less_opsd] candidates={len(candidate_indices)} targets={len(target_indices)} "
        f"feature_type={config.feature_type} target_objective={config.target_objective} "
        f"checkpoints={[c[0] for c in checkpoints]} weights={[round(w, 4) for w in weights]} "
        f"projection_dim={config.projection_dim} projection_seed={config.projection_seed} "
        f"rollout_seed={config.rollout_seed}"
    )

    # One trainer/model instance; per checkpoint we swap in the LoRA adapter weights and
    # (for AdamW modes) load the matching optimizer state. Only one model is resident.
    trainer = build_trainer(args, candidate_dataset)
    backend = OPSDTrainerBackend(trainer)

    def backend_loader(ckpt_id, ckpt_path):
        if ckpt_path is not None:
            load_lora_checkpoint_into_trainer(trainer, ckpt_path)
        opt_view = None
        if config.feature_type in OPTIMIZER_AWARE_FEATURE_TYPES:
            opt_view = build_optimizer_view_for_checkpoint(trainer, ckpt_path)
        return backend, opt_view

    extra_metadata = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "target_dataset_name": args.target_dataset_name,
        "target_dataset_split": args.target_dataset_split,
        "candidate_indices": candidate_indices if len(candidate_indices) <= 10000 else None,
        "target_indices": target_indices,
    }

    result = run_multicheckpoint_selection(
        backend_loader=backend_loader,
        checkpoints=checkpoints,
        checkpoint_weights=weights,
        candidate_dataset=candidate_dataset,
        candidate_indices=candidate_indices,
        target_dataset=target_dataset,
        target_indices=target_indices,
        config=config,
        extra_metadata=extra_metadata,
    )
    return result


def main():
    args = parse_args()

    # Selection never logs to WandB.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.selection_method == "random":
        run_random(args)
    elif args.selection_method == "warmup_subset":
        seed = args.warmup_subset_seed if args.warmup_subset_seed is not None else args.seed
        run_random(args, method_name=WARMUP_RANDOM_METHOD_NAME, seed=seed)
    else:
        run_less_opsd(args)


if __name__ == "__main__":
    main()
