"""CLI for LESS-OPSD gradient-based data selection (static cached-rollout MVP).

Examples
--------
LESS-OPSD selection on a tiny candidate pool::

    python less_opsd_select.py \
      --model_name_or_path Qwen/Qwen3-0.6B \
      --dataset_name siyanzhao/Openthoughts_math_30k_opsd \
      --dataset_split train \
      --candidate_limit 2000 \
      --target_limit 128 \
      --selection_fraction 0.05 \
      --projection_dim 4096 \
      --output_dir outputs/less_opsd_selection/test_run \
      --seed 42

Random baseline (no model needed)::

    python less_opsd_select.py \
      --selection_method random \
      --candidate_limit 2000 \
      --selection_fraction 0.05 \
      --output_dir outputs/less_opsd_selection/random_smoke

This script never calls ``trainer.train()`` and disables WandB by default.

The candidate gradient is the OPSD distillation gradient (via
``OPSDTrainer.compute_loss``), not a supervised cross-entropy gradient. See
``docs/less_opsd_methodology.md``.
"""

from __future__ import annotations

import argparse
import os
import random

from less_opsd_selector import (
    LESSOPSDSelectionConfig,
    run_less_opsd_selection,
    run_random_selection,
)


def parse_args():
    p = argparse.ArgumentParser(description="LESS-OPSD gradient-based data selection.")

    # Method
    p.add_argument(
        "--selection_method",
        type=str,
        default="less_opsd",
        choices=["less_opsd", "random"],
        help="'less_opsd' = gradient alignment selection; 'random' = random baseline.",
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

    # Selection
    p.add_argument("--selection_fraction", type=float, default=0.05)
    p.add_argument("--selection_num_examples", type=int, default=None)
    p.add_argument("--projection_dim", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_batch_size", type=int, default=1)
    p.add_argument("--save_candidate_features", action="store_true", default=False)
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

    return p.parse_args()


def build_indices(num_candidates_available, num_targets_available, args, same_dataset):
    """Build deterministic candidate/target index lists.

    When candidate and target come from the same dataset, the two index sets are disjoint
    by default (unless --allow_candidate_target_overlap is set).
    """
    rng = random.Random(args.seed)

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

        tgt_perm = list(range(num_targets_available))
        rng.shuffle(tgt_perm)
        target_indices = sorted(tgt_perm[: (target_limit or 0)])

    return candidate_indices, target_indices


def make_config(args) -> LESSOPSDSelectionConfig:
    return LESSOPSDSelectionConfig(
        projection_dim=args.projection_dim,
        seed=args.seed,
        candidate_limit=args.candidate_limit,
        target_limit=args.target_limit,
        selection_fraction=args.selection_fraction,
        selection_num_examples=args.selection_num_examples,
        gradient_batch_size=args.gradient_batch_size,
        normalize_per_example=True,
        score_metric="dot",
        output_dir=args.output_dir,
        save_candidate_features=args.save_candidate_features,
        method=args.selection_method,
    )


def run_random(args):
    """Random baseline: no model, no dataset gradients required."""
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    n = len(dataset)
    candidate_indices, _ = build_indices(n, n, args, same_dataset=True)

    config = make_config(args)
    result = run_random_selection(candidate_indices, config)
    print(
        f"[random] selected {len(result['selected_indices'])}/{len(candidate_indices)} "
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

    print(
        f"[less_opsd] candidates={len(candidate_indices)} targets={len(target_indices)} "
        f"projection_dim={args.projection_dim} seed={args.seed}"
    )

    # The trainer must see the full dataset so absolute indices map correctly.
    trainer = build_trainer(args, candidate_dataset)

    config = make_config(args)
    result = run_less_opsd_selection(
        trainer=trainer,
        candidate_dataset=candidate_dataset,
        candidate_indices=candidate_indices,
        target_dataset=target_dataset,
        target_indices=target_indices,
        config=config,
    )
    print(
        f"[less_opsd] selected {len(result['selected_indices'])}/{len(candidate_indices)} "
        f"-> {result['paths']['selected_indices']}"
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
    else:
        run_less_opsd(args)


if __name__ == "__main__":
    main()
