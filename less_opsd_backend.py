"""OPSD self-distillation backend for LESS-OPSD gradient-based selection.

This module implements the ``DistillationGradientBackend`` protocol (see
``less_opsd_selector.py``) on top of the repository's existing ``OPSDTrainer``:

  * candidate loss = the *existing* OPSD distillation loss over a fresh on-policy student
    rollout (``prepare_on_policy_distillation_batch`` -> ``compute_loss``);
  * target objective ``opsd`` = the same loss on target examples;
  * target objective ``reference_ce`` = supervised cross-entropy on the target example's
    reference solution (target side only — candidate features always use the OPSD loss).

It also contains the checkpoint utilities: loading a Hugging Face Trainer checkpoint's
LoRA adapter weights into the trainer's PEFT model, loading the matching AdamW optimizer
state, and recovering the logged learning rate for ``checkpoint_weighting=learning_rate``.

The generic selector never reads dataset fields directly; only this backend knows about
``problem`` / ``solution`` columns and teacher construction.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from less_opsd_selector import (
    AdamWStateView,
    OptimizerStateError,
    load_adamw_state_view_from_checkpoint,
)


def _move_inputs_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()
    }


class OPSDTrainerBackend:
    """Backend that adapts an ``OPSDTrainer`` to the generic selection machinery.

    Responsibilities: build student/teacher prompts (via the trainer's collator),
    generate on-policy rollouts, run teacher scoring, and compute losses. All existing
    trainer options (fixed_teacher, EMA teacher, reason_first, thinking flags, JSD/tinker
    loss, top_k_loss, jsd_token_clip) are honored because the trainer's own code paths
    are reused unchanged.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    # -- protocol --------------------------------------------------------------------
    @property
    def student_model(self):
        return self.trainer.model

    def prepare_candidate_example(self, raw_example: dict, generation_seed: int) -> Any:
        """Collate one example and generate an on-policy rollout (seeded)."""
        trainer = self.trainer
        model = trainer.model
        model.train()

        with torch.no_grad():
            inputs = trainer.data_collator([raw_example])
        inputs = _move_inputs_to_device(inputs, trainer.accelerator.device)

        # Seed the sampling RNG so rollout m of example i at checkpoint k is reproducible
        # (to the extent the generation backend/hardware allows).
        torch.manual_seed(int(generation_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(generation_seed))

        prepared, _, _ = trainer.prepare_on_policy_distillation_batch(model, inputs)
        return prepared

    def compute_candidate_loss(self, prepared_example: Any) -> torch.Tensor:
        loss = self.trainer.compute_loss(self.trainer.model, prepared_example)
        return loss[0] if isinstance(loss, tuple) else loss

    def prepare_target_example(
        self, raw_example: dict, objective: str, generation_seed: int
    ) -> Any:
        if objective == "opsd":
            return self.prepare_candidate_example(raw_example, generation_seed)
        if objective == "reference_ce":
            return self._prepare_reference_ce_batch(raw_example)
        raise ValueError(f"Unknown target objective {objective!r}.")

    def compute_target_loss(self, prepared_example: Any, objective: str) -> torch.Tensor:
        if objective == "opsd":
            return self.compute_candidate_loss(prepared_example)
        if objective == "reference_ce":
            model = self.trainer.model
            out = model(
                input_ids=prepared_example["input_ids"],
                attention_mask=prepared_example["attention_mask"],
                labels=prepared_example["labels"],
            )
            return out.loss
        raise ValueError(f"Unknown target objective {objective!r}.")

    def target_requires_rollout(self, objective: str) -> bool:
        return objective == "opsd"

    def metadata(self) -> dict:
        trainer = self.trainer
        peft_meta = None
        model = trainer.model
        peft_cfg = getattr(model, "peft_config", None)
        if peft_cfg:
            cfg = peft_cfg.get("default")
            if cfg is not None:
                peft_meta = {
                    "r": getattr(cfg, "r", None),
                    "lora_alpha": getattr(cfg, "lora_alpha", None),
                    "target_modules": sorted(getattr(cfg, "target_modules", []) or []),
                }
        if trainer.use_ema_teacher:
            teacher_mode = "ema"
        elif trainer.fixed_teacher:
            teacher_mode = "fixed_base_model"
        else:
            teacher_mode = "dynamic_same_model"
        return {
            "backend": "opsd_self_distillation",
            "teacher_mode": teacher_mode,
            "fixed_teacher": bool(trainer.fixed_teacher),
            "use_ema_teacher": bool(trainer.use_ema_teacher),
            "reason_first": bool(trainer.reason_first),
            "use_thinking_machines_loss": bool(trainer.use_thinking_machines_loss),
            "top_k_loss": trainer.top_k_loss,
            "jsd_token_clip": trainer.jsd_token_clip,
            "beta": trainer.beta,
            "temperature": trainer.temperature,
            "model_name_or_path": trainer.model_name_or_path,
            "lora": peft_meta,
        }

    # -- reference_ce internals --------------------------------------------------------
    def _prepare_reference_ce_batch(self, raw_example: dict) -> dict[str, torch.Tensor]:
        """Supervised CE batch: student-style prompt + reference solution as the label.

        Mirrors the collator's student prompt format so the CE target gradient lives in
        the same prompt distribution the student is trained on. The prompt tokens are
        masked with -100; only reference-solution tokens contribute to the loss.
        """
        trainer = self.trainer
        tokenizer = trainer.processing_class
        collator = trainer.data_collator

        problem = raw_example["problem"]
        solution = raw_example["solution"]

        student_user_message = (
            f"Problem: {problem}\n\nPlease reason step by step, and put your final answer "
            f"within \\boxed{{}}."
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": student_user_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=getattr(collator, "student_thinking", False),
        )

        max_length = getattr(collator, "max_length", 2048)
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        completion_text = solution + (tokenizer.eos_token or "")
        completion_ids = tokenizer(
            completion_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_length - len(prompt_ids)),
        )["input_ids"]

        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
        labels = input_ids.clone()
        labels[0, : len(prompt_ids)] = -100
        attention_mask = torch.ones_like(input_ids)

        device = trainer.accelerator.device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }


# --------------------------------------------------------------------------------------
# Checkpoint utilities
# --------------------------------------------------------------------------------------
def load_lora_checkpoint_into_trainer(trainer, checkpoint_path: str) -> None:
    """Load a HF Trainer checkpoint's LoRA adapter weights into the trainer's PEFT model.

    The trainer's model must already be a PEFT model with the same LoRA configuration the
    checkpoint was trained with; only adapter weights are swapped (the frozen base model
    is untouched, which also preserves fixed-teacher behavior).
    """
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights

    model = trainer.model
    if not hasattr(model, "peft_config"):
        raise RuntimeError(
            "Trainer model is not a PEFT model; loading a LoRA checkpoint requires the "
            "selection trainer to be built with use_peft (matching the warmup run)."
        )
    adapter_weights = load_peft_weights(checkpoint_path)
    result = set_peft_model_state_dict(model, adapter_weights)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} does not match the current LoRA configuration; "
            f"unexpected keys e.g. {list(unexpected)[:3]}."
        )


def build_optimizer_view_for_checkpoint(trainer, checkpoint_path: str) -> AdamWStateView:
    """Create the trainer's AdamW and load the checkpoint's optimizer state into it.

    Uses ``trainer.create_optimizer()`` so the parameter-group structure matches what the
    HF Trainer used during warmup training — the only safe way to map serialized optimizer
    parameter IDs back onto parameter objects. The state view keeps all moments on CPU.
    """
    trainer.optimizer = None  # force a fresh optimizer with the standard group structure
    optimizer = trainer.create_optimizer()
    view = load_adamw_state_view_from_checkpoint(trainer.model, optimizer, checkpoint_path)

    # Free the optimizer itself; the CPU view holds everything the transforms need.
    trainer.optimizer = None
    del optimizer
    return view


def recover_checkpoint_learning_rate(checkpoint_path: str) -> float:
    """Recover the last logged learning rate from ``trainer_state.json``.

    Uses the ``learning_rate`` recorded in the checkpoint's ``trainer_state.json``
    ``log_history`` (the value the HF Trainer logged at or closest before this
    checkpoint's global step). Raises ``OptimizerStateError`` when unavailable — uniform
    weights are NOT silently substituted.
    """
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_path):
        raise OptimizerStateError(
            f"Cannot recover learning rate: {state_path!r} not found. "
            "checkpoint_weighting='learning_rate' requires HF Trainer checkpoints with "
            "trainer_state.json."
        )
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    lrs = [h["learning_rate"] for h in state.get("log_history", []) if "learning_rate" in h]
    if not lrs:
        raise OptimizerStateError(
            f"trainer_state.json at {checkpoint_path!r} has no logged learning_rate entries; "
            "cannot use checkpoint_weighting='learning_rate' (enable logging_steps during "
            "warmup, or use uniform/explicit weighting)."
        )
    return float(lrs[-1])
