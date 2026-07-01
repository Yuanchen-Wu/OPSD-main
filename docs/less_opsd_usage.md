# LESS-OPSD: Usage

Gradient-based data selection for OPSD on-policy self-distillation. See
[`less_opsd_methodology.md`](./less_opsd_methodology.md) for the conceptual background.

The selection gradient is the **OPSD distillation gradient** computed by
`OPSDTrainer.compute_loss` over a fresh on-policy student rollout — not a supervised
cross-entropy gradient.

---

## Pipeline overview

1. **Select** influential training examples with `less_opsd_select.py` → writes
   `selected_indices.json`.
2. **Train** OPSD on just that subset by passing
   `--selected_indices_path .../selected_indices.json` to `opsd_train.py`.

---

## 1. Tiny smoke LESS-OPSD selection

Runs the whole pipeline on a tiny pool with a small projection so it finishes quickly:

```bash
python less_opsd_select.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --candidate_limit 32 \
  --target_limit 8 \
  --selection_fraction 0.25 \
  --projection_dim 128 \
  --output_dir outputs/less_opsd_selection/smoke \
  --seed 42
```

This loads the model with a LoRA adapter, generates one on-policy rollout per candidate
and per target example, computes the OPSD loss + gradient, projects it with CountSketch,
scores candidates by cosine alignment with the averaged target gradient, and selects the
top 25%.

## 2. Larger LESS-OPSD selection

```bash
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
```

Useful flags (defaults mirror `opsd_train.py` so selection gradients match training):

- `--fixed_teacher`, `--use_ema_teacher`, `--ema_decay`
- `--reason_first`
- `--use_tinker_loss` (thinking-machines reverse-KL loss instead of JSD)
- `--top_k_loss`, `--jsd_token_clip`, `--beta`, `--lmbda`
- `--student_thinking`, `--teacher_thinking` / `--no_teacher_thinking`
- `--temperature`, `--top_p`, `--top_k`, `--max_completion_length`, `--max_length`
- `--lora_r`, `--lora_alpha`, `--lora_target_modules`, `--no_peft`
- `--selection_num_examples N` (overrides `--selection_fraction`)
- `--save_candidate_features` (also dumps `candidate_features.pt`, which can be large)

### Target set behavior

- If `--target_dataset_name` is **not** provided, a deterministic subset of the candidate
  dataset is used as the target set, and it is **disjoint** from the candidate indices by
  default.
- Pass `--allow_candidate_target_overlap` to permit overlap.
- Provide `--target_dataset_name`/`--target_dataset_split` to use a separate target set.

## 3. Random baseline

Same output format as LESS-OPSD, but selection is random (no model required, so it is
fast and cheap):

```bash
python less_opsd_select.py \
  --selection_method random \
  --candidate_limit 2000 \
  --selection_fraction 0.05 \
  --output_dir outputs/less_opsd_selection/random_smoke \
  --seed 42
```

## 4. Train OPSD on the selected subset

Pass the produced file to `opsd_train.py`. When `--selected_indices_path` is omitted,
training behavior is completely unchanged.

```bash
python opsd_train.py \
  --selected_indices_path outputs/less_opsd_selection/smoke/selected_indices.json \
  --selection_method less_opsd \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --attn_implementation sdpa \
  --torch_dtype bfloat16 --bf16 True \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --output_dir outputs/opsd_less_run \
  --run_config less_opsd_smoke \
  --max_steps 10 \
  --max_completion_length 256 \
  --max_length 2048 \
  --beta 0 --lmbda 1 \
  --use_peft --lora_r 16 --lora_alpha 32 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --fixed_teacher \
  --report_to none
```

`opsd_train.py` will print how many examples remain after selection, e.g.
`[LESS-OPSD] Using selected subset with 8 examples (...)`.

---

## Files saved by the selector

All under `--output_dir`:

| File | Always? | Contents |
|---|---|---|
| `selected_indices.json` | yes | `{"selected_indices": [...], "metadata": {...}}` — feed this to `opsd_train.py`. |
| `scores.jsonl` | yes | One JSON object per candidate: `{"original_index", "score", "rank", "selected"}`, ranked by descending score. |
| `selection_config.json` | yes | The full `LESSOPSDSelectionConfig` plus method name and reference repo. |
| `target_feature.pt` | LESS-OPSD only | The averaged, normalized projected target gradient (`[projection_dim]`). |
| `candidate_features.pt` | only with `--save_candidate_features` | All projected candidate features (`[num_candidates, projection_dim]`); can be large. |

`selected_indices.json` is also accepted as a bare list (`[1, 5, 10]`) by
`opsd_train.py`/`load_selected_indices`.

---

## How this differs from original LESS

LESS ([princeton-nlp/LESS](https://github.com/princeton-nlp/LESS), ICML 2024) is used here
as a **methodological reference only** — it is not vendored and is not a runtime
dependency. Key differences:

- **Gradient feature**: original LESS uses the supervised cross-entropy gradient
  `∇ CE(answer | instruction)`. LESS-OPSD uses the **OPSD distillation gradient**
  `∇ L_OPSD(problem, reference_solution, student_rollout)` from `OPSDTrainer.compute_loss`,
  evaluated on an **on-policy student rollout**.
- **Projection**: LESS uses TRAK random projectors (`fast_jl`/CUDA). LESS-OPSD uses a
  dependency-free **CountSketch** hashing projection (no dense projection matrix).
- **Scope**: LESS-OPSD's MVP is **static** (single model, single rollout per example, no
  Adam preconditioning, no multi-checkpoint aggregation, not online/adaptive). These are
  documented as future extensions in the methodology doc.
- **Trainable params**: like LESS, gradients are taken only over trainable (LoRA)
  parameters.
