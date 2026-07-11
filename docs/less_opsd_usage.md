# LESS-OPSD usage (v2): warmup → selection → final training

This guide covers the three-stage, Colab-friendly workflow. See
`docs/less_opsd_methodology.md` for the method itself.

The pipeline is deliberately LESS-style:

1. **Stage A (warmup)** — train briefly on a small *random* subset to obtain checkpoints
   whose AdamW optimizer state encodes useful gradient geometry.
2. **Stage B (selection)** — score the candidate pool against a target set using those
   checkpoints, and save `selected_indices.json`.
3. **Stage C (final training)** — **restart from the original base initialization** and
   run standard OPSD training on the selected subset with fresh on-policy rollouts.

Final training does **not** continue from the warmup model, and never reuses selection
rollouts, features, or cached logits.

## Stage A: warmup

Create a reproducible random warmup subset (artifact format is compatible with
`load_selected_indices` / `--selected_indices_path`):

```bash
python less_opsd_select.py \
  --selection_method warmup_subset \
  --selection_num_examples 500 \
  --warmup_subset_seed 7 \
  --output_dir outputs/warmup_subset
```

Then run a short OPSD warmup on that subset with the existing training script. Make sure
checkpoints are saved (HF Trainer checkpoints include `optimizer.pt`,
`trainer_state.json`, and the LoRA adapter by default):

```bash
python opsd_train.py \
  ... your usual lightweight arguments ... \
  --selected_indices_path outputs/warmup_subset/selected_indices.json \
  --selection_method warmup_random \
  --output_dir outputs/warmup \
  --save_steps 25 --save_total_limit 4 \
  --logging_steps 5      # needed later for checkpoint_weighting=learning_rate
```

This produces e.g. `outputs/warmup/checkpoint-25`, `outputs/warmup/checkpoint-50`, each
containing the LoRA adapter, `optimizer.pt` (AdamW moments), and `trainer_state.json`.

## Stage B: selection

### Colab example (AdamW-aware, 1–2 checkpoints, 2 rollouts)

```bash
python less_opsd_select.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --feature_type adamw_candidate_update \
  --target_objective opsd \
  --checkpoint_paths outputs/warmup/checkpoint-25 outputs/warmup/checkpoint-50 \
  --checkpoint_weighting uniform \
  --num_candidate_rollouts 2 \
  --num_target_rollouts 2 \
  --projection_dim 4096 \
  --candidate_limit 2000 \
  --target_limit 128 \
  --selection_fraction 0.05 \
  --save_every 25 \
  --resume \
  --output_dir outputs/less_opsd_selection/adamw_run
```

Notes:

* `--feature_type adamw_candidate_update` **requires** `--checkpoint_paths` pointing at
  HF Trainer checkpoints that contain `optimizer.pt`. There is no silent fallback.
* `--resume` makes extraction restartable after a Colab disconnect: partial features are
  saved every `--save_every` examples under `<output_dir>/resume/`, guarded by a
  configuration fingerprint. Re-running the same command skips completed examples;
  changing the projection seed, checkpoints, feature type, rollout counts, or target
  objective is rejected with a clear error (use a fresh `--output_dir`).
* `--checkpoint_weighting learning_rate` weights each checkpoint by the last
  `learning_rate` logged in its `trainer_state.json`; `explicit` takes
  `--checkpoint_weights 0.3 0.7` (normalized to sum to one).
* The LoRA flags (`--lora_r`, `--lora_alpha`, `--lora_target_modules`) and the OPSD loss
  flags must match the warmup run, otherwise checkpoint loading fails.
* Selection loads one checkpoint at a time (adapter weights are swapped in place;
  optimizer moments stay on CPU), so memory stays at one-model level.

### Cheap smoke test (raw gradients, no warmup checkpoint needed)

```bash
python less_opsd_select.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --feature_type raw_gradient \
  --num_candidate_rollouts 1 \
  --candidate_limit 32 \
  --target_limit 8 \
  --selection_fraction 0.25 \
  --max_completion_length 128 \
  --projection_dim 4096 \
  --output_dir outputs/less_opsd_selection/raw_smoke
```

This is the v1-equivalent baseline (`less_opsd_raw_static`) run at the base
initialization; useful to verify the environment before spending compute on the
AdamW-aware run.

### Random baseline

```bash
python less_opsd_select.py \
  --selection_method random \
  --candidate_limit 2000 \
  --selection_fraction 0.05 \
  --output_dir outputs/less_opsd_selection/random_smoke
```

## Stage C: final training

Restart from the **original base model** (do not resume from the warmup checkpoint) and
restrict the dataset to the selection:

```bash
python opsd_train.py \
  ... your usual arguments (same base model as the warmup) ... \
  --selected_indices_path outputs/less_opsd_selection/adamw_run/selected_indices.json \
  --selection_method less_opsd \
  --output_dir outputs/final_selected
```

Training behavior is otherwise unchanged: the student generates fresh on-policy rollouts
every step, exactly as without selection. With `--selected_indices_path` unset, training
is completely unaffected by any of the selection code.

## Artifacts

Written to `--output_dir`:

* `selected_indices.json` — selected original dataset indices + full metadata (method
  name/version, feature type, target objective, checkpoint ids/paths/weights, optimizer
  metadata, rollout counts, projection dim/seed, rollout seed, teacher mode, LoRA
  config, dataset name/split, candidate/target indices, config fingerprint).
* `scores.jsonl` — one line per candidate, ranked:
  `{"original_index", "final_score", "checkpoint_scores": [...], "rollout_consistency":
  [...], "selected", "rank", "invalid"}`. Invalid (non-finite / zero-norm feature)
  candidates have `final_score: null` and are never selected.
* `selection_config.json` — the resolved configuration, checkpoint weights, and
  fingerprint.
* `target_feature_<ckpt>.pt` — the normalized target group feature per checkpoint
  (plus a `target_feature.pt` alias for single-checkpoint runs).
* `candidate_features_<ckpt>.pt` — only with `--save_candidate_features`.
* `resume/` — partial feature shards + manifest (safe to delete after a completed run).

## Colab sizing guidance (not hardcoded anywhere)

For an A100/T4 smoke run: `candidate_limit` 32–200, `target_limit` 8–32,
`num_candidate_rollouts` 1–2, one warmup checkpoint, `max_completion_length` 128–256.
For a real run: `candidate_limit` 2000+, `target_limit` 128, 2 rollouts, 1–2 checkpoints.
Wall-clock cost scales as
`(num_candidates * num_candidate_rollouts + num_targets * num_target_rollouts) *
num_checkpoints` generations + backward passes.
