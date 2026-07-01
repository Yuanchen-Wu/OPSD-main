# LESS-OPSD: Gradient-Based Data Selection for On-Policy Self-Distillation

This document describes a **LESS-inspired**, gradient-based data selection method
adapted to this repository's **OPSD (On-Policy Self-Distillation)** training. It
explains what the original LESS method does, which ideas we reuse, why LESS is not
directly applicable to OPSD, and the precise gradient definitions and approximations
used by our implementation.

Reference (methodology only, **not vendored**):
- Repo: https://github.com/princeton-nlp/LESS
- Paper: *LESS: Selecting Influential Data for Targeted Instruction Tuning*, ICML 2024.

---

## 1. What the original LESS does

LESS selects a small, influential subset of an instruction-tuning corpus that is most
useful for a given *target* task. The pipeline is roughly:

1. **Warmup LoRA training** on a small random fraction of the data so the model has a
   meaningful gradient geometry.
2. **Gradient feature extraction** for every candidate training example:
   - Run a forward/backward pass of the supervised instruction-tuning loss
     `CE(answer_i | instruction_i)`.
   - Collect the gradient **only over trainable (LoRA) parameters**.
   - Optionally combine with Adam optimizer moments to form an "Adam-preconditioned"
     gradient (`obtain_gradients_with_adam`).
   - **Project** the (very high-dimensional) gradient down to a few thousand dimensions
     using a random projection (TRAK `CudaProjector`/`BasicProjector`), and store the
     low-dimensional features in a gradient datastore on disk.
3. **Target/validation gradient features** are computed the same way on a few labeled
   examples of the target task.
4. **Influence scoring**: the influence of a training example on the target is the inner
   product (cosine, after normalization) between the projected training gradient and the
   projected target gradient (`matching.py: calculate_influence_score = train @ valid.T`),
   summed over checkpoints with learning-rate weights.
5. **Selection**: rank candidates by score and keep the top fraction / top-k
   (`write_selected_data.py`), then train on the selected subset.

### Parts of the LESS repo reused *conceptually*

| LESS file | Idea we reuse |
|---|---|
| `less/data_selection/collect_grad_reps.py` | Backprop a per-example loss, concatenate **only trainable** `p.grad`, then **project** to a low dimension. We keep the "trainable-params-only + projection" idea; we replace TRAK projection with CountSketch and replace the CE loss with the OPSD loss. |
| `less/data_selection/get_info.py` | Driver that loads a model, asserts only LoRA params require grad, and collects grads for a dataset. Mirrored by our `LESSOPSDGradientExtractor` + CLI. |
| `less/data_selection/matching.py` | `influence = train_features @ target_features.T`. We use the same normalized inner product (cosine) scoring. |
| `less/data_selection/write_selected_data.py` | Sort by score, take top fraction / top-k, persist selection. Mirrored by our artifact writer. |

### Parts we deliberately **do not** reuse

- TRAK / `fast_jl` CUDA projectors and the `trak` dependency (heavy, CUDA-specific).
  We use a dependency-free CountSketch projection instead.
- Adam-preconditioned gradients and multi-checkpoint LR-weighted aggregation (these are
  natural future extensions; the MVP uses a single model and raw SGD-style gradients).
- The supervised cross-entropy loss `CE(answer | instruction)`. **This is the key
  conceptual change** — see below.

---

## 2. Why original LESS is not directly applicable to OPSD

LESS assumes **supervised instruction tuning**: each example is a fixed
`(instruction, answer)` pair and the gradient is taken of a static cross-entropy loss on
the gold answer. The "influence" of an example is defined entirely by that fixed
supervised gradient.

OPSD is **not** supervised fine-tuning on a fixed answer. In OPSD:

- The student only sees the **problem** and **generates its own on-policy rollout**.
- The teacher sees the **problem + a privileged reference solution** and scores the
  student's *own* rollout.
- The training signal is a **distillation divergence** (generalized JSD, or a
  reverse-KL / "thinking-machines" policy-gradient variant) between teacher and student
  token distributions over the **student-generated** tokens — not cross-entropy against a
  fixed gold answer.

Consequences:

1. The "example gradient" must be taken of the **OPSD distillation loss**, evaluated on a
   **freshly generated student rollout**, not of a supervised CE loss on a gold answer.
2. The gradient depends on a **stochastic rollout**, so it is a sample of a distribution,
   not a deterministic quantity (this is the main approximation; see §5).
3. The privileged reference solution enters only through the **teacher context**, exactly
   as in normal OPSD training — it is never a direct label for CE.

Therefore we keep LESS's *machinery* (trainable-grad extraction → projection → cosine
matching → top-k) but replace the *gradient feature* with the OPSD gradient produced by
the existing `OPSDTrainer.compute_loss`.

---

## 3. The OPSD candidate gradient

For a candidate problem `i`, the gradient feature is

```
g_i = grad_theta  L_OPSD(problem_i, reference_solution_i, student_rollout_i)
```

where

- `problem_i` is the problem text from the OPSD dataset row,
- `reference_solution_i` is the privileged reference solution from the same row (used to
  build the **teacher** context),
- `student_rollout_i` is generated **on-policy** by the current/base student model from
  the student prompt,
- `L_OPSD` is the **existing** distillation loss implemented by
  `OPSDTrainer.compute_loss` (generalized JSD by default, or the reverse-KL
  "thinking-machines" variant, honoring `beta`, `temperature`, `top_k_loss`,
  `jsd_token_clip`, `fixed_teacher`, EMA teacher, `student_thinking`/`teacher_thinking`,
  and `reason_first`),
- `theta` are the **trainable parameters only** (LoRA adapter weights for a PEFT model).

Crucially, the rollout and the loss are produced by the **same code path used during
training** (`prepare_on_policy_distillation_batch` → `compute_loss`), so the selection
gradient matches the training gradient by construction.

## 4. The OPSD target gradient

Given a small set of target/validation problems `j ∈ T`, the target feature is the mean
of their (projected, per-example-normalized) OPSD gradients:

```
g_target = mean_{j ∈ T}  project( grad_theta L_OPSD(problem_j, reference_solution_j, student_rollout_j) )
```

which is then normalized:

```
g_target_hat = normalize(g_target)
```

The candidate score is the cosine similarity between projected, normalized gradients:

```
score_i = < normalize(project(g_i)), g_target_hat >
```

and we select the top fraction / top-k candidates by `score_i`.

---

## 5. Static cached-rollout approximation (the MVP)

The MVP implements the **static, single-rollout, single-model** version:

1. Load the initial model (optionally with a LoRA adapter so the gradient lives in the
   LoRA subspace, matching training).
2. For each candidate problem: generate **one** student rollout, build the OPSD
   distillation batch, run `compute_loss`, backprop **once**, project the trainable-param
   gradient with CountSketch, normalize, store on CPU.
3. Do the same for each target problem; average + normalize the target features.
4. Score candidates by cosine alignment; select top-k / top-p.
5. Save selected indices (consumable by `opsd_train.py --selected_indices_path`).

This is intentionally a clean experimental MVP, not a framework. The model is **not**
updated during selection (no `trainer.train()`), so the gradient geometry is that of the
initial/base model — analogous to LESS *before* its multi-checkpoint aggregation.

### Limitations

- **Rollout stochasticity.** Each candidate gradient is computed from a *single* sampled
  rollout. Temperature/top-p/top-k sampling means a different run can yield a different
  gradient and therefore a different score. The MVP fixes seeds for the projection but the
  generation itself is sampled; scores are noisy estimates of an expectation over rollouts.
  *Future:* average gradients over multiple rollouts per candidate.
- **Target-set sensitivity.** The selection is only as good as the target set. A small or
  unrepresentative target subset biases selection toward a narrow slice of the problem
  distribution. *Future:* larger / stratified target sets, or per-cluster targets.
- **High compute cost.** Every candidate requires a full generation + forward + backward.
  This is O(N) generations, which dominates cost. *Future:* cache rollouts, batch
  gradient extraction, or restrict the candidate pool.
- **Single model / single checkpoint.** No Adam preconditioning and no multi-checkpoint
  LR-weighted aggregation (both used by full LESS). *Future:* add optimizer-state
  preconditioning and checkpoint ensembling.
- **Not online / adaptive yet.** Selection is computed once, up front, against a fixed
  model. It does not refresh as the student improves during training. *Future:* periodic
  online gradient refresh during OPSD training.
- **Projection collisions.** CountSketch is a hashing projection; distinct gradient
  coordinates can collide into the same bucket. Larger `projection_dim` reduces collision
  noise at the cost of memory/compute.

---

## 6. Extensibility

The implementation is modular so the following can be added later without disrupting the
MVP:

- **Online gradient refresh** during OPSD training (recompute features every K steps).
- **Multiple rollouts per candidate** (average `g_i` over rollouts to reduce variance).
- **Diversity-aware selection** (e.g. facility-location / submodular selection on top of
  the gradient features instead of pure top-k).
- **CE-proxy baseline (`less_ce_proxy`)** that replaces `L_OPSD` with a plain
  cross-entropy on the reference solution — a closer analogue of original LESS, useful as
  an ablation. This is explicitly a *separate* method, never the default.
