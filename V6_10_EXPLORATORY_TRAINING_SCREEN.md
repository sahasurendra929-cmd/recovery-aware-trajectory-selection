# V6.10 exploratory training-screen protocol

Status: **exploratory engineering screen; not a V6.10 formal stage**.

This protocol asks whether the already materialized A08/A12 recovery-pair
inventory contains enough signal to justify the cost of a future formal
experiment. It is deliberately separated from `v6_10_pipeline_closure_v1` and
cannot authorize Compatibility, Pilot, formal generation, training comparison,
official-test use, or paper evidence.

All artifacts from this protocol must carry:

```text
classification = EXPLORATORY_NOT_FOR_FORMAL_GATE
```

## 1. Fixed source inventory

Use only the immutable, hash-indexed candidate files listed in:

```text
results/v6_10_exploratory_24_task_20260810/
EXPLORATORY_24_TASK_CATALOGUE.json
```

The eligible materialized inventory is exactly 48 pairs, from the two
internally readable sessions below:

| Session | Tasks with candidate files | Materialized pairs |
|---|---:|---:|
| A08 | `retail:98,47,31,92,99,11` | 18 |
| A12 | `retail:104,107,37,21,19,35,30,87,72,7` | 30 |

The empty A08 `retail:16` file and all A05/A07 material are excluded. In
particular, the 21 A05/A07 receipt-declared pairs remain quarantined because
their legacy receipt-index hashes do not match. No raw cross-session causal
score is pooled.

## 2. Research question and non-claim

Exploratory question:

> With a fixed QLoRA token and compute budget, does selecting recovery pairs
> by a **within-session rank of recorded forced-first outcome metadata** give a
> promising held-out recovery signal relative to deterministic stratified
> random selection?

This is a feasibility screen. It does **not** estimate the V6.10 primary
estimand, demonstrate causal necessity, or establish that one training-data
path is better. Its result may only be `PROMISING_EXPLORATORY_SIGNAL`,
`INCONCLUSIVE_EXPLORATORY_SIGNAL`, or `NO_EXPLORATORY_SIGNAL`.

## 3. Frozen task split

The split is by task identity, before materialization or training:

```text
screen_train_tasks =
retail:98, retail:47, retail:92, retail:99, retail:107, retail:37,
retail:21, retail:19, retail:35, retail:30, retail:87, retail:7

screen_eval_tasks =
retail:31, retail:11, retail:104, retail:72
```

No candidate, rendered training example, score, or selector decision from an
evaluation task may enter either training arm. Evaluation tasks are not the
official test and must be called `exploratory held-out tasks` in every report.

## 4. Arms and selection

The train pool contains 12 tasks × 3 pairs = 36 pairs. Each arm selects
exactly 24 pairs (two pairs per train task) and materializes the same clean and
recovery views per selected pair.

### Arm R: deterministic stratified random

For each task, select two of its three pair IDs using SHA-256 ordering of:

```text
exploratory-random-v1 || task_id || candidate_pair_id || 20260810
```

This is the primary comparator.

### Arm S: session-normalized recorded-outcome rank

For a pair with the four already recorded forced-first cells, calculate the
recorded contrast only inside its own runtime session:

\[
c_p = \frac{1}{2}[R(e_1,a_1)+R(e_2,a_2)-R(e_1,a_2)-R(e_2,a_1)],
\]

where each `R` is the mean of the stored cell trials. Rank `c_p` within A08 or
A12 only; do not pool raw scores between sessions. For each train task, select
the two highest ranked pairs, breaking ties by `candidate_pair_id`.

`c_p` is an exploratory selection feature, not a new causal measurement or a
formal selector score.

## 5. Training controls

- Base model: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Objective: QLoRA SFT on the frozen materialized clean/recovery views.
- Arms use identical tokenizer, sequence length, packing, optimizer,
  scheduler, LoRA configuration, seed `20260810`, and maximum optimizer
  steps.
- The smaller realized supervised-token total is the common token budget; the
  other arm is deterministically downsampled or repeated only through a
  registered exact-token materializer. Padding does not count.
- Save a CPU-portable checkpoint and SHA-256 manifest after every epoch and at
  finalization. Never include credentials in checkpoints, logs, or Git.

No 72B teacher, 14B user/judge, official test, or new candidate generation is
used in this screen.

## 6. Exploratory held-out evaluation

Evaluate each trained checkpoint and the frozen untrained base on the four
`screen_eval_tasks` under the same controlled injected-error renderer. Use two
fixed seeds per task: `20260811` and `20260812`. Record environment task
success, repair-call validity, repeated-error rate, tool steps, and generated
tokens. All evaluator runs must use fresh task environments; no reference
suffix or candidate material is revealed to the student.

The reported comparison is descriptive:

```text
mean held-out recovery success(S) - mean held-out recovery success(R)
```

with the per-task results shown. No p-value, confidence interval, or formal
superiority claim is permitted from four exploratory tasks.

## 7. Decision rule

After every artifact hash, task split, token budget, checkpoint, and evaluator
result is verified:

- `PROMISING_EXPLORATORY_SIGNAL`: S exceeds R on the held-out mean and does
  not increase repeated-error rate; this justifies designing a formal
  preregistered low-budget release.
- `INCONCLUSIVE_EXPLORATORY_SIGNAL`: results are tied, mixed across tasks, or
  any implementation-quality condition is incomplete.
- `NO_EXPLORATORY_SIGNAL`: S is lower than R or materially increases repeated
  errors.

None of these outcomes authorizes a V6.10 Pilot or paper claim.

## 8. Runtime and cost envelope

Run serially on one 24-GB GPU:

```text
1× RTX PRO 4000 or 1× RTX 4090
```

Use a PyTorch image and mount the existing network volume at `/workspace`.
No multi-GPU topology is needed. Expected work is 1–2 hours preparation,
2–5 hours for two small QLoRA arms, and 2–5 hours for held-out evaluation;
stop after the registered screen completes. The exact wall time depends on
frozen sequence length and the observed token budget.

## 9. Required outputs

Store large checkpoints and raw evaluator traces only on the RunPod network
volume. Push only the following small, credential-scanned evidence to GitHub:

- `EXPLORATORY_TRAINING_SCREEN_MANIFEST.json` and SHA;
- both arm-selection manifests and SHA;
- token/step budget report;
- checkpoint index and hashes;
- evaluator summary and per-task descriptive table;
- final screen audit and handoff; and
- a report carrying `EXPLORATORY_NOT_FOR_FORMAL_GATE` prominently.
