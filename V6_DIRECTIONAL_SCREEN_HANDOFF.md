# V6 Directional Screen Handoff

Status: **code-only handoff; do not start RunPod or any paid GPU from this
document**.

This is the fastest scientifically interpretable V6 screen. It compares three
training-data arms on the same 24-task Pilot:

- `flawless_only`: successful clean trajectories for the matched tasks;
- `random_stratified_seed_20260806`: one random sibling-recovery pair per task;
- `full_proposed`: one pair per task selected by the frozen causal,
  frozen-base-hardness, and coverage score.

The recovery selection atom is one complete `candidate_pair`, containing
exactly two sibling error branches from the same prefix and environment
snapshot. A branch's failed tool call and observed error are context only. The
positive SFT target is the complete freshly generated successful assistant
recovery suffix; neither the failed call nor the tool error is a label.

This handoff describes what may be uploaded now and what must be implemented
and frozen before a GPU run. It does **not** report an experimental result.

## 1. Frozen scientific boundary

- Benchmark: tau2 commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
- Split source:
  `artifacts/v5_stage0/manifests/split_manifest.json`.
- Inner-train partition identity:
  `c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818`.
- Pilot: exactly 24 outcome-independent tasks: 18 retail and 6 airline, as
  listed in `configs/v6_causal_recovery_selection.yaml`.
- Student: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, QLoRA 4-bit.
- Fast screen: training seed `20260722`; evaluation seeds `20260722`,
  `20260723`, and `20260724`.
- Primary comparison: `full_proposed - random_stratified_seed_20260806`.
- `flawless_only` is a secondary matched-task control. Its actual token budget
  difference must be reported; it must not be post-hoc altered to imitate the
  recovery-arm budget.
- Primary metric: controlled-error end-to-end official task success. Next-call
  exact match, token log-probability, and training loss are diagnostics only.
- Independent unit: task ID. Three evaluation seeds are repeated measurements,
  not three new samples.

The official 60-task test split remains sealed during registry construction,
generation, auditing, measurement, scoring, selection, materialization,
training, checkpoint choice, and smoke testing. It is unsealed once, only
after the complete release contract is frozen and hashed. No test task ID or
content may be exported to a generation worker.

## 2. Compatibility decision

### Candidate and selection pipeline

The V6 registry, generation, runtime audit, frozen-student measurement,
scoring, selector-manifest, and materialization scripts form the intended V6
pipeline:

1. `scripts/prepare_v6_candidate_registry.py`
2. `scripts/run_v6_candidate_generation.py`
3. `scripts/measure_v6_candidate_tokens.py`
4. `scripts/audit_v6_candidates.py`
5. `scripts/score_v6_candidates.py`
6. `scripts/build_v6_selector_manifests.py`
7. `scripts/materialize_v6_sft.py`

These commands may be uploaded and locally tested. They must not be described
as having produced a positive result until their artifacts exist and pass the
gates below.

### Existing V5 trainer: not directly compatible

`scripts/train_v5_sft_causal.py` cannot be used as the V6 experiment runner
without a V6-specific entry point. Its command-line contract and fail-closed
checks are frozen to V5:

- it accepts only V5 arm names and V5 data-provenance protocols;
- it requires the V5 audited directory layout and a fixed 256-row schedule;
- it contains V5/V5.5/V5.6-specific seed, step, and mixture rules;
- `validate_row` permits positive labels only on assistant tool-call messages,
  while V6 supervises the complete fresh assistant recovery suffix, including
  assistant text when present.

The reusable portion is the low-level QLoRA implementation: pinned 7B model,
4-bit loading, message-level causal masks, deterministic sampler, CUDA
preflight, finite-metric audit, and adapter packaging.

The V6-specific training entry point is included:

```text
scripts/train_v6_directional_sft.py
```

It consumes V6 materialized JSONL directly, accepts the three V6 paper-arm
names, preserves full-assistant-suffix labels, re-tokenizes every frozen token
contract, binds source/data/model/tokenizer hashes, uses training seed
`20260722`, and emits a self-contained adapter run manifest.

### Existing V5 end-to-end evaluator: not directly compatible

`scripts/run_v5_sft_causal_eval.py` is also frozen to V5 arm sets,
checkpoint-registry profiles, evaluation-manifest protocols, model aliases,
and dynamic-audit artifacts. Renaming a V6 arm to a V5 arm would destroy the
experimental provenance and is forbidden.

Minimum required additions before official evaluation:

```text
scripts/build_v6_checkpoint_registry.py
scripts/run_v6_end_to_end_eval.py
scripts/summarize_v6_directional_screen.py
```

The evaluator may reuse the V5 tau2 single-tool agent, controlled-error
execution, strict judge, and clean/error rollout machinery, but the new wrapper
must bind V6 arms, V6 checkpoint hashes, one training seed, three evaluation
seeds, and the one-time official-test unseal receipt.

### Proposed-selector weights: resolved and regression-tested

The implemented `full_proposed` weights are bound directly to the frozen
protocol:

```text
causal = 0.50
hardness = 0.25
coverage = 0.25
```

`scripts/build_v6_selector_manifests.py` reads
`FULL_OBJECTIVE_WEIGHTS` from `scripts/v6_selection_protocol.py`; it has no
separate cost term. A regression test fails if the key, value, or sum drifts.
Coverage is recomputed as the frozen weighted `log(1+n)` marginal gain during
selection.

## 3. Code-only preflight

The following is safe on a Mac because it does not start a model server or a
GPU job:

```bash
set -euo pipefail

REPO=/absolute/path/to/recovery-aware-trajectory-selection
TAU2=/absolute/path/to/tau2
cd "$REPO"

test "$(git -C "$TAU2" rev-parse HEAD)" = \
  fc0055dc4e0a316c3f83133267fbd6faaa770992

python3 -m py_compile \
  scripts/v6_selection_protocol.py \
  scripts/prepare_v6_candidate_registry.py \
  scripts/run_v6_candidate_generation.py \
  scripts/audit_v6_candidates.py \
  scripts/measure_v6_candidate_tokens.py \
  scripts/score_v6_candidates.py \
  scripts/build_v6_selector_manifests.py \
  scripts/materialize_v6_sft.py \
  scripts/train_v6_directional_sft.py

python3 -m unittest \
  tests.test_v6_selection_protocol \
  tests.test_v6_candidate_generation \
  tests.test_v6_directional_trainer
```

Run the remaining V6 tests in the repository's pinned test environment:

```bash
python -m pytest -q \
  tests/test_v6_candidate_registry.py \
  tests/test_v6_candidate_audit.py \
  tests/test_v6_candidate_measurement.py \
  tests/test_v6_candidate_scoring.py \
  tests/test_v6_selector_manifests.py \
  tests/test_v6_sft_materializer.py
```

Stop if any test fails. Passing these tests authorizes a code release, not an
experimental claim.

## 4. Artifact namespace and command placeholders

These commands are a frozen launch template for a future authorized host. They
are included so the uploaded branch has a complete interface. Do **not** run
them as part of the present code-upload task.

```bash
set -euo pipefail

REPO=/absolute/path/to/recovery-aware-trajectory-selection
TAU2=/absolute/path/to/tau2
ROOT="$REPO/artifacts/v6_directional_screen"
CONFIG="$REPO/configs/v6_causal_recovery_selection.yaml"
SPLIT="$REPO/artifacts/v5_stage0/manifests/split_manifest.json"

TEACHER_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
TEACHER_REV=5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c
TEACHER_API=http://127.0.0.1:8101/v1

USER_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
USER_REV=539535859b135b0244c91f3e59816150c8056698
USER_API=http://127.0.0.1:8201/v1

STUDENT_MODEL=Qwen/Qwen2.5-7B-Instruct
STUDENT_REV=a09a35458c702b33eeacc393d103063234e8bc28

TRAIN_SEED=20260722
EVAL_SEED_1=20260722
EVAL_SEED_2=20260723
EVAL_SEED_3=20260724
```

### 4.1 Build the outcome-independent registry

```bash
cd "$REPO"
python scripts/prepare_v6_candidate_registry.py \
  --tau2-root "$TAU2" \
  --split-manifest "$SPLIT" \
  --config "$CONFIG" \
  --output "$ROOT/registry.json"
```

Required receipt facts: 24 Pilot tasks, 50 action-identifiable formal-pool
tasks, pinned tau2 commit, official test unused.

### 4.2 Generate the 24-task Pilot

The fastest reproducible launch uses one shard. A later multi-host launch must
keep all pairs for a task on the same shard and add a separately audited merge.

```bash
python scripts/run_v6_candidate_generation.py \
  --tau2-root "$TAU2" \
  --registry "$ROOT/registry.json" \
  --output-dir "$ROOT/generation/pilot" \
  --phase pilot \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-revision "$TEACHER_REV" \
  --teacher-api-base "$TEACHER_API" \
  --user-model "$USER_MODEL" \
  --user-revision "$USER_REV" \
  --user-api-base "$USER_API" \
  --judge-model "$USER_MODEL" \
  --judge-revision "$USER_REV" \
  --judge-api-base "$USER_API" \
  --continuation-seeds 20260806,20260807,20260808 \
  --num-shards 1 \
  --shard-index 0
```

Expected files, not expected values:

```text
generation/pilot/run_contract.json
generation/pilot/tasks/*.json
generation/pilot/candidate_pairs.unscored.jsonl
generation/pilot/generation_receipt.json
```

### 4.3 Measure exact costs and frozen-base hardness

This is the first stage that loads the 7B student. It is not part of the
present upload-only task. It precedes final audit because positive exact token
accounting is itself an acceptance requirement.

```bash
python scripts/measure_v6_candidate_tokens.py \
  --input "$ROOT/generation/pilot/candidate_pairs.unscored.jsonl" \
  --output "$ROOT/scores/pilot.measured.jsonl" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REV" \
  --tokenizer "$STUDENT_MODEL" \
  --tokenizer-revision "$STUDENT_REV" \
  --tau2-root "$TAU2" \
  --load-in-4bit \
  --device cuda \
  --dtype bfloat16
```

### 4.4 Audit real execution and measured labels before scoring

The auditor recomputes action-switch accuracy from the teacher's first,
unretried action in each accepted error context. The balanced error-blind
constant-action baseline is exactly `0.50`; neither value is entered by hand.

```bash
python scripts/audit_v6_candidates.py \
  --registry "$ROOT/registry.json" \
  --candidates "$ROOT/scores/pilot.measured.jsonl" \
  --phase pilot \
  --output-root "$ROOT/audit/pilot"
```

The command exits nonzero unless runtime evidence, exact full-suffix label
masks, token accounting, and the Pilot gate all pass.

### 4.5 Score and select the three fast arms

Run this only when the Section 2 proposed-weight regression test passes.

```bash
python scripts/score_v6_candidates.py \
  --input "$ROOT/audit/pilot/accepted_candidate_pairs.jsonl" \
  --output "$ROOT/scores/pilot.scored.jsonl" \
  --audit-output "$ROOT/scores/pilot.scoring_audit.json"

python scripts/build_v6_selector_manifests.py \
  --scored-pool "$ROOT/scores/pilot.scored.jsonl" \
  --output-dir "$ROOT/manifests" \
  --selectors flawless_only,random_stratified,full_proposed
```

For the fast training screen, use exactly:

```text
manifests/flawless_only.json
manifests/random_stratified_seed_20260806.json
manifests/full_proposed.json
```

The other two random manifests are selection-robustness artifacts; they are
not additional training arms in this fast screen.

### 4.6 Materialize SFT rows

```bash
for ARM in flawless_only random_stratified_seed_20260806 full_proposed
do
  python scripts/materialize_v6_sft.py \
    --manifest "$ROOT/manifests/$ARM.json" \
    --output "$ROOT/data/$ARM/train.jsonl" \
    --audit-output "$ROOT/data/$ARM/audit.json" \
    --tokenizer "$STUDENT_MODEL" \
    --tokenizer-revision "$STUDENT_REV"
done
```

Every recovery pair yields two rows. The audit must prove zero failed-action
labels, zero future leakage, no split candidate pair, and exact agreement with
the manifest's `c_sup` and `c_nonpad`.

## 5. V6 directional SFT training

Do not substitute `train_v5_sft_causal.py`. First resolve and freeze the model
configuration and tokenizer fingerprints without training:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
python scripts/train_v6_directional_sft.py \
  --output-dir "$ROOT/training/model_identity" \
  --identity-only

MODEL_CONFIG_SHA="$(
  python -c 'import json; print(json.load(open("'"$ROOT"'/training/model_identity/model_tokenizer_identities.json"))["model_config_sha256"])'
)"
TOKENIZER_SHA="$(
  python -c 'import json; print(json.load(open("'"$ROOT"'/training/model_identity/model_tokenizer_identities.json"))["tokenizer_sha256"])'
)"

for MANIFEST_NAME in flawless_only random_stratified_seed_20260806 full_proposed
do
  PAPER_ARM="$MANIFEST_NAME"
  if [ "$MANIFEST_NAME" = random_stratified_seed_20260806 ]; then
    PAPER_ARM=random_stratified
  fi
  mkdir -p "$ROOT/training/$MANIFEST_NAME"
  TRAIN_FILE="$ROOT/data/$MANIFEST_NAME/train.jsonl"
  TRAIN_SHA="$(sha256sum "$TRAIN_FILE" | awk '{print $1}')"
  SELECTOR_MANIFEST="$ROOT/manifests/$MANIFEST_NAME.json"
  SELECTOR_MANIFEST_SHA="$(
    python -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifests"][sys.argv[2]]["sha256"])' \
      "$ROOT/manifests/selector_audit.json" "$MANIFEST_NAME"
  )"
  python scripts/train_v6_directional_sft.py \
    --train-file "$TRAIN_FILE" \
    --expected-train-sha256 "$TRAIN_SHA" \
    --selector-manifest "$SELECTOR_MANIFEST" \
    --expected-selector-manifest-sha256 "$SELECTOR_MANIFEST_SHA" \
    --output-dir "$ROOT/training/$MANIFEST_NAME/seed_$TRAIN_SEED" \
    --arm "$PAPER_ARM" \
    --model "$STUDENT_MODEL" \
    --model-revision "$STUDENT_REV" \
    --tokenizer-revision "$STUDENT_REV" \
    --expected-model-config-sha256 "$MODEL_CONFIG_SHA" \
    --expected-tokenizer-sha256 "$TOKENIZER_SHA" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --train-seed "$TRAIN_SEED" \
    --mode formal \
    2>&1 | tee "$ROOT/training/$MANIFEST_NAME/console.log"
done
```

Required per-arm products:

```text
checkpoint_final/adapter_config.json
checkpoint_final/adapter_model.safetensors
command.txt
console.log
training_metrics.json
data_audit.json
audit.json
run_manifest.json
files_sha256.json
```

Run the same command once with `--mode smoke` in fresh output directories
before the formal loop. Training is authorized only when all materialization
contracts are immutable, every expected hash matches, CUDA preflight passes,
every loss and gradient statistic is finite, and QLoRA hyperparameters are
identical across arms.

## 6. Official end-to-end evaluation contract still required

After V6 checkpoint-registry and evaluator scripts exist, pass their exact
`--help` interfaces rather than copying a V5 command. The required logical
grid is:

```text
3 arms × 1 training seed × 3 evaluation seeds × {clean, controlled-error}
```

The future evaluator command must bind:

- the sealed official-test unseal receipt;
- exact V6 checkpoint-registry hash;
- exact selector-manifest and training-run hashes;
- the same frozen user simulator, strict judge, decoding, timeout, step
  budget, error policy, and task set across arms;
- one result row per task, condition, and evaluation seed;
- full task success and final-state evidence, not only tool-call matching.

The summarizer must report paired task-level deltas, task-cluster bootstrap
confidence intervals, clean non-inferiority against the preregistered `-0.05`
margin, runtime/cost, and diagnostic metrics. It must not count the three
evaluation seeds as independent tasks.

## 7. Stop gates

Stop before **formal Pilot generation** if any of these hold:

- tau2 commit or split/partition identity differs;
- an official-test task or its content is visible;
- teacher, user simulator, judge, or student revision is not pinned;
- shared prefix/environment-snapshot or actual-error execution cannot be
  audited;
- an injected failed call changes agent or user database state;
- the clean future is visible while a recovery is generated.

Stop after **Pilot audit** unless all gates pass:

- at least 12 of 24 tasks have at least three accepted sibling pairs;
- at least 48 accepted pairs total;
- at least two domains and three error families;
- independent matched replay and all four forced-first cells complete;
- at least 20% of accepted pairs have `kappa >= 0.75`;
- at least 20% have `kappa <= 0.25`;
- no kappa bucket exceeds 80%;
- teacher/oracle action-switch accuracy is at least 0.80;
- teacher advantage over the error-blind baseline is at least 0.20;
- zero future leakage and zero failed calls as positive labels.

Stop before **selection** if token/hardness provenance is incomplete, low-kappa
pairs were removed, or the proposed weights do not equal
`0.50/0.25/0.25`.

Stop before **training** if any arm does not materialize, recovery-arm `c_sup`
is not exactly matched, a candidate pair was split, model/tokenizer identities
or source/data hashes drift, or a V5 arm/provenance alias is used.

Stop before **official evaluation** if the V6 checkpoint registry/evaluator is
absent, seeds are not frozen, checkpoint selection is unfinished, source or
artifact hashes drift, or the official-test unseal is not a single auditable
event.

Stop before a **scientific claim** unless end-to-end official task success and
task-level uncertainty are available. A successful pipeline, lower training
loss, next-call exact match, or a positive point estimate alone does not prove
the V6 selection method works.

## 8. Current code-upload definition of done

The present task is complete when:

1. V6 protocol/config, registry, generation, measurement, audit, scoring,
   selector, materializer, directional trainer, tests, and this handoff are
   committed;
2. the proposed-weight constant mismatch is fixed and regression-tested;
3. local code-only tests pass;
4. the branch is pushed to GitHub;
5. no Pod is deployed or started, no GPU experiment is launched, and no
   unobserved metric is reported.

The V6 checkpoint registry, official evaluator, and task-cluster summarizer
remain a subsequent code milestone. Until they exist, this branch is a
complete candidate-selection/data-construction plus training release, not a
one-command end-to-end experimental runner.
