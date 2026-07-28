# Recovery-Aware Trajectory Selection for Tool-Using Agents

This repository studies a simple question:

> Under a fixed training-token budget, which successful tool-use trajectories should be retained so that an agent can handle errors and corrections more reliably?

The project has completed an offline pilot and a small end-to-end pipeline
validation. It does not claim an end-to-end agent improvement yet.

## Current experiment versions

- **v1.1:** engineering baseline currently being allowed to finish unchanged;
  its 512-token context protocol is not the paper-quality protocol.
- **v2:** isolated corrected protocol with tokenizer-exact/source-controlled
  selection budgets, full system-policy retention, fixed compute, resumable
  evaluation, a zero-shot control, and task-cluster uncertainty. Its completed
  RTX 5060 result is audited in [`V2_RESULT_AUDIT.md`](V2_RESULT_AUDIT.md).
- **v3:** one-variable constrained-recovery diagnostic. It freezes the V2
  model, prompts, labels, token/source budgets, compute, and held-out evaluator,
  then changes only trajectory selection. The completed result preserved the
  predeclared non-recovery floor but did not improve recovery. See
  [`BASELINE_V3_HANDOFF.md`](BASELINE_V3_HANDOFF.md).
- **v4:** objective-level follow-up on the exact V3 trajectory set. It compares
  matched Clean-SFT with Standard V3, then compares DPO with a
  chosen-exposure-matched continued-SFT control. See
  [`BASELINE_V4_HANDOFF.md`](BASELINE_V4_HANDOFF.md).
- **v5 Stage 0:** a 20-run paired τ²-bench development smoke test using
  Qwen2.5-7B-Instruct. The pipeline and fault audit passed, while clean and
  error-injected task success were both 10% on different tasks. This validates
  execution and measurement only. See
  [`STAGE0_V5_RESULT_REPORT.md`](STAGE0_V5_RESULT_REPORT.md).
- **v5 Stage 1 (preflight stopped before training):** a controlled SFT
  data-composition/mechanism screen on full multi-turn trajectories. Under
  matched task, supervised-token, non-padding-token, and optimizer-step
  budgets, it compares flawless demonstrations,
  deliberately positive supervision of one failed action, masked failure
  context followed by later verified-success tool actions, and a 50/50
  clean–post-fault mixture by supervised-token mass. It trains Qwen2.5-7B
  with QLoRA and evaluates an
  additional untrained-base
  control on paired clean/error-injected end-to-end validation tasks. Its
  isolated manifest at `data/processed/v5_stage1_protocol` covers multiple
  read-only fault families across all 83 generation tasks and all 21
  validation tasks. Its frozen split input is the repository-tracked
  `artifacts/v5_stage0/manifests/split_manifest.json`; generated
  `data/processed/v5_stage0` files are not Stage-1 inputs. The claim is
  post-fault task robustness, not proof of semantic repair; per-family results
  are descriptive only. See
  [`V5_STAGE1_SFT_HANDOFF.md`](V5_STAGE1_SFT_HANDOFF.md). Its registered
  three-same-seed-pairs-per-task pool proved operationally infeasible: the
  complete initial generation yielded no eligible paired slot, and the
  corrected pilot remained far below the frozen threshold. No four-arm
  comparison was made. See
  [`V5_STAGE1_PREFLIGHT_RESULT.md`](V5_STAGE1_PREFLIGHT_RESULT.md).
- **v5.2 (stopped before training):** preserved the V5
  question, four SFT arms, 7B student, and end-to-end evaluation, but fixed
  the data-pool design. It runs a fixed 12 attempts per task and condition,
  pairs independently eligible clean and post-fault trajectories at task
  level, caps each task at two pairs, and introduced a fail-closed 40-task /
  48-pair data gate. Its generation/audit attempt exposed remaining task
  compatibility, strict-judge, matching, and controller-completion problems;
  no four-arm training comparison was claimed. See
  [`V5_2_SINGLE_HOST_HANDOFF.md`](V5_2_SINGLE_HOST_HANDOFF.md).
- **v5.3 (implemented, awaiting the preregistered RunPod pilot):** filters five
  ground-truth-incompatible tasks before any rollout, freezes the remaining 78
  tasks into 70 arm-train and 8 loss-validation tasks, uses a 32B-AWQ teacher
  plus a 14B-AWQ user/judge, and makes 12 disjoint-seed attempts per condition.
  A 24-task, 576-rollout pilot must yield at least 15 tasks with one pair and
  4 tasks with a second pair before formal generation is authorized. The
  controller also binds runtime context/concurrency preflights, global
  tolerance-aware four-arm matching, checkpoint provenance, full end-to-end
  result coverage, and recomputed summaries. See
  [`V5_3_SINGLE_HOST_HANDOFF.md`](V5_3_SINGLE_HOST_HANDOFF.md).
- **v5.4 (completed data-feasibility pilot, no training):** replaces two
  independent clean/error rollouts with a counterfactual branch from a shared
  clean prefix. It produced 22 audited pairs over 13 tasks in two domains, but
  stopped one task below the frozen 14-task coverage gate. The official test
  was not used and training was not authorized.
- **v5.5 (audit-first data release):** structurally screens the pinned
  inner-train split for reference-grounded read-only identifier lookups, then
  executes every reference path and two controlled mutations before task
  registration. The pinned screen leaves 36 executable candidates and freezes
  24 tasks / 48 pairs. An independent environment replay must verify every
  error, correction and final database hash before training is authorized.
  See [`V5_5_AUDIT_FIRST_HANDOFF.md`](V5_5_AUDIT_FIRST_HANDOFF.md).
  The complete natural-conversation dose-response and sealed-test design is
  frozen in
  [`V5_5_FULL_EXPERIMENT_PLAN.md`](V5_5_FULL_EXPERIMENT_PLAN.md).

Never compare or merge v1.1 with v2/v3/v4 outputs. V3 may be paired only with the
audited V2 `random_success` result because those two share the frozen examples
and evaluation protocol. V4 reuses the V3 selection and the same 959-example
generation evaluator, while adding objective-aligned outcome annotations.

## Current evidence-bound claim

In the first τ-bench historical-retail pilot, recovery enrichment changed the
model's action prior but did **not** improve offline repair-call exact match.
Distribution constraints reduced the overall damage but still produced no
recovery gain. V4 therefore tested the narrower mechanism suggested by the
error analysis: failed calls should not remain positive SFT labels, and an
observed successful repair should be preferred to replaying the failed call in
the same post-error context.

V5 Stage 0 establishes that the project can now measure full task completion
after a controlled tool failure. Its 10-pair result is too small and its 7B
baseline success rate is too low to update the scientific claim.

The V5 Stage-1 preflight establishes a narrower negative engineering result:
the original same-seed 3/3 pairing requirement cannot supply the registered
training pool with the frozen generation stack. It does not establish whether
post-fault supervision helps or harms. V5.2 exposed further implementation and
attainability risks. V5.3 is the current preregistered continuation; it has not
yet produced a positive scientific result, and formal generation is prohibited
unless its train-only pilot returns `GO_FORMAL_GENERATION`. V5.4 subsequently
showed that shared-prefix counterfactual branching materially improves pair
yield, but missed its task-coverage gate by one. V5.5 is the audit-first
continuation: it has passed a local 48-pair construction and independent
environment-replay integration test, but this is still data evidence rather
than a model-quality result. The next scientific result requires the frozen
three-arm SFT comparison and held-out end-to-end task-success evaluation.

## Repository map

```text
configs/          frozen experiment contracts
data/             source and schema documentation; no benchmark data is committed
scripts/          reproducible offline-pilot code
results/          versioned, observed pilot outputs
artifacts/        audited end-to-end summaries and raw development trajectories
PROJECT_SPEC.md   research specification and evidence rules
CONTRIBUTING.md   collaboration and reporting rules
```

## Quick start: reproduce the data-selection pilot

The pilot requires only Python 3.10+ and the public historical trajectories from the legacy τ-bench repository. No API key is required.

```bash
git clone https://github.com/sierra-research/tau-bench.git data/raw/tau-bench
python3 scripts/run_data_baseline.py \
  --data-dir data/raw/tau-bench/historical_trajectories \
  --output-dir results/reproduced_pilot
```

The script will:

1. read successful retail trajectories;
2. split data by task ID before selection;
3. construct equal estimated-token-budget `random_success`, `shortest_success`, and `recovery_balanced` subsets;
4. save selected-trajectory manifests and selection statistics.

`estimated_tokens` is a deterministic character/4 proxy used only in this no-model pilot. Fine-tuning experiments must use the exact tokenizer of the training model.

## What counts as an error-resolution event?

The current conservative rule labels an event only when:

1. a tool response matches a clear error pattern;
2. a later tool call changes its tool name or arguments; and
3. the complete trajectory is environment-successful.

The label also records whether a user spoke before the corrective tool call. This distinction matters: **user-assisted error resolution is not the same as agent-initiated recovery.**

## Roadmap

- [x] Reproducible equal-budget trajectory-selection pilot
- [x] Task-group split and selected-trajectory manifests
- [x] Error-resolution audit fields
- [x] Token-exact, source-controlled QLoRA v2 protocol and audit
- [x] Complete and audit the single-seed V2 RTX 5060 baseline
- [x] Freeze the constrained-recovery V3 selector and overnight protocol
- [x] Run the V3 diagnostic on RTX 5060 and pair it with V2 Random
- [x] Build and audit matched Clean-SFT plus 79 strict V4 preference pairs
- [x] Run 20 paired τ²-bench Stage-0 end-to-end development trajectories
- [x] Verify all ten injected failures are real and database-preserving
- [x] Freeze the V5 Stage-1 controlled SFT data, training, and end-to-end evaluation code
- [x] Isolate the Stage-1 multi-fault manifests from immutable Stage-0 artifacts
- [x] Dynamically audit all 83 generation and 21 validation injected calls
- [x] Record the fail-closed V5 Stage-1 paired-pool infeasibility result
- [x] Implement and test the V5.2 task-level, fixed-attempt data-pool protocol
- [x] Stop V5.2 before training when generation/audit assumptions failed
- [x] Implement and locally audit the V5.3 pilot-gated 78-task protocol
- [ ] Run the V5.3 24-task pilot and obtain a frozen GO/NO-GO decision
- [ ] If and only if GO, run V5.3 formal generation and pass its 40/48 gate
- [ ] Run the four one-seed 7B QLoRA screening arms plus untrained-base control
- [ ] Confirm the selected method with a preregistered stronger fixed user/judge
- [ ] Replicate a gate-passing mixture on three seeds before unsealing test
- [ ] Run the V4 Clean-SFT / continued-SFT / DPO diagnostic
- [ ] Confirm any screened V4 signal on fresh held-out tasks and three seeds
- [ ] Agent-initiated repair taxonomy and controlled error injection
- [ ] FACES: coverage over error, failed tool, repair action, arguments, and state transitions
- [ ] Executable τ³ evaluation and unseen tool-combination tests
- [ ] Budgeted submodular objective and approximation analysis

## Scope and limitations

The legacy τ-bench repository states that its historical tasks are outdated and recommends τ³-bench for current research. We use the historical corpus only for a no-key, overnight offline pilot. Raw benchmark data and model caches are not committed. Audited formal adapters may appear only on dedicated result branches so that a reported run can be independently checked.

## License

Code in this repository is released under the [MIT License](LICENSE). The source benchmark retains its own license and citation requirements.
