# Recovery-Aware Trajectory Selection for Tool-Using Agents

This repository studies a simple question:

> Under a fixed training-token budget, which successful tool-use trajectories should be retained so that an agent can handle errors and corrections more reliably?

The project is currently at the **offline pilot** stage. It does not claim an end-to-end agent improvement yet.

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

Never compare or merge v1.1 with v2/v3/v4 outputs. V3 may be paired only with the
audited V2 `random_success` result because those two share the frozen examples
and evaluation protocol. V4 reuses the V3 selection and the same 959-example
generation evaluator, while adding objective-aligned outcome annotations.

## Current evidence-bound claim

In the first τ-bench historical-retail pilot, recovery enrichment changed the
model's action prior but did **not** improve offline repair-call exact match.
Distribution constraints reduced the overall damage but still produced no
recovery gain. V4 therefore tests the narrower mechanism suggested by the
error analysis: failed calls should not remain positive SFT labels, and an
observed successful repair should be preferred to replaying the failed call in
the same post-error context.

## Repository map

```text
configs/          frozen experiment contracts
data/             source and schema documentation; no benchmark data is committed
scripts/          reproducible offline-pilot code
results/          versioned, observed pilot outputs
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
