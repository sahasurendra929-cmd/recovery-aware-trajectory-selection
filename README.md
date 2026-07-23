# Recovery-Aware Trajectory Selection for Tool-Using Agents

This repository studies a simple question:

> Under a fixed training-token budget, which successful tool-use trajectories should be retained so that an agent can handle errors and corrections more reliably?

The project is currently at the **offline pilot** stage. It does not claim an end-to-end agent improvement yet.

## Current experiment versions

- **v1.1:** engineering baseline currently being allowed to finish unchanged;
  its 512-token context protocol is not the paper-quality protocol.
- **v2:** isolated corrected protocol with tokenizer-exact/source-controlled
  selection budgets, full system-policy retention, fixed compute, resumable
  evaluation, a zero-shot control, and task-cluster uncertainty. See
  [`BASELINE_V2_HANDOFF.md`](BASELINE_V2_HANDOFF.md).

Never compare or merge v1.1 and v2 outputs. Their data construction, context,
arm definitions, and test-example counts differ.

## Current evidence-bound claim

In the first τ-bench historical-retail pilot, equal-budget sampling changed the amount and type of error-resolution signal retained in the training subset. A coarse recovery quota alone did **not** improve a transparent offline repair-call predictor. This is useful negative evidence: future selection must cover the error type, failed tool, repair action, arguments, and state transition—not merely increase the number of traces that contain an error.

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
- [ ] Complete QLoRA v2 GPU results across three seeds
- [ ] Agent-initiated repair taxonomy and controlled error injection
- [ ] FACES: coverage over error, failed tool, repair action, arguments, and state transitions
- [ ] Executable τ³ evaluation and unseen tool-combination tests
- [ ] Budgeted submodular objective and approximation analysis

## Scope and limitations

The legacy τ-bench repository states that its historical tasks are outdated and recommends τ³-bench for current research. We use the historical corpus only for a no-key, overnight offline pilot. The raw benchmark data and model checkpoints are intentionally not committed to this repository.

## License

Code in this repository is released under the [MIT License](LICENSE). The source benchmark retains its own license and citation requirements.
