# V5 Stage 0 result report

## Outcome

Stage 0 passed its infrastructure and measurement gates. It produced 20
complete end-to-end τ²-bench trajectories on ten paired development tasks:
ten clean runs and ten runs with one deterministic read-only tool failure.
There were no infrastructure, CUDA, model-server, or context-window failures.

This is an end-to-end baseline, not evidence for the recovery-aware training
claim.

## Frozen setup

- Benchmark: τ²-bench v1.0.1 at
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Run commit: `8e01f344737872dff02a155dd7dfd1ed5490fff8`
- Seed: `20260722`
- Development tasks: seven retail and three airline tasks, sampled only from
  the official training split
- Official test split: sealed and unused
- Model: Qwen2.5-7B-Instruct
- Hardware: NVIDIA GeForce RTX 4090, 24,564 MiB
- Serving: vLLM 0.10.2 with Transformers 4.55.4
- Primary metric: official τ²-bench composite task reward

## Results

| Measurement | Clean | Error-injected |
|---|---:|---:|
| Full task success | 1/10 (10%) | 1/10 (10%) |
| Max-steps termination | 4/10 | 1/10 |
| Too-many-errors termination | 1/10 | 1/10 |
| User-stop termination | 5/10 | 8/10 |
| Prompt tokens | 1,895,649 | 860,999 |
| Completion tokens | 29,195 | 12,649 |

Paired outcomes were: zero tasks succeeded in both conditions, one succeeded
only when clean, one succeeded only after the injected error, and eight failed
in both. The average success delta is therefore zero, but the two conditions
did not succeed on the same task. With ten pairs, this is descriptive only.

All ten frozen injected calls produced real tool errors. None repeated the
identical failing call. Six of ten error trajectories later produced at least
one valid tool result. An independent preflight executed all ten injections
against fresh task environments and confirmed that both the agent and user
database hashes remained unchanged in every case.

The 20 simulations took 704.1 seconds of summed task runtime. The error runs
used fewer tokens mainly because four retail trajectories stopped almost
immediately after the injected failure; this must not be presented as a
cost-efficiency improvement.

## Interpretation

The positive result is that the project now has a reproducible end-to-end
agent evaluation path. It measures actual task completion, executes controlled
tool failures, preserves the database, and retains full trajectories for
auditing.

The scientific recovery hypothesis is not confirmed or rejected. A 10% base
success rate creates a floor effect: most tasks fail for general planning and
tool-use reasons before recovery quality can be isolated. The equal 10% rates
also hide completely discordant paired successes, so reporting only the mean
would be misleading.

The next experiment should first establish a stronger end-to-end base model.
After that gate is passed, compare recovery-data mixture ratios rather than an
all-recovery arm, use substantially more paired trajectories and seeds, and
keep official composite task success as the primary outcome. Recovery
diagnostics remain secondary mechanism measurements.

## Artifact map

- `artifacts/v5_stage0/stage0_summary.json`: aggregate and paired metrics
- `artifacts/v5_stage0/injection_audit.json`: ten-call read-only audit
- `artifacts/v5_stage0/raw/`: complete τ²-bench result JSON and per-run logs
- `artifacts/v5_stage0/manifests/`: split, smoke, and data-audit manifests
- `artifacts/v5_stage0/runtime_versions.json`: frozen runtime versions
- `artifacts/v5_stage0/full_run.log`: complete batch log

Raw result hashes are recorded in `stage0_summary.json`.
