# V5 Stage-1 Full-Generation Pool-Gate Result

Status: **FAIL-CLOSED — do not train the four SFT arms**

The corrected generation pipeline completed all four registered shards on
4×RTX 4090. The subsequent formal audit stopped before data construction or
training because the frozen paired-pool requirement was not met.

This full-run result supersedes the preliminary early-stop counts below. Its
machine-readable record is
`artifacts/v5_stage1_preflight/full_generation_gate_result.json`.

## Corrected full-run evidence

- Generation source: `65362cc8817a20146c86d721f7e329e09c073336`
- Processing/audit source: `e3b940954aa7dcd2d70afdbf1b200942c1767173`
- Benchmark: pinned tau2
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Inputs: 83 derived inner-train tasks × 3 trials × clean/error
- Generation contracts: 4/4 `COMPLETE`
- Official test used: **false**

Across 249 simulations per condition, 42 clean rollouts and 28 error-condition
rollouts passed their condition-specific eligibility checks. Only 18 rollout
seeds were eligible in both conditions:

| Paired slots on one task | Number of tasks |
|---:|---:|
| 0 | 70 |
| 1 | 9 |
| 2 | 3 |
| 3 | 1 |

The frozen gate requires 40 distinct task IDs with three paired slots each
(120 slots total). The observed pool has only one such task. The formal
preparation command therefore stopped with:

```text
RuntimeError: only 1 tasks have 3 eligible paired slots; need 40
```

One clean rollout with terminal reward 1 contained no structured,
result-linked assistant tool action: its would-be calls were serialized into
ordinary assistant text. Processing commit `e3b9409` now excludes this case as
`clean:no_verified_successful_tool_action`; it does not invent a label.
All repository tests pass after the correction: **148 passed, 12 skipped**.

No four-arm dataset, QLoRA checkpoint, or validation metric was produced.
Changing the minimum task/slot gate, resampling until success, or duplicating
the 18 pairs would change the frozen protocol and is intentionally not done.

## Earlier preliminary evidence

The remainder of this document preserves the preliminary preflight record for
historical traceability.

Status: **FAIL-CLOSED — do not train the four SFT arms**

The Stage-1 pipeline successfully generated all four registered inner-train
shards and kept the official test split sealed. Data preparation then rejected
the pool because it contained zero eligible clean/error paired slots, against a
frozen minimum of 120 paired slots from at least 40 task IDs.

## What ran

- Benchmark: pinned tau2 `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Split hash: `a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`
- Tasks: 83 derived inner-train tasks, three trials, clean and controlled-error
  conditions
- Hardware: 4×RTX 4090
- Complete generation source: `fa4494ef6c913a28f7aa5d30986e66e0b6402a8c`
- Official test used: **false**

All four generation contracts reached `COMPLETE`. Their SHA-256 values are
recorded in `artifacts/v5_stage1_preflight/result.json`.

## Fail-closed pool audit

Across 249 clean and 249 error-condition simulations:

| Audit result | Clean | Error |
|---|---:|---:|
| Eligible | 4 | 1 |
| Final failure | 12 | 11 |
| Multiple tool calls in one assistant turn | 111 | 138 |
| Mixed text and tool call | 66 | 37 |
| No messages / infrastructure or invalid GT task | 56 | 61 |
| Not exactly one controlled failure | — | 1 |

No rollout seed was eligible in both conditions:

- paired slots: **0** (required: 120)
- tasks with any eligible pair: **0** (required: at least 40)
- tasks with three eligible pairs: **0** (required: at least 40)

The preparation script therefore stopped before constructing arm data.
No QLoRA training or validation evaluation was started.

## Interface corrections and early-stop evidence

The audit exposed two implementation mismatches:

1. vLLM was served with a 16,384-token context although requests exceeded that
   length.
2. Qwen/Hermes could return parallel calls or text plus a call despite the
   single-tool-action contract.

The following fail-closed/interface fixes were added with regression tests:

- empty and malformed simulations are excluded rather than becoming labels;
- generation and processing commits are tracked separately;
- the vLLM serving context is 32,768;
- `parallel_tool_calls=false` is sent;
- tool calls are executed serially (first ordered call, then replan);
- removed text and deferred calls retain SHA-256 audit metadata.

All V5 tests passed after the final correction: **102 passed, 4 skipped**.

A corrected clean-only early run was stopped after 21 completed cases. Eighteen
were scoreable and four succeeded (22.22%). Error-condition generation had not
started. Even before accounting for the lower expected recovery success rate,
this was incompatible with the frozen requirement that at least 40 tasks each
produce three clean/error successful pairs. Continuing the same run would spend
GPU time without making the registered pool attainable.

## Claim boundary

This result does **not** compare Perfect-success, Failure-raw, Repair-50, or
Repair-100. It does **not** support a claim about whether failure exposure,
failure supervision, or repair-only supervision is better.

It establishes a preflight result: with the registered 14B-AWQ teacher,
7B-AWQ user simulator, three trials, and strict three-pair-per-task pool
criterion, the V5 Stage-1 training pool is infeasible. Reporting four-arm
metrics would require changing the protocol.

## Required decision for a V5.1 run

Freeze one revision before spending more GPU time:

1. use a materially stronger trajectory teacher/user simulator and retain the
   three-pair criterion; or
2. pre-register a larger maximum attempt pool and select a fixed number of
   eligible pairs per task without using validation/test outcomes; or
3. redesign the pool around independently eligible slots and explicitly
   weaken the task-level three-pair requirement.

Option 1 is the cleanest causal continuation. Option 2 is cheaper only if a
pilot demonstrates a much higher paired success probability than observed
here.
