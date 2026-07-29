# V6 Stage 3 Attempt 1 — Candidate Generation Failure

Date: 2026-07-30 (Asia/Shanghai)

Status: **FAIL-CLOSED**

Source commit:
`9f620d438884270f3231101924ff9ef2f6cc5d09`

## Frozen attempt

- Phase: Pilot
- Teacher: pinned Qwen2.5-32B-Instruct-AWQ, tensor parallel 2
- User simulator and judge: pinned Qwen2.5-14B-Instruct-AWQ
- Continuation seeds: `20260806`, `20260807`, `20260808`
- Default clean attempts per task: 2
- Default recovery attempts per branch: 2
- Official test used: false

Both OpenAI-compatible services were healthy. The first 14B request incurred a
normal one-time warm-up delay, after which both models generated successfully.

## Failure

The first generated Pilot task, `airline:12`, produced official reward `0.0`
for both frozen clean attempts. Both attempts contained tool calls, but neither
was an officially successful clean trajectory. The generator therefore raised:

```text
V6GenerationError: airline:12: no successful clean rollout with an observed tool call
```

No candidate pair was emitted. This is the intended fail-closed behavior.

## Diagnosis

- Not an SSH, CUDA, OOM, API, model-loading, or process crash.
- Not caused by missing tool calls.
- The teacher made task-level decisions that failed the official tau2 judge in
  both registered attempts.
- Re-running the identical source, seeds, and attempt count would reproduce the
  same failure and is not justified.
- Increasing the clean-attempt allowance changes the generation contract. It
  must therefore be a separately named diagnostic protocol and cannot be
  merged into the frozen Attempt 1 result.

The LiteLLM "model is not mapped" messages affect only dollar-cost estimation
and did not interrupt inference.

## Preserved evidence

- `reports/artifacts/stage3_attempt1/run_contract.json`
- `reports/logs/stage3_attempt1_generation.log`
- Full tau2 per-attempt logs remain in the immutable RunPod run directory.

## Decision

The original V6 Pilot generation contract cannot proceed beyond the first task.
Preserve this negative infrastructure result. Launch a versioned diagnostic
generation attempt with a larger, preregistered clean-attempt allowance while
keeping the task set, models, revisions, continuation seeds, error construction,
selection methods, and all downstream gates unchanged.

