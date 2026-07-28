# V5.5.1 preregistration: unadapted-base task-success floor

## Original frozen result

V5.5 Stage B completed all 9 batches and passed its 378-row integrity audit,
but every R0, R50, and R100 clean and controlled-error task had official
reward zero. The frozen selector therefore reported
`positive_screen=false`, and the terminal decision is
`NO_GO_REFERENCE_SCREEN`.

This document is committed before any V5.5.1 GPU execution.

## Diagnostic evidence and hypothesis

All checkpoints emit tool calls, so the result is not explained by a silent
endpoint. R0 terminates with 4.1 tool errors per trajectory on average;
R50/R100 terminate with approximately 8.0/8.3. Training expanded 48 source
pairs into 512 rows and 73,248 supervised tokens per arm, while final training
loss was 0.03-0.04. The same unadapted Qwen2.5-7B model had non-zero success in
the earlier Stage-0 infrastructure sample.

Primary hypothesis:

> Reference-path SFT caused a common end-to-end capability collapse; the
> unadapted base model retains a non-zero success floor under the exact V5.5
> derived-validation evaluator.

## Single primary change

Add one diagnostic-only `base_control` arm that routes the already served,
revision-pinned unadapted base model through the existing evaluator.

Everything else remains frozen:

- the same 21 derived-validation tasks;
- clean and controlled-error conditions;
- evaluation seed `20260815`;
- three deterministic shards;
- temperature 0, maximum 512 generated tokens, and maximum 60 steps;
- the same frozen user simulator/judge and revisions;
- the same task/error manifests;
- no training, no checkpoint selection, and no data regeneration;
- no official-test access.

Only one evaluation seed is used because this revision diagnoses the existence
of a task-success floor; it is not an arm comparison or confirmatory result.

## Registered outcomes

`PASS_BASE_FLOOR_DIAGNOSTIC` requires:

- complete 42/42 task-condition rows;
- zero missing, duplicate, or extra rows;
- `official_test_used=false`;
- at least one clean task with official reward 1.

If it passes, the next revision may test one preregistered reduction in
reference-pair repeated exposure. V5.5.1 itself does not establish recovery
benefit and does not authorize Stage C-E.

`FAIL_BASE_FLOOR_DIAGNOSTIC` applies if all clean rewards are zero with an
otherwise valid grid. The next and only justified direction is to diagnose
evaluator/model compatibility or task eligibility before further SFT.

Any infrastructure failure is reported separately and does not count as a
scientific zero.

## Interpretation boundary

The official test remains sealed. A positive base floor identifies a likely
SFT degradation mechanism but cannot rescue the original Stage B result,
select a recovery dose, or support a natural-recovery claim.
