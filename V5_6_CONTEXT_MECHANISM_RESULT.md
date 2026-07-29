# V5.6 Matched-versus-Shuffled Error-Context Result

## Decision

**NO EVIDENCE OF A LEARNED MATCHED-ERROR-CONTEXT MECHANISM.**

All three checkpoints assigned slightly higher probability to the identical
correct repair action under the true error context than under a shuffled
within-domain error context. However, the task-clustered uncertainty intervals
all crossed zero, the sign-flip tests were not significant, and the three point
estimates were nearly indistinguishable. In particular, training on matched
error context did not separate from either flawless-success training or
shuffled-error-context training.

This result is compatible with successful-suffix imitation without reliable use
of the task-matched tool error. It is a derived-validation mechanism diagnostic,
not an end-to-end recovery or task-success result.

## Frozen estimand

For each held-out reference pair:

```text
delta_context =
  log p(correct repair action | target prefix, true error event)
  - log p(correct repair action | target prefix, shuffled error event)
```

The paired independent unit is `task_identity`. The official test remained
sealed and was not accessed.

## Results

| Arm | Steps | Final train loss | Tasks | Mean delta_context | Task-bootstrap 95% CI | Sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| `perfect_success` | 32 | 0.115603 | 13 | 1.136373 | [-0.884875, 3.103464] | 0.310069 |
| `repair_25_true` | 32 | 0.117441 | 13 | 1.170838 | [-0.857792, 3.166181] | 0.300570 |
| `repair_25_shuffled` | 32 | 0.121301 | 13 | 1.150322 | [-0.825813, 3.079475] | 0.295570 |

The matched-context arm exceeds the flawless arm by only `0.034465` log
probability units and the shuffled-context arm by only `0.020516`. These are
descriptive differences; V5.6 preregistered inference concerns each
task-clustered `delta_context`, and none excludes zero.

## Contract and execution audit

- Source commit: `809e1607fbbccc9331050895e006d9c70ed5d87c`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Training seed: `20260805`
- Training: QLoRA NF4, bfloat16 compute, 32 fixed optimizer steps per arm
- Schedule: 512 rows and 73,248 supervised tokens per arm
- Recovery mass: 0% for `perfect_success`, exactly 25% for both repair arms
- Failed actions positively labeled: 0
- Input pool: 48 independently audited reference pairs over 24 inner-train tasks
- Scoring set: 26 derived-validation pairs over 13 task identities
- Official test used: false
- Train/derived-validation overlap: 0
- All three training loss audits: finite
- Preflight: pass

Checkpoint fingerprints:

- `perfect_success`: `d9db8024387c13c2f3f81d32a5eefc61883fb993dbab69867cdc782f00b0ea11`
- `repair_25_true`: `163a50dfd14e4bdc46c5f94a0dfc147efa316ca7093a7cecd000f73f2aadb09d`
- `repair_25_shuffled`: `07a31985024e263a5adca8532cc4a6f8d4aecb41b8485251dbbf337d11eb3ec7`

## Scoring repair discovered during execution

The frozen V5.6 scorer referenced the removed private function
`prepare_v5_sft_causal._token_ids`, so the first scoring attempts failed before
model loading. The result branch contains a minimal repair that renders token
IDs with the same recursive JSON canonicalization, message normalization, tool
schema canonicalization, and chat-template flags as the training data contract.
A regression test directly exercises repair-span extraction. The focused V5.6
suite passes 4/4 tests after the repair.

No training data, checkpoint, score target, seed, evaluation pair, or statistical
procedure changed. Both initially attempted arms were rescored from scratch
after the fix; no partial score file was retained.

## Evidence boundary and next decision

V5.6 does not support the hypothesis that 25% matched recovery supervision
teaches the 7B model to use the semantic content of the observed tool error when
choosing the first correct repair action. Because even `perfect_success`
produced a similar positive point estimate, the shared signal likely reflects
properties of the target prefix or true-versus-shuffled prompt construction
rather than a mechanism learned specifically from matched recovery examples.

Do not escalate this result to a natural-recovery or end-to-end task-success
claim. A follow-up should first improve the diagnostic's sensitivity—for
example, by using harder task-matched donors or an interaction estimand that
directly contrasts the change from `perfect_success` to `repair_25_true`
against the change to `repair_25_shuffled`—before spending on a larger
confirmation run.
