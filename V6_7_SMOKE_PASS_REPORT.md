# V6.7 targeted smoke PASS

Date: 2026-07-30 (Asia/Shanghai)

## Decision

The preregistered V6.7 `retail:104` targeted smoke passed. The complete
24-task Pilot is authorized under
`v6_7_natural_clean_completion_v1`. This decision does not authorize
training or unsealing the official test.

## Frozen inputs

- Source commit: `6921a3f82f7c1bfc350ee4200ced72899328ffdd`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- User/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- Registry internal SHA-256:
  `6f1e93e278fb6e71858aaee7a2387e01ea5d4849c9a9fbdad30998b544609ec4`
- Registry file SHA-256:
  `e670df18eb3fb68b320d57500cf9163b240b20eda26e7c9b0517d05ab0b451e7`
- Run-contract file SHA-256:
  `40f075e4e3cb9118b2cfa0ff7faef0181390987c557f627934305affbdf53890`
- Run-contract semantic SHA-256:
  `ad8f7578e0274524c1c120b90870bbd905ad5d08859f91d8943cfb066a0daf27`
- Candidate JSONL SHA-256:
  `aaa4e4df1b6ae5ede478cf47f91c69a5ed07409147608187b7d0e78f93c6d922`
- Generation-receipt SHA-256:
  `676637739f4c9f29fe400dc7016570a2102cf024eb45b5a8eecd3adc0441035c`

## Engineering recovery

The migrated shared volume was missing NumPy binary-library files from the
old Tau2 virtual environment and then exhibited blocking imports when a
replacement environment was placed on the same shared filesystem. No
scientific input had been consumed at that point. A fresh environment and an
exact checkout of the frozen Tau2 commit were created on the Pod-local
filesystem. Model revisions, data, tasks, seeds, budgets, thresholds, and
protocol semantics were unchanged.

Remote code-only verification passed: 136 tests and 6 subtests.

## Smoke audit

- Expected/completed tasks: 1/1
- Accepted candidate pairs: 3/3
- Recovery branches: 6/6
- Forced-first cells: 12/12
- Continuation trials: 36/36
- Clean official task success: 1.0 for every pair
- Matched-recovery official task success: 1.0 for every branch
- Forced-first task success: 1.0 for every cell and trial
- Independent replay: PASS for every matched recovery and forced-first cell
- Failed positive labels: 0
- Official test used: false
- Training started: false

The natural renderer repaired the exact V6.6 failure: the frozen deterministic
tool actions and final database state remained valid, while the strict
natural-language assertion judge accepted the direct user-facing completion.

## Next

Run all 24 frozen Pilot tasks and 72 candidate pairs in a new V6.7 Pilot
directory. Require an atomic generation receipt, complete task receipts,
matching hashes, all replay/label audits, and the preregistered Pilot gate
before allowing scoring or training.
