# V6.4 Pilot generation NO-GO

Date: 2026-07-30 (Asia/Shanghai)

## Decision

V6.4 `v6_4_forced_correction_reference_tail_v1` is **NO-GO** at Pilot
candidate generation. The fixed 24-task Pilot was started, but generation
stopped on the second task. V6.4 results must not be pooled with V6, V6.1,
V6.2, V6.3, or any later protocol.

The official test remained sealed and unused. No training, checkpoint
selection, or official evaluation was authorized.

## Frozen runtime identity

- V6.4 implementation commit:
  `50fd9f7ae3bd84f1b3f21afdc8c2d8b481c6302c`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- User/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- Registered Pilot size: 24 tasks / 72 candidate pairs
- Process exit code: `1`

## Observed coverage

- Completed task receipts: `1 / 24`
- Completed candidate pairs: `3 / 72`
- Completed branches: `6`
- Completed JSONL rows: `0` (the atomic merged output was not written)
- Completed task: `airline:12`
- Failing task: `airline:14`
- Failure cell:
  `v6:pilot:airline:14:prefix:initial:candidate-pair:01:branch:1`

The `airline:12` task receipt is internally PASS: all three pairs and six
branches were materialized, every matched replay had official task success
`1.0`, the shared-prefix/snapshot audits passed, and
`official_test_used=false`. This partial receipt does not authorize the Pilot
gate.

## Failure

The failing branch selected `book_reservation` as the forced corrective
action. It is reference action index `1`; reference action index `0` is
`cancel_reservation`. V6.4 executes the forced action and only the reference
actions strictly after its unique reference index. Consequently it omitted
the required cancellation and official task success was not `1.0`.

The generator failed closed with:

```text
deterministic reference tail did not produce a successful matched recovery
```

This is a protocol construction defect, not an SSH, CUDA, model-service, or
random sampling failure. An initial-state prefix combined with arbitrary
registered reference actions is not generally compatible with replaying only
the suffix after the forced action. V6.4 happens to work when omitted earlier
reference actions are unnecessary, but that property is not guaranteed by
the frozen registry.

## Integrity evidence

| Artifact | SHA-256 |
|---|---|
| Pilot run contract | `543c6a08ca59d5046f6ce96aba19c019fcf8b4c78f81bee9689e2b8d041aa7b3` |
| `airline:12` task receipt | `e5fed5621970950c44d202625db1f83d5b1e2a299bfdc7528fbd7ac4798fc285` |
| Pilot console log | `f1445384f8bb036383fa99a27cb475f7d96dfeceaf2634e11855d8ec131ef769` |
| Pilot exit-code file | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |

The large task receipt and raw model logs remain on the RunPod volume. They
are not committed to GitHub because they are large runtime artifacts; their
hashes and semantic audit are recorded here.

## Next protocol

V6.4 is closed as NO-GO. A later protocol may test one explicitly
preregistered change: execute the forced corrective reference action first,
then execute every other frozen reference action exactly once while preserving
the original relative order among those remaining actions. Such a run must
use a new protocol/version directory and retain this failure.

