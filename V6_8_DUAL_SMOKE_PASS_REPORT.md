# V6.8 dual targeted smoke PASS

Date: 2026-07-30 (Asia/Shanghai)

## Decision

Both preregistered V6.8 targeted smokes passed. The complete 24-task Pilot is
authorized under `v6_8_explicit_assertion_renderer_v1`. Training and
official-test unsealing remain unauthorized.

## Frozen inputs

- Preregistration commit:
  `8fd6172236e2f51f3be18b2e5705870fd244fd1f`
- Implementation commit:
  `4a5815333f75ac00d6bc9561f4d0aa8d3bac64f6`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- User/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- Registry internal SHA-256:
  `d93b22dba94db13f6670bb5ef629bf341b0febdfeb2f4798a430dfd52ec9f2bd`
- Registry file SHA-256:
  `dbd2c6fe7176fd121a6e8db38c92778c5e7362f1120ab2e4c9708ea35436779c`
- Semantic generation-contract SHA-256:
  `64fd96d8d3e63027a74c5b7486e729107ac78a54cce8f4aaa64fb20c96bf7f89`

Remote code-only verification passed: 150 tests and 6 subtests.

## `retail:16` blocker smoke

- Candidate pairs: 3/3
- Recovery branches: 6/6
- Forced-first cells: 12/12
- Continuation trials: 36/36
- Every clean, matched-recovery, cell, and trial task success: 1.0
- Every independent replay: PASS
- Failed positive labels: 0
- Official test used: false
- Run-contract file SHA-256:
  `6dce702ae12a3fd94b3994259d2a25c849ef7189d8eb4131879638a49e9ce875`
- Candidate JSONL SHA-256:
  `7adffdd97a0f1d2cbc985664bd1af1bdc67fe9a3d06094ed7037b0bdbed1d8a9`
- Generation-receipt SHA-256:
  `85ac3b945f6ae7fa8799d947f4d5dbe7ae7749a80fa1d6576dab2978d40b2e9a`
- Execution-log SHA-256:
  `d479ea9756a6712ff6a8ccd23b371d9612dbfb1497ae6cb13a04f8bf3960ae40`

The strict NL assertion judge accepted the explicitly user-directed refund
statement. This repairs the exact V6.7 Pilot failure.

## `retail:104` regression smoke

- Candidate pairs: 3/3
- Recovery branches: 6/6
- Forced-first cells: 12/12
- Continuation trials: 36/36
- Every clean, matched-recovery, cell, and trial task success: 1.0
- Every independent replay: PASS
- Failed positive labels: 0
- Official test used: false
- Run-contract file SHA-256:
  `b3744670113235c8598a171dc53e300a80d2fda4722e94772e852322dda9617f`
- Candidate JSONL SHA-256:
  `76f8271ced3320d89ff5eb4960b45ac690e5e69498147761f56a75c89f11f7b8`
- Generation-receipt SHA-256:
  `7211eabd18529194b38aba1fcc56fa3cc28d08aed22f89ecbbed4e269dbf1009`
- Execution-log SHA-256:
  `2a985c0ad4094d287ec9bee6c8195ebdc9bfa3d22454edddcabf76c360b2e8ed`

The V6.8 renderer therefore preserves the prior `retail:104` repair.

## Next

Run all 24 frozen Pilot tasks and 72 candidate pairs in a fresh V6.8 Pilot
directory. Require all task artifacts, the atomic merged JSONL, PASS generation
receipt, exact hashes, replay/label audits, and the frozen Pilot GO/NO-GO gate
before scoring or training.
