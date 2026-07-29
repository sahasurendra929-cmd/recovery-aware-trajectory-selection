# V6.5 Pilot generation NO-GO

Date: 2026-07-30 (Asia/Shanghai)

## Decision

V6.5 `v6_5_forced_correction_reference_completion_v1` is **NO-GO** at
Pilot candidate generation. The fixed 24-task Pilot stopped on `retail:104`
after both preregistered clean-source attempts received official reward
`0.0`.

This is a candidate-supply result, not a selector or training result. V6.5
must not be pooled with V6 through V6.4 or with any successor protocol. The
official test remained sealed and unused; no training or evaluation was
authorized.

## Frozen runtime identity

- implementation commit:
  `941159e5708d0f66b21d8d07cc856031434e9955`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- user/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- registered Pilot size: 24 tasks / 72 candidate pairs
- clean attempts per task: 2
- process result: nonzero exit with fail-closed `V6GenerationError`

## Observed coverage

- completed task receipts: `7 / 24`
- completed candidate pairs: `21 / 72`
- atomic merged candidate JSONL: not emitted
- atomic generation receipt: not emitted
- completed tasks:
  - `airline:4`
  - `airline:12`
  - `airline:14`
  - `airline:21`
  - `airline:33`
  - `airline:40`
  - `retail:1`
- failing task: `retail:104`

All seven completed receipts are internally PASS. Each contains three
candidate pairs, clean official reward `1.0`, matched-recovery reward `1.0`,
passing independent replays, shared sibling prefixes and environment
snapshots, and `official_test_used=false`. Partial receipts do not authorize
the Pilot gate.

## Failure

The source interaction for `retail:104` is long and requires multiple returns,
tracking disclosures, and modification of a pending order. In both frozen
attempts, the user simulator repeatedly supplied invented mnemonic order IDs,
including `PNDG35791`, rather than one of the five real order IDs returned by
the environment. The teacher queried the invented ID at the end of the
60-step budget. Both attempts therefore ended with official reward `0.0`.

The generator correctly failed closed with:

```text
V6GenerationError: retail:104: no successful clean rollout with an observed tool call
```

This is a deterministic clean-prefix supply defect under the frozen
teacher/user interaction contract, not an SSH, CUDA, model-service, replay,
or reference-completion failure. Increasing attempts, changing the task set,
or silently replacing the user simulator after observing this result is not
allowed within V6.5.

## Integrity evidence

| Artifact | SHA-256 |
|---|---|
| Pilot run contract | `e74a522b731d958544bbbe7190a995695dad5d4173d2304cd9425522e5dd8389` |
| Pilot console log | `ffe6a11fb9816aa528f49a3e8a21a7497a8e9b77cff38b796be14753fd0903a9` |
| Pre-pipe console capture | `d033750f1805d0cfd29247bffd0decaac31744f12faad925868470a42a7c0d9c` |
| `retail:104` attempt 1 task log | `3647e7378dc1db49777490a1bb52c3568c9d33e5978df180c0832983cd2ba816` |
| `retail:104` attempt 2 task log | `a8871eb4c21b15bf78d4bae264c375f0588325abf46de2b8247818c92b68c952` |

Completed task receipt hashes:

| Task | SHA-256 |
|---|---|
| `airline:4` | `fbedb30ee2d3f4506f5f35694c5600639a95c170a8b79ae2e859fc16fd0e354a` |
| `airline:12` | `20e2b59f0ddaae4cba5f37bd3dc499cebe2f93ed742497972e4e99d5985ceed8` |
| `airline:14` | `49b22cb6bb79fecd35406aef5a104b7c5f9a83b970ae34f48bd48b0f0499292b` |
| `airline:21` | `9b3d88e94319e15fd25da50b831209b121c339d3c0dcb2c9fd225cb627120c7b` |
| `airline:33` | `ba86c363e3b27e40425e52caece773a3b14eac6de12518da5df6f86be17841a1` |
| `airline:40` | `3582cb04c43690f7e5a7be87bfb9ac9d4d32ee98187b70754e163b1b06a453f9` |
| `retail:1` | `c74bbbe438c206d5f5ee033dca9b2df0e160aedbbae52e9bdfe21b92b78171b6` |

Large task receipts and raw model logs remain on the RunPod volume. They are
not committed to GitHub; the hashes above bind this report to the exact
bytes.

## Successor boundary

A successor must be preregistered and use a new versioned run directory. It
may change the clean-prefix construction so that candidate supply no longer
depends on a stochastic teacher/user conversation completing the entire
task. The task IDs, gates, error branches, deterministic V6.5 recovery
completion, selector weights, student, budgets, evaluation grid, and
statistics must otherwise remain frozen. V6.5 remains a preserved NO-GO
result regardless of the successor outcome.
