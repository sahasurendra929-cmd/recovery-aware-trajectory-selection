# V6 Nightly Progress

Updated: 2026-07-30 (Asia/Shanghai)

## Current protocol

V6.5 (`v6_5_forced_correction_reference_completion_v1`) is active. V6 through
V6.4 remain preserved NO-GO results and are not pooled.

## Completed

- V6.3 preregistration and implementation frozen at source commit
  `799c05aae9f6a234bf0fbd55cad278d76c791c79`.
- 90 V6 tests and 6 subtests passed locally.
- GitHub push and exact commit readback passed.
- Runtime registry built from the pinned Tau2 commit.
- Registry contains 24 Pilot tasks / 72 Pilot pairs and 50 formal tasks /
  150 formal pairs.
- Official test remains sealed and unused.
- V6.4 `airline:12` smoke passed with 3/3 candidate pairs, 6/6 matched
  recoveries, 12/12 forced-first cells, 36/36 repeated trials, and all replay
  and label audits passing.

## Running

V6.5 Pilot generation has stopped fail-closed at `retail:104`. Seven task
receipts were written before the failure. Both preregistered clean attempts
for the failing task received official reward `0.0`; no merged candidate
JSONL or generation receipt was emitted. GPU model services remain healthy
while the successor protocol is preregistered.

## Next

Preserve and publish the V6.5 NO-GO evidence, preregister one clean-prefix
construction change as V6.6, then run a targeted smoke and restart the fixed
Pilot in a new versioned directory.
# 2026-07-30 — V6.4 Pilot NO-GO

- Restored SSH access after the RunPod migration using the existing local
  V6-specific key and the migrated TCP endpoint.
- The persistent V6.4 Pilot exited `1`; no candidate-generation process
  remains.
- One of 24 task receipts completed (`airline:12`, 3 pairs / 6 branches,
  all matched rewards `1.0`); atomic candidate JSONL was not emitted.
- `airline:14` failed on the first pair because V6.4 forced reference action
  index 1 and replayed only the strict tail, omitting required reference
  action index 0.
- Classified as a scientific/protocol construction defect. V6.4 is NO-GO;
  official test remains sealed and training remains unauthorized.
- Full decision and hashes: `V6_4_PILOT_NO_GO_REPORT.md`.
- Next: preregister V6.5 with one change—forced action first, then all other
  reference actions once in their original relative order.
# 2026-07-30 — V6.5 smoke PASS

- Preregistered V6.5 before observing new outcomes; the only scientific delta
  is forced corrective action first followed by every other frozen reference
  action exactly once in original relative order.
- 97 V6 tests and 6 subtests passed.
- Built a fresh V6.5 registry: 24 Pilot tasks / 72 Pilot pairs / 50 formal
  tasks; official test sealed.
- `airline:14` smoke passed: 3 pairs, 6 branches, 12 forced-first cells, and
  36 repeated trials.
- All matched and crossed trials had reward `1.0`; all independent replays and
  label audits passed. This repairs the exact V6.4 missing-action failure.
- Full evidence: `V6_5_SMOKE_PASS_REPORT.md`.
- Next: complete fixed 24-task V6.5 Pilot in a fresh output directory.

## V6.5 full Pilot started

- Started the complete fixed 24-task / 72-pair Pilot in the persistent
  `v65-pilot` RunPod session.
- Output directory:
  `/workspace/v6_5_20260730/artifacts/v6_5_directional_screen/generation/pilot`
- Completion is determined from the process state, atomic generation receipt,
  and 24 task receipts; directory existence alone is not accepted.
- Official test remains sealed; no training is authorized yet.

## V6.5 full Pilot checkpoint

- Seven of 24 task receipts are complete, representing 21 of 72 candidate
  pairs.
- Completed: `airline:4`, `airline:12`, `airline:14`, `airline:21`,
  `airline:33`, `airline:40`, and `retail:1`.
- The persistent process is currently working on `retail:104`.
- Intermediate receipt audit passes; this is not yet a stage-completion
  decision because the atomic merged generation receipt has not been emitted.

## V6.5 Pilot NO-GO

- Both frozen `retail:104` clean attempts ended at the 60-step budget with
  official reward `0.0`.
- The user simulator repeatedly supplied invented mnemonic order IDs rather
  than one of the real order IDs returned by the environment.
- The generator failed closed; official test remained sealed and no training
  was authorized.
- Full evidence and hashes: `V6_5_PILOT_NO_GO_REPORT.md`.
- Next: preregister V6.6 before implementing or observing successor outcomes.

## V6.6 targeted smoke NO-GO

- V6.6 replaced the stochastic full-task clean source with exactly one user
  turn followed by deterministic reference replay.
- 106 V6 tests and 6 subtests passed after fixing three pre-outcome plumbing
  defects.
- On the valid `retail:104` smoke, DB, all five action checks, and the
  communicate checker passed, but the strict NL assertion judge rejected the
  meta-level completion text in both attempts.
- Official reward was `0.0`; no V6.6 Pilot is authorized.
- Full evidence: `V6_6_SMOKE_NO_GO_REPORT.md`.
- Next: preregister V6.7 with only a direct-natural-language renderer for the
  deleted clean completion message.

## V6.7 targeted smoke PASS

- Implemented the preregistered natural direct completion renderer at source
  commit `6921a3f82f7c1bfc350ee4200ced72899328ffdd`.
- Local and remote code-only suites passed: 136 tests and 6 subtests.
- Rebuilt the runtime environment on Pod-local storage after the migrated
  shared-volume environment failed binary-library and import checks.
- Built a fresh registry with 24 Pilot tasks / 72 Pilot pairs and 50 formal
  tasks / 150 formal pairs; official test remains sealed.
- `retail:104` smoke passed: 3 pairs, 6 branches, 12 forced-first cells, and
  36 continuation trials.
- Every clean, matched-recovery, cell, and trial task-success value was 1.0;
  all independent replays and label audits passed.
- Full evidence: `V6_7_SMOKE_PASS_REPORT.md`.
- Next: start the complete 24-task V6.7 Pilot in a fresh persistent session.

## V6.7 full Pilot started

- Smoke report commit `71085d8887b0b93d8e221f278a1fcf2d930911a6`
  was pushed and read back exactly from GitHub.
- The original frozen V6 branch remained unchanged at
  `9f620d438884270f3231101924ff9ef2f6cc5d09`.
- Started the complete 24-task / 72-pair Pilot in persistent session
  `v67_pilot`.
- The process uses the frozen V6.7 registry, single-turn user prefix,
  deterministic reference completion, natural direct renderer, and the
  preregistered three continuation seeds.
- Completion requires 24 task receipts plus an atomic merged JSONL and PASS
  generation receipt. Official test remains sealed.

## V6.7 Pilot NO-GO

- Ten of 24 tasks and 30 of 72 candidate pairs completed with PASS task
  artifacts before the generator failed closed at `retail:16`.
- The prior blocker `retail:104` passed in the complete Pilot with all clean,
  matched, cell, trial, replay, and label checks passing.
- For `retail:16`, all nine reference actions, DB, and communicate checks
  passed, and the assistant explicitly stated the correct total refund.
- The frozen 14B NL judge nevertheless rejected the statement twice, so both
  clean official rewards were 0.0.
- No atomic merged JSONL or generation receipt was emitted. Official test
  remains sealed and training remains unauthorized.
- Full evidence: `V6_7_PILOT_NO_GO_REPORT.md`.
- Next: preregister a separately versioned V6.8 explicit assertion renderer,
  then run targeted `retail:16` and `retail:104` smokes before a fresh Pilot.

## V6.8 dual targeted smoke PASS

- Preserved and published V6.7 Pilot NO-GO before preregistering V6.8.
- V6.8 changes only deterministic final-message rendering: assertions and
  communicate values are explicitly marked as statements directly to the user.
- Local and remote code-only suites passed: 150 tests and 6 subtests.
- Fresh registry: 24 Pilot tasks / 72 Pilot pairs and 50 formal tasks /
  150 formal pairs; official test remains sealed.
- Both `retail:16` and `retail:104` smokes passed with 3 pairs, 6 branches,
  12 forced-first cells, and 36 continuation trials each.
- Every clean, matched, cell, and trial task-success value was 1.0; all replay
  and label audits passed.
- Full evidence: `V6_8_DUAL_SMOKE_PASS_REPORT.md`.
- Next: start the complete 24-task V6.8 Pilot in a fresh persistent session.

## V6.8 full Pilot started

- Dual-smoke report commit `22dfef8c50b3e429fe7636a09168bc1c575844f3`
  was pushed and read back exactly from GitHub.
- The original frozen V6 branch remained unchanged at
  `9f620d438884270f3231101924ff9ef2f6cc5d09`.
- Started the complete 24-task / 72-pair V6.8 Pilot in persistent session
  `v68_pilot`, with unbuffered pane-level log capture.
- Completion requires 24 task artifacts, atomic merged JSONL, and a PASS
  generation receipt. Official test remains sealed.

## V6.8 Pilot NO-GO

- Eleven tasks / 33 pairs completed with PASS task artifacts.
- `retail:104` and `retail:16` both passed in the complete Pilot.
- `retail:19` matched recovery passed its forced action, all seven action
  checks, DB, and both communicate checks.
- The final message literally contained both required refund/savings facts,
  but the frozen judge rejected the awkward `directly: that ...` sentences.
- No merged JSONL or generation receipt was emitted; official test remains
  sealed and training unauthorized.
- Full evidence: `V6_8_PILOT_NO_GO_REPORT.md`.
- Next: preregister V6.9 with only leading-`that` grammar normalization.

## V6.9 triple targeted smoke PASS

- V6.9 was preregistered before observing any V6.9 outcomes and changes only
  one leading, case-sensitive `that ` after the frozen tell-user prefix.
- A missing 70-task arm-train count in the copied registry config was repaired
  before runtime outcomes; local and remote suites passed 155 tests plus 6
  subtests.
- Fresh registry: 24 Pilot tasks / 72 pairs and 50 formal tasks / 150 pairs;
  official test remains sealed.
- `retail:19`, `retail:16`, and `retail:104` all passed.
- Combined coverage: 9 pairs, 18 branches, 36 forced-first cells, and 108
  continuation trials. All task-success and replay checks passed; failed
  positive labels were zero.
- Full evidence: `V6_9_TRIPLE_SMOKE_PASS_REPORT.md`.
- Next: push/read back this stage and start a fresh complete V6.9 Pilot.

## V6.9 full Pilot started

- Targeted-smoke report commit
  `717981b50db34a09f701ffd35d1ddccc8afa51bb` was pushed and read back
  exactly from GitHub.
- The original frozen V6 branch remained unchanged at
  `9f620d438884270f3231101924ff9ef2f6cc5d09`.
- Started a fresh 24-task / 72-pair Pilot in persistent session `v69_pilot`.
- The run binds the V6.9 registry and semantic generation contract, the
  pinned 72B/14B revisions, three frozen continuation seeds, and the
  `explicit_user_direct_v3` renderer.
- V6.7, V6.8, and targeted-smoke outcomes are not inputs to this run.
- Official test remains sealed. Training remains unauthorized until the
  complete Pilot generation, measurement, scoring, and audit gates pass.

## V6.9 Pilot generation NO-GO

- Fourteen tasks / 42 pairs completed with per-task PASS artifacts.
- Both prior full-Pilot blockers, `retail:16` and `retail:19`, passed.
- Generation then failed closed at `retail:31`: the forced recovery call
  matched a non-singleton number of frozen reference actions.
- No merged JSONL or final generation receipt was emitted. Official test
  remains sealed and training remains unauthorized.
- The 14 task files, ordered hash list, run contract, and full diagnostic log
  are retained on RunPod.
- Full evidence: `V6_9_PILOT_NO_GO_REPORT.md`.
- Next: publish this NO-GO, diagnose reference-slot identity, regression-test
  an implementation repair if the scientific protocol remains unchanged, and
  run a fresh versioned attempt.

## V6.9 registered-reference-slot repair PASS

- `retail:31` contains legitimate repeated read-only reference calls.
- The registry had already frozen unique action indices; runtime now preserves
  those indices through matched recovery and all four forced-first cells.
- Local and remote suites passed 157 tests plus 6 subtests.
- A fresh `retail:31` regression passed 3 pairs, 6 branches, 12 cells, and 36
  trials with all task-success and replay checks passing.
- Full evidence: `V6_9_REFERENCE_SLOT_REPAIR_REPORT.md`.
- Next: start a new full 24-task Pilot attempt without mixing artifacts from
  the earlier source commit.

## V6.9 full Pilot attempt 2 started

- Repair report commit `aa1a81d1c21328bb9dc550ffeea7fbba0c1b07ea`
  was pushed and read back exactly.
- The original frozen V6 branch remained unchanged at
  `9f620d438884270f3231101924ff9ef2f6cc5d09`.
- Started a fresh 24-task / 72-pair run in persistent session `v69_pilot2`.
- Attempt 2 has its own output directory and reruns every task; no task JSON
  from the prior source commit is copied or merged.
- Official test remains sealed and training remains unauthorized.

## V6.9 Pilot attempt 2 NO-GO

- Fifteen tasks / 45 pairs completed with per-task PASS artifacts.
- The repaired repeated-reference task `retail:31` passed in the complete
  Pilot.
- Generation then failed closed at `retail:35` because its frozen reference
  trajectory intentionally starts with an unsuccessful email lookup, while the
  deterministic replay incorrectly prohibited every reference tool error.
- No merged JSONL or generation receipt was emitted. Official test remains
  sealed and training remains unauthorized.
- Full evidence: `V6_9_PILOT_ATTEMPT2_NO_GO_REPORT.md`.
- Next: publish this NO-GO, add a regression fixture, repair expected reference
  error replay without changing the scientific protocol, run a fresh
  `retail:35` smoke, and then start a new complete Pilot attempt.
