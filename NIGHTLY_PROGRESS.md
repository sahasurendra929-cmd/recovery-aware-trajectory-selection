# V6 Nightly Progress

Updated: 2026-07-30 (Asia/Shanghai)

## Current protocol

V6.4 (`v6_4_forced_correction_reference_tail_v1`) is active. V6 through V6.3
remain preserved NO-GO results and are not pooled.

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

The complete fixed 24-task / 72-pair V6.4 Pilot is running in a persistent
remote session. GPU model services remain healthy.

## Next

Run the complete fixed 24-task Pilot, validate all 72 pairs and receipts, then
perform token/hardness measurement and the preregistered Pilot gate audit.
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
