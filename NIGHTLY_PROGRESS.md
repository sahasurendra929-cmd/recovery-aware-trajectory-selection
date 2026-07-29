# V6 Nightly Progress

Updated: 2026-07-30 (Asia/Shanghai)

## Current protocol

V6.3 (`v6_3_deterministic_reference_clean_replay_v1`) is active. V6, V6.1,
and V6.2 remain preserved candidate-supply NO-GO results and are not pooled.

## Completed

- V6.3 preregistration and implementation frozen at source commit
  `799c05aae9f6a234bf0fbd55cad278d76c791c79`.
- 90 V6 tests and 6 subtests passed locally.
- GitHub push and exact commit readback passed.
- Runtime registry built from the pinned Tau2 commit.
- Registry contains 24 Pilot tasks / 72 Pilot pairs and 50 formal tasks /
  150 formal pairs.
- Official test remains sealed and unused.

## Running

The `airline:12` V6.3 smoke is running in a persistent remote session.
Its deterministic clean replay has already executed successfully and the
generator has advanced into fresh recovery rollout generation. GPU model
services remain healthy.

## Next

Validate the smoke task receipt, pair count, replay hashes, matched/crossed
cells, and exit status. If it passes, publish a smoke report and begin the
complete 24-task Pilot. If it fails, preserve the evidence and repair only an
identified implementation fault without changing the frozen V6.3 contract.
