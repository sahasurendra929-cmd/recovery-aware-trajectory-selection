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

The `airline:12` V6.3 smoke finished with exit code 1. Its deterministic clean
replay passed, but both registered fresh recovery attempts received reward
`0.0`; no candidate or task receipt was emitted. This is preserved as a V6.3
recovery-supply NO-GO. GPU model services remain healthy while the separately
versioned successor is prepared.

## Next

Preregister V6.4 with a constrained/reference-completed recovery definition,
test it before observing V6.4 outcomes, then run the same smoke and authorize
the complete Pilot only if every receipt and audit passes.
