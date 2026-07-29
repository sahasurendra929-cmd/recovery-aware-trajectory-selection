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

The complete fixed 24-task V6.4 Pilot is authorized next. GPU model services
remain healthy.

## Next

Run the complete fixed 24-task Pilot, validate all 72 pairs and receipts, then
perform token/hardness measurement and the preregistered Pilot gate audit.
