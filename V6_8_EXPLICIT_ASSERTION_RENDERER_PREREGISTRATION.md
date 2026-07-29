# V6.8 explicit assertion renderer preregistration

Preregistered: 2026-07-30 (Asia/Shanghai), before any V6.8 outcome.

## Motivation

V6.7 failed closed at the frozen Pilot task `retail:16`. The deterministic
clean replay executed every registered action, reached the correct database
state, passed the communicate check, and ended with the literal sentence
`The total refund amount is $8,276.23.` The frozen 14B NL judge nevertheless
classified the fact as tool output rather than a statement to the user in two
repeated attempts.

V6.7 is preserved as `NO_GO` in `V6_7_PILOT_NO_GO_REPORT.md`. V6.8 results
must not be merged with V6.7.

## Single scientific change

Replace `natural_direct_v1` with `explicit_user_direct_v2` for deterministic
clean and deterministic recovery final messages.

The V2 renderer:

1. removes the generic `The requested work is complete.` boilerplate;
2. renders each `communicate_info` value as
   `I am providing the requested information directly to you: <value>.`;
3. renders `Agent should tell the user <fact>` as
   `I am telling you directly: <fact>`;
4. renders `Agent should provide <fact>` as
   `I am providing this directly to you: <fact>`;
5. applies the already frozen V6.7 deterministic rewrite table to other
   supported assertion shapes, then prefixes the result with
   `I am confirming this directly to you:`;
6. fails closed on every unsupported meta-level assertion shape;
7. uses the same renderer for clean and recovery confirmation.

The renderer is deterministic. It does not call a model, inspect any outcome,
or expose the deleted clean future to recovery generation.

## Frozen invariants

V6.8 does not change:

- the 24 Pilot task IDs or 72 registered candidate pairs;
- the formal pool or partition;
- reference actions, ordering, error branches, or snapshots;
- the single-turn user prefix;
- teacher, user/judge, or student model and revisions;
- decoding, attempt counts, continuation seeds, or rollout budgets;
- Pilot/formal gates or kappa thresholds;
- selector definitions or full-proposed weights;
- training budget, optimizer, seeds, or arms;
- primary metric, clean non-inferiority margin, bootstrap procedure, or
  official-test sealing.

## Pre-outcome tests

Before a V6.8 runtime outcome:

- test exact V2 rendering for `retail:16`;
- test exact V2 rendering for the prior `retail:104` assertion;
- test communicate-only and mixed communicate/assertion cases;
- test every supported V6.7 assertion family;
- test fail-closed behavior for unknown meta-level shapes;
- test that the semantic run contract binds `explicit_user_direct_v2`;
- run the complete V6 code-only suite.

## Runtime authorization

Run two targeted Pilot smokes in fresh V6.8 directories:

1. `retail:16`, the V6.7 Pilot blocker;
2. `retail:104`, the V6.6 blocker and V6.7 smoke task.

Each smoke must contain exactly 3 candidate pairs, 6 branches, 12 forced-first
cells, and 36 continuation trials. Every clean, matched-recovery, cell, and
trial official task-success value must be 1.0; all independent replays and
label audits must pass; official test must remain unused.

Only if both smokes pass may the complete 24-task V6.8 Pilot run in a new
versioned directory. A failure is reported as V6.8 NO-GO before any successor
change.
