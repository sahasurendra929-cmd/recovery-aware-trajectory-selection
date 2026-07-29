# V6.9 complementizer normalization preregistration

Preregistered: 2026-07-30 (Asia/Shanghai), before any V6.9 outcome.

V6.8 is preserved as Pilot `NO_GO` at `retail:19`. Its explicit renderer
produced `I am telling you directly: that returning ...` and
`I am telling you directly: that exchanging ...`; the frozen judge rejected
the facts despite literal presence and otherwise perfect action, DB, and
communicate checks.

## Single scientific change

In `explicit_user_direct_v3`, after removing the exact frozen prefix
`Agent should tell the user `, remove exactly one leading case-sensitive
`that ` from the remaining fact before rendering it as
`I am telling you directly: <fact>`.

No other V6.8 rendering rule changes. The same renderer remains shared by
deterministic clean and recovery confirmations and fails closed on unsupported
assertions.

## Frozen invariants

Task IDs, registry, partition, actions, action ordering, error branches,
prefix, models, revisions, decoding, attempts, continuation seeds, budgets,
gates, selector weights, training design, evaluation design, statistics, and
official-test sealing are unchanged. V6.9 results must not be merged with
V6.8.

## Tests and authorization

Before runtime, test exact normalization for both `retail:19` assertions and
unchanged rendering for assertions without leading `that`; run the complete
V6 suite.

Targeted smokes, in order:

1. `retail:19`;
2. `retail:16`;
3. `retail:104`.

Each must pass all 3 pairs, 6 branches, 12 cells, 36 trials, replay checks,
label audits, and official-test seal checks. Only then may a fresh complete
24-task Pilot start.
