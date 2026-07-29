# V6.9 `retail:31` slot-fix smoke: crossed-cell failure

The first post-NO-GO regression attempt produced no candidate-pair artifact
and no generation receipt. It verified that the registered reference index
was forwarded through matched recovery, but a crossed forced-first cell still
called deterministic reference completion without the registered index and
failed closed with the original ambiguous-value error.

This is the same implementation defect on a second call path, not a new
scientific result. The attempted run used the unchanged V6.9 registry,
models, revisions, seeds, renderer, and runtime contract.

- run-contract SHA-256:
  `ec78d013a586b0454b91fff526c7c57021daa031cf066c8c74fb72cdc7a1e757`
- diagnostic-log SHA-256:
  `aa27021a67597504f041663949cdfd7bcf0e8662830de00c35287a5b5902bab8`
- candidate output emitted: false
- official test used: false

Recovery: pass each branch's frozen `reference_action_index` into all four
forced-first cells according to the forced action (`a1` or `a2`), retain the
same semantic-call validation, rerun the full V6 code-only suite, and start a
fresh regression output directory.
