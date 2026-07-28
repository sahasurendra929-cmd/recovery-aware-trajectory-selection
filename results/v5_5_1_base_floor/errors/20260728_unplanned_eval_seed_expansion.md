# V5.5.1 unplanned evaluation-seed expansion

## Terminal classification

`IMPLEMENTATION_DEFECT_WITH_PRESERVED_EXTRA_DIAGNOSTICS`

The preregistered V5.5.1 base-floor diagnostic specified one evaluation seed,
`20260815`. The controller source at
`1c00867594b304d3191ccdd76016bfdcecf164db` correctly restricted the arm and
training-seed grid, but `phase_evaluate` still iterated the global three-seed
tuple. It therefore ran `20260816` and `20260817` after the preregistered batch.

## Reproduction

```text
/root/v55-train/bin/python-v55 scripts/run_v5_5_full.py \
  --phase evaluate \
  --experiment-mode base-diagnostic \
  --source-commit 1c00867594b304d3191ccdd76016bfdcecf164db \
  --results-root .../results/v5_5_1_base_floor \
  --registry .../results/v5_5_full/checkpoint_registry.json
```

The persistent controller log contains three `EVAL_COMPLETE` events and the
command ledger contains nine shard commands. The output contains nine
`metrics.json` files and 126 rows.

## Impact and evidence handling

- The frozen `20260815` batch completed first and remains independently
  auditable as 3 shards / 42 rows.
- `20260816` and `20260817` are accidental, post-preregistration diagnostics.
  They are preserved without alteration but are excluded from the V5.5.1
  success gate and from any confirmatory claim.
- No official-test task was opened. Every row states
  `official_test_used=false` and `claim_level=diagnostic_only`.
- The healthy controller was not interrupted. It completed naturally, wrote
  all nine metrics, and released all four GPUs.
- Both detached postprocess watchers failed closed rather than summarize the
  wrong grid. Their observed states were respectively:
  `3 batches / 9 commands / 9 metrics / 0 rows` (wrong filename in v1) and
  `3 batches / 9 commands / 9 metrics / 126 rows` (correct filename in v2).

## Root cause and minimal fix

The controller used `full.EVALUATION_SEEDS` directly in both evaluation and
summary dispatch. The fix adds `selected_evaluation_seeds(mode)`, returning
only the first frozen evaluation seed for `base-diagnostic` and the unchanged
three-seed tuple for `reference-screen` and `full`.

The generic formal summarizer also assumes the R0 control is present, so it
cannot validly summarize a base-only diagnostic. A dedicated fail-closed
diagnostic summarizer now requires:

- exactly the preregistered arm and seeds;
- exactly 3 complete shards and 42 unique task-condition rows;
- the exact 21-task clean/error validation grid;
- the frozen evaluation source commit;
- `official_test_used=false` and `claim_level=diagnostic_only` on every row.

Its preregistered diagnostic gate is unchanged: 42/42 rows must be complete
and at least one clean official success must be observed.

## Safe reuse and recovery

Do not rerun training, regenerate data, or rerun `20260815`. Recompute the
V5.5.1 diagnostic summary only from:

```text
results/v5_5_1_base_floor/evaluation/base_control/20260805/20260815/
```

Keep `20260816` and `20260817` in the no-weight evidence package under their
original paths with this report. Future `base-diagnostic` evaluation dispatch
must use the fixed controller and must reject any pre-existing incomplete
target directory before launching a shard.
