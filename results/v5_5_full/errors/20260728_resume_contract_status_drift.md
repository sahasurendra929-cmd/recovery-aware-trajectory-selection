# V5.5 reference-screen resume contract-status drift

## Status

Reproducible post-run implementation defect. The completed retry10 evaluation
artifacts are not invalidated, but a future `--phase evaluate` resume would
reject them.

## Frozen run

- evaluation source commit:
  `ee2d860440d0b082be0c84ab089ca7a049fcb4c8`
- tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- experiment mode: `reference-screen`
- result: 9/9 batches, 27/27 shard metrics, 378/378 rows
- official test used: `false`

## Reproduction

Inspect any completed retry10 shard:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path(
    "results/v5_5_full/evaluation/perfect_success/"
    "20260805/20260815/shard-0/run_contract.json"
)
print(json.loads(p.read_text())["status"])
PY
```

Observed:

```text
COMPLETE
```

At source commit `ee2d860`, `evaluation_batch_complete()` in
`scripts/run_v5_5_full.py` requires:

```python
contract.get("status") == "PASS"
```

Therefore the resume predicate returns false for a protocol-valid completed
shard. The controller would then see the existing output directory and fail
closed with `incomplete evaluation batch exists`.

## Root cause

The strict resume hardening used the metrics status vocabulary (`PASS`) for
the evaluator run-contract status. The evaluator's frozen contract vocabulary
is `COMPLETE`; metrics and summary audits use `PASS`.

## Impact

- No retry10 row, metric, contract, seed, manifest, or model output was changed.
- The uninterrupted retry10 controller did not call the resume predicate after
  each newly completed batch; it waited for successful shard exit codes and
  emitted all nine `EVAL_COMPLETE` events.
- The summary independently accepted exactly 378 rows with no missing,
  duplicate, or extra rows.
- The defect affects only reuse of these completed batches in a later resume.

## Registered minimal fix

Require the exact evaluator contract status `COMPLETE`, retain every other
strict arm/seed/source/official-test invariant, and add a regression test using
a real-protocol-shaped contract. Do not change evaluation data, tasks, seeds,
budgets, statistics, or completed artifacts.
