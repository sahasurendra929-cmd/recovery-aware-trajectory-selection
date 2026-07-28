# V5.5.2 post-summary audit field-path failure

## Failure evidence

After all 9 evaluation batches, 27 shard metrics, and 378 task rows completed,
the detached recovery watcher successfully ran the frozen summarizer. Its
inline post-summary audit then raised:

```text
KeyError: 'statistical_analysis'
```

The traceback remains in `/workspace/v5_5_logs/v552_recovery.log`. The
generated summary was not deleted or rewritten.

## Root cause and impact

The watcher expected `summary["statistical_analysis"]["selection"]`, while the
versioned `v5_5_task_cluster_summary_v1` schema stores the same field at
`summary["statistics"]["selection"]`. This defect is limited to the handoff
watcher's final status write. Evaluation, the summarizer, raw task rows,
metrics, hashes, and the official-test seal are unaffected and remain
reusable.

## Minimal fix and recovery

`scripts/audit_v5_5_screen_terminal.py` validates the actual versioned summary
schema, separates integrity failure from a valid scientific NO-GO, and writes
one of `GO_STAGE_C`, `NO_GO_REFERENCE_SCREEN`, or `FAIL_CLOSED`. Regression
tests cover all three paths.

Recovery must reuse the completed summary and run only this terminal audit.
It must not retrain, rerun evaluation, or modify frozen seeds, tasks, budgets,
statistics, or gates.
