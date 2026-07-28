# V5.5 reference-screen: reproducible first-attempt context-window overflow

## Status

Permanent evaluation failure on evaluation seed `20260817`. The frozen runner's
built-in second attempt completed the affected task for seed `20260816`, but both
attempts exceeded the context window for seed `20260817`. The result-interface
audit then rejected the failed simulation because it lacked messages, and the
evaluation controller exited after cleaning up its services.

## Source and protocol

- Evaluation source commit:
  `f7a76299ce4b8d2af97d6bf5cebb4486d5f59db8`
- tau2-bench commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Experiment mode: `reference-screen`
- Arm: `repair_50`
- Training seed: `20260805`
- Affected task: retail Task 95
- Agent model: `v55-r50-seed-20260805`
- Agent endpoint for shard 1: `http://127.0.0.1:8102/v1`
- User/judge endpoint: `http://127.0.0.1:8201/v1`
- Frozen maximum context length: 32768 tokens

Controller launch command:

```text
/root/v55-train/bin/python-v55 scripts/run_v5_5_full.py --phase evaluate --experiment-mode reference-screen --source-commit f7a76299ce4b8d2af97d6bf5cebb4486d5f59db8 --tau2-root /workspace/repos/tau2-bench --train-python /root/v55-train/bin/python-v55 --serve-python /workspace/venvs/v5_4_4500/bin/python-v55serve
```

## Reproduction evidence

The same first-attempt request was rejected in two consecutive evaluation
batches:

1. Evaluation seed `20260816`, shard 1, at `2026-07-28 11:48:41 UTC`.
2. Evaluation seed `20260817`, shard 1, at `2026-07-28 12:06:56 UTC`.

Both logs report:

```text
Task 95 failed (attempt 1/2)
ContextWindowExceededError
maximum context length is 32768 tokens
request has 33002 input tokens
```

Remote evidence:

```text
results/v5_5_full/evaluation/repair_50/20260805/20260816/shard-1.console.log
results/v5_5_full/evaluation/repair_50/20260805/20260817/shard-1.console.log
```

The exact shard command for the second occurrence was:

```text
/workspace/venvs/v5_4_4500/bin/python-v55serve scripts/run_v5_5_end_to_end_eval.py --tau2-root /workspace/repos/tau2-bench --split-manifest /workspace/repos/recovery-aware-trajectory-selection-v55/artifacts/v5_stage0/manifests/split_manifest.json --validation-manifest /workspace/repos/recovery-aware-trajectory-selection-v55/data/processed/v5_stage1_protocol/validation_manifest.json --protocol-audit /workspace/repos/recovery-aware-trajectory-selection-v55/data/processed/v5_stage1_protocol/audit.json --checkpoint-registry /workspace/repos/recovery-aware-trajectory-selection-v55/results/v5_5_full/checkpoint_registry.json --evaluation-source-commit f7a76299ce4b8d2af97d6bf5cebb4486d5f59db8 --arm repair_50 --training-seed 20260805 --evaluation-seed 20260817 --agent-api-base http://127.0.0.1:8102/v1 --user-api-base http://127.0.0.1:8201/v1 --output-dir /workspace/repos/recovery-aware-trajectory-selection-v55/results/v5_5_full/evaluation/repair_50/20260805/20260817/shard-1 --condition both --shard-index 1 --num-shards 3
```

## Observed recovery

For evaluation seed `20260816`, the frozen runner emitted `Retry 1/1`, reran
Task 95, and completed shard 1 with:

- `status: PASS`
- `rows: 14`
- `official_test_used: false`

For evaluation seed `20260817`, the second attempt reproduced the exact same
`33002 > 32768` rejection at `2026-07-28 12:08:29 UTC`. The runner emitted:

```text
Task 95 failed after 2 attempts
Task 95 failed permanently after 2 attempts
```

The remaining task in that condition completed, after which the result-interface
audit failed:

```text
RuntimeError: retail_error.shard-001-of-003.json simulation[4] lacks messages
```

The controller propagated the shard exit:

```text
RuntimeError: evaluation failed with exit 1:
/workspace/repos/recovery-aware-trajectory-selection-v55/results/v5_5_full/evaluation/repair_50/20260805/20260817/shard-1.console.log
```

At failure time, shard 0 and shard 2 had each produced 14 rows and metrics.
Shard 1 retained its console log, run contract, retail-clean output, and the
retail-error output containing the failed simulation. No evaluation processes or
vLLM services remained after controller cleanup.

## Handling decision

No seed, manifest, shard assignment, context limit, max-steps setting, or frozen
protocol was changed before this report was updated and pushed. Evaluation seed
`20260817` cannot complete under the current implementation because the same
request deterministically exceeds the serving limit on both attempts. A fix must
preserve the frozen task set and produce all 14 shard rows; skipping the failed
simulation or accepting a missing row is not permitted.
