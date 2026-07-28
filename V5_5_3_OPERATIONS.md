# V5.5.3 persistent execution and recovery

The final reduced-exposure screen is launched on RunPod with
`scripts/run_v5_5_3_remote_driver.sh` under both `nohup` and `setsid`.
Consequently, the driver and its active controller survive SSH and local
Codex/network disconnection.

The driver runs four restart-separated phases:

1. `train`
2. `registry`
3. `evaluate`
4. `summarize`

It writes the current phase, state, PID, and UTC timestamp to
`results/v5_5_3_reduced_exposure/ops/persistent_driver.status`. The complete
combined log is kept outside Git while running and is copied into the audited
result package afterward.

## Failure and recovery

The driver uses `set -euo pipefail` and stops on the first failed phase. It
does not reinterpret incomplete output as success. Preserve the log and
partial result directory before repairing a deterministic failure.

Recovery starts at the failed controller phase, not at `all`:

- a failed `train` phase may be resumed only after auditing which arm outputs
  are complete; never overwrite a complete checkpoint;
- after complete training, `registry` is safe only when the registry does not
  already exist;
- `evaluate` uses the controller's batch-level completion checks and must
  reuse complete batches rather than deleting them;
- `summarize` runs only after all 9 batches and 27 shards are complete.

Every recovery command must retain the exact pushed `SOURCE_COMMIT`, data
hashes, training seed, evaluation seeds, results root, checkpoint registry,
and official-test seal. No recovery action may delete or overwrite V5.5,
V5.5.1, or V5.5.2 evidence.

After summarize, run the independent terminal audit, package only non-weight
artifacts, push them to the V5.5.3 results branch, and verify GitHub readback.
If the frozen gate fails, this is the third diagnostic revision and the next
action is the final termination report—not another training revision.
