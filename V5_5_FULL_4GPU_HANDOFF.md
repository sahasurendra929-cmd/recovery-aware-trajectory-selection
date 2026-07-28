# V5.5 Full Experiment — Four-GPU Handoff

## Scope

This handoff schedules the complete design in
`V5_5_FULL_EXPERIMENT_PLAN.md`. It does not authorize treating the existing
reference-grounded 48-pair pilot as natural-conversation training data.

The current branch freezes the scientific protocol. The natural-conversation
generator, V5.5 arm builder and V5.5 training/evaluation provenance profile
must pass smoke tests in a subsequent implementation commit before formal GPU
work starts.

## Human decision points

Only three decisions require the coordinator:

1. approve the natural-pool audit after the 24-task/48-pair training gate
   passes;
2. approve continuation from the three-arm screen to the five-arm grid;
3. approve the one-time opening of the sealed 60-task official test after
   \(r^\*\), checkpoints and analysis code are frozen.

Agents may perform all other registered steps without changing thresholds.

## Stage B — reference-grounded mechanism screen

The first evaluation does not wait for natural generation. Build the registered
R0/R50/R100 diagnostic arms from the already audited 24-task/48-pair pool, then
train one seed and evaluate derived validation. This guarantees an early
evaluable checkpoint path, but the result must be marked reference-grounded.

## Stage C — natural data construction

Recommended 4×4090 layout:

| GPU | Service/work |
|---|---|
| 0–1 | 32B-AWQ ground-truth-guided teacher, tensor parallel 2 |
| 2 | 14B-AWQ frozen user simulator |
| 3 | frozen judge/replay worker and overflow generation |

The controller shards only inner-train task/seed slots. All workers write
append-only terminal records. A single coordinator merges and audits after all
shards stop.

Required artifacts:

```text
artifacts/v5_5_full/data/
  slot_registry.jsonl
  clean_sources.jsonl
  counterfactual_pairs.jsonl
  pair_audit.json
  cost_report.json
  hashes.json
```

Fail closed unless the audit reports at least 24 task IDs, 48 pairs, both
domains, no more than two pairs per task, no failed positive labels and no
test access.

The 30-task/60-pair paper target is additionally required before official-test
confirmation.

## Stage D — natural fast screen

One run per GPU:

| GPU | Run |
|---|---|
| 0 | R0 perfect-only, seed 20260805 |
| 1 | R50, seed 20260805 |
| 2 | R100 recovery-only, seed 20260805 |
| 3 | tokenizer/data smoke, then evaluation user/judge service |

All three runs use identical optimizer steps and supervised-token budgets.
Evaluate all 21 derived-validation tasks under clean and controlled-error
conditions.

Continue only under the frozen screen rule in the full plan. Do not search for
a different metric after seeing results.

## Stage E — five-arm, three-seed grid

There are 15 independent training runs:

```text
R0, R25, R50, R75, R100
×
20260805, 20260806, 20260807
```

Queue them across four GPUs. Checkpoints must be stored under:

```text
results/v5_5_full/training/<arm>/<seed>/checkpoint_final/
```

Every directory also contains:

```text
command.txt
console.log
training_metrics.json
run_manifest.json
data_identity.json
```

No arm may reuse an adapter from a different seed or data hash.

## Stage E evaluation

For every checkpoint, evaluate:

- all 21 validation task IDs;
- clean;
- controlled in-family error;
- controlled out-of-family error;
- all three frozen evaluation seeds.

Store one terminal row per:

```text
(arm, training_seed, task_id, condition, evaluation_seed)
```

The aggregator must reject missing, duplicate or extra rows. Select \(r^\*\)
using only the frozen validation utility.

## Stage F — sealed confirmation

Before opening the official test, commit:

- selected \(r^\*\);
- R0 and \(r^\*\) checkpoint hashes;
- all evaluator and aggregator source;
- error schedules;
- evaluation seeds;
- primary and secondary estimands.

Then evaluate only R0 and \(r^\*\) on all 60 sealed tasks. Do not retrain,
change prompts or choose another checkpoint after any test result is visible.

## Stop/continue states

| State | Meaning |
|---|---|
| `REFERENCE_SCREEN_COMPLETE` | first diagnostic evaluation is available |
| `FAIL_DATA_GATE` | natural data is insufficient; natural training prohibited |
| `PASS_DATA_GATE` | natural fast three-arm screen authorized |
| `STOP_NULL_SCREEN` | no preliminary recovery signal; report null |
| `GO_DOSE_RESPONSE` | five-arm/three-seed validation authorized |
| `FREEZE_SELECTED_ARM` | \(r^\*\) and code frozen; test may be requested |
| `FINAL_COMPLETE` | 60-task paired confirmation and audit complete |

## What four GPUs do not solve

More GPUs reduce wall-clock time but do not increase the number of independent
tasks. Confidence intervals must cluster by task ID. Repeated seeds and
trajectories cannot be reported as hundreds of independent test examples.
