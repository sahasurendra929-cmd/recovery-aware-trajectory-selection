# V5.4 Counterfactual Recovery-Branch Pilot

## What this pilot asks

V5.3 generated a clean trajectory and an error trajectory independently.
An eligible pair therefore required two complete simulations to succeed by
chance. V5.4 asks whether we can obtain more valid, diverse recovery pairs per
unit of compute by:

1. obtaining one end-to-end successful clean trajectory;
2. choosing a read-only call using a frozen hash rule;
3. replacing only that call with one schema-valid, task-incorrect error;
4. executing that error against the original task state;
5. deleting the clean suffix;
6. asking the same frozen teacher to recover from the observed error.

The clean and recovery records share the same task and prefix. The failed call
and error result are context, never positive labels.

This is a **data-feasibility pilot only**. It does not train a model, use the
official test, or establish the paper's downstream task-success claim.

## Why the comparison is fair

Both V5.3 and V5.4 have a maximum budget of 288 trajectory slots over the same
24 derived-inner-train tasks. V5.4 registers 8 clean-search and 4
recovery-branch slots per task before outcomes are observed. Unused slots are
terminally accounted and cannot be moved to a favorable task.

The comparison reports:

- eligible pairs per executed slot;
- distinct paired tasks;
- eligible pairs per GPU-hour and GPU-hours per pair;
- real/consequential error pass rate;
- recovery success rate;
- retail and airline coverage.

V5.3's observed 8 paired tasks and 12 capped pairs are historical diagnostics,
not V5.4 training data.

## Frozen state machine

For each task:

1. Execute clean slots 1–8 in order.
2. At the first eligible end-to-end clean success, mark remaining clean slots
   `CLEAN_SUCCESS_EARLY_STOP`.
3. If no clean succeeds, mark recovery slots `NO_CLEAN_PREFIX`.
4. Enumerate read-only calls in the accepted clean trajectory.
5. Select exactly one injection position with
   `SHA256(protocol, task_identity, clean_seed) mod eligible_positions`.
6. Reject before recovery if the perturbation is not schema-valid,
   task-incorrect, consequential, naturalistic, singular, and prefix-replay
   safe.
7. Execute up to four registered recovery seeds.
8. After two eligible pairs, mark remaining branches `PAIR_CAP_REACHED`.

No slot is replaced, rescued, reassigned, or silently omitted.

## What counts as a real error

A primary-gate error must satisfy every item:

- the original clean call was correct;
- the injected call is valid under the same tool schema;
- it is wrong for this task, not merely different;
- the environment produces an observable consequence;
- it matches the frozen natural-error taxonomy;
- the injection target is read-only;
- exactly one error is injected.

Out-of-distribution stress errors may be retained as
`CONTROLLED_STRESS_TEST`, but cannot enter the primary gate.

## Leakage and label audit

Before recovery generation, the branch builder retains only:

- task/system/tool information;
- the clean shared prefix;
- the injected call;
- the actual tool error result.

All clean messages at and after the branch point are removed. The audit stores
SHA-256 values for the shared prefix, removed clean future, recovery prompt,
and injected error event. It checks prefix identity and verifies:

```text
failed_call_supervised = false
error_result_supervised = false
clean_future_present_in_prompt = false
```

Only newly generated post-error recovery assistant actions and final response
may later become supervised targets.

## Pilot preparation and unit tests

From a clean checkout:

```bash
python scripts/prepare_v5_4_pilot.py \
  --output-root artifacts/v5_4_counterfactual_pilot/protocol

python -m pytest -q \
  tests/test_v5_4_pilot_protocol.py \
  tests/test_v5_4_branch_manifest.py
```

The preparation command creates exactly 24 task rows and 288 registered slot
rows. It refuses to overwrite a nonempty protocol directory.

## Branch-builder interface

The clean normalizer must write one JSON object with:

```text
task_identity
seed
eligible_clean_success=true
messages
```

The perturbation executor must write one JSON object containing the frozen
taxonomy fields, the injected tool call, and its actual tool error. Then run:

```bash
python scripts/build_v5_4_branch_manifest.py \
  --clean CLEAN.json \
  --perturbation ERROR.json \
  --recovery-seed SEED \
  --output BRANCH.json
```

The output is a leakage-safe tau2 initial-history request. Its last message is
the actual tool error, so tau2 routes the next turn to the recovery agent. The
environment replays the prefix and checks tool-output consistency.

## Final audit

After all slots have exactly one terminal row and all accepted pairs have
complete strict-judge evidence:

```bash
python scripts/audit_v5_4_pilot.py \
  --slots artifacts/v5_4_counterfactual_pilot/slot_terminal.jsonl \
  --pairs artifacts/v5_4_counterfactual_pilot/eligible_pairs.jsonl \
  --output-root artifacts/v5_4_counterfactual_pilot/final
```

`GO_FULL_V5_4` requires all of the following:

- all 288 slots terminal;
- all claimed pairs pass real-error and leakage audits;
- at least 14 paired tasks and 17 capped pairs;
- more than V5.3's observed 8 tasks and 12 pairs;
- higher eligible-pair yield per executed slot than V5.3;
- both retail and airline represented.

Even a GO keeps `training_authorized=false`. It authorizes freezing a full
V5.4 training/evaluation protocol, not interpreting pilot data as a model
result.

## Required implementation barrier before paid generation

Run one task end to end before the full budget:

1. successful clean trajectory;
2. deterministic read-only position;
3. actual schema-valid error;
4. tau2 prefix replay;
5. fresh 128-token recovery;
6. zero clean-suffix leakage;
7. strict judge and terminal-slot receipt.

Do not launch the 288-slot pilot if this smoke test fails. This barrier is
important because the branch manifest and scientific audit are implemented
here, while the live RunPod adapter must still prove compatibility with the
pinned tau2/vLLM environment.

## Outputs and claim boundary

Push only code, configs, small manifests, receipts, logs, summaries, and
SHA-256 inventories. Do not push weights, model cache, secrets, or raw
unbounded trajectories.

Valid conclusions:

- V5.4 improved or failed to improve quality-adjusted paired-data yield;
- which audit layer removed data;
- whether task/domain coverage improved.

Invalid conclusions:

- recovery training improves task success;
- DPO/SFT/PPO is better;
- official-test performance improved.

Those questions belong to the next full training and end-to-end evaluation
stage.
