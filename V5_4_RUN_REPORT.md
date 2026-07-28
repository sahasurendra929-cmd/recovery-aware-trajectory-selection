# V5.4 Counterfactual Pilot Run Report

## Status

The frozen V5.4 pilot completed operationally, but its confirmatory decision is
`NO_GO_STOP`. This is a data-feasibility result, not an infrastructure failure,
and it does not authorize training.

- Registered terminal slots: 288 / 288
- Execution errors: 0
- Eligible recovery pairs: 22
- Tasks with at least one eligible pair: 13 / 24
- Frozen minimum task coverage: 14 / 24
- Official test use: false
- Training authorized: false
- Total GPU time: 4.271849377382443 GPU-hours

All frozen audit checks passed except `tasks_with_pair_at_least_14`. The
eligible-pair count exceeded its minimum of 17, but task coverage missed its
minimum by one task. The original V5.4 decision must therefore remain
`NO_GO_STOP`; lowering the threshold or selectively rerunning failed seeds
after seeing the result would invalidate the frozen evaluation.

## Runtime correction applied before the pilot

The original recovery runner inserted an assistant tool call into Tau2 message
history without its corresponding `ToolMessage`. Tau2 then failed with:

```text
ValueError('Tool message expected. Got None.')
```

The committed runner replays the clean prefix in a fresh environment, executes
the injected read-only call, and inserts the paired assistant tool call and
actual tool result into recovery history. `tests/test_v5_4_live_runner.py`
contains the regression coverage. The smoke run and all 288 registered pilot
slots completed after this correction.

## Post-run protocol defect discovered

The mutation fallback is broader than the declared
`wrong_identifier_same_type` taxonomy. When a selected read-only call has no
identifier-like argument, `mutate_identifier()` falls back to changing the
first non-empty string argument.

For `retail:69`, the selected clean call was:

```text
find_user_id_by_name_zip(first_name="Emma", last_name="Smith", zip="10192")
```

The injected call changed `first_name="Emma"` to `first_name="Emm0"`. This is
an arbitrary string/name corruption, not a same-type wrong identifier. All four
recovery attempts for this task were ineligible.

Two related audit weaknesses should also be corrected in a successor protocol:

1. Read-only injection-site eligibility checks the tool name but not whether the
   call contains a valid identifier field.
2. `clean_action_correct` and `injected_action_task_incorrect` are asserted by
   the runner rather than independently established by the audit.

This defect was found after the frozen V5.4 decision and must not be used to
rewrite that outcome.

## Recommended V5.4.1 changes

1. Restrict injection candidates to read-only calls containing explicitly
   allowlisted identifier fields such as `order_id`, `reservation_id`,
   `user_id`, and `flight_number`.
2. Remove the arbitrary-string fallback. Skip a call when no valid identifier
   can be mutated.
3. Derive and validate the perturbation taxonomy from the actual mutated field.
4. Independently verify clean-call correctness and injected-call
   task-incorrectness.
5. Persist full recovery traces and structured ineligibility reasons.
6. Freeze a new V5.4.1 protocol and rerun all 24 tasks independently.

## Artifact layout

- `artifacts/v5_4_counterfactual_pilot/full/run_receipt.json`: full-run receipt
- `artifacts/v5_4_counterfactual_pilot/full/slot_terminal.jsonl`: terminal slot ledger
- `artifacts/v5_4_counterfactual_pilot/full/eligible_pairs.jsonl`: 22 audited pairs
- `artifacts/v5_4_counterfactual_pilot/full/clean/`: 16 clean source trajectories
- `artifacts/v5_4_counterfactual_pilot/full/final/`: audit, cost, coverage,
  decision, and checksums
- `artifacts/v5_4_counterfactual_pilot/smoke/run_receipt.json`: passing smoke receipt

The checksums in `full/final/SHA256SUMS` were verified against the downloaded
RunPod artifacts before the Pod was stopped.
