# V6.10 exploratory 24-task cross-session analysis

Classification: **EXPLORATORY_NOT_FOR_FORMAL_GATE**

## Scope

This report analyzes the stored multi-session task inventory only. It is an
engineering and candidate-metadata report, not a Compatibility release, Pilot,
formal-pool, training, or paper result.

## Verified inventory

- Terminal task receipts: **24 / 24**.
- Receipt-declared candidate pairs: **69**.
- Directly materialized and parseable candidate pairs: **48**.
- Nonempty candidate files: **16 / 17**.
- Structural parse failures: **0**.
- Duplicate candidate-pair IDs: **0**.
- Required-field violations: **0**.

The materialized candidate inventory is session-stratified: A08 contains
18 receipt-declared pairs and A12 contains
30 receipt-declared pairs. The A05/A07
records declare 21 pairs but are quarantined because their legacy receipt-index hashes do not match.

## Recorded quality metadata

For all 48 materialized pairs:

- real injected error is recorded as executed;
- independent replay and cross replay are recorded as complete;
- matched recovery replay is recorded as successful;
- no future leakage is recorded;
- failed positive-label count is zero; and
- official-test use is false.

Metadata-all-true check for applicable quality flags: **True**.

## What this supports

The evidence supports operational conclusions only: the A08/A12 candidate
files are structurally readable, deduplicated, and carry the expected recovery
metadata. They can be used for session-stratified candidate review, failure
taxonomy, cost/throughput diagnostics, and regression fixtures.

## What this does not support

It does **not** establish that any recovery-path selection rule produces better
training data. No matched training arms, fixed token-budget training runs,
held-out end-to-end evaluation, or valid cross-session joint causal measurement
was run. The closure status remains `FAIL` and joint cross-session measurement is `PROHIBITED`.

Accordingly, formal/Pilot/training authorization is false. Any claim that one
path-selection rule improves model training would require a separately frozen,
controlled training-and-evaluation experiment.

## Recommended economical next step

Stop this analysis Pod after GitHub synchronization. Use the retained catalogue
for offline/manual review. Only fund a new formal run if the project needs a
causal training-quality claim; then preregister a low-budget controlled design
instead of treating this cross-session pool as formal evidence.
