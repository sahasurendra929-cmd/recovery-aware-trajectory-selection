# V6.1 Preregistration — 72B Trajectory Teacher

Status: **frozen before any V6.1 model outcome is observed**

Parent source commit:
`9f620d438884270f3231101924ff9ef2f6cc5d09`

## Reason for a new protocol

Frozen V6 failed before candidate-pool construction. Its pinned 32B teacher
received official reward `0.0` in both registered clean attempts for the first
Pilot task, `airline:12`. A post-failure engineering diagnostic, excluded from
scientific use, also received reward `0.0` in eight of eight attempts.

The original V6 result remains a candidate-supply NO-GO. It will not be
overwritten, relabeled, or pooled with V6.1.

## Single scientific change

Replace only the trajectory teacher:

- old: `Qwen/Qwen2.5-32B-Instruct-AWQ`
  revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`
- new: `Qwen/Qwen2.5-72B-Instruct-AWQ`
  revision `698703eae6604af048a3d2f509995dc302088217`

Serving remains AWQ with tensor parallel size 2. The user simulator and strict
judge remain the pinned 14B model on the third GPU.

## Unchanged contract

- tau2 commit and split
- fixed 24 Pilot task IDs
- clean attempts: 2
- recovery attempts: 2
- continuation seeds
- candidate-pair definition and audits
- Pilot and formal GO/NO-GO thresholds
- student model and QLoRA contract
- all selector definitions
- `full_proposed` weights: causal 0.50, hardness 0.25, coverage 0.25
- three directional-screen arms
- training and evaluation seeds
- primary metric and contrast
- clean non-inferiority margin
- task-cluster bootstrap with 10,000 replicates
- official-test sealing and one-time unseal rule

## Hypothesis and stopping rule

The larger teacher may supply successful clean prefixes and fresh recovery
suffixes without changing the selection estimand. V6.1 stops fail-closed under
the same generator, audit, and Pilot gates. No additional clean attempts,
replacement tasks, seed changes, or teacher changes are permitted within this
protocol.

V6.1 results must be reported separately even if the entire downstream screen
completes.

