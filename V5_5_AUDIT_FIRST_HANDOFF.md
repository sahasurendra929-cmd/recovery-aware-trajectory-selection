# V5.5 Audit-First Counterfactual Recovery

## Decision

V5.4 is frozen as a data-feasibility result: 22 pairs across 13 tasks, one task
below its registered coverage gate. Its records are not silently relabeled or
used as V5.5 positives.

V5.5 keeps the same research direction—whether recovery-aware trajectory
selection improves tool agents—but changes how a controlled pair is obtained.
It does not wait for two independent LLM trajectories to both succeed.

For a task \(x\), V5.4 approximately paid for:

\[
P(\mathrm{pair}) \approx
P(\mathrm{clean\ success})P(\mathrm{recovery\ success}).
\]

V5.5 first screens a reference solution for an auditable read-only identifier
lookup, then executes:

\[
h \rightarrow a^\star \rightarrow y^\star \rightarrow s
\]

and the counterfactual branch:

\[
h \rightarrow \tilde a \rightarrow e
\rightarrow a^\star \rightarrow y^\star \rightarrow s.
\]

Here \(h\) is the shared prefix, \(\tilde a\) is one schema-valid wrong
identifier call, \(e\) is the actual tool error, \(a^\star\) is the registered
correct lookup, and \(s\) is the remaining reference-action suffix. The failed
call and error result are context, never positive labels.

## What this fixes

- Injection sites are selected before rollout outcomes are observed.
- “Correct”, “wrong” and “recovered” are executed and recomputed by the
  independent auditor, not asserted by the data producer.
- Names and arbitrary strings cannot be mutated; only registered identifier
  fields are eligible.
- The recovery prompt ends at the actual error. The clean suffix is not placed
  in that prompt.
- Two deterministic, distinct mutations are registered per task.
- The official test split remains sealed.
- A CPU-only structural preflight proves that the pool is attainable before
  GPU spending.

The pinned inner-train split contains 45 structurally eligible tasks:
12 airline and 33 retail. A real-tool execution screen rejects reference paths
that themselves contain an error; the current pinned environment leaves
36 executable tasks (12 airline, 24 retail). The frozen manifest then selects
24 tasks and registers 48 pairs.

## Important limitation

The tau2 reference actions are one valid action path used to derive the target
database state. They are **not** a naturally occurring expert conversation.
V5.5 therefore calls the data `reference-grounded`, not `natural error data`.
A successful audit authorizes a training comparison; it is not evidence that
task success improved.

The paper-level claim still requires end-to-end evaluation with a user
simulator on held-out tasks, including clean and controlled-error conditions.

## Frozen gate

Training is authorized only when all checks pass:

- 24 registered tasks and 48 registered pair IDs are present;
- at least 16 tasks have two audited pairs;
- at least 32 pairs pass every audit;
- both airline and retail are represented;
- injected calls produce actual tool errors;
- correction calls succeed;
- clean and recovery branches reach the same reference end state;
- failed events are absent from the positive-label list;
- pair IDs are globally unique and the two error-event hashes are distinct
  within each task;
- official test use is false.

The implementation currently requires all 48 registered rows to be emitted
without an audit failure. The 16-task/32-pair values are scientific coverage
minimums, not permission to hide failed registered rows.

## Commands

Run from the repository root.

```bash
python3 scripts/prepare_v5_5_manifest.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest data/processed/v5_stage0/split_manifest.json \
  --output artifacts/v5_5/manifest.json
```

The command must report 45 structurally eligible tasks, 36 executable eligible
tasks, 24 registered tasks and 48 pairs.

Create the tau2 environment once and reuse it across versions:

```bash
python3.12 -m venv /workspace/venvs/tau2-v55
source /workspace/venvs/tau2-v55/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'data/raw/tau2-bench[dev]'
```

Execute the registered environment pairs:

```bash
python scripts/run_v5_5_reference_pairs.py \
  --tau2-root data/raw/tau2-bench \
  --manifest artifacts/v5_5/manifest.json \
  --output artifacts/v5_5/pairs.jsonl
```

Audit independently:

```bash
python scripts/audit_v5_5_pairs.py \
  --tau2-root data/raw/tau2-bench \
  --manifest artifacts/v5_5/manifest.json \
  --pairs artifacts/v5_5/pairs.jsonl \
  --output artifacts/v5_5/audit.json
```

Do not start training unless the status is
`PASS_TRAINING_AUTHORIZED`.

Run tests:

```bash
python -m pytest -q \
  tests/test_v5_5_protocol.py \
  tests/test_v5_5_manifest.py
```

## Training and evaluation after a pass

The registered comparison is:

1. `perfect_reference`: clean reference-grounded examples only;
2. `recovery_reference`: actual error in context, reference correction and
   suffix supervised;
3. `perfect_recovery_50_50`: equal token/sample budget mixture.

Use Qwen2.5-7B-Instruct and matched optimizer steps/tokens. Model loss is
assistant-tool-call causal cross entropy:

\[
\mathcal L_{\mathrm{SFT}}(\theta)
=-\sum_{t\in M}\log p_\theta(y_t\mid y_{<t}),
\]

where \(M\) excludes the injected failed call and its tool result.

The primary outcome is held-out end-to-end task success. Controlled recovery
success, repeated-error rate, clean success and cost are secondary. The
training adapter and end-to-end evaluator must be released as a separate
frozen commit because the current V5.5 release is the data-construction gate.

## Expected duration

- CPU structural preflight: under one minute.
- 48 local environment replays and audit: usually under 10 minutes.
- environment installation on a fresh pod: roughly 5–15 minutes.
- three 7B QLoRA arms on three 4090-class GPUs: roughly 1–3 hours after the
  training release is frozen.
- end-to-end evaluation is the dominant cost and should be budgeted separately.
