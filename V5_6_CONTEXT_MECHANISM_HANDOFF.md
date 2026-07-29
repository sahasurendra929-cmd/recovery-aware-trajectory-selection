# V5.6: Matched versus Shuffled Error Context

## Decision question

V5.5.3 is a valid reference-grounded NO-GO: increasing recovery-data exposure
did not improve controlled-error success.  V5.6 does not tune that result.
It tests the narrower mechanism hypothesis: whether the error result itself is
used when predicting the first correct repair action.

For a held-out pair, the primary paired quantity is:

\[
\Delta_{\mathrm{context}} =
\log p(a_r\mid h,e_{\mathrm{true}})-
\log p(a_r\mid h,e_{\mathrm{shuffled}}).
\]

Both prompts share the same target task prefix and exact `a_r` tokens.  The
shuffled prompt replaces the whole failed-call/error-result event with a
deterministically chosen event from another task in the same domain.

## Frozen arms

| Arm | Clean rows/block | Recovery rows/block | Error context |
|---|---:|---:|---|
| `perfect_success` | 4 | 0 | none |
| `repair_25_true` | 3 | 1 | task-matched error |
| `repair_25_shuffled` | 3 | 1 | cross-task shuffled error |

There are 128 four-row blocks (512 rows per arm).  Every block has the same
target pair in all three arms.  The recovery target remains the original
successful suffix; failed actions and tool errors remain context-only.

## Required gates

- Input pairs must pass the existing independent pair audit and keep official
  test access false.
- A shuffled donor must be in the same domain but have a different task ID.
- The donor failed call and its tool error are moved together.
- Each arm must have exactly the scheduled 0% or 25% recovery supervised-token
  mass, zero failed positive labels, and matching target-pair exposure.
- Derived validation pairs are never used for SFT.  No V5.6 command implements
  an official-test entry point.

## Commands

Materialize training data using the already audited inner-train reference pool:

```bash
python scripts/prepare_v5_6_context_mechanism.py \
  --tau2-root data/raw/tau2-bench \
  --pairs artifacts/v5_5/pairs.jsonl \
  --pair-audit artifacts/v5_5/audit.json \
  --pair-manifest artifacts/v5_5/manifest.json \
  --pair-mode reference \
  --tokenizer-revision <pinned-qwen-revision> \
  --output-dir data/processed/v5_6_context
```

Train each arm with the existing frozen trainer, passing the generated arm
name, its SHA-256, the same 32-step/1.25e-5 V5.5.3 budget, and this source
commit.  The trainer now validates V5.6's distinct audit protocol.

Score a checkpoint on separately constructed/audited derived-validation pairs:

```bash
python scripts/score_v5_6_context.py \
  --tau2-root data/raw/tau2-bench \
  --pairs artifacts/v5_6_validation/pairs.jsonl \
  --pair-mode reference \
  --model-revision <pinned-qwen-revision> \
  --adapter <adapter-dir> \
  --arm repair_25_true \
  --output results/v5_6_context/repair_25_true/scores.jsonl

python scripts/score_v5_6_context.py \
  --score-jsonl results/v5_6_context/repair_25_true/scores.jsonl \
  --output results/v5_6_context/repair_25_true/summary.json
```

The score is a mechanism diagnostic, not end-to-end tau2 task success and not
a confirmation-level claim about natural error recovery.
