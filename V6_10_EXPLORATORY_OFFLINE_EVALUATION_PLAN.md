# V6.10 exploratory offline recovery-evaluation plan

Status: **planned development experiment; not a V6.10 formal stage**.

## Purpose

The completed A08/A12 training screen established that the two frozen 24-pair
training arms can train successfully.  Its official end-to-end evaluator cannot
legally consume cross-session exploratory adapters, so this follow-up uses an
explicitly *offline* held-out diagnostic.  It is intended to choose the most
useful hypothesis and design for a later continuous-runtime experiment, not to
claim task success or path-selection superiority.

Every output must carry:

```text
classification = EXPLORATORY_NOT_FOR_FORMAL_GATE
```

## Inputs and immutable boundary

Use only the existing hash-indexed A08/A12 materialized pool and the adapters
already produced by the training screen:

- sessions: A08 and A12 only;
- train adapters: `R_random_stratified` and
  `S_session_normalized_recorded_outcome_rank`;
- held-out task identities: `retail:31`, `retail:11`, `retail:104`, and
  `retail:72`;
- held-out instances: all three candidate pairs and both sibling branches for
  each task (24 recovery-prefix instances total);
- baseline: the same frozen, unadapted Qwen2.5-7B revision used for training.

The A05/A07 quarantine, empty A08 `retail:16` candidate file, original
72B/14B services, official test, and formal evaluator remain excluded.

## Evaluation object

For every held-out branch, construct the exact frozen recovery prefix ending
with the recorded failed assistant tool call and error result.  The evaluator
provides the system instruction and registered tool schemas, but never reveals
the recorded recovery suffix to the model at generation time.

Generate one continuation from each of the three models (base, R, S) at two
fixed decoding seeds.  Store the raw generated continuation only on the
RunPod volume.  The GitHub evidence contains hashes and compact per-instance
metrics only.

## Descriptive metrics

Metrics are diagnostics, not environment task success:

1. **JSON/tool-call validity** — a continuation contains at least one parsable
   tool call whose name appears in the branch's registered schema.
2. **First-repair action agreement** — the first generated valid tool-call
   name and canonical arguments exactly match the frozen recorded recovery
   suffix's first action.  This is reference agreement, not an oracle reward.
3. **Repeated-failure avoidance** — the first generated tool call is not an
   exact repeat of the known failed call.
4. **Recovery continuation NLL** — teacher-forced negative log likelihood of
   the recorded recovery suffix under each model.  This is reported separately
   from free generation and cannot be interpreted as task success.
5. **Length and generation failures** — generated tokens, parse failures,
   out-of-schema calls, and CUDA/runtime failures.

Report each metric by task, session, branch, decoding seed, and model.  Do not
pool raw forced-first outcome values across A08 and A12.

## Controls

- Same base revision, tokenizer, prompt renderer, maximum new tokens, decoding
  settings, device, and random seeds for base/R/S.
- Adapter loading is isolated: one model instance at a time; clear CUDA cache
  between model variants.
- Verify every adapter checkpoint manifest before use.  Abort on a missing or
  mismatched file hash.
- Save an atomic progress receipt after each `(model, task, pair, branch,
  seed)` item, so a Pod interruption resumes only unfinished items.
- Treat any failed item as a typed failure; never replace it with a score.

## Interpretation and decision

Possible outcomes are strictly limited to:

- `PROMISING_OFFLINE_EXPLORATORY_SIGNAL`: S has a consistent descriptive
  advantage over R across more than one held-out task without worse
  repeated-failure avoidance or validity;
- `INCONCLUSIVE_OFFLINE_EXPLORATORY_SIGNAL`: mixed, tied, or incomplete
  diagnostics; or
- `NO_OFFLINE_EXPLORATORY_SIGNAL`: S is descriptively worse on the primary
  diagnostics.

These labels do **not** establish recovery-task success, causal mechanism,
training-data superiority, formal Compatibility, Pilot readiness, or paper
evidence.  A promising result only justifies a subsequent continuous-runtime,
pre-registered end-to-end evaluation.

## Runtime and cost

One 24-GB GPU is sufficient: RTX PRO 4000 or RTX 4090, with the existing
network volume mounted at `/workspace`.  The Qwen 7B base must be re-downloaded
to local ephemeral disk when a new Pod starts; adapters and inputs remain on
the network volume.  Expected wall time is approximately 2–5 GPU hours,
depending on prompt length and generation limits.  No 72B or 14B model is
needed.

## Required deliverables

- frozen offline-evaluation manifest and SHA-256;
- per-item atomic receipts and failure ledger;
- adapter checkpoint verification index;
- credential-scanned metric summary and final report;
- an explicit `EXPLORATORY_OFFLINE_EVALUATION_COMPLETE` or typed blocked
  result; and
- GitHub sync of small manifests, receipts index, hashes, summary, and report.

Raw continuations, token-level outputs, and adapter weights remain on the
RunPod network volume unless an independently approved large-artifact archive
is configured.
