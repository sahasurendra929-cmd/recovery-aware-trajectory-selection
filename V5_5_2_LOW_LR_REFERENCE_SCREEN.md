# V5.5.2 low-learning-rate reference screen preregistration

## Prior evidence

Original V5.5 Stage B completed its frozen 3-arm × 3-evaluation-seed grid and
returned zero clean and error success for every adapted arm. It is permanently
recorded as `NO_GO_REFERENCE_SCREEN`.

V5.5.1 then evaluated the unadapted, revision-pinned base on the same frozen
validation environment. Its preregistered 42-row grid was complete and
observed 3/21 clean and 4/21 error successes. The base-floor gate passed. This
shows that the evaluator can produce successes and makes adaptation-induced
capability collapse the leading explanation for the original all-zero result.

## Hypothesis

The original QLoRA update magnitude was too large for the small 48-pair
reference-grounded pool. Reducing only the learning rate should retain more of
the base model's clean capability while still allowing the recovery-mixture
arms to learn from their frozen schedules.

## Single primary change

Change the formal QLoRA learning rate:

```text
V5.5:   1.00e-4
V5.5.2: 1.25e-5
```

This is an eight-fold reduction in update magnitude. No other training,
evaluation, data, decoding, or statistical setting changes.

## Frozen invariants

- base model and revision;
- LoRA rank, alpha, target modules, quantization, optimizer, scheduler,
  warmup, weight decay, clipping, batch size, and gradient accumulation;
- 512-row arm schedules and their exact hashes;
- 64 optimizer steps and one frozen training seed (`20260805`);
- R0, R50, and R100 arms;
- 21 derived-validation tasks under clean and injected-error conditions;
- evaluation seeds `20260815`, `20260816`, and `20260817`;
- 3 shards per batch, 60 interaction steps, 512 generation tokens;
- fixed user simulator, strict judge, tool parser, fault families, and
  official-test seal;
- task-level paired statistics, 10,000 task-cluster bootstrap replicates,
  clean non-inferiority margin, Holm adjustment, and tie rule.

Training data is reused byte-for-byte. It is not regenerated, resampled, or
filtered. Original V5.5 checkpoints and results are not overwritten.

## Success and continuation gate

V5.5.2 uses the original frozen reference-screen gate:

1. all 9 batches / 27 shards / 378 rows are complete and hash-auditable;
2. official-test use is false everywhere;
3. at least one non-control arm passes the frozen clean non-inferiority
   task-bootstrap-CI filter;
4. the selected arm has strictly positive paired error-success delta versus
   R0 (`positive_screen=true`).

Only that full gate authorizes progression to Stage C under the V5.5.2
protocol. A non-zero score by itself is diagnostic and does not authorize
Stage C.

## Interpretation boundaries and failure handling

This remains a reference-grounded diagnostic screen, not a confirmatory
natural-recovery result. It cannot support a formal recovery-training claim.

If the gate fails, V5.5.2 is recorded and pushed as NO-GO without changing the
threshold. One final V5.5.3 revision may then test a preregistered reduction in
repeated pair exposure; no seed selection, task deletion, official-test
tuning, or post-hoc statistical change is allowed.
