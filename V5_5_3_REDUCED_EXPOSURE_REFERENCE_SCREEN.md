# V5.5.3 reduced-exposure reference screen preregistration

## Prior evidence

Original V5.5 completed the frozen reference screen and returned an all-zero
adapted floor. V5.5.1 established a non-zero unadapted-base capability floor.
V5.5.2 reduced only the learning rate from `1e-4` to `1.25e-5`; its complete
378-row screen retained clean capability, but both recovery arms had an error
success delta of `-0.0417` versus R0. It is permanently recorded as
`NO_GO_REFERENCE_SCREEN`.

## Hypothesis

At the lower learning rate, consuming all 512 scheduled microbatches still
repeats the 48 underlying reference pairs too many times. Halving total
optimizer steps should reduce adaptation-induced interference while retaining
the exact frozen recovery-mixture schedules.

## Single primary change

Relative to V5.5.2, change only the formal optimizer-step budget:

```text
V5.5.2: 64 steps × 8 accumulated microbatches = 512 exposures
V5.5.3: 32 steps × 8 accumulated microbatches = 256 exposures
```

The learning rate remains `1.25e-5`. The 512-row arm schedules remain
byte-for-byte unchanged; deterministic seed `20260805` fixes their sampling
order. This is a training-dose change, not a data-selection change.

## Frozen invariants

- base model and revision;
- LoRA rank, alpha, target modules, quantization, optimizer, scheduler,
  warmup, weight decay, clipping, batch size, and gradient accumulation;
- all 512-row arm schedule files and hashes;
- learning rate `1.25e-5` and training seed `20260805`;
- R0, R50, and R100 arms;
- all validation tasks, evaluation seeds, shards, decoding, simulator, judge,
  parser, fault families, statistical rules, tie rule, and official-test seal
  from V5.5.2.

No data is regenerated, resampled, filtered, or overwritten. Earlier
checkpoints and results remain immutable.

## Success and continuation gate

The original frozen reference-screen gate remains unchanged:

1. all 9 batches / 27 shards / 378 rows are complete and hash-auditable;
2. official-test use is false everywhere;
3. at least one recovery arm passes the frozen clean non-inferiority
   task-bootstrap-CI filter;
4. the selected arm has strictly positive paired error-success delta versus
   R0 (`positive_screen=true`).

Only the complete gate authorizes Stage C under the V5.5.3 protocol.

## Interpretation and terminal rule

This is the third and final permitted diagnostic revision. It remains a
reference-grounded diagnostic and cannot support a natural-recovery claim.
If the gate fails, no further V5.5.x tuning is allowed: preserve the result,
produce the full termination and compute-accounting report, push all
non-weight evidence, verify GitHub readback, and stop GPU consumption.
