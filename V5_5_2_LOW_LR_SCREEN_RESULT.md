# V5.5.2 low-learning-rate reference-screen result

## Terminal result

`NO_GO_REFERENCE_SCREEN`

The low-learning-rate revision completed all 9 arm-by-evaluation-seed
batches, 27 shards, and 378 task-condition rows. The summary and independent
terminal audit both report that the official test remained sealed.

## Frozen validation result

| Arm | Clean success | In-family error success | Error delta vs R0 | 95% task-bootstrap CI | Clean noninferiority CI |
|---|---:|---:|---:|---:|---|
| R0 | 0.1587 | 0.2292 | 0.0000 | [0.0000, 0.0000] | yes |
| R50 | 0.2222 | 0.1875 | -0.0417 | [-0.2917, 0.2083] | yes |
| R100 | 0.1905 | 0.1875 | -0.0417 | [-0.2917, 0.2083] | no |

The frozen selection rule nominates R50 after the clean filter, but
`positive_screen` is false because neither non-control arm has a strictly
positive paired error-recovery delta. Stage C is therefore not authorized
under V5.5.2.

## Integrity evidence

The versioned terminal audit verifies:

- 27/27 shard metrics are `PASS`;
- 27/27 run contracts are `COMPLETE`;
- every shard contains exactly 14 task rows;
- 378 total rows have no missing, extra, or duplicate grid entries;
- all 27 rows-file SHA-256 values match the frozen summary;
- every metric, contract, and task row reports `official_test_used: false`;
- the result is structurally `PASS` while scientifically
  `NO_GO_REFERENCE_SCREEN`.

## Post-summary watcher defect

The detached watcher initially used the obsolete field path
`statistical_analysis.selection` and raised a preserved `KeyError` after the
summarizer had succeeded. Commit `7422a28` documents and fixes that isolated
handoff defect; commit `e1d846b` adds full-grid hash verification. No training
or evaluation was repeated.

## Next protocol action

This is the second diagnostic revision after the original V5.5 NO-GO.
Stage C–E remain closed. The only remaining permitted diagnostic revision is
V5.5.3, which must be preregistered before execution and must not change the
frozen validation tasks, seeds, statistical rules, or official-test seal.
