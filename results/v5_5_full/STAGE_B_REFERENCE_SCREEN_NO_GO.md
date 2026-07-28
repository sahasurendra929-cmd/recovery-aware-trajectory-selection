# V5.5 Stage B reference-screen terminal decision

## Decision

`NO_GO_REFERENCE_SCREEN`

The Stage B execution is complete and passes its result-integrity audit, but
it does not pass the frozen scientific positive-screen criterion. This is a
diagnostic reference-grounded result and does not authorize a positive model
claim or silent progression under the original V5.5 interpretation.

## Completed design

- arms: R0 perfect-only, R50, and R100 recovery-only
- training seed: `20260805`
- evaluation seeds: `20260815`, `20260816`, `20260817`
- derived-validation tasks: 21
- conditions: clean and frozen controlled error
- batches: 9/9
- shards: 27/27
- task-level rows: 378/378
- task-cluster bootstrap replicates: 10,000
- official test used: `false`
- claim level: `diagnostic_only`

The integrity audit found no missing, duplicate, or extra rows. It also
verified all 162 result-file hashes recorded by the 27 run contracts, exact
arm/seed/evaluation-source identities, unique row keys, a single checkpoint
registry identity, and the official-test seal.

## Frozen results

| Arm | Clean success | In-family error success | Error delta vs R0 |
|---|---:|---:|---:|
| R0 | 0.000 | 0.000 | +0.000 |
| R50 | 0.000 | 0.000 | +0.000 |
| R100 | 0.000 | 0.000 | +0.000 |

The frozen selector reports `r50` only because the registered tie rule chooses
the lower non-zero recovery dose. It simultaneously reports
`positive_screen=false`; therefore `r50` is not a scientifically supported
winner.

## Interpretation boundary

All arms failed every evaluated task in both primary conditions. The result
does not distinguish recovery-dose effects because the reference-screen
checkpoints have no measurable end-to-end task-success floor. It is not
permissible to reinterpret zero-versus-zero as recovery improvement, select a
favorable seed, remove tasks, or weaken the statistical rule.

Before Stage C-E can proceed under a V5.5.x revision, a preregistered minimal
diagnostic must identify why the common task-success floor is zero. Any
revision must change one primary factor, preserve the official-test seal, and
publish its protocol delta and success criterion before GPU execution.
