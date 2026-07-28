# V5.5.1 terminal state

## `PASS_BASE_FLOOR_DIAGNOSTIC`

The preregistered unadapted-base diagnostic completed on the exact 21 frozen
derived-validation tasks under clean and injected-error conditions.

- expected/observed rows: `42 / 42`
- clean successes: `3 / 21` (`14.29%`)
- error successes: `4 / 21` (`19.05%`)
- unique task-condition rows: `42`
- verified registered result hashes: `18 / 18`
- official test used: `false`
- claim level: `diagnostic_only`

The preregistered gate required a complete 42-row grid and at least one clean
official success. It therefore passes.

## Interpretation boundary

The unadapted frozen Qwen2.5-7B base has a non-zero capability floor under the
same evaluator, tasks, decoding, user simulator, strict judge, and error
protocol for which all three original V5.5 adapted arms scored zero. This
supports the diagnosis that the original Stage B zero floor is consistent
with adaptation-induced capability collapse, rather than an evaluator that is
incapable of producing any success.

This diagnostic does not establish a positive recovery-training effect and
does not authorize Stage C-E by itself. The next revision must preregister one
training-dose reduction while keeping the frozen validation protocol intact.

## Accidental extra diagnostics

Because of a controller scope defect, evaluation seeds `20260816` and
`20260817` also ran. They are preserved, independently hash-audited, and
excluded from this terminal decision. They were not consulted to define or
change the success gate.

Authoritative machine-readable evidence:

- `summary/summary.json`
- `integrity_audit.json`
- `errors/20260728_unplanned_eval_seed_expansion.md`
- `ops/postprocess_v1_status.json`
- `ops/postprocess_v2_status.json`
