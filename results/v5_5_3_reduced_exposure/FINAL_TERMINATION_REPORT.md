# V5.5 Reference-Screen Final Termination Report

## Terminal decision

**NO-GO / FAIL-CLOSED.**

The original V5.5 reference screen and all three permitted diagnostic/revision
cycles have reached auditable terminal states without satisfying the
pre-registered positive-screen criterion. Stage C, Stage D, and Stage E were
therefore not started. This is an allowed non-success terminal state under the
V5.5 execution plan; further GPU tuning is prohibited without a new,
independently approved protocol.

The final V5.5.3 screen itself completed successfully as an experiment:

- 9/9 arm × evaluation-seed batches complete;
- 27/27 shards complete;
- 378/378 task-level rows present;
- no missing, extra, or duplicate rows;
- every shard has a complete run contract and passing metrics;
- all 27 summary row hashes independently verified;
- official test remained sealed (`official_test_used=false`);
- terminal integrity audit: `PASS`;
- scientific terminal status: `NO_GO_REFERENCE_SCREEN`.

## Final V5.5.3 result

| Arm | Clean success | In-family controlled-error success | Delta error vs R0 | 95% task-bootstrap CI |
|---|---:|---:|---:|---:|
| R0 | 0.254 | 0.188 | +0.000 | [+0.000, +0.000] |
| R50 | 0.286 | 0.104 | -0.083 | [-0.229, +0.000] |
| R100 | 0.206 | 0.062 | -0.125 | [-0.271, -0.021] |

The frozen selector identifies R50 after the clean non-inferiority filter, but
`positive_screen=false`: recovery training did not improve controlled-error
success over R0. R100 was materially worse on the primary recovery comparison.
These results are diagnostic only and do not authorize confirmatory claims.

## Revision history and diagnosed causes

1. **Original V5.5** — complete, audited, and reported
   `NO_GO_REFERENCE_SCREEN`. All adapted arms collapsed to zero on the screen.
2. **V5.5.1 base-floor diagnostic** — established that the frozen evaluation
   stack and unadapted base model had non-zero performance. This excluded a
   universal evaluation-floor explanation and localized the primary failure to
   adaptation-induced degradation.
3. **V5.5.2 low-learning-rate revision** — changed only the learning rate from
   `1e-4` to `1.25e-5`, preserving tasks, seeds, training exposure, evaluation,
   statistics, and official-test seal. It removed the total collapse but did
   not produce a positive recovery delta.
4. **V5.5.3 reduced-exposure revision** — retained the V5.5.2 learning rate and
   changed only training steps from 64 to 32 (gradient accumulation 8),
   reducing exposure from 512 to 256 examples per arm. Clean performance was
   retained for R50, but controlled-error recovery remained below R0.

The evidence rules out the initial catastrophic learning-rate/exposure regime
as the sole explanation. The residual blocker is scientific: under the frozen
reference-grounded data construction and validation screen, increasing
recovery-example dose does not transfer into improved controlled-error task
success. The monotone degradation from R0 to R50 to R100 on the primary
controlled-error measure is inconsistent with proceeding to natural-data
construction or a larger dose-response matrix under this protocol.

## Compute and evidence retained

Across the original screen and revisions, the project retained:

- three complete 3-arm reference screens (27 batches, 81 shards, 1,134
  task-level rows);
- one 3-evaluation-seed base-floor diagnostic (9 shards, 126 task-level rows);
- nine trained adaptation checkpoints used for evaluation, while excluding
  checkpoint weights and optimizer state from GitHub;
- preflight reports, frozen manifests, commands, environment/source commits,
  task-level rows, shard metrics, run contracts, summaries, audits, logs, and
  cryptographic hashes needed to recompute and verify reported results.

The RunPod wall-clock execution records and command manifests are preserved in
the corresponding result branches. No official-test data was used.

## Stage A–E terminal accounting

- **Stage A:** previously audited prerequisites reused; no unnecessary
  regeneration was performed.
- **Stage B:** complete for original V5.5, V5.5.2, and V5.5.3; V5.5.1 supplied
  the scoped base-floor diagnostic. Every terminal result is preserved.
- **Stage C:** not executed because Stage B never met the scientific
  continuation gate.
- **Stage D:** not executable because Stage C was not authorized.
- **Stage E:** not executable because neither Stage B nor the downstream gates
  were satisfied.

This is fail-closed protocol compliance, not an implementation failure.

## Recommended next step

Do not run another V5.5.x seed, dose, or hyperparameter search. A future effort
should begin as a new protocol and test a mechanism-level hypothesis before
renting a full GPU matrix. The highest-value options are:

1. audit whether recovery examples teach post-error state reconstruction rather
   than merely imitate the reference continuation;
2. measure token-/turn-level credit assignment and tool-state mismatch around
   the injected error;
3. redesign natural recovery collection around verified causal recovery
   transitions, with an independent data-quality gate;
4. preregister a small mechanism probe and a fresh held-out validation screen
   before any Stage C–E-scale execution.

Any such work must use a new version, preserve the present NO-GO evidence, and
must not use the official test for protocol selection.
