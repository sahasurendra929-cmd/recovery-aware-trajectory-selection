# V6.7 Pilot NO-GO

Date: 2026-07-30 (Asia/Shanghai)

## Decision

V6.7 is `NO_GO` at the frozen 24-task Pilot generation stage. Ten tasks and
30 candidate pairs completed and passed their task-level audits before the
generator failed closed at `retail:16`. No atomic merged candidate JSONL or
generation receipt was emitted. Scoring, training, and official-test unsealing
remain unauthorized.

This NO-GO is preserved as the V6.7 result. A successor protocol may continue
only after a separately versioned preregistration; successor results must not
be merged with V6.7.

## Frozen inputs

- Scientific implementation commit:
  `6921a3f82f7c1bfc350ee4200ced72899328ffdd`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Registry internal SHA-256:
  `6f1e93e278fb6e71858aaee7a2387e01ea5d4849c9a9fbdad30998b544609ec4`
- Pilot run-contract file SHA-256:
  `d7dcd72a1b9cafa0400c4839974a5054178d9739f8a8dc53606ce9f222f31eb7`
- Semantic generation-contract SHA-256:
  `63f62705cda8a0815d7a9e5a5dda1e53e4f472ee9a676150dc07e920301368a8`
- Ten-task checksum-list SHA-256:
  `3f95f7f8ba4c68bc946875a2ce6907dc440f9923937bc109ba427cd978eccf92`
- Diagnostic log SHA-256:
  `01e13bf3c8db8a02d0668665cc840a233019d89443df801552a1068820047352`

Two post-failure logging-only commits (`dbaf9bd` and `1f47e12`) exposed the
official reward breakdown. They did not change trajectories, renderer output,
models, tasks, seeds, evaluator, or gates.

## Completed-prefix audit

- Completed task artifacts: 10/24
- Completed candidate pairs: 30/72
- Task artifact status: 10 PASS
- Shared-prefix checks: all PASS
- Shared-environment-snapshot checks: all PASS
- Clean official reward: 1.0 for every completed task
- Pair quality, matched replay, and label checks: all PASS
- Official test used: false
- Atomic merged JSONL emitted: false
- Training started: false

The previously blocking `retail:104` task passed in the full Pilot: all three
clean controls, six matched recoveries, 12 forced-first cells, and 36
continuation trials had task success 1.0; all independent replay checks passed.

## Failure evidence

The first incomplete task was `retail:16`. Two deterministic clean attempts
were reproduced with the frozen inputs. In both attempts:

- all nine frozen reference actions executed successfully;
- all nine action checks were 1.0;
- final DB check was 1.0;
- communicate check for `8276.23` passed;
- the assistant's final natural-language message explicitly contained
  `The total refund amount is $8,276.23.`;
- the frozen 14B NL assertion judge nevertheless returned 0.0 and stated that
  the amount had not been explicitly communicated.

The aggregate official reward was therefore 0.0 in both attempts because the
registered reward basis is `DB` plus `NL_ASSERTION`. The generator correctly
failed closed.

## Classification

`FROZEN_JUDGE_RENDERER_COMPATIBILITY_FAILURE`.

This is not a database, tool-execution, reference-action, task-supply,
infrastructure, or official-test leak failure. The direct V6.7 sentence is
semantically correct, but it is not reliably recognized by the frozen strict
judge in the full Pilot context. Changing the renderer wording is a scientific
protocol change and must not be applied under the V6.7 label.

## Next

Preregister V6.8 with exactly one change: a more explicit first-person
assertion renderer that marks evaluator assertions as direct statements to
the user and removes generic completion boilerplate that the frozen judge may
misattribute to tool output. Keep task IDs, registry, actions, models,
revisions, seeds, budgets, gates, selector weights, training, and evaluation
protocol unchanged. Run `retail:16` and the prior `retail:104` as targeted
smokes before restarting the complete Pilot in a fresh versioned directory.
