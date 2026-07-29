# V6.9 registered-reference-slot repair report

Date: 2026-07-30 (Asia/Shanghai)

## Decision

The `retail:31` implementation repair passes its targeted regression barrier.
A fresh complete V6.9 Pilot attempt is authorized under the unchanged
scientific protocol.

The failed first Pilot and both regression attempts remain preserved. The new
complete Pilot will use a new output directory and will rerun all 24 tasks;
it will not mix task artifacts produced by different source commits.

## Root cause

Tau2 `retail:31` legitimately repeats three read-only
`get_order_details` calls in its frozen reference action sequence. The V6
outcome-independent registry already records each selected action's unique
`reference_action_index`, but deterministic recovery discarded that identity
and attempted to rediscover the slot from tool name and arguments.

That value-based lookup was ambiguous. The generator correctly failed closed
when it found more than one identical call.

## Repair

Runtime recovery now:

1. accepts the registry's frozen `reference_action_index`;
2. validates that the indexed action has exactly the same tool-call semantics
   as the forced call;
3. removes only that indexed occurrence from deterministic completion;
4. executes every other frozen reference action once, including any identical
   read-only occurrence at another index;
5. propagates the same index through matched recovery and all four
   matched/crossed forced-first cells.

No task, threshold, model, revision, seed, decoding value, candidate identity,
branch definition, renderer, selector weight, training budget, metric, or
statistical rule changed.

## Validation

- local V6 suite: 157 tests plus 6 subtests passed;
- remote Linux V6 suite: 157 tests plus 6 subtests passed;
- targeted task: `retail:31`;
- accepted pairs: 3;
- sibling branches: 6;
- forced-first cells: 12;
- continuation trials: 36;
- all cell and trial task-success values: 1.0;
- all matched and independent replays: pass;
- failed positive labels: 0;
- official test used: false.

## Provenance and fingerprints

- complete repair source commit:
  `cc81f81af02ef4b571cb61c5787e666b159f0474`
- registry internal SHA-256:
  `e234874ec1542470f4458f814ca1a1c6481199f9b2da5de0f58314b07e874a99`
- semantic generation contract SHA-256:
  `1d6003547e020c3c869616e3889e646df704fb0f023fee660ff54172151cbf65`
- targeted run-contract SHA-256:
  `5a557a8307c1457b1ff25b2eda8048e256c2e1aa3e58d6486d3b09d3f7331474`
- targeted candidate JSONL SHA-256:
  `1ae0511d6f88dd301717f51bce631b486079ad5fc1ce2a6e547dce149ee57f5a`
- targeted generation-receipt SHA-256:
  `835277757ca1816ed70166313ab12a7f38fdf3e183de05cbd73709cddc0ef1d5`

The first targeted attempt stopped before emitting candidate output because
the crossed-cell path was initially missed. Its separate failure record is
`artifacts/v6_directional_screen/ops/errors/20260730_v69_smoke31_crossed_cell.md`.

## Next authorization

Start V6.9 Pilot attempt 2 in a fresh directory, rerunning all 24 frozen
tasks with the same registry and semantic generation contract. Measurement,
scoring, selection, training, and official-test unsealing remain unauthorized
until the complete Pilot and its downstream preregistered gates pass.
