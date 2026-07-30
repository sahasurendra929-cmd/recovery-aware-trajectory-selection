# V6.9 Pilot Attempt 2 NO-GO Report

Date: 2026-07-30
Branch: `codex/v6.9-complementizer-normalization`
Source commit: `aa1a81d1c21328bb9dc550ffeea7fbba0c1b07ea`
Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`

## Result

The second fresh V6.9 Pilot generation attempt failed closed at `retail:35`.
Fifteen of the 24 frozen Pilot tasks completed and emitted 45 of the expected
72 candidate pairs. In particular, `retail:31` passed, confirming that the
registered-reference-slot repair worked in the complete Pilot.

No atomic merged candidate file or final generation receipt was emitted.
Official test remains sealed, and training remains unauthorized.

This NO-GO is an auditable intermediate engineering result. It does not end
the complete V6 program: the failure will be regression-tested and repaired
without changing tasks, thresholds, models, seeds, selector weights, training
budget, or statistical protocol, followed by a fresh versioned attempt.

## Failure

The deterministic clean reference replay executed frozen action `35_0`,
`find_user_id_by_email` for a deliberately invalid email address. Tau2
returned `Error: User not found`, and the generator raised:

```text
V6GenerationError: reference action 35_0 returned a tool error
```

Task `retail:35` has no initialization override. Its frozen reference action
sequence intentionally contains an unsuccessful lookup followed by a corrected
lookup and the successful task actions. Therefore, the first tool error is part
of the benchmark reference trajectory rather than an infrastructure failure or
an invented candidate error.

The current deterministic replay incorrectly assumes that every reference
action must individually return a non-error result. The repair must retain and
replay expected reference tool errors, while continuing to fail closed for
unexpected execution exceptions, state drift, missing actions, evaluator
failure, or unsuccessful final task reward.

## Retained evidence

- Completed task artifacts: 15
- Candidate pairs retained: 45
- Expected final Pilot tasks / pairs: 24 / 72
- Run-contract SHA-256:
  `73b45678880fd12cf8254416bcdfb931e79f82cf639e36d41dcc9d9d20ec827d`
- Ordered completed-task checksum-list SHA-256:
  `2e9463e3e0d2e25a5906f1ccd9616cd843bb3bca2c913ecb79bfe61bb001b25d`
- Diagnostic log SHA-256:
  `7095685f81ded1c959ec04125b671146a82571ff271b38428d896fede57b3254`
- Persistent attempt directory:
  `/workspace/v6_9_20260730/artifacts/v6_9_directional_screen/generation/pilot_attempt2`
- Official test used: no

Raw benchmark trajectories and logs remain on RunPod and are not committed to
GitHub.

## Recovery authorization

This is classified as
`EXPECTED_REFERENCE_TOOL_ERROR_REPLAY_INVARIANT_DEFECT`.
It is an implementation defect, not a scientific-design change. Recovery is
authorized only after:

1. adding a fixture that contains an expected failed reference lookup followed
   by a successful correction;
2. proving the failed tool result is retained in the reconstructed trajectory;
3. proving final official reward and deterministic replay checks still gate
   success;
4. passing the complete code-only suite;
5. running a fresh `retail:35` smoke in a new output directory; and
6. starting the next complete Pilot in another fresh attempt directory.
