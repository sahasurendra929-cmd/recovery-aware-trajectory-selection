# V6.9 Pilot generation NO-GO report

Date: 2026-07-30 (Asia/Shanghai)

## Decision

The first complete V6.9 Pilot attempt is `NO_GO`. The generator completed 14
of 24 frozen tasks and 42 of 72 candidate pairs with per-task `PASS`
artifacts, then failed closed while materializing `retail:31`.

This NO-GO is preserved before any repair. It does not authorize measurement,
scoring, selection, training, or official-test unsealing. It also does not end
the broader V6 execution: the failure will be diagnosed, regression-tested,
and either repaired under the unchanged protocol or superseded by an
explicitly preregistered successor if a scientific contract must change.

## Failure

The exception was:

```text
V6GenerationError:
forced recovery action must match exactly one frozen reference action
```

It arose in
`deterministic_reference_completion_simulation()` while constructing a
matched recovery for `retail:31`. The process stopped atomically: no merged
candidate JSONL and no complete generation receipt were emitted.

Initial classification is an implementation/data-contract ambiguity in
mapping a frozen forced call back to the task's reference action sequence.
The frozen gate correctly rejected a match cardinality other than one. No
threshold, task ID, model, revision, seed, selector weight, training budget,
or metric has been changed.

## Preserved progress

All six frozen airline tasks passed. The following eight retail tasks also
passed:

`retail:1`, `retail:104`, `retail:107`, `retail:11`, `retail:16`,
`retail:19`, `retail:21`, and `retail:30`.

In particular, the complete Pilot independently passed both prior blocker
tasks:

- V6.7 blocker `retail:16`;
- V6.8 blocker `retail:19`.

Thus, 14 task artifacts / 42 candidate pairs are preserved. They will not be
presented as a complete V6.9 Pilot and will not be silently merged with a
repaired attempt.

## Provenance and fingerprints

- Pilot launch source commit:
  `717981b50db34a09f701ffd35d1ddccc8afa51bb`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- registry internal SHA-256:
  `e234874ec1542470f4458f814ca1a1c6481199f9b2da5de0f58314b07e874a99`
- run-contract file SHA-256:
  `73b45678880fd12cf8254416bcdfb931e79f82cf639e36d41dcc9d9d20ec827d`
- ordered 14-task checksum-list SHA-256:
  `a831257d531454a7e2706bec5ecc050745b29c203ab99262e564163e2fb219e0`
- diagnostic log SHA-256:
  `4566e05aa630e028ae4fd2ef539c327f47b1dc67e3d760718fbf4d35588a3bc9`
- official test used: false
- training started: false

Raw task artifacts, the ordered checksum list, and the diagnostic log remain
on the retained RunPod volume.

## Required recovery

Determine whether `retail:31` contains repeated identical canonical reference
calls or whether the matcher discards the already known registered reference
slot. If the forced action is already bound to a unique frozen reference
index and the runtime unnecessarily re-identifies it by value, repair the
plumbing to preserve and use that slot, add a regression fixture for repeated
identical calls, rerun code-only tests, and start a fresh versioned attempt
under the unchanged V6.9 scientific protocol.

If resolving the ambiguity requires changing candidate identity, task
contents, branch definition, continuation seeds, or another scientific
contract, first preregister a new V6.x protocol and retain this NO-GO as the
V6.9 result.
