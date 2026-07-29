# V6.3 Smoke NO-GO Report

Status: **NO-GO; V6.3 Pilot candidate generation is not authorized**

## Frozen provenance

- source commit: `799c05aae9f6a234bf0fbd55cad278d76c791c79`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- design protocol: `v6_3_deterministic_reference_clean_replay_v1`
- smoke task: `airline:12`
- candidate branch:
  `v6:pilot:airline:12:prefix:initial:candidate-pair:01:branch:1`
- recovery attempts: 2, with the unchanged registered seeds
- official test: sealed and unused

## Result

The deterministic clean reference replay passed and candidate construction
advanced past clean-source validation, error injection, and the registered
forced first corrective action. Both fresh 72B recovery continuations then
received official reward `0.0`. No candidate pair or task receipt was emitted.
The generator failed closed with:

`no fresh successful recovery whose first action matched the registered corrective call`

The registered corrective action was present. The failure occurred later:
both continuations changed dates, flights, and cabin state outside the frozen
reference resolution, producing a final database state inconsistent with the
task. Each attempt consumed nearly the full 60-step budget. Increasing seeds
or attempt count after observing this result is not permitted within V6.3.

This is a recovery-supply NO-GO, not evidence about selector performance.
V6.3 must not proceed to the 24-task Pilot and must not be pooled with any
other V6 version.

## Artifact fingerprints

- registry file SHA-256:
  `abc50cb5835f80e505f778babb2603be68dbabe6d5e7d2d682a898deedc511c6`
- registry internal SHA-256:
  `0df6aecdccdaac74330739286f20e4fa79e0c3b0223c20275d4d27073bb69483`
- run contract SHA-256:
  `dd0a1b107b03667c82f376aa4d31b9b15496852c9a4e1258cba2920cbb6f917a`
- semantic generation contract SHA-256:
  `d71fe9233564ce21ba8ab17abb47465a71832da64044e9b222a47de944326774`
- execution log SHA-256:
  `dec8f2e02cc6d17be9ba51a96462abef0be89671d9c7fd77945e9b645e50eb47`
- recovery attempt 1 task log SHA-256:
  `3a6293ba99886f3e453b1959ba4e5515a882954e98322d28232af5dc19f1f3c9`
- recovery attempt 2 task log SHA-256:
  `74f01342161c3b05a21a9795f3079629ea1840cdef7fe5498b952ddbae67ba31`

Raw logs remain on the RunPod volume; their hashes bind this report to the
exact bytes without uploading large debug traces.

## Successor boundary

A successor may constrain the continuation after the registered corrective
action by deterministically executing the remaining frozen reference
resolution. This changes the scientific object from unconstrained fresh
teacher recovery to constrained/reference-completed recovery and therefore
requires a separate V6.4 preregistration and conclusion boundary. Tasks,
errors, gates, selectors, training, evaluation, and statistics must otherwise
remain unchanged.
