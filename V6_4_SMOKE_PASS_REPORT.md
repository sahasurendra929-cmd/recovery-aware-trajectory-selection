# V6.4 Smoke PASS Report

Status: **PASS; complete Pilot generation is authorized**

## Provenance

- source commit: `50fd9f7ae3bd84f1b3f21afdc8c2d8b481c6302c`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- design protocol: `v6_4_forced_correction_reference_tail_v1`
- smoke task: `airline:12`
- official test: sealed and unused

## Completeness and audits

- candidate pairs: 3 / 3
- candidate branches: 6
- forced-first cells: 12
- forced-first repeated-measurement trials: 36
- matched recovery official success: 6 / 6
- forced-first cell success: 12 / 12
- forced-first trial success: 36 / 36
- matched and forced-first independent replay: PASS
- failed/error result positive labels: 0
- clean-future overlap: 0
- task receipt: PASS
- generation receipt: PASS
- exit code: 0

All matched and crossed smoke cells succeeded, so this single task supplies no
causal separation. This is recorded without changing the Pilot gate; causal
variation is evaluated only after the complete fixed 24-task Pilot.

## Artifact fingerprints

- registry file SHA-256:
  `bd97264ac38eeb8247025a1f90e2cf27f2625c749190436e945dd57c8c58ef49`
- run contract SHA-256:
  `0a439c6c2dd1477275f504bffb45cd4647731f099075b187e81e2316bade22c8`
- semantic generation contract SHA-256:
  `acc4216e67f6e7437dc220d0fc7b35f3e2a508778001bb4c967f1615dc7fa88f`
- generation receipt SHA-256:
  `9136f1c4a387ef54e6d71f71aabba74796ce9ce3f9dc649b281a8f7623511a35`
- unscored candidate JSONL SHA-256:
  `f41412f34197a668fc5a8191936b65fe2cd5e07e3fcaaadc6df9740e4d704fb9`
- task receipt SHA-256:
  `0fe505a26cce5183b50d3864bedd2f49779ed64821464c5b2391b70e75f7cac2`
- execution log SHA-256:
  `f319b2dbd96317ee96c90b7e6cbb265b9191a837ec2c8b16d214b7da7bb6411b`

Large task and debug logs remain on the RunPod volume. The complete fixed
Pilot must use a separate output directory and may proceed only under the
identical semantic generation contract.
