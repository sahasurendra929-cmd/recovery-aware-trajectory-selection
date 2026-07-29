# V6.2 Smoke NO-GO Report

Status: **NO-GO; V6.2 candidate generation is not authorized**

## Frozen provenance

- source commit: `73339266c9e8ae70ca242b443e1743056e43a1c6`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- design protocol: `v6_2_reference_guided_clean_source_v1`
- smoke task: `airline:12`
- clean agent mode: `reference_guided` (`tau2` `llm_agent_gt`)
- clean attempts: 2, using the unchanged registered attempt seeds
- official test: sealed and unused

## Result

Both clean attempts received official reward `0.0`. No candidate pair or task
receipt was emitted. The generator correctly failed closed with:

`V6GenerationError: airline:12: no successful clean rollout with an observed tool call`

Inspection of the trace shows that the reference-guided LLM treated the
reference actions as advice rather than a hard execution constraint. It
performed an additional reservation upgrade for both passengers and charged
USD 1,200, despite the task requiring the policy-compliant refusal of a
one-passenger cabin change and only the addition of two free bags. The second
attempt repeated the same substantive violation.

This is a candidate-supply failure, not a positive or negative selector result.
V6.2 must not proceed to its Pilot pool and must not be pooled with V6, V6.1,
or any successor.

## Artifact fingerprints

- registry file SHA-256:
  `5bf9a885b969199ca34ab0d2c3d3a0064b8b872913ef496a642a179a63303c3d`
- registry internal SHA-256:
  `ed868dd460cd6b66f5874fdb7829c24112e8ff4cca81bd4f3a2ee9b08f1323c1`
- run contract SHA-256:
  `e062fcb859175b6288ac28fa222c92955346b308fb3b7cc158f45b14d5727852`
- semantic generation contract SHA-256:
  `17ef426ea9a86390f61fa14ab97142e33596e3edc39f55b3706a338c71399136`
- redacted execution log SHA-256:
  `291192501d3317df7c396a7e7a41e632d8418851e38441a4d56fca86e24bc532`

The large raw execution log remains on the RunPod volume. It is not committed
to GitHub; the hash above binds the report to those bytes.

## Successor decision

The failure demonstrates that merely exposing reference actions to an LLM
does not guarantee a successful clean source. A successor may replace only
this prerequisite with deterministic execution and independent replay of the
already-frozen reference actions. Such a change alters candidate construction
and therefore requires a separately preregistered V6.3 protocol. Recovery
suffix generation, task IDs, gates, selectors, training, evaluation, and
statistics must remain unchanged.
