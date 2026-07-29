# V6.5 reference-completion smoke PASS

Date: 2026-07-30 (Asia/Shanghai)

## Decision

The preregistered V6.5 smoke on `airline:14`, the exact task that exposed the
V6.4 construction defect, is **PASS**. V6.5 is authorized to run the complete
fixed 24-task Pilot. This smoke does not itself authorize training.

## Identity

- Protocol: `v6_5_forced_correction_reference_completion_v1`
- Source commit:
  `941159e5708d0f66b21d8d07cc856031434e9955`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- User/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- Registry internal hash:
  `cff633ffe45823d3a6b48ec37c2a2b4ae26c912ede4cc6d63f07b4af7d490838`
- Registry file SHA-256:
  `a5c7dbdbe8ffbd3741cacc448e7e6bcaca0eb36b07c44b30f8fe3c255a3260cc`

## Audit

- Tasks: `1 / 1`
- Candidate pairs: `3 / 3`
- Branches: `6`
- Forced-first cells: `12`
- Repeated forced-first trials: `36`
- Matched recovery rewards: all `1.0`
- Matched and crossed forced-first trial rewards: all `1.0`
- Matched independent replays: all PASS
- Forced-cell independent replays: all PASS
- Failed/error-result positive labels: zero
- Future-message overlap: zero
- Official test used: false
- Training started: false

The V6.4 failure mode is directly repaired: when `book_reservation` is forced
first, the previously omitted `cancel_reservation` action is still executed
exactly once as part of reference completion.

## Integrity evidence

| Artifact | SHA-256 |
|---|---|
| Run contract | `02da61b10279dc7674f41e66cc2ecdab91d424a63133ca5f7806239078ea4500` |
| Generation receipt | `b86e1dfe027cfc2709dc70441a16535682e4670872823bf224953c9e413acccd` |
| Candidate JSONL | `b2ee7535c49729d18673315c476f4bfdf42bb1da6e9d7523d2d1f98d4ce0f6e7` |
| Task receipt | `df5b11851182d05ee1a1e043892ca01671405bdf452fc39e982ad8afc5abd63b` |
| Console log | `cb82bdc638ed92914d1f0c9a3a61a7f191761de056ffcc2c0ac2fda28b297656` |

Large runtime artifacts remain on the RunPod volume. Only their hashes and
audited summaries are committed to GitHub.

## Boundary

All smoke cells succeeding also means this task supplies no matched-versus-
crossed causal separation. The complete Pilot gate is unchanged and may still
return NO-GO. V6.5 remains a constrained/reference-completed recovery
construction, not unconstrained teacher recovery.

