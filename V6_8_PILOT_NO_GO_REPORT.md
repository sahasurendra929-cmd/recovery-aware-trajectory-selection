# V6.8 Pilot NO-GO

Date: 2026-07-30 (Asia/Shanghai)

V6.8 is `NO_GO` at the frozen Pilot generation stage. Eleven of 24 tasks and
33 of 72 candidate pairs completed with PASS task artifacts before the
generator failed closed at `retail:19`, candidate pair 1, branch 1. No atomic
merged JSONL or generation receipt was emitted. Training and official-test
unsealing remain unauthorized.

## Provenance and prefix audit

- Scientific implementation commit:
  `4a5815333f75ac00d6bc9561f4d0aa8d3bac64f6`
- Logging-only diagnostic commit:
  `e3bc65cb2a0b8c63625904f4f3d4276501b37634`
- Tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Registry internal SHA-256:
  `d93b22dba94db13f6670bb5ef629bf341b0febdfeb2f4798a430dfd52ec9f2bd`
- Pilot run-contract file SHA-256:
  `2198cb5c63d26316d354cbf425a98ca7135ee660d517896aa5ed8accd0157ecd`
- Eleven-task checksum-list SHA-256:
  `7877ff8f55433939a0560c6b6017765f8ec3e6810746e46f3aaea4fb568d35ce`
- Diagnostic log SHA-256:
  `c5a28e088cb0db5639340561ef1c77f950a45c207a42fee8d1386a96d355b325`

All 11 completed tasks, 33 pairs, clean rewards, replay checks, and label
audits passed. `retail:104` and `retail:16` both passed in the complete Pilot.
Official test remained unused.

## Failure evidence

For the failing matched recovery:

- the registered forced-first action matched;
- all seven reference action checks were 1.0;
- DB check was 1.0;
- communicate checks for `54.04` and `41.64` passed;
- the final assistant message explicitly said that returning the water bottle
  gives a refund of `$54.04` and exchanging the pet bed and office chair saves
  `$41.64` total;
- both frozen NL assertions nevertheless failed, with the judge claiming that
  refund and savings had not been stated.

The V6.8 renderer produced the grammatically awkward forms
`I am telling you directly: that returning ...` and
`I am telling you directly: that exchanging ...`. The generator correctly
failed closed.

Classification:
`LEADING_COMPLEMENTIZER_RENDERER_JUDGE_COMPATIBILITY_FAILURE`.

## Next

Preregister V6.9 with one renderer grammar normalization: after the frozen
`Agent should tell the user` prefix is removed, strip exactly one leading
case-sensitive `that ` complementizer before constructing the explicit direct
statement. Preserve every other protocol input and gate. Run targeted smokes
for `retail:19`, `retail:16`, and `retail:104` before restarting the Pilot in
a new versioned directory.
