# V6.9 triple targeted-smoke PASS report

Date: 2026-07-30 (Asia/Shanghai)

## Decision

V6.9 passes its preregistered targeted-smoke barrier and is authorized to
start a fresh 24-task Pilot. This is an engineering/runtime authorization, not
an experimental result and not an authorization to unseal official test.

The three targeted tasks were:

- `retail:19`, the V6.8 Pilot failure;
- `retail:16`, the V6.7 Pilot failure;
- `retail:104`, the earlier completion-renderer failure.

All three generation receipts are `PASS`. The combined audit covers 9
candidate pairs, 18 sibling branches, 36 forced-first cells, and 108
continuation trials. Every cell and every trial has official task success
1.0. Every matched recovery and independent replay passed. Every pair
executed a real tool error, had no future leakage, and recorded zero failed
positive labels. Official test was not used.

## Frozen provenance

- source commit at launch:
  `ca7747e015313aa99119ddd36b57f037fedf1098`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- registry internal SHA-256:
  `e234874ec1542470f4458f814ca1a1c6481199f9b2da5de0f58314b07e874a99`
- registry file SHA-256:
  `fd9bf1d4b39d7429837fbb3b042c029944ca5cb45a5f9290bf0b4ff161180ee9`
- semantic generation contract SHA-256:
  `1d6003547e020c3c869616e3889e646df704fb0f023fee660ff54172151cbf65`
- code-only validation on both local Windows and remote Linux:
  155 tests plus 6 subtests passed.

The registry contains 24 Pilot tasks / 72 Pilot pairs and 50 formal tasks /
150 formal pairs. The registry builder reported 222 pairs across both phases,
with official test unused.

## Artifact fingerprints

Candidate JSONL SHA-256:

- `retail:19`:
  `50f15d5533786c96cb0bf3b733dc6a4550764f658dc622d73568a16e62d91766`
- `retail:16`:
  `bdccde205298631b8efdfc9678f3921fb62fe6579bd730f3f365215bc8cff0b7`
- `retail:104`:
  `0a7a0dac499d5a6ff7421a3c78b58303bb23068b82b189f3ed62ca51e4d6a76e`

Generation-receipt SHA-256:

- `retail:19`:
  `2c4129272aac18304e28a88fdda815a44e25acc3905eb90d96023f64faef43bc`
- `retail:16`:
  `7341da80423515a2c8d6cb459b5cc9d8be7ac9777e61e3fe8cd490ebc10be084`
- `retail:104`:
  `7882ebc4866b9295a2fa09472dab377ee4d480b0aadf213f053971f494b4d2e6`

Raw trajectories and service logs remain on the retained RunPod volume. They
are not committed because they are large and may contain benchmark task
content. The hashes above, frozen contracts, source, tests, and audit summary
are the public verification layer.

## Recovery history

The first persistent-launch attempt produced no runtime outcome: its remote
shell variables were expanded by the Windows control shell before transfer.
The resulting log was empty and no task output directory was created. The
operator switched to explicit per-task persistent commands. This launch
failure did not alter the scientific protocol or consume a result.

## Next authorization

Start a fresh V6.9 Pilot output directory using all 24 frozen Pilot tasks, the
same registry, models, revisions, continuation seeds, decoding, clean mode,
recovery mode, and `explicit_user_direct_v3` renderer. Do not merge V6.7,
V6.8, or targeted-smoke outcomes into the V6.9 Pilot. Official test remains
sealed.
