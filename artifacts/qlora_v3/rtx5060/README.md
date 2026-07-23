# QLoRA V3 RTX 5060 audit package

This package contains the minimal allowlisted artifacts needed to audit the constrained-recovery V3 selection, formal QLoRA training, complete 959-example evaluation, and aggregate report.

## Claim boundary

V3 is an exploratory diagnostic of offline next-tool-call imitation on the already inspected V2 test set. It is not an end-to-end Agent-success evaluation, not executable tool success, and not confirmatory or paper-final evidence.

- Frozen source commit: `7e0419d9b0941902ae149a68498ce9a19b1ea2f1`

## Reference status

- V2 reference status: `compatible`
- Compatible paired V2 reference: `yes`
- Direction label: `non_recovery_preserved_without_recovery_gain`
- Aggregator note: Compatible V2 random_success predictions were rescored and paired by example ID.

## Deliberate exclusions

No raw trajectories, processed train/validation/test JSONL, virtual environment, model/tokenizer cache, or smoke-test checkpoint is included. `UPLOAD_MANIFEST.json` hashes every packaged file except itself.
