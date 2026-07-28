# Prompt for the RunPod/Codex experiment agent

You are the V5.5 experiment executor. Work only in:

`/workspace/recovery-aware-trajectory-selection`

Protocol:

`v5_5_reference_grounded_counterfactual_recovery_v1`

Your job in this release is data construction and independent audit, not
training and not an end-to-end result claim.

1. Clone or update the repository and checkout the exact V5.5 commit supplied
   by the coordinator. Record `git rev-parse HEAD`.
2. Confirm the nested tau2 checkout is exactly
   `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
3. Reuse `/workspace/venvs/tau2-v55` if it exists and imports tau2. Otherwise
   create it once with Python 3.12 and install
   `-e 'data/raw/tau2-bench[dev]'`.
4. Run the two V5.5 test files. Stop on any failure.
5. Run the CPU structural preflight exactly as written in
   `V5_5_AUDIT_FIRST_HANDOFF.md`. Confirm:
   - structurally eligible tasks = 45;
   - executable eligible tasks = 36;
   - executable airline = 12, executable retail = 24;
   - registered tasks = 24;
   - registered pairs = 48;
   - official test used = false.
6. Run `run_v5_5_reference_pairs.py` for all 48 rows. Do not edit the
   manifest, skip rows, change mutations or substitute outputs.
7. Run `audit_v5_5_pairs.py`.
8. If status is not `PASS_TRAINING_AUTHORIZED`, stop fail-closed and upload
   the manifest, pairs produced so far, audit/error log and environment
   versions. Do not train.
9. If status passes, still do not train in this release. Package:
   - `artifacts/v5_5/manifest.json`
   - `artifacts/v5_5/pairs.jsonl`
   - `artifacts/v5_5/audit.json`
   - test log
   - Python/package versions
   - SHA-256 file manifest
10. Commit artifacts to a new results branch and push it. Never force-push and
    never place model weights in Git.

Do not claim these reference actions are natural expert conversations. Do not
use the official test split. Do not change the gate to obtain a pass.
