# V5.3 12-hour exploratory screen handoff

## 1. Scope and claim boundary

`V5.3-12h-screen` is a separate, one-seed exploratory experiment intended to
return an audited directional result within a twelve-hour wall-clock budget.
It is **not** the frozen V5.3 feasibility pilot or formal V5.3 experiment.

The following boundaries are immutable:

- frozen V5.3 configs, seeds, rollout bytes, checkpoints, metrics, decisions,
  and result directories are not modified or reused;
- outcome-free V5.3 protocol manifests and deterministic pilot task selection
  are consumed read-only and hash-bound as shared inputs;
- no screen trajectory, checkpoint, metric, or decision may satisfy a formal
  V5.3 gate;
- the official test split remains sealed;
- a positive result is not guaranteed;
- predicted, partial, missing, or imputed values are never reported as
  measured results.

The strongest permitted statement is:

> In a one-seed exploratory derived-validation screen under the registered
> task, model, data, and evaluator scope, Repair-50 produced the reported
> paired difference from Perfect-success.

The screen cannot establish paper-level generalization or replace a later
preregistered multi-seed confirmation.

## 2. Immutable source and persistent workspace

Publish one reviewed implementation commit `C`. A later release manifest
records `C`; the implementation commit does not try to contain its own Git
SHA. At launch, inject the recorded `C`:

```bash
export V5_3_12H_SCREEN_COMMIT='<EXECUTION_COMMIT_C_FROM_RELEASE_MANIFEST>'
export REPO=/workspace/repos/recovery-aware-trajectory-selection
export TAU2="$REPO/data/raw/tau2-bench"
export SERVE_VENV=/workspace/venvs/v5-2-serve
export TRAIN_VENV=/workspace/venvs/v5-2-train
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export SCREEN_RUNTIME="$REPO/artifacts/v5_3_12h_screen"
```

Before any costly work, validate:

```bash
[[ "$V5_3_12H_SCREEN_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -d /workspace
test -x "$SERVE_VENV/bin/python"
test -x "$TRAIN_VENV/bin/python"
test -d "$HF_HOME"
```

Use `/workspace` for repositories, environments, Hugging Face caches, runtime
logs, and retained adapters so a Pod migration or restart can reuse them.

Do **not** run `pip install`, upgrade, downgrade, recreate either virtual
environment, reinstall CUDA/PyTorch/vLLM, or delete a working model cache.
Preflight must compare the existing environment against the frozen package
contract. If it is absent or incompatible, stop with
`ENVIRONMENT_MISMATCH_NO_RUN`; publish the preflight receipt before requesting
a separately reviewed environment repair.

`prefetch` may download only missing files for the exact pinned model
revisions into the persistent Hugging Face cache. It must not reinstall the
Python environment.

The isolated screen requires at least **80 GiB free on `/workspace`** when
preflight runs. This is intentionally different from, and does not modify,
the formal V5.3 controller's 140-GiB default. The screen floor consists of
44 GiB to complete all three pinned model repositories if their cache is
empty (their published files total 41.52 GiB, rounded up), 12 GiB for the
bounded 288 generation rollouts, up to 210 evaluation rollouts, four LoRA
adapters, processed data, logs, and transient runtime files, plus a 24-GiB
post-run safety reserve. If the exact model cache is already complete, its
44-GiB allowance remains additional headroom; do not delete or redownload it.

Keep the execution checkout detached and clean at the published commit.
The tau2 checkout remains pinned at:

```text
fc0055dc4e0a316c3f83133267fbd6faaa770992
```

## 3. Registered screen

The protocol is defined by:

```text
configs/v5_3_12h_screen.yaml
scripts/v5_3_12h_protocol.py
scripts/prepare_v5_3_12h_screen.py
scripts/run_v5_3_12h_supervisor.py
tests/test_v5_3_12h_protocol.py
```

The dedicated single-host controller is:

```text
scripts/run_v5_3_12h_screen.py
```

Do not substitute `scripts/run_v5_3_single_host.py`; that controller belongs
to the frozen V5.3 protocol.

The controller exposes auditable scientific stages plus one entry point:

```text
preflight, self-test, prefetch, protocol, generate, prepare, train, registry,
evaluate, summarize, status, stop-services, all
```

The normal run is launched through the detached supervisor. The supervisor
owns authoritative T0, a boot-bound process-group ledger, singleton locking,
and crash-safe TERM→KILL cleanup; the controller owns scientific barriers:

```bash
"$TRAIN_VENV/bin/python" scripts/run_v5_3_12h_supervisor.py start \
  --runtime-root "$SCREEN_RUNTIME" \
  --terminal-receipt \
  "$REPO/results/v5_3_12h_screen/TERMINAL_STATUS.json" \
  -- \
  "$TRAIN_VENV/bin/python" scripts/run_v5_3_12h_screen.py \
    --stage all \
    --expected-source-commit "$V5_3_12H_SCREEN_COMMIT"
```

No manual vLLM service, generation client, trainer, or evaluator may be
started in parallel with the controller.

`--stage all` does not push GitHub evidence and cannot call the RunPod
management plane. Those are explicit outer-Codex-operator duties. The outer
operator must verify push access before launch, publish bounded receipts, then
Stop Pod and verify `Stopped`. If the controller dies or SSH is lost, run:

```bash
"$TRAIN_VENV/bin/python" scripts/run_v5_3_12h_screen.py --stage stop-services
```

This emergency cleanup requires neither the source-commit argument nor a live
deadline. A local terminal receipt is not proof that RunPod billing stopped.

## 4. Twelve-hour clock

The controller creates a hash-bound, UTC `deadline_receipt.json` before
preflight. Its timestamp is `T0`.

- extension admission deadline: `T0 + 9 hours`;
- hard screen deadline: `T0 + 12 hours`;
- time limits are independent of metric values;
- changing `T0`, pausing its clock, or restarting with a fresh clock is
  prohibited.

At the hard deadline, stop launching work and terminate any incomplete screen
stage cleanly. If the complete core-evaluation receipt and core summary do not
exist, the terminal status is exactly:

```text
INCOMPLETE_NO_CLAIM
```

Do not aggregate partial core rollouts. If the core result is complete but the
optional extension is incomplete, the core remains reportable and the
extension alone is marked `INCOMPLETE_NO_CLAIM` and excluded from comparison.

## 5. Generation: exactly 288 trajectories

Use the same 24 outcome-free task IDs registered in the V5.3 pilot selection:
18 retail and 6 airline tasks. The screen uses an independent base seed and
six independent trial seeds:

```text
base seed: 20260731
trial seeds: 25987, 293840, 725284, 249400, 591103, 257709
```

For every task, generate:

```text
6 clean-condition attempts
6 controlled-error attempts
```

The complete generation population is therefore:

```text
24 tasks × 2 conditions × 6 attempts = 288 trajectories
```

No extra attempt, replacement, retry-as-new-trial, favorable-seed
substitution, rescue, repaired failure, or relabelled failure is permitted.
The only judge retry is the registered malformed-content retry, which does not
create another trajectory.

Generation topology:

| GPU | Role | Frozen model |
|---:|---|---|
| 0 | user simulator and strict judge | `Qwen/Qwen2.5-14B-Instruct-AWQ` |
| 1-2 | trajectory teacher, tensor parallel 2 | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| 3 | reserve and orchestration | no alternative teacher |

Frozen revisions:

```text
14B: 539535859b135b0244c91f3e59816150c8056698
32B: 5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c
```

All 288 registered attempts, all generation contracts, and all strict-judge
evidence must be complete before the audit may decide the data gate.

## 6. Fail-closed data gate

For each task, construct eligible clean/error pairs and cap its contribution
at two pairs. Training is authorized only when both conditions hold:

```text
at least 14 distinct tasks have one eligible pair
AND
at least 17 eligible pairs remain after the two-pairs-per-task cap
```

The threshold is not rescaled after observing yield. A malformed or incomplete
set of 288 attempts fails before evaluating these counts.

If either gate fails:

1. write the complete eligibility/exclusion audit and hash inventory;
2. set `training_authorized=false`;
3. do not train any arm;
4. do not run model evaluation;
5. upload the audited negative feasibility result;
6. stop the Pod after confirming the upload.

This is a valid exploratory feasibility outcome, not an infrastructure excuse
and not evidence comparing the four arms.

## 7. Four-arm training

Only a passing data gate permits training. Train these four QLoRA arms
concurrently, one per RTX 5090:

| GPU | Arm | Meaning |
|---:|---|---|
| 0 | `perfect_success` | clean, flawless successful supervision |
| 1 | `failure_raw` | raw failure-exposed supervision |
| 2 | `repair_50` | exactly 256 perfect rows + 256 repair rows |
| 3 | `repair_100` | repair-only supervision |

All arms use:

```text
Qwen/Qwen2.5-7B-Instruct
revision a09a35458c702b33eeacc393d103063234e8bc28
seed 20260731
exactly 64 optimizer steps
one training seed
no early stopping or validation-based model selection
```

Task support is matched exactly across arms. Because batch size is one and
each microbatch uses mean-reduced causal loss, every training row contributes
one equally weighted microbatch loss. The causal arm dose is therefore frozen
by **row share**: `0/512`, `512/512`, `256/512`, and `512/512` recovery rows
for Perfect-success, Failure-raw, Repair-50, and Repair-100 respectively.

The constructor still minimizes cross-arm supervised-token and nonpadding
token differences, with 1% and 2% as descriptive targets. These token totals
are reported as secondary content/compute controls and do not override the
exact row-share contract or block training if the deterministic row-exact
match cannot also reach those targets. In particular, the experiment does
not claim that Repair-50 has exactly 50% recovery-token mass.

The failed tool call remains visible as history where registered but is
masked from the repair label. Do not change sequence length, LoRA settings,
quantization, optimizer, batch order, or step count in response to memory or
timing pressure.

Every arm must produce finite training metrics, a complete manifest, and a
hash-bound adapter receipt. A missing arm makes the four-arm stage incomplete.

## 8. Core end-to-end evaluation: exactly 126 rollouts

The claim-bearing screen compares:

```text
base_model
perfect_success
repair_50
```

Use all 21 registered derived-validation tasks, one trial per condition:

```text
3 models × 21 tasks × 2 conditions × 1 trial = 126 core rollouts
```

The official test split must remain unopened.

Evaluation uses task-shard parallelism, not one model arm per GPU:

| GPU | Fixed role |
|---:|---|
| 0 | 14B user simulator and strict judge for every arm |
| 1 | 7B agent endpoint for validation shard 0 |
| 2 | 7B agent endpoint for validation shard 1 |
| 3 | 7B agent endpoint for validation shard 2 |

Each GPU 1-3 endpoint loads the same unadapted 7B base plus all four
hash-registered LoRA aliases. Within a shard, arms run sequentially. Every
request names exactly one registry model alias; vLLM permits at most one
active LoRA per worker, so base requests use no adapter and adapted requests
select only the named adapter. Adapters are not merged, and no state or weight
mutation carries from one arm to the next.

For the claim-bearing core, the controller launches exactly
`base_model`, `perfect_success`, and `repair_50` on all three shards. Although
Failure-raw and Repair-100 aliases are registered on the endpoints, no request
uses them until the hash-valid core receipt passes the metric-independent
hour-nine extension gate.

The fixed evaluator is
`Qwen/Qwen2.5-14B-Instruct-AWQ` at revision
`539535859b135b0244c91f3e59816150c8056698`. It is identical for every arm
and is never replaced by a 7B self-evaluator. Across arm comparisons, the
request-bound model alias changes while the task shard, decoding contract,
user simulator, and strict judge remain fixed.

The primary metric is tau2 composite end-to-end task success. The directional
screen is positive only when:

```text
Repair-50 error successes >= Perfect-success error successes + 2 of 21
AND
Repair-50 clean successes >= Perfect-success clean successes - 1 of 21
```

Report the task-level paired difference, paired interval, and discordant task
counts. Exact repetition of the injected failed call is diagnostic only.
Complete protocol-valid negative or inconclusive outcomes must be reported.

## 9. Outcome-independent hour-nine extension

The optional extension evaluates:

```text
failure_raw
repair_100
```

on the same 21 derived-validation tasks and two conditions:

```text
2 models × 21 tasks × 2 conditions × 1 trial = 84 extension rollouts
```

It is admitted if and only if a complete, hash-valid core receipt exists no
later than `T0 + 9 hours`. Admission must be written before reading any core
metric values. The controller records:

```text
extension_authorized
core_receipt_sha256
core_completed_at_utc
uses_metric_values=false
```

If the receipt arrives after hour nine, skip the extension regardless of
whether the core outcome looks favorable or unfavorable. Skipping this
optional extension does not invalidate a complete core result.

## 10. Stage-by-stage GitHub evidence

The immutable execution checkout must not be used as the result-writing
checkout. Create a separate result worktree or clone from the published source
commit and use one dedicated branch, for example:

```text
results/v5.3-12h-screen-20260726
```

After each terminal stage, copy its bounded receipt, complete stage log,
configuration/hash manifest, and small result files into the result checkout,
then credential-scan, commit, and push. At minimum, publish updates after:

```text
preflight
self-test
prefetch
protocol
generation
data audit/gate
training
core evaluation
extension decision/evaluation
final summary/package
```

Ordinary Git must never contain:

- Hugging Face or vLLM model caches;
- base-model weights or `.safetensors`;
- virtual environments or CUDA packages;
- credentials, tokens, SSH private keys, or environment dumps containing
  secrets;
- unbounded debug directories.

Keep adapters and any large raw artifacts on persistent `/workspace` storage
and publish their paths, sizes, and SHA-256 values. Upload bounded logs,
contracts, audits, metrics, summaries, and inventories to GitHub. If GitHub
authentication or push permission is unavailable, preflight must stop before
generation with `UPLOAD_PATH_UNAVAILABLE_NO_RUN`.

Never force-push or overwrite another result branch.

## 11. Shutdown requirement

After the final package commit is visible on GitHub:

1. stop all vLLM, training, evaluator, and monitoring processes;
2. verify no GPU process remains;
3. flush logs and record the final package commit and URL;
4. use the RunPod management plane to **Stop Pod**.

Killing Python or closing SSH does not stop RunPod billing. The operator must
confirm the Pod status is `Stopped`. Do not terminate/delete the persistent
volume unless separately authorized.

If management-plane authorization is unavailable, emit
`USER_ACTION_REQUIRED_STOP_POD` only after the final upload and immediately
request that one action from the user.

## 12. Required final report

Report:

1. source commit, tau2 commit, config hash, and protocol hash;
2. hardware and exact model revisions;
3. T0, hour-nine deadline, and hour-twelve deadline;
4. preflight, self-test, prefetch, and protocol status;
5. completed trajectories versus exactly 288;
6. strict-judge evidence status and rejection/retry counts;
7. tasks with an eligible pair and capped eligible-pair count;
8. gate decision and whether training started;
9. all four training statuses and adapter hashes, if authorized;
10. complete 126-rollout core results and paired directional decision;
11. hour-nine extension decision and complete extension results, if admitted;
12. exact terminal status, including `INCOMPLETE_NO_CLAIM` where applicable;
13. `official_test_used=false` and `frozen_v5_3_configs_seeds_data_results_modified=false`;
14. GitHub result branch/commits and retained `/workspace` artifact hashes;
15. confirmed final RunPod status and measured wall-clock/GPU-hours.
