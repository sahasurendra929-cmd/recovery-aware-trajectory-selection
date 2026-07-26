# Prompt for the V5.3 12-hour screen on 4×RTX 5090

Obtain implementation commit `C` from the separately published release
manifest. Replace `<EXECUTION_COMMIT_C_FROM_RELEASE_MANIFEST>` with `C`;
the implementation commit is not required to contain its own Git SHA.

---

You are the sole operator of the isolated V5.3 12-hour exploratory screen in:

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

Required detached source commit:

```text
<EXECUTION_COMMIT_C_FROM_RELEASE_MANIFEST>
```

Dedicated controller:

```text
scripts/run_v5_3_12h_screen.py
```

This screen is not the frozen V5.3 pilot or formal V5.3 experiment. Never edit
or reuse frozen V5.3 configs, seeds, raw roots, processed roots, results, or
decisions. Screen bytes, checkpoints, and metrics are permanently ineligible
for formal V5.3. Keep the official test sealed.

Read these files completely before acting:

```text
V5_3_12H_SCREEN_HANDOFF.md
configs/v5_3_12h_screen.yaml
scripts/v5_3_12h_protocol.py
scripts/prepare_v5_3_12h_screen.py
scripts/run_v5_3_12h_supervisor.py
tests/test_v5_3_12h_protocol.py
scripts/run_v5_3_12h_screen.py
```

Fail closed if the prompt, handoff, or published source still contains an
unresolved controller/commit TODO.

Use only persistent `/workspace` resources:

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

Validate that the commit is exactly 40 lowercase hexadecimal characters,
checkout that commit detached, verify a clean tracked worktree, and verify
tau2 at:

```text
fc0055dc4e0a316c3f83133267fbd6faaa770992
```

Do not install, upgrade, downgrade, or recreate CUDA, PyTorch, vLLM, the
serving environment, or the training environment. Reuse the existing
`/workspace/venvs` and `/workspace/cache/huggingface`. If the frozen package
checks fail, upload an `ENVIRONMENT_MISMATCH_NO_RUN` preflight receipt and
stop. The prefetch stage may fetch missing files for pinned model revisions;
it may not modify Python packages.

Before GPU work, verify and record:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader
python3 --version
df -h /workspace
```

Require exactly four RTX 5090 GPUs with at least 30,000 MiB reported
memory, writable persistent storage, clean source/tau2 checkouts, valid
environments, exact model revisions, free required ports, and working GitHub
push authorization.

Create a separate result checkout/branch for bounded evidence. Never change
the detached execution checkout to make result commits. After every terminal
stage, push its complete bounded log, receipt, hashes, and small results to
that branch. Do not push model caches, base weights, `.safetensors`, virtual
environments, credentials, private keys, or unbounded debug trees. Retain
adapters and large raw artifacts in `/workspace` and push their SHA-256
inventory. If stage-by-stage push is unavailable, stop before generation with
`UPLOAD_PATH_UNAVAILABLE_NO_RUN`.

Create the hash-bound UTC deadline receipt before preflight. That timestamp is
T0. Do not pause, reset, or replace it:

```text
extension deadline = T0 + 9 hours
hard deadline = T0 + 12 hours
```

Use `scripts/run_v5_3_12h_screen.py` as the only **scientific experiment**
executor, launched only through the detached crash-safe supervisor. The
supervisor owns authoritative T0, singleton execution, the boot-bound process
group ledger, and TERM→KILL cleanup if the controller or SSH session dies:

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

Do not launch manual vLLM services or substitute
`scripts/run_v5_3_single_host.py`.

The controller and supervisor do not call GitHub or the RunPod management
plane. You, the outer Codex operator, must independently (a) verify push access
before this
command, (b) publish bounded evidence, and (c) Stop Pod and verify `Stopped`
after the command terminates. A controller terminal JSON is not proof that
GitHub was updated or billing stopped. If the controller crashes or loses
SSH, first inspect `supervisor_receipt.json`; use the supervisor `stop`
command or controller `--stage stop-services` only as documented, publish the
failure receipt, and use the RunPod management plane to Stop Pod.

After those four stages pass, run the dedicated controller's generation
stage. It must produce exactly:

```text
24 registered tasks
× 2 conditions (clean and controlled error)
× 6 registered attempts
= 288 trajectories
```

The independent screen schedule is:

```text
base seed: 20260731
trial seeds: 25987, 293840, 725284, 249400, 591103, 257709
```

Generation services are frozen:

- GPU 0: `Qwen/Qwen2.5-14B-Instruct-AWQ`, revision
  `539535859b135b0244c91f3e59816150c8056698`, fixed user simulator and
  strict judge;
- GPUs 1-2: `Qwen/Qwen2.5-32B-Instruct-AWQ`, revision
  `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, TP=2 trajectory teacher;
- GPU 3: orchestration/reserve, never a fallback teacher.

Never add attempts, rescue failures, replace seeds, relabel failures, or
calculate a gate from incomplete shards. All 288 attempts, generation
contracts, and strict-judge evidence must be complete.

Run the audit stage once. The immutable training gate is:

```text
at least 14 distinct tasks with an eligible clean/error pair
AND
at least 17 eligible pairs after capping every task at 2 pairs
```

If the gate fails, upload the complete audited negative feasibility result,
set `training_started=false`, do not train or evaluate, package the terminal
result, stop all processes, confirm the GitHub push, and Stop the RunPod Pod.
Do not lower the threshold or generate more data.

If and only if the gate passes, run the training stage. Train all four 7B
QLoRA arms concurrently, one per GPU:

```text
GPU 0: perfect_success
GPU 1: failure_raw
GPU 2: repair_50
GPU 3: repair_100
```

Use only `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, seed `20260731`, exactly 64
optimizer steps, no early stopping, no validation-based selection, exact
task-support matching, and this exact row-weighted recovery mixture:

```text
perfect_success:   0/512 recovery rows
failure_raw:     512/512 recovery rows
repair_50:       256/512 recovery rows
repair_100:      512/512 recovery rows
```

Row share is the optimization dose because batch size is one and every
microbatch loss is mean-reduced before equal gradient accumulation. Minimize
and report supervised-token and nonpadding-token differences (1% and 2%
targets), but treat them only as secondary content/compute controls; a target
miss does not replace or weaken the exact row-share contract. Do not claim
that Repair-50 is exactly 50% by supervised-token mass. Never alter the
training protocol to handle OOM or save time. A failed tool call may remain in
historical context but must be masked from a repair label exactly as
registered.

After all four training receipts and adapter hashes pass, shut down training
workers and configure shard-parallel claim-bearing evaluation:

```text
GPU 0: fixed 14B user simulator and strict judge
GPU 1: 7B agent endpoint for validation shard 0
GPU 2: 7B agent endpoint for validation shard 1
GPU 3: 7B agent endpoint for validation shard 2
```

Each GPU 1-3 endpoint serves the same base alias and all four hash-registered
LoRA aliases. Run arms sequentially within each shard. Every request must name
exactly one registry alias; vLLM must keep `max_loras=1`, base requests must
activate no adapter, and adapted requests must activate only the named LoRA.
Never merge adapters or carry mutable model state across arms.

The core request set is exactly `base_model`, `perfect_success`, and
`repair_50`. Merely registering Failure-raw and Repair-100 on each endpoint
does not authorize requests to them: those two arms remain forbidden until
the metric-independent hour-nine extension receipt authorizes them.

The evaluator must remain the same pinned 14B model across all arms. Never use
a 7B model to judge itself. Evaluate all 21 registered derived-validation
tasks under clean and error conditions, one trial:

```text
3 agents × 21 tasks × 2 conditions = 126 core rollouts
```

Do not read the official test. Require all 126 contracts and strict-judge
evidence before aggregation; do not publish partial metrics.

The primary directional screen is:

```text
Repair-50 gains at least 2 error-condition successes out of 21
AND
Repair-50 loses no more than 1 clean-condition success out of 21
```

Report task-level paired deltas, a paired interval, and discordant counts.
Failed-call replay is diagnostic only. Report a valid negative or
inconclusive result honestly; a positive result is not guaranteed.

Write the hour-nine extension admission receipt before inspecting metric
values. Run the optional `failure_raw` and `repair_100` extension if and only
if the complete hash-valid 126-rollout core receipt existed by T0+9h. The
decision field must say `uses_metric_values=false`. If admitted, use the same
fixed 14B user/judge and run:

```text
2 agents × 21 tasks × 2 conditions = 84 extension rollouts
```

If core completed after hour nine, skip the extension regardless of outcome.

At T0+12h, launch nothing further and terminate incomplete work cleanly. If a
complete core receipt and summary do not exist, set:

```text
INCOMPLETE_NO_CLAIM
```

Never aggregate partial core outputs. If core is complete but the optional
extension is unfinished, retain the complete core result and mark only the
extension `INCOMPLETE_NO_CLAIM`.

The outer Codex operator should publish bounded stage receipts whenever a
stage finishes; this publishing is outside `--stage all` and must never mutate
the detached execution checkout. The final package must contain bounded logs
and receipts for preflight, self-test,
prefetch, protocol, generation, strict data audit, gate decision, four-arm
training, core evaluation, hour-nine extension decision, optional extension,
summary, hashes, source/environment identities, and official-test seal.

After the final package commit is visible on GitHub:

1. stop all servers, trainers, evaluators, clients, and monitors;
2. verify `nvidia-smi` shows no experiment GPU process;
3. record the final result branch URL and commit;
4. use the RunPod management plane to Stop Pod;
5. verify the Pod status is `Stopped`.

Closing SSH or killing Python is not sufficient to stop billing. Do not delete
the persistent volume. If you cannot access the RunPod management plane, emit
`USER_ACTION_REQUIRED_STOP_POD` after the final push and request exactly that
one action immediately.

Your final response must report:

1. exact source/tau2/config/protocol hashes;
2. hardware, environment checks, and model revisions;
3. T0, hour-nine, and hour-twelve timestamps;
4. stage-by-stage status and GitHub evidence commits;
5. completed generation count out of exactly 288;
6. strict-judge evidence and exclusion counts;
7. observed tasks-with-pair and capped-pair counts;
8. gate decision and `training_started`;
9. all four training receipts and adapter hashes, if trained;
10. complete 126-rollout core metrics and paired directional decision;
11. result-independent extension decision and any complete extension metrics;
12. exact terminal status, including `INCOMPLETE_NO_CLAIM` when required;
13. `official_test_used=false`;
14. `frozen_v5_3_configs_seeds_data_results_modified=false`;
15. result branch URL/commit, retained `/workspace` hashes, measured wall time,
    GPU-hours, and confirmed final Pod status.

Never claim a screen stage completed unless its receipt, hashes, and bounded
GitHub evidence are complete.

---
