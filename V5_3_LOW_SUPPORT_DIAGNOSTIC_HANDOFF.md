# V5.3 low-support diagnostic handoff

## Scope

This is a post-yield exploratory diagnostic. It may run only after the
complete V5.3-12h generation snapshot fails the registered 14-task/17-pair
gate but satisfies the separate 8-task/10-capped-pair gate. It is not formal
V5.3, cannot be pooled into formal V5.3, does not open the official test, and
cannot establish a paper-level result.

The only trained adapters are:

```text
perfect_success
repair_50
```

The only evaluated arms are:

```text
base_model
perfect_success
repair_50
```

The implementation refuses subsets, supersets, a passing source strict gate,
fewer than 8 distinct paired tasks, fewer than 10 capped pairs, more than 2
pairs per task, any non-21-task validation manifest, any non-three-shard
evaluation, and any output outside the isolated diagnostic roots.

## Normative unattended launch

The reviewed single-host controller is the normative execution path. The
expanded commands later in this document explain the frozen stages, but must
not be copied into separate shells: doing that creates gaps in the whole-run
source lock and loses process-group cleanup.

After checking out the newly reviewed diagnostic execution commit, launch
inside a persistent `tmux` session so an SSH disconnect cannot terminate the
controller:

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
export HF_HOME=/workspace/cache/huggingface
export DIAGNOSTIC_COMMIT="$(git rev-parse HEAD^{commit})"
mkdir -p artifacts/v5_3_low_support_diagnostic

tmux new-session -d -s v5_3_low_support \
  "cd '$PWD' && exec /workspace/venvs/v5-2-train/bin/python \
scripts/run_v5_3_low_support_diagnostic.py --stage all \
--expected-processing-commit '$DIAGNOSTIC_COMMIT' \
--hf-home '$HF_HOME' \
> artifacts/v5_3_low_support_diagnostic/controller.log 2>&1"

tmux list-panes -t v5_3_low_support \
  -F 'session=#{session_name} pid=#{pane_pid} dead=#{pane_dead}'
```

Do not replace `tmux` with a bare `&`. The controller also handles
SIGINT/SIGTERM/SIGHUP fail-closed, but `tmux` is what keeps the intended run
alive across an SSH/PTTY disconnect. The controller itself is unattended: it validates
the source failure, runs both smoke arms in parallel, runs both formal arms in
parallel, starts four services, requires exact `/v1/models`, performs exactly
10 endpoint×alias one-token smokes, evaluates three shards concurrently with
three arms serial within each shard, writes the raw-bound summary, and always
cleans child process groups and services in `finally`.

Preflight refuses any pre-existing CUDA compute process. Each service gets an
identity receipt binding its PID, Linux start time, boot ID, launch-command
SHA-256, and post-health command-line SHA-256. Cleanup signals only the matching
process group (TERM, then bounded KILL) and refuses a reused PID.

Status is read-only:

```bash
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_low_support_diagnostic.py \
  --stage status \
  --expected-processing-commit "$DIAGNOSTIC_COMMIT"
```

The only valid success terminal state is:

```text
results/v5_3_low_support_diagnostic/TERMINAL_STATUS.json
status = DIAGNOSTIC_COMPLETE_DESCRIPTIVE_ONLY
```

Any other terminal state is `INCOMPLETE_NO_CLAIM`; never extract partial
metrics from it.

## Immutable roots and commits

Use one clean reviewed diagnostic commit and retain the distinct commit
recorded by the completed source-generation contracts:

```bash
export REPO=/workspace/repos/recovery-aware-trajectory-selection
export TAU2="$REPO/data/raw/tau2-bench"
export TRAIN_PY=/workspace/venvs/v5-2-train/bin/python
export SERVE_PY=/workspace/venvs/v5-2-serve/bin/python
export VLLM=/workspace/venvs/v5-2-serve/bin/vllm
export SOURCE_PROTOCOL="$REPO/data/processed/v5_3_12h_screen_protocol"
export SOURCE_RAW="$REPO/data/raw/v5_3_12h_screen_generation"
export SOURCE_RUNTIME="$REPO/artifacts/v5_3_12h_screen"
export DATA="$REPO/data/processed/v5_3_low_support_diagnostic"
export RESULTS="$REPO/results/v5_3_low_support_diagnostic"
export RUNTIME="$REPO/artifacts/v5_3_low_support_diagnostic"
export STUDENT_REV=a09a35458c702b33eeacc393d103063234e8bc28
export DIAGNOSTIC_COMMIT='<REVIEWED_LOW_SUPPORT_COMMIT>'
export SOURCE_GENERATION_COMMIT='f631dd0d7d795daad037e3548e637045ee1425e7'
export SOURCE_SNAPSHOT="$REPO/V5_3_12H_SOURCE_SNAPSHOT_C3.json"
```

Before GPU work:

```bash
cd "$REPO"
test "$(git rev-parse HEAD^{commit})" = "$DIAGNOSTIC_COMMIT"
git diff --quiet --
git diff --cached --quiet --
test -f "$SOURCE_RUNTIME/controller.lock"
test -f "$SOURCE_SNAPSHOT"
test ! -e "$DATA"
test ! -e "$RESULTS"
test ! -e "$RUNTIME"
```

All three source contracts must be `COMPLETE`, name the fixed
`SOURCE_GENERATION_COMMIT` above, cover exactly 288 cases, and bind complete
strict judge evidence. The source `TERMINAL_STATUS.json` must say exactly
`status=DATA_GATE_FAIL_NO_TRAIN`, `reason=DO_NOT_TRAIN`,
`no_scientific_claim=true`, and `official_test_used=false`; a generic
incomplete terminal is not admissible. The source `audit.json` must
independently recompute to the
registered strict 14-task/17-pair `FAIL_CLOSED` gate. The controller and
preparation script both revalidate those facts from raw bytes. They also
require the reviewed source-snapshot receipt to match exactly 3 contracts,
12 result files, 288 cases, every pre-repair SHA-256, and aggregate
`56418d6b8d06822947b7f67888d32adfe764c26db1cfefa84a4c5a2a2ea47082`.
Changing a result and its contract together therefore still fails admission.

## One-time source audit repair

If the completed source generation ended at the known pre-audit
`DATA_AUDIT_PROTOCOL` interface error, preserve the old terminal and logs,
confirm the source processed root is absent, and run only source `prepare`
from the same reviewed commit that will run this diagnostic:

```bash
"$TRAIN_PY" scripts/run_v5_3_12h_screen.py \
  --stage prepare \
  --expected-source-commit "$DIAGNOSTIC_COMMIT" \
  --expected-generation-source-commit "$SOURCE_GENERATION_COMMIT" \
  --tau2-root "$TAU2" \
  --serve-venv /workspace/venvs/v5-2-serve \
  --train-venv /workspace/venvs/v5-2-train \
  --workspace-root /workspace \
  --protocol-root "$SOURCE_PROTOCOL" \
  --raw-root "$SOURCE_RAW" \
  --processed-root "$REPO/data/processed/v5_3_12h_screen" \
  --results-root "$REPO/results/v5_3_12h_screen" \
  --runtime-root "$SOURCE_RUNTIME"
```

This command must end with the expected no-training terminal
`DATA_GATE_FAIL_NO_TRAIN / DO_NOT_TRAIN`, a hash-bound source audit, and no
source training/evaluation directories. Never use `--stage all` for this
cross-commit recovery.

## Whole-run mutual exclusion

The source V5.3-12h controller and this diagnostic must never overlap. Run all
steps below in one shell that holds a shared lock for its entire lifetime:

```bash
exec 9<"$SOURCE_RUNTIME/controller.lock"
flock -s -n 9
export V5_3_LOW_SUPPORT_LOCK_FD=9
```

If `flock` fails, stop. Do not point the command at another lock file. Do not
restart the V5.3-12h controller until the diagnostic shell exits. The
preparation, training, registry, evaluation, and summary CLIs all verify and
reuse this exact inherited descriptor. This holds the shared lock across the
whole pipeline, including gaps between child commands.

## Prepare and gate

```bash
"$TRAIN_PY" scripts/prepare_v5_3_low_support_diagnostic.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest "$SOURCE_PROTOCOL/source_generation_manifest.json" \
  --validation-manifest "$SOURCE_PROTOCOL/validation_manifest.json" \
  --screen-manifest "$SOURCE_PROTOCOL/screen_manifest.json" \
  --generation-dynamic-audit "$SOURCE_PROTOCOL/generation_dynamic_audit.json" \
  --validation-dynamic-audit "$SOURCE_PROTOCOL/validation_dynamic_audit.json" \
  --source-snapshot-receipt "$SOURCE_SNAPSHOT" \
  --raw-dir "$SOURCE_RAW" \
  --tau2-root "$TAU2" \
  --source-runtime-root "$SOURCE_RUNTIME" \
  --output-dir "$DATA" \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision "$STUDENT_REV" \
  --expected-source-commit "$DIAGNOSTIC_COMMIT" \
  --expected-generation-source-commit "$SOURCE_GENERATION_COMMIT" \
  --local-files-only
```

Continue only if `audit.json` says:

```text
status = PASS
decision = LOW_SUPPORT_DIAGNOSTIC_TRAINING_AUTHORIZED
data_gate.status = PASS_LOW_SUPPORT_DIAGNOSTIC
data_gate.source_strict_gate.status = FAIL_CLOSED
arms = exactly perfect_success and repair_50
```

The script creates exactly 512 rows per arm. `repair_50` has 256 perfect rows
and 256 repair rows; the failed action remains masked context and is never a
positive label. Repeated schedule rows increase optimizer exposure, not the
number of independent examples.

## Smoke and formal training

Run both arms in parallel, first smoke and then formal. Do not select a
checkpoint from validation performance.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for MODE in smoke formal; do
  CUDA_VISIBLE_DEVICES=0 "$TRAIN_PY" scripts/train_v5_sft_causal.py \
    --train-file "$DATA/arms/perfect_success/train.jsonl" \
    --validation-file "$DATA/validation_loss.jsonl" \
    --data-audit "$DATA/audit.json" \
    --data-hashes "$DATA/hashes.json" \
    --output-dir "$RESULTS/training/perfect_success/$MODE" \
    --arm perfect_success --mode "$MODE" \
    --model-revision "$STUDENT_REV" \
    --expected-source-commit "$DIAGNOSTIC_COMMIT" \
    --expected-train-sha256 "$(jq -r '.["arms/perfect_success/train.jsonl"]' "$DATA/hashes.json")" \
    --expected-validation-sha256 "$(jq -r '.["validation_loss.jsonl"]' "$DATA/hashes.json")" \
    --local-files-only &
  P0=$!

  CUDA_VISIBLE_DEVICES=1 "$TRAIN_PY" scripts/train_v5_sft_causal.py \
    --train-file "$DATA/arms/repair_50/train.jsonl" \
    --validation-file "$DATA/validation_loss.jsonl" \
    --data-audit "$DATA/audit.json" \
    --data-hashes "$DATA/hashes.json" \
    --output-dir "$RESULTS/training/repair_50/$MODE" \
    --arm repair_50 --mode "$MODE" \
    --model-revision "$STUDENT_REV" \
    --expected-source-commit "$DIAGNOSTIC_COMMIT" \
    --expected-train-sha256 "$(jq -r '.["arms/repair_50/train.jsonl"]' "$DATA/hashes.json")" \
    --expected-validation-sha256 "$(jq -r '.["validation_loss.jsonl"]' "$DATA/hashes.json")" \
    --local-files-only &
  P1=$!

  wait "$P0"
  wait "$P1"
done
```

Each formal run must finish exactly 64 optimizer steps with finite loss and
produce a hash-bound `checkpoint_final`.

## Registry

```bash
"$TRAIN_PY" scripts/build_v5_checkpoint_registry.py \
  --provenance-profile v5_3_low_support_diagnostic \
  --source-commit "$DIAGNOSTIC_COMMIT" \
  --base-revision "$STUDENT_REV" \
  --arm "perfect_success=$RESULTS/training/perfect_success/formal" \
  --arm "repair_50=$RESULTS/training/repair_50/formal" \
  --output "$RESULTS/checkpoint_registry.json"
```

The registry must contain exactly the base model plus the two adapters and
must preserve the source-generation commit, diagnostic processing commit,
low-support counts, claim boundary, and official-test seal.

## Evaluation topology

Use GPU 0 for the pinned 14B-AWQ user simulator and strict judge. Use GPUs
1-3 for three identical 7B services, each serving the base alias and both
registered LoRA adapters. Freeze:

```text
context = 32768
temperature = 0
max_tokens = 512
max_steps = 60
timeout = 900 seconds
seed = 20260731
trials = 1
shards = 3
```

Start the pinned services:

```bash
mkdir -p "$RESULTS/logs" "$RESULTS/pids"

CUDA_VISIBLE_DEVICES=0 "$VLLM" serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --revision 539535859b135b0244c91f3e59816150c8056698 \
  --tokenizer-revision 539535859b135b0244c91f3e59816150c8056698 \
  --host 127.0.0.1 --port 8001 \
  --served-model-name v5-3-low-support-user-judge \
  --api-key diagnostic-user-judge-local \
  --dtype float16 --quantization awq \
  --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm \
  >"$RESULTS/logs/eval-user-judge-gpu0.log" 2>&1 &
echo $! >"$RESULTS/pids/eval-user-judge-gpu0.pid"

for GPU in 1 2 3; do
  PORT=$((8100 + GPU))
  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve Qwen/Qwen2.5-7B-Instruct \
    --revision "$STUDENT_REV" --tokenizer-revision "$STUDENT_REV" \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name v5-3-low-support-base \
    --api-key diagnostic-agent-local \
    --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --generation-config vllm \
    --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 4 \
    --lora-modules \
      "v5-3-low-support-perfect-success=$RESULTS/training/perfect_success/formal/checkpoint_final" \
      "v5-3-low-support-repair-50=$RESULTS/training/repair_50/formal/checkpoint_final" \
    >"$RESULTS/logs/eval-agent-gpu${GPU}.log" 2>&1 &
  echo $! >"$RESULTS/pids/eval-agent-gpu${GPU}.pid"
done
```

Do not begin evaluation until `/v1/models` on port 8001 exposes exactly the
user/judge alias and each of ports 8101-8103 exposes exactly the base plus both
adapter aliases. Send one one-token smoke request for every endpoint×alias
pair: 1 on the user/judge endpoint plus 3 on each of the three agent endpoints,
for exactly 10 smokes. The controller writes
`runtime_service_evidence.json`; every one of the nine evaluation contracts
must bind its path, SHA-256, and canonical SHA-256. The receipt stores the raw
four `/models` responses and all ten raw smoke responses so their hashes and
derived aliases can be recomputed. Before every low-support evaluation shard
starts, the evaluator requires the live alias sets to be exact and verifies
that each PID/start-time/command-line receipt still names the same live
service. The controller repeats the live identity and exact-alias checks after
every arm barrier, including the final arm, before stopping the services.

For each shard `S` in `0,1,2`, use agent port `8101 + S` and run its three
arms sequentially. The three shard workers may run concurrently. Every
evaluator invocation must include both adapter bindings:

```bash
"$SERVE_PY" scripts/run_v5_sft_causal_eval.py \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest "$SOURCE_PROTOCOL/validation_manifest.json" \
  --dynamic-audit "$SOURCE_PROTOCOL/validation_dynamic_audit.json" \
  --checkpoint-registry "$RESULTS/checkpoint_registry.json" \
  --runtime-service-evidence "$RESULTS/runtime_service_evidence.json" \
  --provenance-profile v5_3_low_support_diagnostic \
  --adapter-dir "perfect_success=$RESULTS/training/perfect_success/formal/checkpoint_final" \
  --adapter-dir "repair_50=$RESULTS/training/repair_50/formal/checkpoint_final" \
  --output-dir "$RESULTS/evaluation/<ARM>" \
  --arm '<ARM>' \
  --agent-model 'openai/v5-3-low-support-<ARM-ALIAS>' \
  --agent-api-base 'http://127.0.0.1:<SHARD-PORT>/v1' \
  --agent-api-key diagnostic-agent-local \
  --user-model openai/v5-3-low-support-user-judge \
  --user-api-base http://127.0.0.1:8001/v1 \
  --user-api-key diagnostic-user-judge-local \
  --judge-model openai/v5-3-low-support-user-judge \
  --judge-api-base http://127.0.0.1:8001/v1 \
  --judge-api-key diagnostic-user-judge-local \
  --condition both \
  --shard-index '<S>' --num-shards 3 \
  --max-steps 60 --timeout 900 --max-tokens 512 \
  --seed 20260731 --num-trials 1
```

Replace `<ARM-ALIAS>` with `base`, `perfect-success`, or `repair-50`.
Do not substitute another model, split, seed, shard count, retry population,
or evaluator. Stop all vLLM services after all nine contracts are complete.

## Raw-bound summary

```bash
"$TRAIN_PY" scripts/summarize_v5_3_low_support_diagnostic.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --evaluation-manifest "$SOURCE_PROTOCOL/validation_manifest.json" \
  --dynamic-audit "$SOURCE_PROTOCOL/validation_dynamic_audit.json" \
  --checkpoint-registry "$RESULTS/checkpoint_registry.json" \
  --results-root "$RESULTS" \
  --output "$RESULTS/diagnostic_summary.json"
```

The summary is valid only with exactly 126 raw cases:

```text
3 arms × 21 derived-validation tasks × 2 conditions × 1 trial
```

It recomputes all metrics from raw evidence, reports the actual distinct task
and capped-pair support, and labels every comparison as descriptive. Missing,
partial, predicted, hand-entered, or imputed results produce no claim.

## Expected runtime on four RTX 5090 GPUs

Starting from an already complete, cached, hash-valid source snapshot:

- preparation and audit: about 10-20 minutes;
- two smoke plus two formal QLoRA runs: about 20-45 minutes;
- service startup, registry, and alias checks: about 10-20 minutes;
- 126 end-to-end validation rollouts: about 45-120 minutes;
- summary and final audit: under 10 minutes.

Expected total is roughly 1.5-3 hours. This estimate is operational, not a
guarantee; long τ2 trajectories or service queueing can extend evaluation.
