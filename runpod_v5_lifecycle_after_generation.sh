#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/repos/recovery-aware-trajectory-selection
raw="$repo/data/raw/v5_sft_causal_generation"
result="$repo/results/v5_sft_causal"
coord="$result/coordinator"
serve_python=/workspace/venvs/v5-stage1-serve/bin/python
train_python=/workspace/venvs/v5-stage1-train/bin/python
source_commit=65362cc8817a20146c86d721f7e329e09c073336
generation_source_commit=65362cc8817a20146c86d721f7e329e09c073336
tau2_commit=fc0055dc4e0a316c3f83133267fbd6faaa770992
model_revision=a09a35458c702b33eeacc393d103063234e8bc28
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$coord"
exec >>"$coord/lifecycle_after_generation.log" 2>&1

stamp() {
  echo "[$(date -Is)] $*"
}

fail_trap() {
  code=$?
  stamp "FAILED exit_code=$code line=${BASH_LINENO[0]}"
  exit "$code"
}
trap fail_trap ERR

verify_source() {
  cd "$repo"
  test "$(git rev-parse HEAD)" = "$source_commit"
  test "$(git -C data/raw/tau2-bench rev-parse HEAD)" = "$tau2_commit"
  test -z "$(git status --porcelain --untracked-files=no)"
  test -z "$(git -C data/raw/tau2-bench status --porcelain --untracked-files=no)"
}

contract_complete() {
  local index=$1
  local path="$raw/run_contract.shard-$(printf '%03d' "$index")-of-004.json"
  test -f "$path" &&
    "$train_python" -c \
      'import json,sys; row=json.load(open(sys.argv[1])); raise SystemExit(0 if row.get("status") == "COMPLETE" and row.get("source_commit") == sys.argv[2] else 1)' \
      "$path" "$generation_source_commit"
}

stamp "WAIT_GENERATION"
while true; do
  complete=0
  for index in 0 1 2 3; do
    if contract_complete "$index"; then
      complete=$((complete + 1))
    fi
  done
  stamp "generation_contracts_complete=$complete/4"
  test "$complete" -eq 4 && break
  sleep 30
done
verify_source

stamp "STOP_GENERATION_SERVERS"
for pid_file in \
  "$result/nodes/node0/user_judge.pid" \
  "$result/nodes/node1/teacher.pid" \
  "$result/nodes/node2/teacher.pid" \
  "$result/nodes/node3/teacher.pid"; do
  if test -f "$pid_file"; then
    pid=$(cat "$pid_file")
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 15
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  stamp "GPU processes remain after graceful server stop"
  nvidia-smi
  exit 1
fi

stamp "PREPARE_AND_AUDIT_DATA"
verify_source
test ! -e "$repo/data/processed/v5_sft_causal"
env -C "$repo" PYTHONPATH=. "$train_python" scripts/prepare_v5_sft_causal.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest data/processed/v5_stage1_protocol/generation_manifest.json \
  --validation-manifest data/processed/v5_stage1_protocol/validation_manifest.json \
  --generation-dynamic-audit data/processed/v5_stage1_protocol/generation_dynamic_audit.json \
  --validation-dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
  --raw-dir data/raw/v5_sft_causal_generation \
  --tau2-root data/raw/tau2-bench \
  --output-dir data/processed/v5_sft_causal \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision "$model_revision" \
  --expected-source-commit "$source_commit" \
  --expected-generation-source-commit "$generation_source_commit"

hash_file="$repo/data/processed/v5_sft_causal/hashes.json"
audit_file="$repo/data/processed/v5_sft_causal/audit.json"
validation_hash=$("$train_python" -c \
  "import json; print(json.load(open('$hash_file'))['validation_loss.jsonl'])")

declare -a arms=(perfect_success failure_raw repair_50 repair_100)
declare -a gpus=(0 1 2 3)
declare -a nodes=(node0 node1 node2 node3)

run_training_phase() {
  local mode=$1
  local -a pids=()
  stamp "TRAIN_${mode^^}_START"
  for slot in 0 1 2 3; do
    arm=${arms[$slot]}
    gpu=${gpus[$slot]}
    node=${nodes[$slot]}
    arm_hash=$("$train_python" -c \
      "import json; print(json.load(open('$hash_file'))['arms/$arm/train.jsonl'])")
    out="$result/$arm/$mode"
    test ! -e "$out"
    mkdir -p "$result/nodes/$node"
    {
      echo "stage=training"
      echo "mode=$mode"
      echo "arm=$arm"
      echo "gpu=$gpu"
      echo "source_commit=$source_commit"
      echo "train_sha256=$arm_hash"
      echo "validation_sha256=$validation_hash"
      echo "started_at=$(date -Is)"
    } > "$result/nodes/$node/training_${mode}_assignment.txt"
    env -C "$repo" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      HF_HOME="$HF_HOME" \
      CUBLAS_WORKSPACE_CONFIG="$CUBLAS_WORKSPACE_CONFIG" \
      PYTHONPATH=. \
      "$train_python" scripts/train_v5_sft_causal.py \
        --train-file "data/processed/v5_sft_causal/arms/$arm/train.jsonl" \
        --validation-file data/processed/v5_sft_causal/validation_loss.jsonl \
        --data-audit data/processed/v5_sft_causal/audit.json \
        --data-hashes data/processed/v5_sft_causal/hashes.json \
        --output-dir "results/v5_sft_causal/$arm/$mode" \
        --arm "$arm" \
        --mode "$mode" \
        --model-revision "$model_revision" \
        --expected-source-commit "$source_commit" \
        --expected-train-sha256 "$arm_hash" \
        --expected-validation-sha256 "$validation_hash" \
        --local-files-only \
      >"$result/nodes/$node/training_${mode}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  test "$failed" -eq 0
  for arm in "${arms[@]}"; do
    test -s "$result/$arm/$mode/run_manifest.json"
    test -s "$result/$arm/$mode/training_metrics.json"
  done
  stamp "TRAIN_${mode^^}_COMPLETE"
}

run_training_phase smoke
verify_source
run_training_phase formal
verify_source

stamp "BUILD_CHECKPOINT_REGISTRY"
env -C "$repo" PYTHONPATH=. "$train_python" scripts/build_v5_checkpoint_registry.py \
  --source-commit "$source_commit" \
  --base-revision "$model_revision" \
  --arm perfect_success=results/v5_sft_causal/perfect_success/formal \
  --arm failure_raw=results/v5_sft_causal/failure_raw/formal \
  --arm repair_50=results/v5_sft_causal/repair_50/formal \
  --arm repair_100=results/v5_sft_causal/repair_100/formal \
  --output results/v5_sft_causal/checkpoint_registry.json
sha256sum "$result/checkpoint_registry.json" > "$coord/checkpoint_registry.sha256"

stamp "START_FOUR_EVALUATION_SERVERS"
declare -a eval_ports=(8200 8201 8202 8203)
declare -a server_pids=()
for slot in 0 1 2 3; do
  gpu=${gpus[$slot]}
  port=${eval_ports[$slot]}
  node=${nodes[$slot]}
  cache="/workspace/cache/v5-eval-gpu$gpu"
  mkdir -p "$cache/vllm" "$cache/torchinductor"
  env -C "$repo" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$HF_HOME" \
    VLLM_CACHE_ROOT="$cache/vllm" \
    TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    /workspace/venvs/v5-stage1-serve/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
      --revision "$model_revision" \
      --served-model-name v5-base \
      --host 127.0.0.1 \
      --port "$port" \
      --api-key stage1-eval-local \
      --dtype bfloat16 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 1 \
      --enable-lora \
      --max-lora-rank 16 \
      --max-loras 1 \
      --max-cpu-loras 4 \
      --lora-modules \
        v5-perfect-success=results/v5_sft_causal/perfect_success/formal/checkpoint_final \
        v5-failure-raw=results/v5_sft_causal/failure_raw/formal/checkpoint_final \
        v5-repair-50=results/v5_sft_causal/repair_50/formal/checkpoint_final \
        v5-repair-100=results/v5_sft_causal/repair_100/formal/checkpoint_final \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --generation-config vllm \
    >"$result/nodes/$node/evaluation_server.log" 2>&1 &
  server_pids+=("$!")
  echo "$!" > "$result/nodes/$node/evaluation_server.pid"
done

for port in "${eval_ports[@]}"; do
  ready=0
  for attempt in $(seq 1 120); do
    if curl -fsS -H "Authorization: Bearer stage1-eval-local" \
      "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  test "$ready" -eq 1
  curl -fsS -H "Authorization: Bearer stage1-eval-local" \
    "http://127.0.0.1:$port/v1/models" \
    >"$result/nodes/node$((port - 8200))/evaluation_models.json"
done

eval_one() {
  local arm=$1
  local model_id=$2
  local shard=$3
  local port=${eval_ports[$shard]}
  local node=${nodes[$shard]}
  stamp "EVAL_START arm=$arm shard=$shard gpu=$shard"
  env -C "$repo" PYTHONPATH=. "$serve_python" scripts/run_v5_sft_causal_eval.py \
    --tau2-root data/raw/tau2-bench \
    --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
    --manifest data/processed/v5_sft_causal/validation_manifest.json \
    --dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
    --checkpoint-registry results/v5_sft_causal/checkpoint_registry.json \
    --adapter-dir perfect_success=results/v5_sft_causal/perfect_success/formal/checkpoint_final \
    --adapter-dir failure_raw=results/v5_sft_causal/failure_raw/formal/checkpoint_final \
    --adapter-dir repair_50=results/v5_sft_causal/repair_50/formal/checkpoint_final \
    --adapter-dir repair_100=results/v5_sft_causal/repair_100/formal/checkpoint_final \
    --output-dir "results/v5_sft_causal/evaluation/$arm" \
    --arm "$arm" \
    --agent-model "$model_id" \
    --agent-api-base "http://127.0.0.1:$port/v1" \
    --agent-api-key stage1-eval-local \
    --user-model openai/v5-base \
    --user-api-base "http://127.0.0.1:$port/v1" \
    --user-api-key stage1-eval-local \
    --judge-model openai/v5-base \
    --judge-api-base "http://127.0.0.1:$port/v1" \
    --judge-api-key stage1-eval-local \
    --condition both \
    --shard-index "$shard" \
    --num-shards 4 \
    --max-steps 60 \
    --timeout 900 \
    --max-tokens 512 \
    --seed 20260722 \
    --num-trials 1 \
    >"$result/nodes/$node/eval_${arm}.log" 2>&1
  stamp "EVAL_COMPLETE arm=$arm shard=$shard"
}

run_eval_wave() {
  local arm=$1
  local model_id=$2
  local -a pids=()
  for shard in 0 1 2 3; do
    eval_one "$arm" "$model_id" "$shard" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  test "$failed" -eq 0
}

# The capacity control is completed on all 21 tasks before trained-arm evaluation.
run_eval_wave base_model openai/v5-base
run_eval_wave perfect_success openai/v5-perfect-success
run_eval_wave failure_raw openai/v5-failure-raw
run_eval_wave repair_50 openai/v5-repair-50
run_eval_wave repair_100 openai/v5-repair-100

stamp "SUMMARIZE"
env -C "$repo" PYTHONPATH=. "$train_python" scripts/summarize_v5_sft_causal.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --evaluation-manifest data/processed/v5_sft_causal/validation_manifest.json \
  --dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
  --checkpoint-registry results/v5_sft_causal/checkpoint_registry.json \
  --arm base_model=results/v5_sft_causal/evaluation/base_model \
  --arm perfect_success=results/v5_sft_causal/evaluation/perfect_success \
  --arm failure_raw=results/v5_sft_causal/evaluation/failure_raw \
  --arm repair_50=results/v5_sft_causal/evaluation/repair_50 \
  --arm repair_100=results/v5_sft_causal/evaluation/repair_100 \
  --output results/v5_sft_causal/mechanism_screen_summary.json

sha256sum "$result/mechanism_screen_summary.json" \
  > "$coord/mechanism_screen_summary.sha256"
stamp "V5_STAGE1_COMPLETE"
