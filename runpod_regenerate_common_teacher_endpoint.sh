#!/usr/bin/env bash
set -Eeuo pipefail

repo=/workspace/repos/recovery-aware-trajectory-selection
raw="$repo/data/raw/v5_sft_causal_generation"
result="$repo/results/v5_sft_causal"
coord="$result/coordinator"
serve=/workspace/venvs/v5-stage1-serve/bin
source_commit=65362cc8817a20146c86d721f7e329e09c073336
tau2_commit=fc0055dc4e0a316c3f83133267fbd6faaa770992
run_tag="source-${source_commit:0:12}"
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$coord"
exec >>"$coord/regenerate_${run_tag}.log" 2>&1
stamp() { echo "[$(date -Is)] $*"; }
declare -a child_pids=()
declare -a service_pids=()
cleanup_owned_processes() {
  local pid
  for pid in "${child_pids[@]}" "${service_pids[@]}"; do
    test -n "$pid" && kill "$pid" 2>/dev/null || true
  done
}
fail_trap() {
  local code=$?
  cleanup_owned_processes
  stamp "FAILED code=$code line=${BASH_LINENO[0]}"
  exit "$code"
}
term_trap() {
  cleanup_owned_processes
  stamp "TERMINATED_WITH_CHILD_CLEANUP"
  exit 143
}
trap fail_trap ERR
trap term_trap INT TERM

cd "$repo"
test "$(git rev-parse HEAD)" = "$source_commit"
test "$(git -C data/raw/tau2-bench rev-parse HEAD)" = "$tau2_commit"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git -C data/raw/tau2-bench status --porcelain --untracked-files=no)"
if pgrep -af 'scripts/run_v5_sft_causal_generate.py' >/dev/null; then
  stamp "REFUSE_START_STALE_GENERATION_WORKERS"
  pgrep -af 'scripts/run_v5_sft_causal_generate.py'
  exit 1
fi

# A source commit is part of every generation contract. Preserve the previous
# complete run byte-for-byte and regenerate all shards for the corrected source.
if test -d "$raw"; then
  archive="${raw}_${run_tag}_predecessor_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$raw" "$archive"
  {
    echo "archived_at=$(date -Is)"
    echo "reason=regenerate_after_fail_closed_generation_interface_audit"
    echo "replacement_source_commit=$source_commit"
    find "$archive" -maxdepth 1 -type f -name '*.json' -print0 |
      sort -z | xargs -0 sha256sum
  } >"$archive/ARCHIVE_MANIFEST.txt"
  stamp "archived_previous_raw=$archive"
fi
mkdir -p "$raw"
if test -d data/processed/v5_sft_causal; then
  processed_archive="data/processed/v5_sft_causal_failed_prepare_$(date -u +%Y%m%dT%H%M%SZ)"
  mv data/processed/v5_sft_causal "$processed_archive"
  stamp "archived_partial_processed=$processed_archive"
fi

mkdir -p "$result/serving_common_endpoint/$run_tag" /workspace/v5-nginx
nginx -s stop -c /workspace/v5_teacher_lb_nginx.conf 2>/dev/null || true
sleep 2
nginx -t -c /workspace/v5_teacher_lb_nginx.conf

start_vllm() {
  local gpu=$1 port=$2 model=$3 revision=$4 served_name=$5 api_key=$6
  local max_seqs=$7 utilization=$8 log=$9
  local cache="/workspace/cache/v5-common-endpoint-gpu$gpu"
  mkdir -p "$cache/vllm" "$cache/torchinductor"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_CACHE_ROOT="$cache/vllm" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    "$serve/vllm" serve "$model" \
      --revision "$revision" --served-model-name "$served_name" \
      --host 127.0.0.1 --port "$port" --api-key "$api_key" \
      --dtype auto --max-model-len 32768 \
      --gpu-memory-utilization "$utilization" --max-num-seqs "$max_seqs" \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --generation-config vllm \
    >"$log" 2>&1 </dev/null &
  LAST_PID=$!
  service_pids+=("$LAST_PID")
}

start_vllm 0 8100 Qwen/Qwen2.5-7B-Instruct-AWQ \
  b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  Qwen/Qwen2.5-7B-Instruct-AWQ stage1-user-local 3 0.88 \
  "$result/serving_common_endpoint/$run_tag/user_judge.log"
echo "$LAST_PID" >"$result/nodes/node0/user_judge.pid"

for slot in 1 2 3; do
  start_vllm "$slot" "$((8110 + slot))" Qwen/Qwen2.5-14B-Instruct-AWQ \
    539535859b135b0244c91f3e59816150c8056698 \
    Qwen/Qwen2.5-14B-Instruct-AWQ stage1-teacher-local 1 0.90 \
    "$result/serving_common_endpoint/$run_tag/teacher_gpu${slot}.log"
  echo "$LAST_PID" >"$result/nodes/node${slot}/teacher.pid"
done

for port in 8100 8111 8112 8113; do
  ready=0
  for attempt in $(seq 1 180); do
    key=stage1-teacher-local
    test "$port" -eq 8100 && key=stage1-user-local
    if curl -fsS -H "Authorization: Bearer $key" \
      "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done
  test "$ready" -eq 1
done

nginx -c /workspace/v5_teacher_lb_nginx.conf
for attempt in $(seq 1 60); do
  curl -fsS -H "Authorization: Bearer stage1-teacher-local" \
    http://127.0.0.1:8101/v1/models >/dev/null 2>&1 && break
  test "$attempt" -lt 60
  sleep 2
done

{
  echo "topology=common_teacher_endpoint_least_conn"
  echo "contract_teacher_api_base=http://127.0.0.1:8101/v1"
  echo "backends=http://127.0.0.1:8111,http://127.0.0.1:8112,http://127.0.0.1:8113"
  echo "source_commit=$source_commit"
  echo "scientific_parameters_changed=false"
  echo "started_at=$(date -Is)"
} >"$coord/common_teacher_endpoint_topology_${run_tag}.txt"

run_shard() {
  local index=$1 node=$2
  local log="$result/nodes/$node/generation_shard${index}_${run_tag}.log"
  env -C "$repo" PYTHONPATH=. "$serve/python" \
    scripts/run_v5_sft_causal_generate.py \
      --tau2-root data/raw/tau2-bench \
      --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
      --manifest data/processed/v5_stage1_protocol/generation_manifest.json \
      --dynamic-audit data/processed/v5_stage1_protocol/generation_dynamic_audit.json \
      --output-dir data/raw/v5_sft_causal_generation \
      --teacher-model Qwen/Qwen2.5-14B-Instruct-AWQ \
      --teacher-revision 539535859b135b0244c91f3e59816150c8056698 \
      --teacher-api-base http://127.0.0.1:8101/v1 \
      --teacher-api-key stage1-teacher-local \
      --user-model Qwen/Qwen2.5-7B-Instruct-AWQ \
      --user-revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
      --user-api-base http://127.0.0.1:8100/v1 \
      --user-api-key stage1-user-local \
      --judge-model Qwen/Qwen2.5-7B-Instruct-AWQ \
      --judge-revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
      --judge-api-base http://127.0.0.1:8100/v1 \
      --judge-api-key stage1-user-local \
      --teacher-mode ground_truth --condition both \
      --shard-index "$index" --num-shards 4 --num-trials 3 \
      --seed 20260722 --max-steps 60 --timeout 900 --max-tokens 512 \
      --expected-source-commit "$source_commit" \
    >"$log" 2>&1 &
  LAST_PID=$!
  child_pids+=("$LAST_PID")
  echo "$LAST_PID" >"$result/nodes/$node/generation_shard${index}.pid"
  stamp "DISPATCH shard=$index worker=$node pid=$LAST_PID"
}

# Three client workers continuously feed the three teacher replicas. The fourth
# shard is queued and claimed by whichever client worker finishes first.
declare -A active=()
for shard in 0 1 2; do
  run_shard "$shard" "node$((shard + 1))"
  active["$LAST_PID"]=$shard
done
pending=3
failed=0
while test "${#active[@]}" -gt 0; do
  for pid in "${!active[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      shard=${active[$pid]}
      if wait "$pid"; then
        stamp "COMPLETE shard=$shard pid=$pid"
      else
        stamp "FAILED_SHARD shard=$shard pid=$pid"
        failed=1
      fi
      unset 'active[$pid]'
      if test "$pending" -eq 3 && test "$failed" -eq 0; then
        run_shard 3 "node$((shard + 1))"
        active["$LAST_PID"]=3
        pending=4
      fi
    fi
  done
  test "${#active[@]}" -gt 0 && sleep 10
done
test "$failed" -eq 0
test "$pending" -eq 4
stamp "ALL_GENERATION_SHARDS_COMPLETE"
