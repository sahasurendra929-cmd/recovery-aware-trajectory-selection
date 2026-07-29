#!/usr/bin/env bash
set -euo pipefail

# V5.6 executor.  With three GPUs the arms run concurrently; with one GPU
# they run sequentially so constrained hosts never rent an idle accelerator.
# The scientific schedule, fixed seeds, and scoring inputs are identical.

REPO_URL="${REPO_URL:-https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git}"
BRANCH="${BRANCH:-codex/v5.6-error-context-mechanism}"
SOURCE_COMMIT="${SOURCE_COMMIT:?set SOURCE_COMMIT to the immutable V5.6 commit}"
BASE_REVISION="${BASE_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
WORKSPACE="${WORKSPACE:-/workspace/v5_6_context}"
REPO="${REPO:-${WORKSPACE}/repo}"
TAU2_ROOT="${TAU2_ROOT:-${REPO}/data/raw/tau2-bench}"
VENV="${VENV:-${WORKSPACE}/venv}"
RESULTS="${RESULTS:-${REPO}/results/v5_6_context}"
STATUS_FILE="${STATUS_FILE:-${RESULTS}/ops/status}"
RUN_LOG="${RUN_LOG:-${WORKSPACE}/v5_6_runpod_runner.log}"
GPU_COUNT="${GPU_COUNT:-3}"
RUN_PHASE="${RUN_PHASE:-all}"
REBUILD_FAILED_ATTEMPT="${REBUILD_FAILED_ATTEMPT:-0}"

if [[ "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "3" ]]; then
  echo "GPU_COUNT must be 1 or 3; got ${GPU_COUNT}" >&2
  exit 2
fi
if [[ "${RUN_PHASE}" != "all" && "${RUN_PHASE}" != "preflight" && "${RUN_PHASE}" != "train" ]]; then
  echo "RUN_PHASE must be all, preflight, or train; got ${RUN_PHASE}" >&2
  exit 2
fi
if [[ "${REBUILD_FAILED_ATTEMPT}" != "0" && "${REBUILD_FAILED_ATTEMPT}" != "1" ]]; then
  echo "REBUILD_FAILED_ATTEMPT must be 0 or 1; got ${REBUILD_FAILED_ATTEMPT}" >&2
  exit 2
fi

mkdir -p "${WORKSPACE}"
# The RunPod console may recycle a container after a startup failure.  Keep a
# durable, append-only transcript on the network volume so the next instance
# can diagnose the precise command that failed without renting GPUs to repeat
# the bootstrap blindly.
exec > >(tee -a "${RUN_LOG}") 2>&1
trap 'code=$?; printf "%s state=FAILED exit=%s command=%q\\n" "$(date -u +%FT%TZ)" "${code}" "${BASH_COMMAND}" >>"${WORKSPACE}/v5_6_runpod_failures.log"; exit "${code}"' ERR
printf '%s state=BOOTSTRAP source=%s\n' "$(date -u +%FT%TZ)" "${SOURCE_COMMIT}"
STAGE="BOOTSTRAP"
record_status() {
  mkdir -p "$(dirname "${STATUS_FILE}")"
  printf '%s state=%s source=%s phase=%s gpu_count=%s\n' \
    "$(date -u +%FT%TZ)" "$1" "${SOURCE_COMMIT}" "${RUN_PHASE}" "${GPU_COUNT}" >"${STATUS_FILE}"
}
on_error() {
  local code="$?"
  printf '%s state=FAILED stage=%s exit=%s command=%q\n' \
    "$(date -u +%FT%TZ)" "${STAGE}" "${code}" "${BASH_COMMAND}" \
    >>"${WORKSPACE}/v5_6_runpod_failures.log"
  if [[ -d "${REPO}" ]]; then
    record_status "FAILED"
  fi
  exit "${code}"
}
trap on_error ERR
if [[ ! -d "${REPO}/.git" ]]; then
  git clone --depth 1 --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO}"
fi
git -C "${REPO}" fetch --depth 1 origin "${BRANCH}"
git -C "${REPO}" checkout --detach "${SOURCE_COMMIT}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${SOURCE_COMMIT}"

if [[ ! -d "${TAU2_ROOT}/.git" ]]; then
  git clone --depth 1 --filter=blob:none https://github.com/sierra-research/tau2-bench.git "${TAU2_ROOT}"
fi
git -C "${TAU2_ROOT}" fetch --depth 1 origin fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C "${TAU2_ROOT}" checkout --detach FETCH_HEAD
git -C "${TAU2_ROOT}" submodule update --init --recursive

python -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
# Some RunPod images export this flag globally, but the optional hf_transfer
# package is not part of the pinned environment.  Keep model downloads on the
# standard Hugging Face HTTP path rather than failing before the V5.6 precheck.
export HF_HUB_ENABLE_HF_TRANSFER=0
python -m pip install --upgrade pip
# The formal trainer rejects runtime drift.  RunPod's base images can expose a
# newer system-site torch, so install the protocol-pinned CUDA 12.8 wheel in
# the experiment venv before resolving the remaining dependencies.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
python -m pip install -r "${REPO}/requirements-gpu-v5-sft.txt"
python -m pip install -e "${TAU2_ROOT}"
cd "${REPO}"
mkdir -p "${RESULTS}/ops"
DATA_ROOT="${REPO}/data/processed/v5_6_context"
PREFLIGHT_RECEIPT="${RESULTS}/ops/preflight_receipt.json"

if [[ "${RUN_PHASE}" != "train" ]]; then
  STAGE="PREPARING"
  record_status "PREPARING"
  if [[ -e "${DATA_ROOT}" || -e "${RESULTS}/training" || -e "${RESULTS}/scores" ]]; then
    if [[ "${REBUILD_FAILED_ATTEMPT}" != "1" ]]; then
      echo "V5.6 artifacts already exist; inspect them or rerun with REBUILD_FAILED_ATTEMPT=1" >&2
      exit 2
    fi
    # These are fixed V5.6 generated paths, never caller-provided targets.
    # Rebuild remains opt-in so a failed attempt cannot be silently erased.
    rm -rf "${REPO}/data/processed/v5_6_context" "${REPO}/results/v5_6_context"
    mkdir -p "${RESULTS}/ops"
  fi

  python -m pytest -q tests/test_v5_6_context_mechanism.py tests/test_v5_sft_causal_train.py
  python scripts/prepare_v5_6_context_mechanism.py \
    --tau2-root "${TAU2_ROOT}" \
    --pairs artifacts/v5_5/pairs.jsonl \
    --pair-audit artifacts/v5_5/audit.json \
    --pair-manifest artifacts/v5_5/manifest.json \
    --pair-mode reference \
    --tokenizer-revision "${BASE_REVISION}" \
    --output-dir "${DATA_ROOT}"
  # The raw/processed Stage-0 tree is intentionally untracked; the committed
  # audit artifact is the immutable split authority for this screen.
  python scripts/prepare_v5_6_validation_manifest.py \
    --tau2-root "${TAU2_ROOT}" \
    --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
    --output "${RESULTS}/validation_manifest.json"
  python scripts/run_v5_5_reference_pairs.py \
    --tau2-root "${TAU2_ROOT}" \
    --manifest "${RESULTS}/validation_manifest.json" \
    --output "${RESULTS}/validation_pairs.jsonl"
  python - "${PREFLIGHT_RECEIPT}" "${SOURCE_COMMIT}" "${BASE_REVISION}" "${DATA_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
source_commit, model_revision = sys.argv[2:4]
data_root = Path(sys.argv[4])
receipt = {
    "status": "PASS",
    "source_commit": source_commit,
    "model_revision": model_revision,
    "data_audit_sha256": hashlib.sha256((data_root / "audit.json").read_bytes()).hexdigest(),
    "data_hashes_sha256": hashlib.sha256((data_root / "hashes.json").read_bytes()).hexdigest(),
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  record_status "PREFLIGHT_COMPLETE"
fi

if [[ "${RUN_PHASE}" == "preflight" ]]; then
  printf '%s action=START_TRAIN_PHASE receipt=%s\n' "$(date -u +%FT%TZ)" "${PREFLIGHT_RECEIPT}" >>"${STATUS_FILE}"
  exit 0
fi

STAGE="VERIFYING_PREFLIGHT"
python - "${PREFLIGHT_RECEIPT}" "${SOURCE_COMMIT}" "${BASE_REVISION}" "${DATA_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
source_commit, model_revision = sys.argv[2:4]
data_root = Path(sys.argv[4])
expected = {
    "status": "PASS",
    "source_commit": source_commit,
    "model_revision": model_revision,
    "data_audit_sha256": hashlib.sha256((data_root / "audit.json").read_bytes()).hexdigest(),
    "data_hashes_sha256": hashlib.sha256((data_root / "hashes.json").read_bytes()).hexdigest(),
}
if json.loads(receipt_path.read_text(encoding="utf-8")) != expected:
    raise SystemExit("preflight receipt is missing, stale, or does not match the checked-out source/data")
PY
VALIDATION_SHA="$(sha256sum "${DATA_ROOT}/validation_loss.jsonl" | awk '{print $1}')"
run_train() {
  local gpu="$1" arm="$2"
  local train="${DATA_ROOT}/arms/${arm}/train.jsonl"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_v5_sft_causal.py \
    --train-file "${train}" --validation-file "${DATA_ROOT}/validation_loss.jsonl" \
    --output-dir "${RESULTS}/training/${arm}/20260805" --arm "${arm}" --mode formal \
    --model-revision "${BASE_REVISION}" --expected-source-commit "${SOURCE_COMMIT}" \
    --expected-train-sha256 "$(sha256sum "${train}" | awk '{print $1}')" \
    --expected-validation-sha256 "${VALIDATION_SHA}" --data-audit "${DATA_ROOT}/audit.json" \
    --data-hashes "${DATA_ROOT}/hashes.json" --training-seed 20260805 \
    --learning-rate 1.25e-5 --formal-steps 32 >"${RESULTS}/training/${arm}.console.log" 2>&1
}
STAGE="TRAINING"
record_status "TRAINING"
mkdir -p "${RESULTS}/training"
if [[ "${GPU_COUNT}" == "3" ]]; then
  run_train 0 perfect_success & p0=$!
  run_train 1 repair_25_true & p1=$!
  run_train 2 repair_25_shuffled & p2=$!
  wait "${p0}" "${p1}" "${p2}"
else
  run_train 0 perfect_success
  run_train 0 repair_25_true
  run_train 0 repair_25_shuffled
fi

run_score() {
  local gpu="$1" arm="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/score_v5_6_context.py \
    --tau2-root "${TAU2_ROOT}" --pairs "${RESULTS}/validation_pairs.jsonl" \
    --pair-mode reference --model-revision "${BASE_REVISION}" \
    --adapter "${RESULTS}/training/${arm}/20260805/checkpoint_final" --arm "${arm}" \
    --output "${RESULTS}/scores/${arm}.jsonl" >"${RESULTS}/scores/${arm}.console.log" 2>&1
  python scripts/score_v5_6_context.py --score-jsonl "${RESULTS}/scores/${arm}.jsonl" \
    --output "${RESULTS}/scores/${arm}.summary.json"
}
STAGE="SCORING"
record_status "SCORING"
mkdir -p "${RESULTS}/scores"
if [[ "${GPU_COUNT}" == "3" ]]; then
  run_score 0 perfect_success & p0=$!
  run_score 1 repair_25_true & p1=$!
  run_score 2 repair_25_shuffled & p2=$!
  wait "${p0}" "${p1}" "${p2}"
else
  run_score 0 perfect_success
  run_score 0 repair_25_true
  run_score 0 repair_25_shuffled
fi
STAGE="COMPLETE"
record_status "COMPLETE"
printf '%s action=TERMINATE_POD\n' "$(date -u +%FT%TZ)" >>"${STATUS_FILE}"
