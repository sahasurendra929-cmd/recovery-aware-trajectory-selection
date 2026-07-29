#!/usr/bin/env bash
set -euo pipefail

# One-Pod, three-GPU V5.6 executor.  The three SFT arms and their subsequent
# score jobs are always launched together, keeping every rented GPU occupied.
# The caller terminates the Pod after this script writes COMPLETE.

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

mkdir -p "${WORKSPACE}"
# The RunPod console may recycle a container after a startup failure.  Keep a
# durable, append-only transcript on the network volume so the next instance
# can diagnose the precise command that failed without renting GPUs to repeat
# the bootstrap blindly.
exec > >(tee -a "${RUN_LOG}") 2>&1
trap 'code=$?; printf "%s state=FAILED exit=%s command=%q\\n" "$(date -u +%FT%TZ)" "${code}" "${BASH_COMMAND}" >>"${WORKSPACE}/v5_6_runpod_failures.log"; exit "${code}"' ERR
printf '%s state=BOOTSTRAP source=%s\n' "$(date -u +%FT%TZ)" "${SOURCE_COMMIT}"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone --depth 1 --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO}"
fi
git -C "${REPO}" fetch --depth 1 origin "${BRANCH}"
git -C "${REPO}" checkout --detach "${SOURCE_COMMIT}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${SOURCE_COMMIT}"

if [[ ! -d "${TAU2_ROOT}/.git" ]]; then
  git clone https://github.com/sierra-research/tau2-bench.git "${TAU2_ROOT}"
fi
git -C "${TAU2_ROOT}" fetch origin
git -C "${TAU2_ROOT}" checkout --detach fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C "${TAU2_ROOT}" submodule update --init --recursive

python -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REPO}/requirements-gpu-v5-sft.txt"
python -m pip install -e "${TAU2_ROOT}"
cd "${REPO}"
mkdir -p "${RESULTS}/ops"
printf '%s state=PREPARING source=%s\n' "$(date -u +%FT%TZ)" "${SOURCE_COMMIT}" >"${STATUS_FILE}"

python -m pytest -q tests/test_v5_6_context_mechanism.py tests/test_v5_sft_causal_train.py
python scripts/prepare_v5_6_context_mechanism.py \
  --tau2-root "${TAU2_ROOT}" \
  --pairs artifacts/v5_5/pairs.jsonl \
  --pair-audit artifacts/v5_5/audit.json \
  --pair-manifest artifacts/v5_5/manifest.json \
  --pair-mode reference \
  --tokenizer-revision "${BASE_REVISION}" \
  --output-dir "${REPO}/data/processed/v5_6_context"
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

DATA_ROOT="${REPO}/data/processed/v5_6_context"
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
printf '%s state=TRAINING gpus=0,1,2\n' "$(date -u +%FT%TZ)" >"${STATUS_FILE}"
mkdir -p "${RESULTS}/training"
run_train 0 perfect_success & p0=$!
run_train 1 repair_25_true & p1=$!
run_train 2 repair_25_shuffled & p2=$!
wait "${p0}" "${p1}" "${p2}"

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
printf '%s state=SCORING gpus=0,1,2\n' "$(date -u +%FT%TZ)" >"${STATUS_FILE}"
mkdir -p "${RESULTS}/scores"
run_score 0 perfect_success & p0=$!
run_score 1 repair_25_true & p1=$!
run_score 2 repair_25_shuffled & p2=$!
wait "${p0}" "${p1}" "${p2}"
printf '%s state=COMPLETE action=TERMINATE_POD\n' "$(date -u +%FT%TZ)" >"${STATUS_FILE}"
