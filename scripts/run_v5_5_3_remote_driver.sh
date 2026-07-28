#!/usr/bin/env bash
set -euo pipefail

# Persistent RunPod handoff for the final preregistered V5.5.3 screen.
# Launch with nohup/setsid and SOURCE_COMMIT set to the pushed preregistration
# commit. Each controller phase is restart-separated so a failure preserves the
# last completed phase and never silently reruns a completed evaluation batch.

ROOT="${ROOT:-/workspace/repos/recovery-aware-trajectory-selection-v55}"
TAU2_ROOT="${TAU2_ROOT:-/workspace/repos/tau2-bench}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/root/v55-train/bin/python-v55}"
SERVE_PYTHON="${SERVE_PYTHON:-/workspace/venvs/v5_4_4500/bin/python-v55serve}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/v5_5_3_reduced_exposure}"
REGISTRY="${REGISTRY:-${RESULTS_ROOT}/checkpoint_registry.json}"
STATUS_FILE="${STATUS_FILE:-${RESULTS_ROOT}/ops/persistent_driver.status}"

: "${SOURCE_COMMIT:?SOURCE_COMMIT must be the pushed 40-character commit}"

mkdir -p "$(dirname "${STATUS_FILE}")"
cd "${ROOT}"

common=(
  --experiment-mode reduced-exposure-screen
  --source-commit "${SOURCE_COMMIT}"
  --tau2-root "${TAU2_ROOT}"
  --train-python "${TRAIN_PYTHON}"
  --serve-python "${SERVE_PYTHON}"
  --results-root "${RESULTS_ROOT}"
  --registry "${REGISTRY}"
  --local-files-only
)

write_status() {
  printf '%s phase=%s state=%s pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$$" >"${STATUS_FILE}"
}

on_error() {
  code=$?
  write_status "${current_phase:-startup}" "FAILED(exit=${code})"
  exit "${code}"
}
trap on_error ERR

for current_phase in train registry evaluate summarize; do
  write_status "${current_phase}" "RUNNING"
  "${TRAIN_PYTHON}" scripts/run_v5_5_full.py \
    --phase "${current_phase}" "${common[@]}"
  write_status "${current_phase}" "COMPLETE"
done

write_status "screen" "COMPLETE"
