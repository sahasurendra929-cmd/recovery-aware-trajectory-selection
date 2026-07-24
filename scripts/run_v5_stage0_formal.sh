#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/repos/recovery-aware-trajectory-selection-v5-stage0}"
TAU2_ROOT="${TAU2_ROOT:-/workspace/repos/tau2-bench-v1.0.1}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/v5-stage0/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts/v5_stage0_formal}"
HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LITELLM_MODEL="openai/${MODEL}"
API_BASE="http://127.0.0.1:8000/v1"
SERVER_PID=""

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY=stage0-local

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "${ARTIFACT_ROOT}"/{logs,manifests,screen/raw,formal/raw,data}
cd "${REPO_ROOT}"

git rev-parse HEAD > "${ARTIFACT_ROOT}/repo_commit.txt"
git -C "${TAU2_ROOT}" rev-parse HEAD > "${ARTIFACT_ROOT}/tau2_commit.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "${ARTIFACT_ROOT}/hardware.txt"

TAU2_ROOT="${TAU2_ROOT}" "${PYTHON_BIN}" -m unittest tests.test_v5_stage0 -v \
  > "${ARTIFACT_ROOT}/logs/unit_tests.log" 2>&1
"${PYTHON_BIN}" scripts/prepare_v5_stage0.py \
  --tau2-root "${TAU2_ROOT}" \
  --output-dir "${ARTIFACT_ROOT}/manifests" \
  > "${ARTIFACT_ROOT}/logs/prepare.log" 2>&1

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${MODEL}" \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm \
  > "${ARTIFACT_ROOT}/logs/vllm.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "${API_BASE}/models" > "${ARTIFACT_ROOT}/logs/models.json"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -200 "${ARTIFACT_ROOT}/logs/vllm.log"
    exit 20
  fi
  sleep 2
done
curl -fsS "${API_BASE}/models" > "${ARTIFACT_ROOT}/logs/models.json"

"${PYTHON_BIN}" scripts/run_v5_stage0.py \
  --tau2-root "${TAU2_ROOT}" \
  --manifest "${ARTIFACT_ROOT}/manifests/smoke_manifest.json" \
  --output-dir "${ARTIFACT_ROOT}/screen/raw" \
  --model "${LITELLM_MODEL}" \
  --api-base "${API_BASE}" \
  --agent-mode ground_truth \
  --condition both \
  --num-trials 1 \
  --seed 20260722 \
  > "${ARTIFACT_ROOT}/logs/teacher_screen.log" 2>&1

"${PYTHON_BIN}" scripts/summarize_v5_stage0.py \
  --manifest "${ARTIFACT_ROOT}/manifests/smoke_manifest.json" \
  --results-dir "${ARTIFACT_ROOT}/screen/raw" \
  --output "${ARTIFACT_ROOT}/screen/summary.json" \
  >> "${ARTIFACT_ROOT}/logs/teacher_screen.log" 2>&1

"${PYTHON_BIN}" - "${ARTIFACT_ROOT}/screen/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "protocol_pass": summary["status"] == "PASS",
    "clean_success_at_least_40pct": summary["clean_task_success"] >= 0.40,
    "recovery_success_at_least_25pct": summary["error_injected_task_success"] >= 0.25,
    "injection_observed_100pct": summary["injected_error_observed_rate"] == 1.0,
    "zero_infrastructure_failures": summary["infrastructure_failures"] == 0,
}
gate = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
with open(sys.argv[1].replace("summary.json", "gate.json"), "w", encoding="utf-8") as f:
    json.dump(gate, f, indent=2)
    f.write("\n")
if gate["status"] != "PASS":
    raise SystemExit("Teacher capability gate failed; formal generation is forbidden")
PY

"${PYTHON_BIN}" scripts/run_v5_stage0.py \
  --tau2-root "${TAU2_ROOT}" \
  --manifest "${ARTIFACT_ROOT}/manifests/formal_manifest.json" \
  --output-dir "${ARTIFACT_ROOT}/formal/raw" \
  --model "${LITELLM_MODEL}" \
  --api-base "${API_BASE}" \
  --agent-mode ground_truth \
  --condition both \
  --num-trials 2 \
  --seed 20260722 \
  > "${ARTIFACT_ROOT}/logs/formal_generation.log" 2>&1

"${PYTHON_BIN}" scripts/build_v5_stage0_data.py \
  --manifest "${ARTIFACT_ROOT}/manifests/formal_manifest.json" \
  --raw-dir "${ARTIFACT_ROOT}/formal/raw" \
  --output-dir "${ARTIFACT_ROOT}/data" \
  --teacher-model "${MODEL}" \
  --teacher-mode ground_truth \
  --tokenizer "${MODEL}" \
  --seed 20260722 \
  > "${ARTIFACT_ROOT}/logs/data_build.log" 2>&1

"${PYTHON_BIN}" - "${ARTIFACT_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and "/logs/" not in p.as_posix()
)
manifest = {
    "status": "PASS",
    "protocol": "v5_stage0_formal_data_construction",
    "files": [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ],
}
(root / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

tar -C "$(dirname "${ARTIFACT_ROOT}")" -czf \
  "${ARTIFACT_ROOT}.tar.gz" "$(basename "${ARTIFACT_ROOT}")"
echo "STAGE0_FORMAL_PASS"
