#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 CONTROLLER_PID REPO MAIN_LOG SOURCE_COMMIT STATUS_DIR" >&2
  exit 64
fi

controller_pid="$1"
repo="$2"
main_log="$3"
source_commit="$4"
status_dir="$5"
results_root="$repo/results/v5_5_full"
watch_log="$status_dir/v551_postprocess_watch.log"
status_file="$status_dir/v551_postprocess_status.json"

mkdir -p "$status_dir"
exec >>"$watch_log" 2>&1
echo "WATCH_START $(date -u --iso-8601=seconds) controller_pid=$controller_pid"

while kill -0 "$controller_pid" 2>/dev/null; do
  sleep 60
done

cd "$repo"
complete_count="$(grep -c '"event": "EVAL_COMPLETE"' "$main_log" || true)"
phase_count="$(grep -c '"event": "PHASE_COMPLETE", "phase": "evaluate"' "$main_log" || true)"
command_count="$(wc -l < "$results_root/evaluation_commands.jsonl")"
metrics_count="$(find "$results_root/evaluation" -type f -name metrics.json | wc -l)"

if [[ "$complete_count" -ne 9 || "$phase_count" -ne 1 || "$command_count" -ne 27 || "$metrics_count" -ne 27 ]]; then
  printf '{"status":"FAIL_CLOSED","completed_batches":%s,"phase_complete_events":%s,"evaluation_commands":%s,"metrics_files":%s,"checked_at":"%s"}\n' \
    "$complete_count" "$phase_count" "$command_count" "$metrics_count" \
    "$(date -u --iso-8601=seconds)" >"$status_file"
  echo "WATCH_FAIL_CLOSED batches=$complete_count phases=$phase_count commands=$command_count metrics=$metrics_count"
  exit 1
fi

if [[ -e "$results_root/summary" || -e "$results_root/summary.log" ]]; then
  printf '{"status":"FAIL_CLOSED","reason":"summary output already exists","checked_at":"%s"}\n' \
    "$(date -u --iso-8601=seconds)" >"$status_file"
  echo "WATCH_FAIL_CLOSED summary output already exists"
  exit 1
fi

/root/v55-train/bin/python-v55 scripts/run_v5_5_full.py \
  --phase summarize \
  --experiment-mode reference-screen \
  --source-commit "$source_commit" \
  --tau2-root /workspace/repos/tau2-bench \
  --train-python /root/v55-train/bin/python-v55 \
  --serve-python /workspace/venvs/v5_4_4500/bin/python-v55serve

/workspace/venvs/v5_4_4500/bin/python-v55serve - "$results_root/summary/summary.json" "$status_file" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

summary_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
grid = summary.get("grid_audit", {})
checks = {
    "summary_status_pass": summary.get("status") == "PASS",
    "grid_status_pass": grid.get("status") == "PASS",
    "observed_rows_378": grid.get("observed_rows") == 378,
    "official_test_unused": summary.get("official_test_used") is False,
}
payload = {
    "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
    "checks": checks,
    "summary": str(summary_path),
    "checked_at": datetime.now(timezone.utc).isoformat(),
}
status_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if payload["status"] != "PASS":
    raise SystemExit("post-summary integrity checks failed")
PY

echo "WATCH_COMPLETE $(date -u --iso-8601=seconds)"
