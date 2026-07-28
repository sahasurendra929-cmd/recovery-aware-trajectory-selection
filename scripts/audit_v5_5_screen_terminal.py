#!/usr/bin/env python3
"""Fail-closed terminal audit for a V5.5 validation screen summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def audit(summary: dict[str, Any]) -> dict[str, Any]:
    grid = summary.get("grid_audit") or {}
    selection = (summary.get("statistics") or {}).get("selection") or {}
    checks = {
        "summary_pass": summary.get("status") == "PASS",
        "grid_pass": grid.get("status") == "PASS",
        "rows_378": grid.get("observed_rows") == 378,
        "no_missing_rows": grid.get("missing_rows") == 0,
        "no_extra_rows": grid.get("extra_rows") == 0,
        "no_duplicate_rows": grid.get("duplicate_rows") == 0,
        "official_test_unused": summary.get("official_test_used") is False,
        "selection_present": isinstance(selection.get("positive_screen"), bool),
    }
    integrity_pass = all(checks.values())
    positive_screen = selection.get("positive_screen") if integrity_pass else None
    if not integrity_pass:
        status = "FAIL_CLOSED"
    elif positive_screen:
        status = "GO_STAGE_C"
    else:
        status = "NO_GO_REFERENCE_SCREEN"
    return {
        "protocol": "v5_5_screen_terminal_audit_v1",
        "status": status,
        "integrity_status": "PASS" if integrity_pass else "FAIL_CLOSED",
        "checks": checks,
        "positive_screen": positive_screen,
        "selected_arm": selection.get("selected_arm"),
        "selected_paper_arm": selection.get("selected_paper_arm"),
        "official_test_used": summary.get("official_test_used"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_result_grid(results_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    expected_hashes = summary.get("result_file_sha256") or {}
    shards = []
    total_rows = 0
    for arm in ("perfect_success", "repair_50", "repair_100"):
        for evaluation_seed in (20260815, 20260816, 20260817):
            for shard_index in range(3):
                relative = Path(
                    arm,
                    "20260805",
                    str(evaluation_seed),
                    f"shard-{shard_index}",
                )
                shard = results_root / "evaluation" / relative
                metrics_path = shard / "metrics.json"
                rows_path = shard / "rows.jsonl"
                contract_path = shard / "run_contract.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                rows = [
                    json.loads(line)
                    for line in rows_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                expected_hash = expected_hashes.get(
                    (relative / "rows.jsonl").as_posix()
                )
                checks = {
                    "metrics_pass": metrics.get("status") == "PASS",
                    "contract_complete": contract.get("status") == "COMPLETE",
                    "rows_14": metrics.get("rows") == 14 and len(rows) == 14,
                    "official_test_unused": (
                        metrics.get("official_test_used") is False
                        and contract.get("official_test_used") is False
                        and all(
                            row.get("official_test_used") is False for row in rows
                        )
                    ),
                    "summary_hash_matches": (
                        isinstance(expected_hash, str)
                        and sha256(rows_path) == expected_hash
                    ),
                }
                if not all(checks.values()):
                    raise RuntimeError(f"shard audit failed: {relative}: {checks}")
                total_rows += len(rows)
                shards.append(
                    {
                        "path": relative.as_posix(),
                        "checks": checks,
                        "metrics_sha256": sha256(metrics_path),
                        "rows_sha256": sha256(rows_path),
                        "run_contract_sha256": sha256(contract_path),
                    }
                )
    return {
        "status": "PASS",
        "shards": len(shards),
        "rows": total_rows,
        "summary_row_hashes": len(expected_hashes),
        "all_summary_row_hashes_verified": len(expected_hashes) == len(shards),
        "files": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = audit(summary)
    if args.results_root is not None and result["integrity_status"] == "PASS":
        result["result_grid"] = audit_result_grid(args.results_root, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "FAIL_CLOSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
