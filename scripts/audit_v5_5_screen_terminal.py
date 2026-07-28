#!/usr/bin/env python3
"""Fail-closed terminal audit for a V5.5 validation screen summary."""
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = audit(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if result["status"] == "FAIL_CLOSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
