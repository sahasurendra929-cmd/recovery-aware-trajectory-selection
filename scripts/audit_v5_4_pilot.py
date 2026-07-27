#!/usr/bin/env python3
"""Fail-closed audit and decision writer for the V5.4 pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

try:
    import v5_4_pilot_protocol as protocol
except ModuleNotFoundError:
    from scripts import v5_4_pilot_protocol as protocol


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise protocol.PilotProtocolError(f"invalid JSONL {path}: {error}") from error
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def decision_markdown(decision: dict[str, Any]) -> str:
    observed = decision["observed"]
    checks = decision["checks"]
    lines = [
        "# V5.4 Counterfactual Pilot Decision",
        "",
        f"Status: **{decision['status']}**",
        "",
        "This is a data-feasibility decision only. Training remains unauthorized.",
        "",
        "## Core observations",
        "",
        f"- tasks with an eligible pair: {observed['tasks_with_pair']} / 24",
        f"- capped eligible pairs: {observed['capped_pairs']}",
        f"- executed / registered slots: {observed['executed_slots']} / 288",
        (
            "- quality-adjusted pairs per executed slot: "
            f"{observed['quality_adjusted_pairs_per_executed_slot']:.6f}"
        ),
        f"- domains represented: {', '.join(observed['domains_with_pair']) or 'none'}",
        "",
        "## Frozen checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "Official test used: **false**.",
            "",
            "A GO authorizes design of the full V5.4 experiment; it does not "
            "authorize treating this pilot as a training result.",
            "",
        ]
    )
    return "\n".join(lines)


def cost_report(
    slots: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    executed = [
        row for row in slots if row.get("status") in protocol.EXECUTED_STATUSES
    ]
    seconds = sum(
        float(row.get("gpu_seconds", 0.0))
        for row in executed
        if not isinstance(row.get("gpu_seconds", 0.0), bool)
    )
    eligible = sum(pair.get("eligible") is True for pair in pairs)
    hours = seconds / 3600.0
    return {
        "protocol": protocol.PROTOCOL,
        "executed_slots": len(executed),
        "total_gpu_seconds": seconds,
        "total_gpu_hours": hours,
        "claimed_eligible_pairs": eligible,
        "eligible_pairs_per_gpu_hour": (
            eligible / hours if hours > 0 else None
        ),
        "gpu_hours_per_eligible_pair": (
            hours / eligible if eligible > 0 else None
        ),
        "missing_gpu_time_rows": sum(
            "gpu_seconds" not in row for row in executed
        ),
    }


def difficulty_report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        pair for pair in pairs if pair.get("eligible") is True
    ]
    task_counts = Counter(
        (
            str(pair.get("difficulty_stratum", "UNASSIGNED")),
            str(pair.get("task_identity")),
        )
        for pair in accepted
    )
    strata = Counter()
    for (stratum, _task), count in task_counts.items():
        if count:
            strata[stratum] += 1
    return {
        "protocol": protocol.PROTOCOL,
        "independent_unit": "task_identity",
        "paired_tasks_by_frozen_difficulty": dict(sorted(strata.items())),
        "unassigned_is_diagnostic_failure": "UNASSIGNED" in strata,
        "note": (
            "Difficulty must be frozen from pre-V5.4 derived-inner-train "
            "evidence; V5.4 outcomes cannot define strata."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v5-3-executed-rollouts", type=int, default=288)
    args = parser.parse_args()
    slots = read_jsonl(args.slots)
    pairs = read_jsonl(args.pairs)
    decision = protocol.pilot_decision(
        slots,
        pairs,
        v5_3_executed_rollouts=args.v5_3_executed_rollouts,
    )
    write_json(args.output_root / "audit_report.json", decision)
    write_json(args.output_root / "cost_report.json", cost_report(slots, pairs))
    write_json(
        args.output_root / "difficulty_coverage.json",
        difficulty_report(pairs),
    )
    (args.output_root / "GO_NO_GO_DECISION.md").write_text(
        decision_markdown(decision), encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
