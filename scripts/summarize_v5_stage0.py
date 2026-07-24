#!/usr/bin/env python3
"""Audit and summarize the paired V5 Stage-0 end-to-end smoke results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_call(tool_call: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": tool_call["name"],
            "arguments": tool_call["arguments"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_simulations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulations = payload.get("simulations")
    if not isinstance(simulations, list):
        raise RuntimeError(f"Missing simulations list in {path}")
    return simulations


def reward(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info") or {}
    value = reward_info.get("reward")
    return float(value) if value is not None else 0.0


def analyze_error_run(
    simulation: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    messages = simulation.get("messages") or []
    expected_id = expected["tool_call_id"]
    injected_index = None
    injected_call = None
    error_index = None
    valid_later_tool_result = False
    repeated_identical = False

    for index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            if tool_call.get("id") == expected_id:
                injected_index = index
                injected_call = tool_call
        if (
            message.get("role") == "tool"
            and message.get("id") == expected_id
            and bool(message.get("error"))
        ):
            error_index = index

    if injected_call is None:
        raise RuntimeError(
            f"Task {simulation.get('task_id')} lacks the injected tool call"
        )
    if canonical_call(injected_call) != canonical_call(
        {
            "name": expected["tool_name"],
            "arguments": expected["arguments"],
        }
    ):
        raise RuntimeError(
            f"Task {simulation.get('task_id')} injected call differs from manifest"
        )

    if error_index is not None:
        target = canonical_call(injected_call)
        for message in messages[error_index + 1 :]:
            if message.get("role") == "tool" and not message.get("error"):
                valid_later_tool_result = True
            for tool_call in message.get("tool_calls") or []:
                if canonical_call(tool_call) == target:
                    repeated_identical = True

    return {
        "injected_call_index": injected_index,
        "injected_error_index": error_index,
        "injected_error_observed": error_index is not None,
        "repeated_identical_error": repeated_identical,
        "valid_post_error_tool_result": valid_later_tool_result,
        "messages_after_error": (
            len(messages) - error_index - 1 if error_index is not None else None
        ),
    }


def summarize(manifest_path: Path, results_dir: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = {row["pair_id"]: row for row in manifest["rows"]}
    by_condition: dict[str, dict[str, dict[str, Any]]] = {
        "clean": {},
        "error": {},
    }
    file_hashes: dict[str, str] = {}
    infrastructure_failures = 0

    for domain in ("retail", "airline"):
        for condition in ("clean", "error"):
            path = results_dir / f"{domain}_{condition}.json"
            if not path.exists():
                raise RuntimeError(f"Missing Stage-0 result file: {path}")
            file_hashes[path.name] = sha256_file(path)
            for simulation in load_simulations(path):
                pair_id = f"{domain}:{simulation['task_id']}"
                if pair_id not in expected_rows:
                    raise RuntimeError(f"Unexpected result task: {pair_id}")
                if pair_id in by_condition[condition]:
                    raise RuntimeError(f"Duplicate {condition} result: {pair_id}")
                if simulation.get("termination_reason") in {
                    "infrastructure_error",
                    "unexpected_error",
                    "context_window_exceeded",
                }:
                    infrastructure_failures += 1
                row = {
                    "pair_id": pair_id,
                    "domain": domain,
                    "task_id": str(simulation["task_id"]),
                    "reward": reward(simulation),
                    "success": reward(simulation) == 1.0,
                    "termination_reason": simulation.get("termination_reason"),
                    "duration_seconds": simulation.get("duration"),
                    "message_count": len(simulation.get("messages") or []),
                }
                if condition == "error":
                    row.update(
                        analyze_error_run(
                            simulation,
                            expected_rows[pair_id]["error_condition"],
                        )
                    )
                by_condition[condition][pair_id] = row

    expected_ids = set(expected_rows)
    if set(by_condition["clean"]) != expected_ids:
        raise RuntimeError("Clean task IDs do not match the manifest")
    if set(by_condition["error"]) != expected_ids:
        raise RuntimeError("Error task IDs do not match the manifest")

    paired_rows = []
    for pair_id in sorted(expected_ids):
        clean = by_condition["clean"][pair_id]
        error = by_condition["error"][pair_id]
        paired_rows.append(
            {
                "pair_id": pair_id,
                "domain": clean["domain"],
                "clean_success": clean["success"],
                "error_success": error["success"],
                "success_delta": error["reward"] - clean["reward"],
                "injected_error_observed": error["injected_error_observed"],
                "repeated_identical_error": error["repeated_identical_error"],
                "valid_post_error_tool_result": error["valid_post_error_tool_result"],
                "clean_duration_seconds": clean["duration_seconds"],
                "error_duration_seconds": error["duration_seconds"],
            }
        )

    total = len(paired_rows)
    clean_successes = sum(row["clean_success"] for row in paired_rows)
    error_successes = sum(row["error_success"] for row in paired_rows)
    injected_observed = sum(row["injected_error_observed"] for row in paired_rows)
    repeated = sum(row["repeated_identical_error"] for row in paired_rows)
    valid_post_error = sum(
        row["valid_post_error_tool_result"] for row in paired_rows
    )
    status = (
        "PASS"
        if total == 10 and injected_observed == 10 and infrastructure_failures == 0
        else "FAIL"
    )
    summary = {
        "protocol": manifest["protocol"],
        "status": status,
        "claim_boundary": "stage0_infrastructure_only",
        "paired_tasks": total,
        "total_end_to_end_runs": total * 2,
        "clean_task_success": clean_successes / total,
        "error_injected_task_success": error_successes / total,
        "paired_task_success_delta": (error_successes - clean_successes) / total,
        "injected_error_observed_rate": injected_observed / total,
        "repeated_identical_error_rate": repeated / total,
        "valid_post_error_tool_result_rate": valid_post_error / total,
        "infrastructure_failures": infrastructure_failures,
        "result_sha256": file_hashes,
        "paired_results": paired_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/v5_stage0/smoke_manifest.json"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/v5_stage0/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v5_stage0/stage0_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(
        manifest_path=args.manifest.resolve(),
        results_dir=args.results_dir.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
