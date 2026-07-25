#!/usr/bin/env python3
"""Execute every frozen Stage-0 fault and prove it is read-only."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def configure_tau2_path(tau2_root: Path) -> None:
    source = tau2_root.resolve() / "src"
    if not source.exists():
        raise FileNotFoundError(f"tau2 source directory not found: {source}")
    sys.path.insert(0, str(source))


def verify(
    tau2_root: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    configure_tau2_path(tau2_root)

    from tau2.data_model.message import ToolCall
    from tau2.registry import registry
    from tau2.runner.build import build_environment

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for expected in manifest["rows"]:
        domain = expected["domain"]
        tasks = {
            str(task.id): task
            for task in registry.get_tasks_loader(domain)(None)
        }
        task = tasks[str(expected["task_id"])]
        environment = build_environment(domain)
        state = task.initial_state
        environment.set_state(
            state.initialization_data if state else None,
            state.initialization_actions if state else None,
            deepcopy(state.message_history)
            if state and state.message_history
            else [],
        )
        db_before = environment.get_db_hash()
        user_db_before = environment.get_user_db_hash()
        injection = expected["error_condition"]
        response = environment.get_response(
            ToolCall(
                id=injection["tool_call_id"],
                name=injection["tool_name"],
                arguments=injection["arguments"],
                requestor="assistant",
            )
        )
        db_after = environment.get_db_hash()
        user_db_after = environment.get_user_db_hash()
        rows.append(
            {
                "pair_id": expected["pair_id"],
                "tool_name": injection["tool_name"],
                "tool_error_observed": response.error,
                "agent_database_unchanged": db_before == db_after,
                "user_database_unchanged": user_db_before == user_db_after,
                "tool_response": response.content,
            }
        )

    passed = (
        len(rows) == manifest["paired_task_count"]
        and all(row["tool_error_observed"] for row in rows)
        and all(row["agent_database_unchanged"] for row in rows)
        and all(row["user_database_unchanged"] for row in rows)
    )
    result = {
        "protocol": manifest["protocol"],
        "status": "PASS" if passed else "FAIL",
        "verified_injections": len(rows),
        "all_tool_errors_observed": all(
            row["tool_error_observed"] for row in rows
        ),
        "all_agent_databases_unchanged": all(
            row["agent_database_unchanged"] for row in rows
        ),
        "all_user_databases_unchanged": all(
            row["user_database_unchanged"] for row in rows
        ),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(
        args.tau2_root.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
