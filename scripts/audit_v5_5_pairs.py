#!/usr/bin/env python3
"""Independent, fail-closed V5.5 pair audit."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import v5_5_protocol as protocol
except ModuleNotFoundError:
    from scripts import v5_5_protocol as protocol


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise protocol.V55ProtocolError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise protocol.V55ProtocolError(
                f"{path}:{line_number}: expected object"
            )
        rows.append(value)
    return rows


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def initialize_environment(domain: str, task: Any) -> Any:
    from tau2.registry import registry

    environment = registry.get_env_constructor(domain)()
    initial = task.initial_state
    environment.set_state(
        initialization_data=(
            deepcopy(initial.initialization_data) if initial else None
        ),
        initialization_actions=(
            deepcopy(initial.initialization_actions) if initial else None
        ),
        message_history=(
            deepcopy(initial.message_history)
            if initial and initial.message_history
            else []
        ),
    )
    return environment


def task_map(domains: set[str]) -> dict[str, Any]:
    from tau2.registry import registry

    result: dict[str, Any] = {}
    for domain in domains:
        for task in registry.get_tasks_loader(domain)(None):
            result[f"{domain}:{task.id}"] = task
    return result


def calls_for(task: Any) -> list[dict[str, Any]]:
    actions = (
        task.evaluation_criteria.actions
        if task.evaluation_criteria and task.evaluation_criteria.actions
        else []
    )
    return [
        {
            "id": f"v55_ref_{index:03d}",
            "name": action.name,
            "arguments": deepcopy(action.arguments),
            "requestor": action.requestor,
        }
        for index, action in enumerate(actions)
    ]


def execute(environment: Any, call: dict[str, Any]) -> dict[str, Any]:
    from tau2.data_model.message import ToolCall

    return environment.get_response(
        ToolCall.model_validate(call)
    ).model_dump(mode="json")


def independently_replay(
    pair: dict[str, Any], row: dict[str, Any], task: Any
) -> dict[str, Any]:
    """Recompute error and end-state evidence without trusting producer flags."""
    checks: dict[str, bool] = {}
    calls = calls_for(task)
    index = int(row["action_index"])
    if index >= len(calls):
        return {**pair, "independent_environment_replay_pass": False}
    correct = deepcopy(calls[index])
    injected = deepcopy(correct)
    injected["id"] = f"v55_error_{row['mutation_variant']}"
    injected["arguments"][row["identifier_key"]] = row["mutated_identifier"]
    checks["registered_clean_call"] = pair.get("clean_call") == correct
    checks["registered_injected_call"] = pair.get("injected_call") == injected
    checks["registered_correction_call"] = pair.get("correction_call") == correct

    clean_env = initialize_environment(str(row["domain"]), task)
    clean_results = [execute(clean_env, call) for call in calls]
    checks["clean_reference_path_has_no_errors"] = not any(
        result.get("error") is True for result in clean_results
    )

    recovery_env = initialize_environment(str(row["domain"]), task)
    recovery_results: list[dict[str, Any]] = []
    for action_index, call in enumerate(calls):
        if action_index == index:
            recovery_results.append(execute(recovery_env, injected))
        recovery_results.append(execute(recovery_env, call))
    observed_error = recovery_results[index]
    observed_correction = recovery_results[index + 1]
    checks["injected_call_actually_errors"] = observed_error.get("error") is True
    checks["stored_error_matches_replay"] = (
        pair.get("injected_result", {}).get("error")
        == observed_error.get("error")
        and pair.get("injected_result", {}).get("content")
        == observed_error.get("content")
    )
    checks["correction_actually_succeeds"] = (
        observed_correction.get("error") is False
    )
    checks["stored_correction_matches_replay"] = (
        pair.get("correction_result", {}).get("error")
        == observed_correction.get("error")
        and pair.get("correction_result", {}).get("content")
        == observed_correction.get("content")
    )
    checks["exactly_one_recovery_error"] = (
        sum(result.get("error") is True for result in recovery_results) == 1
    )
    clean_agent_hash = clean_env.get_db_hash()
    clean_user_hash = clean_env.get_user_db_hash()
    recovery_agent_hash = recovery_env.get_db_hash()
    recovery_user_hash = recovery_env.get_user_db_hash()
    checks["end_state_matches_reference"] = (
        clean_agent_hash == recovery_agent_hash
        and clean_user_hash == recovery_user_hash
    )
    checks["stored_hashes_match_replay"] = (
        pair.get("clean_agent_db_hash") == clean_agent_hash
        and pair.get("clean_user_db_hash") == clean_user_hash
        and pair.get("recovery_agent_db_hash") == recovery_agent_hash
        and pair.get("recovery_user_db_hash") == recovery_user_hash
    )
    verified = deepcopy(pair)
    verified["clean_end_state_matches_reference"] = checks[
        "clean_reference_path_has_no_errors"
    ]
    verified["recovery_end_state_matches_reference"] = checks[
        "end_state_matches_reference"
    ]
    verified["independent_environment_replay_checks"] = checks
    verified["independent_environment_replay_pass"] = all(checks.values())
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_tau2(args.tau2_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = read_jsonl(args.pairs)
    registered = {
        row["pair_id"]: row for row in manifest.get("rows", [])
    }
    if len(registered) != protocol.TARGET_PAIRS:
        raise protocol.V55ProtocolError("manifest is not the frozen 48-pair registry")
    observed_ids = [pair.get("pair_id") for pair in pairs]
    unknown = sorted({value for value in observed_ids if value not in registered})
    missing = sorted(set(registered) - set(observed_ids))
    tasks = task_map({str(row["domain"]) for row in manifest["rows"]})
    verified_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if pair_id not in registered:
            invalid = deepcopy(pair)
            invalid["independent_environment_replay_pass"] = False
            verified_pairs.append(invalid)
            continue
        row = registered[pair_id]
        verified_pairs.append(
            independently_replay(pair, row, tasks[str(row["task_identity"])])
        )
    result = protocol.gate(
        verified_pairs,
        [str(value) for value in manifest.get("task_ids", [])],
    )
    result["registry_checks"] = {
        "exact_pair_count": len(pairs) == protocol.TARGET_PAIRS,
        "no_unknown_pair_ids": not unknown,
        "no_missing_pair_ids": not missing,
    }
    result["unknown_pair_ids"] = unknown
    result["missing_pair_ids"] = missing
    result["independent_replay_failures"] = [
        {
            "pair_id": pair.get("pair_id"),
            "failed_checks": sorted(
                key
                for key, value in pair.get(
                    "independent_environment_replay_checks", {}
                ).items()
                if not value
            ),
        }
        for pair in verified_pairs
        if pair.get("independent_environment_replay_pass") is not True
    ]
    if not all(result["registry_checks"].values()):
        result["status"] = "FAIL_CLOSED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS_TRAINING_AUTHORIZED" else 2)


if __name__ == "__main__":
    main()
