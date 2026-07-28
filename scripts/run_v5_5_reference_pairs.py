#!/usr/bin/env python3
"""Execute V5.5 counterfactual pairs against the pinned tau2 environments.

This is deliberately not an LLM rollout.  It executes one registered tau2
reference action path twice: once unchanged, and once with a single failed
read-only identifier call inserted immediately before the registered correct
lookup.  The resulting recovery label is the complete reference-action suffix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import v5_5_protocol as protocol
except ModuleNotFoundError:
    from scripts import v5_5_protocol as protocol


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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


def execute_call(environment: Any, call: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from tau2.data_model.message import ToolCall

    tool_call = ToolCall.model_validate(call)
    result = environment.get_response(tool_call)
    role = "assistant" if tool_call.requestor == "assistant" else "user"
    message = {
        "role": role,
        "content": None,
        "tool_calls": [tool_call.model_dump(mode="json")],
    }
    return message, result.model_dump(mode="json")


def reference_calls(task: Any) -> list[dict[str, Any]]:
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


def run_path(
    domain: str,
    task: Any,
    calls: list[dict[str, Any]],
    *,
    injection_index: int | None = None,
    injected_call: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]]]:
    environment = initialize_environment(domain, task)
    messages: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if injection_index == index:
            if injected_call is None:
                raise protocol.V55ProtocolError("missing injected call")
            message, result = execute_call(environment, injected_call)
            messages.extend((message, result))
            results.append(result)
        message, result = execute_call(environment, call)
        messages.extend((message, result))
        results.append(result)
    return messages, environment, results


def task_context(task: Any) -> dict[str, Any]:
    return {
        "description": (
            task.description.model_dump(mode="json")
            if task.description is not None
            else None
        ),
        "user_scenario": task.user_scenario.model_dump(mode="json"),
        "note": (
            "Task metadata is retained for research reconstruction. It is not "
            "claimed to be a naturally occurring user utterance."
        ),
    }


def build_pair(row: dict[str, Any], task: Any) -> dict[str, Any]:
    domain = str(row["domain"])
    action_index = int(row["action_index"])
    calls = reference_calls(task)
    if action_index >= len(calls):
        raise protocol.V55ProtocolError("registered action index out of range")
    clean_call = deepcopy(calls[action_index])
    if (
        clean_call["name"] != row["tool_name"]
        or clean_call["arguments"].get(row["identifier_key"])
        != row["correct_identifier"]
    ):
        raise protocol.V55ProtocolError("reference action drift")
    injected_call = deepcopy(clean_call)
    injected_call["id"] = f"v55_error_{row['mutation_variant']}"
    injected_call["arguments"][row["identifier_key"]] = row["mutated_identifier"]

    clean_messages, clean_env, clean_results = run_path(domain, task, calls)
    recovery_messages, recovery_env, recovery_results = run_path(
        domain,
        task,
        calls,
        injection_index=action_index,
        injected_call=injected_call,
    )
    clean_selected_offset = 2 * action_index
    recovery_error_offset = 2 * action_index
    error_result = recovery_messages[recovery_error_offset + 1]
    correction_call_message = recovery_messages[recovery_error_offset + 2]
    correction_result = recovery_messages[recovery_error_offset + 3]
    correction_call = correction_call_message["tool_calls"][0]

    if any(result.get("error") is True for result in clean_results):
        raise protocol.V55ProtocolError(
            f"{row['pair_id']}: reference action path contains a tool error"
        )
    if error_result.get("error") is not True:
        raise protocol.V55ProtocolError(
            f"{row['pair_id']}: mutation did not cause an actual tool error"
        )
    if correction_result.get("error") is True:
        raise protocol.V55ProtocolError(
            f"{row['pair_id']}: registered correct lookup failed"
        )
    if sum(result.get("error") is True for result in recovery_results) != 1:
        raise protocol.V55ProtocolError(
            f"{row['pair_id']}: recovery path contains an unregistered tool error"
        )

    clean_prefix = clean_messages[:clean_selected_offset]
    recovery_prefix = recovery_messages[:recovery_error_offset]
    failed_event = recovery_messages[
        recovery_error_offset : recovery_error_offset + 2
    ]
    supervised = recovery_messages[recovery_error_offset + 2 :]
    clean_agent_hash = clean_env.get_db_hash()
    clean_user_hash = clean_env.get_user_db_hash()
    recovery_agent_hash = recovery_env.get_db_hash()
    recovery_user_hash = recovery_env.get_user_db_hash()
    end_state_match = (
        clean_agent_hash == recovery_agent_hash
        and clean_user_hash == recovery_user_hash
    )
    return {
        "protocol": protocol.PROTOCOL,
        "pair_id": row["pair_id"],
        "task_identity": row["task_identity"],
        "domain": domain,
        "task_id": row["task_id"],
        "mutation_variant": row["mutation_variant"],
        "identifier_key": row["identifier_key"],
        "task_context": task_context(task),
        "clean_call": clean_call,
        "injected_call": injected_call,
        "injected_result": error_result,
        "correction_call": correction_call,
        "correction_result": correction_result,
        "clean_prefix": clean_prefix,
        "recovery_prompt": [*recovery_prefix, *failed_event],
        "failed_event": failed_event,
        "supervised_messages": supervised,
        "supervision_starts_after_error": True,
        "clean_future_present_in_recovery_prompt": False,
        "clean_prefix_sha256": protocol.semantic_sha256(clean_prefix),
        "recovery_prefix_sha256": protocol.semantic_sha256(recovery_prefix),
        "error_event_sha256": protocol.semantic_sha256(failed_event),
        "supervised_suffix_sha256": protocol.semantic_sha256(supervised),
        "clean_end_state_matches_reference": True,
        "recovery_end_state_matches_reference": end_state_match,
        "clean_agent_db_hash": clean_agent_hash,
        "clean_user_db_hash": clean_user_hash,
        "recovery_agent_db_hash": recovery_agent_hash,
        "recovery_user_db_hash": recovery_user_hash,
        "reference_actions_are_natural_dialogue": False,
        "independent_environment_replay_pass": False,
        "official_test_used": False,
    }


def load_tasks(manifest: dict[str, Any]) -> dict[str, Any]:
    from tau2.registry import registry

    domains = {row["domain"] for row in manifest["rows"]}
    result: dict[str, Any] = {}
    for domain in domains:
        for task in registry.get_tasks_loader(domain)(None):
            result[f"{domain}:{task.id}"] = task
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_tau2(args.tau2_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol.PROTOCOL:
        raise protocol.V55ProtocolError("manifest protocol mismatch")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    tasks = load_tasks(manifest)
    for index, row in enumerate(manifest["rows"], 1):
        pair = build_pair(row, tasks[row["task_identity"]])
        append_jsonl(args.output, pair)
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(manifest["rows"]),
                    "pair_id": row["pair_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
