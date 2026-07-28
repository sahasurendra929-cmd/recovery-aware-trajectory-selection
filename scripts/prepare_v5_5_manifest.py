#!/usr/bin/env python3
"""CPU-only structural preflight for the frozen V5.5 task universe."""
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


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def executable_candidates(
    tau2_root: Path, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute every reference path and both mutations before registration."""
    configure_tau2(tau2_root)
    from tau2.data_model.message import ToolCall
    from tau2.registry import registry

    tasks: dict[str, Any] = {}
    for domain in {str(row["domain"]) for row in candidates}:
        for task in registry.get_tasks_loader(domain)(None):
            tasks[f"{domain}:{task.id}"] = task

    def initialize(domain: str, task: Any) -> Any:
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

    def execute(environment: Any, call: dict[str, Any]) -> dict[str, Any]:
        return environment.get_response(
            ToolCall.model_validate(call)
        ).model_dump(mode="json")

    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = str(candidate["task_identity"])
        task = tasks[identity]
        actions = (
            task.evaluation_criteria.actions
            if task.evaluation_criteria and task.evaluation_criteria.actions
            else []
        )
        calls = [
            {
                "id": f"v55_ref_{index:03d}",
                "name": action.name,
                "arguments": deepcopy(action.arguments),
                "requestor": action.requestor,
            }
            for index, action in enumerate(actions)
        ]
        reasons: list[str] = []
        clean = initialize(str(candidate["domain"]), task)
        clean_results = [execute(clean, call) for call in calls]
        if any(result.get("error") is True for result in clean_results):
            reasons.append("reference_path_contains_tool_error")
        if not reasons:
            for variant in range(protocol.PAIRS_PER_TASK):
                injected = deepcopy(calls[int(candidate["action_index"])])
                injected["id"] = f"v55_error_{variant}"
                injected["arguments"][candidate["identifier_key"]] = (
                    protocol.mutate_identifier(
                        str(candidate["correct_identifier"]), variant
                    )
                )
                recovery = initialize(str(candidate["domain"]), task)
                results: list[dict[str, Any]] = []
                for index, call in enumerate(calls):
                    if index == int(candidate["action_index"]):
                        results.append(execute(recovery, injected))
                    results.append(execute(recovery, call))
                if sum(result.get("error") is True for result in results) != 1:
                    reasons.append(
                        f"mutation_{variant}_does_not_create_exactly_one_error"
                    )
        if reasons:
            rejected.append(
                {"task_identity": identity, "reasons": sorted(set(reasons))}
            )
        else:
            passed.append(candidate)
    return passed, rejected


def prepare(
    tau2_root: Path,
    split_manifest: Path,
    output: Path,
    *,
    execute_tools: bool = True,
) -> dict[str, Any]:
    split = load(split_manifest)
    candidates: list[dict[str, Any]] = []
    scanned = 0
    for domain in ("airline", "retail"):
        allowed = {
            str(value)
            for value in split["domains"][domain]["inner_train_ids"]
        }
        tasks_path = tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
        tasks = {str(task["id"]): task for task in load(tasks_path)}
        for task_id in sorted(allowed):
            scanned += 1
            task = tasks[task_id]
            actions = (
                (task.get("evaluation_criteria") or {}).get("actions") or []
            )
            eligible = protocol.eligible_reference_actions(domain, actions)
            if not eligible:
                continue
            site = protocol.select_site(f"{domain}:{task_id}", eligible)
            candidates.append(
                {
                    "task_identity": f"{domain}:{task_id}",
                    "domain": domain,
                    "task_id": task_id,
                    "reference_action_count": len(actions),
                    "eligible_site_count": len(eligible),
                    **site,
                }
            )
    structural_candidates = candidates
    execution_rejections: list[dict[str, Any]] = []
    if execute_tools:
        candidates, execution_rejections = executable_candidates(
            tau2_root, candidates
        )
    selected = protocol.choose_tasks(candidates)
    rows: list[dict[str, Any]] = []
    for selected_task in selected:
        for variant in range(protocol.PAIRS_PER_TASK):
            rows.append(
                {
                    **selected_task,
                    "pair_id": (
                        f"{selected_task['task_identity']}:"
                        f"counterfactual:{variant + 1}"
                    ),
                    "mutation_variant": variant,
                    "mutated_identifier": protocol.mutate_identifier(
                        selected_task["correct_identifier"], variant
                    ),
                    "protocol": protocol.PROTOCOL,
                    "status": "REGISTERED",
                    "official_test_used": False,
                }
            )
    payload = {
        "protocol": protocol.PROTOCOL,
        "design_version": protocol.DESIGN_VERSION,
        "selection_policy": (
            "structural reference-action eligibility only; no rollout, "
            "reward, validation, or test outcomes"
        ),
        "source_split_sha256": protocol.sha256(split),
        "scanned_inner_train_tasks": scanned,
        "structurally_eligible_tasks": len(structural_candidates),
        "eligible_tasks": len(candidates),
        "eligible_domain_counts": {
            domain: sum(row["domain"] == domain for row in candidates)
            for domain in ("airline", "retail")
        },
        "tool_execution_preflight_performed": execute_tools,
        "tool_execution_rejections": execution_rejections,
        "registered_tasks": len(selected),
        "registered_pairs": len(rows),
        "task_ids": [row["task_identity"] for row in selected],
        "rows": rows,
        "official_test_used": False,
    }
    if len(rows) != protocol.TARGET_PAIRS:
        raise protocol.V55ProtocolError("registered pair count drift")
    dump(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="diagnostic only; formal manifests must execute tool preflight",
    )
    args = parser.parse_args()
    result = prepare(
        args.tau2_root.resolve(),
        args.split_manifest.resolve(),
        args.output.resolve(),
        execute_tools=not args.structural_only,
    )
    print(
        json.dumps(
            {
                "status": "STRUCTURAL_PREFLIGHT_PASS",
                "eligible_tasks": result["eligible_tasks"],
                "registered_tasks": result["registered_tasks"],
                "registered_pairs": result["registered_pairs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
