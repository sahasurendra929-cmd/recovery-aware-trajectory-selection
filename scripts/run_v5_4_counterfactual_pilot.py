#!/usr/bin/env python3
"""Live tau2 executor for the V5.4 counterfactual recovery pilot.

The frozen V5.4 branch intentionally shipped the protocol, branch builder and
auditor without a host-specific executor.  This runner supplies that missing
layer while keeping the registered task/seed schedule and fail-closed gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from pydantic import TypeAdapter

try:
    import v5_4_pilot_protocol as protocol
    from build_v5_4_branch_manifest import build_branch, eligible_indices
    from run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )
except ModuleNotFoundError:
    from scripts import v5_4_pilot_protocol as protocol
    from scripts.build_v5_4_branch_manifest import build_branch, eligible_indices
    from scripts.run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tau2-root", type=Path, required=True)
    p.add_argument("--protocol-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--teacher-model", default="v5_4_teacher")
    p.add_argument("--teacher-api-base", default="http://127.0.0.1:18000/v1")
    p.add_argument("--user-model", default="v5_4_judge")
    p.add_argument("--user-api-base", default="http://127.0.0.1:18001/v1")
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--smoke-task", default=protocol.PILOT_TASK_IDS[0])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--timeout", type=float, default=900.0)
    return p.parse_args()


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


def load_slots(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected = protocol.slot_registry()
    if rows != expected:
        raise protocol.PilotProtocolError("slot registry differs from frozen protocol")
    return rows


def register_agent() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.registry import registry

    class V54Agent(LLMAgent):
        def _generate_next_message(self, message, state):
            return normalize_tool_only_message(super()._generate_next_message(message, state))

    def factory(tools, domain_policy, **kwargs):
        return V54Agent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))

    if registry.get_agent_factory("v5_4_live_agent") is None:
        registry.register_agent_factory(factory, "v5_4_live_agent")


def select_task(identity: str):
    from tau2.registry import registry

    domain, task_id = identity.split(":", 1)
    tasks = {str(task.id): task for task in registry.get_tasks_loader(domain)(None)}
    if task_id not in tasks:
        raise protocol.PilotProtocolError(f"missing task {identity}")
    return domain, tasks[task_id]


def run_config(args: argparse.Namespace, domain: str, seed: int):
    from tau2.data_model.simulation import TextRunConfig

    teacher = endpoint_args(args.teacher_api_base, "v5-4-local", max_tokens=args.max_tokens, seed=seed, temperature=.2, top_p=.95)
    user = endpoint_args(args.user_api_base, "v5-4-local", max_tokens=args.max_tokens, seed=seed, temperature=.2, top_p=.95)
    return TextRunConfig(
        domain=domain, agent="v5_4_live_agent", user="user_simulator",
        llm_agent=litellm_openai_model(args.teacher_model), llm_args_agent=teacher,
        llm_user=litellm_openai_model(args.user_model), llm_args_user=user,
        num_trials=1, max_steps=args.max_steps, max_errors=10, timeout=args.timeout,
        max_concurrency=1, seed=seed, log_level="INFO", max_retries=1,
        retry_delay=1.0, auto_resume=False, hallucination_retries=0,
        enforce_communication_protocol=False, verbose_logs=True,
    )


def run_one(args: argparse.Namespace, task: Any, domain: str, seed: int, save_dir: Path):
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_single_task

    started = time.monotonic()
    simulation = run_single_task(
        run_config(args, domain, seed), task, seed=seed,
        evaluation_type=EvaluationType.ALL, save_dir=save_dir, verbose_logs=True,
    )
    return simulation, time.monotonic() - started


def reward(simulation: Any) -> float:
    value = getattr(getattr(simulation, "reward_info", None), "reward", 0.0)
    return float(value or 0.0)


def message_dicts(simulation: Any) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in simulation.messages]


def mutate_identifier(call: dict[str, Any]) -> tuple[dict[str, Any], str]:
    mutated = deepcopy(call)
    arguments = mutated.get("arguments")
    if not isinstance(arguments, dict):
        raise protocol.PilotProtocolError("tool call arguments are not an object")
    preferred = [key for key in arguments if re.search(r"(^|_)(id|number)$|reservation|order|user|flight", key, re.I)]
    candidates = preferred + [key for key, value in arguments.items() if isinstance(value, str) and key not in preferred]
    for key in candidates:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            continue
        tail = "0" if value[-1] != "0" else "1"
        arguments[key] = value[:-1] + tail
        mutated["id"] = f"v5_4_injected_{call.get('id') or 'call'}"
        mutated["requestor"] = "assistant"
        return mutated, key
    raise protocol.PilotProtocolError("selected read-only call has no mutable string identifier")


def parse_messages(values: list[dict[str, Any]]):
    from tau2.data_model.message import Message
    return [TypeAdapter(Message).validate_python(value) for value in values]


def recovery_task(original: Any, prompt_without_error: list[dict[str, Any]]):
    from tau2.data_model.tasks import InitialState

    copied = original.model_copy(deep=True)
    prior = original.initial_state
    copied.initial_state = InitialState(
        initialization_data=deepcopy(prior.initialization_data) if prior else None,
        initialization_actions=deepcopy(prior.initialization_actions) if prior else None,
        message_history=parse_messages(prompt_without_error),
    )
    return copied


def injected_tool_result(
    original: Any,
    domain: str,
    prefix: list[dict[str, Any]],
    injected: dict[str, Any],
) -> dict[str, Any]:
    """Execute the injected read-only call against a freshly replayed environment."""
    from tau2.data_model.message import ToolCall
    from tau2.runner import build_environment

    environment = build_environment(domain)
    prior = original.initial_state
    environment.set_state(
        deepcopy(prior.initialization_data) if prior else None,
        deepcopy(prior.initialization_actions) if prior else None,
        parse_messages(prefix),
    )
    call = TypeAdapter(ToolCall).validate_python(injected)
    return environment.get_response(call).model_dump(mode="json")


def recovery_history(
    prefix: list[dict[str, Any]],
    injected: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        *deepcopy(prefix),
        {"role": "assistant", "content": None, "tool_calls": [deepcopy(injected)]},
        deepcopy(result),
    ]


def tool_result_after_injection(messages: list[dict[str, Any]], injected_id: str) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") == "tool" and message.get("id") == injected_id:
            return message
        if message.get("role") == "tool" and message.get("tool_call_id") == injected_id:
            return message
    return None


def execute_task(args: argparse.Namespace, identity: str, task_slots: list[dict[str, Any]], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domain, task = select_task(identity)
    terminal: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    accepted_clean: dict[str, Any] | None = None
    selected_call: dict[str, Any] | None = None
    selected_index: int | None = None
    clean_future_result: dict[str, Any] | None = None

    clean_slots = [row for row in task_slots if row["slot_kind"] == "clean"]
    recovery_slots = [row for row in task_slots if row["slot_kind"] == "recovery"]
    for row in clean_slots:
        if accepted_clean is not None:
            terminal.append({**row, "status": "CLEAN_SUCCESS_EARLY_STOP"})
            continue
        try:
            sim, seconds = run_one(args, task, domain, row["seed"], root / "logs" / row["slot_id"].replace(":", "_"))
            messages = message_dicts(sim)
            eligible = eligible_indices(messages)
            if reward(sim) == 1.0 and eligible:
                selected_index = protocol.select_injection_index(eligible, task_identity=identity, clean_seed=row["seed"])
                selected_call = deepcopy(messages[selected_index]["tool_calls"][0])
                clean_future_result = messages[selected_index + 1] if selected_index + 1 < len(messages) else None
                accepted_clean = {"task_identity": identity, "seed": row["seed"], "eligible_clean_success": True, "messages": messages}
                terminal.append({**row, "status": "CLEAN_ELIGIBLE", "gpu_seconds": seconds})
                write_json(root / "clean" / f"{identity.replace(':', '_')}.json", accepted_clean)
            else:
                terminal.append({**row, "status": "CLEAN_INELIGIBLE", "gpu_seconds": seconds})
        except Exception as error:
            terminal.append({**row, "status": "EXECUTION_ERROR", "gpu_seconds": 0.0, "error": repr(error)})

    if accepted_clean is None or selected_call is None or selected_index is None:
        terminal.extend({**row, "status": "NO_CLEAN_PREFIX"} for row in recovery_slots)
        return terminal, pairs

    injected, mutated_key = mutate_identifier(selected_call)
    prefix = deepcopy(accepted_clean["messages"][:selected_index])
    try:
        injected_result = injected_tool_result(task, domain, prefix, injected)
        branch_messages = recovery_history(prefix, injected, injected_result)
    except Exception as error:
        terminal.extend(
            {**row, "status": "EXECUTION_ERROR", "gpu_seconds": 0.0, "error": repr(error)}
            for row in recovery_slots
        )
        return terminal, pairs
    for row in recovery_slots:
        if len(pairs) >= protocol.MAX_PAIRS_PER_TASK:
            terminal.append({**row, "status": "PAIR_CAP_REACHED"})
            continue
        try:
            branch_task = recovery_task(task, branch_messages)
            sim, seconds = run_one(args, branch_task, domain, row["seed"], root / "logs" / row["slot_id"].replace(":", "_"))
            messages = message_dicts(sim)
            actual_error = tool_result_after_injection(messages, str(injected.get("id", "")))
            consequential = actual_error is not None and actual_error != clean_future_result
            perturbation = {
                "taxonomy": "wrong_identifier_same_type", "schema_valid": actual_error is not None,
                "clean_action_correct": True, "injected_action_task_incorrect": True,
                "consequential": consequential, "clean_tool_name": selected_call["name"],
                "injected_error_count": 1, "injected_call": injected,
                "actual_error_result": actual_error, "mutated_argument": mutated_key,
            }
            if actual_error is None or not consequential:
                terminal.append({**row, "status": "NON_CONSEQUENTIAL_PERTURBATION", "gpu_seconds": seconds})
                continue
            branch = build_branch(accepted_clean, perturbation, recovery_seed=row["seed"])
            eligible_pair = reward(sim) == 1.0
            pair = {
                **branch["evidence"], "slot_id": row["slot_id"], "eligible": eligible_pair,
                "recovery_reward": reward(sim), "difficulty_stratum": "PREDECLARED_PILOT",
                "official_test_used": False,
            }
            terminal.append({**row, "status": "PAIR_ELIGIBLE" if eligible_pair else "RECOVERY_INELIGIBLE", "gpu_seconds": seconds})
            if eligible_pair:
                pairs.append(pair)
        except Exception as error:
            terminal.append({**row, "status": "EXECUTION_ERROR", "gpu_seconds": 0.0, "error": repr(error)})
    return terminal, pairs


def main() -> None:
    args = parse_args()
    configure_tau2(args.tau2_root)
    register_agent()
    os.environ.setdefault("OPENAI_API_KEY", "v5-4-local")
    judge_args = endpoint_args(args.user_api_base, "v5-4-local", max_tokens=args.max_tokens, seed=protocol.BASE_SEED)
    patch_local_nl_judge(litellm_openai_model(args.user_model), judge_args)
    slots = load_slots(args.protocol_root / "slot_registry.jsonl")
    identities = [args.smoke_task] if args.mode == "smoke" else list(protocol.PILOT_TASK_IDS)
    root = args.output_root / args.mode
    terminal_path = root / "slot_terminal.jsonl"
    pair_path = root / "eligible_pairs.jsonl"
    all_terminal = (
        [json.loads(line) for line in terminal_path.read_text(encoding="utf-8").splitlines() if line]
        if terminal_path.exists() else []
    )
    all_pairs = (
        [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines() if line]
        if pair_path.exists() else []
    )
    existing_ids = [str(row.get("slot_id")) for row in all_terminal]
    if len(existing_ids) != len(set(existing_ids)):
        raise protocol.PilotProtocolError("resume ledger contains duplicate slot ids")
    for identity in identities:
        task_slots = [row for row in slots if row["task_identity"] == identity]
        completed = [row for row in all_terminal if row.get("task_identity") == identity]
        if completed:
            if len(completed) != protocol.SLOTS_PER_TASK:
                raise protocol.PilotProtocolError(f"partial task receipt requires recovery: {identity}")
            continue
        terminal, pairs = execute_task(args, identity, task_slots, root)
        all_terminal.extend(terminal)
        all_pairs.extend(pairs)
        for row in terminal:
            append_jsonl(terminal_path, row)
        for pair in pairs:
            append_jsonl(pair_path, pair)
    smoke_pass = args.mode != "smoke" or (
        len(all_terminal) == protocol.SLOTS_PER_TASK
        and any(row["status"] == "PAIR_ELIGIBLE" for row in all_terminal)
        and all(row["status"] in protocol.TERMINAL_STATUSES for row in all_terminal)
    )
    receipt = {
        "protocol": protocol.PROTOCOL, "mode": args.mode,
        "status": "SMOKE_PASS" if smoke_pass and args.mode == "smoke" else ("PASS" if smoke_pass else "SMOKE_FAIL_NO_RUN"),
        "terminal_slots": len(all_terminal), "eligible_pairs": len(all_pairs),
        "official_test_used": False, "training_started": False,
    }
    write_json(root / "run_receipt.json", receipt)
    if args.mode == "full":
        from audit_v5_4_pilot import cost_report, decision_markdown, difficulty_report

        decision = protocol.pilot_decision(all_terminal, all_pairs)
        final = root / "final"
        write_json(final / "audit_report.json", decision)
        write_json(final / "cost_report.json", cost_report(all_terminal, all_pairs))
        write_json(final / "difficulty_coverage.json", difficulty_report(all_pairs))
        final.mkdir(parents=True, exist_ok=True)
        (final / "GO_NO_GO_DECISION.md").write_text(
            decision_markdown(decision), encoding="utf-8"
        )
    print(json.dumps(receipt, sort_keys=True))
    if not smoke_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
