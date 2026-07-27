#!/usr/bin/env python3
"""Build a leakage-safe recovery request from one successful clean trajectory.

This script is intentionally environment-agnostic.  The RunPod executor must
provide a normalized clean record whose messages alternate as tau2 emitted
them.  The output stops immediately after the injected call's actual tool
error and contains no clean suffix.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import v5_4_pilot_protocol as protocol
except ModuleNotFoundError:
    from scripts import v5_4_pilot_protocol as protocol


def _sha(value: Any) -> str:
    return protocol.canonical_sha256(value)


def _calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    value = message.get("tool_calls")
    return value if isinstance(value, list) else []


def eligible_indices(messages: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for index, message in enumerate(messages):
        calls = _calls(message)
        if (
            message.get("role") == "assistant"
            and len(calls) == 1
            and protocol.is_read_only_tool(str(calls[0].get("name", "")))
        ):
            result.append(index)
    return result


def build_branch(
    clean: dict[str, Any],
    perturbation: dict[str, Any],
    *,
    recovery_seed: int,
) -> dict[str, Any]:
    if clean.get("eligible_clean_success") is not True:
        raise protocol.PilotProtocolError("clean source is not an eligible success")
    task_identity = str(clean.get("task_identity"))
    if task_identity not in protocol.PILOT_TASK_IDS:
        raise protocol.PilotProtocolError("clean source is outside pilot tasks")
    messages = clean.get("messages")
    if not isinstance(messages, list):
        raise protocol.PilotProtocolError("clean source lacks normalized messages")
    selected = protocol.select_injection_index(
        eligible_indices(messages),
        task_identity=task_identity,
        clean_seed=int(clean["seed"]),
    )
    clean_call = _calls(messages[selected])[0]
    if perturbation.get("clean_tool_name") != clean_call.get("name"):
        raise protocol.PilotProtocolError("perturbation tool does not match clean call")
    injected_call = perturbation.get("injected_call")
    actual_error_result = perturbation.get("actual_error_result")
    if not isinstance(injected_call, dict) or not isinstance(
        actual_error_result, dict
    ):
        raise protocol.PilotProtocolError("branch needs call and actual tool error")
    real_checks = protocol.real_error_checks(perturbation)
    if not all(real_checks.values()):
        failed = sorted(key for key, value in real_checks.items() if not value)
        raise protocol.PilotProtocolError(
            "not a primary real error: " + ", ".join(failed)
        )

    prefix = deepcopy(messages[:selected])
    injected_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [deepcopy(injected_call)],
        "metadata": {
            "v5_4_controlled_error": True,
            "positive_label": False,
        },
    }
    error_message = deepcopy(actual_error_result)
    error_message["metadata"] = {
        **dict(error_message.get("metadata") or {}),
        "v5_4_error_result": True,
        "positive_label": False,
    }
    prompt_messages = [*prefix, injected_message, error_message]
    clean_future = messages[selected:]
    evidence = {
        **deepcopy(perturbation),
        "task_identity": task_identity,
        "clean_seed": clean["seed"],
        "recovery_seed": recovery_seed,
        "selected_message_index": selected,
        "shared_prefix_sha256": _sha(prefix),
        "recovery_prefix_sha256": _sha(prefix),
        "clean_future_sha256": _sha(clean_future),
        "recovery_prompt_sha256": _sha(prompt_messages),
        "error_event_sha256": _sha([injected_message, error_message]),
        "clean_future_present_in_prompt": False,
        "failed_call_supervised": False,
        "error_result_supervised": False,
    }
    return {
        "protocol": protocol.PROTOCOL,
        "task_identity": task_identity,
        "clean_source_sha256": _sha(clean),
        "recovery_seed": recovery_seed,
        "messages": prompt_messages,
        "supervision_starts_after_message_index": len(prompt_messages) - 1,
        "evidence": evidence,
        "official_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--perturbation", type=Path, required=True)
    parser.add_argument("--recovery-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean = json.loads(args.clean.read_text(encoding="utf-8"))
    perturbation = json.loads(args.perturbation.read_text(encoding="utf-8"))
    value = build_branch(clean, perturbation, recovery_seed=args.recovery_seed)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
