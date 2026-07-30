#!/usr/bin/env python3
"""Fail-closed, model-free contracts for V6.10 reference execution.

This module deliberately imports neither tau2 nor any model/CUDA package.
The runtime preflight supplies four adapters:

``environment_factory()``
    Create a fresh environment at the frozen task initial state.
``execute_call(environment, call)``
    Execute one tool call and return the assistant call message and tool result.
``state_hashes(environment)``
    Return every state hash whose equality defines state preservation.
``environment_evaluator(messages)``
    Run tau2's deterministic ``EvaluationType.ENV`` evaluator and return
    ``{"reward": ..., ...}``.

Reference actions are identified only by
``(task_identity, reference_action_index)``.  Value matching is never used,
because valid tau2 traces contain repeated semantically identical calls.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


DESIGN_PROTOCOL = "v6_10_pipeline_closure_v1"
RECEIPT_PROTOCOL = "v6_reference_execution_preflight_v1"
ARTIFACT_TYPE = "v6_reference_execution_preflight_receipt"
CONTRACT_VERSION = "1.0"
ENVIRONMENT_REWARD_REQUIRED = 1.0
ENVIRONMENT_REWARD_SCOPE = "TAU2_EVALUATION_TYPE_ENV_STRICT_REPLAY"
FULL_OFFICIAL_REWARD_STATUS = "DEFERRED_TO_DYNAMIC_JUDGE"
TERMINAL_TASK_STATUSES = frozenset({"PASS", "REJECTED"})
VOLATILE_FIELDS = frozenset(
    {
        "timestamp",
        "turn_idx",
        "cost",
        "usage",
        "generation_time_seconds",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

EnvironmentFactory = Callable[[], Any]
ExecuteCall = Callable[
    [Any, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]
]
StateHashes = Callable[[Any], Mapping[str, Any]]
EnvironmentEvaluator = Callable[
    [Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


class ReferenceContractError(RuntimeError):
    """The frozen reference-execution contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(value: Any) -> Any:
    """Remove transport metadata while preserving experimental content."""
    if isinstance(value, list):
        return [semantic(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): semantic(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in VOLATILE_FIELDS
        }
    return value


def semantic_sha256(value: Any) -> str:
    return sha256(semantic(value))


def _action_mapping(action: Any) -> dict[str, Any]:
    if isinstance(action, Mapping):
        value = deepcopy(dict(action))
    elif callable(getattr(action, "model_dump", None)):
        value = deepcopy(action.model_dump(mode="json"))
    else:
        value = {
            key: deepcopy(getattr(action, key))
            for key in ("action_id", "requestor", "name", "arguments")
            if hasattr(action, key)
        }
    name = value.get("name")
    arguments = value.get("arguments")
    requestor = value.get("requestor", "assistant")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(arguments, Mapping)
        or not isinstance(requestor, str)
        or not requestor
    ):
        raise ReferenceContractError(
            f"malformed reference action: {canonical(value)}"
        )
    return {
        "source_action_id": (
            str(value["action_id"]) if value.get("action_id") is not None else None
        ),
        "requestor": requestor,
        "name": name,
        "arguments": deepcopy(dict(arguments)),
    }


def call_semantics(call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requestor": str(call.get("requestor", "assistant")),
        "name": str(call.get("name", "")),
        "arguments": deepcopy(call.get("arguments")),
    }


def build_reference_slots(
    task_identity: str,
    actions: Sequence[Any],
) -> list[dict[str, Any]]:
    """Assign exact, occurrence-preserving identity to every reference action."""
    if not isinstance(task_identity, str) or ":" not in task_identity:
        raise ReferenceContractError("task identity must have domain:id form")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ReferenceContractError(f"{task_identity}: actions are not a sequence")
    if not actions:
        raise ReferenceContractError(f"{task_identity}: reference actions are empty")

    slots: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        parsed = _action_mapping(action)
        semantics = call_semantics(parsed)
        slots.append(
            {
                "task_identity": task_identity,
                "reference_action_index": index,
                "reference_slot_id": f"{task_identity}:reference:{index:03d}",
                "source_action_id": parsed["source_action_id"],
                "call": semantics,
                "call_semantics_sha256": sha256(semantics),
            }
        )

    groups: dict[str, list[int]] = {}
    for slot in slots:
        groups.setdefault(slot["call_semantics_sha256"], []).append(
            slot["reference_action_index"]
        )
    for slot in slots:
        indices = groups[slot["call_semantics_sha256"]]
        slot["semantic_occurrence_ordinal"] = indices.index(
            slot["reference_action_index"]
        )
        slot["semantic_occurrence_count"] = len(indices)
        slot["semantically_identical_reference_indices"] = list(indices)
        slot["reference_slot_sha256"] = sha256(
            {
                key: value
                for key, value in slot.items()
                if key != "reference_slot_sha256"
            }
        )
    return slots


def validate_reference_index(
    slots: Sequence[Mapping[str, Any]],
    index: Any,
) -> int:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(slots)
    ):
        raise ReferenceContractError(f"reference action index out of range: {index!r}")
    if slots[index].get("reference_action_index") != index:
        raise ReferenceContractError(
            "reference slot ordering/index identity drift: "
            f"position={index} slot={slots[index].get('reference_action_index')!r}"
        )
    return index


def structurally_candidate_indices(
    slots: Sequence[Mapping[str, Any]],
    allowed_identifier_keys: Iterable[str],
) -> list[int]:
    """Return exact slots exposing at least one allowlisted string identifier."""
    allowed = set(allowed_identifier_keys)
    result: list[int] = []
    for position, slot in enumerate(slots):
        validate_reference_index(slots, position)
        arguments = (slot.get("call") or {}).get("arguments")
        if not isinstance(arguments, Mapping):
            raise ReferenceContractError("reference slot lacks argument mapping")
        if any(
            key in allowed
            and isinstance(arguments.get(key), str)
            and len(arguments[key]) >= 2
            for key in arguments
        ):
            result.append(position)
    return result


def _state_snapshot(state_hashes: StateHashes, environment: Any) -> dict[str, Any]:
    value = state_hashes(environment)
    if not isinstance(value, Mapping) or not value:
        raise ReferenceContractError("state adapter returned no state hashes")
    return deepcopy(dict(value))


def _call_for_slot(
    slot: Mapping[str, Any],
    *,
    call_id_prefix: str,
) -> dict[str, Any]:
    index = int(slot["reference_action_index"])
    core = slot.get("call")
    if not isinstance(core, Mapping):
        raise ReferenceContractError("reference slot lacks a call")
    return {
        "id": f"{call_id_prefix}-{index:03d}",
        **deepcopy(dict(core)),
    }


def _execute_slots_on_environment(
    environment: Any,
    *,
    slots: Sequence[Mapping[str, Any]],
    ordered_indices: Sequence[int],
    execute_call: ExecuteCall,
    state_hashes: StateHashes,
    call_id_prefix: str,
) -> dict[str, Any]:
    if len(ordered_indices) != len(set(ordered_indices)):
        raise ReferenceContractError("an action index appears twice in one plan")
    initial_state = _state_snapshot(state_hashes, environment)
    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for ordinal, raw_index in enumerate(ordered_indices):
        index = validate_reference_index(slots, raw_index)
        slot = slots[index]
        before = _state_snapshot(state_hashes, environment)
        call = _call_for_slot(
            slot,
            call_id_prefix=f"{call_id_prefix}-{ordinal:03d}",
        )
        try:
            assistant_message, tool_result = execute_call(environment, call)
        except Exception as error:
            raise ReferenceContractError(
                f"{slot['reference_slot_id']}: tool execution raised "
                f"{type(error).__name__}: {error}"
            ) from error
        if not isinstance(assistant_message, Mapping) or not isinstance(
            tool_result, Mapping
        ):
            raise ReferenceContractError(
                f"{slot['reference_slot_id']}: malformed execution adapter output"
            )
        error_flag = tool_result.get("error")
        if not isinstance(error_flag, bool):
            raise ReferenceContractError(
                f"{slot['reference_slot_id']}: result lacks boolean error"
            )
        after = _state_snapshot(state_hashes, environment)
        state_preserving = before == after
        if error_flag and not state_preserving:
            raise ReferenceContractError(
                f"{slot['reference_slot_id']}: failed action changed state"
            )
        assistant_value = deepcopy(dict(assistant_message))
        result_value = deepcopy(dict(tool_result))
        messages.extend((assistant_value, result_value))
        events.append(
            {
                "ordinal": ordinal,
                "task_identity": slot["task_identity"],
                "reference_action_index": index,
                "reference_slot_id": slot["reference_slot_id"],
                "reference_slot_sha256": slot["reference_slot_sha256"],
                "call_semantics_sha256": slot["call_semantics_sha256"],
                "assistant_message": assistant_value,
                "assistant_message_semantic_sha256": semantic_sha256(
                    assistant_value
                ),
                "tool_result_message": result_value,
                "tool_result_semantic_sha256": semantic_sha256(result_value),
                "error": error_flag,
                "state_before": before,
                "state_after": after,
                "state_preserving": state_preserving,
            }
        )
    final_state = _state_snapshot(state_hashes, environment)
    return {
        "ordered_reference_indices": list(ordered_indices),
        "ordered_reference_indices_sha256": sha256(list(ordered_indices)),
        "initial_state": initial_state,
        "events": events,
        "messages": messages,
        "messages_semantic_sha256": semantic_sha256(messages),
        "final_state": final_state,
        "final_state_sha256": sha256(final_state),
    }


def execute_slot_plan(
    *,
    slots: Sequence[Mapping[str, Any]],
    ordered_indices: Sequence[int],
    environment_factory: EnvironmentFactory,
    execute_call: ExecuteCall,
    state_hashes: StateHashes,
    call_id_prefix: str,
) -> dict[str, Any]:
    try:
        environment = environment_factory()
    except Exception as error:
        raise ReferenceContractError(
            f"fresh environment construction failed: {type(error).__name__}: {error}"
        ) from error
    return _execute_slots_on_environment(
        environment,
        slots=slots,
        ordered_indices=ordered_indices,
        execute_call=execute_call,
        state_hashes=state_hashes,
        call_id_prefix=call_id_prefix,
    )


def _execution_signature(execution: Mapping[str, Any]) -> dict[str, Any]:
    events = execution.get("events")
    if not isinstance(events, list):
        raise ReferenceContractError("execution has no event list")
    return {
        "ordered_reference_indices": execution.get("ordered_reference_indices"),
        "initial_state": execution.get("initial_state"),
        "final_state": execution.get("final_state"),
        "events": [
            {
                "reference_action_index": event.get("reference_action_index"),
                "call_semantics_sha256": event.get("call_semantics_sha256"),
                "tool_result_semantic_sha256": event.get(
                    "tool_result_semantic_sha256"
                ),
                "error": event.get("error"),
                "state_before": event.get("state_before"),
                "state_after": event.get("state_after"),
            }
            for event in events
        ],
    }


def _evaluate_environment(
    *,
    prefix: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    rendered_completion: Mapping[str, Any],
    environment_evaluator: EnvironmentEvaluator,
) -> dict[str, Any]:
    execution_messages = execution.get("messages")
    if not isinstance(execution_messages, list):
        raise ReferenceContractError(
            "execution lacks messages for environment reward"
        )
    messages = [
        *deepcopy([dict(value) for value in prefix]),
        *deepcopy([dict(value) for value in execution_messages]),
        deepcopy(dict(rendered_completion)),
    ]
    try:
        result = environment_evaluator(messages)
    except Exception as error:
        raise ReferenceContractError(
            f"environment evaluator raised {type(error).__name__}: {error}"
        ) from error
    if not isinstance(result, Mapping):
        raise ReferenceContractError("environment evaluator returned no mapping")
    reward = result.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ReferenceContractError(
            "environment evaluator returned nonnumeric reward"
        )
    reward = float(reward)
    if not math.isfinite(reward):
        raise ReferenceContractError(
            "environment evaluator returned nonfinite reward"
        )
    return {
        "reward": reward,
        "reward_is_required_one": reward == ENVIRONMENT_REWARD_REQUIRED,
        "reward_info": semantic(result.get("reward_info")),
        "messages_semantic_sha256": semantic_sha256(messages),
    }


def derive_reference_outcome_contract(
    *,
    task_identity: str,
    slots: Sequence[Mapping[str, Any]],
    prefix: Sequence[Mapping[str, Any]],
    rendered_completion: Mapping[str, Any],
    environment_factory: EnvironmentFactory,
    execute_call: ExecuteCall,
    state_hashes: StateHashes,
    environment_evaluator: EnvironmentEvaluator,
) -> dict[str, Any]:
    """Execute raw twice, remove exact expected errors, and replay sanitized."""
    canonical_indices = list(range(len(slots)))
    digest = sha256(task_identity)[:12]
    raw = execute_slot_plan(
        slots=slots,
        ordered_indices=canonical_indices,
        environment_factory=environment_factory,
        execute_call=execute_call,
        state_hashes=state_hashes,
        call_id_prefix=f"v610-raw-{digest}",
    )
    raw_repeat = execute_slot_plan(
        slots=slots,
        ordered_indices=canonical_indices,
        environment_factory=environment_factory,
        execute_call=execute_call,
        state_hashes=state_hashes,
        call_id_prefix=f"v610-raw-{digest}",
    )
    if _execution_signature(raw) != _execution_signature(raw_repeat):
        raise ReferenceContractError(
            f"{task_identity}: raw reference execution is nondeterministic"
        )

    expected_errors = [
        event["reference_action_index"]
        for event in raw["events"]
        if event["error"] is True
    ]
    expected_set = set(expected_errors)
    sanitized_indices = [
        index for index in canonical_indices if index not in expected_set
    ]
    if not sanitized_indices:
        raise ReferenceContractError(
            f"{task_identity}: sanitized reference plan is empty"
        )
    sanitized = execute_slot_plan(
        slots=slots,
        ordered_indices=sanitized_indices,
        environment_factory=environment_factory,
        execute_call=execute_call,
        state_hashes=state_hashes,
        call_id_prefix=f"v610-sanitized-{digest}",
    )
    sanitized_errors = [
        event["reference_action_index"]
        for event in sanitized["events"]
        if event["error"] is True
    ]

    raw_reward = _evaluate_environment(
        prefix=prefix,
        execution=raw,
        rendered_completion=rendered_completion,
        environment_evaluator=environment_evaluator,
    )
    raw_repeat_reward = _evaluate_environment(
        prefix=prefix,
        execution=raw_repeat,
        rendered_completion=rendered_completion,
        environment_evaluator=environment_evaluator,
    )
    sanitized_reward = _evaluate_environment(
        prefix=prefix,
        execution=sanitized,
        rendered_completion=rendered_completion,
        environment_evaluator=environment_evaluator,
    )
    final_equivalent = sanitized["final_state"] == raw["final_state"]
    outcome_pass = (
        raw_reward["reward_is_required_one"] is True
        and raw_repeat_reward["reward_is_required_one"] is True
        and sanitized_reward["reward_is_required_one"] is True
        and not sanitized_errors
        and final_equivalent
    )
    expected_error_rows = [
        {
            "reference_action_index": event["reference_action_index"],
            "reference_slot_id": event["reference_slot_id"],
            "call_semantics_sha256": event["call_semantics_sha256"],
            "tool_result_semantic_sha256": event[
                "tool_result_semantic_sha256"
            ],
            "state_preserving": event["state_preserving"],
        }
        for event in raw["events"]
        if event["error"] is True
    ]
    payload = {
        "protocol": RECEIPT_PROTOCOL,
        "design_protocol": DESIGN_PROTOCOL,
        "environment_reward_scope": ENVIRONMENT_REWARD_SCOPE,
        "task_identity": task_identity,
        "status": "PASS" if outcome_pass else "REJECTED",
        "reference_slots_sha256": sha256(
            [slot["reference_slot_sha256"] for slot in slots]
        ),
        "canonical_reference_indices": canonical_indices,
        "raw_execution_repeat_count": 2,
        "raw_execution_reproducible": True,
        "expected_error_indices": expected_errors,
        "expected_error_set_sha256": sha256(expected_errors),
        "expected_errors": expected_error_rows,
        "all_expected_errors_state_preserving": all(
            row["state_preserving"] for row in expected_error_rows
        ),
        "sanitized_successful_reference_indices": sanitized_indices,
        "sanitized_successful_plan_sha256": sha256(sanitized_indices),
        "sanitized_tool_error_indices": sanitized_errors,
        "sanitized_plan_all_actions_succeeded": not sanitized_errors,
        "sanitized_final_state_equivalent": final_equivalent,
        "raw_reference_environment_reward": raw_reward["reward"],
        "raw_reference_repeat_environment_reward": raw_repeat_reward["reward"],
        "sanitized_environment_reward": sanitized_reward["reward"],
        "raw_reference_environment_reward_info": raw_reward["reward_info"],
        "sanitized_environment_reward_info": sanitized_reward["reward_info"],
        "full_official_reward_status": FULL_OFFICIAL_REWARD_STATUS,
        "raw_reference_messages_semantic_sha256": raw_reward[
            "messages_semantic_sha256"
        ],
        "sanitized_messages_semantic_sha256": sanitized_reward[
            "messages_semantic_sha256"
        ],
        "raw_reference_execution": raw,
        "sanitized_reference_execution": sanitized,
    }
    payload["reference_outcome_contract_sha256"] = sha256(payload)
    return payload


def _forced_order(
    outcome_contract: Mapping[str, Any],
    forced_index: int,
) -> list[int]:
    sanitized = outcome_contract.get("sanitized_successful_reference_indices")
    if not isinstance(sanitized, list) or forced_index not in sanitized:
        raise ReferenceContractError(
            f"reference action {forced_index} is not in sanitized plan"
        )
    return [forced_index, *[index for index in sanitized if index != forced_index]]


def screen_forced_first_slots(
    *,
    slots: Sequence[Mapping[str, Any]],
    outcome_contract: Mapping[str, Any],
    candidate_indices: Sequence[int],
    prefix: Sequence[Mapping[str, Any]],
    rendered_completion: Mapping[str, Any],
    environment_factory: EnvironmentFactory,
    execute_call: ExecuteCall,
    state_hashes: StateHashes,
    environment_evaluator: EnvironmentEvaluator,
) -> dict[str, Any]:
    """Classify exact candidate indices without consulting a candidate registry."""
    if len(candidate_indices) != len(set(candidate_indices)):
        raise ReferenceContractError("candidate reference indices are duplicated")
    if outcome_contract.get("status") != "PASS":
        return {
            "status": "SKIPPED_REFERENCE_PLAN_REJECTED",
            "candidate_reference_indices": list(candidate_indices),
            "eligible_forced_first_reference_indices": [],
            "ineligible_forced_first_reference_indices": list(candidate_indices),
            "rows": [],
        }

    canonical_final = outcome_contract.get("raw_reference_execution", {}).get(
        "final_state"
    )
    expected_errors = set(outcome_contract.get("expected_error_indices") or [])
    rows: list[dict[str, Any]] = []
    for raw_index in candidate_indices:
        index = validate_reference_index(slots, raw_index)
        slot = slots[index]
        if index in expected_errors:
            rows.append(
                {
                    "reference_action_index": index,
                    "reference_slot_id": slot["reference_slot_id"],
                    "call_semantics_sha256": slot["call_semantics_sha256"],
                    "eligible": False,
                    "reason_code": "EXPECTED_ERROR_REFERENCE_ACTION",
                    "forced_first_environment_reward": None,
                }
            )
            continue

        order = _forced_order(outcome_contract, index)
        execution = execute_slot_plan(
            slots=slots,
            ordered_indices=order,
            environment_factory=environment_factory,
            execute_call=execute_call,
            state_hashes=state_hashes,
            call_id_prefix=f"v610-forced-{sha256(slot['reference_slot_id'])[:12]}",
        )
        failing = [
            event["reference_action_index"]
            for event in execution["events"]
            if event["error"] is True
        ]
        environment_reward = _evaluate_environment(
            prefix=prefix,
            execution=execution,
            rendered_completion=rendered_completion,
            environment_evaluator=environment_evaluator,
        )
        final_equivalent = execution["final_state"] == canonical_final
        eligible = (
            not failing
            and final_equivalent
            and environment_reward["reward_is_required_one"] is True
        )
        if failing:
            reason = "FORCED_FIRST_PLAN_TOOL_ERROR"
        elif not final_equivalent:
            reason = "FORCED_FIRST_FINAL_STATE_MISMATCH"
        elif environment_reward["reward_is_required_one"] is not True:
            reason = "FORCED_FIRST_ENVIRONMENT_REWARD_NOT_ONE"
        else:
            reason = "ELIGIBLE"
        rows.append(
            {
                "reference_action_index": index,
                "reference_slot_id": slot["reference_slot_id"],
                "call_semantics_sha256": slot["call_semantics_sha256"],
                "eligible": eligible,
                "reason_code": reason,
                "ordered_reference_indices": order,
                "failing_reference_indices": failing,
                "final_state_equivalent": final_equivalent,
                "forced_first_environment_reward": environment_reward["reward"],
                "messages_semantic_sha256": environment_reward[
                    "messages_semantic_sha256"
                ],
                "execution_sha256": sha256(execution),
            }
        )
    eligible_indices = [
        row["reference_action_index"] for row in rows if row["eligible"] is True
    ]
    ineligible_indices = [
        row["reference_action_index"] for row in rows if row["eligible"] is False
    ]
    return {
        "status": "PASS",
        "candidate_reference_indices": list(candidate_indices),
        "candidate_reference_indices_sha256": sha256(list(candidate_indices)),
        "eligible_forced_first_reference_indices": eligible_indices,
        "eligible_forced_first_reference_indices_sha256": sha256(
            eligible_indices
        ),
        "ineligible_forced_first_reference_indices": ineligible_indices,
        "ineligible_count": len(ineligible_indices),
        "rows": rows,
    }


_PROTECTED_LITERAL_RE = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})"
    r"|(?:[$#]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b[A-Z][A-Z0-9_-]*\d[A-Z0-9_-]*\b)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\b)"
)
_FORBIDDEN_RENDERED_PATTERNS = (
    re.compile(r"directly:\s+that\b", re.IGNORECASE),
    re.compile(r"\b(?:Agent|agent|should)\b"),
    re.compile(r"Check that", re.IGNORECASE),
)


def build_renderer_lint_inputs(
    *,
    task_identity: str,
    communicate_info: Sequence[Any] | None,
    nl_assertions: Sequence[Any] | None,
    fallback: str,
) -> dict[str, Any]:
    communicate = [str(value) for value in (communicate_info or [])]
    assertions = [str(value) for value in (nl_assertions or [])]
    if any(not value.strip() for value in communicate + assertions):
        raise ReferenceContractError(
            f"{task_identity}: renderer source contains an empty item"
        )
    sources = [*communicate, *assertions]
    protected = sorted(
        {
            match.group(0)
            for source in sources
            for match in _PROTECTED_LITERAL_RE.finditer(source)
        }
    )
    return {
        "task_identity": task_identity,
        "communicate_info": communicate,
        "nl_assertions": assertions,
        "fallback": fallback,
        "protected_literals": protected,
        "source_sha256": sha256(
            {
                "communicate_info": communicate,
                "nl_assertions": assertions,
                "fallback": fallback,
            }
        ),
    }


def lint_rendered_completion(
    lint_inputs: Mapping[str, Any],
    rendered_message: Mapping[str, Any],
) -> dict[str, Any]:
    content = rendered_message.get("content")
    checks = {
        "assistant_role": rendered_message.get("role") == "assistant",
        "no_tool_calls": not rendered_message.get("tool_calls"),
        "nonempty_content": isinstance(content, str) and bool(content.strip()),
        "communicate_info_preserved": False,
        "protected_literals_preserved": False,
        "no_meta_level_or_complementizer_artifacts": False,
    }
    if isinstance(content, str):
        checks["communicate_info_preserved"] = all(
            str(value) in content
            for value in (lint_inputs.get("communicate_info") or [])
        )
        checks["protected_literals_preserved"] = all(
            str(value) in content
            for value in (lint_inputs.get("protected_literals") or [])
        )
        checks["no_meta_level_or_complementizer_artifacts"] = not any(
            pattern.search(content) for pattern in _FORBIDDEN_RENDERED_PATTERNS
        )
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "REJECTED",
        "checks": checks,
        "rendered_message": semantic(rendered_message),
        "rendered_message_semantic_sha256": semantic_sha256(rendered_message),
    }


def preflight_task(
    *,
    task_identity: str,
    actions: Sequence[Any],
    allowed_identifier_keys: Iterable[str],
    prefix: Sequence[Mapping[str, Any]],
    environment_factory: EnvironmentFactory,
    execute_call: ExecuteCall,
    state_hashes: StateHashes,
    environment_evaluator: EnvironmentEvaluator,
    renderer_lint_inputs: Mapping[str, Any],
    rendered_message: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all registry-independent reference gates for one task."""
    slots = build_reference_slots(task_identity, actions)
    renderer = lint_rendered_completion(renderer_lint_inputs, rendered_message)
    outcome = derive_reference_outcome_contract(
        task_identity=task_identity,
        slots=slots,
        prefix=prefix,
        rendered_completion=rendered_message,
        environment_factory=environment_factory,
        execute_call=execute_call,
        state_hashes=state_hashes,
        environment_evaluator=environment_evaluator,
    )
    candidate_indices = structurally_candidate_indices(
        slots, allowed_identifier_keys
    )
    screen = screen_forced_first_slots(
        slots=slots,
        outcome_contract=outcome,
        candidate_indices=candidate_indices,
        prefix=prefix,
        rendered_completion=rendered_message,
        environment_factory=environment_factory,
        execute_call=execute_call,
        state_hashes=state_hashes,
        environment_evaluator=environment_evaluator,
    )
    eligible = screen["eligible_forced_first_reference_indices"]
    eligible_semantics = {
        slots[index]["call_semantics_sha256"] for index in eligible
    }
    reasons: list[str] = []
    if renderer["status"] != "PASS":
        reasons.append("COMPLETION_RENDERER_LINT_FAILED")
    if outcome["status"] != "PASS":
        reasons.append("RAW_OR_SANITIZED_REFERENCE_REJECTED")
    if len(eligible_semantics) < 2:
        reasons.append("FEWER_THAN_TWO_DISTINCT_ELIGIBLE_FORCED_FIRST_CALLS")
    passed = not reasons
    duplicate_groups = sorted(
        {
            tuple(slot["semantically_identical_reference_indices"])
            for slot in slots
            if slot["semantic_occurrence_count"] > 1
        }
    )
    row = {
        "protocol": RECEIPT_PROTOCOL,
        "design_protocol": DESIGN_PROTOCOL,
        "environment_reward_scope": ENVIRONMENT_REWARD_SCOPE,
        "task_identity": task_identity,
        "status": "PASS" if passed else "REJECTED",
        "reason_codes": reasons or ["PASS"],
        "reference_action_count": len(slots),
        "reference_slots": slots,
        "reference_slots_sha256": sha256(
            [slot["reference_slot_sha256"] for slot in slots]
        ),
        "duplicate_semantic_reference_groups": [
            list(group) for group in duplicate_groups
        ],
        "evaluation_prefix": semantic(prefix),
        "evaluation_prefix_semantic_sha256": semantic_sha256(prefix),
        "renderer_lint": renderer,
        "reference_outcome_contract": outcome,
        "expected_error_indices": outcome["expected_error_indices"],
        "expected_error_set_sha256": outcome["expected_error_set_sha256"],
        "sanitized_successful_reference_indices": outcome[
            "sanitized_successful_reference_indices"
        ],
        "sanitized_successful_plan_sha256": outcome[
            "sanitized_successful_plan_sha256"
        ],
        # Backward-readable alias. New consumers should use the longer,
        # scientifically explicit field above.
        "sanitized_reference_indices": outcome[
            "sanitized_successful_reference_indices"
        ],
        "eligible_forced_first_reference_indices": eligible,
        "eligible_forced_first_reference_indices_sha256": sha256(eligible),
        "ineligible_forced_first_reference_indices": screen[
            "ineligible_forced_first_reference_indices"
        ],
        "forced_first_slot_screen": screen,
        "raw_reference_environment_reward": outcome[
            "raw_reference_environment_reward"
        ],
        "raw_reference_repeat_environment_reward": outcome[
            "raw_reference_repeat_environment_reward"
        ],
        "sanitized_environment_reward": outcome[
            "sanitized_environment_reward"
        ],
        "full_official_reward_status": FULL_OFFICIAL_REWARD_STATUS,
        "official_test_used": False,
    }
    row["task_preflight_sha256"] = sha256(row)
    return row


def task_error_row(task_identity: str, error: Exception) -> dict[str, Any]:
    """Represent an unclassified task exception; this makes the receipt NO_GO."""
    row = {
        "protocol": RECEIPT_PROTOCOL,
        "design_protocol": DESIGN_PROTOCOL,
        "environment_reward_scope": ENVIRONMENT_REWARD_SCOPE,
        "task_identity": task_identity,
        "status": "ERROR",
        "reason_codes": ["UNCLASSIFIED_TASK_PREFLIGHT_EXCEPTION"],
        "error": f"{type(error).__name__}: {error}",
        "expected_error_indices": [],
        "expected_error_set_sha256": sha256([]),
        "sanitized_successful_reference_indices": [],
        "sanitized_successful_plan_sha256": sha256([]),
        "sanitized_reference_indices": [],
        "eligible_forced_first_reference_indices": [],
        "eligible_forced_first_reference_indices_sha256": sha256([]),
        "ineligible_forced_first_reference_indices": [],
        "raw_reference_environment_reward": None,
        "raw_reference_repeat_environment_reward": None,
        "sanitized_environment_reward": None,
        "full_official_reward_status": FULL_OFFICIAL_REWARD_STATUS,
        "official_test_used": False,
    }
    row["task_preflight_sha256"] = sha256(row)
    return row


def build_preflight_receipt(
    *,
    expected_task_ids: Sequence[str],
    task_results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    task_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    expected = list(expected_task_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ReferenceContractError("expected task identities are empty/duplicated")
    supplied: dict[str, Mapping[str, Any]] = {}
    for row in task_results:
        task = row.get("task_identity")
        if not isinstance(task, str) or not task or task in supplied:
            raise ReferenceContractError("task results have missing/duplicate identity")
        supplied[task] = row
    missing = [task for task in expected if task not in supplied]
    unexpected = sorted(set(supplied) - set(expected))
    ordered = [deepcopy(supplied[task]) for task in expected if task in supplied]
    error_tasks = [
        row["task_identity"]
        for row in ordered
        if row.get("status") not in TERMINAL_TASK_STATUSES
    ]
    pass_tasks = [
        row["task_identity"] for row in ordered if row.get("status") == "PASS"
    ]
    rejected_tasks = [
        row["task_identity"] for row in ordered if row.get("status") == "REJECTED"
    ]
    source = provenance.get("source")
    tau2 = provenance.get("tau2")
    source_clean = isinstance(source, Mapping) and source.get(
        "tracked_worktree_clean"
    ) is True
    tau2_clean = isinstance(tau2, Mapping) and tau2.get(
        "tracked_worktree_clean"
    ) is True
    top_pass = (
        not missing
        and not unexpected
        and not error_tasks
        and source_clean
        and tau2_clean
    )
    task_hashes = {str(key): str(value) for key, value in task_source_hashes.items()}
    receipt = {
        "protocol": RECEIPT_PROTOCOL,
        "design_protocol": DESIGN_PROTOCOL,
        "artifact_type": ARTIFACT_TYPE,
        "contract_version": CONTRACT_VERSION,
        "environment_reward_scope": ENVIRONMENT_REWARD_SCOPE,
        "status": "PASS" if top_pass else "NO_GO",
        "audit_semantics": (
            "PASS means every expected task reached terminal PASS or REJECTED; "
            "only task PASS rows are registry-eligible"
        ),
        "ordered_task_ids": expected,
        "ordered_task_ids_sha256": sha256(expected),
        "task_source_hashes": task_hashes,
        "task_source_hashes_sha256": sha256(task_hashes),
        "provenance": deepcopy(dict(provenance)),
        "summary": {
            "expected_task_count": len(expected),
            "supplied_task_count": len(supplied),
            "pass_task_count": len(pass_tasks),
            "rejected_task_count": len(rejected_tasks),
            "error_task_count": len(error_tasks),
            "missing_task_ids": missing,
            "unexpected_task_ids": unexpected,
            "pass_task_ids": pass_tasks,
            "rejected_task_ids": rejected_tasks,
            "error_task_ids": error_tasks,
            "expected_error_count": sum(
                len(row.get("expected_error_indices") or []) for row in ordered
            ),
            "eligible_forced_first_slot_count": sum(
                len(row.get("eligible_forced_first_reference_indices") or [])
                for row in ordered
            ),
        },
        "task_results": ordered,
        "full_official_reward_status": FULL_OFFICIAL_REWARD_STATUS,
        "official_test_used": False,
    }
    receipt["receipt_sha256"] = sha256(receipt)
    return receipt


def verify_receipt_hash(receipt: Mapping[str, Any]) -> None:
    value = deepcopy(dict(receipt))
    declared = value.pop("receipt_sha256", None)
    observed = sha256(value)
    if declared != observed:
        raise ReferenceContractError(
            f"preflight receipt hash drift: {declared!r} != {observed}"
        )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReferenceContractError(f"{field} is not a lowercase sha256")
    return value


def _require_git_commit(value: Any, field: str) -> str:
    """Validate the frozen repositories' SHA-1 object format."""
    if not isinstance(value, str) or GIT_SHA1_RE.fullmatch(value) is None:
        raise ReferenceContractError(f"{field} is not a 40-hex git commit")
    return value


def verify_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_task_ids: Sequence[str] | None = None,
    expected_source_commit: str | None = None,
    expected_tau2_commit: str | None = None,
    expected_preflight_script_sha256: str | None = None,
    expected_contract_module_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    require_pass: bool = True,
) -> dict[str, Mapping[str, Any]]:
    """Validate schema, content hashes, task coverage, and stale bindings."""
    verify_receipt_hash(receipt)
    if (
        receipt.get("protocol") != RECEIPT_PROTOCOL
        or receipt.get("design_protocol") != DESIGN_PROTOCOL
        or receipt.get("artifact_type") != ARTIFACT_TYPE
        or receipt.get("contract_version") != CONTRACT_VERSION
        or receipt.get("environment_reward_scope") != ENVIRONMENT_REWARD_SCOPE
        or receipt.get("official_test_used") is not False
        or receipt.get("full_official_reward_status")
        != FULL_OFFICIAL_REWARD_STATUS
    ):
        raise ReferenceContractError("reference preflight identity/schema drift")
    if require_pass and receipt.get("status") != "PASS":
        raise ReferenceContractError("reference preflight receipt is not PASS")

    ordered = receipt.get("ordered_task_ids")
    rows = receipt.get("task_results")
    if (
        not isinstance(ordered, list)
        or not ordered
        or len(ordered) != len(set(ordered))
        or not isinstance(rows, list)
        or len(rows) != len(ordered)
    ):
        raise ReferenceContractError("receipt task coverage is malformed")
    if receipt.get("ordered_task_ids_sha256") != sha256(ordered):
        raise ReferenceContractError("ordered task-list hash drift")
    if expected_task_ids is not None and ordered != list(expected_task_ids):
        raise ReferenceContractError("receipt covers a stale/wrong task list")

    source_hashes = receipt.get("task_source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(ordered)
        or receipt.get("task_source_hashes_sha256")
        != sha256(dict(source_hashes))
    ):
        raise ReferenceContractError("task-source coverage/hash drift")
    for task, digest in source_hashes.items():
        _require_sha256(digest, f"task_source_hashes.{task}")

    provenance = receipt.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ReferenceContractError("receipt lacks provenance")
    source = provenance.get("source")
    tau2 = provenance.get("tau2")
    files = provenance.get("files")
    if not all(isinstance(value, Mapping) for value in (source, tau2, files)):
        raise ReferenceContractError("receipt provenance sections are missing")
    required_file_hashes = (
        "preflight_script_sha256",
        "contract_module_sha256",
        "config_sha256",
        "split_manifest_sha256",
    )
    for field in required_file_hashes:
        _require_sha256(files.get(field), f"provenance.files.{field}")
    _require_git_commit(source.get("commit"), "provenance.source.commit")
    _require_sha256(source.get("tree"), "provenance.source.tree")
    _require_git_commit(tau2.get("commit"), "provenance.tau2.commit")
    _require_sha256(tau2.get("tree"), "provenance.tau2.tree")
    if (
        source.get("tracked_worktree_clean") is not True
        or tau2.get("tracked_worktree_clean") is not True
    ):
        raise ReferenceContractError(
            "source and tau2 tracked worktrees must both be clean"
        )

    stale_checks = {
        "source_commit": (source.get("commit"), expected_source_commit),
        "tau2_commit": (tau2.get("commit"), expected_tau2_commit),
        "preflight_script_sha256": (
            files.get("preflight_script_sha256"),
            expected_preflight_script_sha256,
        ),
        "contract_module_sha256": (
            files.get("contract_module_sha256"),
            expected_contract_module_sha256,
        ),
        "config_sha256": (files.get("config_sha256"), expected_config_sha256),
        "split_manifest_sha256": (
            files.get("split_manifest_sha256"),
            expected_split_manifest_sha256,
        ),
    }
    for field, (observed, expected) in stale_checks.items():
        if expected is not None and observed != expected:
            raise ReferenceContractError(
                f"stale preflight binding {field}: {observed!r} != {expected!r}"
            )

    by_task: dict[str, Mapping[str, Any]] = {}
    for expected_identity, row in zip(ordered, rows):
        if not isinstance(row, Mapping) or row.get(
            "task_identity"
        ) != expected_identity:
            raise ReferenceContractError("task result order/identity drift")
        row_copy = deepcopy(dict(row))
        declared = row_copy.pop("task_preflight_sha256", None)
        if declared != sha256(row_copy):
            raise ReferenceContractError(
                f"{expected_identity}: task preflight hash drift"
            )
        status = row.get("status")
        if status not in {*TERMINAL_TASK_STATUSES, "ERROR"}:
            raise ReferenceContractError(
                f"{expected_identity}: unknown task preflight status"
            )
        if status == "PASS":
            outcome = row.get("reference_outcome_contract") or {}
            screen = row.get("forced_first_slot_screen") or {}
            if (
                row.get("environment_reward_scope") != ENVIRONMENT_REWARD_SCOPE
                or outcome.get("environment_reward_scope")
                != ENVIRONMENT_REWARD_SCOPE
                or row.get("raw_reference_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or row.get("raw_reference_repeat_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or row.get("sanitized_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or row.get("full_official_reward_status")
                != FULL_OFFICIAL_REWARD_STATUS
                or (row.get("renderer_lint") or {}).get("status") != "PASS"
                or outcome.get("status") != "PASS"
                or outcome.get("raw_reference_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or outcome.get("raw_reference_repeat_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or outcome.get("sanitized_environment_reward")
                != ENVIRONMENT_REWARD_REQUIRED
                or outcome.get("sanitized_plan_all_actions_succeeded") is not True
                or outcome.get("sanitized_final_state_equivalent") is not True
                or outcome.get("sanitized_tool_error_indices") != []
                or outcome.get("full_official_reward_status")
                != FULL_OFFICIAL_REWARD_STATUS
                or screen.get("status") != "PASS"
                or len(row.get("eligible_forced_first_reference_indices") or [])
                < 2
            ):
                raise ReferenceContractError(
                    f"{expected_identity}: PASS row violates scientific gates"
                )
            if any(
                forbidden in row or forbidden in outcome
                for forbidden in (
                    "raw_reference_official_reward",
                    "raw_reference_repeat_official_reward",
                    "sanitized_official_reward",
                    "official_reward",
                )
            ):
                raise ReferenceContractError(
                    f"{expected_identity}: static receipt claims full official reward"
                )
            screen_eligible = screen.get(
                "eligible_forced_first_reference_indices"
            )
            if screen_eligible != row.get(
                "eligible_forced_first_reference_indices"
            ):
                raise ReferenceContractError(
                    f"{expected_identity}: forced-first eligible-index drift"
                )
            for screen_row in screen.get("rows") or []:
                if screen_row.get("eligible") is True and (
                    screen_row.get("forced_first_environment_reward")
                    != ENVIRONMENT_REWARD_REQUIRED
                    or screen_row.get("final_state_equivalent") is not True
                    or screen_row.get("failing_reference_indices") != []
                ):
                    raise ReferenceContractError(
                        f"{expected_identity}: eligible forced-first row "
                        "violates environment gates"
                    )
        expected_errors = row.get("expected_error_indices")
        sanitized = row.get("sanitized_successful_reference_indices")
        eligible = row.get("eligible_forced_first_reference_indices")
        if not all(
            isinstance(value, list)
            for value in (expected_errors, sanitized, eligible)
        ):
            raise ReferenceContractError(
                f"{expected_identity}: index lists are missing"
            )
        if set(expected_errors) & set(sanitized):
            raise ReferenceContractError(
                f"{expected_identity}: expected-error index retained in sanitized plan"
            )
        if not set(eligible).issubset(set(sanitized)):
            raise ReferenceContractError(
                f"{expected_identity}: eligible index is outside sanitized plan"
            )
        if (
            row.get("expected_error_set_sha256") != sha256(expected_errors)
            or row.get("sanitized_successful_plan_sha256") != sha256(sanitized)
            or row.get("eligible_forced_first_reference_indices_sha256")
            != sha256(eligible)
            or row.get("sanitized_reference_indices") != sanitized
        ):
            raise ReferenceContractError(
                f"{expected_identity}: task index-set hash/alias drift"
            )
        by_task[expected_identity] = row

    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise ReferenceContractError("receipt lacks summary")
    if summary.get("expected_task_count") != len(ordered):
        raise ReferenceContractError("receipt summary task count drift")
    return by_task


def task_contract_from_receipt(
    receipt: Mapping[str, Any],
    task_identity: str,
    *,
    require_task_pass: bool = True,
) -> Mapping[str, Any]:
    """Canonical consumer accessor; callers need not duplicate receipt schema."""
    by_task = verify_preflight_receipt(receipt)
    row = by_task.get(task_identity)
    if row is None:
        raise ReferenceContractError(
            f"reference preflight does not cover {task_identity}"
        )
    if require_task_pass and row.get("status") != "PASS":
        raise ReferenceContractError(
            f"{task_identity}: reference preflight task is not PASS"
        )
    return row


def atomic_write_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    refuse_overwrite: bool = True,
) -> None:
    verify_receipt_hash(receipt)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise ReferenceContractError(f"refusing to overwrite receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(receipt),
                stream,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
