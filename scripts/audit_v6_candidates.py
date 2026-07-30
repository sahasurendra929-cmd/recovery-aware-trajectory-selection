#!/usr/bin/env python3
"""Fail-closed audit for materialized V6 candidate pairs.

The producer is not trusted to declare a pair valid.  This module binds every
result to the structural registry and recomputes:

* full clean/recovery trace and shared-prefix identities;
* the common pre-error environment snapshot;
* exact registered failed calls, real tool errors, and zero failed-call state
  mutation in the agent database and in the user database when Tau2 exposes
  one (otherwise requiring explicit null evidence before and after);
* two distinct canonical corrective families and exact matched first actions;
* matched end-to-end reward plus independent replay;
* zero clean-future leakage and zero failed/error positive labels; and
* forced-first matched/cross replay kappa under one continuation contract.

Low-kappa pairs are deliberately retained as controls.  Kappa is a selection
feature and identifiability stratum, not an acceptance threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import v6_selection_protocol as protocol
except ModuleNotFoundError:
    from scripts import v6_selection_protocol as protocol

try:
    import prepare_v6_candidate_registry as registry_protocol
except ModuleNotFoundError:
    from scripts import prepare_v6_candidate_registry as registry_protocol

try:
    import measure_v6_candidate_tokens as measurement_protocol
except ModuleNotFoundError:
    from scripts import measure_v6_candidate_tokens as measurement_protocol

try:
    import v6_10_selection_protocol as closure_protocol
except ModuleNotFoundError:
    from scripts import v6_10_selection_protocol as closure_protocol

try:
    import v6_reference_contract as reference_contract
except ModuleNotFoundError:
    from scripts import v6_reference_contract as reference_contract


AUDIT_PROTOCOL = "v6_candidate_pair_audit_v1"
AUDIT_VERSION = "1.0"
GENERATION_PROTOCOL = "v6_fresh_recovery_candidate_generation_v1"
FROZEN_STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FROZEN_STUDENT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
V6_10_PIPELINE_CLOSURE_PROTOCOL = closure_protocol.PROTOCOL
V6_10_REGISTRY_PROTOCOL = closure_protocol.REGISTRY_PROTOCOL
V6_10_REFERENCE_PREFLIGHT_PROTOCOL = reference_contract.RECEIPT_PROTOCOL
V6_10_COMPATIBILITY_SEEDS = (20260806,)
V6_10_CONTINUATION_SEEDS = (20260806, 20260807, 20260808)
V6_10_COMPATIBILITY_MIN_TASKS_WITH_THREE_PAIRS = 22
V6_10_COMPATIBILITY_MIN_ACCEPTED_PAIRS = 66
V6_10_TYPED_TASK_REJECTION_CODES = {
    "REFERENCE_PREFLIGHT_TASK_REJECTED",
    "FEWER_THAN_THREE_EXECUTABLE_PAIRS",
    "REFERENCE_ACTION_INELIGIBLE",
    "INJECTED_ERROR_INVALID",
    "CLEAN_GENERATION_EXHAUSTED",
    "MATCHED_POSITIVE_RECOVERY_FAILED",
    "INDEPENDENT_REPLAY_FAILED",
    "POSITIVE_SUFFIX_INVALID",
}
V6_10_VOLATILE_MESSAGE_FIELDS = {
    "timestamp",
    "turn_idx",
    "cost",
    "usage",
    "generation_time_seconds",
}
V6_10_FORBIDDEN_CAUSAL_HELPER_MARKERS = (
    "deterministic_reference",
    "reference_completion",
    "reference_tail",
    "oracle",
)
MEASUREMENT_CONTRACT_FIELDS = (
    "protocol",
    "model_name",
    "model_revision",
    "tokenizer_name",
    "tokenizer_revision",
    "frozen_checkpoint_identity_sha256",
    "tool_schemas_sha256",
    "failed_calls_supervised",
    "full_fresh_assistant_suffix_supervised",
)


class V6CandidateAuditError(RuntimeError):
    """The evidence bundle is malformed or not registry-bound."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_message_payload(value: Any) -> Any:
    """Match the producer's hash of a raw, unnormalized message."""
    if isinstance(value, list):
        return [_semantic_message_payload(item) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _semantic_message_payload(item)
            for key, item in sorted(value.items())
            if key not in V6_10_VOLATILE_MESSAGE_FIELDS
        }
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise V6CandidateAuditError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise V6CandidateAuditError(
                f"{path}:{line_number}: expected an object"
            )
        rows.append(row)
    return rows


def _messages(value: Any) -> list[dict[str, Any]] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        return None
    return deepcopy(value)


def _is_prefix(prefix: list[Any], full: list[Any]) -> bool:
    return len(full) >= len(prefix) and full[: len(prefix)] == prefix


def _compact_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
        return None
    return {"name": name, "arguments": deepcopy(dict(arguments))}


def _first_assistant_call(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not calls:
            continue
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        return _compact_call(calls[0])
    return None


def _call_core_from_registered(branch: Mapping[str, Any]) -> dict[str, Any] | None:
    corrective = branch.get("corrective_action_spec")
    forced = (
        corrective.get("forced_first_action_constructor")
        if isinstance(corrective, Mapping)
        else None
    )
    call = forced.get("tool_call") if isinstance(forced, Mapping) else None
    return _compact_call(call)


def _error_event_calls(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if len(messages) != 2:
        return None, None
    assistant, tool = messages
    calls = assistant.get("tool_calls")
    if (
        assistant.get("role") != "assistant"
        or not isinstance(calls, list)
        or len(calls) != 1
        or tool.get("role") != "tool"
    ):
        return None, None
    return _compact_call(calls[0]), tool


def _valid_success(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isclose(float(value), 1.0, abs_tol=1e-12)
    )


def _official_reward_matches(
    reward_info: Any,
    value: Any,
    *,
    require_success: bool,
) -> bool:
    if (
        not isinstance(reward_info, Mapping)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        return False
    declared = reward_info.get("reward")
    if (
        isinstance(declared, bool)
        or not isinstance(declared, (int, float))
        or not math.isfinite(float(declared))
        or not math.isfinite(float(value))
        or not math.isclose(float(declared), float(value), abs_tol=1e-12)
    ):
        return False
    return not require_success or _valid_success(value)


def _expected_fresh_suffix_mask(
    messages: list[dict[str, Any]],
) -> list[bool]:
    """Label every non-failed assistant output in the fresh suffix.

    This includes ordinary assistant text as well as tool calls.  A tool call
    whose immediately linked tool result is an error is always context, never
    a positive target.
    """
    result: list[bool] = []
    for index, message in enumerate(messages):
        is_assistant_output = (
            message.get("role") == "assistant"
            and (
                message.get("content") not in (None, "")
                or bool(message.get("tool_calls"))
            )
        )
        failed = (
            bool(message.get("tool_calls"))
            and index + 1 < len(messages)
            and messages[index + 1].get("role") == "tool"
            and messages[index + 1].get("error") is True
        )
        result.append(bool(is_assistant_output and not failed))
    return result


def _tool_error_target_indices(
    messages: Any,
    label_mask: Any,
) -> list[int] | None:
    """Return labelled assistant calls whose linked tool result is an error.

    ``None`` means that the evidence is malformed.  An empty list is the only
    passing result.  This distinction is important for V6.10's fail-closed
    positive-data contract.
    """
    parsed = _messages(messages)
    if (
        parsed is None
        or not isinstance(label_mask, list)
        or len(label_mask) != len(parsed)
        or any(not isinstance(value, bool) for value in label_mask)
    ):
        return None
    indices: list[int] = []
    for index, message in enumerate(parsed[:-1]):
        if (
            label_mask[index] is True
            and message.get("role") == "assistant"
            and bool(message.get("tool_calls"))
            and parsed[index + 1].get("role") == "tool"
            and parsed[index + 1].get("error") is True
        ):
            indices.append(index)
    return indices


def _tool_error_indices(messages: Any) -> list[int] | None:
    """Return all failed assistant-tool-call positions in a message sequence."""
    parsed = _messages(messages)
    if parsed is None:
        return None
    return [
        index
        for index, message in enumerate(parsed[:-1])
        if (
            message.get("role") == "assistant"
            and bool(message.get("tool_calls"))
            and parsed[index + 1].get("role") == "tool"
            and parsed[index + 1].get("error") is True
        )
    ]


def _semantic_call(value: Any) -> dict[str, Any] | None:
    """Canonical action semantics used by V6.10 measurement hashes."""
    compact = _compact_call(value)
    if compact is None:
        return None
    requestor = (
        value.get("requestor", "assistant")
        if isinstance(value, Mapping)
        else "assistant"
    )
    return {
        "requestor": str(requestor),
        "name": compact["name"],
        "arguments": deepcopy(compact["arguments"]),
    }


def _raw_probe_call_semantics(value: Any) -> tuple[str, str, str] | None:
    """Reproduce the producer's unnormalized first-call comparison."""
    if not isinstance(value, Mapping):
        return None
    return (
        str(value.get("name", "")),
        canonical(value.get("arguments")),
        str(value.get("requestor", "assistant")),
    )


def _contains_forbidden_causal_helper(
    value: Any,
    *,
    provenance_context: bool = False,
) -> bool:
    """Detect oracle/reference completion provenance inside a causal cell.

    Registered reference indices and forced calls are legitimate.  We scan
    values, not field names, so those exact-index bindings are not confused
    with a continuation policy that was secretly given the reference suffix.
    """
    if isinstance(value, str):
        lowered = value.lower()
        return provenance_context and any(
            marker in lowered
            for marker in V6_10_FORBIDDEN_CAUSAL_HELPER_MARKERS
        )
    if isinstance(value, Mapping):
        provenance_hints = (
            "mode",
            "policy",
            "agent",
            "helper",
            "provenance",
            "source",
            "generator",
            "renderer",
        )
        return any(
            _contains_forbidden_causal_helper(
                item,
                provenance_context=(
                    provenance_context
                    or any(hint in str(key).lower() for hint in provenance_hints)
                ),
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(
            _contains_forbidden_causal_helper(
                item,
                provenance_context=provenance_context,
            )
            for item in value
        )
    return False


def _v610_design_context(
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    registry_root: Mapping[str, Any] | None,
) -> tuple[bool, str | None, str | None]:
    contract = record.get("generation_contract")
    record_design = (
        contract.get("design_protocol")
        if isinstance(contract, Mapping)
        else None
    )
    registry_design = (
        registry_root.get("design_protocol")
        if isinstance(registry_root, Mapping)
        else registered.get("design_protocol")
    )
    is_v610 = V6_10_PIPELINE_CLOSURE_PROTOCOL in {
        record_design,
        registry_design,
    }
    return is_v610, (
        str(record_design) if isinstance(record_design, str) else None
    ), (
        str(registry_design) if isinstance(registry_design, str) else None
    )


def _v610_reference_plan_audit(
    *,
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    registry_root: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Validate the per-task sanitized plan and its registry/receipt binding."""
    generation = record.get("generation_contract")
    runtime = (
        generation.get("runtime_provenance")
        if isinstance(generation, Mapping)
        else None
    )
    receipt = (
        runtime.get("reference_preflight")
        if isinstance(runtime, Mapping)
        else None
    )
    task_row = record.get("reference_plan_binding")
    task_identity = record.get("task_identity")
    expected_errors = (
        task_row.get("expected_error_indices")
        if isinstance(task_row, Mapping)
        else None
    )
    sanitized = (
        task_row.get("sanitized_successful_reference_indices")
        if isinstance(task_row, Mapping)
        else None
    )
    eligible = (
        task_row.get("eligible_forced_first_reference_indices")
        if isinstance(task_row, Mapping)
        else None
    )
    task_hash = (
        task_row.get("task_preflight_sha256")
        if isinstance(task_row, Mapping)
        else None
    )
    plan_hash = (
        task_row.get("sanitized_successful_plan_sha256")
        if isinstance(task_row, Mapping)
        else None
    )
    receipt_hash = (
        receipt.get("receipt_sha256")
        if isinstance(receipt, Mapping)
        else None
    )
    receipt_file_hash = (
        receipt.get("file_sha256")
        if isinstance(receipt, Mapping)
        else None
    )
    registry_receipt_hash = (
        registry_root.get("reference_preflight_receipt_sha256")
        if isinstance(registry_root, Mapping)
        else registered.get("reference_preflight_receipt_sha256")
    )
    registry_receipt_file_hash = (
        registry_root.get("reference_preflight_file_sha256")
        if isinstance(registry_root, Mapping)
        else registered.get("reference_preflight_file_sha256")
    )
    registered_receipt_hash = registered.get(
        "reference_preflight_receipt_sha256"
    )
    registered_receipt_file_hash = registered.get(
        "reference_preflight_file_sha256"
    )
    registered_task_hash = registered.get(
        "reference_task_preflight_sha256"
    )
    registered_plan_hash = registered.get(
        "sanitized_reference_plan_sha256"
    )
    source_commit = (
        runtime.get("observed_source_commit")
        if isinstance(runtime, Mapping)
        else None
    )
    expected_source_commit = (
        runtime.get("expected_source_commit")
        if isinstance(runtime, Mapping)
        else None
    )
    branch_indices: list[int] = []
    branch_bindings_valid = True
    registered_branches = registered.get("branches")
    if isinstance(registered_branches, list):
        for branch in registered_branches:
            corrective = (
                branch.get("corrective_action_spec")
                if isinstance(branch, Mapping)
                else None
            )
            index = (
                corrective.get("reference_action_index")
                if isinstance(corrective, Mapping)
                else None
            )
            if isinstance(index, int) and not isinstance(index, bool):
                branch_indices.append(index)
            binding = (
                branch.get("reference_preflight_binding")
                if isinstance(branch, Mapping)
                else None
            )
            slots = (
                task_row.get("reference_slots_by_index")
                if isinstance(task_row, Mapping)
                else None
            )
            slot = (
                slots.get(str(index))
                if isinstance(slots, Mapping)
                else None
            )
            constructor = (
                corrective.get("forced_first_action_constructor")
                if isinstance(corrective, Mapping)
                else None
            )
            reference_semantics = _semantic_call(
                constructor.get("tool_call")
                if isinstance(constructor, Mapping)
                else None
            )
            branch_bindings_valid = branch_bindings_valid and (
                isinstance(binding, Mapping)
                and set(binding)
                == {
                    "reference_preflight_receipt_sha256",
                    "reference_preflight_file_sha256",
                    "reference_task_preflight_sha256",
                    "sanitized_reference_plan_sha256",
                    "reference_action_index",
                    "reference_slot_id",
                    "reference_slot_sha256",
                    "call_semantics_sha256",
                }
                and binding.get("reference_preflight_receipt_sha256")
                == registry_receipt_hash
                and binding.get("reference_preflight_file_sha256")
                == registry_receipt_file_hash
                and binding.get("reference_task_preflight_sha256")
                == task_hash
                and binding.get("sanitized_reference_plan_sha256")
                == plan_hash
                and binding.get("reference_action_index") == index
                and isinstance(slot, Mapping)
                and binding.get("reference_slot_id")
                == slot.get("reference_slot_id")
                and binding.get("reference_slot_sha256")
                == slot.get("reference_slot_sha256")
                and binding.get("call_semantics_sha256")
                == slot.get("call_semantics_sha256")
                and (
                    reference_semantics is None
                    or binding.get("call_semantics_sha256")
                    == sha256(reference_semantics)
                )
            )
    else:
        branch_bindings_valid = False
    index_lists_valid = (
        isinstance(expected_errors, list)
        and isinstance(sanitized, list)
        and isinstance(eligible, list)
        and all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0
            for values in (expected_errors, sanitized, eligible)
            for index in values
        )
        and all(len(values) == len(set(values)) for values in (
            expected_errors,
            sanitized,
            eligible,
        ))
    )
    checks = {
        "generation_contract_hash_bound": (
            isinstance(generation, Mapping)
            and record.get("generation_contract_sha256")
            == sha256(dict(generation))
        ),
        "generation_contract_design_bound": (
            isinstance(generation, Mapping)
            and generation.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
        ),
        "preflight_receipt_identity_bound": (
            isinstance(receipt, Mapping)
            and receipt.get("protocol")
            == V6_10_REFERENCE_PREFLIGHT_PROTOCOL
            and receipt.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
            and receipt.get("status") == "PASS"
            and receipt.get("tau2_commit") == protocol.TAU2_COMMIT
            and receipt.get("official_test_used") is False
            and _sha256_hex(receipt_hash)
            and _sha256_hex(receipt_file_hash)
            and receipt_hash == registry_receipt_hash
            and receipt_hash == registered_receipt_hash
            and receipt_file_hash == registry_receipt_file_hash
            and receipt_file_hash == registered_receipt_file_hash
            and record.get("reference_preflight_receipt_sha256")
            == registry_receipt_hash
            and record.get("reference_preflight_file_sha256")
            == registry_receipt_file_hash
        ),
        "source_commit_bound": (
            isinstance(source_commit, str)
            and len(source_commit) == 40
            and source_commit == expected_source_commit
            and isinstance(receipt, Mapping)
            and receipt.get("source_commit") == source_commit
        ),
        "task_preflight_row_hash_bound": (
            isinstance(task_row, Mapping)
            and task_row.get("task_identity") == task_identity
            and task_row.get("official_test_used") is False
            and _sha256_hex(task_hash)
            and task_hash == registered_task_hash
            and record.get("reference_task_preflight_sha256")
            == registered_task_hash
            and task_row.get("reference_preflight_receipt_sha256")
            == registry_receipt_hash
            and task_row.get("reference_preflight_file_sha256")
            == registry_receipt_file_hash
        ),
        "sanitized_plan_hash_bound": (
            index_lists_valid
            and plan_hash == sha256(sanitized)
            and plan_hash == registered_plan_hash
            and record.get("sanitized_reference_plan_sha256")
            == registered_plan_hash
            and task_row.get("expected_error_set_sha256")
            == sha256(expected_errors)
            and task_row.get(
                "eligible_forced_first_reference_indices_sha256"
            )
            == sha256(eligible)
            and not (set(expected_errors) & set(sanitized))
            and set(eligible).issubset(set(sanitized))
        ),
        "sanitized_outcome_pass": (
            isinstance(task_row, Mapping)
            and _valid_success(
                task_row.get("raw_reference_environment_reward")
            )
            and _valid_success(
                task_row.get("sanitized_environment_reward")
            )
            and task_row.get("dynamic_full_official_reward_required")
            is True
        ),
        "registered_reference_indices_exact_and_eligible": (
            len(branch_indices) == 2
            and len(set(branch_indices)) == 2
            and index_lists_valid
            and all(
                index in sanitized and index in eligible
                for index in branch_indices
            )
        ),
        "registered_branch_preflight_bindings_exact": (
            branch_bindings_valid
        ),
        "official_test_not_used_anywhere": (
            record.get("official_test_used") is False
            and isinstance(generation, Mapping)
            and generation.get("official_test_used", False) is False
            and isinstance(runtime, Mapping)
            and runtime.get("official_test_used", False) is False
        ),
    }
    normalized = {
        "reference_preflight_receipt_sha256": receipt_hash,
        "reference_preflight_file_sha256": receipt_file_hash,
        "reference_task_preflight_sha256": task_hash,
        "sanitized_reference_plan_sha256": plan_hash,
        "expected_error_indices": deepcopy(expected_errors),
        "sanitized_successful_reference_indices": deepcopy(sanitized),
        "eligible_forced_first_reference_indices": deepcopy(eligible),
    }
    return normalized, checks


def _v610_unforced_measurement_audit(
    *,
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    measurement = record.get("teacher_unforced_first_action_measurement")
    trials = (
        measurement.get("trials")
        if isinstance(measurement, Mapping)
        else None
    )
    trial = (
        trials[0]
        if isinstance(trials, list)
        and len(trials) == 1
        and isinstance(trials[0], Mapping)
        else None
    )
    corrective = registered.get("corrective_action_spec")
    constructor = (
        corrective.get("forced_first_action_constructor")
        if isinstance(corrective, Mapping)
        else None
    )
    reference_call = (
        constructor.get("tool_call")
        if isinstance(constructor, Mapping)
        else None
    )
    raw_response = (
        trial.get("raw_response")
        if isinstance(trial, Mapping)
        else None
    )
    raw_calls = (
        raw_response.get("tool_calls")
        if isinstance(raw_response, Mapping)
        else None
    )
    valid_single_tool_call = (
        isinstance(raw_response, Mapping)
        and raw_response.get("role") == "assistant"
        and raw_response.get("content") in (None, "")
        and isinstance(raw_calls, list)
        and len(raw_calls) == 1
        and isinstance(raw_calls[0], Mapping)
    )
    raw_first_action = (
        deepcopy(dict(raw_calls[0])) if valid_single_tool_call else None
    )
    observed_call = _compact_call(raw_first_action)
    observed_semantics = _raw_probe_call_semantics(raw_first_action)
    reference_semantics = _raw_probe_call_semantics(reference_call)
    matched = (
        raw_first_action is not None
        and reference_semantics is not None
        and observed_semantics == reference_semantics
    )
    registered_seed = registered.get("recovery_seed")
    expected_measurement_keys = {
        "mode",
        "forced_first",
        "gold_suffix_visible",
        "fresh_recovery_generated",
        "raw_first_response_generated",
        "teacher_model",
        "teacher_revision",
        "measurement_seed",
        "measurement_seed_sha256",
        "trial_count",
        "generation_count",
        "matched_registered_corrective_accuracy",
        "normalization_applied",
        "normalization_forbidden",
        "task_success_status",
        "trials",
        "official_test_used",
    }
    expected_trial_keys = {
        "seed",
        "generation_count",
        "task_success",
        "task_success_status",
        "official_reward_info",
        "first_action_observed",
        "first_action",
        "first_action_matched_registered_corrective",
        "first_response_status",
        "malformed_reason",
        "first_action_semantics_sha256",
        "raw_response",
        "raw_response_sha256",
        "normalization_applied",
        "normalization_forbidden",
        "user_continuation_generated",
        "judge_invoked",
        "full_rollout_generated",
        "retry_count",
        "wall_seconds",
    }
    expected_probe_contract = {
        "raw_teacher_response": True,
        "assistant_generations_per_error_context": 1,
        "normalization_applied": False,
        "user_continuation_generated": False,
        "judge_invoked": False,
        "task_success_status": (
            "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE"
        ),
        "retry_count": 0,
    }
    wall_seconds = trial.get("wall_seconds") if isinstance(trial, Mapping) else None
    checks = {
        "measurement_mode_is_unforced_unretried": (
            isinstance(measurement, Mapping)
            and set(measurement) == expected_measurement_keys
            and measurement.get("mode")
            == "teacher_unforced_raw_first_response"
            and measurement.get("forced_first") is False
            and measurement.get("gold_suffix_visible") is False
            and measurement.get("fresh_recovery_generated") is False
            and measurement.get("raw_first_response_generated") is True
            and measurement.get("trial_count") == 1
            and measurement.get("generation_count") == 1
            and measurement.get("official_test_used") is False
            and isinstance(trials, list)
            and len(trials) == 1
            and isinstance(trial, Mapping)
            and set(trial) == expected_trial_keys
        ),
        "measurement_model_bound": (
            isinstance(measurement, Mapping)
            and measurement.get("teacher_model")
            == generation_contract.get("teacher_model")
            and measurement.get("teacher_revision")
            == generation_contract.get("teacher_revision")
        ),
        "measurement_seed_registry_bound": (
            isinstance(registered_seed, int)
            and not isinstance(registered_seed, bool)
            and isinstance(measurement, Mapping)
            and measurement.get("measurement_seed") == registered_seed
            and measurement.get("measurement_seed_sha256")
            == sha256(registered_seed)
            and isinstance(trial, Mapping)
            and trial.get("seed") == registered_seed
        ),
        "raw_first_response_shape_and_hash_exact": (
            isinstance(raw_response, Mapping)
            and isinstance(trial, Mapping)
            and trial.get("raw_response_sha256")
            == sha256(_semantic_message_payload(raw_response))
            and trial.get("first_action") == raw_first_action
            and trial.get("first_action_observed")
            is valid_single_tool_call
            and trial.get("first_response_status")
            == (
                "VALID_SINGLE_TOOL_CALL"
                if valid_single_tool_call
                else "INCORRECT_OR_MALFORMED"
            )
            and trial.get("malformed_reason")
            == (
                None
                if valid_single_tool_call
                else "RAW_FIRST_RESPONSE_NOT_ONE_TOOL_ONLY_CALL"
            )
            and isinstance(wall_seconds, (int, float))
            and not isinstance(wall_seconds, bool)
            and math.isfinite(float(wall_seconds))
            and float(wall_seconds) >= 0.0
        ),
        "unforced_first_action_recomputed": (
            isinstance(trial, Mapping)
            and trial.get(
                "first_action_matched_registered_corrective"
            )
            is matched
            and (
                trial.get("first_action_semantics_sha256")
                == sha256(observed_semantics)
                if observed_semantics is not None
                else trial.get("first_action_semantics_sha256") is None
            )
        ),
        "unforced_probe_no_normalization_or_continuation": (
            isinstance(measurement, Mapping)
            and measurement.get("normalization_applied") is False
            and measurement.get("normalization_forbidden") is True
            and isinstance(trial, Mapping)
            and trial.get("generation_count") == 1
            and trial.get("normalization_applied") is False
            and trial.get("normalization_forbidden") is True
            and trial.get("user_continuation_generated") is False
            and trial.get("judge_invoked") is False
            and trial.get("full_rollout_generated") is False
            and trial.get("retry_count") == 0
            and "independent_replay" not in trial
            and "all_replays_pass" not in measurement
        ),
        "unforced_probe_task_success_deferred": (
            isinstance(measurement, Mapping)
            and measurement.get("task_success_status")
            == "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE"
            and isinstance(trial, Mapping)
            and trial.get("task_success") is None
            and trial.get("task_success_status")
            == "NOT_MEASURED_FIRST_RESPONSE_ONLY"
            and trial.get("official_reward_info") is None
        ),
        "unforced_probe_generation_contract_bound": (
            generation_contract.get("first_action_measurement_mode")
            == "teacher_unforced"
            and generation_contract.get(
                "teacher_unforced_first_action_generated"
            )
            is True
            and generation_contract.get("teacher_unforced_probe_contract")
            == expected_probe_contract
        ),
        "measurement_accuracy_recomputed": (
            isinstance(measurement, Mapping)
            and isinstance(
                measurement.get(
                    "matched_registered_corrective_accuracy"
                ),
                (int, float),
            )
            and not isinstance(
                measurement.get(
                    "matched_registered_corrective_accuracy"
                ),
                bool,
            )
            and math.isclose(
                float(
                    measurement[
                        "matched_registered_corrective_accuracy"
                    ]
                ),
                float(matched),
                abs_tol=1e-12,
            )
        ),
    }
    normalized = {
        "teacher_first_attempt_action_matched": matched,
        "teacher_first_attempt_action": observed_call,
        "teacher_first_attempt_measurement_source": (
            "teacher_unforced_first_action_measurement"
        ),
        "teacher_unforced_first_action_measurement": deepcopy(
            measurement
        ),
    }
    return normalized, checks


def _v610_causal_cells_audit(
    *,
    cells: Any,
    registered: Mapping[str, Any],
    phase: Any,
    role_separation: Any,
) -> dict[str, bool]:
    expected_seeds = (
        V6_10_COMPATIBILITY_SEEDS
        if phase == "compatibility"
        else V6_10_CONTINUATION_SEEDS
        if phase in {"pilot", "formal"}
        else ()
    )
    branches = registered.get("branches")
    branch_indices: list[Any] = []
    branch_calls: list[dict[str, Any] | None] = []
    branch_preflight_bindings: list[Mapping[str, Any] | None] = []
    if isinstance(branches, list):
        for branch in branches:
            corrective = (
                branch.get("corrective_action_spec")
                if isinstance(branch, Mapping)
                else None
            )
            branch_indices.append(
                corrective.get("reference_action_index")
                if isinstance(corrective, Mapping)
                else None
            )
            branch_calls.append(
                _call_core_from_registered(branch)
                if isinstance(branch, Mapping)
                else None
            )
            binding = (
                branch.get("reference_preflight_binding")
                if isinstance(branch, Mapping)
                else None
            )
            branch_preflight_bindings.append(
                binding if isinstance(binding, Mapping) else None
            )
    action_slot = {
        "q_e1_a1": 0,
        "q_e1_a2": 1,
        "q_e2_a1": 0,
        "q_e2_a2": 1,
    }
    exact_cells = (
        isinstance(cells, Mapping)
        and set(cells) == set(protocol.FORCED_FIRST_CELLS)
    )
    cell_modes_valid = exact_cells
    exact_indices_valid = (
        exact_cells
        and len(branch_indices) == 2
        and len(branch_preflight_bindings) == 2
    )
    exact_calls_valid = exact_cells and len(branch_calls) == 2
    seeds_valid = exact_cells
    helper_free = exact_cells
    full_official_reward_valid = exact_cells
    if exact_cells:
        for name in protocol.FORCED_FIRST_CELLS:
            cell = cells[name]
            slot = action_slot[name]
            gold_access = (
                cell.get("gold_access_audit")
                if isinstance(cell, Mapping)
                else None
            )
            trials = (
                cell.get("trials")
                if isinstance(cell, Mapping)
                else None
            )
            cell_modes_valid = cell_modes_valid and (
                isinstance(cell, Mapping)
                and cell.get("forced_first_only") is True
                and cell.get("continuation_mode") == "fresh_teacher"
                and cell.get("gold_suffix_visible") is False
                and cell.get("fresh_recovery_generated") is True
                and isinstance(gold_access, Mapping)
                and gold_access.get(
                    "gold_reference_actions_visible_to_continuation"
                )
                is False
                and gold_access.get(
                    "evaluation_criteria_visible_to_continuation"
                )
                is False
            )
            exact_indices_valid = exact_indices_valid and (
                isinstance(cell, Mapping)
                and cell.get("forced_reference_action_index")
                == branch_indices[slot]
                and isinstance(branch_preflight_bindings[slot], Mapping)
                and cell.get("forced_reference_slot_id")
                == branch_preflight_bindings[slot].get(
                    "reference_slot_id"
                )
                and cell.get("reference_task_preflight_sha256")
                == branch_preflight_bindings[slot].get(
                    "reference_task_preflight_sha256"
                )
                and cell.get("sanitized_reference_plan_sha256")
                == branch_preflight_bindings[slot].get(
                    "sanitized_reference_plan_sha256"
                )
                and cell.get("reference_preflight_receipt_sha256")
                == branch_preflight_bindings[slot].get(
                    "reference_preflight_receipt_sha256"
                )
                and cell.get("reference_preflight_file_sha256")
                == branch_preflight_bindings[slot].get(
                    "reference_preflight_file_sha256"
                )
            )
            exact_calls_valid = exact_calls_valid and (
                isinstance(trials, list)
                and bool(trials)
                and all(
                    isinstance(trial, Mapping)
                    and _compact_call(trial.get("forced_call"))
                    == branch_calls[slot]
                    for trial in trials
                )
            )
            seeds = (
                [trial.get("seed") for trial in trials]
                if isinstance(trials, list)
                and all(isinstance(trial, Mapping) for trial in trials)
                else None
            )
            seeds_valid = seeds_valid and (
                bool(expected_seeds)
                and seeds == list(expected_seeds)
                and cell.get("continuation_seed_set_sha256")
                == sha256(list(expected_seeds))
            )
            helper_free = helper_free and not (
                isinstance(cell, Mapping)
                and _contains_forbidden_causal_helper(cell)
            )
            full_official_reward_valid = (
                full_official_reward_valid
                and isinstance(trials, list)
                and len(trials) == len(expected_seeds)
                and all(
                    isinstance(trial, Mapping)
                    and _official_reward_matches(
                        trial.get("official_reward_info"),
                        trial.get("task_success"),
                        require_success=False,
                    )
                    for trial in trials
                )
            )
    return {
        "v610_causal_role_separation_bound": (
            isinstance(role_separation, Mapping)
            and role_separation.get("causal_cell_continuation_mode")
            == "fresh_teacher"
            and role_separation.get("first_action_measurement_mode")
            == "teacher_unforced"
            and role_separation.get("causal_cells_gold_free") is True
        ),
        "v610_all_four_cells_fresh_teacher_gold_free": cell_modes_valid,
        "v610_causal_cells_exact_registered_reference_indices": (
            exact_indices_valid
        ),
        "v610_causal_cells_exact_registered_forced_calls": exact_calls_valid,
        "v610_compatibility_cells_use_one_registered_seed": (
            seeds_valid if phase == "compatibility" else True
        ),
        "v610_scientific_cells_use_three_registered_seeds": (
            seeds_valid if phase in {"pilot", "formal"} else True
        ),
        "v610_causal_cells_have_no_reference_or_oracle_helper": helper_free,
        "v610_causal_cells_use_full_official_task_success": (
            full_official_reward_valid
        ),
    }


def _forced_cell_trials_recompute(cell: Mapping[str, Any]) -> bool:
    trials = cell.get("trials")
    if not isinstance(trials, list) or not trials:
        return False
    seeds: list[int] = []
    rewards: list[float] = []
    forced_calls: set[str] = set()
    replay_passes: list[bool] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            return False
        seed = trial.get("seed")
        reward = trial.get("task_success")
        replay = trial.get("independent_replay")
        forced_call = trial.get("forced_call")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not 0.0 <= float(reward) <= 1.0
            or not isinstance(replay, Mapping)
            or replay.get("pass") is not True
            or not isinstance(forced_call, Mapping)
        ):
            return False
        seeds.append(seed)
        rewards.append(float(reward))
        replay_passes.append(replay.get("pass") is True)
        forced_calls.add(canonical(forced_call))
    recomputed = sum(rewards) / len(rewards)
    return (
        len(seeds) == len(set(seeds))
        and len(forced_calls) == 1
        and cell.get("continuation_seed_set_sha256") == sha256(seeds)
        and isinstance(cell.get("task_success"), (int, float))
        and not isinstance(cell.get("task_success"), bool)
        and math.isclose(
            float(cell["task_success"]), recomputed, abs_tol=1e-12
        )
        and cell.get("independent_replay_pass") is all(replay_passes)
    )


def _nonempty_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _explicit_optional_hash(mapping: Any, key: str) -> bool:
    """Accept an exposed nonempty hash or an explicit null-unavailable value."""

    return (
        isinstance(mapping, Mapping)
        and key in mapping
        and (
            mapping.get(key) is None
            or _nonempty_hash(mapping.get(key))
        )
    )


def _sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _measurement_contract_payload(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Mirror the exact payload hashed by the frozen measurement stage."""
    return {
        key: provenance.get(key) for key in MEASUREMENT_CONTRACT_FIELDS
    }


def _recompute_branch_token_costs(
    branches: Any,
) -> dict[str, int] | None:
    """Sum both measured branch contracts and their redundant declarations."""
    if (
        not isinstance(branches, list)
        or len(branches) != 2
        or any(not isinstance(branch, Mapping) for branch in branches)
    ):
        return None
    totals = {
        "contract_supervised_tokens": 0,
        "contract_sequence_tokens": 0,
        "declared_supervised_tokens": 0,
        "declared_sequence_tokens": 0,
    }
    for branch in branches:
        contract = branch.get("token_contract")
        if not isinstance(contract, Mapping):
            return None
        contract_sup = contract.get("supervised_tokens")
        contract_nonpad = contract.get("sequence_tokens")
        declared_sup = branch.get("supervised_target_tokens")
        declared_nonpad = branch.get("nonpadding_tokens")
        if not (
            _positive_int(contract_sup)
            and _positive_int(contract_nonpad)
            and contract_sup <= contract_nonpad
            and _positive_int(declared_sup)
            and _positive_int(declared_nonpad)
            and declared_sup <= declared_nonpad
        ):
            return None
        totals["contract_supervised_tokens"] += contract_sup
        totals["contract_sequence_tokens"] += contract_nonpad
        totals["declared_supervised_tokens"] += declared_sup
        totals["declared_sequence_tokens"] += declared_nonpad
    return totals


def _trace_audit(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    prefix = _messages(record.get("shared_prefix"))
    snapshot = record.get("environment_snapshot")
    clean = record.get("clean_trace")
    snapshot_state = (
        snapshot.get("state") if isinstance(snapshot, Mapping) else None
    )
    clean_messages = (
        _messages(clean.get("messages")) if isinstance(clean, Mapping) else None
    )
    computed_prefix_hash = sha256(prefix) if prefix is not None else None
    computed_snapshot_hash = (
        sha256(snapshot_state)
        if isinstance(snapshot_state, (dict, list))
        else None
    )
    computed_clean_hash = (
        sha256(clean_messages) if clean_messages is not None else None
    )
    training_system = record.get("training_system_message")
    computed_training_system_hash = (
        sha256(training_system)
        if isinstance(training_system, Mapping)
        else None
    )
    checks = {
        "shared_prefix_complete": prefix is not None,
        "stored_shared_prefix_hash_matches": (
            computed_prefix_hash is not None
            and record.get("shared_prefix_sha256") == computed_prefix_hash
        ),
        "environment_snapshot_complete": (
            isinstance(snapshot, Mapping)
            and isinstance(snapshot_state, (dict, list))
            and _nonempty_hash(snapshot.get("agent_db_hash"))
            and _explicit_optional_hash(snapshot, "user_db_hash")
        ),
        "stored_snapshot_hash_matches": (
            computed_snapshot_hash is not None
            and isinstance(snapshot, Mapping)
            and snapshot.get("state_sha256") == computed_snapshot_hash
        ),
        "clean_trace_complete": (
            clean_messages is not None
            and isinstance(clean, Mapping)
            and _valid_success(clean.get("official_task_success"))
            and clean.get("final_state_valid") is True
            and _nonempty_hash(clean.get("final_agent_db_hash"))
            and _explicit_optional_hash(clean, "final_user_db_hash")
        ),
        "clean_trace_starts_with_shared_prefix": (
            prefix is not None
            and clean_messages is not None
            and _is_prefix(prefix, clean_messages)
        ),
        "stored_clean_trace_hash_matches": (
            computed_clean_hash is not None
            and isinstance(clean, Mapping)
            and clean.get("trace_sha256") == computed_clean_hash
        ),
        "training_system_policy_message_bound": (
            isinstance(training_system, Mapping)
            and training_system.get("role") == "system"
            and isinstance(training_system.get("content"), str)
            and bool(training_system["content"].strip())
            and record.get("training_system_message_sha256")
            == computed_training_system_hash
        ),
    }
    normalized = {
        "shared_prefix": prefix,
        "shared_prefix_sha256": computed_prefix_hash,
        "environment_snapshot": deepcopy(snapshot),
        "environment_snapshot_sha256": computed_snapshot_hash,
        "user_db_hash_available": (
            isinstance(snapshot, Mapping)
            and snapshot.get("user_db_hash") is not None
        ),
        "clean_trace": deepcopy(clean),
        "clean_trace_sha256": computed_clean_hash,
        "training_system_message": deepcopy(training_system),
        "training_system_message_sha256": computed_training_system_hash,
    }
    return normalized, checks


def _branch_audit(
    *,
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    shared: Mapping[str, Any],
    is_v610: bool = False,
    generation_contract: Mapping[str, Any] | None = None,
    v610_reference_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    branch_id = registered.get("branch_id")
    error_event = _messages(record.get("error_event_messages"))
    recovery_prompt = _messages(record.get("recovery_prompt"))
    recovery_suffix = _messages(record.get("recovery_suffix"))
    full_trace = _messages(record.get("full_trace"))
    supervised = _messages(record.get("supervised_messages"))
    training_prompt = _messages(
        record.get("training_prompt", record.get("prompt"))
    )
    training_full = _messages(record.get("training_full_trace"))
    event_call, event_result = (
        _error_event_calls(error_event)
        if error_event is not None
        else (None, None)
    )
    registered_error = _compact_call(
        (registered.get("injection_spec") or {}).get("error_call")
    )
    reference_call = _call_core_from_registered(registered)
    first_recovery_call = (
        _first_assistant_call(recovery_suffix)
        if recovery_suffix is not None
        else None
    )
    registered_task = str(registered.get("task_identity", ""))
    registered_domain = (
        registered_task.split(":", 1)[0] if ":" in registered_task else ""
    )
    expected_error_family = (
        f"{registered.get('tool_name')}::{registered.get('identifier_key')}"
    )
    expected_coverage = {
        "domain": registered_domain,
        "failed_tool": registered.get("tool_name"),
        "error_family": expected_error_family,
        "corrective_action": registered.get("corrective_family"),
        "recovery_length_bin": (
            "short"
            if isinstance(recovery_suffix, list) and len(recovery_suffix) <= 4
            else "medium"
            if isinstance(recovery_suffix, list) and len(recovery_suffix) <= 10
            else "long"
        ),
        "recovery_mode": (
            "user_assisted"
            if isinstance(recovery_suffix, list)
            and any(message.get("role") == "user" for message in recovery_suffix)
            else "agent_initiated"
        ),
    }
    observed_coverage = record.get("coverage")
    prefix = shared.get("shared_prefix")
    clean = shared.get("clean_trace")
    clean_messages = (
        clean.get("messages") if isinstance(clean, Mapping) else None
    )
    clean_future = (
        clean_messages[len(prefix) :]
        if isinstance(prefix, list)
        and isinstance(clean_messages, list)
        and _is_prefix(prefix, clean_messages)
        else None
    )
    expected_prompt = (
        [*prefix, *error_event]
        if isinstance(prefix, list) and isinstance(error_event, list)
        else None
    )
    expected_full = (
        [*expected_prompt, *recovery_suffix]
        if isinstance(expected_prompt, list)
        and isinstance(recovery_suffix, list)
        else None
    )
    system_message = shared.get("training_system_message")
    expected_training_prompt = (
        [deepcopy(system_message), *expected_prompt]
        if isinstance(system_message, Mapping)
        and isinstance(expected_prompt, list)
        else None
    )
    expected_training_full = (
        [*expected_training_prompt, *recovery_suffix]
        if isinstance(expected_training_prompt, list)
        and isinstance(recovery_suffix, list)
        else None
    )
    clean_future_hashes = {
        canonical(message) for message in (clean_future or [])
    }
    prompt_tail = (
        recovery_prompt[len(prefix) :]
        if isinstance(prefix, list)
        and isinstance(recovery_prompt, list)
        and _is_prefix(prefix, recovery_prompt)
        else []
    )
    future_overlap = clean_future_hashes & {
        canonical(message) for message in prompt_tail
    }
    failed_overlap = (
        {canonical(message) for message in (error_event or [])}
        & {canonical(message) for message in (supervised or [])}
    )
    snapshot = shared.get("environment_snapshot")
    db = record.get("database_hashes")
    execution = record.get("tool_execution_evidence")
    matched = record.get("matched_replay")
    label = record.get("label_audit")
    producer = record.get("producer_evidence")
    produced_recovery = (
        producer.get("matched_recovery")
        if isinstance(producer, Mapping)
        else None
    )
    produced_reference_plan = (
        produced_recovery.get("reference_plan_binding")
        if isinstance(produced_recovery, Mapping)
        else None
    )
    attempts = (
        produced_recovery.get("attempts")
        if isinstance(produced_recovery, Mapping)
        else None
    )
    first_attempt = (
        attempts[0]
        if isinstance(attempts, list)
        and attempts
        and isinstance(attempts[0], Mapping)
        else None
    )
    first_attempt_action = (
        _compact_call(first_attempt.get("first_action"))
        if isinstance(first_attempt, Mapping)
        else None
    )
    first_attempt_matched = (
        first_attempt_action is not None
        and reference_call is not None
        and first_attempt_action == reference_call
    )
    unforced_normalized: dict[str, Any] = {}
    unforced_checks: dict[str, bool] = {}
    if is_v610:
        unforced_normalized, unforced_checks = (
            _v610_unforced_measurement_audit(
                record=record,
                registered=registered,
                generation_contract=(
                    generation_contract
                    if isinstance(generation_contract, Mapping)
                    else {}
                ),
            )
        )
    measured_suffix_mask = record.get("fresh_recovery_label_mask")
    measured_full_mask = record.get("full_assistant_label_mask")
    producer_full_mask = record.get("label_mask")
    expected_suffix_mask = (
        _expected_fresh_suffix_mask(recovery_suffix)
        if recovery_suffix is not None
        else None
    )
    expected_producer_full_mask = (
        [False] * len(recovery_prompt) + expected_suffix_mask
        if recovery_prompt is not None
        and expected_suffix_mask is not None
        else None
    )
    expected_measured_full_mask = (
        [False] * len(training_prompt) + expected_suffix_mask
        if training_prompt is not None
        and expected_suffix_mask is not None
        else None
    )
    computed_error_hash = sha256(error_event) if error_event is not None else None
    computed_trace_hash = sha256(full_trace) if full_trace is not None else None
    call_key = sha256(first_recovery_call) if first_recovery_call else None
    reference_key = sha256(reference_call) if reference_call else None
    expected_agent = (
        snapshot.get("agent_db_hash") if isinstance(snapshot, Mapping) else None
    )
    expected_user = (
        snapshot.get("user_db_hash") if isinstance(snapshot, Mapping) else None
    )
    clean_final_agent = (
        clean.get("final_agent_db_hash") if isinstance(clean, Mapping) else None
    )
    clean_final_user = (
        clean.get("final_user_db_hash") if isinstance(clean, Mapping) else None
    )
    token_contract = record.get("token_contract")
    contract_sup = (
        token_contract.get("supervised_tokens")
        if isinstance(token_contract, Mapping)
        else None
    )
    contract_nonpad = (
        token_contract.get("sequence_tokens")
        if isinstance(token_contract, Mapping)
        else None
    )
    declared_sup = record.get("supervised_target_tokens")
    declared_nonpad = record.get("nonpadding_tokens")
    recovery_error_indices = _tool_error_indices(recovery_suffix)
    recovery_error_targets = _tool_error_target_indices(
        recovery_suffix,
        measured_suffix_mask,
    )
    checks = {
        "branch_id_registry_bound": record.get("branch_id") == branch_id,
        "branch_slot_hash_registry_bound": (
            record.get("registered_branch_slot_sha256")
            == registered.get("branch_slot_sha256")
        ),
        "branch_prefix_matches_pair": (
            record.get("shared_prefix_sha256")
            == shared.get("shared_prefix_sha256")
        ),
        "branch_snapshot_matches_pair": (
            record.get("environment_snapshot_sha256")
            == shared.get("environment_snapshot_sha256")
        ),
        "error_event_complete": (
            error_event is not None
            and event_call is not None
            and isinstance(event_result, Mapping)
        ),
        "error_call_matches_frozen_injection": (
            event_call is not None and event_call == registered_error
        ),
        "real_tool_error_observed": (
            isinstance(event_result, Mapping)
            and event_result.get("error") is True
        ),
        "stored_error_event_hash_matches": (
            computed_error_hash is not None
            and record.get("error_event_sha256") == computed_error_hash
        ),
        "tool_execution_evidence_complete": (
            isinstance(execution, Mapping)
            and execution.get("executed_in_pinned_environment") is True
            and execution.get("tau2_commit") == protocol.TAU2_COMMIT
            and event_call is not None
            and execution.get("tool_call_sha256") == sha256(event_call)
            and isinstance(event_result, Mapping)
            and execution.get("tool_result_sha256") == sha256(event_result)
        ),
        "failed_injection_agent_db_unchanged": (
            isinstance(db, Mapping)
            and db.get("agent_before_error") == expected_agent
            and db.get("agent_after_error") == expected_agent
            and _nonempty_hash(expected_agent)
        ),
        "failed_injection_user_db_unchanged": (
            isinstance(db, Mapping)
            and "user_before_error" in db
            and "user_after_error" in db
            and _explicit_optional_hash(snapshot, "user_db_hash")
            and db.get("user_before_error") == expected_user
            and db.get("user_after_error") == expected_user
        ),
        "recovery_prompt_exactly_prefix_plus_error": (
            recovery_prompt is not None
            and expected_prompt is not None
            and recovery_prompt == expected_prompt
        ),
        "full_trace_exactly_prompt_plus_fresh_suffix": (
            full_trace is not None
            and expected_full is not None
            and full_trace == expected_full
        ),
        "training_prompt_injects_policy_exactly_once": (
            training_prompt is not None
            and training_prompt == expected_training_prompt
        ),
        "training_full_trace_exactly_prompt_plus_fresh_suffix": (
            training_full is not None
            and training_full == expected_training_full
        ),
        "stored_full_trace_hash_matches": (
            computed_trace_hash is not None
            and record.get("full_trace_sha256") == computed_trace_hash
        ),
        "fresh_recovery_suffix_nonempty": recovery_suffix is not None,
        "corrective_first_action_matches_reference": (
            first_recovery_call is not None
            and reference_call is not None
            and first_recovery_call == reference_call
        ),
        "coverage_recomputed_from_frozen_execution": (
            isinstance(observed_coverage, Mapping)
            and all(
                observed_coverage.get(field) == value
                for field, value in expected_coverage.items()
            )
        ),
        "clean_future_leakage_zero": (
            not future_overlap
            and isinstance(label, Mapping)
            and label.get("future_message_overlap_count") == 0
            and label.get("clean_future_visible") is False
        ),
        "failed_positive_labels_zero": (
            not failed_overlap
            and isinstance(label, Mapping)
            and label.get("failed_positive_label_count") == 0
            and label.get("error_result_positive_label_count") == 0
            and supervised == recovery_suffix
        ),
        "frozen_student_label_mask_exact": (
            isinstance(measured_suffix_mask, list)
            and measured_suffix_mask == expected_suffix_mask
            and isinstance(measured_full_mask, list)
            and measured_full_mask == expected_measured_full_mask
            and isinstance(producer_full_mask, list)
            and producer_full_mask == expected_producer_full_mask
            and bool(expected_suffix_mask)
            and any(expected_suffix_mask)
        ),
        "frozen_student_token_contract_matches_declared_cost": (
            isinstance(token_contract, Mapping)
            and _positive_int(contract_sup)
            and _positive_int(contract_nonpad)
            and contract_sup <= contract_nonpad
            and _positive_int(declared_sup)
            and _positive_int(declared_nonpad)
            and declared_sup <= declared_nonpad
            and declared_sup == contract_sup
            and declared_nonpad == contract_nonpad
            and record.get("token_measurement_status")
            == "MEASURED_FROZEN_STUDENT"
        ),
        "matched_reward_and_independent_replay": (
            isinstance(matched, Mapping)
            and _valid_success(matched.get("official_task_success"))
            and matched.get("independent_replay_pass") is True
            and matched.get("final_state_valid") is True
            and matched.get("final_agent_db_hash") == clean_final_agent
            and isinstance(clean, Mapping)
            and "final_user_db_hash" in clean
            and "final_user_db_hash" in matched
            and matched.get("final_user_db_hash") == clean_final_user
            and _nonempty_hash(matched.get("independent_replay_audit_sha256"))
        ),
        "teacher_first_attempt_action_recomputed": (
            isinstance(first_attempt, Mapping)
            and first_attempt.get("first_action_matched")
            is first_attempt_matched
        ),
        "official_test_not_used": (
            record.get("official_test_used") is False
        ),
    }
    if is_v610:
        checks.update(
            {
                "v610_positive_recovery_suffix_has_no_tool_errors": (
                    recovery_error_indices == []
                ),
                "v610_positive_recovery_has_zero_tool_error_targets": (
                    recovery_error_targets == []
                ),
                "v610_recovery_positive_is_sanitized_plan_bound": (
                    isinstance(produced_recovery, Mapping)
                    and produced_recovery.get("continuation_mode")
                    == "deterministic_reference_completion"
                    and produced_recovery.get("fresh_recovery_generated")
                    is False
                    and produced_recovery.get("gold_suffix_used") is True
                    and isinstance(produced_reference_plan, Mapping)
                    and produced_reference_plan.get(
                        "task_preflight_sha256"
                    )
                    == (
                        v610_reference_binding or {}
                    ).get("reference_task_preflight_sha256")
                    and produced_reference_plan.get(
                        "sanitized_successful_plan_sha256"
                    )
                    == (
                        v610_reference_binding or {}
                    ).get("sanitized_reference_plan_sha256")
                    and record.get("sanitized_reference_plan_sha256")
                    == (
                        v610_reference_binding or {}
                    ).get("sanitized_reference_plan_sha256")
                ),
                "v610_observed_branch_preflight_binding_exact": (
                    isinstance(
                        registered.get("reference_preflight_binding"),
                        Mapping,
                    )
                    and record.get("reference_preflight_binding")
                    == registered.get("reference_preflight_binding")
                    and record.get("reference_action_index")
                    == (
                        registered.get("corrective_action_spec") or {}
                    ).get("reference_action_index")
                    and record.get("reference_slot_id")
                    == registered["reference_preflight_binding"].get(
                        "reference_slot_id"
                    )
                    and record.get("sanitized_reference_plan_sha256")
                    == registered["reference_preflight_binding"].get(
                        "sanitized_reference_plan_sha256"
                    )
                ),
                "v610_recovery_positive_has_full_official_success": (
                    isinstance(produced_recovery, Mapping)
                    and _valid_success(
                        produced_recovery.get("official_task_success")
                    )
                    and isinstance(attempts, list)
                    and len(attempts) == 1
                    and isinstance(attempts[0], Mapping)
                    and _official_reward_matches(
                        attempts[0].get("official_reward_info"),
                        attempts[0].get("official_reward"),
                        require_success=True,
                    )
                    and isinstance(matched, Mapping)
                    and _valid_success(
                        matched.get("official_task_success")
                    )
                ),
                **unforced_checks,
            }
        )
        # A deterministic matched-positive constructor is allowed to use the
        # sanitized plan.  It is not evidence of natural teacher switching.
        checks["teacher_first_attempt_action_recomputed"] = unforced_checks[
            "unforced_first_action_recomputed"
        ]
        first_attempt_matched = bool(
            unforced_normalized.get(
                "teacher_first_attempt_action_matched", False
            )
        )
        first_attempt_action = unforced_normalized.get(
            "teacher_first_attempt_action"
        )
    recovery_suffix_provenance: dict[str, Any] | None = None
    if is_v610 and all(
        checks.get(name) is True
        for name in (
            "v610_positive_recovery_suffix_has_no_tool_errors",
            "v610_positive_recovery_has_zero_tool_error_targets",
            "v610_recovery_positive_is_sanitized_plan_bound",
            "v610_recovery_positive_has_full_official_success",
            "matched_reward_and_independent_replay",
        )
    ):
        recovery_suffix_provenance = {
            "protocol": "v6_10_audited_recovery_suffix_provenance_v1",
            "origin": "sanitized_deterministic_reference_plan",
            "fresh_recovery_generated": False,
            "gold_suffix_used": True,
            "reference_preflight_receipt_sha256": (
                v610_reference_binding.get(
                    "reference_preflight_receipt_sha256"
                )
            ),
            "reference_task_preflight_sha256": (
                v610_reference_binding.get(
                    "reference_task_preflight_sha256"
                )
            ),
            "sanitized_reference_plan_sha256": (
                v610_reference_binding.get(
                    "sanitized_reference_plan_sha256"
                )
            ),
            "matched_recovery_evidence_sha256": sha256(
                produced_recovery
            ),
        }
    if is_v610:
        checks[
            "v610_audited_recovery_suffix_provenance_constructed"
        ] = recovery_suffix_provenance is not None
    normalized = {
        **{
            key: deepcopy(value)
            for key, value in registered.items()
            if key != "branch_slot_sha256"
        },
        "branch_slot_sha256": registered.get("branch_slot_sha256"),
        "error_family": expected_error_family,
        "failed_tool": registered.get("tool_name"),
        "corrective_action": registered.get("corrective_family"),
        "error_event_sha256": computed_error_hash,
        "first_recovery_action_key": call_key,
        "reference_action_key": reference_key,
        "first_recovery_action": first_recovery_call,
        "error_event_messages": deepcopy(error_event),
        "recovery_prompt": deepcopy(recovery_prompt),
        "prompt": deepcopy(training_prompt),
        "training_prompt": deepcopy(training_prompt),
        "recovery_suffix": deepcopy(recovery_suffix),
        "fresh_recovery_suffix": deepcopy(recovery_suffix),
        "full_trace": deepcopy(full_trace),
        "training_full_trace": deepcopy(training_full),
        "full_recovery_messages": deepcopy(full_trace),
        "supervised_messages": deepcopy(supervised),
        "recovery_suffix_provenance": deepcopy(
            recovery_suffix_provenance
        ),
        "label_mask": deepcopy(producer_full_mask),
        "fresh_recovery_label_mask": deepcopy(measured_suffix_mask),
        "full_assistant_label_mask": deepcopy(measured_full_mask),
        "coverage": deepcopy(expected_coverage),
        "tool_schemas": deepcopy(record.get("tool_schemas")),
        "token_contract": deepcopy(record.get("token_contract")),
        "token_measurement_status": record.get("token_measurement_status"),
        "supervised_target_tokens": record.get(
            "supervised_target_tokens"
        ),
        "nonpadding_tokens": record.get("nonpadding_tokens"),
        "agent_db_hash_before_error": (
            db.get("agent_before_error") if isinstance(db, Mapping) else None
        ),
        "agent_db_hash_after_error": (
            db.get("agent_after_error") if isinstance(db, Mapping) else None
        ),
        "user_db_hash_before_error": (
            db.get("user_before_error") if isinstance(db, Mapping) else None
        ),
        "user_db_hash_after_error": (
            db.get("user_after_error") if isinstance(db, Mapping) else None
        ),
        "user_db_hash_available": expected_user is not None,
        "identifier_field_allowlisted": checks[
            "error_call_matches_frozen_injection"
        ],
        "identifier_mutation_type_preserving": (
            type(registered.get("original_identifier"))
            is type(registered.get("mutated_identifier"))
            and registered.get("original_identifier")
            != registered.get("mutated_identifier")
        ),
        "actual_tool_error": checks["real_tool_error_observed"],
        "state_unchanged_after_error": (
            checks["failed_injection_agent_db_unchanged"]
            and checks["failed_injection_user_db_unchanged"]
        ),
        "corrective_first_action_is_reference_call": checks[
            "corrective_first_action_matches_reference"
        ],
        "matched_task_success": checks[
            "matched_reward_and_independent_replay"
        ],
        "teacher_first_attempt_action_matched": first_attempt_matched,
        "teacher_first_attempt_action": first_attempt_action,
        **unforced_normalized,
        "evidence_sha256": sha256(record),
        "audit_checks": checks,
    }
    return normalized, checks


def audit_candidate_pair(
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    *,
    registry_sha256: str,
    registry_root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    is_v610, record_design_protocol, registry_design_protocol = (
        _v610_design_context(record, registered, registry_root)
    )
    generation_contract = record.get("generation_contract")
    generation_contract = (
        generation_contract
        if isinstance(generation_contract, Mapping)
        else {}
    )
    v610_reference_binding: dict[str, Any] = {}
    v610_reference_checks: dict[str, bool] = {}
    if is_v610:
        v610_reference_binding, v610_reference_checks = (
            _v610_reference_plan_audit(
                record=record,
                registered=registered,
                registry_root=registry_root,
            )
        )
    shared, shared_checks = _trace_audit(record)
    record_branches = record.get("branches")
    registered_branches = registered.get("branches")
    branches_well_formed = (
        isinstance(record_branches, list)
        and isinstance(registered_branches, list)
        and len(record_branches) == len(registered_branches) == 2
    )
    branch_rows: list[dict[str, Any]] = []
    branch_checks: dict[str, dict[str, bool]] = {}
    if branches_well_formed:
        supplied_by_id = {
            row.get("branch_id"): row
            for row in record_branches
            if isinstance(row, Mapping)
        }
        if len(supplied_by_id) == 2:
            for registered_branch in registered_branches:
                branch_id = registered_branch.get("branch_id")
                supplied = supplied_by_id.get(branch_id)
                if not isinstance(supplied, Mapping):
                    continue
                normalized, checks = _branch_audit(
                    record=supplied,
                    registered=registered_branch,
                    shared=shared,
                    is_v610=is_v610,
                    generation_contract=generation_contract,
                    v610_reference_binding=v610_reference_binding,
                )
                branch_rows.append(normalized)
                branch_checks[str(branch_id)] = checks
    families = [
        row.get("canonical_corrective_family") for row in branch_rows
    ]
    forced_cells = record.get("forced_first_cells")
    try:
        kappa = protocol.forced_first_common_continuation_kappa(
            forced_cells
        )
        forced_cells_valid = (
            isinstance(forced_cells, Mapping)
            and all(
                isinstance(forced_cells.get(name), Mapping)
                and forced_cells[name].get("independent_replay_pass") is True
                and _forced_cell_trials_recompute(forced_cells[name])
                for name in protocol.FORCED_FIRST_CELLS
            )
        )
    except (protocol.V6ProtocolError, TypeError):
        kappa = None
        forced_cells_valid = False
    v610_causal_checks: dict[str, bool] = {}
    if is_v610:
        v610_causal_checks = _v610_causal_cells_audit(
            cells=forced_cells,
            registered=registered,
            phase=record.get("phase"),
            role_separation=record.get("continuation_role_separation"),
        )
    token = record.get("token_accounting")
    c_sup = (
        token.get("supervised_target_tokens")
        if isinstance(token, Mapping)
        else None
    )
    c_nonpad = (
        token.get("nonpadding_tokens")
        if isinstance(token, Mapping)
        else None
    )
    branch_token_costs = _recompute_branch_token_costs(record_branches)
    record_clean_view = record.get("clean_view")
    measurement = record.get("token_measurement_provenance")
    tool_schemas = record.get("tool_schemas")
    computed_measurement_contract_sha256 = (
        sha256(_measurement_contract_payload(measurement))
        if isinstance(measurement, Mapping)
        else None
    )
    clean_training_messages = (
        _messages(record_clean_view.get("messages"))
        if isinstance(record_clean_view, Mapping)
        else None
    )
    expected_clean_training_messages = (
        [
            deepcopy(shared.get("training_system_message")),
            *deepcopy((shared.get("clean_trace") or {}).get("messages", [])),
        ]
        if isinstance(shared.get("training_system_message"), Mapping)
        and isinstance((shared.get("clean_trace") or {}).get("messages"), list)
        else None
    )
    clean_label_mask = (
        record_clean_view.get("label_mask")
        if isinstance(record_clean_view, Mapping)
        else None
    )
    clean_error_indices = _tool_error_indices(clean_training_messages)
    clean_error_targets = _tool_error_target_indices(
        clean_training_messages,
        clean_label_mask,
    )
    top_checks = {
        "registry_sha256_bound": (
            record.get("registry_sha256") == registry_sha256
        ),
        "candidate_pair_id_bound": (
            record.get("candidate_pair_id")
            == registered.get("candidate_pair_id")
        ),
        "registered_candidate_pair_hash_bound": (
            record.get("registered_candidate_pair_sha256")
            == registered.get("candidate_pair_sha256")
        ),
        "phase_bound": record.get("phase") == registered.get("phase"),
        "task_identity_bound": (
            record.get("task_identity") == registered.get("task_identity")
        ),
        "official_test_not_used": (
            record.get("official_test_used") is False
        ),
        "shared_trace_and_snapshot_valid": all(shared_checks.values()),
        "exactly_two_registry_bound_branches": (
            branches_well_formed
            and len(branch_rows) == 2
            and len(branch_checks) == 2
        ),
        "all_branch_audits_pass": (
            len(branch_checks) == 2
            and all(
                checks and all(checks.values())
                for checks in branch_checks.values()
            )
        ),
        "two_distinct_canonical_corrective_families": (
            len(families) == 2
            and all(isinstance(value, str) and value for value in families)
            and len(set(families)) == 2
            and families
            == registered.get("expected_canonical_corrective_families")
        ),
        "forced_first_replays_complete": forced_cells_valid,
        "positive_token_accounting": (
            _positive_int(c_sup)
            and _positive_int(c_nonpad)
            and c_nonpad >= c_sup
        ),
        "token_accounting_recomputed_from_both_branches": (
            branch_token_costs is not None
            and c_sup
            == branch_token_costs["contract_supervised_tokens"]
            == branch_token_costs["declared_supervised_tokens"]
            and c_nonpad
            == branch_token_costs["contract_sequence_tokens"]
            == branch_token_costs["declared_sequence_tokens"]
        ),
        "clean_training_view_injects_policy_exactly_once": (
            clean_training_messages is not None
            and clean_training_messages == expected_clean_training_messages
            and isinstance(record_clean_view, Mapping)
            and isinstance(record_clean_view.get("label_mask"), list)
            and len(record_clean_view["label_mask"])
            == len(clean_training_messages)
        ),
        "frozen_student_measurement_provenance_bound": (
            isinstance(measurement, Mapping)
            and measurement.get("protocol")
            == measurement_protocol.PROTOCOL
            and measurement.get("model_name") == FROZEN_STUDENT_MODEL
            and measurement.get("model_revision") == FROZEN_STUDENT_REVISION
            and measurement.get("tokenizer_name") == FROZEN_STUDENT_MODEL
            and measurement.get("tokenizer_revision")
            == FROZEN_STUDENT_REVISION
            and measurement.get("load_in_4bit") is True
            and measurement.get("measurement_uses_train_pool_only") is True
            and measurement.get("official_test_used") is False
            and measurement.get("validation_used") is False
            and measurement.get("failed_calls_supervised") is False
            and measurement.get(
                "full_fresh_assistant_suffix_supervised"
            )
            is True
            and _sha256_hex(
                measurement.get("frozen_checkpoint_identity_sha256")
            )
            and _sha256_hex(
                measurement.get("raw_candidate_record_sha256")
            )
            and _sha256_hex(measurement.get("source_pool_sha256"))
            and isinstance(tool_schemas, list)
            and bool(tool_schemas)
            and all(
                isinstance(schema, Mapping) for schema in tool_schemas
            )
            and measurement.get("tool_schemas_sha256")
            == sha256(tool_schemas)
            and measurement.get("training_system_message_sha256")
            == shared.get("training_system_message_sha256")
        ),
        "measurement_contract_sha256_recomputed": (
            isinstance(measurement, Mapping)
            and _sha256_hex(
                measurement.get("measurement_contract_sha256")
            )
            and measurement.get("measurement_contract_sha256")
            == computed_measurement_contract_sha256
        ),
    }
    if is_v610:
        top_checks.update(
            {
                "v610_design_protocol_registry_and_record_bound": (
                    record_design_protocol
                    == V6_10_PIPELINE_CLOSURE_PROTOCOL
                    and registry_design_protocol
                    == V6_10_PIPELINE_CLOSURE_PROTOCOL
                ),
                "v610_registry_protocol_bound": (
                    isinstance(registry_root, Mapping)
                    and registry_root.get("protocol")
                    == V6_10_REGISTRY_PROTOCOL
                ),
                "v610_positive_clean_suffix_has_no_tool_errors": (
                    clean_error_indices == []
                ),
                "v610_positive_clean_has_zero_tool_error_targets": (
                    clean_error_targets == []
                ),
                "v610_clean_positive_is_sanitized_plan_bound": (
                    isinstance(record_clean_view, Mapping)
                    and record_clean_view.get(
                        "sanitized_reference_plan_sha256"
                    )
                    == v610_reference_binding.get(
                        "sanitized_reference_plan_sha256"
                    )
                    and isinstance(record.get("clean_trace"), Mapping)
                    and record["clean_trace"].get(
                        "sanitized_reference_plan_sha256"
                    )
                    == v610_reference_binding.get(
                        "sanitized_reference_plan_sha256"
                    )
                ),
                "v610_clean_positive_has_full_official_success": (
                    isinstance(record.get("clean_trace"), Mapping)
                    and _official_reward_matches(
                        record["clean_trace"].get(
                            "official_reward_info"
                        ),
                        record["clean_trace"].get(
                            "official_task_success"
                        ),
                        require_success=True,
                    )
                ),
                "v610_frozen_dynamic_judge_bound": (
                    isinstance(generation_contract.get("judge_model"), str)
                    and bool(generation_contract["judge_model"])
                    and isinstance(
                        generation_contract.get("judge_revision"), str
                    )
                    and bool(generation_contract["judge_revision"])
                ),
                "v610_no_future_leakage_or_official_test": (
                    record.get("official_test_used") is False
                    and isinstance(record_clean_view, Mapping)
                    and record_clean_view.get("official_test_used") is False
                    and all(
                        isinstance(branch, Mapping)
                        and branch.get("official_test_used") is False
                        and (
                            branch.get("label_audit") or {}
                        ).get("clean_future_visible")
                        is False
                        for branch in (record_branches or [])
                    )
                ),
                **v610_reference_checks,
                **v610_causal_checks,
            }
        )
    accepted = all(top_checks.values())
    failed_positive_labels = sum(
        int(
            (
                next(
                    (
                        row.get("label_audit", {}).get(
                            "failed_positive_label_count", 1
                        )
                        for row in record_branches
                        if isinstance(row, Mapping)
                        and row.get("branch_id") == branch_id
                    ),
                    1,
                )
            )
        )
        for branch_id in branch_checks
    )
    normalized = {
        **{
            key: deepcopy(value)
            for key, value in registered.items()
            if key not in {"branches", "candidate_pair_sha256"}
        },
        "registered_candidate_pair_sha256": registered.get(
            "candidate_pair_sha256"
        ),
        "design_protocol": (
            V6_10_PIPELINE_CLOSURE_PROTOCOL
            if is_v610
            else protocol.PROTOCOL
        ),
        "generation_contract": deepcopy(record.get("generation_contract")),
        "generation_contract_sha256": record.get(
            "generation_contract_sha256"
        ),
        "reference_plan_binding": deepcopy(
            record.get("reference_plan_binding")
        ),
        "v610_reference_binding": deepcopy(v610_reference_binding),
        "continuation_role_separation": deepcopy(
            record.get("continuation_role_separation")
        ),
        "prefix_sha256": shared.get("shared_prefix_sha256"),
        "environment_snapshot_sha256": shared.get(
            "environment_snapshot_sha256"
        ),
        "user_db_hash_available": shared.get(
            "user_db_hash_available"
        ),
        "branches": branch_rows,
        "forced_first_cells": deepcopy(forced_cells),
        "forced_first_q": deepcopy(forced_cells),
        "first_action_logprobs": deepcopy(
            record.get("first_action_logprobs")
        ),
        "tool_schemas": deepcopy(record.get("tool_schemas")),
        "token_measurement_provenance": deepcopy(
            record.get("token_measurement_provenance")
        ),
        "kappa": kappa,
        "c_sup": c_sup,
        "c_nonpad": c_nonpad,
        "cost": {"c_sup": c_sup, "c_nonpad": c_nonpad},
        "shared_prefix": deepcopy(shared.get("shared_prefix")),
        "training_system_message": deepcopy(
            shared.get("training_system_message")
        ),
        "training_system_message_sha256": shared.get(
            "training_system_message_sha256"
        ),
        "clean_trace": deepcopy(shared.get("clean_trace")),
        "clean_view": {
            "clean_id": f"{registered.get('task_identity')}:clean",
            "messages": deepcopy(
                (record.get("clean_view") or {}).get("messages")
            ),
            "raw_messages": deepcopy(
                (record.get("clean_view") or {}).get("raw_messages")
            ),
            "label_mask": deepcopy(
                (record.get("clean_view") or {}).get("label_mask")
            ),
            "tool_schemas": deepcopy(
                (record.get("clean_view") or {}).get(
                    "tool_schemas", record.get("tool_schemas")
                )
            ),
            "official_task_success": (
                (shared.get("clean_trace") or {}).get(
                    "official_task_success"
                )
            ),
            "source_trajectory_sha256": (
                (shared.get("clean_trace") or {}).get("trace_sha256")
            ),
            "c_sup": (
                (record.get("clean_view") or {}).get("c_sup")
            ),
            "c_nonpad": (
                (record.get("clean_view") or {}).get("c_nonpad")
            ),
            "token_contract": deepcopy(
                (record.get("clean_view") or {}).get("token_contract")
            ),
            "token_measurement_status": (
                (record.get("clean_view") or {}).get(
                    "token_measurement_status"
                )
            ),
            "producer_label_mask": deepcopy(
                (record.get("clean_view") or {}).get(
                    "producer_label_mask"
                )
            ),
            "official_test_used": (
                (record.get("clean_view") or {}).get(
                    "official_test_used"
                )
            ),
            "sanitized_reference_plan_sha256": (
                (record.get("clean_view") or {}).get(
                    "sanitized_reference_plan_sha256"
                )
            ),
        },
        "quality": {
            "real_error_executed": (
                len(branch_rows) == 2
                and all(row.get("actual_tool_error") is True for row in branch_rows)
            ),
            "matched_recovery_replay_success": (
                len(branch_rows) == 2
                and all(
                    row.get("matched_task_success") is True
                    for row in branch_rows
                )
            ),
            "cross_replay_complete": forced_cells_valid,
            "no_future_leakage": (
                len(branch_checks) == 2
                and all(
                    checks.get("clean_future_leakage_zero") is True
                    for checks in branch_checks.values()
                )
            ),
            "independent_replay_audited": (
                len(branch_checks) == 2
                and all(
                    checks.get("matched_reward_and_independent_replay")
                    is True
                    for checks in branch_checks.values()
                )
                and forced_cells_valid
            ),
            "failed_positive_labels": failed_positive_labels,
            "official_test_used": False,
        },
        "source_evidence_sha256": sha256(record),
    }
    protocol_checks = (
        protocol.audit_candidate_pair(normalized) if accepted else {}
    )
    if accepted and not all(protocol_checks.values()):
        accepted = False
        top_checks["frozen_protocol_candidate_pair_audit"] = False
    else:
        top_checks["frozen_protocol_candidate_pair_audit"] = accepted
    normalized["audit_status"] = "ACCEPTED" if accepted else "REJECTED"
    normalized["audit_checks"] = {
        "top": top_checks,
        "shared": shared_checks,
        "branches": branch_checks,
        "frozen_protocol": protocol_checks,
    }
    normalized["kappa_bucket"] = (
        "UNAVAILABLE"
        if kappa is None
        else "LOW"
        if kappa <= protocol.KAPPA_LOW_MAX
        else "HIGH"
        if kappa >= protocol.KAPPA_HIGH_MIN
        else "MIDDLE"
    )
    normalized["audited_candidate_pair_sha256"] = sha256(normalized)
    return normalized


def _choice_sets(
    *,
    registry: Mapping[str, Any],
    accepted: list[dict[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        grouped[str(row["choice_set_id"])].append(row)
    registered_choices = {
        row["choice_set_id"]: row
        for row in registry.get("choice_sets", [])
        if row.get("phase") == phase
    }
    result: list[dict[str, Any]] = []
    for choice_id, rows in sorted(grouped.items()):
        registered = registered_choices.get(choice_id)
        if not isinstance(registered, Mapping):
            continue
        rows.sort(key=lambda row: str(row["candidate_pair_id"]))
        choice = {
            "choice_set_id": choice_id,
            "task_identity": registered["task_identity"],
            "domain": registered["domain"],
            "prefix_sha256": rows[0]["prefix_sha256"],
            "environment_snapshot_sha256": rows[0][
                "environment_snapshot_sha256"
            ],
            "candidate_pairs": rows,
        }
        choice["audit_checks"] = protocol.audit_choice_set(choice)
        result.append(choice)
    return result


def _v610_phase_spec(phase: str) -> dict[str, Any]:
    specs = {
        "compatibility": {
            "task_ids": closure_protocol.COMPATIBILITY_TASK_IDS,
            "task_ids_sha256": (
                closure_protocol.COMPATIBILITY_TASK_IDS_SHA256
            ),
            "gate": {
                "minimum_tasks_with_three_accepted_pairs": (
                    V6_10_COMPATIBILITY_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": (
                    V6_10_COMPATIBILITY_MIN_ACCEPTED_PAIRS
                ),
            },
        },
        "pilot": {
            "task_ids": closure_protocol.PROSPECTIVE_PILOT_TASK_IDS,
            "task_ids_sha256": (
                closure_protocol.PROSPECTIVE_PILOT_TASK_IDS_SHA256
            ),
            "gate": {
                "minimum_tasks_with_three_accepted_pairs": (
                    closure_protocol.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": (
                    closure_protocol.PROSPECTIVE_MIN_ACCEPTED_PAIRS
                ),
                "minimum_domains": protocol.MIN_DOMAINS,
                "minimum_error_families": protocol.MIN_ERROR_FAMILIES,
            },
        },
        "formal": {
            "task_ids": closure_protocol.FORMAL_TASK_IDS,
            "task_ids_sha256": closure_protocol.FORMAL_TASK_IDS_SHA256,
            "gate": {
                "minimum_tasks_with_three_accepted_pairs": (
                    protocol.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": (
                    protocol.FORMAL_MIN_ACCEPTED_PAIRS
                ),
                "screen_minimum_tasks": (
                    protocol.SCREEN_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_domains": protocol.MIN_DOMAINS,
                "minimum_error_families": protocol.MIN_ERROR_FAMILIES,
            },
        },
    }
    try:
        return specs[phase]
    except KeyError as error:
        raise V6CandidateAuditError(
            f"V6.10 phase must be compatibility, pilot, or formal: {phase}"
        ) from error


def _v610_phase_population_checks(
    registry: Mapping[str, Any],
    phase: str,
) -> dict[str, bool]:
    spec = _v610_phase_spec(phase)
    expected_tasks = list(spec["task_ids"])
    phase_registry = registry.get("phase_registry")
    phase_row = (
        phase_registry.get(phase)
        if isinstance(phase_registry, Mapping)
        else None
    )
    gates = registry.get("gates")
    gate_row = (
        gates.get(phase) if isinstance(gates, Mapping) else None
    )
    return {
        f"registered_population_is_frozen_{len(expected_tasks)}": (
            isinstance(phase_row, Mapping)
            and phase_row.get("task_ids") == expected_tasks
            and phase_row.get("task_ids_sha256")
            == spec["task_ids_sha256"]
            and phase_row.get("task_ids_sha256")
            == sha256(expected_tasks)
            and phase_row.get("task_count") == len(expected_tasks)
            and phase_row.get("all_tasks_require_terminal_receipts")
            is True
        ),
        "phase_gate_thresholds_match_frozen_v610_contract": (
            isinstance(gate_row, Mapping)
            and dict(gate_row) == spec["gate"]
        ),
    }


def _v610_gate(
    *,
    registry: Mapping[str, Any],
    choice_sets: list[dict[str, Any]],
    phase: str,
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
    """Apply V6.10's three disjoint, preregistered phase gates."""
    spec = _v610_phase_spec(phase)
    expected_tasks = set(spec["task_ids"])
    summary = protocol._accepted_pool_summary(choice_sets)
    population_checks = _v610_phase_population_checks(registry, phase)
    pool_tasks = set(summary["pairs_by_task"])
    common_checks = {
        **population_checks,
        "accepted_tasks_belong_to_registered_phase_population": (
            pool_tasks <= expected_tasks
        ),
        "unique_candidate_pair_ids": summary[
            "unique_candidate_pair_ids"
        ],
    }
    if phase == "compatibility":
        checks = {
            **common_checks,
            "at_least_22_tasks_with_three_pairs": (
                summary["tasks_with_at_least_three_pairs"]
                >= V6_10_COMPATIBILITY_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "at_least_66_accepted_pairs": (
                summary["accepted_candidate_pairs"]
                >= V6_10_COMPATIBILITY_MIN_ACCEPTED_PAIRS
            ),
        }
        return {
            "protocol": V6_10_PIPELINE_CLOSURE_PROTOCOL,
            "phase": "compatibility",
            "role": "engineering_authorization_only",
            "status": (
                "COMPATIBILITY_RELEASE_AUTHORIZED"
                if all(checks.values())
                else "STOP_COMPATIBILITY_NO_GO"
            ),
            "checks": checks,
            "pool": summary,
            "identifiability": {
                "status": "NOT_APPLIED_ENGINEERING_GATE"
            },
            "official_test_used": False,
        }

    identification = protocol.identifiability_gate(
        summary["kappas"],
        teacher_action_switch_accuracy=teacher_action_switch_accuracy,
        error_blind_action_switch_accuracy=(
            error_blind_action_switch_accuracy
        ),
    )
    common_scientific_checks = {
        **common_checks,
        "both_domains": len(summary["domains"]) >= protocol.MIN_DOMAINS,
        "at_least_three_error_families": (
            len(summary["error_families"])
            >= protocol.MIN_ERROR_FAMILIES
        ),
        "identifiability": identification["status"] == "PASS",
    }
    if phase == "pilot":
        checks = {
            **common_scientific_checks,
            "at_least_16_tasks_with_three_pairs": (
                summary["tasks_with_at_least_three_pairs"]
                >= closure_protocol.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "at_least_48_accepted_pairs": (
                summary["accepted_candidate_pairs"]
                >= closure_protocol.PROSPECTIVE_MIN_ACCEPTED_PAIRS
            ),
        }
        status = "GO_FORMAL_POOL" if all(checks.values()) else "STOP_NO_GO"
    else:
        tasks = summary["tasks_with_at_least_three_pairs"]
        pairs = summary["pairs_from_qualifying_tasks"]
        formal_counts = (
            tasks >= protocol.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
            and pairs >= protocol.FORMAL_MIN_ACCEPTED_PAIRS
        )
        screen_counts = (
            protocol.SCREEN_MIN_TASKS_WITH_THREE_PAIRS
            <= tasks
            < protocol.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
        )
        checks = {
            **common_scientific_checks,
            "at_least_48_tasks_with_three_pairs": (
                tasks >= protocol.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
            ),
            "at_least_144_pairs_from_qualifying_tasks": (
                pairs >= protocol.FORMAL_MIN_ACCEPTED_PAIRS
            ),
            "screen_band_40_to_47_tasks": screen_counts,
        }
        if all(common_scientific_checks.values()) and formal_counts:
            status = "FORMAL_SELECTION_AUTHORIZED"
        elif all(common_scientific_checks.values()) and screen_counts:
            status = "SCREEN_ONLY"
        else:
            status = "STOP_INSUFFICIENT_POOL"
    return {
        "protocol": V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "phase": phase,
        "status": status,
        "checks": checks,
        "pool": summary,
        "identifiability": identification,
        "official_test_used": False,
    }


def _gate(
    *,
    registry: Mapping[str, Any],
    choice_sets: list[dict[str, Any]],
    phase: str,
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
    if (
        registry.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        return _v610_gate(
            registry=registry,
            choice_sets=choice_sets,
            phase=phase,
            teacher_action_switch_accuracy=teacher_action_switch_accuracy,
            error_blind_action_switch_accuracy=(
                error_blind_action_switch_accuracy
            ),
        )
    if phase == "pilot":
        return protocol.pilot_gate(
            choice_sets,
            registered_task_ids=registry["phase_registry"]["pilot"][
                "task_ids"
            ],
            teacher_action_switch_accuracy=teacher_action_switch_accuracy,
            error_blind_action_switch_accuracy=(
                error_blind_action_switch_accuracy
            ),
        )
    return protocol.formal_pool_gate(
        choice_sets,
        teacher_action_switch_accuracy=teacher_action_switch_accuracy,
        error_blind_action_switch_accuracy=error_blind_action_switch_accuracy,
    )


def _action_switch_diagnostics(
    accepted: Iterable[Mapping[str, Any]],
) -> tuple[float, float, int]:
    """Recompute teacher sensitivity from the first, unretried action.

    Each candidate pair contributes two balanced error contexts with two
    distinct registered corrective actions.  An error-blind constant-action
    policy can therefore match exactly one of the two contexts (0.5).
    """
    values: list[bool] = []
    for pair in accepted:
        branches = pair.get("branches")
        if not isinstance(branches, list) or len(branches) != 2:
            continue
        is_v610 = (
            pair.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
        )
        for branch in branches:
            if not isinstance(branch, Mapping):
                continue
            if (
                is_v610
                and branch.get(
                    "teacher_first_attempt_measurement_source"
                )
                != "teacher_unforced_first_action_measurement"
            ):
                continue
            values.append(
                branch.get("teacher_first_attempt_action_matched")
                is True
            )
    teacher = (
        sum(int(value) for value in values) / len(values)
        if values
        else 0.0
    )
    return teacher, 0.5, len(values)


def _v610_generation_closure_audit(
    *,
    registry: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    phase: str,
    generation_closure: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Prove every task executed atomically and has a scientific outcome."""
    closure = (
        generation_closure
        if isinstance(generation_closure, Mapping)
        else {}
    )
    generation = closure.get("generation_receipt")
    generation = generation if isinstance(generation, Mapping) else {}
    ledger = closure.get("failure_ledger")
    ledger_rows = (
        ledger
        if isinstance(ledger, list)
        and all(isinstance(row, Mapping) for row in ledger)
        else []
    )
    task_receipts_value = closure.get("task_receipts")
    task_receipts = (
        task_receipts_value
        if isinstance(task_receipts_value, list)
        and all(
            isinstance(row, Mapping) for row in task_receipts_value
        )
        else []
    )
    expected_tasks = list(_v610_phase_spec(phase)["task_ids"])
    expected_task_set = set(expected_tasks)
    receipt_identities = [
        str(row.get("task_identity")) for row in task_receipts
    ]
    receipts_by_task = {
        str(row.get("task_identity")): row for row in task_receipts
    }
    receipt_hashes = {
        task: row.get("task_receipt_sha256")
        for task, row in receipts_by_task.items()
    }
    registered_by_task: dict[str, set[str]] = defaultdict(set)
    for row in registry.get("candidate_pairs", []):
        if isinstance(row, Mapping) and row.get("phase") == phase:
            registered_by_task[str(row.get("task_identity"))].add(
                str(row.get("candidate_pair_id"))
            )
    submitted_by_task: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        if isinstance(row, Mapping) and row.get("phase") == phase:
            submitted_by_task[str(row.get("task_identity"))].add(
                str(row.get("candidate_pair_id"))
            )
    pass_tasks = {
        task
        for task, row in receipts_by_task.items()
        if row.get("scientific_outcome") == "ACCEPTED"
    }
    rejected_tasks = {
        task
        for task, row in receipts_by_task.items()
        if row.get("scientific_outcome") == "REJECTED"
    }
    task_hashes_valid = all(
        isinstance(row.get("task_receipt_sha256"), str)
        and row.get("task_receipt_sha256")
        == sha256(
            {
                key: value
                for key, value in row.items()
                if key != "task_receipt_sha256"
            }
        )
        for row in task_receipts
    )
    task_contracts_valid = all(
        row.get("protocol") == GENERATION_PROTOCOL
        and row.get("status") == "PASS"
        and row.get("execution_status") == "PASS"
        and row.get("scientific_outcome")
        in {"ACCEPTED", "REJECTED"}
        and row.get("registry_sha256") == registry.get("registry_sha256")
        and row.get("official_test_used") is False
        and isinstance(
            row.get("semantic_generation_contract"), Mapping
        )
        and row.get("semantic_generation_contract_sha256")
        == sha256(row.get("semantic_generation_contract"))
        and isinstance(row.get("runtime_provenance"), Mapping)
        and row.get("runtime_provenance")
        == row["semantic_generation_contract"].get(
            "runtime_provenance", {}
        )
        and row.get("attempt_id")
        == row["runtime_provenance"].get("attempt_id")
        for row in task_receipts
    )
    pass_receipts_valid = True
    rejected_receipts_valid = True
    generated_pairs_by_id: dict[str, Mapping[str, Any]] = {}
    generated_pair_ids_unique = True
    for task, row in receipts_by_task.items():
        expected_pair_ids = registered_by_task.get(task, set())
        pairs = row.get("candidate_pairs")
        pairs_list = (
            pairs
            if isinstance(pairs, list)
            and all(isinstance(pair, Mapping) for pair in pairs)
            else []
        )
        observed_pair_ids = {
            str(pair.get("candidate_pair_id")) for pair in pairs_list
        }
        if row.get("scientific_outcome") == "ACCEPTED":
            for pair in pairs_list:
                pair_id = str(pair.get("candidate_pair_id"))
                if pair_id in generated_pairs_by_id:
                    generated_pair_ids_unique = False
                generated_pairs_by_id[pair_id] = pair
            pass_receipts_valid = pass_receipts_valid and (
                bool(expected_pair_ids)
                and len(observed_pair_ids) == len(pairs_list)
                and observed_pair_ids == expected_pair_ids
                and row.get("candidate_pair_count") == len(pairs_list)
                and all(
                    pair.get("protocol") == GENERATION_PROTOCOL
                    and pair.get("official_test_used") is False
                    and pair.get("task_identity") == task
                    and pair.get("registry_sha256")
                    == registry.get("registry_sha256")
                    and pair.get("generation_contract")
                    == row.get("semantic_generation_contract")
                    and pair.get("generation_contract_sha256")
                    == row.get(
                        "semantic_generation_contract_sha256"
                    )
                    and pair.get("generated_candidate_pair_sha256")
                    == sha256(
                        {
                            key: value
                            for key, value in pair.items()
                            if key
                            != "generated_candidate_pair_sha256"
                        }
                    )
                    for pair in pairs_list
                )
            )
        elif row.get("scientific_outcome") == "REJECTED":
            rejected_receipts_valid = rejected_receipts_valid and (
                row.get("reason_code")
                in V6_10_TYPED_TASK_REJECTION_CODES
                and isinstance(row.get("error_type"), str)
                and bool(row.get("error_type"))
                and isinstance(row.get("error"), str)
                and bool(row.get("error"))
                and row.get("expected_candidate_pair_ids")
                == sorted(expected_pair_ids)
                and pairs == []
                and row.get("candidate_pair_count") == 0
                and row.get("training_started") is False
            )
        else:
            pass_receipts_valid = False
            rejected_receipts_valid = False
    ledger_by_task = {
        str(row.get("task_identity")): row for row in ledger_rows
    }
    ledger_valid = (
        len(ledger_by_task) == len(ledger_rows)
        and set(ledger_by_task) == rejected_tasks
        and all(
            ledger_by_task[task]
            == {
                "task_identity": task,
                "reason_code": receipt.get("reason_code"),
                "error_type": receipt.get("error_type"),
                "error": receipt.get("error"),
                "task_receipt_sha256": receipt.get(
                    "task_receipt_sha256"
                ),
                "attempt_id": receipt.get("attempt_id"),
                "official_test_used": False,
            }
            for task, receipt in receipts_by_task.items()
            if task in rejected_tasks
        )
    )
    generation_contract_hashes = {
        row.get("semantic_generation_contract_sha256")
        for row in task_receipts
    }
    run_contract_hashes = {
        row.get("run_contract_sha256") for row in task_receipts
    }
    attempt_ids = {row.get("attempt_id") for row in task_receipts}
    expected_submitted_ids = set().union(
        *(registered_by_task.get(task, set()) for task in pass_tasks)
    )
    submitted_ids = {
        str(row.get("candidate_pair_id")) for row in candidates
    }
    submitted_content_bound = (
        generated_pair_ids_unique
        and submitted_ids == set(generated_pairs_by_id)
        and all(
            isinstance(row, Mapping)
            and isinstance(
                generated_pairs_by_id.get(
                    str(row.get("candidate_pair_id"))
                ),
                Mapping,
            )
            and row.get("generated_candidate_pair_sha256")
            == generated_pairs_by_id[
                str(row.get("candidate_pair_id"))
            ].get("generated_candidate_pair_sha256")
            and isinstance(
                row.get("token_measurement_provenance"), Mapping
            )
            and row["token_measurement_provenance"].get(
                "raw_candidate_record_sha256"
            )
            == sha256(
                generated_pairs_by_id[
                    str(row.get("candidate_pair_id"))
                ]
            )
            and row.get("registry_sha256")
            == registry.get("registry_sha256")
            and isinstance(row.get("generation_contract"), Mapping)
            and row.get("generation_contract")
            == generated_pairs_by_id[
                str(row.get("candidate_pair_id"))
            ].get("generation_contract")
            and row.get("generation_contract_sha256")
            == generated_pairs_by_id[
                str(row.get("candidate_pair_id"))
            ].get("generation_contract_sha256")
            and isinstance(
                row["generation_contract"].get("runtime_provenance"),
                Mapping,
            )
            and row["generation_contract"]["runtime_provenance"].get(
                "attempt_id"
            )
            == generated_pairs_by_id[
                str(row.get("candidate_pair_id"))
            ].get("generation_contract", {}).get(
                "runtime_provenance", {}
            ).get("attempt_id")
            for row in candidates
        )
    )
    closure_hashes_valid = (
        closure.get("generation_receipt_sha256")
        == sha256(generation)
        and closure.get("failure_ledger_sha256")
        == sha256(ledger_rows)
        and closure.get("task_receipts_sha256")
        == sha256(dict(sorted(receipt_hashes.items())))
    )
    generation_valid = (
        generation.get("protocol") == GENERATION_PROTOCOL
        and generation.get("status") == "PASS"
        and generation.get("execution_status") == "PASS"
        and generation.get("phase") == phase
        and generation.get("expected_tasks") == len(expected_tasks)
        and generation.get("completed_tasks") == len(expected_tasks)
        and generation.get("missing_tasks") == []
        and generation.get("passing_tasks") == len(pass_tasks)
        and generation.get("rejected_tasks") == len(rejected_tasks)
        and generation.get("accepted_candidate_pairs")
        == len(expected_submitted_ids)
        and generation.get("expected_candidate_pairs")
        == len(expected_submitted_ids)
        and generation.get("task_receipt_sha256_by_task")
        == dict(sorted(receipt_hashes.items()))
        and generation.get("task_receipts_sha256")
        == sha256(dict(sorted(receipt_hashes.items())))
        and generation.get("registry_sha256")
        == registry.get("registry_sha256")
        and len(generation_contract_hashes) == 1
        and all(_sha256_hex(value) for value in generation_contract_hashes)
        and generation.get("semantic_generation_contract_sha256")
        in generation_contract_hashes
        and len(run_contract_hashes) == 1
        and all(_sha256_hex(value) for value in run_contract_hashes)
        and generation.get("run_contract_sha256")
        in run_contract_hashes
        and len(attempt_ids) == 1
        and all(
            isinstance(value, str) and bool(value)
            for value in attempt_ids
        )
        and generation.get("attempt_id") in attempt_ids
        and generation.get("training_started") is False
        and generation.get("official_test_used") is False
    )
    checks = {
        "generation_closure_bundle_present": bool(closure),
        "closure_semantic_hashes_recomputed": closure_hashes_valid,
        "exact_phase_task_receipt_population": (
            len(receipt_identities) == len(set(receipt_identities))
            and set(receipt_identities) == expected_task_set
        ),
        "all_task_receipt_hashes_valid": task_hashes_valid,
        "all_task_receipts_share_frozen_run_contract": (
            task_contracts_valid
            and len(generation_contract_hashes) == 1
            and len(run_contract_hashes) == 1
            and len(attempt_ids) == 1
        ),
        "passing_tasks_have_all_registered_pairs": pass_receipts_valid,
        "rejected_tasks_are_exact_typed_empty_receipts": (
            rejected_receipts_valid
        ),
        "failure_ledger_exactly_matches_typed_rejections": ledger_valid,
        "submitted_pairs_exactly_partition_to_passing_tasks": (
            submitted_ids == expected_submitted_ids
            and all(
                not submitted_by_task.get(task)
                for task in rejected_tasks
            )
        ),
        "submitted_candidate_content_bound_to_pass_receipts": (
            submitted_content_bound
        ),
        "generation_receipt_recomputes_terminal_closure": generation_valid,
    }
    normalized = {
        "generation_receipt_sha256": closure.get(
            "generation_receipt_sha256"
        ),
        "failure_ledger_sha256": closure.get("failure_ledger_sha256"),
        "task_receipts_sha256": closure.get("task_receipts_sha256"),
        "expected_tasks": len(expected_tasks),
        "passing_tasks": len(pass_tasks),
        "rejected_tasks": len(rejected_tasks),
        "accepted_candidate_pairs": len(expected_submitted_ids),
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
    }
    normalized["generation_closure_audit_sha256"] = sha256(
        {"normalized": normalized, "checks": checks}
    )
    return normalized, checks


def audit_candidates(
    *,
    registry: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    phase: str,
    generation_closure: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    registry_hash = registry.get("registry_sha256")
    registry_design_protocol = registry.get("design_protocol")
    allowed_phases = (
        {"compatibility", "pilot", "formal"}
        if registry_design_protocol
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
        else {"pilot", "formal"}
    )
    if phase not in allowed_phases:
        raise V6CandidateAuditError(
            "phase must be "
            + ", ".join(sorted(allowed_phases))
        )
    expected_registry_protocol = (
        V6_10_REGISTRY_PROTOCOL
        if registry_design_protocol
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
        else registry_protocol.REGISTRY_PROTOCOL
    )
    if (
        registry.get("protocol")
        != expected_registry_protocol
        or not _nonempty_hash(registry_hash)
        or sha256(
            {
                key: value
                for key, value in registry.items()
                if key != "registry_sha256"
            }
        )
        != registry_hash
        or registry.get("official_test_used") is not False
        or registry.get("official_test_sealed") is not True
    ):
        raise V6CandidateAuditError("registry identity or official-test seal failed")
    registered = {
        row["candidate_pair_id"]: row
        for row in registry.get("candidate_pairs", [])
        if row.get("phase") == phase
    }
    if not registered:
        raise V6CandidateAuditError(f"registry has no {phase} candidate pairs")
    rows = list(candidates)
    generation_closure_audit: dict[str, Any] = {}
    generation_closure_checks: dict[str, bool] = {}
    if (
        registry_design_protocol
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        (
            generation_closure_audit,
            generation_closure_checks,
        ) = _v610_generation_closure_audit(
            registry=registry,
            candidates=rows,
            phase=phase,
            generation_closure=generation_closure,
        )
    ids = [row.get("candidate_pair_id") for row in rows]
    duplicates = sorted(
        key for key, count in Counter(ids).items() if count > 1
    )
    unknown = sorted(
        str(value) for value in set(ids) if value not in registered
    )
    wrong_phase = sorted(
        str(row.get("candidate_pair_id"))
        for row in rows
        if row.get("phase") != phase
    )
    audited: list[dict[str, Any]] = []
    for row in rows:
        pair_id = row.get("candidate_pair_id")
        if pair_id not in registered or row.get("phase") != phase:
            continue
        audited.append(
            audit_candidate_pair(
                row,
                registered[pair_id],
                registry_sha256=str(registry_hash),
                registry_root=registry,
            )
        )
    accepted = [
        row for row in audited if row["audit_status"] == "ACCEPTED"
    ]
    rejected = [
        row for row in audited if row["audit_status"] != "ACCEPTED"
    ]
    choice_sets = _choice_sets(
        registry=registry,
        accepted=accepted,
        phase=phase,
    )
    teacher_action_switch_accuracy, error_blind_action_switch_accuracy, switch_n = (
        _action_switch_diagnostics(accepted)
    )
    gate = _gate(
        registry=registry,
        choice_sets=choice_sets,
        phase=phase,
        teacher_action_switch_accuracy=teacher_action_switch_accuracy,
        error_blind_action_switch_accuracy=error_blind_action_switch_accuracy,
    )
    global_checks = {
        "registry_hash_valid": True,
        "official_test_sealed": True,
        "no_duplicate_candidate_pair_ids": not duplicates,
        "no_unknown_candidate_pair_ids": not unknown,
        "all_rows_match_requested_phase": not wrong_phase,
        "official_test_never_used": all(
            row.get("official_test_used") is False for row in rows
        ),
    }
    if (
        registry_design_protocol
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        global_checks["v610_phase_population_registry_bound"] = all(
            _v610_phase_population_checks(registry, phase).values()
        )
        global_checks[
            "v610_generation_closure_terminal_and_bound"
        ] = (
            bool(generation_closure_checks)
            and all(generation_closure_checks.values())
        )
    if not all(global_checks.values()):
        gate = deepcopy(gate)
        gate["status"] = {
            "compatibility": "STOP_COMPATIBILITY_NO_GO",
            "pilot": "STOP_NO_GO",
            "formal": "STOP_INSUFFICIENT_POOL",
        }[phase]
        gate["checks"]["global_registry_and_seal_checks"] = False
    missing = sorted(set(registered) - set(ids))
    bucket_counts = Counter(
        str(row["kappa_bucket"]) for row in accepted
    )
    report: dict[str, Any] = {
        "protocol": AUDIT_PROTOCOL,
        "audit_version": AUDIT_VERSION,
        "design_protocol": registry_design_protocol,
        "phase": phase,
        "registry_sha256": registry_hash,
        "official_test_used": False,
        "official_test_sealed": True,
        "global_checks": global_checks,
        "generation_closure_audit": deepcopy(
            generation_closure_audit
        ),
        "generation_closure_checks": deepcopy(
            generation_closure_checks
        ),
        "registered_candidate_pairs": len(registered),
        "submitted_candidate_pairs": len(rows),
        "audited_candidate_pairs": len(audited),
        "accepted_candidate_pairs": len(accepted),
        "rejected_candidate_pairs": len(rejected),
        "missing_candidate_pair_ids": missing,
        "unknown_candidate_pair_ids": unknown,
        "duplicate_candidate_pair_ids": duplicates,
        "wrong_phase_candidate_pair_ids": wrong_phase,
        "accepted_choice_sets": len(choice_sets),
        "action_switch_diagnostic": {
            "definition": (
                "first unretried teacher action matched the registered "
                "error-specific corrective action"
            ),
            "teacher_action_switch_accuracy": (
                teacher_action_switch_accuracy
            ),
            "error_blind_balanced_pair_accuracy": (
                error_blind_action_switch_accuracy
            ),
            "branch_contexts": switch_n,
            "computed_from_audited_evidence": True,
        },
        "kappa_bucket_counts": dict(sorted(bucket_counts.items())),
        "low_kappa_retention_policy": (
            "accepted low-kappa pairs are retained as selector controls"
        ),
        "rejected": [
            {
                "candidate_pair_id": row["candidate_pair_id"],
                "failed_checks": sorted(
                    [
                        f"top.{name}"
                        for name, passed in row["audit_checks"]["top"].items()
                        if not passed
                    ]
                    + [
                        f"shared.{name}"
                        for name, passed in row["audit_checks"]["shared"].items()
                        if not passed
                    ]
                    + [
                        f"branch.{branch_id}.{name}"
                        for branch_id, checks in row["audit_checks"][
                            "branches"
                        ].items()
                        for name, passed in checks.items()
                        if not passed
                    ]
                ),
            }
            for row in rejected
        ],
        "gate": gate,
        "accepted_candidate_pair_hashes": {
            row["candidate_pair_id"]: row["audited_candidate_pair_sha256"]
            for row in accepted
        },
    }
    report["audit_report_sha256"] = sha256(report)
    freeze_rows = [
        {
            "candidate_pair_id": row["candidate_pair_id"],
            "choice_set_id": row["choice_set_id"],
            "task_identity": row["task_identity"],
            "domain": row["domain"],
            "kappa": row["kappa"],
            "kappa_bucket": row["kappa_bucket"],
            "c_sup": row["c_sup"],
            "c_nonpad": row["c_nonpad"],
            "corrective_families": [
                branch["canonical_corrective_family"]
                for branch in row["branches"]
            ],
            "audited_candidate_pair_sha256": row[
                "audited_candidate_pair_sha256"
            ],
            "source_evidence_sha256": row["source_evidence_sha256"],
        }
        for row in accepted
    ]
    freeze: dict[str, Any] = {
        "protocol": "v6_candidate_pair_freeze_manifest_v1",
        "phase": phase,
        "status": (
            "FROZEN"
            if gate["status"]
            in {
                "COMPATIBILITY_RELEASE_AUTHORIZED",
                "GO_FORMAL_POOL",
                "FORMAL_SELECTION_AUTHORIZED",
                "SCREEN_ONLY",
            }
            and all(global_checks.values())
            else "NOT_FROZEN_FAIL_CLOSED"
        ),
        "selection_authorized": (
            gate["status"] == "FORMAL_SELECTION_AUTHORIZED"
            and all(global_checks.values())
        ),
        "screen_only": gate["status"] == "SCREEN_ONLY",
        "registry_sha256": registry_hash,
        "audit_report_sha256": report["audit_report_sha256"],
        "official_test_used": False,
        "official_test_sealed": True,
        "independent_unit": "task_identity",
        "selection_unit": "candidate_pair",
        "low_kappa_pairs_preserved": [
            row["candidate_pair_id"]
            for row in freeze_rows
            if row["kappa_bucket"] == "LOW"
        ],
        "middle_kappa_pairs_preserved": [
            row["candidate_pair_id"]
            for row in freeze_rows
            if row["kappa_bucket"] == "MIDDLE"
        ],
        "high_kappa_pairs_preserved": [
            row["candidate_pair_id"]
            for row in freeze_rows
            if row["kappa_bucket"] == "HIGH"
        ],
        "candidate_pairs": freeze_rows,
        "candidate_pair_hash_stream_sha256": sha256(
            [
                row["audited_candidate_pair_sha256"]
                for row in freeze_rows
            ]
        ),
        "gate": gate,
    }
    freeze["freeze_manifest_sha256"] = sha256(freeze)
    return report, accepted, freeze


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("compatibility", "pilot", "formal"),
        required=True,
    )
    parser.add_argument("--generation-receipt", type=Path)
    parser.add_argument("--failure-ledger", type=Path)
    parser.add_argument("--task-receipts-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    candidates = read_jsonl(args.candidates)
    closure_paths = (
        args.generation_receipt,
        args.failure_ledger,
        args.task_receipts_dir,
    )
    if any(path is not None for path in closure_paths) and not all(
        path is not None for path in closure_paths
    ):
        raise V6CandidateAuditError(
            "generation receipt, failure ledger, and task receipts "
            "directory must be supplied together"
        )
    if (
        registry.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
        and not all(path is not None for path in closure_paths)
    ):
        raise V6CandidateAuditError(
            "V6.10 audit requires --generation-receipt, "
            "--failure-ledger, and --task-receipts-dir"
        )
    generation_closure = None
    if all(path is not None for path in closure_paths):
        assert args.generation_receipt is not None
        assert args.failure_ledger is not None
        assert args.task_receipts_dir is not None
        generation_receipt = json.loads(
            args.generation_receipt.read_text(encoding="utf-8")
        )
        failure_ledger = read_jsonl(args.failure_ledger)
        task_receipt_paths = sorted(
            args.task_receipts_dir.glob("*.json")
        )
        if not task_receipt_paths:
            raise V6CandidateAuditError(
                "task receipts directory has no JSON receipts"
            )
        task_receipts = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in task_receipt_paths
        ]
        receipt_hashes = {
            str(row.get("task_identity")): row.get(
                "task_receipt_sha256"
            )
            for row in task_receipts
            if isinstance(row, Mapping)
        }
        generation_closure = {
            "generation_receipt": generation_receipt,
            "generation_receipt_sha256": sha256(
                generation_receipt
            ),
            "generation_receipt_file_sha256": sha256_file(
                args.generation_receipt
            ),
            "failure_ledger": failure_ledger,
            "failure_ledger_sha256": sha256(failure_ledger),
            "failure_ledger_file_sha256": sha256_file(
                args.failure_ledger
            ),
            "task_receipts": task_receipts,
            "task_receipts_sha256": sha256(
                dict(sorted(receipt_hashes.items()))
            ),
            "task_receipt_file_sha256": {
                path.name: sha256_file(path)
                for path in task_receipt_paths
            },
        }
    report, accepted, freeze = audit_candidates(
        registry=registry,
        candidates=candidates,
        phase=args.phase,
        generation_closure=generation_closure,
    )
    if args.output_root.exists():
        raise V6CandidateAuditError(
            f"refusing to overwrite audit directory: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "audit_report.json", report)
    write_jsonl(
        args.output_root / "accepted_candidate_pairs.jsonl", accepted
    )
    write_json(args.output_root / "freeze_manifest.json", freeze)
    print(
        json.dumps(
            {
                "status": report["gate"]["status"],
                "accepted_candidate_pairs": len(accepted),
                "freeze_status": freeze["status"],
                "freeze_manifest_sha256": freeze[
                    "freeze_manifest_sha256"
                ],
                "official_test_used": False,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(
        0
        if freeze["status"] == "FROZEN"
        else 2
    )


if __name__ == "__main__":
    main()
