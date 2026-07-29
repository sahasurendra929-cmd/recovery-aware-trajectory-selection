#!/usr/bin/env python3
"""Fail-closed audit for materialized V6 candidate pairs.

The producer is not trusted to declare a pair valid.  This module binds every
result to the structural registry and recomputes:

* full clean/recovery trace and shared-prefix identities;
* the common pre-error environment snapshot;
* exact registered failed calls, real tool errors, and zero failed-call state
  mutation in both agent and user databases;
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


AUDIT_PROTOCOL = "v6_candidate_pair_audit_v1"
AUDIT_VERSION = "1.0"
FROZEN_STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FROZEN_STUDENT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
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
            and _nonempty_hash(snapshot.get("user_db_hash"))
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
            and _nonempty_hash(clean.get("final_user_db_hash"))
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
            and db.get("user_before_error") == expected_user
            and db.get("user_after_error") == expected_user
            and _nonempty_hash(expected_user)
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
        "evidence_sha256": sha256(record),
        "audit_checks": checks,
    }
    return normalized, checks


def audit_candidate_pair(
    record: Mapping[str, Any],
    registered: Mapping[str, Any],
    *,
    registry_sha256: str,
) -> dict[str, Any]:
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
        "prefix_sha256": shared.get("shared_prefix_sha256"),
        "environment_snapshot_sha256": shared.get(
            "environment_snapshot_sha256"
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


def _gate(
    *,
    registry: Mapping[str, Any],
    choice_sets: list[dict[str, Any]],
    phase: str,
    teacher_action_switch_accuracy: float,
    error_blind_action_switch_accuracy: float,
) -> dict[str, Any]:
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
        values.extend(
            branch.get("teacher_first_attempt_action_matched") is True
            for branch in branches
            if isinstance(branch, Mapping)
        )
    teacher = (
        sum(int(value) for value in values) / len(values)
        if values
        else 0.0
    )
    return teacher, 0.5, len(values)


def audit_candidates(
    *,
    registry: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    phase: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if phase not in {"pilot", "formal"}:
        raise V6CandidateAuditError("phase must be pilot or formal")
    registry_hash = registry.get("registry_sha256")
    if (
        registry.get("protocol")
        != registry_protocol.REGISTRY_PROTOCOL
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
    if not all(global_checks.values()):
        gate = deepcopy(gate)
        gate["status"] = (
            "STOP_NO_GO" if phase == "pilot" else "STOP_INSUFFICIENT_POOL"
        )
        gate["checks"]["global_registry_and_seal_checks"] = False
    missing = sorted(set(registered) - set(ids))
    bucket_counts = Counter(
        str(row["kappa_bucket"]) for row in accepted
    )
    report: dict[str, Any] = {
        "protocol": AUDIT_PROTOCOL,
        "audit_version": AUDIT_VERSION,
        "design_protocol": protocol.PROTOCOL,
        "phase": phase,
        "registry_sha256": registry_hash,
        "official_test_used": False,
        "official_test_sealed": True,
        "global_checks": global_checks,
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
            in {"GO_FORMAL_POOL", "FORMAL_SELECTION_AUTHORIZED", "SCREEN_ONLY"}
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
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    candidates = read_jsonl(args.candidates)
    report, accepted, freeze = audit_candidates(
        registry=registry,
        candidates=candidates,
        phase=args.phase,
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
