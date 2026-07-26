#!/usr/bin/env python3
"""Audit whether the frozen V5.2 Stage-1 data pool is structurally attainable.

This program is deliberately separate from training and evaluation.  It reads
only *inner-train generation* attempt records (or a builder audit containing
the same attempt-level records), then answers one question:

    Can the registered number of task-level clean/error pairs still be built
    within the remaining, fixed generation-attempt budget?

V5.2 pairing is task-level and cross-seed.  If a task has two eligible clean
attempts with seeds 1/2 and two eligible error attempts with seeds 7/8, it has
two constructible pairs even though the seed sets do not intersect.  Thus:

    pairs(task) = min(clean eligible, error eligible, max pairs per task)

The future bound is structural, not a forecast that selects positive results.
Every unspent attempt is optimistically allowed to become eligible.  If even
that upper bound misses a gate, the pool is mathematically unreachable.

Exit codes:

* 0  -- the observed pool is already trainable;
* 10 -- not trainable yet, but still structurally reachable within budget;
* 20 -- mathematically unreachable within the frozen budget;
* 2  -- malformed, duplicate, over-budget, or prohibited-split data.

The script uses only the Python standard library.  It never consumes
derived-validation or official-test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence


EXIT_TRAINABLE = 0
EXIT_DATA_ERROR = 2
EXIT_PENDING_REACHABLE = 10
EXIT_MATHEMATICALLY_UNREACHABLE = 20

DEFAULT_MIN_TASKS = 40
DEFAULT_MIN_PAIRS = 48
DEFAULT_MAX_PAIRS_PER_TASK = 2
DEFAULT_MAX_ATTEMPTS = 12
DEFAULT_CONFIDENCE = 0.95

CONDITIONS = ("clean", "error")
ALLOWED_DOMAINS = ("retail", "airline")
ATTEMPT_FIELDS = (
    "attempt_id",
    "trial",
    "rollout_id",
    "rollout_seed",
    "generation_seed",
    "seed",
)
ROW_LIST_FIELDS = (
    "feasibility_rows",
    "attempt_rows",
    "pool_rows",
    "slot_rows",
    "records",
    "rows",
    "simulations",
)
FORBIDDEN_PATH_MARKERS = (
    "derived_validation",
    "derived_val",
    "official_test",
)
FORBIDDEN_SPLIT_MARKERS = (
    "derived_validation",
    "derived_val",
    "official_test",
    "officialtest",
)
ALLOWED_INNER_TRAIN_SPLITS = {
    "arm_train",
    "derived_inner_train",
    "inner_train",
}
INFRASTRUCTURE_REASONS = {
    "context_window_exceeded",
    "infrastructure_error",
    "invalid_simulation_no_messages",
    "invalid_simulation_no_user_message",
    "malformed",
    "no_messages",
    "unexpected_error",
}


class DataError(RuntimeError):
    """Raised when an input cannot be safely used for a feasibility decision."""


@dataclass(frozen=True)
class Attempt:
    domain: str
    task_id: str
    condition: str
    attempt_id: str
    scoreable: bool
    eligible: bool
    reason: str
    content_key: str
    content_hash_verified: bool
    source: str

    @property
    def task_key(self) -> str:
        return f"{self.domain}:{self.task_id}"


@dataclass
class TaskState:
    attempts: dict[str, dict[str, Attempt]] = field(
        default_factory=lambda: {"clean": {}, "error": {}}
    )

    def add(self, attempt: Attempt) -> None:
        side = self.attempts[attempt.condition]
        if attempt.attempt_id in side:
            previous = side[attempt.attempt_id]
            raise DataError(
                "duplicate attempt for "
                f"{attempt.task_key}/{attempt.condition}/{attempt.attempt_id}: "
                f"{previous.source} and {attempt.source}"
            )
        side[attempt.attempt_id] = attempt

    def eligible_ids(self, condition: str) -> set[str]:
        """Return one deterministic attempt ID per unique eligible content."""

        by_content: dict[str, str] = {}
        for attempt_id, attempt in sorted(self.attempts[condition].items()):
            if attempt.eligible:
                by_content.setdefault(attempt.content_key, attempt_id)
        return set(by_content.values())

    def raw_eligible_count(self, condition: str) -> int:
        return sum(
            attempt.eligible for attempt in self.attempts[condition].values()
        )

    def scoreable_count(self, condition: str) -> int:
        return sum(
            attempt.scoreable for attempt in self.attempts[condition].values()
        )


def _normalise_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _guard_path_before_read(path: Path) -> None:
    marker = _normalise_marker(str(path))
    if any(forbidden in marker for forbidden in FORBIDDEN_PATH_MARKERS):
        raise DataError(
            f"refusing prohibited validation/test result path before read: {path}"
        )


def _guard_payload_metadata(value: Any, *, where: str = "$") -> None:
    """Reject explicit validation/test provenance before extracting outcomes."""

    if isinstance(value, list):
        for index, item in enumerate(value):
            _guard_payload_metadata(item, where=f"{where}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for raw_key, item in value.items():
        key = _normalise_marker(str(raw_key))
        if key == "official_test_used" and item is not False:
            raise DataError(f"{where}.{raw_key}: official-test use is prohibited")
        if key in {
            "derived_validation_used",
            "derived_validation_used_for_supervision",
            "validation_outcomes_used",
        } and item is True:
            raise DataError(
                f"{where}.{raw_key}: derived-validation outcomes are prohibited"
            )
        if key in {
            "source_split",
            "data_split",
            "fit_split",
            "evaluation_split",
            "result_split",
            "split_name",
        } and isinstance(item, str):
            split_marker = _normalise_marker(item)
            if (
                split_marker in {"validation", "val", "test"}
                or any(
                    forbidden in split_marker
                    for forbidden in FORBIDDEN_SPLIT_MARKERS
                )
            ):
                raise DataError(
                    f"{where}.{raw_key}: prohibited result split {item!r}"
                )
        _guard_payload_metadata(item, where=f"{where}.{raw_key}")


def _read_json_or_jsonl(path: Path) -> Any:
    _guard_path_before_read(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    if not text.strip():
        raise DataError(f"empty input: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataError(
                    f"{path}:{line_number}: invalid JSONL record: {exc}"
                ) from exc
        payload = rows
    _guard_payload_metadata(payload)
    return payload


def _require_inner_train_provenance(
    payload: Any, rows: Sequence[dict[str, Any]], *, path: Path
) -> None:
    markers: list[str] = []
    if isinstance(payload, dict):
        for key in ("source_split", "data_split", "fit_split", "result_split"):
            value = payload.get(key)
            if isinstance(value, str):
                markers.append(_normalise_marker(value))
        provenance = payload.get("provenance")
        if isinstance(provenance, dict):
            for key in ("source_split", "data_split", "fit_split", "result_split"):
                value = provenance.get(key)
                if isinstance(value, str):
                    markers.append(_normalise_marker(value))
    for row in rows:
        for key in ("source_split", "data_split", "fit_split", "result_split"):
            value = _nested_value(row, key)
            if isinstance(value, str):
                markers.append(_normalise_marker(value))
    if not markers:
        if _has_bound_inner_train_generation_contract(path):
            return
        raise DataError(
            f"{path}: result input lacks explicit inner-train source_split "
            "provenance or a hash-binding COMPLETE generation contract"
        )
    unexpected = sorted(set(markers) - ALLOWED_INNER_TRAIN_SPLITS)
    if unexpected:
        raise DataError(
            f"{path}: result source_split is not frozen inner-train: {unexpected}"
        )


def _has_bound_inner_train_generation_contract(result_path: Path) -> bool:
    """Accept a tau2 result only when a COMPLETE sidecar binds its exact hash."""

    try:
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DataError(f"cannot hash result input {result_path}: {exc}") from exc
    matches = []
    for contract_path in sorted(result_path.parent.glob("run_contract*.json")):
        _guard_path_before_read(contract_path)
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"invalid generation contract {contract_path}: {exc}") from exc
        _guard_payload_metadata(contract)
        if not isinstance(contract, dict):
            raise DataError(f"generation contract is not an object: {contract_path}")
        hashes = contract.get("result_sha256")
        if not isinstance(hashes, dict) or result_path.name not in hashes:
            continue
        if hashes[result_path.name] != result_sha256:
            raise DataError(
                f"{contract_path}: result SHA-256 mismatch for {result_path.name}"
            )
        if (
            contract.get("protocol") != "v5_stage1_inner_train_generation_run"
            or contract.get("status") != "COMPLETE"
            or contract.get("source_split") != "derived_inner_train"
            or contract.get("official_test_used") is not False
        ):
            raise DataError(
                f"{contract_path}: result is not bound by a COMPLETE "
                "inner-train generation contract"
            )
        matches.append(contract_path)
    if len(matches) > 1:
        raise DataError(
            f"{result_path}: multiple generation contracts bind the same result"
        )
    return len(matches) == 1


def _domain_from_filename(path: Path) -> str | None:
    match = re.search(r"(?:^|[^a-z])(retail|airline)(?:[^a-z]|$)", path.name.lower())
    return match.group(1) if match else None


def _condition_from_filename(path: Path) -> str | None:
    name = _normalise_marker(path.name)
    for condition in CONDITIONS:
        if re.search(rf"(?:^|_){condition}(?:_|$)", name):
            return condition
    return None


def _nested_value(row: dict[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for container_name in ("metadata", "analysis", "audit"):
        container = row.get(container_name)
        if isinstance(container, dict) and name in container:
            return container[name]
    return None


def _normalise_domain_task(
    domain: Any,
    task_id: Any,
    *,
    default_domain: str | None = None,
    where: str,
) -> tuple[str, str]:
    if task_id is None:
        raise DataError(f"{where}: missing task_id")
    task_text = str(task_id).strip()
    domain_text = str(domain).strip().lower() if domain is not None else ""
    if ":" in task_text:
        embedded_domain, embedded_task = task_text.split(":", 1)
        embedded_domain = embedded_domain.strip().lower()
        if domain_text and embedded_domain != domain_text:
            raise DataError(f"{where}: domain/task_id domain mismatch")
        domain_text, task_text = embedded_domain, embedded_task.strip()
    if not domain_text:
        domain_text = default_domain or ""
    if domain_text not in ALLOWED_DOMAINS:
        raise DataError(f"{where}: domain must be retail or airline")
    if not task_text or task_text == "None":
        raise DataError(f"{where}: empty task_id")
    return domain_text, task_text


def _attempt_id(row: dict[str, Any], *, where: str) -> str:
    for field_name in ATTEMPT_FIELDS:
        value = _nested_value(row, field_name)
        if value is not None and not isinstance(value, (dict, list, bool)):
            text = str(value).strip()
            if text:
                return text
    raise DataError(
        f"{where}: each attempt needs one of {', '.join(ATTEMPT_FIELDS)}"
    )


def _content_key(
    row: dict[str, Any], *, attempt_id: str, where: str
) -> tuple[str, bool]:
    for field_name in (
        "trajectory_sha256",
        "simulation_sha256",
        "content_sha256",
        "attempt_sha256",
    ):
        value = _nested_value(row, field_name)
        if value is None:
            continue
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise DataError(f"{where}: {field_name} must be lowercase SHA-256")
        return f"sha256:{value}", True
    messages = row.get("messages")
    if isinstance(messages, list):
        encoded = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"derived:{hashlib.sha256(encoded).hexdigest()}", True
    # An attempt-level builder audit may omit trajectory content.  It remains
    # auditable for cross-seed counts, but the JSON result makes the missing
    # content-level deduplication evidence explicit.
    return f"attempt:{attempt_id}", False


def _tool_link_id(message: dict[str, Any]) -> str | None:
    value = message.get("tool_call_id", message.get("id"))
    return value if isinstance(value, str) and value else None


def _call_id(call: dict[str, Any]) -> str | None:
    value = call.get("id")
    return value if isinstance(value, str) and value else None


def _is_injected(message: dict[str, Any]) -> bool:
    raw = message.get("raw_data")
    return isinstance(raw, dict) and bool(
        raw.get("v5_stage1_injected_fault")
        or raw.get("v5_stage0_injected_fault")
        or raw.get("v5_2_injected_fault")
    )


def _reward_value(row: dict[str, Any]) -> float | None:
    reward_info = row.get("reward_info")
    value = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    value = row.get("reward")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _analyse_raw_simulation(
    row: dict[str, Any], condition: str
) -> tuple[bool, bool, str]:
    """Reproduce the eligibility facts needed by the V5 builder."""

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, False, "invalid_simulation_no_messages"
    if not all(isinstance(message, dict) for message in messages):
        return False, False, "invalid_simulation_malformed_message"
    if not any(message.get("role") == "user" for message in messages):
        return False, False, "invalid_simulation_no_user_message"
    reward = _reward_value(row)
    if reward is None:
        return False, False, "unscoreable_missing_reward"

    outcomes: list[tuple[int, int, bool, bool]] = []
    consumed_results: set[int] = set()
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if calls:
            if (
                message.get("role") != "assistant"
                or not isinstance(calls, list)
                or len(calls) != 1
            ):
                return False, False, "invalid_simulation_non_single_tool_call"
            if message.get("content") not in (None, ""):
                return False, False, "invalid_simulation_mixed_text_and_tool_call"
            call = calls[0]
            if not isinstance(call, dict) or _call_id(call) is None:
                return False, False, "invalid_simulation_malformed_tool_call"
            result_index = index + 1
            if result_index >= len(messages):
                return False, False, "invalid_simulation_missing_tool_result"
            result = messages[result_index]
            if (
                result.get("role") != "tool"
                or _tool_link_id(result) != _call_id(call)
                or type(result.get("error")) is not bool
            ):
                return False, False, "invalid_simulation_malformed_tool_result"
            consumed_results.add(result_index)
            outcomes.append(
                (
                    index,
                    result_index,
                    bool(result["error"]),
                    _is_injected(message),
                )
            )
        elif message.get("role") == "tool" and index not in consumed_results:
            return False, False, "invalid_simulation_orphan_tool_result"

    final_success = reward == 1.0
    failed = [outcome for outcome in outcomes if outcome[2]]
    injected = [outcome for outcome in outcomes if outcome[3]]
    if condition == "clean":
        eligible = final_success and not failed and not injected
        if eligible:
            return True, True, "eligible"
        if not final_success:
            return True, False, "final_failure"
        return True, False, "clean_contains_failure_or_injection"

    injected_failure = (
        len(injected) == 1 and injected[0][2] and len(failed) == 1
    )
    later_success = bool(
        injected
        and any(
            not outcome[2] and outcome[0] > injected[0][1]
            for outcome in outcomes
        )
    )
    eligible = final_success and injected_failure and later_success
    if eligible:
        return True, True, "eligible"
    if not final_success:
        return True, False, "final_failure"
    if not injected_failure:
        return True, False, "not_exactly_one_controlled_failure"
    return True, False, "no_verified_post_error_repair"


def _explicit_attempt_facts(
    row: dict[str, Any],
) -> tuple[bool, bool, str] | None:
    eligible = _nested_value(row, "eligible")
    scoreable = _nested_value(row, "scoreable")
    reason = _nested_value(row, "reason")
    if eligible is None and scoreable is None:
        return None
    if type(eligible) is not bool:
        raise DataError("attempt-level eligible must be a boolean")
    if scoreable is None:
        reason_text = str(reason or "")
        scoreable = not (
            reason_text.startswith("invalid_simulation_")
            or reason_text in INFRASTRUCTURE_REASONS
            or reason_text.startswith("unscoreable_")
        )
    if type(scoreable) is not bool:
        raise DataError("attempt-level scoreable must be a boolean")
    if eligible and not scoreable:
        raise DataError("eligible attempt cannot be unscoreable")
    return scoreable, eligible, str(reason or ("eligible" if eligible else "ineligible"))


def _normalise_attempt(
    row: dict[str, Any],
    *,
    path: Path,
    row_index: int,
    default_condition: str | None,
    default_domain: str | None,
) -> Attempt:
    where = f"{path}:record[{row_index}]"
    condition_value = _nested_value(row, "condition")
    condition = (
        str(condition_value).strip().lower()
        if condition_value is not None
        else default_condition
    )
    if condition not in CONDITIONS:
        raise DataError(f"{where}: condition must be clean or error")
    domain, task_id = _normalise_domain_task(
        _nested_value(row, "domain"),
        _nested_value(row, "task_id"),
        default_domain=default_domain,
        where=where,
    )
    facts = _explicit_attempt_facts(row)
    if facts is None:
        facts = _analyse_raw_simulation(row, condition)
    scoreable, eligible, reason = facts
    attempt_id = _attempt_id(row, where=where)
    content_key, content_hash_verified = _content_key(
        row, attempt_id=attempt_id, where=where
    )
    return Attempt(
        domain=domain,
        task_id=task_id,
        condition=condition,
        attempt_id=attempt_id,
        scoreable=scoreable,
        eligible=eligible,
        reason=reason,
        content_key=content_key,
        content_hash_verified=content_hash_verified,
        source=where,
    )


def _task_ref(
    value: Any, *, default_domain: str | None, where: str
) -> str:
    if isinstance(value, dict):
        domain = value.get("domain")
        task_id = value.get("task_id", value.get("id"))
    else:
        domain = None
        task_id = value
    domain_text, task_text = _normalise_domain_task(
        domain,
        task_id,
        default_domain=default_domain,
        where=where,
    )
    return f"{domain_text}:{task_text}"


def _task_refs_from_value(
    value: Any, *, default_domain: str | None, where: str
) -> set[str]:
    if not isinstance(value, list):
        raise DataError(f"{where}: task universe must be a list")
    return {
        _task_ref(item, default_domain=default_domain, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    }


def _declared_tasks(payload: Any, path: Path) -> tuple[set[str], bool]:
    if not isinstance(payload, dict):
        return set(), False
    default_domain = (
        str(payload.get("domain")).lower()
        if isinstance(payload.get("domain"), str)
        else _domain_from_filename(path)
    )
    result: set[str] = set()
    complete = False
    # Formal V5.2 generation covers 83 inner-train tasks, while only the
    # frozen 75-task arm-training population may enter the feasibility gate.
    # Prefer an explicit gate population over a broader generation universe.
    for field_name in (
        "gate_task_ids",
        "arm_train_task_ids",
        "population_task_ids",
    ):
        if field_name in payload:
            return (
                _task_refs_from_value(
                    payload[field_name],
                    default_domain=default_domain,
                    where=f"{path}:{field_name}",
                ),
                True,
            )
    for field_name in ("planned_task_ids", "expected_task_ids"):
        if field_name in payload:
            result.update(
                _task_refs_from_value(
                    payload[field_name],
                    default_domain=default_domain,
                    where=f"{path}:{field_name}",
                )
            )
            complete = True
    if "task_ids" in payload:
        result.update(
            _task_refs_from_value(
                payload["task_ids"],
                default_domain=default_domain,
                where=f"{path}:task_ids",
            )
        )
        complete = complete or payload.get("task_universe_complete") is True
    domains = payload.get("domains")
    if isinstance(domains, dict):
        gate_result: set[str] = set()
        found_train = False
        for domain, descriptor in domains.items():
            if not isinstance(descriptor, dict) or "train_ids" not in descriptor:
                continue
            found_train = True
            gate_result.update(
                _task_refs_from_value(
                    descriptor["train_ids"],
                    default_domain=str(domain).lower(),
                    where=f"{path}:domains.{domain}.train_ids",
                )
            )
        if found_train:
            return gate_result, True
        found_inner_train = False
        for domain, descriptor in domains.items():
            if not isinstance(descriptor, dict) or "inner_train_ids" not in descriptor:
                continue
            found_inner_train = True
            result.update(
                _task_refs_from_value(
                    descriptor["inner_train_ids"],
                    default_domain=str(domain).lower(),
                    where=f"{path}:domains.{domain}.inner_train_ids",
                )
            )
        complete = complete or found_inner_train
    return result, complete


def _expand_task_container(
    tasks: Any,
    *,
    path: Path,
    default_domain: str | None,
) -> list[dict[str, Any]]:
    if isinstance(tasks, list):
        task_rows = tasks
    elif isinstance(tasks, dict):
        task_rows = []
        for task_key, descriptor in tasks.items():
            if not isinstance(descriptor, dict):
                raise DataError(f"{path}:tasks.{task_key} must be an object")
            row = dict(descriptor)
            row.setdefault("task_id", task_key)
            task_rows.append(row)
    else:
        raise DataError(f"{path}: tasks must be a list or object")

    expanded: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_rows):
        if not isinstance(task, dict):
            raise DataError(f"{path}:tasks[{task_index}] must be an object")
        task_domain = task.get("domain", default_domain)
        task_id = task.get("task_id", task.get("id"))
        for condition in CONDITIONS:
            attempts = task.get(condition)
            if attempts is None:
                attempts = task.get(f"{condition}_attempts")
            if attempts is None:
                continue
            if not isinstance(attempts, list):
                raise DataError(
                    f"{path}:tasks[{task_index}].{condition} must be a list"
                )
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    raise DataError(
                        f"{path}:tasks[{task_index}].{condition} attempt "
                        "must be an object"
                    )
                row = dict(attempt)
                row.setdefault("domain", task_domain)
                row.setdefault("task_id", task_id)
                row.setdefault("condition", condition)
                expanded.append(row)
    return expanded


def _extract_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    default_condition = (
        str(payload.get("condition")).lower()
        if isinstance(payload, dict) and isinstance(payload.get("condition"), str)
        else _condition_from_filename(path)
    )
    default_domain = (
        str(payload.get("domain")).lower()
        if isinstance(payload, dict) and isinstance(payload.get("domain"), str)
        else _domain_from_filename(path)
    )
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        if "tasks" in payload:
            rows.extend(
                _expand_task_container(
                    payload["tasks"],
                    path=path,
                    default_domain=default_domain,
                )
            )
        for condition in CONDITIONS:
            condition_rows = payload.get(condition)
            if isinstance(condition_rows, list):
                for value in condition_rows:
                    if not isinstance(value, dict):
                        raise DataError(
                            f"{path}:{condition} records must be objects"
                        )
                    row = dict(value)
                    row.setdefault("condition", condition)
                    rows.append(row)
        for field_name in ROW_LIST_FIELDS:
            values = payload.get(field_name)
            if isinstance(values, list):
                rows.extend(values)
                break
        if not rows and _nested_value(payload, "task_id") is not None:
            rows = [payload]
    else:
        raise DataError(f"{path}: JSON root must be an object, list, or JSONL")
    if not rows:
        raise DataError(
            f"{path}: no attempt-level rows; aggregate counts cannot prove "
            "task-level cross-seed feasibility"
        )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DataError(f"{path}: record[{index}] must be an object")
        copied = dict(row)
        if default_condition is not None:
            copied.setdefault("condition", default_condition)
        if default_domain is not None:
            copied.setdefault("domain", default_domain)
        result.append(copied)
    return result


def _load_task_universe(path: Path) -> set[str]:
    payload = _read_json_or_jsonl(path)
    if isinstance(payload, list):
        return _task_refs_from_value(
            payload, default_domain=None, where=f"{path}:task_universe"
        )
    declared, complete = _declared_tasks(payload, path)
    if not declared or not complete:
        raise DataError(
            f"{path}: task universe needs planned_task_ids, expected_task_ids, "
            "or domains.*.inner_train_ids"
        )
    return declared


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid Wilson interval counts")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": None,
            "confidence": confidence,
            "lower": None,
            "upper": None,
        }
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / total
    denominator = 1.0 + z * z / total
    centre = (estimate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "confidence": confidence,
        "lower": max(0.0, centre - margin),
        "upper": min(1.0, centre + margin),
    }


def _rate_summary(
    attempts: Sequence[Attempt], confidence: float
) -> dict[str, Any]:
    total = len(attempts)
    scoreable = sum(attempt.scoreable for attempt in attempts)
    eligible = sum(attempt.eligible for attempt in attempts)
    return {
        "attempts": total,
        "scoreable": scoreable,
        "eligible": eligible,
        "scoreable_rate": wilson_interval(
            scoreable, total, confidence=confidence
        ),
        "eligible_rate_all_attempts": wilson_interval(
            eligible, total, confidence=confidence
        ),
        "eligible_rate_given_scoreable": wilson_interval(
            eligible, scoreable, confidence=confidence
        ),
        "reason_counts": dict(
            sorted(Counter(attempt.reason for attempt in attempts).items())
        ),
    }


def audit_attempts(
    attempts: Sequence[Attempt],
    *,
    task_universe: set[str] | None = None,
    universe_complete: bool = False,
    min_tasks: int = DEFAULT_MIN_TASKS,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_pairs_per_task: int = DEFAULT_MAX_PAIRS_PER_TASK,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    if min_tasks <= 0 or min_pairs <= 0:
        raise DataError("min_tasks and min_pairs must be positive")
    if max_pairs_per_task <= 0 or max_attempts <= 0:
        raise DataError("max_pairs_per_task and max_attempts must be positive")
    if max_pairs_per_task > max_attempts:
        raise DataError("max_pairs_per_task cannot exceed max_attempts")
    if not 0.0 < confidence < 1.0:
        raise DataError("confidence must be between zero and one")
    if not attempts:
        raise DataError("no attempts to audit")

    all_attempts = list(attempts)
    if task_universe is None:
        gate_attempts = all_attempts
        excluded_attempts: list[Attempt] = []
    else:
        gate_attempts = [
            attempt
            for attempt in all_attempts
            if attempt.task_key in task_universe
        ]
        excluded_attempts = [
            attempt
            for attempt in all_attempts
            if attempt.task_key not in task_universe
        ]

    states: dict[str, TaskState] = defaultdict(TaskState)
    for attempt in gate_attempts:
        states[attempt.task_key].add(attempt)
    observed_tasks = set(states)
    if task_universe is None:
        universe = set(observed_tasks)
    else:
        universe = set(task_universe)
    if not universe:
        raise DataError("empty task universe")

    task_rows: list[dict[str, Any]] = []
    current_pairs = 0
    current_pair_tasks = 0
    maximum_pairs = 0
    maximum_pair_tasks = 0
    same_attempt_pairs = 0
    two_sided_attempt_tasks = 0
    two_sided_scoreable_tasks = 0
    two_sided_eligible_tasks = 0
    total_remaining = 0

    for task_key in sorted(universe):
        state = states.get(task_key, TaskState())
        attempt_counts = {
            condition: len(state.attempts[condition])
            for condition in CONDITIONS
        }
        if any(value > max_attempts for value in attempt_counts.values()):
            raise DataError(
                f"{task_key}: observed attempts exceed frozen "
                f"{max_attempts}-per-condition budget"
            )
        scoreable_counts = {
            condition: state.scoreable_count(condition)
            for condition in CONDITIONS
        }
        eligible_ids = {
            condition: state.eligible_ids(condition)
            for condition in CONDITIONS
        }
        eligible_counts = {
            condition: len(eligible_ids[condition])
            for condition in CONDITIONS
        }
        raw_eligible_counts = {
            condition: state.raw_eligible_count(condition)
            for condition in CONDITIONS
        }
        pair_count = min(
            eligible_counts["clean"],
            eligible_counts["error"],
            max_pairs_per_task,
        )
        same_count = min(
            len(eligible_ids["clean"] & eligible_ids["error"]),
            max_pairs_per_task,
        )
        remaining = {
            condition: max_attempts - attempt_counts[condition]
            for condition in CONDITIONS
        }
        possible_eligible = {
            condition: eligible_counts[condition] + remaining[condition]
            for condition in CONDITIONS
        }
        possible_pairs = min(
            possible_eligible["clean"],
            possible_eligible["error"],
            max_pairs_per_task,
        )
        current_pairs += pair_count
        maximum_pairs += possible_pairs
        same_attempt_pairs += same_count
        total_remaining += sum(remaining.values())
        current_pair_tasks += pair_count >= 1
        maximum_pair_tasks += possible_pairs >= 1
        two_sided_attempt_tasks += all(
            attempt_counts[condition] > 0 for condition in CONDITIONS
        )
        two_sided_scoreable_tasks += all(
            scoreable_counts[condition] > 0 for condition in CONDITIONS
        )
        two_sided_eligible_tasks += all(
            eligible_counts[condition] > 0 for condition in CONDITIONS
        )
        task_rows.append(
            {
                "task_id": task_key,
                "observed_attempts": attempt_counts,
                "scoreable_attempts": scoreable_counts,
                "eligible_attempts": eligible_counts,
                "eligible_attempts_before_content_dedup": raw_eligible_counts,
                "eligible_content_duplicates_removed": {
                    condition: raw_eligible_counts[condition]
                    - eligible_counts[condition]
                    for condition in CONDITIONS
                },
                "constructible_cross_seed_pairs": pair_count,
                "same_attempt_id_pairs": same_count,
                "cross_seed_additional_pairs": pair_count - same_count,
                "remaining_attempt_budget": remaining,
                "maximum_structural_pairs": possible_pairs,
            }
        )

    current_task_gate = current_pair_tasks >= min_tasks
    current_pair_gate = current_pairs >= min_pairs
    potential_task_gate = maximum_pair_tasks >= min_tasks
    potential_pair_gate = maximum_pairs >= min_pairs
    if current_task_gate and current_pair_gate:
        status = "TRAINABLE"
        exit_code = EXIT_TRAINABLE
        reason = "observed pool satisfies both frozen gates"
    elif potential_task_gate and potential_pair_gate:
        status = "PENDING_REACHABLE"
        exit_code = EXIT_PENDING_REACHABLE
        reason = (
            "observed pool is incomplete, but the optimistic structural upper "
            "bound still satisfies both gates"
        )
    elif not universe_complete:
        raise DataError(
            "observed tasks cannot reach the gates and the complete frozen task "
            "universe was not supplied; pass --task-universe or embed "
            "planned_task_ids"
        )
    else:
        status = "MATHEMATICALLY_UNREACHABLE"
        exit_code = EXIT_MATHEMATICALLY_UNREACHABLE
        blockers = []
        if not potential_task_gate:
            blockers.append(
                f"at most {maximum_pair_tasks} tasks can form a pair; "
                f"need {min_tasks}"
            )
        if not potential_pair_gate:
            blockers.append(
                f"at most {maximum_pairs} pairs can be formed; need {min_pairs}"
            )
        reason = "; ".join(blockers)

    by_condition = {
        condition: _rate_summary(
            [
                attempt
                for attempt in gate_attempts
                if attempt.condition == condition
            ],
            confidence,
        )
        for condition in CONDITIONS
    }
    result = {
        "protocol": "v5_2_stage1_feasibility_audit",
        "status": status,
        "exit_code": exit_code,
        "decision_reason": reason,
        "claim_boundary": {
            "inner_train_generation_only": True,
            "inner_train_source_split_verified": True,
            "official_test_used": False,
            "derived_validation_outcomes_used": False,
            "positive_outcomes_used_to_choose_tasks": False,
            "future_bound_is_optimistic_structural_only": True,
            "cross_seed_pairing": True,
            "deduplicate_before_pairing": True,
        },
        "gates": {
            "minimum_task_ids_with_pair": min_tasks,
            "minimum_constructible_pairs": min_pairs,
            "maximum_pairs_per_task": max_pairs_per_task,
            "maximum_attempts_per_task_per_condition": max_attempts,
        },
        "task_universe": {
            "tasks": len(universe),
            "observed_tasks": len(observed_tasks),
            "complete": universe_complete,
            "excluded_inner_train_tasks": len(
                {attempt.task_key for attempt in excluded_attempts}
            ),
            "excluded_inner_train_attempts": len(excluded_attempts),
            "excluded_attempts_used_for_rates_or_gate": False,
        },
        "attempt_rates": {
            "overall": _rate_summary(gate_attempts, confidence),
            **by_condition,
        },
        "task_level_two_sided_coverage": {
            "tasks_with_attempts_on_both_conditions": two_sided_attempt_tasks,
            "tasks_with_scoreable_attempts_on_both_conditions": (
                two_sided_scoreable_tasks
            ),
            "tasks_with_eligible_attempts_on_both_conditions": (
                two_sided_eligible_tasks
            ),
            "tasks_with_constructible_pair": current_pair_tasks,
            "constructible_cross_seed_pairs": current_pairs,
            "same_attempt_id_pairs": same_attempt_pairs,
            "cross_seed_additional_pairs": current_pairs - same_attempt_pairs,
            "tasks_with_constructible_pair_rate": wilson_interval(
                current_pair_tasks, len(universe), confidence=confidence
            ),
            "constructible_pair_slot_fill_rate": wilson_interval(
                current_pairs,
                len(universe) * max_pairs_per_task,
                confidence=confidence,
            ),
            "cross_seed_additional_pair_fraction": wilson_interval(
                current_pairs - same_attempt_pairs,
                current_pairs,
                confidence=confidence,
            ),
        },
        "structural_budget_bound": {
            "remaining_attempts_across_both_conditions": total_remaining,
            "maximum_tasks_with_pair": maximum_pair_tasks,
            "maximum_constructible_pairs": maximum_pairs,
            "task_gate_reachable": potential_task_gate,
            "pair_gate_reachable": potential_pair_gate,
            "current_task_deficit": max(0, min_tasks - current_pair_tasks),
            "current_pair_deficit": max(0, min_pairs - current_pairs),
        },
        "task_audit": task_rows,
        "content_deduplication": {
            "all_gate_attempts_have_content_hash_evidence": all(
                attempt.content_hash_verified for attempt in gate_attempts
            ),
            "attempts_without_content_hash_evidence": sum(
                not attempt.content_hash_verified for attempt in gate_attempts
            ),
        },
    }
    return result


def audit_paths(
    paths: Sequence[Path],
    *,
    task_universe_path: Path | None = None,
    min_tasks: int = DEFAULT_MIN_TASKS,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_pairs_per_task: int = DEFAULT_MAX_PAIRS_PER_TASK,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    attempts: list[Attempt] = []
    declared_tasks: set[str] = set()
    universe_complete = False
    for path in paths:
        payload = _read_json_or_jsonl(path)
        tasks, complete = _declared_tasks(payload, path)
        declared_tasks.update(tasks)
        universe_complete = universe_complete or complete
        rows = _extract_rows(payload, path)
        _require_inner_train_provenance(payload, rows, path=path)
        default_condition = (
            str(payload.get("condition")).lower()
            if isinstance(payload, dict)
            and isinstance(payload.get("condition"), str)
            else _condition_from_filename(path)
        )
        default_domain = (
            str(payload.get("domain")).lower()
            if isinstance(payload, dict)
            and isinstance(payload.get("domain"), str)
            else _domain_from_filename(path)
        )
        for index, row in enumerate(rows):
            attempts.append(
                _normalise_attempt(
                    row,
                    path=path,
                    row_index=index,
                    default_condition=default_condition,
                    default_domain=default_domain,
                )
            )
    if task_universe_path is not None:
        declared_tasks = _load_task_universe(task_universe_path)
        universe_complete = True
    return audit_attempts(
        attempts,
        task_universe=declared_tasks or None,
        universe_complete=universe_complete,
        min_tasks=min_tasks,
        min_pairs=min_pairs,
        max_pairs_per_task=max_pairs_per_task,
        max_attempts=max_attempts,
        confidence=confidence,
    )


def _write_result(target: str | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if target == "-":
        sys.stdout.write(text)
    elif target is not None:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _human_summary(result: dict[str, Any]) -> str:
    coverage = result["task_level_two_sided_coverage"]
    bound = result["structural_budget_bound"]
    gates = result["gates"]
    return "\n".join(
        (
            f"V5.2 feasibility: {result['status']} "
            f"(exit {result['exit_code']})",
            (
                "Current: "
                f"{coverage['tasks_with_constructible_pair']}/"
                f"{gates['minimum_task_ids_with_pair']} tasks, "
                f"{coverage['constructible_cross_seed_pairs']}/"
                f"{gates['minimum_constructible_pairs']} pairs"
            ),
            (
                "Structural maximum within frozen budget: "
                f"{bound['maximum_tasks_with_pair']} tasks, "
                f"{bound['maximum_constructible_pairs']} pairs"
            ),
            result["decision_reason"],
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="inner-train generation JSON/JSONL shards or builder audit JSON",
    )
    parser.add_argument(
        "--task-universe",
        type=Path,
        help=(
            "optional frozen inner-train task universe; required for an "
            "unreachable decision if inputs do not embed a complete universe"
        ),
    )
    parser.add_argument("--min-tasks", type=int, default=DEFAULT_MIN_TASKS)
    parser.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS)
    parser.add_argument(
        "--max-pairs-per-task",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_TASK,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="maximum attempts per task per condition",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="two-sided Wilson confidence level",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="emit JSON to stdout, or to PATH when one is supplied",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_paths(
            args.inputs,
            task_universe_path=args.task_universe,
            min_tasks=args.min_tasks,
            min_pairs=args.min_pairs,
            max_pairs_per_task=args.max_pairs_per_task,
            max_attempts=args.max_attempts,
            confidence=args.confidence,
        )
    except (DataError, ValueError) as exc:
        error = {
            "protocol": "v5_2_stage1_feasibility_audit",
            "status": "DATA_ERROR",
            "exit_code": EXIT_DATA_ERROR,
            "error": str(exc),
            "official_test_used": False,
            "derived_validation_outcomes_used": False,
        }
        if args.json is not None:
            _write_result(args.json, error)
        else:
            print(f"V5.2 feasibility DATA_ERROR: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    if args.json is not None:
        _write_result(args.json, result)
    else:
        print(_human_summary(result))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
