#!/usr/bin/env python3
"""Frozen contract for V5.5 audit-first counterfactual recovery data.

V5.5 keeps the recovery-data-selection question from V5.4, but replaces
outcome-dependent injection-site discovery with a pre-rollout structural
screen over the pinned tau2 reference action path.  Reference actions are
treated as one valid action backbone, not as a natural expert conversation.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

PROTOCOL = "v5_5_reference_grounded_counterfactual_recovery_v1"
DESIGN_VERSION = "5.5"
BASE_SEED = 20260805
REGISTERED_TASKS = 24
PAIRS_PER_TASK = 2
TARGET_PAIRS = REGISTERED_TASKS * PAIRS_PER_TASK
MIN_TASKS_WITH_PAIRS = 16
MIN_AUDITED_PAIRS = 32
MIN_DOMAINS = 2
OFFICIAL_TEST_USED = False

# Only identifier lookups with unambiguous, schema-valid string arguments are
# eligible.  Names, free text, dates and ZIP codes are deliberately excluded.
READ_ACTIONS: dict[str, dict[str, str]] = {
    "airline": {
        "get_reservation_details": "reservation_id",
        "get_user_details": "user_id",
    },
    "retail": {
        "get_order_details": "order_id",
        "get_user_details": "user_id",
        "get_product_details": "product_id",
        "get_item_details": "item_id",
    },
}


class V55ProtocolError(RuntimeError):
    """The frozen V5.5 contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


TRANSIENT_MESSAGE_KEYS = {
    "timestamp",
    "turn_idx",
    "cost",
    "usage",
    "raw_data",
    "generation_time_seconds",
}


def semantic_message_view(value: Any) -> Any:
    """Remove runtime-only fields before trajectory identity comparisons."""
    if isinstance(value, list):
        return [semantic_message_view(item) for item in value]
    if isinstance(value, dict):
        return {
            key: semantic_message_view(item)
            for key, item in value.items()
            if key not in TRANSIENT_MESSAGE_KEYS
        }
    return value


def semantic_sha256(value: Any) -> str:
    return sha256(semantic_message_view(value))


def task_rank(task_identity: str) -> bytes:
    return hashlib.sha256(
        f"{PROTOCOL}|task|{BASE_SEED}|{task_identity}".encode()
    ).digest()


def action_rank(task_identity: str, action_index: int) -> bytes:
    return hashlib.sha256(
        (
            f"{PROTOCOL}|action|{BASE_SEED}|{task_identity}|"
            f"{action_index}"
        ).encode()
    ).digest()


def mutate_identifier(value: str, variant: int) -> str:
    """Return one of two deterministic, type-preserving typo values."""
    if variant not in (0, 1):
        raise V55ProtocolError("mutation variant must be 0 or 1")
    if not isinstance(value, str) or len(value) < 2:
        raise V55ProtocolError("identifier must be a non-empty string")
    chars = list(value)
    mutable = [i for i, char in enumerate(chars) if char.isalnum()]
    if not mutable:
        raise V55ProtocolError("identifier has no mutable alphanumeric character")
    index = mutable[-1 - (variant % min(2, len(mutable)))]
    original = chars[index]
    if original.isdigit():
        chars[index] = str((int(original) + variant + 1) % 10)
    elif original.isupper():
        chars[index] = chr((ord(original) - 65 + variant + 1) % 26 + 65)
    else:
        chars[index] = chr((ord(original.lower()) - 97 + variant + 1) % 26 + 97)
    mutated = "".join(chars)
    if mutated == value:
        raise V55ProtocolError("mutation did not change identifier")
    return mutated


def eligible_reference_actions(
    domain: str, actions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    allowed = READ_ACTIONS.get(domain, {})
    eligible: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        name = str(action.get("name", ""))
        key = allowed.get(name)
        arguments = action.get("arguments")
        if (
            action.get("requestor", "assistant") != "assistant"
            or key is None
            or not isinstance(arguments, dict)
            or not isinstance(arguments.get(key), str)
            or len(arguments[key]) < 2
        ):
            continue
        eligible.append(
            {
                "action_index": index,
                "tool_name": name,
                "identifier_key": key,
                "correct_identifier": arguments[key],
            }
        )
    return eligible


def select_site(
    task_identity: str, eligible: list[dict[str, Any]]
) -> dict[str, Any]:
    if not eligible:
        raise V55ProtocolError(f"{task_identity}: no eligible reference action")
    return min(
        eligible,
        key=lambda row: action_rank(task_identity, int(row["action_index"])),
    )


def choose_tasks(
    candidates: list[dict[str, Any]], count: int = REGISTERED_TASKS
) -> list[dict[str, Any]]:
    """Select without rollout, reward, validation or test outcomes."""
    unique = {row["task_identity"]: row for row in candidates}
    if len(unique) < count:
        raise V55ProtocolError(
            f"only {len(unique)} structurally eligible tasks; need {count}"
        )
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in unique.values():
        by_domain.setdefault(str(row["domain"]), []).append(row)
    if len(by_domain) < MIN_DOMAINS:
        raise V55ProtocolError("both airline and retail must be represented")
    for rows in by_domain.values():
        rows.sort(key=lambda row: task_rank(str(row["task_identity"])))

    # Guarantee six tasks per domain, then fill the remaining twelve globally.
    selected: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        if len(by_domain[domain]) < 6:
            raise V55ProtocolError(f"{domain}: fewer than six eligible tasks")
        selected.extend(by_domain[domain][:6])
    chosen = {row["task_identity"] for row in selected}
    remainder = sorted(
        (row for row in unique.values() if row["task_identity"] not in chosen),
        key=lambda row: task_rank(str(row["task_identity"])),
    )
    selected.extend(remainder[: count - len(selected)])
    return sorted(selected, key=lambda row: str(row["task_identity"]))


def audit_pair(pair: dict[str, Any]) -> dict[str, bool]:
    """Recompute checks from evidence; producer truth flags are ignored."""
    clean_call = pair.get("clean_call")
    injected_call = pair.get("injected_call")
    error_result = pair.get("injected_result")
    correction_call = pair.get("correction_call")
    if not all(isinstance(x, dict) for x in (clean_call, injected_call, error_result, correction_call)):
        return {"well_formed": False}
    key = pair.get("identifier_key")
    clean_args = clean_call.get("arguments", {})
    injected_args = injected_call.get("arguments", {})
    correction_args = correction_call.get("arguments", {})
    failed_event = pair.get("failed_event")
    supervised = pair.get("supervised_messages")
    checks = {
        "well_formed": True,
        "same_tool_schema": (
            clean_call.get("name")
            == injected_call.get("name")
            == correction_call.get("name")
        ),
        "identifier_changed": (
            isinstance(key, str)
            and clean_args.get(key) != injected_args.get(key)
            and correction_args.get(key) == clean_args.get(key)
        ),
        "only_identifier_changed": (
            isinstance(key, str)
            and {
                k: v for k, v in clean_args.items() if k != key
            }
            == {k: v for k, v in injected_args.items() if k != key}
        ),
        "actual_tool_error": error_result.get("error") is True,
        "failed_event_masked": (
            isinstance(failed_event, list)
            and len(failed_event) == 2
            and pair.get("supervision_starts_after_error") is True
        ),
        "failed_event_not_supervised": (
            isinstance(supervised, list)
            and isinstance(failed_event, list)
            and not (
                {canonical(message) for message in failed_event}
                & {canonical(message) for message in supervised}
            )
        ),
        "future_removed_from_prompt": (
            pair.get("clean_future_present_in_recovery_prompt") is False
        ),
        "shared_prefix_exact": (
            pair.get("clean_prefix_sha256")
            == pair.get("recovery_prefix_sha256")
        ),
        "correct_call_succeeded": pair.get("correction_result", {}).get("error") is False,
        "clean_end_state_valid": (
            pair.get("clean_end_state_matches_reference") is True
            and bool(pair.get("clean_agent_db_hash"))
        ),
        "recovery_end_state_valid": (
            pair.get("recovery_end_state_matches_reference") is True
            and pair.get("clean_agent_db_hash")
            == pair.get("recovery_agent_db_hash")
            and pair.get("clean_user_db_hash")
            == pair.get("recovery_user_db_hash")
        ),
        "independent_environment_replay": (
            pair.get("independent_environment_replay_pass") is True
        ),
        "official_test_sealed": pair.get("official_test_used") is False,
    }
    return checks


def gate(pairs: list[dict[str, Any]], registered_tasks: list[str]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for pair in pairs:
        checks = audit_pair(pair)
        if checks and all(checks.values()):
            accepted.append(pair)
        else:
            failures.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "failed_checks": sorted(k for k, v in checks.items() if not v),
                }
            )
    task_counts: dict[str, int] = {}
    for pair in accepted:
        identity = str(pair.get("task_identity"))
        task_counts[identity] = task_counts.get(identity, 0) + 1
    fully_paired = sorted(
        identity for identity, count in task_counts.items() if count >= PAIRS_PER_TASK
    )
    domains = sorted(
        {str(pair.get("domain")) for pair in accepted if pair.get("domain")}
    )
    unique_pair_ids = {pair.get("pair_id") for pair in accepted}
    task_error_events = {
        (pair.get("task_identity"), pair.get("error_event_sha256"))
        for pair in accepted
    }
    checks = {
        "registered_task_count": len(set(registered_tasks)) == REGISTERED_TASKS,
        "minimum_fully_paired_tasks": len(fully_paired) >= MIN_TASKS_WITH_PAIRS,
        "minimum_audited_pairs": len(accepted) >= MIN_AUDITED_PAIRS,
        "both_domains": len(domains) >= MIN_DOMAINS,
        "unique_pair_ids": len(unique_pair_ids) == len(accepted),
        "distinct_error_events_within_task": (
            len(task_error_events) == len(accepted)
        ),
        "zero_pair_audit_failures": not failures,
        "official_test_sealed": all(
            pair.get("official_test_used") is False for pair in pairs
        ),
    }
    return {
        "protocol": PROTOCOL,
        "status": "PASS_TRAINING_AUTHORIZED" if all(checks.values()) else "FAIL_CLOSED",
        "checks": checks,
        "observed": {
            "registered_tasks": len(set(registered_tasks)),
            "audited_pairs": len(accepted),
            "fully_paired_tasks": len(fully_paired),
            "domains": domains,
        },
        "fully_paired_task_ids": fully_paired,
        "pair_failures": failures,
        "claim_boundary": (
            "A pass authorizes the registered SFT comparison; it does not "
            "establish improved end-to-end task success."
        ),
    }


def validate_constants() -> None:
    if TARGET_PAIRS != 48 or MIN_AUDITED_PAIRS > TARGET_PAIRS:
        raise V55ProtocolError("V5.5 pair budget drift")
    rng = random.Random(BASE_SEED)
    if rng.randint(0, 1_000_000) != 538364:
        raise V55ProtocolError("V5.5 seed drift")
