#!/usr/bin/env python3
"""Frozen, dependency-free contract for the V5.4 counterfactual pilot.

V5.4 is a data-feasibility experiment, not a training or paper-result run.
It first searches for one successful clean trajectory per registered task,
then branches from a deterministically selected read-only tool-call prefix.
The clean future is unavailable to the recovery generator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Any, Iterable


PROTOCOL = "v5_4_counterfactual_branch_pilot_v1"
DESIGN_VERSION = "5.4-pilot"
BASE_SEED = 20260804
TRIAL_SEEDS = (
    253882,
    150999,
    692585,
    709707,
    602924,
    390225,
    22158,
    106545,
    106867,
    350607,
    153438,
    113820,
)
CLEAN_SLOTS_PER_TASK = 8
RECOVERY_SLOTS_PER_TASK = 4
SLOTS_PER_TASK = CLEAN_SLOTS_PER_TASK + RECOVERY_SLOTS_PER_TASK
MAX_PAIRS_PER_TASK = 2
TASKS = 24
MAX_EXECUTED_SLOTS = TASKS * SLOTS_PER_TASK
MIN_TASKS_WITH_PAIR = 14
MIN_CAPPED_PAIRS = 17
V5_3_OBSERVED_TASKS_WITH_PAIR = 8
V5_3_OBSERVED_CAPPED_PAIRS = 12

PILOT_TASK_IDS = (
    "airline:1",
    "airline:11",
    "airline:14",
    "airline:33",
    "airline:38",
    "airline:40",
    "retail:2",
    "retail:8",
    "retail:10",
    "retail:15",
    "retail:19",
    "retail:25",
    "retail:30",
    "retail:35",
    "retail:54",
    "retail:67",
    "retail:69",
    "retail:72",
    "retail:85",
    "retail:92",
    "retail:93",
    "retail:104",
    "retail:106",
    "retail:110",
)

FORMAL_V5_3_SEEDS = {
    20260722,
    574293,
    816256,
    420309,
    70872,
    419659,
    596026,
    55413,
    256204,
    120794,
    83442,
    692054,
    873496,
}
PILOT_V5_3_SEEDS = {
    20260730,
    733980,
    626956,
    183324,
    266579,
    559726,
    176907,
    35514,
    769350,
    589887,
    418549,
    589641,
    621567,
}
SCREEN_V5_3_SEEDS = {
    20260731,
    25987,
    293840,
    725284,
    249400,
    591103,
    257709,
}

TERMINAL_STATUSES = {
    "CLEAN_ELIGIBLE",
    "CLEAN_INELIGIBLE",
    "CLEAN_SUCCESS_EARLY_STOP",
    "NO_CLEAN_PREFIX",
    "PAIR_ELIGIBLE",
    "RECOVERY_INELIGIBLE",
    "PAIR_CAP_REACHED",
    "NON_CONSEQUENTIAL_PERTURBATION",
    "CONTROLLED_STRESS_TEST",
    "EXECUTION_ERROR",
}
EXECUTED_STATUSES = {
    "CLEAN_ELIGIBLE",
    "CLEAN_INELIGIBLE",
    "PAIR_ELIGIBLE",
    "RECOVERY_INELIGIBLE",
    "NON_CONSEQUENTIAL_PERTURBATION",
    "CONTROLLED_STRESS_TEST",
    "EXECUTION_ERROR",
}
PRIMARY_ERROR_TAXONOMY = {
    "wrong_identifier_same_type",
    "stale_identifier_same_type",
    "wrong_read_only_tool_same_schema",
    "schema_valid_argument_substitution",
}
READ_ONLY_PREFIXES = (
    "get_",
    "find_",
    "list_",
    "search_",
    "lookup_",
    "check_",
)
READ_ONLY_EXACT = {
    "calculate",
    "verify_identity",
}


class PilotProtocolError(RuntimeError):
    """V5.4 evidence is incomplete, inconsistent, or protocol-invalid."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_constants() -> None:
    rng = random.Random(BASE_SEED)
    observed = tuple(rng.randint(0, 1_000_000) for _ in TRIAL_SEEDS)
    if observed != TRIAL_SEEDS:
        raise PilotProtocolError("V5.4 seed derivation drift")
    if len(PILOT_TASK_IDS) != TASKS or len(set(PILOT_TASK_IDS)) != TASKS:
        raise PilotProtocolError("V5.4 task universe must be 24 unique tasks")
    if (
        {BASE_SEED, *TRIAL_SEEDS}
        & (FORMAL_V5_3_SEEDS | PILOT_V5_3_SEEDS | SCREEN_V5_3_SEEDS)
    ):
        raise PilotProtocolError("V5.4 seeds overlap a V5.3 schedule")
    if SLOTS_PER_TASK != 12 or MAX_EXECUTED_SLOTS != 288:
        raise PilotProtocolError("V5.4 slot budget drift")


def is_read_only_tool(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in READ_ONLY_EXACT or normalized.startswith(
        READ_ONLY_PREFIXES
    )


def slot_registry() -> list[dict[str, Any]]:
    """Return all 288 immutable slots before any outcome is observed."""
    validate_constants()
    rows: list[dict[str, Any]] = []
    for task_identity in PILOT_TASK_IDS:
        domain, task_id = task_identity.split(":", 1)
        for index, seed in enumerate(TRIAL_SEEDS):
            kind = "clean" if index < CLEAN_SLOTS_PER_TASK else "recovery"
            rows.append(
                {
                    "slot_id": f"{task_identity}:{kind}:{index + 1:02d}",
                    "task_identity": task_identity,
                    "domain": domain,
                    "task_id": task_id,
                    "slot_index": index + 1,
                    "slot_kind": kind,
                    "seed": seed,
                    "protocol": PROTOCOL,
                    "status": "REGISTERED",
                }
            )
    return rows


def select_injection_index(
    eligible_indices: Iterable[int],
    *,
    task_identity: str,
    clean_seed: int,
) -> int:
    """Select a location without consulting branch success or test outcomes."""
    ordered = sorted(set(eligible_indices))
    if not ordered:
        raise PilotProtocolError("clean trajectory has no read-only injection site")
    digest = hashlib.sha256(
        f"{PROTOCOL}|{task_identity}|{clean_seed}".encode("utf-8")
    ).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def real_error_checks(record: dict[str, Any]) -> dict[str, bool]:
    taxonomy = record.get("taxonomy")
    return {
        "schema_valid": record.get("schema_valid") is True,
        "clean_action_correct": record.get("clean_action_correct") is True,
        "injected_action_task_incorrect": (
            record.get("injected_action_task_incorrect") is True
        ),
        "consequential": record.get("consequential") is True,
        "naturalistic_taxonomy": taxonomy in PRIMARY_ERROR_TAXONOMY,
        "read_only_injection": is_read_only_tool(
            str(record.get("clean_tool_name", ""))
        ),
        "single_injected_error": record.get("injected_error_count") == 1,
    }


def leakage_checks(record: dict[str, Any]) -> dict[str, bool]:
    hashes = (
        "shared_prefix_sha256",
        "clean_future_sha256",
        "recovery_prompt_sha256",
        "error_event_sha256",
    )
    return {
        "future_removed": record.get("clean_future_present_in_prompt") is False,
        "prefix_identity": (
            record.get("recovery_prefix_sha256")
            == record.get("shared_prefix_sha256")
        ),
        "hashes_present": all(
            isinstance(record.get(key), str) and len(record[key]) == 64
            for key in hashes
        ),
        "failed_call_not_positive_label": (
            record.get("failed_call_supervised") is False
        ),
        "error_result_not_positive_label": (
            record.get("error_result_supervised") is False
        ),
    }


def validate_slot_records(records: list[dict[str, Any]]) -> None:
    expected = {row["slot_id"]: row for row in slot_registry()}
    observed = {str(row.get("slot_id")): row for row in records}
    if len(records) != MAX_EXECUTED_SLOTS or set(observed) != set(expected):
        raise PilotProtocolError("all 288 registered slots need one terminal row")
    for slot_id, row in observed.items():
        frozen = expected[slot_id]
        if any(row.get(key) != frozen[key] for key in ("task_identity", "seed", "slot_kind")):
            raise PilotProtocolError(f"slot identity drift: {slot_id}")
        if row.get("status") not in TERMINAL_STATUSES:
            raise PilotProtocolError(f"nonterminal or unknown status: {slot_id}")
    for task in PILOT_TASK_IDS:
        rows = sorted(
            (row for row in records if row["task_identity"] == task),
            key=lambda row: row["slot_index"],
        )
        clean = rows[:CLEAN_SLOTS_PER_TASK]
        recovery = rows[CLEAN_SLOTS_PER_TASK:]
        successes = [
            index for index, row in enumerate(clean)
            if row["status"] == "CLEAN_ELIGIBLE"
        ]
        if len(successes) > 1:
            raise PilotProtocolError(f"multiple accepted clean prefixes: {task}")
        if successes:
            accepted = successes[0]
            if any(
                row["status"] not in {"CLEAN_INELIGIBLE", "EXECUTION_ERROR"}
                for row in clean[:accepted]
            ):
                raise PilotProtocolError(f"invalid pre-success clean state: {task}")
            if any(
                row["status"] != "CLEAN_SUCCESS_EARLY_STOP"
                for row in clean[accepted + 1:]
            ):
                raise PilotProtocolError(f"clean early-stop drift: {task}")
            if any(row["status"] == "NO_CLEAN_PREFIX" for row in recovery):
                raise PilotProtocolError(f"false NO_CLEAN_PREFIX: {task}")
        else:
            if any(
                row["status"] not in {"CLEAN_INELIGIBLE", "EXECUTION_ERROR"}
                for row in clean
            ) or any(row["status"] != "NO_CLEAN_PREFIX" for row in recovery):
                raise PilotProtocolError(f"no-clean terminal state drift: {task}")
        pair_positions = [
            index for index, row in enumerate(recovery)
            if row["status"] == "PAIR_ELIGIBLE"
        ]
        if len(pair_positions) > MAX_PAIRS_PER_TASK:
            raise PilotProtocolError(f"pair cap exceeded: {task}")
        if len(pair_positions) == MAX_PAIRS_PER_TASK:
            second = pair_positions[-1]
            if any(
                row["status"] != "PAIR_CAP_REACHED"
                for row in recovery[second + 1:]
            ):
                raise PilotProtocolError(f"pair-cap early-stop drift: {task}")


def pilot_decision(
    slot_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
    *,
    v5_3_executed_rollouts: int = 288,
) -> dict[str, Any]:
    """Compute the preregistered V5.4 go/no-go decision, fail closed."""
    validate_slot_records(slot_records)
    slots_by_id = {row["slot_id"]: row for row in slot_records}
    claimed_pairs = [
        pair for pair in pair_records if pair.get("eligible") is True
    ]
    valid_pairs: list[dict[str, Any]] = []
    all_claimed_pairs_audit_valid = True
    for pair in claimed_pairs:
        slot_id = str(pair.get("slot_id"))
        slot = slots_by_id.get(slot_id)
        if slot is None or slot.get("status") != "PAIR_ELIGIBLE":
            all_claimed_pairs_audit_valid = False
            continue
        if pair.get("task_identity") != slot.get("task_identity"):
            all_claimed_pairs_audit_valid = False
            continue
        error = real_error_checks(pair)
        leakage = leakage_checks(pair)
        if not all(error.values()) or not all(leakage.values()):
            all_claimed_pairs_audit_valid = False
            continue
        valid_pairs.append(pair)
    valid_slot_ids = [str(pair.get("slot_id")) for pair in valid_pairs]
    if len(valid_slot_ids) != len(set(valid_slot_ids)):
        all_claimed_pairs_audit_valid = False
    eligible_slot_ids = {
        row["slot_id"] for row in slot_records
        if row["status"] == "PAIR_ELIGIBLE"
    }
    if set(valid_slot_ids) != eligible_slot_ids:
        all_claimed_pairs_audit_valid = False

    counts = Counter(str(pair.get("task_identity")) for pair in valid_pairs)
    if any(task not in PILOT_TASK_IDS for task in counts):
        raise PilotProtocolError("pair outside frozen task universe")
    capped = {task: min(counts[task], MAX_PAIRS_PER_TASK) for task in PILOT_TASK_IDS}
    tasks_with_pair = sum(value >= 1 for value in capped.values())
    capped_pairs = sum(capped.values())
    executed_slots = sum(
        row["status"] in EXECUTED_STATUSES for row in slot_records
    )
    v54_yield = capped_pairs / executed_slots if executed_slots else 0.0
    v53_yield = V5_3_OBSERVED_CAPPED_PAIRS / v5_3_executed_rollouts
    domains = {
        task.split(":", 1)[0] for task, value in capped.items() if value
    }
    checks = {
        "all_slots_terminal": True,
        "all_claimed_pairs_audit_valid": all_claimed_pairs_audit_valid,
        "zero_future_leakage": all(
            all(leakage_checks(pair).values()) for pair in claimed_pairs
        ),
        "zero_failed_positive_labels": all(
            pair.get("failed_call_supervised") is False
            and pair.get("error_result_supervised") is False
            for pair in claimed_pairs
        ),
        "tasks_with_pair_at_least_14": tasks_with_pair >= MIN_TASKS_WITH_PAIR,
        "capped_pairs_at_least_17": capped_pairs >= MIN_CAPPED_PAIRS,
        "beats_v5_3_tasks": tasks_with_pair > V5_3_OBSERVED_TASKS_WITH_PAIR,
        "beats_v5_3_pairs": capped_pairs > V5_3_OBSERVED_CAPPED_PAIRS,
        "slot_yield_beats_v5_3": v54_yield > v53_yield,
        "both_domains_represented": domains == {"retail", "airline"},
    }
    return {
        "protocol": PROTOCOL,
        "status": "GO_FULL_V5_4" if all(checks.values()) else "NO_GO_STOP",
        "training_authorized": False,
        "official_test_used": False,
        "checks": checks,
        "observed": {
            "tasks_with_pair": tasks_with_pair,
            "capped_pairs": capped_pairs,
            "executed_slots": executed_slots,
            "registered_slots": MAX_EXECUTED_SLOTS,
            "quality_adjusted_pairs_per_executed_slot": v54_yield,
            "v5_3_pairs_per_rollout": v53_yield,
            "domains_with_pair": sorted(domains),
        },
        "claim_boundary": {
            "data_feasibility_only": True,
            "training_result": False,
            "paper_level_confirmation": False,
            "official_test_sealed": True,
        },
    }


validate_constants()
