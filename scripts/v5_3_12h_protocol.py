#!/usr/bin/env python3
"""Immutable constants and small audit helpers for the V5.3 12-hour screen.

This protocol is deliberately *not* the frozen V5.3 feasibility pilot.  It
uses a separate seed schedule and artifact namespace, and none of its raw
bytes, checkpoints, metrics, or decisions may satisfy a V5.3 formal barrier.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import random
from typing import Any


PROTOCOL = "v5_3_12h_exploratory_screen_v1"
DESIGN_VERSION = "5.3-12h-screen"
DATA_PROTOCOL = "v5_3_12h_screen_sft_data_v1"
GENERATION_SUBPROTOCOL = "v5_3_12h_screen_generation_v1"
REGISTRY_PROFILE = "v5_3_12h_screen"

BASE_SEED = 20260731
TRIAL_SEEDS = (
    25987,
    293840,
    725284,
    249400,
    591103,
    257709,
)
ATTEMPTS_PER_TASK_PER_CONDITION = 6
TASKS = 24
CONDITIONS = ("clean", "error")
EXPECTED_ROLLOUTS = TASKS * len(CONDITIONS) * ATTEMPTS_PER_TASK_PER_CONDITION
NUM_GENERATION_SHARDS = 3

MIN_TASKS_WITH_PAIR = 14
MIN_CAPPED_PAIRS = 17
MAX_PAIRS_PER_TASK = 2

TRAINED_ARMS = (
    "perfect_success",
    "failure_raw",
    "repair_50",
    "repair_100",
)
# The screen trains with batch size one and mean-reduced token loss, then
# averages eight microbatch losses per optimizer step.  Consequently each
# trajectory row, not each supervised token, is one mixture-weight unit.
ARM_RECOVERY_ROW_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_50": 0.5,
    "repair_100": 1.0,
}
CORE_EVAL_ARMS = ("base_model", "perfect_success", "repair_50")
EXTENSION_EVAL_ARMS = ("failure_raw", "repair_100")
ALL_EVAL_ARMS = (*CORE_EVAL_ARMS, *EXTENSION_EVAL_ARMS)
EVALUATION_TASKS = 21
EVALUATION_CONDITIONS = ("clean", "error")
EVALUATION_TRIALS = 1
CORE_EVALUATION_ROLLOUTS = (
    len(CORE_EVAL_ARMS)
    * EVALUATION_TASKS
    * len(EVALUATION_CONDITIONS)
    * EVALUATION_TRIALS
)
CORE_ROLLOUTS = CORE_EVALUATION_ROLLOUTS
EXTENSION_EVALUATION_ROLLOUTS = (
    len(EXTENSION_EVAL_ARMS)
    * EVALUATION_TASKS
    * len(EVALUATION_CONDITIONS)
    * EVALUATION_TRIALS
)

HOUR_9 = 9
HARD_RESULT_TARGET_HOURS = 12
TASK_BOOTSTRAP_SEED = 20260731
TASK_BOOTSTRAP_REPLICATES = 100_000
CONFIDENCE_LEVEL = 0.95
ERROR_SUCCESS_GAIN_COUNT = 2
CLEAN_SUCCESS_MAX_LOSS_COUNT = 1

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

FORMAL_BASE_SEED = 20260722
FORMAL_TRIAL_SEEDS = (
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
)
PILOT_BASE_SEED = 20260730
PILOT_TRIAL_SEEDS = (
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
)

MODEL_IDS = {
    "base_model": "openai/v5-3-12h-base",
    "perfect_success": "openai/v5-3-12h-perfect-success",
    "failure_raw": "openai/v5-3-12h-failure-raw",
    "repair_50": "openai/v5-3-12h-repair-50",
    "repair_100": "openai/v5-3-12h-repair-100",
}
USER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
USER_JUDGE_MODEL_ID = "openai/v5-3-12h-user-judge"
USER_JUDGE_REVISION = "539535859b135b0244c91f3e59816150c8056698"

CLAIM_BOUNDARY = {
    "exploratory_screen_only": True,
    "formal_v5_3_result": False,
    "screen_outputs_may_enter_formal_v5_3": False,
    "paper_level_confirmation": False,
    "official_test_used": False,
    "official_test_sealed": True,
    "positive_result_guaranteed": False,
    "allowed_claim": (
        "one-seed exploratory derived-validation directional screen under "
        "the exact registered task, model, data, and evaluator scope"
    ),
}


class ScreenProtocolError(RuntimeError):
    """The isolated screen protocol was malformed or drifted."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_constants() -> None:
    derived = random.Random(BASE_SEED)
    observed = tuple(derived.randint(0, 1_000_000) for _ in range(6))
    if observed != TRIAL_SEEDS:
        raise ScreenProtocolError(
            f"screen trial-seed derivation drift: {observed}"
        )
    if len(PILOT_TASK_IDS) != TASKS or len(set(PILOT_TASK_IDS)) != TASKS:
        raise ScreenProtocolError("screen task IDs must be 24 unique values")
    if (
        set(TRIAL_SEEDS) & set(FORMAL_TRIAL_SEEDS)
        or set(TRIAL_SEEDS) & set(PILOT_TRIAL_SEEDS)
        or BASE_SEED in {FORMAL_BASE_SEED, PILOT_BASE_SEED}
    ):
        raise ScreenProtocolError("screen seeds overlap frozen V5.3 seeds")
    if EXPECTED_ROLLOUTS != 288 or CORE_EVALUATION_ROLLOUTS != 126:
        raise ScreenProtocolError("screen rollout arithmetic drift")
    if set(TRAINED_ARMS) != set(MODEL_IDS) - {"base_model"}:
        raise ScreenProtocolError("screen trained-arm alias set drift")
    if (
        set(ARM_RECOVERY_ROW_RATIOS) != set(TRAINED_ARMS)
        or any(
            ratio * 512 != int(ratio * 512)
            for ratio in ARM_RECOVERY_ROW_RATIOS.values()
        )
    ):
        raise ScreenProtocolError("screen row-weighted arm mixture drift")


def data_gate(task_pair_counts: dict[str, int]) -> dict[str, Any]:
    if set(task_pair_counts) != set(PILOT_TASK_IDS):
        raise ScreenProtocolError("data gate task universe drift")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in task_pair_counts.values()
    ):
        raise ScreenProtocolError("data gate pair counts must be nonnegative ints")
    capped = {
        task_id: min(value, MAX_PAIRS_PER_TASK)
        for task_id, value in task_pair_counts.items()
    }
    tasks_with_pair = sum(value >= 1 for value in capped.values())
    capped_pairs = sum(capped.values())
    checks = {
        "tasks_with_pair_at_least_14": tasks_with_pair
        >= MIN_TASKS_WITH_PAIR,
        "capped_pairs_at_least_17": capped_pairs >= MIN_CAPPED_PAIRS,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "training_authorized": all(checks.values()),
        "checks": checks,
        "observed": {
            "tasks": TASKS,
            "tasks_with_pair": tasks_with_pair,
            "capped_pairs": capped_pairs,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        },
        "thresholds": {
            "minimum_tasks_with_pair": MIN_TASKS_WITH_PAIR,
            "minimum_capped_pairs": MIN_CAPPED_PAIRS,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        },
    }


def make_deadline_receipt(started_at: datetime) -> dict[str, Any]:
    if started_at.tzinfo is None:
        raise ScreenProtocolError("screen start time must be timezone-aware")
    started = started_at.astimezone(timezone.utc)
    payload = {
        "protocol": f"{PROTOCOL}:deadline_v1",
        "started_at_utc": started.isoformat(),
        "extension_admission_deadline_utc": (
            started + timedelta(hours=HOUR_9)
        ).isoformat(),
        "core_result_target_deadline_utc": (
            started + timedelta(hours=HARD_RESULT_TARGET_HOURS)
        ).isoformat(),
        "extension_rule": (
            "run failure_raw and repair_100 iff the complete core evaluation "
            "receipt exists by the hour-9 deadline; never inspect metrics"
        ),
        "result_independent": True,
        "official_test_used": False,
    }
    return {**payload, "canonical_sha256": canonical_sha256(payload)}


def validate_deadline_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "protocol",
        "started_at_utc",
        "extension_admission_deadline_utc",
        "core_result_target_deadline_utc",
        "extension_rule",
        "result_independent",
        "official_test_used",
        "canonical_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ScreenProtocolError("deadline receipt schema drift")
    body = {
        key: value
        for key, value in receipt.items()
        if key != "canonical_sha256"
    }
    if receipt.get("canonical_sha256") != canonical_sha256(body):
        raise ScreenProtocolError("deadline receipt SHA drift")
    try:
        started = datetime.fromisoformat(str(receipt["started_at_utc"]))
        extension = datetime.fromisoformat(
            str(receipt["extension_admission_deadline_utc"])
        )
        core = datetime.fromisoformat(
            str(receipt["core_result_target_deadline_utc"])
        )
    except ValueError as error:
        raise ScreenProtocolError("deadline receipt timestamp drift") from error
    if any(value.tzinfo is None for value in (started, extension, core)):
        raise ScreenProtocolError("deadline receipt timestamps must be aware")
    expected = make_deadline_receipt(started)
    if receipt != expected:
        raise ScreenProtocolError("deadline receipt derived values drift")
    return receipt


def extension_admission(
    deadline_receipt: dict[str, Any],
    *,
    core_completed_at: datetime,
    core_complete: bool,
) -> dict[str, Any]:
    validate_deadline_receipt(deadline_receipt)
    if core_completed_at.tzinfo is None:
        raise ScreenProtocolError("core completion time must be timezone-aware")
    if not core_complete:
        allowed = False
        reason = "CORE_INCOMPLETE"
    else:
        deadline = datetime.fromisoformat(
            str(deadline_receipt["extension_admission_deadline_utc"])
        )
        completed = core_completed_at.astimezone(timezone.utc)
        allowed = completed <= deadline
        reason = "BEFORE_HOUR_9" if allowed else "AFTER_HOUR_9"
    return {
        "protocol": f"{PROTOCOL}:extension_admission_v1",
        "extension_authorized": allowed,
        "reason": reason,
        "uses_metric_values": False,
        "official_test_used": False,
    }


def directional_gate(
    *,
    perfect_error_successes: int,
    repair_error_successes: int,
    perfect_clean_successes: int,
    repair_clean_successes: int,
) -> dict[str, Any]:
    values = (
        perfect_error_successes,
        repair_error_successes,
        perfect_clean_successes,
        repair_clean_successes,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= EVALUATION_TASKS
        for value in values
    ):
        raise ScreenProtocolError("directional counts must be ints in 0..21")
    error_delta = repair_error_successes - perfect_error_successes
    clean_delta = repair_clean_successes - perfect_clean_successes
    checks = {
        "error_gain_at_least_2_of_21": error_delta
        >= ERROR_SUCCESS_GAIN_COUNT,
        "clean_loss_at_most_1_of_21": clean_delta
        >= -CLEAN_SUCCESS_MAX_LOSS_COUNT,
    }
    positive = all(checks.values())
    return {
        "status": "DIRECTIONAL_POSITIVE_SCREEN"
        if positive
        else "EXPLORATORY_INCONCLUSIVE",
        "checks": checks,
        "counts": {
            "perfect_error_successes": perfect_error_successes,
            "repair_50_error_successes": repair_error_successes,
            "error_success_delta_count": error_delta,
            "perfect_clean_successes": perfect_clean_successes,
            "repair_50_clean_successes": repair_clean_successes,
            "clean_success_delta_count": clean_delta,
        },
        "rates": {
            "error_success_delta": error_delta / EVALUATION_TASKS,
            "clean_success_delta": clean_delta / EVALUATION_TASKS,
        },
        "primary_metric": "tau2_composite_end_to_end_task_success",
        "replay_metric_role": "diagnostic_only",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


validate_constants()
