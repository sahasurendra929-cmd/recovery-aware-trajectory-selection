#!/usr/bin/env python3
"""Frozen constants for the post-yield V5.3 low-support diagnostic.

This profile is intentionally downstream of a failed V5.3-12h strict data
gate.  It reuses the same complete, hash-bound generation snapshot without
relaxing trajectory eligibility or same-attempt-slot pairing.  Its only
purpose is an engineering/scientific diagnostic under visibly low training
support; it can never satisfy a formal V5.3 barrier or a paper-level claim.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import v5_3_12h_protocol as source
except ModuleNotFoundError:
    from scripts import v5_3_12h_protocol as source


PROTOCOL = "v5_3_low_support_diagnostic_v1"
DESIGN_VERSION = "5.3-low-support-diagnostic"
DATA_PROTOCOL = "v5_3_low_support_diagnostic_sft_data_v1"
REGISTRY_PROFILE = "v5_3_low_support_diagnostic"
SOURCE_SCREEN_PROTOCOL = source.PROTOCOL
SOURCE_DATA_PROTOCOL = source.DATA_PROTOCOL
SOURCE_RUNTIME_ROOT = "artifacts/v5_3_12h_screen"

BASE_SEED = source.BASE_SEED
TRIAL_SEEDS = source.TRIAL_SEEDS
ATTEMPTS_PER_TASK_PER_CONDITION = source.ATTEMPTS_PER_TASK_PER_CONDITION
TASKS = source.TASKS
CONDITIONS = source.CONDITIONS
EXPECTED_ROLLOUTS = source.EXPECTED_ROLLOUTS
NUM_GENERATION_SHARDS = source.NUM_GENERATION_SHARDS
PILOT_TASK_IDS = source.PILOT_TASK_IDS
MAX_PAIRS_PER_TASK = 2

MIN_DISTINCT_TASKS = 8
MIN_CAPPED_PAIRS = 10
SCHEDULE_ROWS = 512
OPTIMIZER_STEPS = 64
ARTIFACT_NAMESPACE = {
    "processed_root": "data/processed/v5_3_low_support_diagnostic",
    "results_root": "results/v5_3_low_support_diagnostic",
    "runtime_root": "artifacts/v5_3_low_support_diagnostic",
}

TRAINED_ARMS = ("perfect_success", "repair_50")
ARM_RECOVERY_ROW_RATIOS = {
    "perfect_success": 0.0,
    "repair_50": 0.5,
}
EVAL_ARMS = ("base_model", "perfect_success", "repair_50")
EVALUATION_TASKS = source.EVALUATION_TASKS
EVALUATION_CONDITIONS = source.EVALUATION_CONDITIONS
EVALUATION_TRIALS = source.EVALUATION_TRIALS
EVALUATION_SHARDS = 3
EVALUATION_ROLLOUTS = (
    len(EVAL_ARMS)
    * EVALUATION_TASKS
    * len(EVALUATION_CONDITIONS)
    * EVALUATION_TRIALS
)

MODEL_IDS = {
    "base_model": "openai/v5-3-low-support-base",
    "perfect_success": "openai/v5-3-low-support-perfect-success",
    "repair_50": "openai/v5-3-low-support-repair-50",
}
USER_JUDGE_MODEL = source.USER_JUDGE_MODEL
USER_JUDGE_MODEL_ID = "openai/v5-3-low-support-user-judge"
USER_JUDGE_REVISION = source.USER_JUDGE_REVISION

CLAIM_BOUNDARY = {
    "post_yield_exploratory_diagnostic": True,
    "trigger_requires_strict_gate_fail_closed": True,
    "low_support_is_explicit": True,
    "formal_v5_3_result": False,
    "paper_level_confirmation": False,
    "diagnostic_outputs_may_enter_formal_v5_3": False,
    "official_test_used": False,
    "official_test_sealed": True,
    "extensions_allowed": False,
    "may_satisfy_formal_v5_3_gate": False,
    "may_enter_formal_v5_3_training": False,
    "may_be_pooled_with_formal_v5_3": False,
    "may_select_hyperparameters_or_training_arms": False,
    "positive_result_guaranteed": False,
    "allowed_claim": (
        "post-yield, one-seed, low-support directional diagnostic on the "
        "frozen 21-task derived-validation split under the exact registered "
        "model, data, and evaluator scope"
    ),
}


class LowSupportProtocolError(RuntimeError):
    """The diagnostic protocol or its provenance drifted."""


def artifact_root(repo_root: Path, key: str) -> Path:
    """Resolve one exact isolated root; arbitrary output roots are forbidden."""

    if key not in ARTIFACT_NAMESPACE:
        raise LowSupportProtocolError(f"unknown artifact root key: {key}")
    return (repo_root.resolve() / ARTIFACT_NAMESPACE[key]).resolve()


def require_whole_run_source_lock(repo_root: Path) -> int:
    """Acquire/verify the inherited shared lock held for the whole diagnostic."""

    try:
        import fcntl
    except ModuleNotFoundError as error:
        raise LowSupportProtocolError(
            "low-support diagnostic requires Linux flock support"
        ) from error
    raw_fd = os.environ.get("V5_3_LOW_SUPPORT_LOCK_FD")
    try:
        descriptor = int(raw_fd or "")
    except ValueError as error:
        raise LowSupportProtocolError(
            "V5_3_LOW_SUPPORT_LOCK_FD must name the inherited lock descriptor"
        ) from error
    if descriptor < 3:
        raise LowSupportProtocolError(
            "V5_3_LOW_SUPPORT_LOCK_FD must be at least 3"
        )
    expected = (
        repo_root.resolve() / SOURCE_RUNTIME_ROOT / "controller.lock"
    ).resolve()
    try:
        proc_link = Path(f"/proc/self/fd/{descriptor}")
        if proc_link.is_symlink():
            observed = Path(os.readlink(proc_link)).resolve()
        elif hasattr(fcntl, "F_GETPATH"):
            raw_path = fcntl.fcntl(
                descriptor, fcntl.F_GETPATH, b"\0" * 1024
            )
            observed = Path(
                raw_path.split(b"\0", 1)[0].decode("utf-8")
            ).resolve()
        else:
            observed = Path(
                os.readlink(Path(f"/dev/fd/{descriptor}"))
            ).resolve()
    except (OSError, ValueError) as error:
        raise LowSupportProtocolError(
            "inherited whole-run source lock descriptor is unavailable"
        ) from error
    if observed != expected:
        raise LowSupportProtocolError(
            "whole-run lock descriptor targets the wrong source controller "
            f"lock: expected={expected}, observed={observed}"
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as error:
        raise LowSupportProtocolError(
            "source V5.3-12h controller owns the lock; diagnostic overlap "
            "is forbidden"
        ) from error
    return descriptor


def canonical(value: Any) -> str:
    return source.canonical(value)


def canonical_sha256(value: Any) -> str:
    return source.canonical_sha256(value)


def validate_constants() -> None:
    source.validate_constants()
    if (
        DESIGN_VERSION != "5.3-low-support-diagnostic"
        or DATA_PROTOCOL != "v5_3_low_support_diagnostic_sft_data_v1"
        or REGISTRY_PROFILE != "v5_3_low_support_diagnostic"
        or SOURCE_RUNTIME_ROOT != "artifacts/v5_3_12h_screen"
        or MIN_DISTINCT_TASKS != 8
        or MIN_CAPPED_PAIRS != 10
        or MAX_PAIRS_PER_TASK != 2
        or SCHEDULE_ROWS != 512
        or OPTIMIZER_STEPS != 64
        or ARTIFACT_NAMESPACE
        != {
            "processed_root": (
                "data/processed/v5_3_low_support_diagnostic"
            ),
            "results_root": "results/v5_3_low_support_diagnostic",
            "runtime_root": "artifacts/v5_3_low_support_diagnostic",
        }
        or TRAINED_ARMS != ("perfect_success", "repair_50")
        or EVAL_ARMS != ("base_model", "perfect_success", "repair_50")
        or EVALUATION_SHARDS != 3
        or EVALUATION_ROLLOUTS != 126
    ):
        raise LowSupportProtocolError("low-support constants drift")
    if set(ARM_RECOVERY_ROW_RATIOS) != set(TRAINED_ARMS):
        raise LowSupportProtocolError("low-support mixture arm set drift")
    if ARM_RECOVERY_ROW_RATIOS != {
        "perfect_success": 0.0,
        "repair_50": 0.5,
    }:
        raise LowSupportProtocolError("low-support mixture ratios drift")
    if CLAIM_BOUNDARY["formal_v5_3_result"] is not False:
        raise LowSupportProtocolError("formal claim boundary drift")
    if CLAIM_BOUNDARY["official_test_sealed"] is not True:
        raise LowSupportProtocolError("official-test seal drift")


def data_gate(task_pair_counts: dict[str, int]) -> dict[str, Any]:
    """Authorize only low-support diagnostics after the strict gate fails."""

    validate_constants()
    if set(task_pair_counts) != set(PILOT_TASK_IDS):
        raise LowSupportProtocolError("data gate task universe drift")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in task_pair_counts.values()
    ):
        raise LowSupportProtocolError(
            "data gate pair counts must be nonnegative integers"
        )
    capped = {
        task_id: min(value, MAX_PAIRS_PER_TASK)
        for task_id, value in task_pair_counts.items()
    }
    tasks_with_pair = sum(value >= 1 for value in capped.values())
    capped_pairs = sum(capped.values())
    source_strict_gate = source.data_gate(task_pair_counts)
    checks = {
        "source_strict_gate_is_fail_closed": (
            source_strict_gate["status"] == "FAIL_CLOSED"
            and source_strict_gate["training_authorized"] is False
        ),
        "distinct_tasks_at_least_8": tasks_with_pair >= MIN_DISTINCT_TASKS,
        "capped_pairs_at_least_10": capped_pairs >= MIN_CAPPED_PAIRS,
    }
    authorized = all(checks.values())
    return {
        "status": (
            "PASS_LOW_SUPPORT_DIAGNOSTIC"
            if authorized
            else "FAIL_CLOSED"
        ),
        "training_authorized": authorized,
        "checks": checks,
        "observed": {
            "tasks": TASKS,
            "tasks_with_pair": tasks_with_pair,
            "capped_pairs": capped_pairs,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        },
        "thresholds": {
            "minimum_distinct_tasks": MIN_DISTINCT_TASKS,
            "minimum_capped_pairs": MIN_CAPPED_PAIRS,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        },
        "source_strict_gate": source_strict_gate,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


validate_constants()
