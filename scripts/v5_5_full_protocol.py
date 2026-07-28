#!/usr/bin/env python3
"""Dependency-free preregistration contract for the complete V5.5 study."""
from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL = "v5_5_full_recovery_dose_response_v1"
DESIGN_VERSION = "5.5-full"
DATA_PILOT_PROTOCOL = "v5_5_reference_grounded_counterfactual_recovery_v1"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REPLICATION_MODEL = "Qwen/Qwen2.5-14B-Instruct"

RECOVERY_RATIOS = {
    "r0_perfect": 0.00,
    "r25": 0.25,
    "r50": 0.50,
    "r75": 0.75,
    "r100_recovery": 1.00,
}
SCREEN_ARMS = ("r0_perfect", "r50", "r100_recovery")
CONFIRMATORY_ARMS = tuple(RECOVERY_RATIOS)
TRAINING_SEEDS = (20260805, 20260806, 20260807)
EVALUATION_SEEDS = (20260815, 20260816, 20260817)

INNER_TRAIN_TASKS = 83
DERIVED_VALIDATION_TASKS = 21
SEALED_TEST_TASKS = 60
TOTAL_BENCHMARK_TASKS = 164
REFERENCE_PILOT_TASKS = 24
REFERENCE_PILOT_PAIRS = 48
MIN_NATURAL_TRAIN_TASKS = 24
MIN_NATURAL_PAIRS = 48
PAPER_TARGET_NATURAL_TASKS = 30
PAPER_TARGET_NATURAL_PAIRS = 60
MAX_PAIRS_PER_TASK = 2

TRAINING_MIXTURE_BASIS = "supervised_token_mass"
FAILED_ACTION_LABELS = 0
CLEAN_NONINFERIORITY_MARGIN = 0.05
ALPHA = 0.05
BOOTSTRAP_REPLICATES = 10_000
PRIMARY_METRIC = "tau2_end_to_end_task_success"
PRIMARY_CONDITION = "controlled_error_in_family"
OFFICIAL_TEST_ACCESS = "once_after_arm_selection_and_protocol_freeze"


class FullProtocolError(RuntimeError):
    """The complete V5.5 preregistration is internally inconsistent."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def primary_estimand(success: dict[str, float], selected_arm: str) -> float:
    """Paired task-success difference against the perfect-only baseline."""
    if selected_arm not in RECOVERY_RATIOS or selected_arm == "r0_perfect":
        raise FullProtocolError("selected arm must be a non-zero recovery arm")
    return float(success[selected_arm]) - float(success["r0_perfect"])


def clean_noninferiority(success: dict[str, float], selected_arm: str) -> bool:
    if selected_arm not in RECOVERY_RATIOS:
        raise FullProtocolError("unknown selected arm")
    return (
        float(success[selected_arm])
        - float(success["r0_perfect"])
        >= -CLEAN_NONINFERIORITY_MARGIN - 1e-12
    )


def validation_utility(
    *,
    clean_success: float,
    error_success: float,
    baseline_clean_success: float,
    penalty: float = 1.0,
) -> float:
    clean_drop_beyond_margin = max(
        0.0,
        baseline_clean_success - clean_success - CLEAN_NONINFERIORITY_MARGIN,
    )
    return error_success - penalty * clean_drop_beyond_margin


def validate_design() -> None:
    if tuple(RECOVERY_RATIOS.values()) != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise FullProtocolError("dose-response grid drift")
    if set(SCREEN_ARMS) - set(CONFIRMATORY_ARMS):
        raise FullProtocolError("screen arms are not confirmatory arms")
    if len(TRAINING_SEEDS) != 3 or len(set(TRAINING_SEEDS)) != 3:
        raise FullProtocolError("three unique training seeds are required")
    if set(TRAINING_SEEDS) & set(EVALUATION_SEEDS):
        raise FullProtocolError("training and evaluation seeds overlap")
    if (
        REFERENCE_PILOT_PAIRS
        != REFERENCE_PILOT_TASKS * MAX_PAIRS_PER_TASK
    ):
        raise FullProtocolError("reference pilot pair count drift")
    if MIN_NATURAL_PAIRS < MIN_NATURAL_TRAIN_TASKS * MAX_PAIRS_PER_TASK:
        raise FullProtocolError("natural pair gate is below task coverage gate")
    if (
        PAPER_TARGET_NATURAL_PAIRS
        < PAPER_TARGET_NATURAL_TASKS * MAX_PAIRS_PER_TASK
    ):
        raise FullProtocolError("paper target is below task coverage target")
    if (
        INNER_TRAIN_TASKS
        + DERIVED_VALIDATION_TASKS
        + SEALED_TEST_TASKS
        != TOTAL_BENCHMARK_TASKS
    ):
        raise FullProtocolError("tau2 split counts do not cover 164 tasks")
    if FAILED_ACTION_LABELS != 0:
        raise FullProtocolError("failed actions must never be positive labels")
    if not 0.0 < ALPHA < 1.0 or BOOTSTRAP_REPLICATES < 1_000:
        raise FullProtocolError("statistical contract drift")


def frozen_summary() -> dict[str, Any]:
    validate_design()
    return {
        "protocol": PROTOCOL,
        "design_version": DESIGN_VERSION,
        "benchmark": {"tau2_commit": TAU2_COMMIT},
        "model": MODEL,
        "replication_model": REPLICATION_MODEL,
        "arms": RECOVERY_RATIOS,
        "screen_arms": list(SCREEN_ARMS),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "data_gate": {
            "minimum_natural_train_tasks": MIN_NATURAL_TRAIN_TASKS,
            "minimum_natural_pairs": MIN_NATURAL_PAIRS,
            "paper_target_natural_tasks": PAPER_TARGET_NATURAL_TASKS,
            "paper_target_natural_pairs": PAPER_TARGET_NATURAL_PAIRS,
            "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        },
        "evaluation": {
            "derived_validation_tasks": DERIVED_VALIDATION_TASKS,
            "sealed_test_tasks": SEALED_TEST_TASKS,
            "primary_metric": PRIMARY_METRIC,
            "primary_condition": PRIMARY_CONDITION,
            "clean_noninferiority_margin": CLEAN_NONINFERIORITY_MARGIN,
            "alpha": ALPHA,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "official_test_access": OFFICIAL_TEST_ACCESS,
        },
        "failed_action_labels": FAILED_ACTION_LABELS,
        "official_test_used_for_selection": False,
    }


def main() -> None:
    print(json.dumps(frozen_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
