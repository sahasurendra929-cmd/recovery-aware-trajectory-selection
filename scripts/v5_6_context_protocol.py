#!/usr/bin/env python3
"""Frozen, dependency-free contract for the V5.6 error-context mechanism screen."""
from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL = "v5_6_error_context_mechanism_v1"
DATA_PROTOCOL = "v5_6_error_context_sft_data_v1"
DESIGN_VERSION = "5.6-context-mechanism"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TRAINING_SEEDS = (20260805,)
EVALUATION_SEEDS = (20260815, 20260816, 20260817)
ARMS = ("perfect_success", "repair_25_true", "repair_25_shuffled")
RECOVERY_RATIO = {"perfect_success": 0.0, "repair_25_true": 0.25, "repair_25_shuffled": 0.25}
SCHEDULE_ROWS = 512
BLOCK_SIZE = 4
MAX_SEQUENCE_TOKENS = 10_240
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05


class ContextProtocolError(RuntimeError):
    """The preregistered V5.6 mechanism contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_design() -> None:
    if len(ARMS) != 3 or set(RECOVERY_RATIO) != set(ARMS):
        raise ContextProtocolError("three-arm grid drift")
    if SCHEDULE_ROWS % BLOCK_SIZE or SCHEDULE_ROWS != 512:
        raise ContextProtocolError("schedule must consist of 128 four-row blocks")
    if MAX_SEQUENCE_TOKENS != 10_240:
        raise ContextProtocolError("V5.6 must retain the validated 10,240-token context contract")
    if RECOVERY_RATIO["perfect_success"] != 0.0 or any(
        RECOVERY_RATIO[arm] != 0.25 for arm in ARMS[1:]
    ):
        raise ContextProtocolError("recovery-dose contract drift")
    if len(TRAINING_SEEDS) != 1 or not EVALUATION_SEEDS:
        raise ContextProtocolError("screen seed contract drift")
    if set(TRAINING_SEEDS) & set(EVALUATION_SEEDS):
        raise ContextProtocolError("training/evaluation seed overlap")
    if BOOTSTRAP_REPLICATES < 1000 or not 0 < ALPHA < 1:
        raise ContextProtocolError("statistics contract drift")


def frozen_summary() -> dict[str, Any]:
    validate_design()
    return {
        "protocol": PROTOCOL,
        "design_version": DESIGN_VERSION,
        "model": MODEL,
        "tau2_commit": TAU2_COMMIT,
        "arms": {arm: {"recovery_supervised_token_ratio": RECOVERY_RATIO[arm]} for arm in ARMS},
        "primary_estimand": "mean_task_delta_context",
        "delta_context": "logp(correct_repair_action | true_error_context) - logp(correct_repair_action | shuffled_error_context)",
        "shuffle_rule": "deterministic within-domain task-derangement; donor task differs from target task",
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "evaluation_split": "derived_validation_only",
        "official_test_used": False,
        "official_test_sealed": True,
    }


if __name__ == "__main__":
    print(json.dumps(frozen_summary(), indent=2, sort_keys=True))
