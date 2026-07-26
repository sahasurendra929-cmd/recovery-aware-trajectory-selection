#!/usr/bin/env python3
"""Pure and runtime helpers for the frozen V5.3 train-only pilot.

This module deliberately does not modify the pinned tau2 checkout or any V5.2
source file.  Its functions are dependency-free and unit-testable.  Formal
generation and the pilot share the strict runtime judge implemented in
``v5_strict_nl_judge.py``; this module independently tests the same critical
JSON/cardinality contract and owns the pilot thresholds and diagnostics.

Malformed judge output is never converted to an empty list.  Consequently,
Python's vacuous ``all([])`` behavior cannot award a successful NL score.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence


PROTOCOL = "v5_3_train_only_feasibility_pilot"
SEED = 20260722
PILOT_BASE_SEED = 20260730
FORMAL_BASE_SEED = 20260722
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
PILOT_TASKS = 24
FORMAL_TRAIN_TASKS = 70
PILOT_MIN_ONE_PAIR_TASKS = 15
PILOT_MIN_TWO_PAIR_TASKS = 4
PILOT_MIN_ONE_PAIR_RATE = 15 / 24
PILOT_MIN_SECOND_PAIR_RATE = 4 / 24
PILOT_MIN_PROJECTED_TASKS = 43.75
PILOT_MIN_PROJECTED_PAIRS = 55 + 5 / 12
FORMAL_MIN_TASKS = 40
FORMAL_MIN_PAIRS = 48
MAX_PAIRS_PER_TASK = 2
ATTEMPTS_PER_TASK_PER_CONDITION = 12

TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
TEACHER_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
USER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
USER_JUDGE_REVISION = "539535859b135b0244c91f3e59816150c8056698"

GT_INCOMPATIBLE_INNER_TRAIN_TASKS = frozenset(
    {
        ("retail", "24"),
        ("airline", "0"),
        ("airline", "10"),
        ("airline", "28"),
        ("airline", "34"),
    }
)

PILOT_FAMILY_QUOTAS = {
    "retail_missing_product_id": 4,
    "retail_missing_order_id": 5,
    "retail_missing_user_email": 5,
    "retail_missing_user_id": 4,
    "airline_missing_flight_number": 2,
    "airline_missing_reservation_id": 2,
    "airline_missing_user_id": 2,
}

_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE,
)


class JudgeContractError(RuntimeError):
    """The judge response cannot be used as a scientific label."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_trial_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Reproduce pinned tau2's ``random.seed``/``randint`` trial schedule."""

    import random

    generator = random.Random(base_seed)
    return tuple(generator.randint(0, 1_000_000) for _ in range(count))


def validate_frozen_seed_schedules() -> None:
    if expected_trial_seeds(PILOT_BASE_SEED, 12) != PILOT_TRIAL_SEEDS:
        raise RuntimeError("V5.3 pilot trial seed schedule drift")
    if expected_trial_seeds(FORMAL_BASE_SEED, 12) != FORMAL_TRIAL_SEEDS:
        raise RuntimeError("V5.3 formal trial seed schedule drift")
    if set(PILOT_TRIAL_SEEDS) & set(FORMAL_TRIAL_SEEDS):
        raise RuntimeError("V5.3 pilot/formal trial seeds overlap")


def _extract_single_json_object(content: str) -> tuple[dict[str, Any], str]:
    if not isinstance(content, str) or not content.strip():
        raise JudgeContractError("judge response is empty or non-text")
    stripped = content.strip()
    match = _FENCE_RE.fullmatch(stripped)
    if match:
        payload_text = match.group("body").strip()
        method = "single_markdown_json_fence"
    else:
        if stripped.startswith("```") or stripped.endswith("```"):
            raise JudgeContractError("judge response has malformed Markdown fencing")
        payload_text = stripped
        method = "bare_json"
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise JudgeContractError(
            f"judge response is not one valid JSON value: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise JudgeContractError("judge response top level must be an object")
    return payload, method


def parse_judge_response(
    content: str,
    expected_outcomes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse and validate one complete judge response.

    The expected outcomes must be covered exactly once.  Provider row order
    may vary, so accepted rows are normalized back to registered order.  This
    rejects missing, duplicated, or invented assertions and makes it
    impossible for a nonempty assertion set to produce ``all([])``.
    """

    expected = list(expected_outcomes)
    if not expected or any(
        not isinstance(value, str) or not value for value in expected
    ):
        raise JudgeContractError("expected outcomes must be a nonempty string list")
    if len(set(expected)) != len(expected):
        raise JudgeContractError("expected outcomes contain duplicates")
    expected_set = set(expected)
    payload, method = _extract_single_json_object(content)
    if set(payload) != {"results"}:
        raise JudgeContractError("judge top level must contain only 'results'")
    results = payload["results"]
    if not isinstance(results, list):
        raise JudgeContractError("judge 'results' must be a list")
    if len(results) != len(expected):
        raise JudgeContractError(
            "judge result count does not match expected outcome count"
        )
    by_outcome: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise JudgeContractError(f"judge result {index} is not an object")
        if set(row) != {"expectedOutcome", "reasoning", "metExpectation"}:
            raise JudgeContractError(
                f"judge result {index} has an unexpected schema"
            )
        assertion = row["expectedOutcome"]
        if assertion not in expected_set:
            raise JudgeContractError(
                f"judge result {index} contains an unknown assertion"
            )
        if assertion in by_outcome:
            raise JudgeContractError(
                f"judge result {index} duplicates an assertion"
            )
        if type(row["metExpectation"]) is not bool:
            raise JudgeContractError(
                f"judge result {index}.metExpectation must be boolean"
            )
        reasoning = row["reasoning"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise JudgeContractError(
                f"judge result {index}.reasoning must be nonempty text"
            )
        by_outcome[assertion] = {
            "expectedOutcome": assertion,
            "reasoning": reasoning.strip(),
            "metExpectation": row["metExpectation"],
        }
    if set(by_outcome) != expected_set:
        raise JudgeContractError("judge results do not exactly cover assertions")
    normalized = [by_outcome[assertion] for assertion in expected]
    metadata = {
        "parse_method": method,
        "raw_sha256": sha256_text(content),
        "normalized_sha256": sha256_text(
            canonical_json({"results": normalized})
        ),
        "result_count": len(normalized),
        "met_count": sum(row["metExpectation"] for row in normalized),
    }
    return normalized, metadata


def classify_user_loop_db0(simulations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report, but never repair or relabel, early-stop/loop failure modes."""

    total = 0
    user_stop_db0 = 0
    repeated_user_text = 0
    max_steps = 0
    rows = []
    for simulation in simulations:
        total += 1
        reward_info = simulation.get("reward_info") or {}
        db_check = reward_info.get("db_check") or {}
        db_reward = db_check.get("db_reward")
        termination = simulation.get("termination_reason")
        messages = simulation.get("messages") or []
        normalized_user_messages: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            normalized = " ".join(
                content.replace("###STOP###", "")
                .replace("###TRANSFER###", "")
                .replace("###OUT-OF-SCOPE###", "")
                .lower()
                .split()
            )
            if normalized:
                normalized_user_messages.append(normalized)
        repeated = len(set(normalized_user_messages)) < len(normalized_user_messages)
        is_user_stop_db0 = termination == "user_stop" and db_reward == 0.0
        is_max_steps = termination == "max_steps"
        user_stop_db0 += is_user_stop_db0
        repeated_user_text += repeated
        max_steps += is_max_steps
        if is_user_stop_db0 or repeated or is_max_steps:
            rows.append(
                {
                    "task_id": str(simulation.get("task_id")),
                    "trial": simulation.get("trial"),
                    "seed": simulation.get("seed"),
                    "termination_reason": termination,
                    "user_stop_db0": is_user_stop_db0,
                    "repeated_user_text": repeated,
                    "max_steps": is_max_steps,
                }
            )
    return {
        "total_simulations": total,
        "user_stop_db0": user_stop_db0,
        "exact_or_normalized_user_repeat": repeated_user_text,
        "max_steps": max_steps,
        "diagnostic_only": True,
        "attempts_repaired_or_relabelled": 0,
        "rows": rows,
    }
