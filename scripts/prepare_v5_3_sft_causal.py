#!/usr/bin/env python3
"""Build the V5.3 task-level, cross-attempt SFT screen.

V5.3 deliberately leaves the historical V5 builder unchanged.  It reuses
V5's trajectory eligibility, causal label masking, tokenization, and schedule
helpers, but changes two data-design decisions:

* five structurally ground-truth-incompatible task IDs are excluded before
  sharding, then the remaining inner-train IDs are split *before generation
  outcomes are read* into 70 arm-train and 8 loss-validation tasks; and
* each condition gets twelve attempts.  Eligible clean and post-fault
  attempts are independently hash ordered within a task, then zipped by rank.
  Thus clean/error attempts need not share a seed.

No official-test task may enter this module.  A failed 40-task/48-pair gate
writes an audit-only bundle and raises before any arm JSONL is created.
The four arms share task support and matched budgets, but their exact
pair-occurrence schedules may differ. Consequently ``failure_raw`` is a
descriptive negative-exposure control, not an isolated causal test of label
masking against ``repair_100``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    import prepare_v5_sft_causal as v5
except ModuleNotFoundError:
    from scripts import prepare_v5_sft_causal as v5

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


DESIGN_PROTOCOL = "v5_3_task_level_cross_seed_sft_screen"
DESIGN_VERSION = "5.3"
GENERATION_MANIFEST_PROTOCOL = "v5_3_multifault_data_construction"
PARTITION_PROTOCOL = "v5_3_inner_train_loss_validation_split"
PARTITION_SALT = "v5.3-loss-validation"
PAIR_ORDER_SALT = "v5.3-eligible-attempt-order"
ATTEMPTS_PER_TASK = 12
MAX_PAIRS_PER_TASK = 2
MIN_TRAIN_TASKS = 40
MIN_TRAIN_PAIRS = 48
MAX_SEQUENCE_TOKENS = 8192
EXPECTED_GENERATION_SHARDS = 3
EXPECTED_TEACHER_API_BASE_BY_SHARD = {
    0: "http://127.0.0.1:8011/v1",
    1: "http://127.0.0.1:8011/v1",
    2: "http://127.0.0.1:8011/v1",
}
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
TEACHER_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
USER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
USER_JUDGE_REVISION = "539535859b135b0244c91f3e59816150c8056698"
GENERATION_TEMPERATURE = 0.2
GENERATION_TOP_P = 0.95
DERIVED_TRIAL_SEEDS = [
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
]
LOSS_VALIDATION_COUNTS = {"retail": 6, "airline": 2}
GT_INCOMPATIBLE_TASK_IDS = {
    "airline:0",
    "airline:10",
    "airline:28",
    "airline:34",
    "retail:24",
}
EXPECTED_PARTITION_SHA256 = (
    "c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818"
)
EXPECTED_LOSS_VALIDATION_IDS = {
    "retail": ["22", "23", "52", "63", "96", "103"],
    "airline": ["15", "47"],
}


class PoolGateError(RuntimeError):
    """Raised after writing an audit-only FAIL_CLOSED bundle."""


def _numeric_task_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError as error:
        raise RuntimeError(f"V5.3 task ID is not numeric: {task_id!r}") from error


def _compact_sha256(value: Any) -> str:
    return hashlib.sha256(v5.canonical(value).encode("utf-8")).hexdigest()


def build_task_partition(
    split: dict[str, Any],
    *,
    split_manifest_sha256: str,
    seed: int = v5.SEED,
    strict_formal: bool = True,
) -> dict[str, Any]:
    """Create the frozen 70/8 partition without observing rollout outcomes."""

    domains: dict[str, Any] = {}
    all_train: set[str] = set()
    all_loss: set[str] = set()
    for domain in ("retail", "airline"):
        row = split["domains"][domain]
        inner = [
            str(value)
            for value in row["inner_train_ids"]
            if f"{domain}:{value}" not in GT_INCOMPATIBLE_TASK_IDS
        ]
        if len(inner) != len(set(inner)):
            raise RuntimeError(f"{domain}: duplicate inner-train task IDs")
        ordered = sorted(
            inner,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{PARTITION_SALT}|{seed}|{domain}|{task_id}".encode("utf-8")
                ).hexdigest(),
                _numeric_task_key(task_id),
            ),
        )
        count = LOSS_VALIDATION_COUNTS[domain]
        loss_ids = sorted(ordered[:count], key=_numeric_task_key)
        train_ids = sorted(ordered[count:], key=_numeric_task_key)
        train_keys = {f"{domain}:{task_id}" for task_id in train_ids}
        loss_keys = {f"{domain}:{task_id}" for task_id in loss_ids}
        if train_keys & loss_keys or train_keys | loss_keys != {
            f"{domain}:{task_id}" for task_id in inner
        }:
            raise RuntimeError(f"{domain}: V5.3 partition is not a disjoint cover")
        all_train.update(train_keys)
        all_loss.update(loss_keys)
        domains[domain] = {
            "train_ids": train_ids,
            "loss_validation_ids": loss_ids,
            "counts": {
                "train": len(train_ids),
                "loss_validation": len(loss_ids),
            },
        }
    payload = {
        "protocol": PARTITION_PROTOCOL,
        "seed": seed,
        "source_split_manifest_sha256": split_manifest_sha256,
        "domains": domains,
    }
    digest = _compact_sha256(payload)
    if strict_formal:
        if len(all_train) != 70 or len(all_loss) != 8:
            raise RuntimeError(
                "V5.3 partition count drift: "
                f"train={len(all_train)}, loss_validation={len(all_loss)}"
            )
        observed_loss = {
            domain: domains[domain]["loss_validation_ids"]
            for domain in ("retail", "airline")
        }
        if observed_loss != EXPECTED_LOSS_VALIDATION_IDS:
            raise RuntimeError(
                f"V5.3 frozen loss-validation IDs drift: {observed_loss}"
            )
        if digest != EXPECTED_PARTITION_SHA256:
            raise RuntimeError(
                "V5.3 frozen partition digest drift: "
                f"observed={digest}, expected={EXPECTED_PARTITION_SHA256}"
            )
    return {
        **payload,
        "canonical_sha256": digest,
        "selection_inputs": ["domain", "task_id", "protocol_seed"],
        "ground_truth_incompatible_task_ids": sorted(
            GT_INCOMPATIBLE_TASK_IDS
        ),
        "ground_truth_compatibility_uses_generation_outcomes": False,
        "selection_uses_generation_outcomes": False,
        "selection_uses_validation_or_test": False,
        "official_test_used": False,
    }


def _attempt_index(simulation: dict[str, Any]) -> int:
    values = [
        simulation.get(name)
        for name in ("attempt_index", "attempt", "trial")
        if simulation.get(name) is not None
    ]
    if not values:
        raise RuntimeError("simulation lacks attempt_index/attempt/trial")
    try:
        index = int(values[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid attempt index: {values[0]!r}") from error
    if any(int(value) != index for value in values[1:]):
        raise RuntimeError("simulation attempt_index/attempt/trial disagree")
    if not 0 <= index < ATTEMPTS_PER_TASK:
        raise RuntimeError(
            f"attempt index {index} is outside [0,{ATTEMPTS_PER_TASK - 1}]"
        )
    return index


def _attempt_seed(simulation: dict[str, Any]) -> str:
    seed = simulation.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise RuntimeError("simulation requires a scalar deterministic seed")
    value = str(seed)
    if not value:
        raise RuntimeError("simulation seed is empty")
    return value


def attempt_identity(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return the only fields allowed to influence candidate ordering."""

    return {
        "attempt_index": _attempt_index(simulation),
        "attempt_seed": _attempt_seed(simulation),
    }


def attempt_order_key(
    *,
    domain: str,
    task_id: str,
    condition: str,
    simulation: dict[str, Any],
    seed: int = v5.SEED,
) -> tuple[str, int, str]:
    identity = attempt_identity(simulation)
    digest = hashlib.sha256(
        (
            f"{PAIR_ORDER_SALT}|{seed}|{domain}|{task_id}|{condition}|"
            f"{identity['attempt_index']}|{identity['attempt_seed']}"
        ).encode("utf-8")
    ).hexdigest()
    return digest, identity["attempt_index"], identity["attempt_seed"]


def _load_attempts(
    raw_dir: Path,
    split: dict[str, Any],
    *,
    allowed_task_ids: set[str],
) -> tuple[
    dict[str, dict[str, dict[int, dict[str, Any]]]],
    list[Path],
]:
    """Load exactly twelve attempts per task and condition.

    Unlike V5, clean and error seeds are intentionally not required to match.
    """

    by_condition: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        "clean": defaultdict(dict),
        "error": defaultdict(dict),
    }
    paths: list[Path] = []
    expected_indices = set(range(ATTEMPTS_PER_TASK))
    for domain in ("retail", "airline"):
        allowed = {
            str(value)
            for value in split["domains"][domain]["inner_train_ids"]
            if f"{domain}:{value}" in allowed_task_ids
        }
        for condition in ("clean", "error"):
            for path in v5._raw_paths(raw_dir, domain, condition):
                paths.append(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                simulations = payload.get("simulations")
                if not isinstance(simulations, list):
                    raise RuntimeError(f"{path}: simulations must be a list")
                for simulation in simulations:
                    if not isinstance(simulation, dict):
                        raise RuntimeError(f"{path}: simulation must be an object")
                    task_id = str(simulation.get("task_id"))
                    if task_id not in allowed:
                        raise RuntimeError(
                            f"{path}: {domain}:{task_id} is not inner-train"
                        )
                    index = _attempt_index(simulation)
                    _attempt_seed(simulation)
                    task_key = f"{domain}:{task_id}"
                    if index in by_condition[condition][task_key]:
                        raise RuntimeError(
                            f"duplicate {condition} attempt {task_key}:{index}"
                        )
                    by_condition[condition][task_key][index] = simulation
        for task_id in allowed:
            task_key = f"{domain}:{task_id}"
            for condition in ("clean", "error"):
                observed = set(by_condition[condition].get(task_key, {}))
                if observed != expected_indices:
                    raise RuntimeError(
                        f"{task_key}/{condition}: attempts differ from "
                        f"0..{ATTEMPTS_PER_TASK - 1}; observed={sorted(observed)}"
                    )
    if set(by_condition["clean"]) != set(by_condition["error"]):
        raise RuntimeError("V5.3 clean/error task coverage differs")
    return by_condition, sorted(set(paths))


def _canonical_trajectory_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove provider call-ID noise without removing trajectory semantics."""

    normalized: list[dict[str, Any]] = []
    call_ids: dict[str, str] = {}
    next_call = 1
    for message_index, message in enumerate(messages):
        role = message.get("role")
        # The training constructor installs one frozen system message later.
        # A provider-emitted copy is therefore not part of rollout identity.
        if role == "system":
            continue
        compact: dict[str, Any] = {
            "role": role,
            "content": message.get("content"),
        }
        calls = message.get("tool_calls") or []
        if calls:
            compact_calls = []
            for call_index, call in enumerate(calls):
                name, arguments, original_id = v5._function_call(
                    call,
                    (
                        f"messages[{message_index}]"
                        f".tool_calls[{call_index}]"
                    ),
                )
                if original_id not in call_ids:
                    call_ids[original_id] = f"call_{next_call:04d}"
                    next_call += 1
                compact_calls.append(
                    {
                        "id": call_ids[original_id],
                        "name": name,
                        "arguments": arguments,
                    }
                )
            compact["tool_calls"] = compact_calls
        if role == "tool":
            original_id = v5._tool_link_id(message)
            if original_id is None or original_id not in call_ids:
                raise RuntimeError(
                    "tool result cannot be canonicalized without linked call"
                )
            compact["tool_call_id"] = call_ids[original_id]
            compact["error"] = message.get("error")
            if message.get("name") is not None:
                compact["name"] = message["name"]
        normalized.append(compact)
    return normalized


def semantic_trajectory_sha256(messages: list[dict[str, Any]]) -> str:
    return v5.sha256_text(
        v5.canonical(_canonical_trajectory_messages(messages))
    )


def _rank_and_filter(
    *,
    domain: str,
    task_id: str,
    condition: str,
    attempts: dict[int, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    ordered = sorted(
        attempts.values(),
        key=lambda simulation: attempt_order_key(
            domain=domain,
            task_id=task_id,
            condition=condition,
            simulation=simulation,
            seed=seed,
        ),
    )
    eligible: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_trajectories: set[str] = set()
    for rank, simulation in enumerate(ordered):
        analysis = v5.analyze_simulation(simulation, condition=condition)
        identity = attempt_identity(simulation)
        trajectory_sha = semantic_trajectory_sha256(
            analysis.get("messages") or []
        )
        reason = analysis["reason"]
        accepted = bool(analysis["eligible"])
        if accepted and trajectory_sha in seen_trajectories:
            accepted = False
            reason = "duplicate_eligible_trajectory"
        if accepted:
            seen_trajectories.add(trajectory_sha)
            eligible.append(
                {
                    "simulation": simulation,
                    "analysis": analysis,
                    "identity": identity,
                    "rank": rank,
                    "trajectory_sha256": trajectory_sha,
                }
            )
        reasons[reason] += 1
        audit_rows.append(
            {
                "domain": domain,
                "task_id": task_id,
                "source_split": "derived_inner_train",
                "condition": condition,
                **identity,
                "attempt_id": str(identity["attempt_index"]),
                "stable_rank": rank,
                "stable_order_sha256": attempt_order_key(
                    domain=domain,
                    task_id=task_id,
                    condition=condition,
                    simulation=simulation,
                    seed=seed,
                )[0],
                "trajectory_sha256": trajectory_sha,
                "scoreable": not reason.startswith("invalid_simulation_"),
                "eligible": accepted,
                "reason": reason,
            }
        )
    return eligible, audit_rows, reasons


def _check_injected_fault(
    analysis: dict[str, Any],
    fault: dict[str, Any],
    *,
    where: str,
) -> None:
    injected = analysis["injected"]
    if len(injected) != 1:
        raise RuntimeError(f"{where}: eligible error rollout lacks one injection")
    observed = injected[0]["name"], v5.canonical(injected[0]["arguments"])
    expected = fault["tool_name"], v5.canonical(fault["arguments"])
    if observed != expected:
        raise RuntimeError(
            f"{where}: injected fault differs from frozen task manifest"
        )


def _filter_training_compatible_attempts(
    *,
    domain: str,
    task_id: str,
    condition: str,
    eligible: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    reasons: Counter[str],
    fault: dict[str, Any],
    context: dict[str, Any],
    tokenizer: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply the frozen no-truncation 8,192-token contract before pairing."""

    audit_by_identity = {
        (row["attempt_index"], row["attempt_seed"]): row
        for row in audit_rows
    }
    compatible: list[dict[str, Any]] = []
    variants = (
        ("perfect_success",)
        if condition == "clean"
        else ("failure_raw", "repair_masked")
    )
    for candidate in eligible:
        identity = candidate["identity"]
        if condition == "error":
            _check_injected_fault(
                candidate["analysis"],
                fault,
                where=(
                    f"{domain}:{task_id}:attempt-"
                    f"{identity['attempt_index']}:training-fit"
                ),
            )
        sequence_tokens: dict[str, int] = {}
        for variant in variants:
            source = v5.make_source_example(
                domain=domain,
                task_id=task_id,
                trial=f"training-fit-{identity['attempt_index']}",
                pair_id=(
                    f"{domain}:{task_id}:training-fit:"
                    f"{condition}:{identity['attempt_index']}"
                ),
                analysis=candidate["analysis"],
                context=context,
                variant=variant,
                fault=fault,
                tokenizer=tokenizer,
                seed=seed,
            )
            sequence_tokens[variant] = int(
                source["token_contract"]["sequence_tokens"]
            )
        candidate["training_sequence_tokens"] = sequence_tokens
        audit_row = audit_by_identity[
            (identity["attempt_index"], identity["attempt_seed"])
        ]
        audit_row["training_sequence_tokens"] = sequence_tokens
        if max(sequence_tokens.values()) > MAX_SEQUENCE_TOKENS:
            audit_row["eligible"] = False
            audit_row["reason"] = "training_sequence_exceeds_8192"
            reasons["eligible"] -= 1
            reasons["training_sequence_exceeds_8192"] += 1
            continue
        compatible.append(candidate)
    if reasons.get("eligible") == 0:
        del reasons["eligible"]
    return compatible


def _materialize_pair(
    *,
    domain: str,
    task_id: str,
    pair_rank: int,
    clean: dict[str, Any],
    error: dict[str, Any],
    fault: dict[str, Any],
    context: dict[str, Any],
    tokenizer: Any,
    seed: int,
    partition: str,
) -> dict[str, Any]:
    identity = {
        "domain": domain,
        "task_id": task_id,
        "clean": clean["identity"],
        "error": error["identity"],
    }
    pair_hash = _compact_sha256(identity)
    pair_id = f"{domain}:{task_id}:v5.3:{pair_rank}:{pair_hash[:16]}"
    trial = (
        f"clean-{clean['identity']['attempt_index']}-"
        f"error-{error['identity']['attempt_index']}"
    )
    variants = {
        "perfect_success": v5.make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=clean["analysis"],
            context=context,
            variant="perfect_success",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        ),
        "failure_raw": v5.make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=error["analysis"],
            context=context,
            variant="failure_raw",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        ),
        "repair_masked": v5.make_source_example(
            domain=domain,
            task_id=task_id,
            trial=trial,
            pair_id=pair_id,
            analysis=error["analysis"],
            context=context,
            variant="repair_masked",
            fault=fault,
            tokenizer=tokenizer,
            seed=seed,
        ),
    }
    overlong = {
        name: source["token_contract"]["sequence_tokens"]
        for name, source in variants.items()
        if source["token_contract"]["sequence_tokens"] > MAX_SEQUENCE_TOKENS
    }
    if overlong:
        raise RuntimeError(
            f"{pair_id}: training sequence exceeds "
            f"{MAX_SEQUENCE_TOKENS}: {overlong}"
        )
    pair_contract = {
        "protocol": "v5_3_independent_eligible_rank_zip",
        "pair_rank": pair_rank,
        "pair_identity_sha256": pair_hash,
        "clean": clean["identity"],
        "error": error["identity"],
        "same_seed": clean["identity"]["attempt_seed"]
        == error["identity"]["attempt_seed"],
        "selection_uses_validation_or_test": False,
    }
    for source in variants.values():
        source["metadata"].update(
            {
                "design_protocol": DESIGN_PROTOCOL,
                "design_version": DESIGN_VERSION,
                "v5_3_partition": partition,
                "pair_contract": deepcopy(pair_contract),
                "clean_attempt_index": clean["identity"]["attempt_index"],
                "clean_attempt_seed": clean["identity"]["attempt_seed"],
                "error_attempt_index": error["identity"]["attempt_index"],
                "error_attempt_seed": error["identity"]["attempt_seed"],
            }
        )
    return {
        "pair_id": pair_id,
        "pair_identity_sha256": pair_hash,
        "domain": domain,
        "task_id": task_id,
        "partition": partition,
        "pair_rank": pair_rank,
        "clean_attempt": clean["identity"],
        "error_attempt": error["identity"],
        **variants,
    }


def _write_audit_bundle(
    *,
    output_dir: Path,
    partition: dict[str, Any],
    attempt_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    v5.write_json(output_dir / "task_partition.json", partition)
    v5.write_jsonl(output_dir / "attempt_audit.jsonl", attempt_rows)
    v5.write_json(output_dir / "audit.json", audit)
    hashes = {
        name: v5.sha256_file(output_dir / name)
        for name in ("task_partition.json", "attempt_audit.jsonl", "audit.json")
    }
    v5.write_json(output_dir / "hashes.json", hashes)


def _refine_schedule(
    task_ids: list[str],
    initial: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    *,
    target_tokens: int,
    target_sequence: int,
    target_recovery_ratio: float,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    """Deterministically improve V5's greedy two-budget schedule.

    With two source pairs per task, the original one-pass chooser can leave a
    feasible shared target several percent away. Coordinate refinement never
    changes the common task schedule or candidate support and does not inspect
    validation outcomes.
    """

    rows = list(initial)
    total_tokens = sum(
        row["token_contract"]["supervised_tokens"] for row in rows
    )
    total_sequence = sum(
        row["token_contract"]["sequence_tokens"] for row in rows
    )
    recovery_tokens = sum(
        row["token_contract"]["supervised_tokens"]
        for row in rows
        if row["metadata"]["source"] == "failure_rich"
    )

    def score(tokens: int, sequence: int, recovery: int) -> tuple[float, float, float]:
        return (
            abs(tokens - target_tokens) / max(1, target_tokens)
            + abs(sequence - target_sequence) / max(1, target_sequence)
            + 4.0
            * abs(recovery / max(1, tokens) - target_recovery_ratio),
            abs(tokens - target_tokens) / max(1, target_tokens),
            abs(sequence - target_sequence) / max(1, target_sequence),
        )

    position_order = sorted(
        range(len(rows)),
        key=lambda position: (
            hashlib.sha256(
                f"{seed}|{salt}|position|{position}".encode("utf-8")
            ).hexdigest(),
            position,
        ),
    )
    for _ in range(24):
        changed = False
        for position in position_order:
            current = rows[position]
            current_tokens = current["token_contract"]["supervised_tokens"]
            current_sequence = current["token_contract"]["sequence_tokens"]
            current_recovery = (
                current_tokens
                if current["metadata"]["source"] == "failure_rich"
                else 0
            )
            ordered_options = v5.deterministic_order(
                candidates[task_ids[position]],
                seed,
                f"{salt}:refine:{position}",
            )
            best = (
                score(total_tokens, total_sequence, recovery_tokens),
                v5.sha256_text(f"{seed}:{salt}:{position}:{current['id']}"),
                current,
                total_tokens,
                total_sequence,
                recovery_tokens,
            )
            for option in ordered_options:
                option_tokens = option["token_contract"]["supervised_tokens"]
                option_sequence = option["token_contract"]["sequence_tokens"]
                option_recovery = (
                    option_tokens
                    if option["metadata"]["source"] == "failure_rich"
                    else 0
                )
                candidate_tokens = total_tokens - current_tokens + option_tokens
                candidate_sequence = (
                    total_sequence - current_sequence + option_sequence
                )
                candidate_recovery = (
                    recovery_tokens - current_recovery + option_recovery
                )
                candidate = (
                    score(
                        candidate_tokens,
                        candidate_sequence,
                        candidate_recovery,
                    ),
                    v5.sha256_text(
                        f"{seed}:{salt}:{position}:{option['id']}"
                    ),
                    option,
                    candidate_tokens,
                    candidate_sequence,
                    candidate_recovery,
                )
                if candidate[:2] < best[:2]:
                    best = candidate
            if best[2]["id"] != current["id"]:
                rows[position] = best[2]
                total_tokens = best[3]
                total_sequence = best[4]
                recovery_tokens = best[5]
                changed = True
        if not changed:
            break
    return rows


def _relative_range(values: list[int]) -> float:
    """Return the registered max-minus-min range relative to the minimum."""

    if not values or min(values) <= 0:
        raise RuntimeError("matching totals must be positive")
    return (max(values) - min(values)) / min(values)


def _interval_distance(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _source_recovery_tokens(row: dict[str, Any]) -> int:
    source = row["metadata"]["source"]
    if source == "failure_rich":
        return row["token_contract"]["supervised_tokens"]
    if source == "perfect_success":
        return 0
    raise RuntimeError(f"unknown matching source type: {source!r}")


def _beam_arm_schedules(
    task_ids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    *,
    target_tokens: float,
    target_sequence: float,
    target_recovery_ratio: float | None,
    target_recovery_row_ratio: float | None = None,
    seed: int,
    salt: str,
    beam_width: int,
    final_width: int,
) -> list[dict[str, Any]]:
    """Bounded deterministic multiple-choice DP for one arm.

    The state contains supervised, nonpadding, recovery-token, and recovery-row
    totals.  The row dimension is needed by the 12-hour screen because its
    batch-size-one mean loss gives each trajectory equal optimization weight;
    supervised-token mass is not the mixture weight in that profile.
    Candidate order is hash-derived from protocol inputs.  Rewards, validation
    results, and official-test outcomes are not inputs to either ordering or
    pruning.
    """

    if not task_ids or beam_width <= 0 or final_width <= 0:
        raise RuntimeError("invalid tolerance-aware matching search budget")
    option_lists: list[list[dict[str, Any]]] = []
    for position, task_id in enumerate(task_ids):
        options = candidates.get(task_id) or []
        if not options:
            raise RuntimeError(f"matching task has no candidates: {task_id}")
        option_lists.append(
            v5.deterministic_order(
                options,
                seed,
                f"{salt}:beam:{position}:{task_id}",
            )
        )

    size = len(task_ids)
    suffix_token_min = [0] * (size + 1)
    suffix_token_max = [0] * (size + 1)
    suffix_sequence_min = [0] * (size + 1)
    suffix_sequence_max = [0] * (size + 1)
    suffix_recovery_min = [0] * (size + 1)
    suffix_recovery_max = [0] * (size + 1)
    suffix_recovery_rows_min = [0] * (size + 1)
    suffix_recovery_rows_max = [0] * (size + 1)
    for position in range(size - 1, -1, -1):
        options = option_lists[position]
        token_values = [
            row["token_contract"]["supervised_tokens"] for row in options
        ]
        sequence_values = [
            row["token_contract"]["sequence_tokens"] for row in options
        ]
        recovery_values = [_source_recovery_tokens(row) for row in options]
        recovery_row_values = [
            int(row["metadata"]["source"] == "failure_rich")
            for row in options
        ]
        suffix_token_min[position] = (
            suffix_token_min[position + 1] + min(token_values)
        )
        suffix_token_max[position] = (
            suffix_token_max[position + 1] + max(token_values)
        )
        suffix_sequence_min[position] = (
            suffix_sequence_min[position + 1] + min(sequence_values)
        )
        suffix_sequence_max[position] = (
            suffix_sequence_max[position + 1] + max(sequence_values)
        )
        suffix_recovery_min[position] = (
            suffix_recovery_min[position + 1] + min(recovery_values)
        )
        suffix_recovery_max[position] = (
            suffix_recovery_max[position + 1] + max(recovery_values)
        )
        suffix_recovery_rows_min[position] = (
            suffix_recovery_rows_min[position + 1]
            + min(recovery_row_values)
        )
        suffix_recovery_rows_max[position] = (
            suffix_recovery_rows_max[position + 1]
            + max(recovery_row_values)
        )

    def optimistic_score(
        tokens: int,
        sequence: int,
        recovery: int,
        recovery_rows: int,
        next_position: int,
    ) -> tuple[float, float, float, float, float]:
        token_distance = _interval_distance(
            target_tokens,
            tokens + suffix_token_min[next_position],
            tokens + suffix_token_max[next_position],
        ) / max(1.0, target_tokens)
        sequence_distance = _interval_distance(
            target_sequence,
            sequence + suffix_sequence_min[next_position],
            sequence + suffix_sequence_max[next_position],
        ) / max(1.0, target_sequence)
        token_ratio_distance = 0.0
        if target_recovery_ratio is not None:
            minimum_ratio = (
                recovery + suffix_recovery_min[next_position]
            ) / max(1, tokens + suffix_token_max[next_position])
            maximum_ratio = (
                recovery + suffix_recovery_max[next_position]
            ) / max(1, tokens + suffix_token_min[next_position])
            token_ratio_distance = _interval_distance(
                target_recovery_ratio,
                minimum_ratio,
                maximum_ratio,
            )
        row_ratio_distance = 0.0
        if target_recovery_row_ratio is not None:
            minimum_row_ratio = (
                recovery_rows + suffix_recovery_rows_min[next_position]
            ) / size
            maximum_row_ratio = (
                recovery_rows + suffix_recovery_rows_max[next_position]
            ) / size
            row_ratio_distance = _interval_distance(
                target_recovery_row_ratio,
                minimum_row_ratio,
                maximum_row_ratio,
            )
        return (
            token_distance
            + sequence_distance
            + 4.0 * token_ratio_distance
            + 4.0 * row_ratio_distance,
            token_distance,
            sequence_distance,
            token_ratio_distance,
            row_ratio_distance,
        )

    # Nodes retain only selected beam predecessors, avoiding an O(n^2) copy of
    # growing choice tuples.  Node zero is the root.
    nodes: list[tuple[int, dict[str, Any] | None]] = [(-1, None)]
    states: list[tuple[int, int, int, int, int]] = [(0, 0, 0, 0, 0)]
    for position, options in enumerate(option_lists):
        # Equal totals have equivalent future reachability.  Keeping the first
        # hash-ordered path is therefore lossless for the registered budgets.
        expanded: dict[
            tuple[int, int, int, int],
            tuple[int, dict[str, Any]],
        ] = {}
        for tokens, sequence, recovery, recovery_rows, node_id in states:
            for option in options:
                option_tokens = option["token_contract"]["supervised_tokens"]
                option_sequence = option["token_contract"]["sequence_tokens"]
                option_is_recovery = int(
                    option["metadata"]["source"] == "failure_rich"
                )
                key = (
                    tokens + option_tokens,
                    sequence + option_sequence,
                    recovery + _source_recovery_tokens(option),
                    recovery_rows + option_is_recovery,
                )
                expanded.setdefault(key, (node_id, option))
        ranked = sorted(
            expanded.items(),
            key=lambda item: (
                optimistic_score(*item[0], position + 1),
                item[0],
            ),
        )[:beam_width]
        states = []
        for totals, (parent_id, option) in ranked:
            nodes.append((parent_id, option))
            states.append((*totals, len(nodes) - 1))

    ranked_final = sorted(
        states,
        key=lambda state: (
            abs(state[0] - target_tokens) / max(1.0, target_tokens)
            + abs(state[1] - target_sequence) / max(1.0, target_sequence)
            + (
                0.0
                if target_recovery_ratio is None
                else 4.0
                * abs(
                    state[2] / max(1, state[0])
                    - target_recovery_ratio
                )
            )
            + (
                0.0
                if target_recovery_row_ratio is None
                else 4.0
                * abs(state[3] / size - target_recovery_row_ratio)
            ),
            abs(state[0] - target_tokens),
            abs(state[1] - target_sequence),
            (
                0.0
                if target_recovery_ratio is None
                else abs(
                    state[2] / max(1, state[0])
                    - target_recovery_ratio
                )
            ),
            (
                0.0
                if target_recovery_row_ratio is None
                else abs(state[3] / size - target_recovery_row_ratio)
            ),
            state[:4],
        ),
    )[:final_width]
    schedules: list[dict[str, Any]] = []
    for tokens, sequence, recovery, recovery_rows, node_id in ranked_final:
        rows: list[dict[str, Any]] = []
        while node_id:
            parent_id, row = nodes[node_id]
            assert row is not None
            rows.append(row)
            node_id = parent_id
        rows.reverse()
        if len(rows) != len(task_ids):
            raise RuntimeError("matching beam predecessor chain is incomplete")
        schedules.append(
            {
                "rows": rows,
                "supervised_tokens": tokens,
                "nonpadding_tokens": sequence,
                "recovery_supervised_tokens": recovery,
                "recovery_supervised_token_ratio": (
                    recovery / max(1, tokens)
                ),
                "recovery_rows": recovery_rows,
                "recovery_row_ratio": recovery_rows / size,
                "selection_sha256": _compact_sha256(
                    [row["id"] for row in rows]
                ),
            }
        )
    return schedules


def _matching_target_points(
    endpoint_bounds_by_arm: dict[str, dict[str, tuple[int, int]]],
    *,
    expected_arms: tuple[str, ...] = v5.CORE_ARMS,
) -> list[tuple[float, float]]:
    """Construct deterministic search landmarks without requiring overlap."""

    def landmarks(field: str) -> tuple[float, float, float, float, float]:
        bounds = [
            endpoint_bounds_by_arm[arm][field] for arm in expected_arms
        ]
        lower_bridge = max(row[0] for row in bounds)
        upper_bridge = min(row[1] for row in bounds)
        gap_center = (lower_bridge + upper_bridge) / 2.0
        arm_midpoint = float(median([(low + high) / 2.0 for low, high in bounds]))
        outer_center = (
            min(low for low, _ in bounds) + max(high for _, high in bounds)
        ) / 2.0
        return (
            gap_center,
            arm_midpoint,
            float(lower_bridge),
            float(upper_bridge),
            outer_center,
        )

    tokens = landmarks("supervised_tokens")
    sequences = landmarks("nonpadding_tokens")
    proposed = [
        (tokens[0], sequences[0]),
        (tokens[1], sequences[1]),
        (tokens[0], sequences[1]),
        (tokens[1], sequences[0]),
        (tokens[2], sequences[2]),
        (tokens[3], sequences[3]),
        (tokens[4], sequences[4]),
    ]
    result: list[tuple[float, float]] = []
    for point in proposed:
        if point not in result:
            result.append(point)
    return result


def _deterministic_tolerance_aware_joint_match(
    task_ids: list[str],
    arm_candidates: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    expected_arms: tuple[str, ...] = v5.CORE_ARMS,
    recovery_ratios: dict[str, float] | None,
    recovery_row_ratios: dict[str, float] | None = None,
    seed: int,
    supervised_tolerance: float = 0.01,
    sequence_tolerance: float = 0.02,
    recovery_ratio_tolerance: float = 0.01,
    recovery_row_ratio_tolerance: float = 0.0,
    enforce_cross_arm_budget_tolerances: bool = True,
    beam_width_schedule: tuple[int, ...] = (128, 512),
    final_width: int = 16,
) -> tuple[dict[str, list[dict[str, Any]]] | None, dict[str, Any]]:
    """Search jointly for the exact registered tolerance-compatible arms.

    This is a bounded search, not an integer-programming proof system.
    Therefore failure is explicitly reported as inconclusive rather than as
    evidence that no feasible schedule exists.
    """

    if (
        not expected_arms
        or len(expected_arms) != len(set(expected_arms))
        or set(arm_candidates) != set(expected_arms)
    ):
        raise RuntimeError("matching candidate arms drift")
    if recovery_ratios is None and recovery_row_ratios is None:
        raise RuntimeError("matching requires a registered recovery mixture")
    for name, values in (
        ("recovery token ratios", recovery_ratios),
        ("recovery row ratios", recovery_row_ratios),
    ):
        if values is not None and (
            not set(expected_arms) <= set(values)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in values.values()
            )
        ):
            raise RuntimeError(f"matching {name} drift")
    endpoint_bounds_by_arm = {
        arm: {
            "supervised_tokens": v5.endpoint_bounds(
                task_ids,
                arm_candidates[arm],
                "supervised_tokens",
            ),
            "nonpadding_tokens": v5.endpoint_bounds(
                task_ids,
                arm_candidates[arm],
                "sequence_tokens",
            ),
        }
        for arm in expected_arms
    }
    target_points = _matching_target_points(
        endpoint_bounds_by_arm,
        expected_arms=expected_arms,
    )
    search_attempts: list[dict[str, Any]] = []
    best_observed: dict[str, Any] | None = None

    for beam_width in beam_width_schedule:
        for target_tokens, target_sequence in target_points:
            finalists = {
                arm: _beam_arm_schedules(
                    task_ids,
                    arm_candidates[arm],
                    target_tokens=target_tokens,
                    target_sequence=target_sequence,
                    target_recovery_ratio=(
                        recovery_ratios[arm]
                        if recovery_ratios is not None
                        else None
                    ),
                    target_recovery_row_ratio=(
                        recovery_row_ratios[arm]
                        if recovery_row_ratios is not None
                        else None
                    ),
                    seed=seed,
                    salt=f"v5.3:{arm}",
                    beam_width=beam_width,
                    final_width=final_width,
                )
                for arm in expected_arms
            }
            attempt_best: dict[str, Any] | None = None
            feasible: list[
                tuple[
                    tuple[float, float, float, tuple[str, ...]],
                    tuple[dict[str, Any], ...],
                    dict[str, Any],
                ]
            ] = []
            for combination in product(
                *(finalists[arm] for arm in expected_arms)
            ):
                token_gap = _relative_range(
                    [row["supervised_tokens"] for row in combination]
                )
                sequence_gap = _relative_range(
                    [row["nonpadding_tokens"] for row in combination]
                )
                ratio_deviations = (
                    [
                        abs(
                            row["recovery_supervised_token_ratio"]
                            - recovery_ratios[arm]
                        )
                        for arm, row in zip(expected_arms, combination)
                    ]
                    if recovery_ratios is not None
                    else [0.0] * len(expected_arms)
                )
                row_ratio_deviations = (
                    [
                        abs(
                            row["recovery_row_ratio"]
                            - recovery_row_ratios[arm]
                        )
                        for arm, row in zip(expected_arms, combination)
                    ]
                    if recovery_row_ratios is not None
                    else [0.0] * len(expected_arms)
                )
                identity = tuple(row["selection_sha256"] for row in combination)
                score = (
                    token_gap / max(supervised_tolerance, 1e-12)
                    + sequence_gap / max(sequence_tolerance, 1e-12)
                    + sum(
                        value / max(recovery_ratio_tolerance, 1e-12)
                        for value in ratio_deviations
                    )
                    + sum(
                        value / max(recovery_row_ratio_tolerance, 1e-12)
                        for value in row_ratio_deviations
                    ),
                    token_gap,
                    sequence_gap,
                    identity,
                )
                observation = {
                    "supervised_token_relative_range": token_gap,
                    "nonpadding_token_relative_range": sequence_gap,
                    "maximum_recovery_ratio_deviation": max(ratio_deviations),
                    "maximum_recovery_row_ratio_deviation": max(
                        row_ratio_deviations
                    ),
                    "selection_sha256_by_arm": {
                        arm: row["selection_sha256"]
                        for arm, row in zip(expected_arms, combination)
                    },
                }
                if attempt_best is None or score < attempt_best["_score"]:
                    attempt_best = {**observation, "_score": score}
                if (
                    (
                        not enforce_cross_arm_budget_tolerances
                        or (
                            token_gap <= supervised_tolerance + 1e-12
                            and sequence_gap <= sequence_tolerance + 1e-12
                        )
                    )
                    and max(ratio_deviations)
                    <= recovery_ratio_tolerance + 1e-12
                    and max(row_ratio_deviations)
                    <= recovery_row_ratio_tolerance + 1e-12
                ):
                    feasible.append((score, combination, observation))
            attempt_record = {
                "beam_width": beam_width,
                "target_supervised_tokens": target_tokens,
                "target_nonpadding_tokens": target_sequence,
                "finalists_per_arm": {
                    arm: len(finalists[arm]) for arm in expected_arms
                },
                "best_observed": (
                    {
                        key: value
                        for key, value in attempt_best.items()
                        if key != "_score"
                    }
                    if attempt_best is not None
                    else None
                ),
            }
            search_attempts.append(attempt_record)
            if attempt_best is not None and (
                best_observed is None
                or attempt_best["_score"] < best_observed["_score"]
            ):
                best_observed = attempt_best
            if feasible:
                _, selected, observation = min(feasible, key=lambda row: row[0])
                schedules = {
                    arm: candidate["rows"]
                    for arm, candidate in zip(expected_arms, selected)
                }
                certificate = {
                    "algorithm": (
                        "deterministic_tolerance_aware_joint_beam_dp_v2"
                    ),
                    "search_status": "FEASIBLE_MATCH_FOUND",
                    "beam_width_schedule": list(beam_width_schedule),
                    "successful_beam_width": beam_width,
                    "target_points": [
                        {
                            "supervised_tokens": token_point,
                            "nonpadding_tokens": sequence_point,
                        }
                        for token_point, sequence_point in target_points
                    ],
                    "selected_target_supervised_tokens": target_tokens,
                    "selected_target_nonpadding_tokens": target_sequence,
                    "endpoint_bounds_by_arm": endpoint_bounds_by_arm,
                    "supervised_tokens_by_arm": {
                        arm: row["supervised_tokens"]
                        for arm, row in zip(expected_arms, selected)
                    },
                    "nonpadding_tokens_by_arm": {
                        arm: row["nonpadding_tokens"]
                        for arm, row in zip(expected_arms, selected)
                    },
                    "recovery_supervised_token_ratio_by_arm": {
                        arm: row["recovery_supervised_token_ratio"]
                        for arm, row in zip(expected_arms, selected)
                    },
                    "recovery_row_ratio_by_arm": {
                        arm: row["recovery_row_ratio"]
                        for arm, row in zip(expected_arms, selected)
                    },
                    "recovery_rows_by_arm": {
                        arm: row["recovery_rows"]
                        for arm, row in zip(expected_arms, selected)
                    },
                    "recovery_mixture_basis": (
                        "row_mean_microbatch_equal_weight"
                        if recovery_row_ratios is not None
                        else "supervised_token_mass"
                    ),
                    "cross_arm_budget_tolerances_gating": (
                        enforce_cross_arm_budget_tolerances
                    ),
                    "actual_supervised_token_relative_range": observation[
                        "supervised_token_relative_range"
                    ],
                    "actual_nonpadding_token_relative_range": observation[
                        "nonpadding_token_relative_range"
                    ],
                    "tolerances": {
                        "supervised_token_relative_range": supervised_tolerance,
                        "nonpadding_token_relative_range": sequence_tolerance,
                        "recovery_supervised_token_ratio": (
                            recovery_ratio_tolerance
                            if recovery_ratios is not None
                            else None
                        ),
                        "recovery_row_ratio": (
                            recovery_row_ratio_tolerance
                            if recovery_row_ratios is not None
                            else None
                        ),
                    },
                    "search_attempts": search_attempts,
                    "candidate_order_uses_reward_loss_validation_or_test": False,
                    "mathematical_infeasibility_claimed": False,
                }
                return schedules, certificate

    certificate = {
        "algorithm": "deterministic_tolerance_aware_joint_beam_dp_v2",
        "search_status": "MATCHING_SEARCH_INCONCLUSIVE",
        "beam_width_schedule": list(beam_width_schedule),
        "target_points": [
            {
                "supervised_tokens": token_point,
                "nonpadding_tokens": sequence_point,
            }
            for token_point, sequence_point in target_points
        ],
        "endpoint_bounds_by_arm": endpoint_bounds_by_arm,
        "best_observed": (
            {
                key: value
                for key, value in best_observed.items()
                if key != "_score"
            }
            if best_observed is not None
            else None
        ),
        "tolerances": {
            "supervised_token_relative_range": supervised_tolerance,
            "nonpadding_token_relative_range": sequence_tolerance,
            "recovery_supervised_token_ratio": (
                recovery_ratio_tolerance
                if recovery_ratios is not None
                else None
            ),
            "recovery_row_ratio": (
                recovery_row_ratio_tolerance
                if recovery_row_ratios is not None
                else None
            ),
        },
        "recovery_mixture_basis": (
            "row_mean_microbatch_equal_weight"
            if recovery_row_ratios is not None
            else "supervised_token_mass"
        ),
        "cross_arm_budget_tolerances_gating": (
            enforce_cross_arm_budget_tolerances
        ),
        "search_attempts": search_attempts,
        "candidate_order_uses_reward_loss_validation_or_test": False,
        "search_space_exhausted": False,
        "mathematical_infeasibility_claimed": False,
    }
    return None, certificate


def prepare(
    *,
    split_manifest: Path,
    generation_manifest: Path,
    raw_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    domain_contexts: dict[str, dict[str, Any]],
    validation_manifest: Path | None = None,
    generation_dynamic_audit: Path | None = None,
    validation_dynamic_audit: Path | None = None,
    expected_source_commit: str | None = None,
    expected_generation_source_commit: str | None = None,
    seed: int = v5.SEED,
    schedule_rows: int = v5.SCHEDULE_ROWS,
    strict_split_counts: bool = True,
    strict_generation_contracts: bool = True,
    strict_dynamic_audits: bool = True,
    token_ratio_tolerance: float = 0.01,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"output directory must be absent: {output_dir}")
    if not tokenizer_revision:
        raise RuntimeError("V5.3 requires a pinned tokenizer revision")
    split = v5.load_split(split_manifest, strict_counts=strict_split_counts)
    split_sha = v5.sha256_file(split_manifest)
    partition = build_task_partition(
        split,
        split_manifest_sha256=split_sha,
        seed=seed,
        strict_formal=strict_split_counts,
    )
    historical_inner_train_task_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["inner_train_ids"]
    }
    all_task_ids = historical_inner_train_task_ids - GT_INCOMPATIBLE_TASK_IDS
    if (
        len(historical_inner_train_task_ids) != 83
        or len(all_task_ids) != 78
        or not GT_INCOMPATIBLE_TASK_IDS <= historical_inner_train_task_ids
    ):
        raise RuntimeError("V5.3 effective inner-train universe drift")
    generation_payload = json.loads(
        generation_manifest.read_text(encoding="utf-8")
    )
    faults = v5.validate_multifault_manifest(
        generation_payload,
        expected_protocol=GENERATION_MANIFEST_PROTOCOL,
        expected_source_split="derived_inner_train",
        expected_task_ids=all_task_ids,
        split_manifest_sha256=split_sha,
    )
    expected_gt_filter = {
        "protocol": "v5_3_gt_compatibility_filter_v1",
        "policy": "exclude_before_sharding",
        "teacher_mode": "ground_truth",
        "source_task_count": 83,
        "included_task_count": 78,
        "excluded_task_ids": sorted(GT_INCOMPATIBLE_TASK_IDS),
        "exclusion_reason": "no_expected_tool_actions",
        "selection_uses_rollouts_rewards_validation_or_test": False,
        "official_test_used": False,
    }
    if generation_payload.get("gt_compatibility_filter") != expected_gt_filter:
        raise RuntimeError("V5.3 GT compatibility filter contract drift")
    dynamic_identities: dict[str, Any] = {}
    if strict_dynamic_audits:
        if (
            generation_dynamic_audit is None
            or validation_dynamic_audit is None
            or validation_manifest is None
        ):
            raise RuntimeError("formal V5.3 preparation requires both dynamic audits")
        dynamic_identities["generation"] = load_complete_dynamic_audit(
            generation_dynamic_audit,
            manifest_path=generation_manifest,
            split_manifest_path=split_manifest,
            expected_source_split="derived_inner_train",
            expected_task_ids=all_task_ids,
        )
        validation_payload = v5.load_validation_manifest(
            validation_manifest,
            split=split,
            split_manifest_path=split_manifest,
        )
        dynamic_identities["validation"] = load_complete_dynamic_audit(
            validation_dynamic_audit,
            manifest_path=validation_manifest,
            split_manifest_path=split_manifest,
            expected_source_split="derived_validation",
            expected_task_ids={
                f"{row['domain']}:{row['task_id']}"
                for row in validation_payload["rows"]
            },
        )
    else:
        validation_payload = (
            json.loads(validation_manifest.read_text(encoding="utf-8"))
            if validation_manifest is not None
            else None
        )
    generation_contract_audit = None
    if strict_generation_contracts:
        if expected_source_commit is None:
            raise RuntimeError("formal V5.3 preparation needs expected source commit")
        generation_contract_audit = v5.validate_generation_contracts(
            raw_dir=raw_dir,
            split_manifest=split_manifest,
            split=split,
            generation_manifest=generation_manifest,
            expected_trials=ATTEMPTS_PER_TASK,
            expected_seed=seed,
            expected_source_commit=expected_source_commit,
            expected_generation_source_commit=expected_generation_source_commit,
            dynamic_audit_identity=dynamic_identities["generation"],
            expected_shards=EXPECTED_GENERATION_SHARDS,
            expected_teacher_api_base_by_shard=(
                EXPECTED_TEACHER_API_BASE_BY_SHARD
            ),
            expected_task_ids=all_task_ids,
            expected_manifest_protocol=GENERATION_MANIFEST_PROTOCOL,
            expected_teacher={
                "model": TEACHER_MODEL,
                "revision": TEACHER_REVISION,
                "mode": "ground_truth",
            },
            expected_user={
                "model": USER_JUDGE_MODEL,
                "revision": USER_JUDGE_REVISION,
            },
            expected_judge={
                "model": USER_JUDGE_MODEL,
                "revision": USER_JUDGE_REVISION,
                "protocol": "v5_strict_nl_judge_v1",
                "content_attempts": 2,
                "schema_failure": "fail_closed",
                "raw_response_audit": True,
            },
            expected_decoding={
                "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P,
                "max_tokens": 512,
                "parallel_tool_calls": False,
                "parallel_tool_call_normalization": (
                    "execute_first_then_replan"
                ),
                "mixed_tool_call_content_normalization": (
                    "drop_text_preserve_sha256"
                ),
                "max_steps": 60,
                "task_timeout_seconds": 900,
                "seed": v5.SEED,
                "derived_trial_seeds": DERIVED_TRIAL_SEEDS,
            },
            expected_contract_fields={
                "generation_manifest_protocol": (
                    GENERATION_MANIFEST_PROTOCOL
                ),
                "gt_compatibility_filter": expected_gt_filter,
            },
        )
    pools, raw_paths = _load_attempts(
        raw_dir,
        split,
        allowed_task_ids=all_task_ids,
    )
    train_task_keys = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in partition["domains"][domain]["train_ids"]
    }
    loss_task_keys = all_task_ids - train_task_keys
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    attempt_audit_rows: list[dict[str, Any]] = []
    exclusion_reasons = {"clean": Counter(), "error": Counter()}
    task_yield: dict[str, Any] = {}
    for task_key in sorted(all_task_ids):
        domain, task_id = task_key.split(":", 1)
        ranked: dict[str, list[dict[str, Any]]] = {}
        for condition in ("clean", "error"):
            eligible, rows, reasons = _rank_and_filter(
                domain=domain,
                task_id=task_id,
                condition=condition,
                attempts=pools[condition][task_key],
                seed=seed,
            )
            eligible = _filter_training_compatible_attempts(
                domain=domain,
                task_id=task_id,
                condition=condition,
                eligible=eligible,
                audit_rows=rows,
                reasons=reasons,
                fault=faults[task_key]["error_condition"],
                context=domain_contexts[domain],
                tokenizer=tokenizer,
                seed=seed,
            )
            ranked[condition] = eligible
            attempt_audit_rows.extend(rows)
            exclusion_reasons[condition].update(reasons)
        selected_count = min(
            len(ranked["clean"]),
            len(ranked["error"]),
            MAX_PAIRS_PER_TASK,
        )
        partition_name = (
            "arm_train" if task_key in train_task_keys else "loss_validation"
        )
        task_pairs: list[dict[str, Any]] = []
        fault = faults[task_key]["error_condition"]
        for pair_rank, (clean, error) in enumerate(
            zip(
                ranked["clean"][:selected_count],
                ranked["error"][:selected_count],
            )
        ):
            _check_injected_fault(
                error["analysis"],
                fault,
                where=f"{task_key}:pair-{pair_rank}",
            )
            task_pairs.append(
                _materialize_pair(
                    domain=domain,
                    task_id=task_id,
                    pair_rank=pair_rank,
                    clean=clean,
                    error=error,
                    fault=fault,
                    context=domain_contexts[domain],
                    tokenizer=tokenizer,
                    seed=seed,
                    partition=partition_name,
                )
            )
        selected_by_task[task_key] = task_pairs
        task_yield[task_key] = {
            "partition": partition_name,
            "clean_eligible_attempts": len(ranked["clean"]),
            "error_eligible_attempts": len(ranked["error"]),
            "selected_pairs": len(task_pairs),
            "selected_pair_ids": [pair["pair_id"] for pair in task_pairs],
        }
    train_pairs = {
        task: rows
        for task, rows in selected_by_task.items()
        if task in train_task_keys and rows
    }
    loss_pairs = {
        task: rows
        for task, rows in selected_by_task.items()
        if task in loss_task_keys and rows
    }
    train_pair_count = sum(len(rows) for rows in train_pairs.values())
    loss_pair_count = sum(len(rows) for rows in loss_pairs.values())
    gate_failures = []
    if len(train_pairs) < MIN_TRAIN_TASKS:
        gate_failures.append(
            f"eligible arm-train tasks {len(train_pairs)} < {MIN_TRAIN_TASKS}"
        )
    if train_pair_count < MIN_TRAIN_PAIRS:
        gate_failures.append(
            f"eligible arm-train pairs {train_pair_count} < {MIN_TRAIN_PAIRS}"
        )
    if loss_pair_count == 0:
        gate_failures.append("loss-validation partition has no eligible pair")
    cross_seed_pairs = sum(
        pair["clean_attempt"]["attempt_seed"]
        != pair["error_attempt"]["attempt_seed"]
        for rows in selected_by_task.values()
        for pair in rows
    )
    all_selected_pair_count = sum(len(rows) for rows in selected_by_task.values())
    base_audit = {
        "protocol": "v5_stage1_sft_causal",
        "design_protocol": DESIGN_PROTOCOL,
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "model_tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "split_manifest_sha256": split_sha,
        "inner_train_partition_sha256": partition["canonical_sha256"],
        "arm_train_task_ids": sorted(train_task_keys),
        "loss_validation_task_ids": sorted(loss_task_keys),
        "ground_truth_incompatible_task_ids": sorted(
            GT_INCOMPATIBLE_TASK_IDS
        ),
        "official_test_used": False,
        "derived_validation_used_for_supervision": False,
        "generation_manifest_sha256": v5.sha256_file(generation_manifest),
        "raw_files": {str(path): v5.sha256_file(path) for path in raw_paths},
        "generation_contracts": generation_contract_audit,
        "dynamic_audits": dynamic_identities,
        "attempts_per_task_per_condition": ATTEMPTS_PER_TASK,
        "maximum_pairs_per_task": MAX_PAIRS_PER_TASK,
        "maximum_training_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "runtime_truncation": False,
        "candidate_order_inputs": [
            "protocol_seed",
            "domain",
            "task_id",
            "condition",
            "attempt_index",
            "attempt_seed",
        ],
        "candidate_order_uses_reward_loss_or_validation": False,
        "pairing": "independently_rank_eligible_conditions_then_zip_by_rank",
        "same_seed_required": False,
        "task_yield": task_yield,
        "exclusions": {
            condition: dict(sorted(counter.items()))
            for condition, counter in exclusion_reasons.items()
        },
        "formal_data_gate": {
            "required_distinct_train_tasks": MIN_TRAIN_TASKS,
            "required_train_pairs": MIN_TRAIN_PAIRS,
            "observed_distinct_train_tasks": len(train_pairs),
            "observed_train_pairs": train_pair_count,
            "observed_loss_validation_tasks": len(loss_pairs),
            "observed_loss_validation_pairs": loss_pair_count,
        },
        "cross_seed_pair_fraction": (
            cross_seed_pairs / all_selected_pair_count
            if all_selected_pair_count
            else 0.0
        ),
    }
    if gate_failures:
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "gate_failures": gate_failures,
            "arm_files_written": False,
        }
        _write_audit_bundle(
            output_dir=output_dir,
            partition=partition,
            attempt_rows=attempt_audit_rows,
            audit=audit,
        )
        raise PoolGateError("V5.3 data gate failed: " + "; ".join(gate_failures))

    eligible_train_tasks = sorted(train_pairs)
    scheduled_tasks = v5.task_schedule(
        eligible_train_tasks,
        schedule_rows,
        seed,
    )
    perfect_by_task = {
        task: [pair["perfect_success"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    raw_by_task = {
        task: [pair["failure_raw"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    repair_by_task = {
        task: [pair["repair_masked"] for pair in pairs]
        for task, pairs in train_pairs.items()
    }
    repair_50_candidates = {
        task: perfect_by_task[task] + repair_by_task[task]
        for task in eligible_train_tasks
    }
    endpoint_sources, matching_certificate = (
        _deterministic_tolerance_aware_joint_match(
            scheduled_tasks,
            {
                "perfect_success": perfect_by_task,
                "failure_raw": raw_by_task,
                "repair_50": repair_50_candidates,
                "repair_100": repair_by_task,
            },
            recovery_ratios=v5.ARM_RATIOS,
            seed=seed,
            supervised_tolerance=0.01,
            sequence_tolerance=0.02,
            recovery_ratio_tolerance=token_ratio_tolerance,
        )
    )
    if endpoint_sources is None:
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "gate_failures": ["MATCHING_SEARCH_INCONCLUSIVE"],
            "matching": matching_certificate,
            "arm_files_written": False,
        }
        _write_audit_bundle(
            output_dir=output_dir,
            partition=partition,
            attempt_rows=attempt_audit_rows,
            audit=audit,
        )
        raise PoolGateError(
            "V5.3 matching gate failed: MATCHING_SEARCH_INCONCLUSIVE"
        )
    target_tokens = matching_certificate["selected_target_supervised_tokens"]
    target_sequence = matching_certificate[
        "selected_target_nonpadding_tokens"
    ]
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_audit: dict[str, Any] = {}
    task_multisets: dict[str, Counter[Any]] = {}
    for arm in v5.CORE_ARMS:
        rows = [
            v5.source_clone(
                source,
                arm=arm,
                slot=position,
                fit_split="train_schedule",
            )
            for position, source in enumerate(endpoint_sources[arm])
        ]
        arm_rows[arm] = rows
        total_tokens = sum(
            row["token_contract"]["supervised_tokens"] for row in rows
        )
        sequence_tokens = sum(
            row["token_contract"]["sequence_tokens"] for row in rows
        )
        recovery_tokens = sum(
            row["token_contract"]["supervised_tokens"]
            for row in rows
            if row["metadata"]["source"] == "failure_rich"
        )
        ratio = recovery_tokens / max(1, total_tokens)
        if abs(ratio - v5.ARM_RATIOS[arm]) > token_ratio_tolerance:
            raise RuntimeError(
                f"{arm}: recovery supervised-token ratio {ratio:.6f} "
                f"differs from {v5.ARM_RATIOS[arm]:.2f}"
            )
        failed_labels = sum(
            sum(
                row["label_mask"][index]
                for index in row["metadata"]["failed_assistant_message_indices"]
            )
            for row in rows
        )
        if arm == "failure_raw" and failed_labels != len(rows):
            raise RuntimeError("failure_raw does not label one failed action per row")
        if arm != "failure_raw" and failed_labels:
            raise RuntimeError(f"{arm}: failed action became a positive label")
        task_multisets[arm] = Counter(
            (row["metadata"]["domain"], row["metadata"]["task_id"])
            for row in rows
        )
        arm_audit[arm] = {
            "rows": len(rows),
            "supervised_tokens": total_tokens,
            "nonpadding_tokens": sequence_tokens,
            "recovery_supervised_tokens": recovery_tokens,
            "recovery_supervised_token_ratio": ratio,
            "controlled_failed_action_labels": failed_labels,
        }
    reference_tasks = task_multisets[v5.CORE_ARMS[0]]
    if any(task_multisets[arm] != reference_tasks for arm in v5.CORE_ARMS[1:]):
        raise RuntimeError("V5.3 core-arm task multisets differ")
    core_tokens = [arm_audit[arm]["supervised_tokens"] for arm in v5.CORE_ARMS]
    core_sequences = [arm_audit[arm]["nonpadding_tokens"] for arm in v5.CORE_ARMS]
    token_relative_range = (
        (max(core_tokens) - min(core_tokens)) / min(core_tokens)
    )
    sequence_relative_range = (
        (max(core_sequences) - min(core_sequences)) / min(core_sequences)
    )
    if token_relative_range > 0.01 or sequence_relative_range > 0.02:
        matching_certificate = {
            **matching_certificate,
            "search_status": "MATCHING_SEARCH_INCONCLUSIVE",
            "post_selection_verification": "FAILED",
            "post_selection_supervised_token_relative_range": (
                token_relative_range
            ),
            "post_selection_nonpadding_token_relative_range": (
                sequence_relative_range
            ),
            "search_space_exhausted": False,
            "mathematical_infeasibility_claimed": False,
        }
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "gate_failures": ["MATCHING_SEARCH_INCONCLUSIVE"],
            "matching": matching_certificate,
            "arms": arm_audit,
            "arm_files_written": False,
        }
        _write_audit_bundle(
            output_dir=output_dir,
            partition=partition,
            attempt_rows=attempt_audit_rows,
            audit=audit,
        )
        raise PoolGateError(
            "V5.3 matching gate failed: MATCHING_SEARCH_INCONCLUSIVE"
        )
    validation_loss = []
    for task_id in sorted(loss_pairs):
        for pair in loss_pairs[task_id]:
            for source in (pair["perfect_success"], pair["repair_masked"]):
                validation_loss.append(
                    v5.source_clone(
                        source,
                        arm="validation_loss",
                        slot=len(validation_loss),
                        fit_split="validation_loss",
                    )
                )
    train_source_ids = {
        row["metadata"]["source_example_id"]
        for rows in arm_rows.values()
        for row in rows
    }
    validation_source_ids = {
        row["metadata"]["source_example_id"] for row in validation_loss
    }
    if train_source_ids & validation_source_ids:
        raise RuntimeError("V5.3 train/loss-validation source overlap")
    output_dir.mkdir(parents=True, exist_ok=False)
    v5.write_json(output_dir / "task_partition.json", partition)
    v5.write_jsonl(output_dir / "attempt_audit.jsonl", attempt_audit_rows)
    master_rows = [
        pair
        for task_id in sorted(selected_by_task)
        for pair in selected_by_task[task_id]
    ]
    v5.write_jsonl(output_dir / "paired_master_pool.jsonl", master_rows)
    for arm, rows in arm_rows.items():
        path = output_dir / "arms" / arm / "train.jsonl"
        v5.write_jsonl(path, rows)
        arm_audit[arm]["sha256"] = v5.sha256_file(path)
    v5.write_jsonl(output_dir / "validation_loss.jsonl", validation_loss)
    if validation_payload is not None:
        v5.write_json(output_dir / "validation_manifest.json", validation_payload)
    audit = {
        **base_audit,
        "status": "PASS",
        "decision": "TRAINING_AUTHORIZED",
        "claim_scope": "v5_3_task_level_cross_attempt_sft_screen",
        "semantic_repair_claim_allowed": False,
        "failure_raw_mechanism_causal_claim_allowed": False,
        "failure_raw_interpretation": (
            "descriptive_negative_exposure_control_only"
        ),
        "eligible_distinct_tasks": len(train_pairs),
        "paired_master_slots": train_pair_count,
        "train_schedule_rows_per_arm": schedule_rows,
        "shared_target_supervised_tokens": target_tokens,
        "shared_target_nonpadding_tokens": target_sequence,
        "core_task_id_multiset_equal": True,
        "common_eligible_pair_support": True,
        "core_pair_id_occurrence_multiset_equal": False,
        "core_pair_id_occurrence_multiset_equal_required": False,
        "train_validation_source_overlap": 0,
        "cross_arm_supervised_token_relative_range": token_relative_range,
        "cross_arm_nonpadding_token_relative_range": sequence_relative_range,
        "matching": matching_certificate,
        "label_guarantees": {
            "per_tool_call_adjacent_outcome_verified": True,
            "final_failure_never_positive_sft": True,
            "raw_labels_only_controlled_failed_action": True,
            "repair_failure_is_context_not_label": True,
            "repair_has_verified_post_error_success": True,
            "causal_prefix_tokenization_verified": True,
            "call_ids_condition_neutral": True,
            "official_test_and_derived_validation_label_leakage": 0,
        },
        "arms": arm_audit,
        "validation_loss": {
            "rows": len(validation_loss),
            "distinct_tasks": len(loss_pairs),
            "source_split": "inner_train",
            "sha256": v5.sha256_file(output_dir / "validation_loss.jsonl"),
        },
    }
    v5.write_json(output_dir / "audit.json", audit)
    hash_paths = [
        output_dir / "task_partition.json",
        output_dir / "attempt_audit.jsonl",
        output_dir / "paired_master_pool.jsonl",
        output_dir / "validation_loss.jsonl",
        output_dir / "audit.json",
        *[
            output_dir / "arms" / arm / "train.jsonl"
            for arm in v5.CORE_ARMS
        ],
    ]
    if (output_dir / "validation_manifest.json").is_file():
        hash_paths.append(output_dir / "validation_manifest.json")
    v5.write_json(
        output_dir / "hashes.json",
        {
            str(path.relative_to(output_dir)): v5.sha256_file(path)
            for path in hash_paths
        },
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--generation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--validation-dynamic-audit", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=v5.MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-generation-source-commit")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokenizer_revision != v5.MODEL_REVISION:
        raise RuntimeError(
            "V5.3 tokenizer revision drift: "
            f"{args.tokenizer_revision} != {v5.MODEL_REVISION}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    contexts = v5.load_tau2_contexts(args.tau2_root)
    audit = prepare(
        split_manifest=args.split_manifest.resolve(),
        generation_manifest=args.generation_manifest.resolve(),
        validation_manifest=args.validation_manifest.resolve(),
        generation_dynamic_audit=args.generation_dynamic_audit.resolve(),
        validation_dynamic_audit=args.validation_dynamic_audit.resolve(),
        expected_source_commit=args.expected_source_commit,
        expected_generation_source_commit=args.expected_generation_source_commit,
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        domain_contexts=contexts,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
