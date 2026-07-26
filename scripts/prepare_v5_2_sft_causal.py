#!/usr/bin/env python3
"""Build the V5.2 task-level, cross-attempt SFT screen.

V5.2 deliberately leaves the historical V5 builder unchanged.  It reuses
V5's trajectory eligibility, causal label masking, tokenization, and schedule
helpers, but changes two data-design decisions:

* the 83 inner-train task IDs are split *before generation outcomes are read*
  into 75 arm-train and 8 loss-validation tasks; and
* each condition gets twelve attempts.  Eligible clean and post-fault
  attempts are independently hash ordered within a task, then zipped by rank.
  Thus clean/error attempts need not share a seed.

No official-test task may enter this module.  A failed 40-task/48-pair gate
writes an audit-only bundle and raises before any arm JSONL is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    import prepare_v5_sft_causal as v5
except ModuleNotFoundError:
    from scripts import prepare_v5_sft_causal as v5

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


DESIGN_PROTOCOL = "v5_2_task_level_cross_seed_sft_screen"
DESIGN_VERSION = "5.2"
PARTITION_PROTOCOL = "v5_2_inner_train_loss_validation_split"
PARTITION_SALT = "v5.2-loss-validation"
PAIR_ORDER_SALT = "v5.2-eligible-attempt-order"
ATTEMPTS_PER_TASK = 12
MAX_PAIRS_PER_TASK = 2
MIN_TRAIN_TASKS = 40
MIN_TRAIN_PAIRS = 48
MAX_SEQUENCE_TOKENS = 8192
EXPECTED_GENERATION_SHARDS = 3
EXPECTED_TEACHER_API_BASE_BY_SHARD = {
    0: "http://127.0.0.1:8011/v1",
    1: "http://127.0.0.1:8012/v1",
    2: "http://127.0.0.1:8013/v1",
}
LOSS_VALIDATION_COUNTS = {"retail": 6, "airline": 2}
EXPECTED_PARTITION_SHA256 = (
    "b476ec66996485445dd0b65f9fc982347526ff70fbec5769bbe9805aa70ce50a"
)
EXPECTED_LOSS_VALIDATION_IDS = {
    "retail": ["21", "28", "43", "99", "110", "112"],
    "airline": ["14", "27"],
}


class PoolGateError(RuntimeError):
    """Raised after writing an audit-only FAIL_CLOSED bundle."""


def _numeric_task_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError as error:
        raise RuntimeError(f"V5.2 task ID is not numeric: {task_id!r}") from error


def _compact_sha256(value: Any) -> str:
    return hashlib.sha256(v5.canonical(value).encode("utf-8")).hexdigest()


def build_task_partition(
    split: dict[str, Any],
    *,
    split_manifest_sha256: str,
    seed: int = v5.SEED,
    strict_formal: bool = True,
) -> dict[str, Any]:
    """Create the frozen 75/8 partition without observing rollout outcomes."""

    domains: dict[str, Any] = {}
    all_train: set[str] = set()
    all_loss: set[str] = set()
    for domain in ("retail", "airline"):
        row = split["domains"][domain]
        inner = [str(value) for value in row["inner_train_ids"]]
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
            raise RuntimeError(f"{domain}: V5.2 partition is not a disjoint cover")
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
        if len(all_train) != 75 or len(all_loss) != 8:
            raise RuntimeError(
                "V5.2 partition count drift: "
                f"train={len(all_train)}, loss_validation={len(all_loss)}"
            )
        observed_loss = {
            domain: domains[domain]["loss_validation_ids"]
            for domain in ("retail", "airline")
        }
        if observed_loss != EXPECTED_LOSS_VALIDATION_IDS:
            raise RuntimeError(
                f"V5.2 frozen loss-validation IDs drift: {observed_loss}"
            )
        if digest != EXPECTED_PARTITION_SHA256:
            raise RuntimeError(
                "V5.2 frozen partition digest drift: "
                f"observed={digest}, expected={EXPECTED_PARTITION_SHA256}"
            )
    return {
        **payload,
        "canonical_sha256": digest,
        "selection_inputs": ["domain", "task_id", "protocol_seed"],
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
            str(value) for value in split["domains"][domain]["inner_train_ids"]
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
        raise RuntimeError("V5.2 clean/error task coverage differs")
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
    pair_id = f"{domain}:{task_id}:v5.2:{pair_rank}:{pair_hash[:16]}"
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
        "protocol": "v5_2_independent_eligible_rank_zip",
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
                "v5_2_partition": partition,
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
        raise RuntimeError("V5.2 requires a pinned tokenizer revision")
    split = v5.load_split(split_manifest, strict_counts=strict_split_counts)
    split_sha = v5.sha256_file(split_manifest)
    partition = build_task_partition(
        split,
        split_manifest_sha256=split_sha,
        seed=seed,
        strict_formal=strict_split_counts,
    )
    all_task_ids = {
        f"{domain}:{task_id}"
        for domain in ("retail", "airline")
        for task_id in split["domains"][domain]["inner_train_ids"]
    }
    generation_payload = json.loads(
        generation_manifest.read_text(encoding="utf-8")
    )
    faults = v5.validate_multifault_manifest(
        generation_payload,
        expected_protocol="v5_stage1_multifault_data_construction",
        expected_source_split="derived_inner_train",
        expected_task_ids=all_task_ids,
        split_manifest_sha256=split_sha,
    )
    dynamic_identities: dict[str, Any] = {}
    if strict_dynamic_audits:
        if (
            generation_dynamic_audit is None
            or validation_dynamic_audit is None
            or validation_manifest is None
        ):
            raise RuntimeError("formal V5.2 preparation requires both dynamic audits")
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
            raise RuntimeError("formal V5.2 preparation needs expected source commit")
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
        )
    pools, raw_paths = _load_attempts(raw_dir, split)
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
        raise PoolGateError("V5.2 data gate failed: " + "; ".join(gate_failures))

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
    token_bounds = [
        v5.endpoint_bounds(scheduled_tasks, pool, "supervised_tokens")
        for pool in (perfect_by_task, raw_by_task, repair_by_task)
    ]
    sequence_bounds = [
        v5.endpoint_bounds(scheduled_tasks, pool, "sequence_tokens")
        for pool in (perfect_by_task, raw_by_task, repair_by_task)
    ]
    token_low = max(row[0] for row in token_bounds)
    token_high = min(row[1] for row in token_bounds)
    sequence_low = max(row[0] for row in sequence_bounds)
    sequence_high = min(row[1] for row in sequence_bounds)
    matching_failures = []
    if token_low > token_high:
        matching_failures.append(
            "common task schedule has no shared supervised-token target"
        )
    if sequence_low > sequence_high:
        matching_failures.append(
            "common task schedule has no shared nonpadding-token target"
        )
    if matching_failures:
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "gate_failures": matching_failures,
            "matching": {
                "supervised_token_endpoint_bounds": token_bounds,
                "nonpadding_token_endpoint_bounds": sequence_bounds,
                "common_pair_support": True,
                "pair_occurrence_multiset_required": False,
            },
            "arm_files_written": False,
        }
        _write_audit_bundle(
            output_dir=output_dir,
            partition=partition,
            attempt_rows=attempt_audit_rows,
            audit=audit,
        )
        raise PoolGateError(
            "V5.2 matching gate failed: " + "; ".join(matching_failures)
        )
    target_tokens = (token_low + token_high) // 2
    target_sequence = (sequence_low + sequence_high) // 2
    endpoint_sources = {
        "perfect_success": v5.choose_candidates(
            scheduled_tasks,
            perfect_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="v5.2:perfect_success",
        ),
        "failure_raw": v5.choose_candidates(
            scheduled_tasks,
            raw_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="v5.2:failure_raw",
        ),
        "repair_100": v5.choose_candidates(
            scheduled_tasks,
            repair_by_task,
            target_tokens=target_tokens,
            target_sequence=target_sequence,
            seed=seed,
            salt="v5.2:repair_100",
        ),
    }
    endpoint_sources["repair_50"] = v5.mixture_schedule(
        scheduled_tasks,
        perfect_by_task,
        repair_by_task,
        ratio=0.5,
        target_tokens=target_tokens,
        target_sequence=target_sequence,
        seed=seed,
        salt="v5.2:repair_50",
    )
    endpoint_sources["perfect_success"] = _refine_schedule(
        scheduled_tasks,
        endpoint_sources["perfect_success"],
        perfect_by_task,
        target_tokens=target_tokens,
        target_sequence=target_sequence,
        target_recovery_ratio=0.0,
        seed=seed,
        salt="v5.2:perfect_success",
    )
    endpoint_sources["failure_raw"] = _refine_schedule(
        scheduled_tasks,
        endpoint_sources["failure_raw"],
        raw_by_task,
        target_tokens=target_tokens,
        target_sequence=target_sequence,
        target_recovery_ratio=1.0,
        seed=seed,
        salt="v5.2:failure_raw",
    )
    endpoint_sources["repair_100"] = _refine_schedule(
        scheduled_tasks,
        endpoint_sources["repair_100"],
        repair_by_task,
        target_tokens=target_tokens,
        target_sequence=target_sequence,
        target_recovery_ratio=1.0,
        seed=seed,
        salt="v5.2:repair_100",
    )
    repair_50_candidates = {
        task: perfect_by_task[task] + repair_by_task[task]
        for task in eligible_train_tasks
    }
    endpoint_sources["repair_50"] = _refine_schedule(
        scheduled_tasks,
        endpoint_sources["repair_50"],
        repair_50_candidates,
        target_tokens=target_tokens,
        target_sequence=target_sequence,
        target_recovery_ratio=0.5,
        seed=seed,
        salt="v5.2:repair_50",
    )
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
        raise RuntimeError("V5.2 core-arm task multisets differ")
    core_tokens = [arm_audit[arm]["supervised_tokens"] for arm in v5.CORE_ARMS]
    core_sequences = [arm_audit[arm]["nonpadding_tokens"] for arm in v5.CORE_ARMS]
    token_relative_range = (
        (max(core_tokens) - min(core_tokens)) / min(core_tokens)
    )
    sequence_relative_range = (
        (max(core_sequences) - min(core_sequences)) / min(core_sequences)
    )
    if token_relative_range > 0.01 or sequence_relative_range > 0.02:
        matching_failures = []
        if token_relative_range > 0.01:
            matching_failures.append(
                f"supervised-token relative range {token_relative_range:.6f} > 0.01"
            )
        if sequence_relative_range > 0.02:
            matching_failures.append(
                f"nonpadding-token relative range {sequence_relative_range:.6f} > 0.02"
            )
        audit = {
            **base_audit,
            "status": "FAIL_CLOSED",
            "decision": "DO_NOT_TRAIN",
            "gate_failures": matching_failures,
            "matching": {
                "target_supervised_tokens": target_tokens,
                "target_nonpadding_tokens": target_sequence,
                "supervised_token_relative_range": token_relative_range,
                "nonpadding_token_relative_range": sequence_relative_range,
                "common_pair_support": True,
                "pair_occurrence_multiset_required": False,
                "task_occurrence_multiset_equal": True,
            },
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
            "V5.2 matching gate failed: " + "; ".join(matching_failures)
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
        raise RuntimeError("V5.2 train/loss-validation source overlap")
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
        "claim_scope": "v5_2_task_level_cross_attempt_sft_screen",
        "semantic_repair_claim_allowed": False,
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
            "V5.2 tokenizer revision drift: "
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
