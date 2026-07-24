#!/usr/bin/env python3
"""Fail-closed paired aggregation for the frozen QLoRA V4 experiment.

The aggregator compares four checkpoints on the same 959 frozen examples:

* ``standard_v3``: the completed Standard V3 SFT control;
* ``clean_sft``: failed gold actions are not positive SFT labels;
* ``continued_sft``: the chosen-only continuation control;
* ``dpo``: preference training from the same Clean-SFT checkpoint.

Predictions, rather than stored summary metrics, are the source of truth.  Every
generated string is reparsed and rescored against the prepare-time outcome
audit.  The script refuses partial evaluations, foreign/duplicate IDs, target
drift, a changed frozen test hash, or ``limited=true`` results.

The three planned contrasts separate supervision cleaning, additional chosen
exposure, and the DPO objective. Confidence intervals use 10,000 paired
task-cluster bootstrap draws. Cluster sign-flip p-values are reported as a
sensitivity analysis; Holm correction is restricted to the two preregistered
primary recovery-success full-call comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluate_tool_actions_v3 import (
    FROZEN_FORMAL_EXAMPLES,
    FROZEN_MAX_NEW_TOKENS,
    FROZEN_MAX_PROMPT_TOKENS,
    FROZEN_MODEL,
    FROZEN_MODEL_REVISION,
    FROZEN_SEED,
    METRIC_KEYS,
    normalize_call,
    parse_call,
)

PROTOCOL = "qlora_v4_paired_aggregation"
EXPECTED_TEST_SHA256 = (
    "0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96"
)
EXPECTED_TEST_OUTCOMES_SHA256 = (
    "9a4ec2b1e25ee512e5946e8ac770b0fbf6b0ed0d5b61f994c643fc04cd227b57"
)
EXPECTED_TEST_PREFERENCE_PAIRS_SHA256 = (
    "b85548e9f1c041032358172e10b7f7f53f91710d1d15f26dfaa606a07799cf74"
)
EXPECTED_TEST_PREFERENCE_PAIRS = 48
BOOTSTRAP_SAMPLES = 10_000
SIGN_FLIP_SAMPLES = 10_000
FAILED_REPLAY_METRIC = "exact_failed_call_replay_rate_on_strict_pairs"
FROZEN_TAG = "v4-frozen-20260724"
CLEAN_TRAIN_SCHEDULE_SHA256 = (
    "4b28da48082ef5bd3396e7df4b5b723c4efffe4b2e5438f47c8c2ca9d709f386"
)
PREFERENCE_TRAIN_SCHEDULE_SHA256 = (
    "f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341"
)
STANDARD_V3_PROTOCOL = "qlora_v3"
V4_EVALUATION_PROTOCOL = "qlora_v4"
STANDARD_V3_CHECKPOINT_FINGERPRINT = (
    "3cdfa858353e8f7ea6da0d5558c21014bacbd58b2092f7095d6b5925f147825c"
)
STANDARD_V3_METRICS_SHA256 = (
    "880db45dcb6dc6eea497aa32dff26c5d59a4ab3b570c458c6f132757ea9d61f4"
)
STANDARD_V3_CONTRACT_SHA256 = (
    "2b941573f85b9c1d33622c4e6fde42d10af194981fe028baa7e210d39a455471"
)
STANDARD_V3_PREDICTIONS_SHA256 = (
    "491e1613b20eb11b176d9aac61e19b4e3472257d3a2576125c76c6e03cb24de3"
)

ARM_ORDER = ("standard_v3", "clean_sft", "continued_sft", "dpo")
CONTRASTS = {
    "clean_sft_minus_standard_v3": ("clean_sft", "standard_v3"),
    "continued_sft_minus_clean_sft": ("continued_sft", "clean_sft"),
    "dpo_minus_continued_sft": ("dpo", "continued_sft"),
}
PRIMARY_CONTRASTS = (
    "clean_sft_minus_standard_v3",
    "dpo_minus_continued_sft",
)
GROUPS = (
    "overall",
    "outcome_success",
    "failed_gold",
    "non_recovery_success",
    "recovery_success",
    "agent_initiated",
    "user_assisted",
)
EXPECTED_GROUP_COUNTS = {
    "overall": 959,
    "outcome_success": 902,
    "failed_gold": 57,
    "non_recovery_success": 852,
    "recovery_success": 50,
}
GROUP_DEFINITIONS = {
    "overall": "All 959 frozen test examples.",
    "outcome_success": "Gold tool call has a verified successful outcome.",
    "failed_gold": "Gold tool call itself has a verified failed outcome.",
    "non_recovery_success": (
        "Verified-success examples with recovery_mode=none."
    ),
    "recovery_success": (
        "Verified-success examples with agent_initiated or user_assisted "
        "recovery_mode."
    ),
    "agent_initiated": (
        "Verified-success recovery examples with no intervening user message."
    ),
    "user_assisted": (
        "Verified-success recovery examples with an intervening user message."
    ),
}
CONTRACT_KEYS = (
    "test_file_sha256",
    "formal_test_examples",
    "evaluated_examples",
    "model",
    "model_revision",
    "base_model_loading",
    "max_prompt_tokens",
    "generation",
    "limited",
)
CORE_EVAL_EXPECTED = {
    "test_file_sha256": EXPECTED_TEST_SHA256,
    "formal_test_examples": FROZEN_FORMAL_EXAMPLES,
    "evaluated_examples": FROZEN_FORMAL_EXAMPLES,
    "model": FROZEN_MODEL,
    "model_revision": FROZEN_MODEL_REVISION,
    "base_model_loading": "nf4_4bit",
    "max_prompt_tokens": FROZEN_MAX_PROMPT_TOKENS,
    "generation": {
        "do_sample": False,
        "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
        "batch_size": 1,
    },
    "limited": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing required JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"blank line in formal JSONL {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def require_exact(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label}: {observed!r} != {expected!r}")


def require_close(label: str, observed: Any, expected: Any) -> None:
    if observed is None or expected is None:
        require_exact(label, observed, expected)
        return
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"{label}: {observed!r} != recomputed {expected!r}")


def normalize_required_call(value: Any, label: str) -> dict[str, Any]:
    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = parse_call(candidate)
    normalized = normalize_call(candidate)
    if normalized is None:
        raise RuntimeError(f"{label} is not a valid tool call")
    return normalized


def first_present(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, Any] | None:
    for key in keys:
        if key in row:
            return key, row[key]
    return None


def normalize_outcome_class(row: dict[str, Any], example_id: str) -> str:
    signals: list[tuple[str, str]] = []
    string_fields = (
        "outcome_class",
        "label_class",
        "gold_outcome",
        "target_outcome",
        "outcome",
        "linked_tool_outcome",
    )
    success_values = {
        "outcome_success",
        "success",
        "succeeded",
        "successful",
        "verified_success",
        "ok",
    }
    failure_values = {
        "failed_gold",
        "failure",
        "failed",
        "error",
        "verified_failure",
        "not_found",
        "generic_error",
    }
    for key in string_fields:
        value = row.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in success_values:
            signals.append((key, "outcome_success"))
        elif normalized in failure_values:
            signals.append((key, "failed_gold"))

    boolean_fields = {
        "target_succeeded": ("outcome_success", "failed_gold"),
        "gold_succeeded": ("outcome_success", "failed_gold"),
        "tool_success": ("outcome_success", "failed_gold"),
        "is_failed_gold": ("failed_gold", "outcome_success"),
        "gold_is_failure": ("failed_gold", "outcome_success"),
        "target_failed": ("failed_gold", "outcome_success"),
        "is_outcome_success_target": ("outcome_success", "failed_gold"),
    }
    for key, (when_true, when_false) in boolean_fields.items():
        value = row.get(key)
        if isinstance(value, bool):
            signals.append((key, when_true if value else when_false))
    if not signals:
        raise RuntimeError(
            f"test outcome lacks an unambiguous success/failure label: {example_id}"
        )
    classes = {value for _, value in signals}
    if len(classes) != 1:
        raise RuntimeError(
            f"conflicting outcome labels for {example_id}: {signals!r}"
        )
    return signals[0][1]


def normalize_recovery_mode(row: dict[str, Any], example_id: str) -> str:
    value = row.get("recovery_mode")
    if value not in {"none", "agent_initiated", "user_assisted"}:
        raise RuntimeError(
            f"invalid or missing recovery_mode for {example_id}: {value!r}"
        )
    return value


def normalize_outcomes(
    path: Path,
    expected_examples: int = FROZEN_FORMAL_EXAMPLES,
    expected_group_counts: dict[str, int] | None = EXPECTED_GROUP_COUNTS,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise RuntimeError("test outcome lacks a nonempty string example_id")
        if example_id in seen:
            raise RuntimeError(f"duplicate test outcome example_id: {example_id}")
        seen.add(example_id)
        task_key = row.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise RuntimeError(f"test outcome lacks task_key: {example_id}")
        target_tool = row.get("target_tool")
        if not isinstance(target_tool, str) or not target_tool:
            raise RuntimeError(f"test outcome lacks target_tool: {example_id}")
        recovery_mode = normalize_recovery_mode(row, example_id)
        outcome_class = normalize_outcome_class(row, example_id)
        expected_recovery_success = (
            outcome_class == "outcome_success" and recovery_mode != "none"
        )
        if "is_recovery_success_target" in row:
            require_exact(
                f"is_recovery_success_target for {example_id}",
                row.get("is_recovery_success_target"),
                expected_recovery_success,
            )
        expected_non_recovery_success = (
            outcome_class == "outcome_success" and recovery_mode == "none"
        )
        if "is_non_recovery_success_target" in row:
            require_exact(
                f"is_non_recovery_success_target for {example_id}",
                row.get("is_non_recovery_success_target"),
                expected_non_recovery_success,
            )
        normalized.append(
            {
                "example_id": example_id,
                "task_key": task_key,
                "trace_id": row.get("trace_id"),
                "source": row.get("source"),
                "recovery_mode": recovery_mode,
                "prior_error_type": row.get("prior_error_type"),
                "outcome_class": outcome_class,
                "target_tool": target_tool,
            }
        )
    require_exact("test outcome rows", len(normalized), expected_examples)
    if expected_group_counts is not None:
        counts = {
            name: len(select_group(normalized, name))
            for name in expected_group_counts
        }
        for name, expected in expected_group_counts.items():
            require_exact(f"frozen {name} examples", counts[name], expected)
        require_exact(
            "agent/user partition of recovery-success",
            len(select_group(normalized, "agent_initiated"))
            + len(select_group(normalized, "user_assisted")),
            counts["recovery_success"],
        )
    return normalized


def normalize_preference_pairs(
    path: Path,
    outcomes: list[dict[str, Any]],
    expected_pairs: int = EXPECTED_TEST_PREFERENCE_PAIRS,
) -> dict[str, dict[str, Any]]:
    outcome_by_id = {row["example_id"]: row for row in outcomes}
    eligible_ids = {
        row["example_id"] for row in select_group(outcomes, "recovery_success")
    }
    pairs: dict[str, dict[str, Any]] = {}
    seen_pair_ids: set[str] = set()
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or example_id not in outcome_by_id:
            raise RuntimeError(
                f"test preference pair has foreign/missing example_id: {example_id!r}"
            )
        if example_id in pairs:
            raise RuntimeError(f"duplicate preference example_id: {example_id}")
        pair_id = row.get("pair_id", example_id)
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen_pair_ids:
            raise RuntimeError(f"invalid or duplicate preference pair_id: {pair_id!r}")
        seen_pair_ids.add(pair_id)
        prompt = row.get("prompt")
        chosen_text = row.get("chosen")
        rejected_text = row.get("rejected")
        if (
            not isinstance(prompt, str)
            or not isinstance(chosen_text, str)
            or not isinstance(rejected_text, str)
        ):
            raise RuntimeError(
                f"preference pair lacks canonical prompt/chosen/rejected text: "
                f"{example_id}"
            )
        chosen = normalize_required_call(
            chosen_text, f"chosen for {example_id}"
        )
        rejected = normalize_required_call(
            rejected_text, f"rejected for {example_id}"
        )
        if example_id not in eligible_ids:
            raise RuntimeError(
                f"preference pair is not a recovery-success example: {example_id}"
            )
        require_exact(
            f"preference chosen target tool for {example_id}",
            chosen["name"],
            outcome_by_id[example_id]["target_tool"],
        )
        if chosen == rejected:
            raise RuntimeError(f"chosen equals rejected for preference pair {pair_id}")
        if "task_key" in row:
            require_exact(
                f"preference task_key for {example_id}",
                row.get("task_key"),
                outcome_by_id[example_id]["task_key"],
            )
        pairs[example_id] = {
            "pair_id": pair_id,
            "example_id": example_id,
            "task_key": outcome_by_id[example_id]["task_key"],
            "recovery_mode": outcome_by_id[example_id]["recovery_mode"],
            "chosen": chosen,
            "rejected": rejected,
            "pair_content_sha256": hashlib.sha256(
                canonical_json(
                    {
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                    }
                ).encode("utf-8")
            ).hexdigest(),
        }
    require_exact("strict test preference pair rows", len(pairs), expected_pairs)
    if not set(pairs).issubset(eligible_ids):
        raise RuntimeError("test preference IDs are not a recovery-success subset")
    return pairs


def select_group(items: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "overall":
        return items
    if name == "outcome_success":
        return [
            item for item in items if item["outcome_class"] == "outcome_success"
        ]
    if name == "failed_gold":
        return [item for item in items if item["outcome_class"] == "failed_gold"]
    if name == "non_recovery_success":
        return [
            item
            for item in items
            if item["outcome_class"] == "outcome_success"
            and item["recovery_mode"] == "none"
        ]
    if name == "recovery_success":
        return [
            item
            for item in items
            if item["outcome_class"] == "outcome_success"
            and item["recovery_mode"] != "none"
        ]
    if name == "agent_initiated":
        return [
            item
            for item in items
            if item["outcome_class"] == "outcome_success"
            and item["recovery_mode"] == "agent_initiated"
        ]
    if name == "user_assisted":
        return [
            item
            for item in items
            if item["outcome_class"] == "outcome_success"
            and item["recovery_mode"] == "user_assisted"
        ]
    raise KeyError(name)


def validate_core_contract(
    arm: str,
    metrics: dict[str, Any],
    contract: dict[str, Any],
    predictions_path: Path,
    expected_test_sha256: str,
    expected_examples: int,
    reference_contract: dict[str, Any] | None,
) -> None:
    expected_protocol = (
        STANDARD_V3_PROTOCOL
        if arm == "standard_v3"
        else V4_EVALUATION_PROTOCOL
    )
    require_exact(
        f"{arm} metrics protocol",
        metrics.get("protocol"),
        expected_protocol,
    )
    require_exact(
        f"{arm} contract protocol",
        contract.get("protocol"),
        expected_protocol,
    )
    require_exact(
        f"{arm} evaluator family",
        contract.get("training_and_evaluation_protocol"),
        "qlora_v2_frozen",
    )
    expected = dict(CORE_EVAL_EXPECTED)
    expected["test_file_sha256"] = expected_test_sha256
    expected["formal_test_examples"] = expected_examples
    expected["evaluated_examples"] = expected_examples
    for key, value in expected.items():
        require_exact(f"{arm} metrics {key}", metrics.get(key), value)
        require_exact(f"{arm} contract {key}", contract.get(key), value)
    for key in CONTRACT_KEYS:
        require_exact(
            f"{arm} contract mirrored in metrics: {key}",
            metrics.get(key),
            contract.get(key),
        )
    fingerprint = contract.get("checkpoint_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError(f"{arm} contract lacks checkpoint_fingerprint")
    require_exact(
        f"{arm} metrics checkpoint_fingerprint",
        metrics.get("checkpoint_fingerprint"),
        fingerprint,
    )
    require_exact(
        f"{arm} prediction hash",
        metrics.get("predictions_sha256"),
        sha256_file(predictions_path),
    )
    if reference_contract is not None:
        for key in CONTRACT_KEYS:
            require_exact(
                f"{arm}/standard_v3 frozen evaluation field {key}",
                contract.get(key),
                reference_contract.get(key),
            )


def validate_new_arm_training_identity(
    arm: str,
    result_dir: Path,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    manifest_path = result_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    require_exact(f"{arm} source tag", manifest.get("source_tag"), FROZEN_TAG)
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit)
    ):
        raise RuntimeError(f"{arm} run manifest lacks a full source commit")
    require_exact(f"{arm} manifest arm", manifest.get("arm"), arm)
    output = manifest.get("output_checkpoint")
    if not isinstance(output, dict):
        raise RuntimeError(f"{arm} manifest lacks output_checkpoint")
    require_exact(
        f"{arm} training/evaluation checkpoint identity",
        output.get("checkpoint_fingerprint"),
        evaluation_fingerprint,
    )
    if arm == "clean_sft":
        require_exact(
            "clean_sft training protocol",
            manifest.get("protocol"),
            "qlora_v4_clean_sft",
        )
        require_exact("clean_sft smoke flag", manifest.get("smoke_test"), False)
        require_exact("clean_sft failed labels", manifest.get("failed_action_labels"), 0)
        require_exact("clean_sft steps", manifest.get("max_steps"), 68)
        require_exact(
            "clean_sft schedule hash",
            manifest.get("train_file_sha256"),
            CLEAN_TRAIN_SCHEDULE_SHA256,
        )
        require_exact(
            "clean_sft held-out access",
            manifest.get("held_out_test_accessed"),
            False,
        )
        require_exact(
            "clean_sft finite loss audit",
            (manifest.get("loss_audit") or {}).get("finite"),
            True,
        )
    else:
        require_exact(
            f"{arm} training protocol",
            manifest.get("protocol"),
            "qlora_v4_preference_continuation",
        )
        require_exact(f"{arm} formal flag", manifest.get("formal_result"), True)
        require_exact(
            f"{arm} pair schedule hash",
            (manifest.get("data") or {}).get("train_schedule_sha256"),
            PREFERENCE_TRAIN_SCHEDULE_SHA256,
        )
        compute = manifest.get("compute_contract") or {}
        require_exact(f"{arm} optimizer steps", compute.get("optimizer_steps"), 18)
        require_exact(
            f"{arm} scheduled microbatches",
            compute.get("scheduled_microbatches"),
            144,
        )
        require_exact(
            f"{arm} held-out access",
            (manifest.get("data") or {}).get("held_out_test_accessed"),
            False,
        )
        require_exact(
            f"{arm} finite loss audit",
            (manifest.get("loss_audit") or {}).get("finite"),
            True,
        )
    return manifest


def rescore_predictions(
    arm: str,
    predictions_path: Path,
    outcomes: list[dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    canonical_targets: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    expected = {row["example_id"]: row for row in outcomes}
    observed: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(predictions_path):
        example_id = item.get("example_id")
        if not isinstance(example_id, str):
            raise RuntimeError(f"{arm} prediction lacks string example_id")
        if example_id not in expected:
            raise RuntimeError(f"{arm} prediction contains foreign ID: {example_id}")
        if example_id in observed:
            raise RuntimeError(f"{arm} prediction duplicates ID: {example_id}")
        observed[example_id] = item
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise RuntimeError(
            f"{arm} predictions incomplete: {len(observed)}/{len(expected)}; "
            f"first missing={missing[:3]}"
        )

    rescored: list[dict[str, Any]] = []
    for outcome in outcomes:
        example_id = outcome["example_id"]
        item = observed[example_id]
        generated = item.get("generated_text")
        if not isinstance(generated, str):
            raise RuntimeError(
                f"{arm} generated_text is not a string: {example_id}"
            )
        prediction = parse_call(generated)
        stored_target = normalize_required_call(
            item.get("target"), f"{arm} stored target for {example_id}"
        )
        if canonical_targets is None:
            target = stored_target
        else:
            target = canonical_targets[example_id]
            require_exact(
                f"{arm} target agrees with Standard V3 for {example_id}",
                stored_target,
                target,
            )
        require_exact(
            f"{arm} target tool agrees with outcome audit for {example_id}",
            target["name"],
            outcome["target_tool"],
        )
        scored = {
            **outcome,
            "target": target,
            "generated_text": generated,
            "prediction": prediction,
            "json_valid": prediction is not None,
            "tool_name_correct": bool(
                prediction and prediction["name"] == target["name"]
            ),
            "arguments_exact": bool(
                prediction and prediction["arguments"] == target["arguments"]
            ),
            "full_call_exact": prediction == target,
            "has_preference_pair": example_id in pairs,
            "rejected_repeat": (
                prediction == pairs[example_id]["rejected"]
                if example_id in pairs
                else None
            ),
        }
        required_stored = (
            "task_key",
            "recovery_mode",
            "prediction",
            "target",
            *METRIC_KEYS,
        )
        for key in required_stored:
            if key not in item:
                raise RuntimeError(
                    f"{arm} stored prediction lacks {key!r}: {example_id}"
                )
            stored_value = (
                stored_target
                if key == "target"
                else item.get(key)
            )
            require_exact(
                f"{arm} stored {key} for {example_id}",
                stored_value,
                scored[key],
            )
        for key in ("trace_id", "source", "prior_error_type"):
            if outcome.get(key) is not None and key in item:
                require_exact(
                    f"{arm} stored {key} for {example_id}",
                    item.get(key),
                    outcome.get(key),
                )
        rescored.append(scored)
    return rescored


def metric_mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item[key] for item in items if item.get(key) is not None]
    return (
        sum(bool(value) for value in values) / len(values)
        if values
        else None
    )


def task_macro_mean(items: list[dict[str, Any]], key: str) -> float | None:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item.get(key) is not None:
            by_task[item["task_key"]].append(item)
    if not by_task:
        return None
    return sum(metric_mean(rows, key) for rows in by_task.values()) / len(by_task)


def reported_metric_name(group_name: str, metric_name: str) -> str:
    if group_name == "failed_gold" and metric_name == "full_call_exact":
        return "failed_action_imitation_rate"
    if metric_name == "rejected_repeat":
        return FAILED_REPLAY_METRIC
    return metric_name


def group_summary(
    items: list[dict[str, Any]],
    group_name: str,
) -> dict[str, Any]:
    tasks = {item["task_key"] for item in items}
    metrics: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        reported_key = reported_metric_name(group_name, key)
        metrics[reported_key] = {
            "correct": sum(bool(item[key]) for item in items),
            "rate": metric_mean(items, key),
            "task_macro_rate": task_macro_mean(items, key),
        }
        if reported_key == "failed_action_imitation_rate":
            metrics[reported_key]["interpretation"] = "lower_is_better"
    pair_items = [item for item in items if item["has_preference_pair"]]
    metrics[FAILED_REPLAY_METRIC] = {
        "repeats": sum(bool(item["rejected_repeat"]) for item in pair_items),
        "eligible_examples": len(pair_items),
        "rate": metric_mean(pair_items, "rejected_repeat"),
        "task_macro_rate": task_macro_mean(pair_items, "rejected_repeat"),
    }
    return {
        "examples": len(items),
        "tasks": len(tasks),
        "metrics": metrics,
    }


def validate_stored_overall(
    arm: str,
    metrics: dict[str, Any],
    rescored: list[dict[str, Any]],
) -> None:
    stored = metrics.get("groups", {}).get("overall")
    if not isinstance(stored, dict):
        raise RuntimeError(f"{arm} metrics lacks groups.overall")
    require_exact(
        f"{arm} stored overall examples",
        stored.get("examples"),
        len(rescored),
    )
    micro = stored.get("micro")
    if not isinstance(micro, dict):
        raise RuntimeError(f"{arm} metrics overall lacks micro metrics")
    for key in METRIC_KEYS:
        require_close(
            f"{arm} stored overall {key}",
            micro.get(key),
            metric_mean(rescored, key),
        )


def nested_values_for_key(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if nested_key == key:
                found.append(nested_value)
            found.extend(nested_values_for_key(nested_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(nested_values_for_key(item, key))
    return found


def require_recorded_value(
    label: str,
    documents: Iterable[dict[str, Any]],
    keys: Iterable[str],
    expected: Any,
) -> None:
    observed: list[tuple[str, Any]] = []
    for document in documents:
        for key in keys:
            observed.extend((key, value) for value in nested_values_for_key(document, key))
    if not observed:
        raise RuntimeError(
            f"{label} is not recorded under any accepted key: {tuple(keys)!r}"
        )
    disagreements = [(key, value) for key, value in observed if value != expected]
    if disagreements:
        raise RuntimeError(
            f"{label} recorded value drift: {disagreements!r}; expected {expected!r}"
        )


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is non-finite: {value!r}")
    return result


def validate_optional_pair_scores(
    arm: str,
    score_dir: Path,
    pairs: dict[str, dict[str, Any]],
    expected_checkpoint_fingerprint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = score_dir / "metrics.json"
    scores_path = score_dir / "pair_scores.jsonl"
    manifest_path = score_dir / "score_manifest.json"
    metrics = load_json(metrics_path)
    manifest = load_json(manifest_path)
    rows = read_jsonl(scores_path)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise RuntimeError(f"{arm} pair-score manifest lacks protocol")
    require_exact(
        f"{arm} pair-score split",
        manifest.get("split"),
        "test",
    )
    require_exact(
        f"{arm} pair-score training_performed",
        manifest.get("training_performed"),
        False,
    )
    require_exact(f"{arm} pair-score limited", manifest.get("limited"), False)
    require_exact(f"{arm} pair-score complete", manifest.get("complete"), True)
    for key in ("expected_pairs", "completed_pairs", "pair_count"):
        require_exact(
            f"{arm} pair-score manifest {key}",
            manifest.get(key),
            len(pairs),
        )
    require_exact(
        f"{arm} pair-score manifest output rows",
        manifest.get("outputs", {}).get("pair_scores_rows"),
        len(pairs),
    )
    require_exact(f"{arm} pair-score rows", len(rows), len(pairs))
    require_exact(
        f"{arm} pair-score metrics count",
        metrics.get("pair_count"),
        len(pairs),
    )
    require_exact(
        f"{arm} pair-score EOS contract",
        metrics.get("completion_eos_included"),
        True,
    )
    require_recorded_value(
        f"{arm} pair-score input hash",
        (metrics, manifest),
        (
            "preference_pairs_sha256",
            "pairs_file_sha256",
            "pair_file_sha256",
            "input_pairs_sha256",
            "test_preference_pairs_sha256",
        ),
        EXPECTED_TEST_PREFERENCE_PAIRS_SHA256,
    )
    require_recorded_value(
        f"{arm} pair-score output hash",
        (metrics, manifest),
        ("pair_scores_sha256", "scores_sha256", "output_scores_sha256"),
        sha256_file(scores_path),
    )
    require_recorded_value(
        f"{arm} pair-score checkpoint fingerprint",
        (metrics, manifest),
        ("checkpoint_fingerprint", "adapter_fingerprint"),
        expected_checkpoint_fingerprint,
    )
    require_recorded_value(
        f"{arm} pair-score metrics hash",
        (manifest,),
        ("metrics_sha256",),
        sha256_file(metrics_path),
    )
    require_exact(
        f"{arm} pair-score strict input hash audit",
        manifest.get("data", {}).get("strict_pair_file_hash_matches_expected"),
        True,
    )

    by_pair_id: dict[str, dict[str, Any]] = {}
    expected_by_pair_id = {pair["pair_id"]: pair for pair in pairs.values()}
    for expected_index, raw in enumerate(rows):
        require_exact(
            f"{arm} pair-score row split",
            raw.get("split"),
            "test",
        )
        require_exact(
            f"{arm} pair-score row index",
            raw.get("pair_index"),
            expected_index,
        )
        pair_id = raw.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in expected_by_pair_id:
            raise RuntimeError(
                f"{arm} pair score has foreign/missing pair_id: {pair_id!r}"
            )
        if pair_id in by_pair_id:
            raise RuntimeError(f"{arm} pair scores duplicate pair_id: {pair_id}")
        content_hash = raw.get("pair_content_sha256")
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(char not in "0123456789abcdef" for char in content_hash)
        ):
            raise RuntimeError(
                f"{arm} pair score has invalid pair_content_sha256: {pair_id}"
            )
        expected_pair = expected_by_pair_id[pair_id]
        require_exact(
            f"{arm} pair content for {pair_id}",
            content_hash,
            expected_pair["pair_content_sha256"],
        )
        task_key = raw.get("task_key", raw.get("task_id"))
        require_exact(
            f"{arm} pair-score task for {pair_id}",
            task_key,
            expected_pair["task_key"],
        )
        chosen_tokens = raw.get("chosen_tokens_including_eos")
        rejected_tokens = raw.get("rejected_tokens_including_eos")
        if (
            isinstance(chosen_tokens, bool)
            or not isinstance(chosen_tokens, int)
            or chosen_tokens <= 0
            or isinstance(rejected_tokens, bool)
            or not isinstance(rejected_tokens, int)
            or rejected_tokens <= 0
        ):
            raise RuntimeError(
                f"{arm} pair score has invalid EOS-inclusive token counts: {pair_id}"
            )
        chosen_sum = finite_number(
            raw.get("chosen_summed_logp_including_eos"),
            f"{arm} chosen summed logp for {pair_id}",
        )
        rejected_sum = finite_number(
            raw.get("rejected_summed_logp_including_eos"),
            f"{arm} rejected summed logp for {pair_id}",
        )
        summed_margin = finite_number(
            raw.get("summed_logp_margin_chosen_minus_rejected"),
            f"{arm} summed margin for {pair_id}",
        )
        chosen_per_token = finite_number(
            raw.get("chosen_per_token_logp_including_eos"),
            f"{arm} chosen normalized logp for {pair_id}",
        )
        rejected_per_token = finite_number(
            raw.get("rejected_per_token_logp_including_eos"),
            f"{arm} rejected normalized logp for {pair_id}",
        )
        normalized_margin = finite_number(
            raw.get("per_token_normalized_margin_chosen_minus_rejected"),
            f"{arm} normalized margin for {pair_id}",
        )
        require_close(
            f"{arm} recomputed summed margin for {pair_id}",
            summed_margin,
            chosen_sum - rejected_sum,
        )
        require_close(
            f"{arm} recomputed chosen per-token logp for {pair_id}",
            chosen_per_token,
            chosen_sum / chosen_tokens,
        )
        require_close(
            f"{arm} recomputed rejected per-token logp for {pair_id}",
            rejected_per_token,
            rejected_sum / rejected_tokens,
        )
        require_close(
            f"{arm} recomputed normalized margin for {pair_id}",
            normalized_margin,
            chosen_per_token - rejected_per_token,
        )
        summed_correct = raw.get("summed_logp_correct")
        normalized_correct = raw.get("per_token_normalized_correct")
        if not isinstance(summed_correct, bool) or not isinstance(
            normalized_correct, bool
        ):
            raise RuntimeError(f"{arm} pair score lacks boolean correctness: {pair_id}")
        require_exact(
            f"{arm} summed correctness for {pair_id}",
            summed_correct,
            summed_margin > 0.0,
        )
        require_exact(
            f"{arm} normalized correctness for {pair_id}",
            normalized_correct,
            normalized_margin > 0.0,
        )
        by_pair_id[pair_id] = {
            "pair_id": pair_id,
            "example_id": expected_pair["example_id"],
            "task_key": task_key,
            "summed_logp_correct": summed_correct,
            "summed_logp_margin": summed_margin,
            "per_token_normalized_correct": normalized_correct,
            "per_token_normalized_margin": normalized_margin,
        }
    require_exact(
        f"{arm} pair-score ID set",
        set(by_pair_id),
        set(expected_by_pair_id),
    )
    normalized_rows = [
        by_pair_id[pair["pair_id"]]
        for pair in sorted(pairs.values(), key=lambda item: item["pair_id"])
    ]
    summed_margins = [row["summed_logp_margin"] for row in normalized_rows]
    normalized_margins = [
        row["per_token_normalized_margin"] for row in normalized_rows
    ]
    recomputed = {
        "pair_count": len(normalized_rows),
        "pair_accuracy_summed_logp": metric_mean(
            normalized_rows, "summed_logp_correct"
        ),
        "summed_logp_correct_count": sum(
            row["summed_logp_correct"] for row in normalized_rows
        ),
        "summed_logp_tie_count": sum(
            row["summed_logp_margin"] == 0.0 for row in normalized_rows
        ),
        "mean_summed_logp_margin": statistics.fmean(summed_margins),
        "median_summed_logp_margin": statistics.median(summed_margins),
        "per_token_normalized_pair_accuracy": metric_mean(
            normalized_rows, "per_token_normalized_correct"
        ),
        "per_token_normalized_correct_count": sum(
            row["per_token_normalized_correct"] for row in normalized_rows
        ),
        "mean_per_token_normalized_margin": statistics.fmean(
            normalized_margins
        ),
        "median_per_token_normalized_margin": statistics.median(
            normalized_margins
        ),
        "completion_eos_included": True,
    }
    for key, expected in recomputed.items():
        if isinstance(expected, float):
            require_close(f"{arm} pair-score metric {key}", metrics.get(key), expected)
        else:
            require_exact(f"{arm} pair-score metric {key}", metrics.get(key), expected)
    return (
        {
            "status": "complete",
            "score_dir": str(score_dir),
            "metrics_sha256": sha256_file(metrics_path),
            "pair_scores_sha256": sha256_file(scores_path),
            "score_manifest_sha256": sha256_file(manifest_path),
            "metrics": recomputed,
        },
        normalized_rows,
    )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise RuntimeError("cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_task_aggregates(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    key: str,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    control_by_id = {item["example_id"]: item for item in control}
    if set(control_by_id) != {item["example_id"] for item in treatment}:
        raise RuntimeError(f"paired ID mismatch for metric {key}")
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in treatment:
        other = control_by_id[item["example_id"]]
        treatment_value = item.get(key)
        control_value = other.get(key)
        if treatment_value is None or control_value is None:
            continue
        task_key = item["task_key"]
        require_exact(
            f"paired task_key for {item['example_id']}",
            other["task_key"],
            task_key,
        )
        by_task[task_key][0] += int(bool(treatment_value)) - int(
            bool(control_value)
        )
        by_task[task_key][1] += 1
    tasks = sorted(by_task)
    if not tasks:
        raise RuntimeError(f"no eligible paired examples for metric {key}")
    return tasks, {task: tuple(by_task[task]) for task in tasks}


def paired_task_bootstrap(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    key: str,
    seed: int,
    samples: int,
) -> list[float]:
    tasks, aggregates = paired_task_aggregates(treatment, control, key)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        numerator = 0
        denominator = 0
        for _ in tasks:
            difference, count = aggregates[rng.choice(tasks)]
            numerator += difference
            denominator += count
        draws.append(numerator / denominator)
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def paired_cluster_sign_flip_p(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    key: str,
    seed: int,
    samples: int,
) -> float:
    tasks, aggregates = paired_task_aggregates(treatment, control, key)
    observed = abs(sum(aggregates[task][0] for task in tasks))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        permuted = sum(
            aggregates[task][0] * (-1 if rng.random() < 0.5 else 1)
            for task in tasks
        )
        if abs(permuted) >= observed:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def paired_metric(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    key: str,
    seed: int,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> dict[str, Any]:
    control_by_id = {item["example_id"]: item for item in control}
    if set(control_by_id) != {item["example_id"] for item in treatment}:
        raise RuntimeError(f"paired ID mismatch for metric {key}")
    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in treatment:
        other = control_by_id[item["example_id"]]
        if item.get(key) is not None and other.get(key) is not None:
            eligible.append((item, other))
    if not eligible:
        raise RuntimeError(f"no paired observations for metric {key}")
    treatment_items = [pair[0] for pair in eligible]
    control_items = [pair[1] for pair in eligible]
    treatment_rate = metric_mean(treatment_items, key)
    control_rate = metric_mean(control_items, key)
    treatment_only_true = sum(
        bool(left[key]) and not bool(right[key]) for left, right in eligible
    )
    control_only_true = sum(
        not bool(left[key]) and bool(right[key]) for left, right in eligible
    )
    delta = treatment_rate - control_rate
    result = {
        "eligible_examples": len(eligible),
        "treatment": treatment_rate,
        "control": control_rate,
        "delta_treatment_minus_control": delta,
        "treatment_only_true": treatment_only_true,
        "control_only_true": control_only_true,
        "paired_ties": len(eligible)
        - treatment_only_true
        - control_only_true,
        "task_cluster_bootstrap_95_delta": paired_task_bootstrap(
            treatment_items,
            control_items,
            key,
            seed,
            bootstrap_samples,
        ),
        "task_cluster_sign_flip_two_sided_p": paired_cluster_sign_flip_p(
            treatment_items,
            control_items,
            key,
            seed + 1,
            sign_flip_samples,
        ),
    }
    if key == "rejected_repeat":
        result["improvement_control_minus_treatment"] = -delta
        result["interpretation"] = "lower_is_better"
    else:
        result["improvement_treatment_minus_control"] = delta
        result["interpretation"] = "higher_is_better"
    return result


def paired_numeric_metric(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    key: str,
    seed: int,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> dict[str, Any]:
    control_by_id = {item["example_id"]: item for item in control}
    require_exact(
        f"paired numeric ID set for {key}",
        set(control_by_id),
        {item["example_id"] for item in treatment},
    )
    by_task: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    differences: list[float] = []
    treatment_values: list[float] = []
    control_values: list[float] = []
    for item in treatment:
        other = control_by_id[item["example_id"]]
        require_exact(
            f"paired numeric task for {item['example_id']}",
            other["task_key"],
            item["task_key"],
        )
        treatment_value = finite_number(
            item.get(key), f"treatment {key} for {item['example_id']}"
        )
        control_value = finite_number(
            other.get(key), f"control {key} for {item['example_id']}"
        )
        difference = treatment_value - control_value
        treatment_values.append(treatment_value)
        control_values.append(control_value)
        differences.append(difference)
        by_task[item["task_key"]][0] += difference
        by_task[item["task_key"]][1] += 1
    tasks = sorted(by_task)
    if not tasks:
        raise RuntimeError(f"no paired numeric values for {key}")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        numerator = 0.0
        denominator = 0
        for _ in tasks:
            difference_sum, count = by_task[rng.choice(tasks)]
            numerator += float(difference_sum)
            denominator += int(count)
        draws.append(numerator / denominator)
    observed = abs(sum(differences))
    rng = random.Random(seed + 1)
    extreme = 0
    for _ in range(sign_flip_samples):
        permuted = sum(
            float(by_task[task][0]) * (-1 if rng.random() < 0.5 else 1)
            for task in tasks
        )
        if abs(permuted) >= observed:
            extreme += 1
    treatment_mean = statistics.fmean(treatment_values)
    control_mean = statistics.fmean(control_values)
    return {
        "eligible_examples": len(differences),
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "delta_treatment_minus_control": treatment_mean - control_mean,
        "task_cluster_bootstrap_95_delta": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
        "task_cluster_sign_flip_two_sided_p": (
            (extreme + 1) / (sign_flip_samples + 1)
        ),
        "interpretation": "higher_is_better",
    }


def compare_pair_scores(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> dict[str, Any]:
    return {
        "treatment_arm": "dpo",
        "control_arm": "continued_sft",
        "pairs": len(treatment),
        "metrics": {
            "pair_accuracy_summed_logp": paired_metric(
                treatment,
                control,
                "summed_logp_correct",
                seed,
                bootstrap_samples,
                sign_flip_samples,
            ),
            "mean_summed_logp_margin": paired_numeric_metric(
                treatment,
                control,
                "summed_logp_margin",
                seed + 10,
                bootstrap_samples,
                sign_flip_samples,
            ),
            "per_token_normalized_pair_accuracy": paired_metric(
                treatment,
                control,
                "per_token_normalized_correct",
                seed + 20,
                bootstrap_samples,
                sign_flip_samples,
            ),
            "mean_per_token_normalized_margin": paired_numeric_metric(
                treatment,
                control,
                "per_token_normalized_margin",
                seed + 30,
                bootstrap_samples,
                sign_flip_samples,
            ),
        },
    }


def paired_comparison(
    treatment_name: str,
    control_name: str,
    arm_items: dict[str, list[dict[str, Any]]],
    seed: int,
    bootstrap_samples: int,
    sign_flip_samples: int,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group_index, group_name in enumerate(GROUPS):
        treatment = select_group(arm_items[treatment_name], group_name)
        control = select_group(arm_items[control_name], group_name)
        group_metrics: dict[str, Any] = {}
        for metric_index, key in enumerate((*METRIC_KEYS, "rejected_repeat")):
            # Rejected-repeat is defined only on the 48 strict preference pairs.
            # It remains useful when a larger group is selected because the
            # paired_metric function transparently restricts to eligible IDs.
            if key == "rejected_repeat" and not any(
                item["has_preference_pair"] for item in treatment
            ):
                continue
            reported_key = reported_metric_name(group_name, key)
            paired_result = paired_metric(
                treatment,
                control,
                key,
                seed + group_index * 100 + metric_index * 2,
                bootstrap_samples,
                sign_flip_samples,
            )
            if reported_key == "failed_action_imitation_rate":
                paired_result.pop("improvement_treatment_minus_control", None)
                paired_result["improvement_control_minus_treatment"] = -paired_result[
                    "delta_treatment_minus_control"
                ]
                paired_result["interpretation"] = "lower_is_better"
            group_metrics[reported_key] = paired_result
        groups[group_name] = {
            "examples": len(treatment),
            "metrics": group_metrics,
        }
    return {
        "treatment_arm": treatment_name,
        "control_arm": control_name,
        "groups": groups,
    }


def holm_adjust(
    named_p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(named_p_values.items(), key=lambda item: item[1])
    adjusted_sorted: list[tuple[str, float, float]] = []
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * p_value)
        running = max(running, adjusted)
        adjusted_sorted.append((name, p_value, running))
    return {
        name: {
            "raw_p": raw,
            "holm_adjusted_p": adjusted,
            "reject_at_alpha_0_05": adjusted <= alpha,
        }
        for name, raw, adjusted in adjusted_sorted
    }


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# QLoRA V4 paired comparison",
        "",
        (
            "All values are recomputed from complete prediction JSONL files. "
            "Intervals are paired 10,000-draw task-cluster bootstrap intervals."
        ),
        "",
    ]
    for contrast_name, contrast in report["paired_comparisons"].items():
        lines.extend(
            [
                f"## {contrast_name}",
                "",
                "| group / metric | treatment | control | delta | treatment-only true | control-only true | task-cluster 95% CI |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for group_name in GROUPS:
            metric_name = reported_metric_name(group_name, "full_call_exact")
            metric = contrast["groups"][group_name]["metrics"][metric_name]
            interval = metric["task_cluster_bootstrap_95_delta"]
            display_group = (
                f"{group_name} / {metric_name}"
                if group_name == "failed_gold"
                else group_name
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        display_group,
                        f"{metric['treatment']:.6f}",
                        f"{metric['control']:.6f}",
                        f"{metric['delta_treatment_minus_control']:+.6f}",
                        str(metric["treatment_only_true"]),
                        str(metric["control_only_true"]),
                        f"[{interval[0]:+.6f}, {interval[1]:+.6f}]",
                    )
                )
                + " |"
            )
        rejected = contrast["groups"]["recovery_success"]["metrics"][
            FAILED_REPLAY_METRIC
        ]
        lines.extend(
            [
                "",
                (
                    "Rejected-repeat rate on the 48 strict preference-pair examples: "
                    f"treatment={rejected['treatment']:.6f}, "
                    f"control={rejected['control']:.6f}, "
                    f"delta={rejected['delta_treatment_minus_control']:+.6f} "
                    "(lower is better)."
                ),
                "",
            ]
        )
    pair_scoring = report["pair_scoring"]
    lines.extend(
        [
            "## Required chosen/rejected log-probability scoring",
            "",
            "| arm | status | summed-logp accuracy | mean summed margin | normalized accuracy | mean normalized margin |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        summary = pair_scoring["arms"][arm]
        if summary["status"] != "complete":
            lines.append(f"| {arm} | {summary['status']} | — | — | — | — |")
            continue
        metrics = summary["metrics"]
        lines.append(
            "| "
            + " | ".join(
                (
                    arm,
                    "complete",
                    f"{metrics['pair_accuracy_summed_logp']:.6f}",
                    f"{metrics['mean_summed_logp_margin']:+.6f}",
                    f"{metrics['per_token_normalized_pair_accuracy']:.6f}",
                    f"{metrics['mean_per_token_normalized_margin']:+.6f}",
                )
            )
            + " |"
        )
    score_comparison = pair_scoring["dpo_minus_continued_sft"]
    if score_comparison["status"] == "complete":
        lines.extend(
            [
                "",
                "DPO minus continued-SFT pair-score deltas:",
                "",
            ]
        )
        for metric_name, metric in score_comparison["metrics"].items():
            delta = metric["delta_treatment_minus_control"]
            lines.append(f"- {metric_name}: {delta:+.6f}")
    gate = report["dpo_exploratory_screening_gate"]
    lines.extend(
        [
            "",
            "## Frozen exploratory screening gate",
            "",
            (
                f"Decision: **{gate['decision']}**. This gate only decides "
                "whether to run fresh three-seed confirmation; it is not a "
                "paper-final claim."
            ),
            "",
            report["multiple_testing_note"],
            "",
        ]
    )
    lines.extend(
        [
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    fields = (
        "comparison",
        "treatment_arm",
        "control_arm",
        "group",
        "metric",
        "eligible_examples",
        "treatment",
        "control",
        "delta_treatment_minus_control",
        "treatment_only_true",
        "control_only_true",
        "paired_ties",
        "ci_low",
        "ci_high",
        "cluster_sign_flip_p",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for contrast_name, contrast in report["paired_comparisons"].items():
            for group_name, group in contrast["groups"].items():
                for metric_name, metric in group["metrics"].items():
                    interval = metric["task_cluster_bootstrap_95_delta"]
                    writer.writerow(
                        {
                            "comparison": contrast_name,
                            "treatment_arm": contrast["treatment_arm"],
                            "control_arm": contrast["control_arm"],
                            "group": group_name,
                            "metric": metric_name,
                            "eligible_examples": metric["eligible_examples"],
                            "treatment": metric["treatment"],
                            "control": metric["control"],
                            "delta_treatment_minus_control": metric[
                                "delta_treatment_minus_control"
                            ],
                            "treatment_only_true": metric[
                                "treatment_only_true"
                            ],
                            "control_only_true": metric["control_only_true"],
                            "paired_ties": metric["paired_ties"],
                            "ci_low": interval[0],
                            "ci_high": interval[1],
                            "cluster_sign_flip_p": metric[
                                "task_cluster_sign_flip_two_sided_p"
                            ],
                        }
                    )
    temporary.replace(path)


def aggregate(
    result_dirs: dict[str, Path],
    outcomes_path: Path,
    preferences_path: Path,
    *,
    pair_score_dirs: dict[str, Path | None] | None = None,
    expected_test_sha256: str = EXPECTED_TEST_SHA256,
    expected_examples: int = FROZEN_FORMAL_EXAMPLES,
    expected_group_counts: dict[str, int] | None = EXPECTED_GROUP_COUNTS,
    expected_outcomes_sha256: str | None = EXPECTED_TEST_OUTCOMES_SHA256,
    expected_preferences_sha256: str | None = (
        EXPECTED_TEST_PREFERENCE_PAIRS_SHA256
    ),
    expected_preference_pairs: int = EXPECTED_TEST_PREFERENCE_PAIRS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    sign_flip_samples: int = SIGN_FLIP_SAMPLES,
) -> dict[str, Any]:
    require_exact("V4 arm names", tuple(result_dirs), ARM_ORDER)
    if bootstrap_samples <= 0 or sign_flip_samples <= 0:
        raise RuntimeError("bootstrap/sign-flip sample counts must be positive")
    outcomes = normalize_outcomes(
        outcomes_path,
        expected_examples=expected_examples,
        expected_group_counts=expected_group_counts,
    )
    if expected_outcomes_sha256 is not None:
        require_exact(
            "frozen test outcome audit hash",
            sha256_file(outcomes_path),
            expected_outcomes_sha256,
        )
    if expected_preferences_sha256 is not None:
        require_exact(
            "frozen test preference-pair hash",
            sha256_file(preferences_path),
            expected_preferences_sha256,
        )
    pairs = normalize_preference_pairs(
        preferences_path,
        outcomes,
        expected_pairs=expected_preference_pairs,
    )

    arm_items: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, Any] = {}
    training_manifests: dict[str, dict[str, Any]] = {}
    input_artifacts: dict[str, Any] = {
        "test_outcomes": {
            "path": str(outcomes_path),
            "sha256": sha256_file(outcomes_path),
            "rows": len(outcomes),
        },
        "test_preference_pairs": {
            "path": str(preferences_path),
            "sha256": sha256_file(preferences_path),
            "rows": len(pairs),
        },
        "arms": {},
    }
    reference_contract: dict[str, Any] | None = None
    canonical_targets: dict[str, dict[str, Any]] | None = None
    for arm in ARM_ORDER:
        result_dir = result_dirs[arm]
        metrics_path = result_dir / "metrics.json"
        contract_path = result_dir / "metrics.contract.json"
        predictions_path = result_dir / "metrics.predictions.jsonl"
        if arm == "standard_v3":
            require_exact(
                "Standard V3 metrics artifact",
                sha256_file(metrics_path),
                STANDARD_V3_METRICS_SHA256,
            )
            require_exact(
                "Standard V3 contract artifact",
                sha256_file(contract_path),
                STANDARD_V3_CONTRACT_SHA256,
            )
            require_exact(
                "Standard V3 predictions artifact",
                sha256_file(predictions_path),
                STANDARD_V3_PREDICTIONS_SHA256,
            )
        metrics = load_json(metrics_path)
        contract = load_json(contract_path)
        validate_core_contract(
            arm,
            metrics,
            contract,
            predictions_path,
            expected_test_sha256,
            expected_examples,
            reference_contract,
        )
        if arm == "standard_v3":
            reference_contract = contract
            require_exact(
                "Standard V3 checkpoint fingerprint",
                contract.get("checkpoint_fingerprint"),
                STANDARD_V3_CHECKPOINT_FINGERPRINT,
            )
        else:
            training_manifests[arm] = validate_new_arm_training_identity(
                arm,
                result_dir,
                contract["checkpoint_fingerprint"],
            )
        rescored = rescore_predictions(
            arm,
            predictions_path,
            outcomes,
            pairs,
            canonical_targets,
        )
        if arm == "standard_v3":
            canonical_targets = {
                row["example_id"]: row["target"] for row in rescored
            }
            for example_id, pair in pairs.items():
                require_exact(
                    f"test preference chosen equals Standard V3 target for {example_id}",
                    pair["chosen"],
                    canonical_targets[example_id],
                )
        validate_stored_overall(arm, metrics, rescored)
        arm_items[arm] = rescored
        arm_summaries[arm] = {
            "result_dir": str(result_dir),
            "protocol": metrics.get("protocol"),
            "checkpoint_fingerprint": metrics.get("checkpoint_fingerprint"),
            "groups": {
                group: group_summary(select_group(rescored, group), group)
                for group in GROUPS
            },
        }
        input_artifacts["arms"][arm] = {
            "result_dir": str(result_dir),
            "metrics_sha256": sha256_file(metrics_path),
            "contract_sha256": sha256_file(contract_path),
            "predictions_sha256": sha256_file(predictions_path),
            "checkpoint_fingerprint": contract["checkpoint_fingerprint"],
        }
        if arm != "standard_v3":
            input_artifacts["arms"][arm]["run_manifest_sha256"] = sha256_file(
                result_dir / "run_manifest.json"
            )

    fingerprints = {
        arm: summary["checkpoint_fingerprint"]
        for arm, summary in arm_summaries.items()
    }
    if len(set(fingerprints.values())) != len(ARM_ORDER):
        raise RuntimeError(
            f"four arm checkpoints are not distinct: {fingerprints}"
        )
    source_commits = {
        manifest["source_commit"] for manifest in training_manifests.values()
    }
    require_exact("new-arm source commit count", len(source_commits), 1)
    clean_fingerprint = fingerprints["clean_sft"]
    comparison_ids: set[str] = set()
    for arm in ("continued_sft", "dpo"):
        manifest = training_manifests[arm]
        require_exact(
            f"{arm} Clean-SFT initialization",
            (manifest.get("clean_sft_initialization") or {}).get(
                "checkpoint_fingerprint"
            ),
            clean_fingerprint,
        )
        comparison_id = manifest.get("comparison_contract_id")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise RuntimeError(f"{arm} lacks comparison_contract_id")
        comparison_ids.add(comparison_id)
    require_exact(
        "continued-SFT/DPO comparison contract count",
        len(comparison_ids),
        1,
    )

    comparisons = {
        name: paired_comparison(
            treatment,
            control,
            arm_items,
            FROZEN_SEED + index * 10_000,
            bootstrap_samples,
            sign_flip_samples,
        )
        for index, (name, (treatment, control)) in enumerate(CONTRASTS.items())
    }
    supplied_pair_score_dirs = pair_score_dirs or {}
    required_pair_score_arms = {
        "clean_sft",
        "continued_sft",
        "dpo",
    }
    if set(supplied_pair_score_dirs) != required_pair_score_arms:
        raise RuntimeError(
            "pair-score arms must be exactly "
            f"{sorted(required_pair_score_arms)}; received "
            f"{sorted(supplied_pair_score_dirs)}"
        )
    pair_scoring: dict[str, Any] = {
        "standard_v3": {
            "status": "not_available",
            "reason": (
                "The existing Standard V3 generation artifact has no frozen "
                "chosen/rejected log-probability scoring pass."
            ),
        }
    }
    pair_score_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ("clean_sft", "continued_sft", "dpo"):
        score_dir = supplied_pair_score_dirs.get(arm)
        if score_dir is None:
            raise RuntimeError(f"required pair-score directory missing for {arm}")
        summary, rows = validate_optional_pair_scores(
            arm,
            score_dir,
            pairs,
            arm_summaries[arm]["checkpoint_fingerprint"],
        )
        pair_scoring[arm] = summary
        pair_score_rows[arm] = rows
    pair_score_comparison: dict[str, Any] = {
        "status": "complete",
        **compare_pair_scores(
            pair_score_rows["dpo"],
            pair_score_rows["continued_sft"],
            FROZEN_SEED + 50_000,
            bootstrap_samples,
            sign_flip_samples,
        ),
    }
    primary_p_values = {
        name: comparison["groups"]["recovery_success"]["metrics"][
            "full_call_exact"
        ]["task_cluster_sign_flip_two_sided_p"]
        for name, comparison in comparisons.items()
        if name in PRIMARY_CONTRASTS
    }
    dpo_contrast = comparisons["dpo_minus_continued_sft"]
    recovery_test = dpo_contrast["groups"]["recovery_success"]["metrics"][
        "full_call_exact"
    ]
    non_recovery_test = dpo_contrast["groups"][
        "non_recovery_success"
    ]["metrics"]["full_call_exact"]
    replay_test = dpo_contrast["groups"]["recovery_success"]["metrics"][
        FAILED_REPLAY_METRIC
    ]
    ranking_test = pair_score_comparison["metrics"][
        "pair_accuracy_summed_logp"
    ]
    screening_thresholds = {
        "recovery_success_min_delta": 0.04,
        "exact_failed_call_replay_max_delta": -(2 / 48),
        "pair_ranking_min_delta": 3 / 48,
        "non_recovery_success_min_delta": -0.02,
    }
    screening_criteria = {
        "recovery_success": (
            recovery_test["delta_treatment_minus_control"]
            >= screening_thresholds["recovery_success_min_delta"]
        ),
        "exact_failed_call_replay": (
            replay_test["delta_treatment_minus_control"]
            <= screening_thresholds["exact_failed_call_replay_max_delta"]
        ),
        "pair_ranking": (
            ranking_test["delta_treatment_minus_control"]
            >= screening_thresholds["pair_ranking_min_delta"]
        ),
        "non_recovery_success": (
            non_recovery_test["delta_treatment_minus_control"]
            >= screening_thresholds["non_recovery_success_min_delta"]
        ),
    }
    directional_advance = all(screening_criteria.values())
    statistically_supported = (
        recovery_test["task_cluster_bootstrap_95_delta"][0] > 0.0
        and non_recovery_test["task_cluster_bootstrap_95_delta"][0]
        >= screening_thresholds["non_recovery_success_min_delta"]
    )
    report = {
        "protocol": PROTOCOL,
        "valid": True,
        "frozen_test_sha256": expected_test_sha256,
        "examples": expected_examples,
        "bootstrap": {
            "unit": "task_key",
            "paired": True,
            "samples": bootstrap_samples,
            "interval": "percentile_95",
            "seed": FROZEN_SEED,
        },
        "cluster_sign_flip": {
            "unit": "task_key",
            "samples": sign_flip_samples,
            "two_sided": True,
        },
        "group_definitions": GROUP_DEFINITIONS,
        "input_artifacts": input_artifacts,
        "arms": arm_summaries,
        "paired_comparisons": comparisons,
        "pair_scoring": {
            "completion_eos_included": True,
            "strict_pair_examples": len(pairs),
            "standard_v3_missing_is_nonblocking": True,
            "arms": pair_scoring,
            "dpo_minus_continued_sft": pair_score_comparison,
            "note": (
                "Summed completion log probability is primary. Per-token "
                "normalization is diagnostic because it changes the preference "
                "quantity. These scores are separate from generation exact-match "
                "and rejected-repeat metrics."
            ),
        },
        "holm_primary_recovery_success_full_call": {
            "family": list(PRIMARY_CONTRASTS),
            "alpha": 0.05,
            "method": "Holm",
            "tests": holm_adjust(primary_p_values),
        },
        "dpo_exploratory_screening_gate": {
            "frozen_before_v4_results": True,
            "purpose": (
                "method-advancement screen only; not a paper-final "
                "significance claim"
            ),
            "thresholds": screening_thresholds,
            "criteria_pass": screening_criteria,
            "directional_advance_gate_pass": directional_advance,
            "statistically_supported_exploratory_signal": (
                statistically_supported
            ),
            "decision": (
                "advance_to_fresh_three_seed_confirmation"
                if directional_advance
                else "do_not_claim_positive_mechanism_from_v4"
            ),
        },
        "multiple_testing_note": (
            "Only the two preregistered primary recovery-success full-call "
            "tests receive Holm adjustment. All other p-values are "
            "uncorrected exploratory diagnostics."
        ),
        "claim_boundary": (
            "Offline next-tool-call evaluation on the already inspected frozen "
            "V2/V3 test set. Single-seed task-cluster intervals do not cover "
            "training randomness and cannot establish end-to-end Agent success, "
            "cross-seed robustness, or paper-final confirmation."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-v3-result-dir",
        type=Path,
        default=Path("results/qlora_v3/constrained_recovery"),
    )
    parser.add_argument(
        "--clean-sft-result-dir",
        type=Path,
        default=Path("results/qlora_v4/clean_sft"),
    )
    parser.add_argument(
        "--continued-sft-result-dir",
        type=Path,
        default=Path("results/qlora_v4/continued_sft"),
    )
    parser.add_argument(
        "--dpo-result-dir",
        type=Path,
        default=Path("results/qlora_v4/dpo"),
    )
    parser.add_argument(
        "--clean-sft-pair-score-dir",
        type=Path,
        required=True,
        help=(
            "Required clean_sft directory containing metrics.json, "
            "pair_scores.jsonl, and score_manifest.json."
        ),
    )
    parser.add_argument(
        "--continued-sft-pair-score-dir",
        type=Path,
        required=True,
        help=(
            "Required continued_sft directory containing strict pair scores."
        ),
    )
    parser.add_argument(
        "--dpo-pair-score-dir",
        type=Path,
        required=True,
        help="Required dpo directory containing strict pair scores.",
    )
    parser.add_argument(
        "--test-outcomes",
        type=Path,
        default=Path("data/processed/qlora_v4/evaluation/test_outcomes.jsonl"),
    )
    parser.add_argument(
        "--test-preference-pairs",
        type=Path,
        default=Path(
            "data/processed/qlora_v4/evaluation/test_preference_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis_v4"),
    )
    args = parser.parse_args()

    report = aggregate(
        {
            "standard_v3": args.standard_v3_result_dir,
            "clean_sft": args.clean_sft_result_dir,
            "continued_sft": args.continued_sft_result_dir,
            "dpo": args.dpo_result_dir,
        },
        args.test_outcomes,
        args.test_preference_pairs,
        pair_score_dirs={
            "clean_sft": args.clean_sft_pair_score_dir,
            "continued_sft": args.continued_sft_pair_score_dir,
            "dpo": args.dpo_pair_score_dir,
        },
    )
    # No output directory or partial analysis is created until every arm and
    # every prepare-time audit artifact has passed validation.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.output_dir / "comparison.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        args.output_dir / "comparison.md",
        render_markdown(report),
    )
    write_csv(args.output_dir / "comparison.csv", report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": PROTOCOL,
                "examples": report["examples"],
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
