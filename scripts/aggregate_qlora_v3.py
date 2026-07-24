#!/usr/bin/env python3
"""Audit and aggregate V3, optionally against the frozen V2 random arm.

Predictions are treated as the source of truth: this script reparses every
generated string and recomputes the core metrics before accepting the stored
``metrics.json``.  When a compatible V2 ``random_success`` result is supplied,
it also reports paired, same-example differences.  If the reference directory
is genuinely absent, a standalone V3 report is still produced, but no
directional judgement is allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_tool_actions_v3 import (
    FROZEN_FORMAL_EXAMPLES,
    FROZEN_MAX_NEW_TOKENS,
    FROZEN_MAX_PROMPT_TOKENS,
    FROZEN_MODEL,
    FROZEN_MODEL_REVISION,
    FROZEN_SEED,
    METRIC_KEYS,
    checkpoint_fingerprint,
    normalize_call,
    parse_call,
)
from run_qlora_v3 import load_and_validate_contract

BUILD_PROTOCOL = "qlora_v3_constrained_recovery"
V3_EVAL_PROTOCOL = "qlora_v3"
V2_EVAL_PROTOCOL = "qlora_v2"
ARM = "constrained_recovery"
EXPECTED_VALIDATION_SHA256 = "71f66c41394a50e3d992b7b860bb444774e7215ada3ad95e7d749b911916057a"
EXPECTED_TEST_SHA256 = "0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96"
V2_RANDOM_TRAIN_SCHEDULE_SHA256 = "e0bfc19652f339a4ce748acf7f63a5f2f7d0be18658f265b9806345e36012743"
V2_RANDOM_CHECKPOINT_FINGERPRINT = "f0a4d03b660d0e86265b66e64fa7f988cbce9cb419c9a63e9f201ca26f84559f"
EXPECTED_SOURCE_BUDGETS = {
    "gpt-4o-retail": 523_182,
    "sonnet-35-new-retail": 1_167_747,
}
FROZEN_GENERATION = {
    "do_sample": False,
    "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
    "batch_size": 1,
}
GROUPS = (
    "overall",
    "non_recovery",
    "recovery",
    "agent_initiated",
    "user_assisted",
)
V2_STORED_GROUPS = (
    "overall",
    "recovery",
    "agent_initiated",
    "user_assisted",
)
EXPECTED_PACKAGES = {
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "bitsandbytes": "0.46.0",
    "datasets": "3.6.0",
    "accelerate": "1.7.0",
    "sentencepiece": "0.2.0",
    "safetensors": "0.5.3",
    "numpy": "1.26.4",
}
V2_RANDOM_OVERALL_FULL_CALL = 0.3305526590198123
V2_RANDOM_NON_RECOVERY_CORRECT = 286
NON_RECOVERY_EXAMPLES = 906
V2_RANDOM_RECOVERY_CORRECT = 31
RECOVERY_EXAMPLES = 53
OVERALL_RETENTION_FLOOR = V2_RANDOM_OVERALL_FULL_CALL - 0.02
NON_RECOVERY_RETENTION_FLOOR = (
    V2_RANDOM_NON_RECOVERY_CORRECT / NON_RECOVERY_EXAMPLES
) - 0.02
RECOVERY_SIGNAL_FLOOR = (V2_RANDOM_RECOVERY_CORRECT + 2) / RECOVERY_EXAMPLES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"blank line in formal JSONL artifact {path}:{line_number}")
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
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {observed!r} != recomputed {expected!r}")


def mean(items: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(item[key]) for item in items) / len(items) if items else None


def select_group(items: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "overall":
        return items
    if name == "non_recovery":
        return [item for item in items if item["recovery_mode"] == "none"]
    if name == "recovery":
        return [item for item in items if item["recovery_mode"] != "none"]
    if name == "agent_initiated":
        return [item for item in items if item["recovery_mode"] == "agent_initiated"]
    if name == "user_assisted":
        return [item for item in items if item["recovery_mode"] == "user_assisted"]
    raise KeyError(name)


def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_key"]].append(item)
    return {
        "examples": len(items),
        "tasks": len(by_task),
        "micro": {key: mean(items, key) for key in METRIC_KEYS},
        "task_macro": {
            key: (
                sum(mean(task_items, key) for task_items in by_task.values()) / len(by_task)
                if by_task
                else None
            )
            for key in METRIC_KEYS
        },
    }


def rescore_predictions(
    predictions_path: Path,
    test_rows: list[dict[str, Any]],
    expected_protocol_label: str,
) -> list[dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for row in test_rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or example_id in expected:
            raise RuntimeError(f"invalid or duplicate frozen test example_id: {example_id!r}")
        expected[example_id] = row
    observed: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(predictions_path):
        example_id = item.get("example_id")
        if not isinstance(example_id, str):
            raise RuntimeError(f"{expected_protocol_label} prediction lacks a string example_id")
        if example_id not in expected:
            raise RuntimeError(f"{expected_protocol_label} prediction contains foreign ID: {example_id}")
        if example_id in observed:
            raise RuntimeError(f"{expected_protocol_label} predictions duplicate ID: {example_id}")
        observed[example_id] = item
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise RuntimeError(
            f"{expected_protocol_label} predictions are incomplete: "
            f"{len(observed)}/{len(expected)}; first missing={missing[:3]}"
        )

    rescored: list[dict[str, Any]] = []
    for row in test_rows:
        item = observed[row["example_id"]]
        generated = item.get("generated_text")
        if not isinstance(generated, str):
            raise RuntimeError(f"{expected_protocol_label} generated_text is not a string: {row['example_id']}")
        prediction = parse_call(generated)
        target = normalize_call(json.loads(row["completion"]))
        if target is None:
            raise RuntimeError(f"invalid frozen test target: {row['example_id']}")
        rescored_item = {
            "example_id": row["example_id"],
            "trace_id": row["trace_id"],
            "task_key": row["task_key"],
            "source": row["source"],
            "recovery_mode": row["recovery_mode"],
            "prior_error_type": row.get("prior_error_type"),
            "generated_text": generated,
            "prediction": prediction,
            "target": target,
            "json_valid": prediction is not None,
            "tool_name_correct": bool(prediction and prediction["name"] == target["name"]),
            "arguments_exact": bool(prediction and prediction["arguments"] == target["arguments"]),
            "full_call_exact": prediction == target,
        }
        for key in (
            "trace_id",
            "task_key",
            "source",
            "recovery_mode",
            "prior_error_type",
            "prediction",
            "target",
            *METRIC_KEYS,
        ):
            if item.get(key) != rescored_item[key]:
                raise RuntimeError(
                    f"{expected_protocol_label} stored prediction field {key!r} "
                    f"does not match recomputation for {row['example_id']}"
                )
        rescored.append(rescored_item)
    return rescored


def validate_stored_groups(
    metrics: dict[str, Any],
    rescored: list[dict[str, Any]],
    label: str,
    stored_groups: tuple[str, ...] = GROUPS,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for group_name in GROUPS:
        summary = group_summary(select_group(rescored, group_name))
        if group_name not in stored_groups:
            summaries[group_name] = summary
            continue
        stored = metrics.get("groups", {}).get(group_name, {})
        require_exact(f"{label} {group_name} example count", stored.get("examples"), summary["examples"])
        require_exact(f"{label} {group_name} task count", stored.get("tasks"), summary["tasks"])
        for key in METRIC_KEYS:
            require_close(
                f"{label} {group_name} {key}",
                stored.get("micro", {}).get(key),
                summary["micro"][key],
            )
            require_close(
                f"{label} {group_name} task-macro {key}",
                stored.get("task_macro", {}).get(key),
                summary["task_macro"][key],
            )
        summaries[group_name] = summary
    return summaries


def validate_evaluation_contract(
    metrics: dict[str, Any],
    contract: dict[str, Any],
    protocol: str,
    label: str,
) -> None:
    expected = {
        "protocol": protocol,
        "test_file_sha256": EXPECTED_TEST_SHA256,
        "formal_test_examples": FROZEN_FORMAL_EXAMPLES,
        "evaluated_examples": FROZEN_FORMAL_EXAMPLES,
        "model": FROZEN_MODEL,
        "model_revision": FROZEN_MODEL_REVISION,
        "base_model_loading": "nf4_4bit",
        "max_prompt_tokens": FROZEN_MAX_PROMPT_TOKENS,
        "generation": FROZEN_GENERATION,
        "limited": False,
    }
    for key, value in expected.items():
        require_exact(f"{label} metrics {key}", metrics.get(key), value)
        require_exact(f"{label} contract {key}", contract.get(key), value)
    if protocol == V3_EVAL_PROTOCOL:
        require_exact(
            f"{label} frozen training/evaluation declaration",
            contract.get("training_and_evaluation_protocol"),
            "qlora_v2_frozen",
        )
    require_exact(
        f"{label} checkpoint fingerprint",
        metrics.get("checkpoint_fingerprint"),
        contract.get("checkpoint_fingerprint"),
    )
    for key, value in contract.items():
        require_exact(f"{label} contract mirrored in metrics: {key}", metrics.get(key), value)


def validate_v3_training(
    result_dir: Path,
    build: dict[str, Any],
) -> None:
    manifest = load_json(result_dir / "run_manifest.json")
    expected = {
        "protocol": "qlora_v2",
        "objective": "completion_only_causal_language_model_cross_entropy",
        "model": FROZEN_MODEL,
        "model_revision": FROZEN_MODEL_REVISION,
        "seed": FROZEN_SEED,
        "max_seq_len": 2048,
        "max_steps": 68,
        "batch_size": 1,
        "grad_accum": 16,
        "learning_rate": 1e-4,
        "smoke_test": False,
    }
    for key, value in expected.items():
        require_exact(f"V3 run manifest {key}", manifest.get(key), value)
    require_exact(
        "V3 run manifest train schedule hash",
        manifest.get("train_file_sha256"),
        build["arm"]["train_schedule_sha256"],
    )
    require_exact(
        "V3 run manifest validation hash",
        manifest.get("validation_file_sha256"),
        EXPECTED_VALIDATION_SHA256,
    )
    checkpoint = result_dir / "checkpoint_final"
    expected_fingerprint = checkpoint_fingerprint(checkpoint)
    metrics = load_json(result_dir / "metrics.json")
    require_exact(
        "V3 evaluated checkpoint fingerprint",
        metrics.get("checkpoint_fingerprint"),
        expected_fingerprint,
    )
    training_metrics = load_json(result_dir / "training_metrics.json")
    for key, value in training_metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"V3 training metric {key} is non-finite")
    log_path = result_dir / "training_log.json"
    if not log_path.exists():
        raise RuntimeError(f"missing required artifact: {log_path}")
    training_log = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(training_log, list) or not training_log:
        raise RuntimeError("V3 training log is empty or invalid")
    loss_values = [
        value
        for entry in training_log
        if isinstance(entry, dict)
        for key, value in entry.items()
        if "loss" in key.lower() and isinstance(value, (int, float))
    ]
    if not loss_values or any(not math.isfinite(float(value)) for value in loss_values):
        raise RuntimeError("V3 training log lacks a complete finite loss history")


def validate_execution_audit(results_root: Path) -> None:
    preflight = load_json(results_root / "preflight_environment.json")
    if not str(preflight.get("torch", "")).startswith("2.7.1"):
        raise RuntimeError(f"V3 preflight torch drift: {preflight.get('torch')!r}")
    require_exact("V3 preflight CUDA runtime", preflight.get("cuda_runtime"), "12.8")
    require_exact("V3 preflight BF16", preflight.get("bf16_supported"), True)
    require_exact("V3 preflight package pins", preflight.get("packages"), EXPECTED_PACKAGES)
    commands_path = results_root / "commands.json"
    if not commands_path.exists():
        raise RuntimeError(f"missing execution command audit: {commands_path}")
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    if not isinstance(commands, list):
        raise RuntimeError("commands.json must contain a list")
    completed = [
        entry
        for entry in commands
        if isinstance(entry, dict)
        and entry.get("status") == "complete"
        and entry.get("returncode") == 0
        and isinstance(entry.get("argv"), list)
    ]
    formal_train = [
        entry
        for entry in completed
        if any(str(part).endswith("train_qlora_v2.py") for part in entry["argv"])
        and "--smoke-test" not in entry["argv"]
    ]
    formal_evaluate = [
        entry
        for entry in completed
        if any(str(part).endswith("evaluate_tool_actions_v3.py") for part in entry["argv"])
        and "--resume" in entry["argv"]
        and "--limit" not in entry["argv"]
    ]
    if len(formal_train) != 1:
        raise RuntimeError(
            f"commands.json must contain exactly one completed formal V3 train; found {len(formal_train)}"
        )
    if not formal_evaluate:
        raise RuntimeError("commands.json lacks a completed, unlimited V3 evaluation")


def validate_v3_selection(
    build: dict[str, Any],
    selection_manifest: dict[str, Any],
    processed_dir: Path,
    selection_dir: Path,
) -> None:
    require_exact("V3 build protocol", build.get("protocol"), BUILD_PROTOCOL)
    require_exact("V3 frozen train/eval declaration", build.get("training_and_evaluation_protocol"), "qlora_v2_frozen")
    require_exact("V3 build seed", build.get("seed"), FROZEN_SEED)
    require_exact("V3 build model", build.get("model", {}).get("name"), FROZEN_MODEL)
    require_exact(
        "V3 build revision",
        build.get("model", {}).get("resolved_revision"),
        FROZEN_MODEL_REVISION,
    )
    require_exact(
        "V3 build source budgets",
        build.get("selection", {}).get("source_token_budgets"),
        EXPECTED_SOURCE_BUDGETS,
    )
    require_exact("V3 validation hash", build.get("hashes", {}).get("validation_jsonl"), EXPECTED_VALIDATION_SHA256)
    require_exact("V3 test hash", build.get("hashes", {}).get("test_jsonl"), EXPECTED_TEST_SHA256)
    require_exact("V3 test examples", build.get("shared_splits", {}).get("test_examples"), FROZEN_FORMAL_EXAMPLES)
    require_exact("V3 schedule rows", build.get("arm", {}).get("scheduled_microbatches"), 1088)
    require_exact(
        "V3 schedule covers every selected example",
        build.get("arm", {}).get("scheduled_unique_examples"),
        build.get("arm", {}).get("examples"),
    )
    require_exact("V3 arm source quotas", build.get("arm", {}).get("source_token_counts"), EXPECTED_SOURCE_BUDGETS)
    require_exact("V3 selection manifest protocol", selection_manifest.get("protocol"), BUILD_PROTOCOL)
    require_exact(
        "V3 selection manifest budgets",
        selection_manifest.get("source_token_budgets"),
        EXPECTED_SOURCE_BUDGETS,
    )
    manifest_path = selection_dir / f"{ARM}_manifest.json"
    require_exact("V3 selection manifest hash", build.get("hashes", {}).get("manifest_json"), sha256_file(manifest_path))
    require_exact(
        "V3 train file hash",
        build.get("arm", {}).get("train_file_sha256"),
        sha256_file(processed_dir / ARM / "train.jsonl"),
    )
    require_exact(
        "V3 schedule file hash",
        build.get("arm", {}).get("train_schedule_sha256"),
        sha256_file(processed_dir / ARM / "train_schedule.jsonl"),
    )


def paired_task_bootstrap(
    v3_items: list[dict[str, Any]],
    v2_items: list[dict[str, Any]],
    key: str,
    seed: int,
    samples: int = 2000,
) -> list[float | None]:
    pairs_by_task: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for v3, v2 in zip(v3_items, v2_items):
        require_exact("paired bootstrap example order", v3["example_id"], v2["example_id"])
        pairs_by_task[v3["task_key"]].append((bool(v3[key]), bool(v2[key])))
    tasks = sorted(pairs_by_task)
    if not tasks:
        return [None, None]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(tasks) for _ in tasks]
        pairs = [pair for task in sampled for pair in pairs_by_task[task]]
        draws.append(
            sum(int(v3_value) - int(v2_value) for v3_value, v2_value in pairs) / len(pairs)
        )
    draws.sort()

    def percentile(probability: float) -> float:
        position = (len(draws) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(draws) - 1)
        fraction = position - lower
        return draws[lower] * (1 - fraction) + draws[upper] * fraction

    return [percentile(0.025), percentile(0.975)]


def paired_group(
    v3_all: list[dict[str, Any]],
    v2_all: list[dict[str, Any]],
    group_name: str,
    seed: int,
) -> dict[str, Any]:
    v3_items = select_group(v3_all, group_name)
    v2_lookup = {item["example_id"]: item for item in select_group(v2_all, group_name)}
    if set(item["example_id"] for item in v3_items) != set(v2_lookup):
        raise RuntimeError(f"V2/V3 paired ID mismatch for group {group_name}")
    v2_items = [v2_lookup[item["example_id"]] for item in v3_items]
    metrics: dict[str, Any] = {}
    for offset, key in enumerate(("tool_name_correct", "full_call_exact")):
        v3_value = mean(v3_items, key)
        v2_value = mean(v2_items, key)
        wins = sum(bool(v3[key]) and not bool(v2[key]) for v3, v2 in zip(v3_items, v2_items))
        losses = sum(not bool(v3[key]) and bool(v2[key]) for v3, v2 in zip(v3_items, v2_items))
        metrics[key] = {
            "v3": v3_value,
            "v2_random_success": v2_value,
            "delta_v3_minus_v2": v3_value - v2_value if v3_value is not None and v2_value is not None else None,
            "v3_only_correct": wins,
            "v2_only_correct": losses,
            "paired_ties": len(v3_items) - wins - losses,
            "task_cluster_bootstrap_95_delta": paired_task_bootstrap(
                v3_items,
                v2_items,
                key,
                seed + offset,
            ),
        }
    return {"examples": len(v3_items), "metrics": metrics}


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/qlora_v3"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v3"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis_v3"))
    parser.add_argument(
        "--v2-results-root",
        type=Path,
        help="Directory whose random_success child contains the compatible V2 reference.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    v3_summaries: dict[str, dict[str, Any]] | None = None
    v3_rescored: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    result_dir = args.results_root / ARM
    test_path = args.processed_dir / "shared" / "test.jsonl"
    try:
        require_exact("physical V3 test hash", sha256_file(test_path), EXPECTED_TEST_SHA256)
        test_rows = read_jsonl(test_path)
        require_exact("physical V3 test rows", len(test_rows), FROZEN_FORMAL_EXAMPLES)
        build = load_and_validate_contract(args.processed_dir, args.selection_dir)
        selection_manifest = load_json(args.selection_dir / f"{ARM}_manifest.json")
        validate_v3_selection(build, selection_manifest, args.processed_dir, args.selection_dir)
        validate_execution_audit(args.results_root)
        validate_v3_training(result_dir, build)
        metrics = load_json(result_dir / "metrics.json")
        contract = load_json(result_dir / "metrics.contract.json")
        validate_evaluation_contract(metrics, contract, V3_EVAL_PROTOCOL, "V3")
        predictions_path = result_dir / "metrics.predictions.jsonl"
        require_exact(
            "V3 predictions hash recorded in metrics",
            metrics.get("predictions_sha256"),
            sha256_file(predictions_path),
        )
        resume_audit_path = result_dir / "metrics.resume_recovery.jsonl"
        resume_audit = read_jsonl(resume_audit_path) if resume_audit_path.exists() else []
        require_exact(
            "V3 resume recovery event count",
            metrics.get("resume_recovery_events"),
            len(resume_audit),
        )
        require_exact(
            "V3 resume recovery audit",
            metrics.get("resume_recovery_audit"),
            resume_audit,
        )
        v3_rescored = rescore_predictions(predictions_path, test_rows, "V3")
        v3_summaries = validate_stored_groups(metrics, v3_rescored, "V3")
    except Exception as exc:
        errors.append(f"V3 validation failed: {exc}")

    reference_status = "not_requested"
    reference_note = "No V2 results root was supplied."
    v2_summaries: dict[str, dict[str, Any]] | None = None
    paired: dict[str, Any] | None = None
    judgement = {
        "allowed": False,
        "label": "not_permitted",
        "reason": "A complete compatible V2 random_success reference is required.",
    }
    if args.v2_results_root is not None:
        reference_dir = args.v2_results_root / "random_success"
        if not reference_dir.exists():
            reference_status = "missing"
            reference_note = (
                f"{reference_dir} does not exist. V3 standalone metrics are valid "
                "if their audit passed, but no direction judgement is permitted."
            )
        else:
            required_reference = (
                reference_dir / "metrics.json",
                reference_dir / "metrics.contract.json",
                reference_dir / "metrics.predictions.jsonl",
                reference_dir / "run_manifest.json",
            )
            missing = [str(path) for path in required_reference if not path.exists()]
            if missing:
                reference_status = "incompatible"
                reference_note = f"V2 reference directory exists but is incomplete: {missing}"
                errors.append(reference_note)
            elif v3_rescored is not None:
                try:
                    test_rows = read_jsonl(test_path)
                    v2_metrics = load_json(reference_dir / "metrics.json")
                    v2_contract = load_json(reference_dir / "metrics.contract.json")
                    validate_evaluation_contract(v2_metrics, v2_contract, V2_EVAL_PROTOCOL, "V2 random_success")
                    require_exact(
                        "V2 random_success checkpoint fingerprint",
                        v2_metrics.get("checkpoint_fingerprint"),
                        V2_RANDOM_CHECKPOINT_FINGERPRINT,
                    )
                    v2_manifest = load_json(reference_dir / "run_manifest.json")
                    expected_v2_manifest = {
                        "protocol": "qlora_v2",
                        "model": FROZEN_MODEL,
                        "model_revision": FROZEN_MODEL_REVISION,
                        "seed": FROZEN_SEED,
                        "max_seq_len": 2048,
                        "max_steps": 68,
                        "batch_size": 1,
                        "grad_accum": 16,
                        "smoke_test": False,
                    }
                    for key, value in expected_v2_manifest.items():
                        require_exact(f"V2 run manifest {key}", v2_manifest.get(key), value)
                    require_exact(
                        "V2 validation hash",
                        v2_manifest.get("validation_file_sha256"),
                        EXPECTED_VALIDATION_SHA256,
                    )
                    require_exact(
                        "V2 random_success train schedule hash",
                        v2_manifest.get("train_file_sha256"),
                        V2_RANDOM_TRAIN_SCHEDULE_SHA256,
                    )
                    v3_manifest = load_json(result_dir / "run_manifest.json")
                    for key in ("torch", "transformers", "cuda_runtime", "gpu"):
                        require_exact(
                            f"V2/V3 training environment {key}",
                            v3_manifest.get("environment", {}).get(key),
                            v2_manifest.get("environment", {}).get(key),
                        )
                    v2_rescored = rescore_predictions(
                        reference_dir / "metrics.predictions.jsonl",
                        test_rows,
                        "V2 random_success",
                    )
                    v2_summaries = validate_stored_groups(
                        v2_metrics,
                        v2_rescored,
                        "V2 random_success",
                        V2_STORED_GROUPS,
                    )
                    paired = {
                        group: paired_group(
                            v3_rescored,
                            v2_rescored,
                            group,
                            FROZEN_SEED + index * 10,
                        )
                        for index, group in enumerate(GROUPS)
                    }
                    overall_metric = paired["overall"]["metrics"]["full_call_exact"]
                    non_recovery_metric = paired["non_recovery"]["metrics"][
                        "full_call_exact"
                    ]
                    recovery_metric = paired["recovery"]["metrics"]["full_call_exact"]
                    require_close(
                        "frozen V2 Random overall full-call baseline",
                        overall_metric["v2_random_success"],
                        V2_RANDOM_OVERALL_FULL_CALL,
                    )
                    require_exact(
                        "frozen non-recovery subset size",
                        paired["non_recovery"]["examples"],
                        NON_RECOVERY_EXAMPLES,
                    )
                    require_close(
                        "frozen V2 Random non-recovery full-call baseline",
                        non_recovery_metric["v2_random_success"],
                        V2_RANDOM_NON_RECOVERY_CORRECT / NON_RECOVERY_EXAMPLES,
                    )
                    require_exact(
                        "frozen recovery subset size",
                        paired["recovery"]["examples"],
                        RECOVERY_EXAMPLES,
                    )
                    require_close(
                        "frozen V2 Random recovery full-call baseline",
                        recovery_metric["v2_random_success"],
                        V2_RANDOM_RECOVERY_CORRECT / RECOVERY_EXAMPLES,
                    )
                    preserves_overall = overall_metric["v3"] >= OVERALL_RETENTION_FLOOR
                    preserves_non_recovery = (
                        non_recovery_metric["v3"]
                        >= NON_RECOVERY_RETENTION_FLOOR
                    )
                    improves_recovery = recovery_metric["v3"] >= RECOVERY_SIGNAL_FLOOR
                    if preserves_non_recovery and improves_recovery:
                        label = "diagnostic_directional_support"
                        reason = (
                            "V3 is within 2 percentage points of V2 Random on "
                            "the 906 non-recovery examples and gets at least 2 "
                            "more of the 53 recovery examples correct."
                        )
                    elif improves_recovery:
                        label = "recovery_gain_with_non_recovery_tradeoff"
                        reason = (
                            "V3 gets at least 2 more recovery examples correct, "
                            "but non-recovery EM drops by more than 2 points."
                        )
                    elif preserves_non_recovery:
                        label = "non_recovery_preserved_without_recovery_gain"
                        reason = (
                            "Non-recovery EM is preserved, but the predeclared "
                            "recovery gain is absent."
                        )
                    else:
                        label = "no_directional_support"
                        reason = "Neither predeclared diagnostic condition is satisfied."
                    judgement = {
                        "allowed": True,
                        "label": label,
                        "reason": reason,
                        "predeclared_diagnostic_conditions": {
                            "overall_full_call_absolute_min": OVERALL_RETENTION_FLOOR,
                            "overall_full_call_delta_min": -0.02,
                            "non_recovery_full_call_absolute_min": NON_RECOVERY_RETENTION_FLOOR,
                            "non_recovery_full_call_delta_min": -0.02,
                            "non_recovery_correct_baseline": V2_RANDOM_NON_RECOVERY_CORRECT,
                            "non_recovery_examples": NON_RECOVERY_EXAMPLES,
                            "recovery_full_call_absolute_min": RECOVERY_SIGNAL_FLOOR,
                            "recovery_correct_min": V2_RANDOM_RECOVERY_CORRECT + 2,
                            "recovery_examples": RECOVERY_EXAMPLES,
                        },
                        "preserves_overall": preserves_overall,
                        "preserves_non_recovery": preserves_non_recovery,
                        "improves_recovery": improves_recovery,
                        "claim_boundary": (
                            "exploratory diagnostic on the already inspected V2 "
                            "test set; not confirmatory or paper-final evidence"
                        ),
                    }
                    reference_status = "compatible"
                    reference_note = "Compatible V2 random_success predictions were rescored and paired by example ID."
                except Exception as exc:
                    reference_status = "incompatible"
                    reference_note = f"V2 reference validation failed: {exc}"
                    errors.append(reference_note)
            else:
                reference_status = "not_evaluated_due_to_v3_failure"
                reference_note = (
                    "The V2 directory is present, but paired validation was "
                    "not attempted because the V3 audit failed."
                )

    payload = {
        "protocol": V3_EVAL_PROTOCOL,
        "valid": not errors and v3_summaries is not None,
        "claim_boundary": (
            "offline next-tool-call imitation on an already inspected test set; "
            "not executable Agent success and not confirmatory paper evidence"
        ),
        "errors": errors,
        "v3": {
            "arm": ARM,
            "groups": v3_summaries,
            "metrics_path": str(result_dir / "metrics.json"),
        },
        "v2_reference": {
            "status": reference_status,
            "note": reference_note,
            "root": str(args.v2_results_root) if args.v2_results_root is not None else None,
            "groups": v2_summaries,
        },
        "paired_comparison": paired,
        "direction_judgement": judgement,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    csv_rows = []
    if v3_summaries:
        for group in GROUPS:
            v3_group = v3_summaries[group]
            row = {
                "group": group,
                "examples": v3_group["examples"],
                "v3_tool_accuracy": v3_group["micro"]["tool_name_correct"],
                "v3_full_call_exact": v3_group["micro"]["full_call_exact"],
                "v2_tool_accuracy": None,
                "v2_full_call_exact": None,
                "delta_tool_accuracy": None,
                "delta_full_call_exact": None,
            }
            if paired:
                row.update({
                    "v2_tool_accuracy": paired[group]["metrics"]["tool_name_correct"]["v2_random_success"],
                    "v2_full_call_exact": paired[group]["metrics"]["full_call_exact"]["v2_random_success"],
                    "delta_tool_accuracy": paired[group]["metrics"]["tool_name_correct"]["delta_v3_minus_v2"],
                    "delta_full_call_exact": paired[group]["metrics"]["full_call_exact"]["delta_v3_minus_v2"],
                })
            csv_rows.append(row)
    csv_fields = [
        "group",
        "examples",
        "v3_tool_accuracy",
        "v3_full_call_exact",
        "v2_tool_accuracy",
        "v2_full_call_exact",
        "delta_tool_accuracy",
        "delta_full_call_exact",
    ]
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# QLoRA V3 constrained-recovery diagnostic",
        "",
        "Offline held-out next-tool-call imitation on an already inspected V2 test set; "
        "this is neither executable Agent success nor confirmatory paper evidence.",
        "",
        f"- V3 audit: **{'PASS' if payload['valid'] else 'FAIL'}**",
        f"- V2 random reference: **{reference_status}**",
        f"- Direction label: **{judgement['label']}**",
        f"- Interpretation: {judgement['reason']}",
        "",
        "| group | n | V3 tool acc. | V3 full-call EM | V2 random tool acc. | V2 random full-call EM | Δ tool | Δ full-call |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['group']} | {row['examples']} | {pct(row['v3_tool_accuracy'])} | "
            f"{pct(row['v3_full_call_exact'])} | {pct(row['v2_tool_accuracy'])} | "
            f"{pct(row['v2_full_call_exact'])} | {pct(row['delta_tool_accuracy'])} | "
            f"{pct(row['delta_full_call_exact'])} |"
        )
    lines.extend(["", f"Reference note: {reference_note}"])
    if errors:
        lines.extend(["", "## Blocking audit errors", "", *[f"- {error}" for error in errors]])
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
    if errors or v3_summaries is None:
        raise SystemExit("V3 aggregation blocked by incompatible or missing V3 artifacts")


if __name__ == "__main__":
    main()
