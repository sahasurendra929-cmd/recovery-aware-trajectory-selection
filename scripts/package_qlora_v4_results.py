#!/usr/bin/env python3
"""Fail-closed packaging for the frozen four-arm QLoRA V4 experiment.

The package is deliberately the last independent release gate.  It revalidates
all three newly trained checkpoints, their generation evaluations and strict
pair-scoring passes; pins the existing Standard V3 reference by Git commit and
artifact hashes; reruns the deterministic aggregate; and byte-compares every
published analysis file before copying anything into the upload directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aggregate_qlora_v4 as aggregate_v4
import prepare_qlora_v4 as prepare_v4
import run_qlora_v4 as run_v4
import train_preference_v4 as preference_v4
import train_qlora_v4_sft as clean_sft_v4


ROOT = Path(__file__).resolve().parents[1]
FROZEN_TAG = run_v4.FROZEN_TAG
TRAINED_ARMS = ("clean_sft", "continued_sft", "dpo")
ARM_ORDER = aggregate_v4.ARM_ORDER
CONTRASTS = aggregate_v4.CONTRASTS
CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors")
EVALUATION_FILES = (
    "metrics.json",
    "metrics.contract.json",
    "metrics.predictions.jsonl",
)
FORMAL_EXAMPLES = aggregate_v4.FROZEN_FORMAL_EXAMPLES
STRICT_PAIR_EXAMPLES = aggregate_v4.EXPECTED_TEST_PREFERENCE_PAIRS
TEST_SHA256 = aggregate_v4.EXPECTED_TEST_SHA256
PAIR_SHA256 = aggregate_v4.EXPECTED_TEST_PREFERENCE_PAIRS_SHA256
ANALYSIS_PROTOCOL = aggregate_v4.PROTOCOL
EVALUATION_PROTOCOL = aggregate_v4.V4_EVALUATION_PROTOCOL
SCORE_PROTOCOL = f"{preference_v4.PROTOCOL}_pair_scoring"

DATA_HASH_PATHS = {
    "clean_train_unique_sha256": Path("clean_sft/train_unique.jsonl"),
    "clean_train_schedule_sha256": Path("clean_sft/train_schedule.jsonl"),
    "clean_validation_sha256": Path("clean_sft/validation.jsonl"),
    "replacement_map_sha256": Path("clean_sft/replacement_map.jsonl"),
    "preference_train_pairs_sha256": Path("preference/train_pairs.jsonl"),
    "preference_train_schedule_sha256": Path("preference/train_schedule.jsonl"),
    "preference_validation_pairs_sha256": Path(
        "preference/validation_pairs.jsonl"
    ),
    "preference_smoke_pairs_sha256": Path("preference/smoke_pairs.jsonl"),
    "test_outcomes_sha256": Path("evaluation/test_outcomes.jsonl"),
    "test_preference_pairs_sha256": Path(
        "evaluation/test_preference_pairs.jsonl"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_exact(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label}: {observed!r} != {expected!r}")


def validate_static_constants() -> None:
    for label, value in (
        ("aggregate", aggregate_v4.FROZEN_TAG),
        ("Clean-SFT trainer", clean_sft_v4.FROZEN_TAG),
        ("preference trainer", preference_v4.FROZEN_TAG),
    ):
        require_exact(f"{label} frozen tag", value, FROZEN_TAG)
    require_exact(
        "Clean-SFT schedule hash constant",
        clean_sft_v4.EXPECTED_TRAIN_SHA256,
        prepare_v4.EXPECTED_OUTPUT_SHA256["clean_train_schedule_sha256"],
    )
    require_exact(
        "preference schedule hash constant",
        run_v4.PREFERENCE_SCHEDULE_SHA256,
        prepare_v4.EXPECTED_OUTPUT_SHA256[
            "preference_train_schedule_sha256"
        ],
    )
    require_exact(
        "strict test-pair hash constant",
        PAIR_SHA256,
        prepare_v4.EXPECTED_OUTPUT_SHA256[
            "test_preference_pairs_sha256"
        ],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reject_nonfinite(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value in {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{location}[{index}]")


def load_json_value(path: Path) -> Any:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"missing or unsafe JSON artifact: {path}",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    reject_nonfinite(value, str(path))
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = load_json_value(path)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_json_list(path: Path) -> list[dict[str, Any]]:
    value = load_json_value(path)
    require(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, dict) for row in value),
        f"expected a nonempty JSON object list: {path}",
    )
    return value


def load_jsonl(path: Path, expected: int | None = None) -> list[dict[str, Any]]:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"missing or unsafe JSONL artifact: {path}",
    )
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"invalid UTF-8 JSONL artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        require(bool(line.strip()), f"blank line in formal JSONL {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        require(isinstance(row, dict), f"non-object JSONL row {path}:{line_number}")
        reject_nonfinite(row, f"{path}:{line_number}")
        rows.append(row)
    if expected is not None:
        require_exact(f"{path} row count", len(rows), expected)
    return rows


def git_output(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def discovered_v4_source_paths() -> tuple[Path, ...]:
    """Return every release-relevant V4 source/config/document in the checkout."""
    paths = {
        ROOT / "README.md",
        ROOT / "BASELINE_V4_HANDOFF.md",
        ROOT / "V4_RTX5060_AGENT_PROMPT.md",
        ROOT / "requirements-gpu-v4.txt",
        ROOT / "configs" / "qlora_v4.yaml",
    }
    for pattern in ("*v4*.py", "*V4*.py"):
        paths.update((ROOT / "scripts").glob(pattern))
        paths.update((ROOT / "tests").glob(pattern))
    for pattern in ("*v4*", "*V4*"):
        paths.update((ROOT / "configs").glob(pattern))
    for pattern in ("*v4*.md", "*V4*.md", "*v4*.txt", "*V4*.txt"):
        paths.update(ROOT.glob(pattern))
    return tuple(sorted(paths))


def source_commit() -> tuple[str, dict[str, str]]:
    head = git_output("rev-parse", "HEAD^{commit}").decode("ascii").strip()
    tagged = (
        git_output("rev-parse", f"refs/tags/{FROZEN_TAG}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    require(head == tagged, f"HEAD {head} differs from {FROZEN_TAG}={tagged}")
    dirty = (
        git_output("status", "--porcelain", "--untracked-files=no")
        .decode("utf-8")
        .strip()
    )
    require(not dirty, "tracked source/config differs from the frozen tag")

    source_hashes: dict[str, str] = {}
    for path in discovered_v4_source_paths():
        require(
            path.is_file() and not path.is_symlink(),
            f"required V4 source/config/document is missing or unsafe: {path}",
        )
        relative = path.relative_to(ROOT).as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        require(
            tracked.returncode == 0,
            f"V4 source/config/document is not tracked by Git: {relative}",
        )
        # Hash the canonical Git blob.  A clean Windows checkout may represent
        # the same tracked text with CRLF, so comparing raw working-tree bytes
        # would incorrectly reject an otherwise identical frozen checkout.
        frozen_payload = git_output("show", f"{FROZEN_TAG}:{relative}")
        source_hashes[relative] = sha256_bytes(frozen_payload)
    require(bool(source_hashes), "no tracked V4 release sources were discovered")
    return head, source_hashes


def checkpoint_fingerprint(path: Path) -> str:
    """Return the shared V2/V3/V4 adapter fingerprint."""
    digest = hashlib.sha256()
    for filename in CHECKPOINT_FILES:
        target = path / filename
        require(
            target.is_file() and not target.is_symlink() and target.stat().st_size > 0,
            f"incomplete checkpoint file: {target}",
        )
        file_hash = sha256_file(target)
        # This is the established V2/V3 evaluator convention.  Do not add
        # separators without versioning every producer and consumer together.
        digest.update(filename.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> tuple[str, dict[str, str]]:
    fingerprint = checkpoint_fingerprint(path)
    hashes = {
        filename: sha256_file(path / filename) for filename in CHECKPOINT_FILES
    }
    require(
        (path / "adapter_model.safetensors").stat().st_size > 16,
        f"adapter weights are implausibly small: {path}",
    )
    adapter_config = load_json(path / "adapter_config.json")
    require(bool(adapter_config), f"empty adapter configuration: {path}")
    return fingerprint, hashes


def validate_source_identity(
    document: dict[str, Any],
    commit: str,
    label: str,
) -> None:
    require_exact(f"{label} source tag", document.get("source_tag"), FROZEN_TAG)
    require_exact(f"{label} source commit", document.get("source_commit"), commit)


def validate_training_log(
    log: list[dict[str, Any]],
    expected_steps: int,
    label: str,
) -> None:
    steps = [
        row["step"]
        for row in log
        if isinstance(row.get("step"), int) and not isinstance(row.get("step"), bool)
    ]
    require(steps, f"{label} training log contains no integer step")
    require_exact(f"{label} final logged step", max(steps), expected_steps)
    numeric_loss = False
    numeric_grad = False
    for row in log:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                require(math.isfinite(float(value)), f"non-finite {label} {key}")
                numeric_loss = numeric_loss or "loss" in key.lower()
                numeric_grad = numeric_grad or key == "grad_norm"
    require(numeric_loss, f"{label} training log contains no numeric loss")
    require(numeric_grad, f"{label} training log contains no numeric grad_norm")


def validate_runtime_environment(
    environment: Any,
    label: str,
    *,
    full_library_set: bool,
) -> None:
    require(isinstance(environment, dict), f"{label} environment audit is missing")
    require(
        isinstance(environment.get("python"), str)
        and environment["python"].startswith("3.11"),
        f"{label} did not use frozen Python 3.11",
    )
    require(
        isinstance(environment.get("torch"), str)
        and environment["torch"].startswith("2.7.1"),
        f"{label} did not use frozen torch 2.7.1",
    )
    expected = {
        "transformers": "4.52.4",
        "cuda_runtime": "12.8",
        "deterministic_algorithms_enabled": True,
        "cublas_workspace_config": ":4096:8",
    }
    if full_library_set:
        expected.update(
            {
                "peft": "0.15.2",
                "bitsandbytes": "0.46.0",
                "datasets": "3.6.0",
                "accelerate": "1.7.0",
                "trl": "0.18.2",
                "bf16_supported": True,
            }
        )
    else:
        expected["bf16"] = True
    for key, value in expected.items():
        require_exact(f"{label} environment {key}", environment.get(key), value)


def require_checkpoint_record(
    record: Any,
    fingerprint: str,
    file_hashes: dict[str, str],
    label: str,
) -> None:
    require(isinstance(record, dict), f"{label} lacks a checkpoint record")
    require_exact(
        f"{label} checkpoint fingerprint",
        record.get("checkpoint_fingerprint"),
        fingerprint,
    )
    require_exact(
        f"{label} checkpoint file hashes",
        record.get("file_sha256"),
        file_hashes,
    )


def validate_training(
    result_dir: Path,
    arm: str,
    checkpoint_fp: str,
    checkpoint_hashes: dict[str, str],
    clean_fp: str,
    clean_hashes: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    manifest_path = result_dir / "run_manifest.json"
    metrics_path = result_dir / "training_metrics.json"
    log_path = result_dir / "training_log.json"
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    log = load_json_list(log_path)
    validate_source_identity(manifest, commit, f"{arm} training")
    require_exact(f"{arm} manifest arm", manifest.get("arm"), arm)
    require_exact(f"{arm} model", manifest.get("model"), preference_v4.MODEL)
    require_exact(
        f"{arm} model revision",
        manifest.get("model_revision"),
        preference_v4.MODEL_REVISION,
    )
    require_checkpoint_record(
        manifest.get("output_checkpoint"),
        checkpoint_fp,
        checkpoint_hashes,
        f"{arm} output",
    )

    if arm == "clean_sft":
        expected_top_level = {
            "protocol": "qlora_v4_clean_sft",
            "objective": "matched_clean_completion_only_cross_entropy",
            "failed_action_labels": 0,
            "seed": clean_sft_v4.SEED,
            "max_seq_len": clean_sft_v4.MAX_SEQUENCE_TOKENS,
            "max_steps": clean_sft_v4.EXPECTED_STEPS,
            "batch_size": clean_sft_v4.FORMAL_BATCH_SIZE,
            "grad_accum": clean_sft_v4.FORMAL_GRAD_ACCUM,
            "learning_rate": clean_sft_v4.FORMAL_LEARNING_RATE,
            "lora_dropout": 0.05,
            "formal_schedule_rows": clean_sft_v4.EXPECTED_ROWS,
            "train_file_sha256": clean_sft_v4.EXPECTED_TRAIN_SHA256,
            "validation_file_sha256": clean_sft_v4.EXPECTED_VALIDATION_SHA256,
            "smoke_test": False,
            "held_out_test_accessed": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        }
        for key, expected in expected_top_level.items():
            require_exact(f"clean_sft manifest {key}", manifest.get(key), expected)
        loss_audit = manifest.get("loss_audit")
        require(
            isinstance(loss_audit, dict)
            and loss_audit.get("finite") is True
            and loss_audit.get("loss_values_checked", 0) > 0
            and loss_audit.get("grad_norm_values_checked", 0) > 0
            and loss_audit.get("validation_loss_values_checked", 0) > 0,
            "clean_sft loss audit is incomplete",
        )
        require(
            isinstance(metrics.get("train_loss"), (int, float))
            and not isinstance(metrics.get("train_loss"), bool),
            "clean_sft training metrics lacks train_loss",
        )
        validate_runtime_environment(
            manifest.get("environment"),
            "clean_sft",
            full_library_set=False,
        )
        validate_training_log(log, clean_sft_v4.EXPECTED_STEPS, arm)
    else:
        expected_objective = {
            "continued_sft": "chosen_completion_only_causal_cross_entropy",
            "dpo": "sigmoid_direct_preference_optimization",
        }[arm]
        require_exact(
            f"{arm} protocol",
            manifest.get("protocol"),
            preference_v4.PROTOCOL,
        )
        require_exact(f"{arm} formal flag", manifest.get("formal_result"), True)
        require_exact(f"{arm} objective", manifest.get("objective"), expected_objective)
        require_checkpoint_record(
            manifest.get("clean_sft_initialization"),
            clean_fp,
            clean_hashes,
            f"{arm} Clean-SFT initialization",
        )
        data = manifest.get("data")
        require(isinstance(data, dict), f"{arm} data contract is missing")
        expected_data = {
            "train_schedule_sha256": run_v4.PREFERENCE_SCHEDULE_SHA256,
            "expected_train_schedule_sha256": run_v4.PREFERENCE_SCHEDULE_SHA256,
            "train_schedule_hash_matches_expected": True,
            "held_out_test_accessed": False,
        }
        for key, expected in expected_data.items():
            require_exact(f"{arm} data {key}", data.get(key), expected)
        split_audit = data.get("split_audit") or {}
        require_exact(
            f"{arm} data expected split",
            split_audit.get("expected_split"),
            "train",
        )
        require_exact(
            f"{arm} data split audit",
            split_audit.get("all_rows_match_expected_split"),
            True,
        )
        compute = manifest.get("compute_contract")
        require(isinstance(compute, dict), f"{arm} compute contract is missing")
        expected_compute = {
            "seed": preference_v4.SEED,
            "max_prompt_tokens": preference_v4.MAX_PROMPT_TOKENS,
            "max_completion_tokens": preference_v4.MAX_COMPLETION_TOKENS,
            "max_sequence_tokens": preference_v4.MAX_SEQUENCE_TOKENS,
            "optimizer_steps": preference_v4.FORMAL_MAX_STEPS,
            "microbatch_size": preference_v4.FORMAL_BATCH_SIZE,
            "gradient_accumulation_steps": preference_v4.FORMAL_GRAD_ACCUM,
            "scheduled_microbatches": preference_v4.FORMAL_SCHEDULE_ROWS,
            "chosen_exposures": preference_v4.FORMAL_SCHEDULE_ROWS,
            "learning_rate": preference_v4.FORMAL_LEARNING_RATE,
            "lr_scheduler_type": preference_v4.LR_SCHEDULER,
            "warmup_steps": preference_v4.WARMUP_STEPS,
            "optimizer": "paged_adamw_8bit",
            "optimizer_state_reused_from_clean_sft": False,
            "scheduler_state_reused_from_clean_sft": False,
            "sampler": "SequentialSampler",
            "row_order_shuffled": False,
            "bf16": True,
            "quantization": "NF4 double-quantized base",
            "gradient_checkpointing": True,
            "dropout": 0.0,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        }
        for key, expected in expected_compute.items():
            require_exact(f"{arm} compute {key}", compute.get(key), expected)
        require(
            isinstance(manifest.get("loss_audit"), dict)
            and manifest["loss_audit"].get("finite") is True,
            f"{arm} finite loss audit is missing",
        )
        require_exact(
            f"{arm} policy state changed",
            (manifest.get("parameter_audit") or {}).get("policy_state_changed"),
            True,
        )
        require_exact(f"{arm} metrics arm", metrics.get("arm"), arm)
        require_exact(
            f"{arm} metrics global step",
            metrics.get("global_step"),
            preference_v4.FORMAL_MAX_STEPS,
        )
        require_exact(f"{arm} metrics finite loss", metrics.get("finite_loss"), True)
        validate_runtime_environment(
            manifest.get("environment"),
            arm,
            full_library_set=True,
        )
        require_exact(
            f"{arm} frozen library contract",
            manifest.get("library_contract"),
            {
                "transformers": "4.52.4",
                "peft": "0.15.2",
                "trl": "0.18.2",
            },
        )
        validate_training_log(log, preference_v4.FORMAL_MAX_STEPS, arm)
        dpo_audit = manifest.get("dpo_reference_audit")
        if arm == "continued_sft":
            require_exact("continued_sft DPO audit", dpo_audit, None)
        else:
            require(isinstance(dpo_audit, dict), "dpo reference audit is missing")
            expected_dpo = {
                "policy_and_reference_identical_at_initialization": True,
                "reference_state_unchanged": True,
                "reference_frozen_in_every_context": True,
                "reference_tensors_in_optimizer": 0,
                "reference_requires_grad_tensors_after_training": 0,
                "policy_state_changed": True,
                "beta": preference_v4.DPO_BETA,
                "loss_type": preference_v4.DPO_LOSS_TYPE,
                "label_smoothing": preference_v4.DPO_LABEL_SMOOTHING,
                "reference_free": False,
                "use_logits_to_keep": True,
            }
            for key, expected in expected_dpo.items():
                require_exact(f"dpo audit {key}", dpo_audit.get(key), expected)
            require(
                isinstance(dpo_audit.get("reference_context_entries"), int)
                and dpo_audit["reference_context_entries"] > 0,
                "dpo never used the frozen reference context",
            )
            initial = dpo_audit.get("initial_numerical_audit") or {}
            require_exact(
                "dpo initial policy/reference match",
                initial.get("policy_reference_logps_match"),
                True,
            )
            require_exact(
                "dpo initial loss audit",
                initial.get("initial_dpo_loss_within_tolerance"),
                True,
            )

    return {
        "paths": (manifest_path, metrics_path, log_path),
        "manifest": manifest,
        "run_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_fingerprint": checkpoint_fp,
        "checkpoint_file_sha256": checkpoint_hashes,
    }


def validate_eval(
    result_dir: Path,
    checkpoint_fp: str,
) -> dict[str, Any]:
    metrics_path = result_dir / "metrics.json"
    contract_path = result_dir / "metrics.contract.json"
    predictions_path = result_dir / "metrics.predictions.jsonl"
    metrics = load_json(metrics_path)
    contract = load_json(contract_path)
    predictions = load_jsonl(predictions_path, FORMAL_EXAMPLES)
    require_exact("evaluation metrics protocol", metrics.get("protocol"), EVALUATION_PROTOCOL)
    require_exact("evaluation contract protocol", contract.get("protocol"), EVALUATION_PROTOCOL)
    require_exact(
        "evaluation family",
        contract.get("training_and_evaluation_protocol"),
        "qlora_v2_frozen",
    )
    for key, expected in {
        **aggregate_v4.CORE_EVAL_EXPECTED,
        "test_file_sha256": TEST_SHA256,
        "formal_test_examples": FORMAL_EXAMPLES,
        "evaluated_examples": FORMAL_EXAMPLES,
    }.items():
        require_exact(f"evaluation metrics {key}", metrics.get(key), expected)
        require_exact(f"evaluation contract {key}", contract.get(key), expected)
    for key in aggregate_v4.CONTRACT_KEYS:
        require_exact(
            f"evaluation metrics/contract {key}",
            metrics.get(key),
            contract.get(key),
        )
    require_exact(
        "evaluation metrics checkpoint fingerprint",
        metrics.get("checkpoint_fingerprint"),
        checkpoint_fp,
    )
    require_exact(
        "evaluation contract checkpoint fingerprint",
        contract.get("checkpoint_fingerprint"),
        checkpoint_fp,
    )
    prediction_hash = sha256_file(predictions_path)
    require_exact(
        "evaluation prediction hash",
        metrics.get("predictions_sha256"),
        prediction_hash,
    )
    ids = [row.get("example_id") for row in predictions]
    require(
        len(set(ids)) == FORMAL_EXAMPLES
        and all(isinstance(value, str) and value for value in ids),
        "evaluation prediction IDs are incomplete or duplicated",
    )
    return {
        "paths": (metrics_path, contract_path, predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "contract_sha256": sha256_file(contract_path),
        "predictions_sha256": prediction_hash,
        "checkpoint_fingerprint": checkpoint_fp,
    }


def validate_score(
    score_dir: Path,
    checkpoint_fp: str,
    checkpoint_hashes: dict[str, str],
    commit: str,
) -> dict[str, Any]:
    metrics_path = score_dir / "metrics.json"
    scores_path = score_dir / "pair_scores.jsonl"
    manifest_path = score_dir / "score_manifest.json"
    metrics = load_json(metrics_path)
    rows = load_jsonl(scores_path, STRICT_PAIR_EXAMPLES)
    manifest = load_json(manifest_path)
    validate_source_identity(manifest, commit, f"{score_dir.name} pair score")
    expected_manifest = {
        "protocol": SCORE_PROTOCOL,
        "mode": "score",
        "split": "test",
        "training_performed": False,
        "limited": False,
        "complete": True,
        "expected_pairs": STRICT_PAIR_EXAMPLES,
        "completed_pairs": STRICT_PAIR_EXAMPLES,
        "pair_count": STRICT_PAIR_EXAMPLES,
        "preference_pairs_sha256": PAIR_SHA256,
        "model": preference_v4.MODEL,
        "model_revision": preference_v4.MODEL_REVISION,
    }
    for key, expected in expected_manifest.items():
        require_exact(f"pair-score manifest {key}", manifest.get(key), expected)
    require_checkpoint_record(
        manifest.get("adapter"),
        checkpoint_fp,
        checkpoint_hashes,
        f"{score_dir.name} pair-score adapter",
    )
    require_exact(
        "pair-score top-level checkpoint fingerprint",
        manifest.get("checkpoint_fingerprint"),
        checkpoint_fp,
    )
    data = manifest.get("data") or {}
    for key, expected in {
        "strict_pair_file_sha256": PAIR_SHA256,
        "expected_strict_pair_file_sha256": PAIR_SHA256,
        "strict_pair_file_hash_matches_expected": True,
    }.items():
        require_exact(f"pair-score data {key}", data.get(key), expected)
    require_exact(
        "pair-score split audit",
        (data.get("split_audit") or {}).get("all_rows_match_expected_split"),
        True,
    )
    scoring = manifest.get("scoring_contract") or {}
    for key, expected in {
        "completion_eos_included": True,
        "runtime_truncation_count": 0,
        "dropout": 0.0,
        "quantization": "NF4 double-quantized base",
        "bf16": True,
    }.items():
        require_exact(f"pair-score contract {key}", scoring.get(key), expected)
    require_exact("pair-score metrics protocol", metrics.get("protocol"), SCORE_PROTOCOL)
    require_exact("pair-score metrics split", metrics.get("split"), "test")
    require_exact(
        "pair-score metrics pair count",
        metrics.get("pair_count"),
        STRICT_PAIR_EXAMPLES,
    )
    require_exact(
        "pair-score metrics EOS contract",
        metrics.get("completion_eos_included"),
        True,
    )
    validate_runtime_environment(
        manifest.get("environment"),
        f"{score_dir.name} pair score",
        full_library_set=True,
    )
    score_hash = sha256_file(scores_path)
    metrics_hash = sha256_file(metrics_path)
    require_exact(
        "pair-score manifest score hash",
        manifest.get("pair_scores_sha256"),
        score_hash,
    )
    outputs = manifest.get("outputs") or {}
    require_exact("pair-score output metrics hash", outputs.get("metrics_sha256"), metrics_hash)
    require_exact("pair-score output score hash", outputs.get("pair_scores_sha256"), score_hash)
    require_exact(
        "pair-score output row count",
        outputs.get("pair_scores_rows"),
        STRICT_PAIR_EXAMPLES,
    )
    pair_ids = [row.get("pair_id") for row in rows]
    require(
        len(set(pair_ids)) == STRICT_PAIR_EXAMPLES
        and all(isinstance(value, str) and value for value in pair_ids),
        "pair-score rows have incomplete or duplicate pair IDs",
    )
    return {
        "paths": (metrics_path, scores_path, manifest_path),
        "metrics_sha256": metrics_hash,
        "pair_scores_sha256": score_hash,
        "score_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_fingerprint": checkpoint_fp,
    }


def validate_data_audit(processed_dir: Path) -> dict[str, Any]:
    summary_path = processed_dir / "build_summary.json"
    audit_path = processed_dir / "contract_audit.json"
    summary = load_json(summary_path)
    audit = load_json(audit_path)
    require_exact("V4 build protocol", summary.get("protocol"), prepare_v4.PROTOCOL)
    require_exact("V4 build seed", summary.get("seed"), prepare_v4.SEED)
    require_exact("V4 build counts", summary.get("counts"), prepare_v4.EXPECTED_COUNTS)
    require_exact(
        "V4 build model",
        summary.get("model"),
        {"name": prepare_v4.MODEL, "revision": prepare_v4.MODEL_REVISION},
    )
    require_exact("V4 audit protocol", audit.get("protocol"), prepare_v4.PROTOCOL)
    require_exact("V4 audit valid", audit.get("valid"), True)
    require_exact("V4 audit errors", audit.get("errors"), [])
    require_exact("V4 audit counts", audit.get("counts"), prepare_v4.EXPECTED_COUNTS)
    require_exact(
        "V4 audit frozen hashes",
        audit.get("hashes"),
        prepare_v4.EXPECTED_OUTPUT_SHA256,
    )
    checks = audit.get("checks")
    require(isinstance(checks, dict) and bool(checks), "V4 data audit has no checks")
    failures: dict[str, Any] = {}
    for key, value in checks.items():
        if isinstance(value, bool):
            passed = value is True
        elif isinstance(value, (int, float)):
            passed = not isinstance(value, bool) and value == 0
        else:
            passed = False
        if not passed:
            failures[key] = value
    require(not failures, f"V4 data audit contains failed checks: {failures}")
    require_exact(
        "V4 package data-hash key set",
        set(DATA_HASH_PATHS),
        set(prepare_v4.EXPECTED_OUTPUT_SHA256),
    )
    for key, relative in DATA_HASH_PATHS.items():
        require_exact(
            f"V4 processed artifact {key}",
            sha256_file(processed_dir / relative),
            prepare_v4.EXPECTED_OUTPUT_SHA256[key],
        )
    return {
        "paths": (summary_path, audit_path),
        "build_summary_sha256": sha256_file(summary_path),
        "contract_audit_sha256": sha256_file(audit_path),
    }


def validate_standard_v3(result_dir: Path) -> dict[str, Any]:
    require_exact(
        "runner/aggregator Standard V3 fingerprint constant",
        run_v4.V3_CHECKPOINT_FINGERPRINT,
        aggregate_v4.STANDARD_V3_CHECKPOINT_FINGERPRINT,
    )
    expected_hashes = {
        "metrics.json": aggregate_v4.STANDARD_V3_METRICS_SHA256,
        "metrics.contract.json": aggregate_v4.STANDARD_V3_CONTRACT_SHA256,
        "metrics.predictions.jsonl": aggregate_v4.STANDARD_V3_PREDICTIONS_SHA256,
    }
    require_exact(
        "runner/aggregator Standard V3 artifact hashes",
        run_v4.V3_RESULT_SHA256,
        expected_hashes,
    )
    git_output("cat-file", "-e", f"{run_v4.V3_RESULT_COMMIT}^{{commit}}")
    paths: list[Path] = []
    for filename, expected_hash in expected_hashes.items():
        path = result_dir / filename
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f"missing or unsafe Standard V3 artifact: {path}",
        )
        require_exact(
            f"Standard V3 local {filename}",
            sha256_file(path),
            expected_hash,
        )
        frozen_payload = git_output(
            "show",
            f"{run_v4.V3_RESULT_COMMIT}:{run_v4.V3_ARTIFACT_PREFIX}/{filename}",
        )
        require_exact(
            f"Standard V3 pinned-commit {filename}",
            sha256_bytes(frozen_payload),
            expected_hash,
        )
        require(
            path.read_bytes() == frozen_payload,
            f"Standard V3 local file differs from pinned commit: {filename}",
        )
        paths.append(path)
    metrics = load_json(result_dir / "metrics.json")
    contract = load_json(result_dir / "metrics.contract.json")
    require_exact("Standard V3 metrics protocol", metrics.get("protocol"), "qlora_v3")
    require_exact("Standard V3 contract protocol", contract.get("protocol"), "qlora_v3")
    require_exact(
        "Standard V3 checkpoint fingerprint",
        contract.get("checkpoint_fingerprint"),
        run_v4.V3_CHECKPOINT_FINGERPRINT,
    )
    require_exact(
        "Standard V3 metrics fingerprint",
        metrics.get("checkpoint_fingerprint"),
        run_v4.V3_CHECKPOINT_FINGERPRINT,
    )
    require_exact("Standard V3 test hash", contract.get("test_file_sha256"), TEST_SHA256)
    require_exact(
        "Standard V3 formal examples",
        contract.get("formal_test_examples"),
        FORMAL_EXAMPLES,
    )
    require_exact(
        "Standard V3 evaluated examples",
        contract.get("evaluated_examples"),
        FORMAL_EXAMPLES,
    )
    require_exact("Standard V3 limited flag", contract.get("limited"), False)
    return {
        "paths": tuple(paths),
        "file_sha256": expected_hashes,
        "checkpoint_fingerprint": run_v4.V3_CHECKPOINT_FINGERPRINT,
        "source_branch": run_v4.V3_RESULT_BRANCH,
        "source_commit": run_v4.V3_RESULT_COMMIT,
        "source_artifact_prefix": run_v4.V3_ARTIFACT_PREFIX,
    }


def validate_commands(path: Path, commit: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    required_stages = {
        "train-clean-sft",
        "train-continued_sft",
        "train-dpo",
        "evaluate-clean_sft",
        "evaluate-continued_sft",
        "evaluate-dpo",
        "score-clean_sft",
        "score-continued_sft",
        "score-dpo",
        "aggregate-v4",
    }
    completed = {
        row.get("stage")
        for row in rows
        if type(row.get("returncode")) is int
        and row.get("returncode") == 0
        and isinstance(row.get("finished_utc"), str)
        and bool(row.get("finished_utc"))
        and row.get("source_commit") == commit
    }
    missing = required_stages - completed
    require(not missing, f"command audit lacks successful frozen stages: {sorted(missing)}")
    return rows


def require_report_binding(
    comparison: dict[str, Any],
    training: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    standard: dict[str, Any],
    data_audit: dict[str, Any],
) -> None:
    require_exact("comparison valid", comparison.get("valid"), True)
    require_exact("comparison protocol", comparison.get("protocol"), ANALYSIS_PROTOCOL)
    require_exact("comparison examples", comparison.get("examples"), FORMAL_EXAMPLES)
    require_exact("comparison test hash", comparison.get("frozen_test_sha256"), TEST_SHA256)
    require_exact(
        "comparison arm order",
        tuple((comparison.get("arms") or {}).keys()),
        ARM_ORDER,
    )
    require_exact(
        "comparison contrast order",
        tuple((comparison.get("paired_comparisons") or {}).keys()),
        tuple(CONTRASTS),
    )
    require_exact(
        "comparison bootstrap samples",
        (comparison.get("bootstrap") or {}).get("samples"),
        aggregate_v4.BOOTSTRAP_SAMPLES,
    )
    require_exact(
        "comparison sign-flip samples",
        (comparison.get("cluster_sign_flip") or {}).get("samples"),
        aggregate_v4.SIGN_FLIP_SAMPLES,
    )
    require_exact(
        "comparison bootstrap unit",
        (comparison.get("bootstrap") or {}).get("unit"),
        "task_key",
    )
    require_exact(
        "comparison paired bootstrap",
        (comparison.get("bootstrap") or {}).get("paired"),
        True,
    )
    input_artifacts = comparison.get("input_artifacts") or {}
    require_exact(
        "comparison outcome hash",
        (input_artifacts.get("test_outcomes") or {}).get("sha256"),
        prepare_v4.EXPECTED_OUTPUT_SHA256["test_outcomes_sha256"],
    )
    require_exact(
        "comparison preference hash",
        (input_artifacts.get("test_preference_pairs") or {}).get("sha256"),
        prepare_v4.EXPECTED_OUTPUT_SHA256["test_preference_pairs_sha256"],
    )
    input_arms = input_artifacts.get("arms") or {}
    require_exact("comparison input-arm order", tuple(input_arms), ARM_ORDER)
    for arm in ARM_ORDER:
        evaluation = evaluations[arm]
        expected = {
            "metrics_sha256": evaluation["metrics_sha256"],
            "contract_sha256": evaluation["contract_sha256"],
            "predictions_sha256": evaluation["predictions_sha256"],
            "checkpoint_fingerprint": evaluation["checkpoint_fingerprint"],
        }
        if arm != "standard_v3":
            expected["run_manifest_sha256"] = training[arm][
                "run_manifest_sha256"
            ]
        for key, value in expected.items():
            require_exact(
                f"comparison {arm} {key}",
                (input_arms.get(arm) or {}).get(key),
                value,
            )
        require_exact(
            f"comparison {arm} summary checkpoint",
            (comparison["arms"].get(arm) or {}).get("checkpoint_fingerprint"),
            evaluation["checkpoint_fingerprint"],
        )
    for filename, expected in standard["file_sha256"].items():
        report_key = {
            "metrics.json": "metrics_sha256",
            "metrics.contract.json": "contract_sha256",
            "metrics.predictions.jsonl": "predictions_sha256",
        }[filename]
        require_exact(
            f"comparison Standard V3 {report_key}",
            input_arms["standard_v3"].get(report_key),
            expected,
        )

    pair_scoring = comparison.get("pair_scoring")
    require(isinstance(pair_scoring, dict), "comparison lacks required pair_scoring")
    require_exact(
        "comparison strict pair examples",
        pair_scoring.get("strict_pair_examples"),
        STRICT_PAIR_EXAMPLES,
    )
    score_arms = pair_scoring.get("arms") or {}
    require_exact("comparison pair-score arm order", tuple(score_arms), ARM_ORDER)
    require_exact(
        "Standard V3 pair-score status",
        (score_arms.get("standard_v3") or {}).get("status"),
        "not_available",
    )
    for arm in TRAINED_ARMS:
        report_score = score_arms.get(arm) or {}
        require_exact(f"{arm} pair-score status", report_score.get("status"), "complete")
        for key in (
            "metrics_sha256",
            "pair_scores_sha256",
            "score_manifest_sha256",
        ):
            require_exact(
                f"comparison {arm} pair-score {key}",
                report_score.get(key),
                scores[arm][key],
            )
    require_exact(
        "DPO pair-score comparison status",
        (pair_scoring.get("dpo_minus_continued_sft") or {}).get("status"),
        "complete",
    )
    require(bool(data_audit["build_summary_sha256"]), "data audit binding is empty")


def rerender_and_compare(
    analysis_dir: Path,
    recomputed: dict[str, Any],
) -> tuple[Path, Path, Path]:
    expected_json = json.dumps(recomputed, ensure_ascii=False, indent=2) + "\n"
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        expected_paths = {
            "comparison.json": temp / "comparison.json",
            "comparison.md": temp / "comparison.md",
            "comparison.csv": temp / "comparison.csv",
        }
        aggregate_v4.atomic_write_text(expected_paths["comparison.json"], expected_json)
        aggregate_v4.atomic_write_text(
            expected_paths["comparison.md"],
            aggregate_v4.render_markdown(recomputed),
        )
        aggregate_v4.write_csv(expected_paths["comparison.csv"], recomputed)
        for filename, expected_path in expected_paths.items():
            observed_path = analysis_dir / filename
            require(
                observed_path.is_file() and not observed_path.is_symlink(),
                f"missing or unsafe analysis output: {observed_path}",
            )
            require(
                observed_path.read_bytes() == expected_path.read_bytes(),
                f"stale or hand-edited analysis output: {observed_path}",
            )
    return tuple(analysis_dir / name for name in expected_paths)  # type: ignore[return-value]


def validate_analysis(
    analysis_dir: Path,
    result_dirs: dict[str, Path],
    score_dirs: dict[str, Path],
    processed_dir: Path,
    training: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    standard: dict[str, Any],
    data_audit: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    comparison_path = analysis_dir / "comparison.json"
    comparison = load_json(comparison_path)
    require_report_binding(
        comparison,
        training,
        evaluations,
        scores,
        standard,
        data_audit,
    )
    recomputed = aggregate_v4.aggregate(
        result_dirs,
        processed_dir / "evaluation" / "test_outcomes.jsonl",
        processed_dir / "evaluation" / "test_preference_pairs.jsonl",
        pair_score_dirs=score_dirs,
    )
    require_exact("stored/recomputed comparison", comparison, recomputed)
    analysis_paths = rerender_and_compare(analysis_dir, recomputed)
    return recomputed, analysis_paths


def add_file(
    allowlist: list[tuple[Path, Path]],
    source: Path,
    relative: Path,
) -> None:
    require(
        source.is_file() and not source.is_symlink() and source.stat().st_size > 0,
        f"missing or unsafe package input: {source}",
    )
    require(
        ".." not in relative.parts and not relative.is_absolute(),
        f"unsafe package destination: {relative}",
    )
    allowlist.append((source, relative))


def write_generated_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/qlora_v4"),
    )
    parser.add_argument(
        "--standard-v3-result-dir",
        type=Path,
        default=Path("results/qlora_v3/constrained_recovery"),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/analysis_v4"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/qlora_v4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qlora_v4/rtx5060"),
    )
    args = parser.parse_args()

    validate_static_constants()
    commit, source_hashes = source_commit()
    results = ROOT / args.results_root
    standard_dir = ROOT / args.standard_v3_result_dir
    analysis_dir = ROOT / args.analysis_dir
    processed_dir = ROOT / args.processed_dir
    output = ROOT / args.output_dir
    if output.exists():
        require(
            output.is_dir() and not output.is_symlink() and not any(output.iterdir()),
            f"package output is not an empty safe directory: {output}",
        )

    data_audit = validate_data_audit(processed_dir)
    standard = validate_standard_v3(standard_dir)
    checkpoints = {
        arm: results / arm / "checkpoint_final" for arm in TRAINED_ARMS
    }
    checkpoint_identities = {
        arm: checkpoint_identity(path) for arm, path in checkpoints.items()
    }
    fingerprints = {
        arm: identity[0] for arm, identity in checkpoint_identities.items()
    }
    require(
        len(set(fingerprints.values())) == len(TRAINED_ARMS),
        f"new-arm checkpoints are not distinct: {fingerprints}",
    )

    training: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {
        "standard_v3": {
            "metrics_sha256": standard["file_sha256"]["metrics.json"],
            "contract_sha256": standard["file_sha256"]["metrics.contract.json"],
            "predictions_sha256": standard["file_sha256"][
                "metrics.predictions.jsonl"
            ],
            "checkpoint_fingerprint": standard["checkpoint_fingerprint"],
        }
    }
    scores: dict[str, dict[str, Any]] = {}
    comparison_ids: list[str] = []
    for arm in TRAINED_ARMS:
        checkpoint_fp, checkpoint_hashes = checkpoint_identities[arm]
        clean_fp, clean_hashes = checkpoint_identities["clean_sft"]
        result_dir = results / arm
        training[arm] = validate_training(
            result_dir,
            arm,
            checkpoint_fp,
            checkpoint_hashes,
            clean_fp,
            clean_hashes,
            commit,
        )
        evaluations[arm] = validate_eval(result_dir, checkpoint_fp)
        score_dir = results / "pair_scores" / arm
        require(score_dir.is_dir(), f"required pair-score directory missing: {score_dir}")
        scores[arm] = validate_score(
            score_dir,
            checkpoint_fp,
            checkpoint_hashes,
            commit,
        )
        if arm != "clean_sft":
            comparison_id = training[arm]["manifest"].get("comparison_contract_id")
            require(
                isinstance(comparison_id, str) and bool(comparison_id),
                f"{arm} lacks comparison_contract_id",
            )
            comparison_ids.append(comparison_id)
    require(
        len(comparison_ids) == 2 and len(set(comparison_ids)) == 1,
        "continued-SFT and DPO comparison-contract IDs differ",
    )

    result_dirs = {
        "standard_v3": standard_dir,
        **{arm: results / arm for arm in TRAINED_ARMS},
    }
    score_dirs = {
        arm: results / "pair_scores" / arm for arm in TRAINED_ARMS
    }
    comparison, analysis_paths = validate_analysis(
        analysis_dir,
        result_dirs,
        score_dirs,
        processed_dir,
        training,
        evaluations,
        scores,
        standard,
        data_audit,
    )
    commands_path = results / "commands.jsonl"
    validate_commands(commands_path, commit)

    allowlist: list[tuple[Path, Path]] = []
    for arm in TRAINED_ARMS:
        for source in training[arm]["paths"]:
            add_file(allowlist, source, Path("results") / arm / source.name)
        for source in evaluations[arm]["paths"]:
            add_file(allowlist, source, Path("results") / arm / source.name)
        for filename in CHECKPOINT_FILES:
            add_file(
                allowlist,
                checkpoints[arm] / filename,
                Path("results") / arm / "checkpoint_final" / filename,
            )
        for source in scores[arm]["paths"]:
            add_file(
                allowlist,
                source,
                Path("pair_scores") / arm / source.name,
            )
    for source in standard["paths"]:
        add_file(
            allowlist,
            source,
            Path("reference") / "standard_v3" / source.name,
        )
    for source in analysis_paths:
        add_file(allowlist, source, Path("analysis") / source.name)
    for source in data_audit["paths"]:
        add_file(allowlist, source, Path("data_audit") / source.name)
    for source, relative in (
        (ROOT / "configs" / "qlora_v4.yaml", Path("configs/qlora_v4.yaml")),
        (ROOT / "BASELINE_V4_HANDOFF.md", Path("BASELINE_V4_HANDOFF.md")),
        (ROOT / "V4_RTX5060_AGENT_PROMPT.md", Path("V4_RTX5060_AGENT_PROMPT.md")),
        (ROOT / "requirements-gpu-v4.txt", Path("requirements-gpu-v4.txt")),
        (commands_path, Path("results/commands.jsonl")),
    ):
        add_file(allowlist, source, relative)
    require(
        len({relative.as_posix() for _, relative in allowlist}) == len(allowlist),
        "duplicate destination in package allowlist",
    )
    require(
        not any(
            relative.parts
            and relative.parts[0] in {"raw", "processed", "data"}
            and relative.suffix == ".jsonl"
            for _, relative in allowlist
        ),
        "raw/processed JSONL must not be included in the upload package",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for source, relative in allowlist:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        source_provenance = {
            "protocol": "qlora_v4_source_provenance",
            "source_tag": FROZEN_TAG,
            "source_commit": commit,
            "tracked_v4_files_sha256": source_hashes,
        }
        standard_provenance = {
            "protocol": "qlora_v4_standard_v3_provenance",
            "source_branch": standard["source_branch"],
            "source_commit": standard["source_commit"],
            "source_artifact_prefix": standard["source_artifact_prefix"],
            "checkpoint_fingerprint": standard["checkpoint_fingerprint"],
            "file_sha256": standard["file_sha256"],
        }
        write_generated_json(staging / "SOURCE_PROVENANCE.json", source_provenance)
        write_generated_json(
            staging / "reference" / "standard_v3" / "PROVENANCE.json",
            standard_provenance,
        )
        readme = {
            "protocol": "qlora_v4_result_package",
            "source_tag": FROZEN_TAG,
            "source_commit": commit,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arms": list(ARM_ORDER),
            "newly_trained_arms": list(TRAINED_ARMS),
            "standard_v3_source_commit": standard["source_commit"],
            "standard_v3_checkpoint_fingerprint": standard[
                "checkpoint_fingerprint"
            ],
            "analysis_recomputed_and_byte_verified": True,
            "required_pair_scoring_complete": True,
            "raw_or_processed_jsonl_included": False,
            "claim_boundary": comparison.get("claim_boundary"),
            "comparison_contract_id": comparison_ids[0],
        }
        write_generated_json(staging / "README.json", readme)

        manifest_rows = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                manifest_rows.append(
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        upload_manifest = {
            "protocol": "qlora_v4_result_package",
            "source_tag": FROZEN_TAG,
            "source_commit": commit,
            "standard_v3_source_commit": standard["source_commit"],
            "files": manifest_rows,
            "total_bytes": sum(row["bytes"] for row in manifest_rows),
        }
        write_generated_json(staging / "UPLOAD_MANIFEST.json", upload_manifest)
        if output.exists():
            output.rmdir()
        staging.replace(output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "files": len(upload_manifest["files"]) + 1,
                "total_bytes": upload_manifest["total_bytes"],
                "standard_v3_commit": standard["source_commit"],
                "analysis_recomputed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
