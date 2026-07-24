#!/usr/bin/env python3
"""Stage runner for the frozen V4 clean-SFT -> preference experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_qlora_v2 import MODEL, MODEL_REVISION, SEED
from prepare_qlora_v3 import EXPECTED_TEST_SHA256
from prepare_qlora_v4 import (
    EXPECTED_COUNTS,
    EXPECTED_OUTPUT_SHA256,
)

FROZEN_TAG = "v4-frozen-20260724-p2"
V3_RESULT_BRANCH = "results/v3-rtx5060-20260724"
V3_RESULT_COMMIT = "aedf77a5784a364bd76bad42aa0a6cb6fad555b6"
V3_RESULT_SHA256 = {
    "metrics.json": (
        "880db45dcb6dc6eea497aa32dff26c5d59a4ab3b570c458c6f132757ea9d61f4"
    ),
    "metrics.contract.json": (
        "2b941573f85b9c1d33622c4e6fde42d10af194981fe028baa7e210d39a455471"
    ),
    "metrics.predictions.jsonl": (
        "491e1613b20eb11b176d9aac61e19b4e3472257d3a2576125c76c6e03cb24de3"
    ),
}
V3_CHECKPOINT_FINGERPRINT = (
    "3cdfa858353e8f7ea6da0d5558c21014bacbd58b2092f7095d6b5925f147825c"
)
V3_ARTIFACT_PREFIX = (
    "artifacts/qlora_v3/rtx5060/results/qlora_v3/"
    "constrained_recovery"
)
LONGEST_PAIR_ID = (
    "sonnet-35-new-retail:task106:trial4:action28:"
    "prefer_repair_over_action26"
)
PREFERENCE_SCHEDULE_SHA256 = (
    "f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341"
)
PREFERENCE_SMOKE_SHA256 = (
    "86fd923875ba3d11c50d635c409246e6f09437c771a4b4881c02cbba47190eb4"
)
TEST_PREFERENCE_PAIRS_SHA256 = (
    "b85548e9f1c041032358172e10b7f7f53f91710d1d15f26dfaa606a07799cf74"
)

ROOT = Path(__file__).resolve().parents[1]
V3_PROCESSED = ROOT / "data" / "processed" / "qlora_v3"
V4_PROCESSED = ROOT / "data" / "processed" / "qlora_v4"
V4_RESULTS = ROOT / "results" / "qlora_v4"
TEST_FILE = V3_PROCESSED / "shared" / "test.jsonl"
ARM_RESULT_DIRS = {
    "clean_sft": V4_RESULTS / "clean_sft",
    "continued_sft": V4_RESULTS / "continued_sft",
    "dpo": V4_RESULTS / "dpo",
}
SMOKE_MAX_RESERVED_BYTES = 8_053_063_680
FROZEN_SOURCE_FILES = (
    ".gitignore",
    "README.md",
    "BASELINE_V4_HANDOFF.md",
    "V4_RTX5060_AGENT_PROMPT.md",
    "configs/qlora_v4.yaml",
    "requirements-gpu-v4.txt",
    "scripts/aggregate_qlora_v4.py",
    "scripts/evaluate_tool_actions_v4.py",
    "scripts/package_qlora_v4_results.py",
    "scripts/prepare_qlora_v4.py",
    "scripts/run_qlora_v4.py",
    "scripts/train_preference_v4.py",
    "scripts/train_qlora_v4_sft.py",
    "tests/test_v4_package.py",
    "tests/test_v4_protocol.py",
    "tests/test_v4_runtime.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing JSONL: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"invalid JSONL rows: {path}")
    return rows


def checkpoint_fingerprint(path: Path) -> str:
    require_checkpoint(path)
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(filename.encode("utf-8"))
        digest.update(sha256_file(path / filename).encode("ascii"))
    return digest.hexdigest()


def prepare_output_directory(
    path: Path,
    label: str,
    *,
    archive_partial: bool,
) -> None:
    occupied = path.exists() and (
        not path.is_dir() or any(path.iterdir())
    )
    if occupied:
        if not archive_partial:
            raise RuntimeError(
                f"{label} is non-empty: {path}. Confirm no matching process is "
                "still running, then repeat the stage with "
                "--archive-partial-output to preserve the failed attempt."
            )
        archive_root = V4_RESULTS / "failed_attempts"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = archive_root / f"{stamp}_{path.name}"
        suffix = 1
        while destination.exists():
            destination = archive_root / f"{stamp}_{path.name}_{suffix}"
            suffix += 1
        path.replace(destination)
        print(
            f"archived partial {label} output to {destination}",
            flush=True,
        )
    path.mkdir(parents=True, exist_ok=True)


def require_checkpoint(path: Path) -> None:
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        target = path / filename
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"incomplete adapter checkpoint: {target}")


def source_commit(*, require_tag: bool) -> str:
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *FROZEN_SOURCE_FILES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "tracked source/config is dirty; use a clean frozen checkout"
        )
    if require_tag:
        tagged = subprocess.run(
            ["git", "rev-list", "-n", "1", FROZEN_TAG],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tagged != commit:
            raise RuntimeError(
                f"HEAD {commit} differs from frozen tag {FROZEN_TAG}={tagged}"
            )
    return commit


def append_command(record: dict[str, Any]) -> None:
    path = V4_RESULTS / "commands.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.is_file() else b""
    if existing:
        if not existing.endswith(b"\n"):
            raise RuntimeError(
                f"command audit has an unterminated tail: {path}"
            )
        # Refuse to conceal an older corrupt command receipt.
        load_jsonl(path)
    line = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".atomic.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(existing)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def record_validated_completion(stage: str) -> None:
    """Recover a lost runner receipt only after the stage artifacts validate."""
    commit = source_commit(require_tag=True)
    path = V4_RESULTS / "commands.jsonl"
    if path.is_file():
        rows = load_jsonl(path)
        if any(
            row.get("stage") == stage
            and type(row.get("returncode")) is int
            and row.get("returncode") == 0
            and row.get("source_commit") == commit
            and isinstance(row.get("finished_utc"), str)
            and bool(row.get("finished_utc"))
            for row in rows
        ):
            return
    timestamp = utc_now()
    append_command(
        {
            "stage": stage,
            "started_utc": timestamp,
            "finished_utc": timestamp,
            "command": ["<recovered-from-validated-artifacts>"],
            "source_commit": commit,
            "returncode": 0,
            "recovered_from_validated_artifacts": True,
        }
    )


def run_command(command: list[str], stage: str) -> None:
    record = {
        "stage": stage,
        "started_utc": utc_now(),
        "command": command,
        "source_commit": source_commit(require_tag=False),
    }
    append_command(record)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    record.update(
        {
            "finished_utc": utc_now(),
            "returncode": completed.returncode,
        }
    )
    append_command(record)
    if completed.returncode:
        raise RuntimeError(
            f"{stage} failed with exit code {completed.returncode}"
        )


def validate_prepare_audit() -> dict[str, Any]:
    audit_path = V4_PROCESSED / "contract_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("V4 contract audit is missing; run prepare first")
    audit = load_json(audit_path)
    if audit.get("valid") is not True or audit.get("errors") != []:
        raise RuntimeError(f"V4 audit is not valid: {audit}")
    if audit.get("counts") != EXPECTED_COUNTS:
        raise RuntimeError("V4 count contract drift")
    if audit.get("hashes") != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError("V4 output hash contract drift")
    checks = audit.get("checks") or {}
    if not checks:
        raise RuntimeError("V4 audit has no checks")
    failed_checks = {}
    for key, value in checks.items():
        if isinstance(value, bool):
            passed = value is True
        elif isinstance(value, (int, float)):
            passed = value == 0
        else:
            passed = False
        if not passed:
            failed_checks[key] = value
    if failed_checks:
        raise RuntimeError(f"V4 audit checks failed: {failed_checks}")
    if sha256_file(TEST_FILE) != EXPECTED_TEST_SHA256:
        raise RuntimeError("frozen 959-example test hash drift")
    return audit


def validate_clean_training_result(
    result_dir: Path,
    *,
    smoke: bool,
) -> str:
    checkpoint = result_dir / "checkpoint_final"
    fingerprint = checkpoint_fingerprint(checkpoint)
    manifest = load_json(result_dir / "run_manifest.json")
    expected_steps = 2 if smoke else 68
    expected_grad_accum = 8 if smoke else 16
    required = {
        "protocol": "qlora_v4_clean_sft",
        "source_tag": FROZEN_TAG,
        "source_commit": source_commit(require_tag=True),
        "arm": "clean_sft",
        "smoke_test": smoke,
        "failed_action_labels": 0,
        "max_steps": expected_steps,
        "grad_accum": expected_grad_accum,
        "train_file_sha256": EXPECTED_OUTPUT_SHA256[
            "clean_train_schedule_sha256"
        ],
        "validation_file_sha256": EXPECTED_OUTPUT_SHA256[
            "clean_validation_sha256"
        ],
        "held_out_test_accessed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"clean-SFT result drift in {result_dir}: "
                f"{key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if (manifest.get("loss_audit") or {}).get("finite") is not True:
        raise RuntimeError(f"clean-SFT finite-loss audit failed: {result_dir}")
    if (
        manifest.get("output_checkpoint") or {}
    ).get("checkpoint_fingerprint") != fingerprint:
        raise RuntimeError(f"clean-SFT checkpoint fingerprint drift: {result_dir}")
    reserved = (
        manifest.get("environment") or {}
    ).get("peak_cuda_reserved_bytes")
    if (
        isinstance(reserved, bool)
        or not isinstance(reserved, int)
        or reserved < 0
    ):
        raise RuntimeError(
            f"clean-SFT result lacks valid peak reserved memory: {reserved!r}"
        )
    if smoke:
        if reserved > SMOKE_MAX_RESERVED_BYTES:
            raise RuntimeError(
                f"clean-SFT smoke memory gate failed: {reserved!r}"
            )
    return fingerprint


def validate_longest_prompt_smoke(path: Path) -> None:
    metrics = load_json(path)
    if (
        metrics.get("protocol") != "qlora_v4"
        or metrics.get("evaluated_examples") != 1
        or metrics.get("limited") is not True
        or metrics.get("smoke_selection") != "longest_prompt"
        or metrics.get("smoke_forced_new_tokens") != 128
    ):
        raise RuntimeError(f"longest-prompt smoke contract failed: {path}")
    peak_reserved = (
        metrics.get("runtime_memory") or {}
    ).get("peak_cuda_memory_reserved_bytes")
    if (
        not isinstance(peak_reserved, int)
        or peak_reserved > SMOKE_MAX_RESERVED_BYTES
    ):
        raise RuntimeError(
            f"longest-prompt smoke exceeds 7.5 GiB gate: {peak_reserved!r}"
        )
    predictions = path.with_name(path.stem + ".predictions.jsonl")
    rows = load_jsonl(predictions)
    if len(rows) != 1 or rows[0].get("generated_token_count") != 128:
        raise RuntimeError(
            f"longest-prompt smoke did not generate 128 tokens: {path}"
        )


def validate_preference_training_result(
    result_dir: Path,
    arm: str,
    *,
    smoke: bool,
) -> tuple[str, str]:
    checkpoint = result_dir / "checkpoint_final"
    fingerprint = checkpoint_fingerprint(checkpoint)
    clean_fingerprint = checkpoint_fingerprint(
        ARM_RESULT_DIRS["clean_sft"] / "checkpoint_final"
    )
    manifest = load_json(result_dir / "run_manifest.json")
    expected_rows = 16 if smoke else 144
    expected_steps = 2 if smoke else 18
    expected_hash = (
        PREFERENCE_SMOKE_SHA256 if smoke else PREFERENCE_SCHEDULE_SHA256
    )
    required = {
        "protocol": "qlora_v4_preference_continuation",
        "source_tag": FROZEN_TAG,
        "source_commit": source_commit(require_tag=True),
        "formal_result": not smoke,
        "arm": arm,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{arm} result drift in {result_dir}: "
                f"{key}={manifest.get(key)!r}, expected {expected!r}"
            )
    data = manifest.get("data") or {}
    compute = manifest.get("compute_contract") or {}
    if (
        data.get("train_schedule_sha256") != expected_hash
        or data.get("held_out_test_accessed") is not False
        or compute.get("optimizer_steps") != expected_steps
        or compute.get("scheduled_microbatches") != expected_rows
        or compute.get("gradient_accumulation_steps") != 8
        or compute.get("model_accepts_loss_kwargs") is not False
        or compute.get(
            "custom_loss_gradient_accumulation_scaled_by_trainer"
        ) is not True
    ):
        raise RuntimeError(f"{arm} data/compute contract failed: {result_dir}")
    if (
        manifest.get("clean_sft_initialization") or {}
    ).get("checkpoint_fingerprint") != clean_fingerprint:
        raise RuntimeError(f"{arm} Clean-SFT initialization drift")
    if (
        manifest.get("output_checkpoint") or {}
    ).get("checkpoint_fingerprint") != fingerprint:
        raise RuntimeError(f"{arm} output checkpoint fingerprint drift")
    if (manifest.get("loss_audit") or {}).get("finite") is not True:
        raise RuntimeError(f"{arm} finite-loss audit failed")
    if arm == "dpo":
        dpo = manifest.get("dpo_reference_audit") or {}
        if (
            dpo.get("reference_state_unchanged") is not True
            or dpo.get("policy_state_changed") is not True
            or (
                dpo.get("initial_numerical_audit") or {}
            ).get("policy_reference_logps_match") is not True
        ):
            raise RuntimeError(f"DPO reference/policy audit failed: {result_dir}")
    runtime_reserved = (
        manifest.get("runtime") or {}
    ).get("peak_cuda_memory_reserved_bytes")
    if (
        isinstance(runtime_reserved, bool)
        or not isinstance(runtime_reserved, int)
        or runtime_reserved < 0
    ):
        raise RuntimeError(
            f"{arm} lacks valid peak reserved memory: {runtime_reserved!r}"
        )
    if smoke:
        memory = manifest.get("smoke_memory_audit") or {}
        if (
            memory.get("within_limit") is not True
            or memory.get("peak_cuda_memory_reserved_bytes")
            != runtime_reserved
            or runtime_reserved > SMOKE_MAX_RESERVED_BYTES
        ):
            raise RuntimeError(f"{arm} smoke memory gate failed")
    comparison_id = manifest.get("comparison_contract_id")
    if not isinstance(comparison_id, str) or not comparison_id:
        raise RuntimeError(f"{arm} lacks comparison_contract_id")
    return fingerprint, comparison_id


def validate_formal_metrics(result_dir: Path) -> None:
    path = result_dir / "metrics.json"
    metrics = load_json(path)
    contract = load_json(result_dir / "metrics.contract.json")
    predictions_path = result_dir / "metrics.predictions.jsonl"
    predictions = load_jsonl(predictions_path)
    expected_fingerprint = checkpoint_fingerprint(
        result_dir / "checkpoint_final"
    )
    required = {
        "protocol": "qlora_v4",
        "test_file_sha256": EXPECTED_TEST_SHA256,
        "formal_test_examples": 959,
        "evaluated_examples": 959,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "base_model_loading": "nf4_4bit",
        "max_prompt_tokens": 1664,
        "limited": False,
        "checkpoint_fingerprint": expected_fingerprint,
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"formal metric contract drift in {path}: "
                f"{key}={metrics.get(key)!r}, expected {expected!r}"
            )
        if contract.get(key) != expected:
            raise RuntimeError(
                f"formal contract drift in {result_dir}: "
                f"{key}={contract.get(key)!r}, expected {expected!r}"
            )
    generation = metrics.get("generation") or {}
    if generation != {
        "do_sample": False,
        "max_new_tokens": 128,
        "batch_size": 1,
    }:
        raise RuntimeError(f"generation contract drift in {path}")
    if contract.get("generation") != generation:
        raise RuntimeError(f"metrics/contract generation drift in {result_dir}")
    if len(predictions) != 959:
        raise RuntimeError(f"formal predictions incomplete in {result_dir}")
    ids = [row.get("example_id") for row in predictions]
    expected_ids = [row.get("example_id") for row in load_jsonl(TEST_FILE)]
    if ids != expected_ids or len(set(ids)) != 959:
        raise RuntimeError(
            f"formal prediction IDs/order differ from frozen test in {result_dir}"
        )
    for index, row in enumerate(predictions):
        if not isinstance(row.get("generated_text"), str):
            raise RuntimeError(
                f"formal prediction {index} lacks generated_text in {result_dir}"
            )
        if not isinstance(row.get("target"), dict):
            raise RuntimeError(
                f"formal prediction {index} lacks target in {result_dir}"
            )
        if any(
            not isinstance(row.get(key), bool)
            for key in (
                "json_valid",
                "tool_name_correct",
                "arguments_exact",
                "full_call_exact",
            )
        ):
            raise RuntimeError(
                f"formal prediction {index} has invalid score flags in {result_dir}"
            )
    if metrics.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError(f"formal prediction hash drift in {result_dir}")
    overall = ((metrics.get("groups") or {}).get("overall") or {})
    micro = overall.get("micro") or {}
    if overall.get("examples") != 959:
        raise RuntimeError(f"formal overall example count drift in {result_dir}")
    for key in (
        "json_valid",
        "tool_name_correct",
        "arguments_exact",
        "full_call_exact",
    ):
        value = micro.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise RuntimeError(
                f"formal micro metric {key} is invalid in {result_dir}: {value!r}"
            )
        recomputed = sum(bool(row[key]) for row in predictions) / 959
        if not math.isclose(
            float(value),
            recomputed,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"formal micro metric {key} disagrees with predictions in "
                f"{result_dir}: {value!r} != {recomputed!r}"
            )
    memory = metrics.get("runtime_memory") or {}
    for key in (
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    ):
        value = memory.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"formal runtime memory {key} is invalid in {result_dir}: "
                f"{value!r}"
            )


def validate_pair_score_result(result_dir: Path, arm: str) -> None:
    metrics = load_json(result_dir / "metrics.json")
    manifest = load_json(result_dir / "score_manifest.json")
    scores_path = result_dir / "pair_scores.jsonl"
    scores = load_jsonl(scores_path)
    expected_fingerprint = checkpoint_fingerprint(
        ARM_RESULT_DIRS[arm] / "checkpoint_final"
    )
    if (
        manifest.get("protocol")
        != "qlora_v4_preference_continuation_pair_scoring"
        or manifest.get("source_tag") != FROZEN_TAG
        or manifest.get("source_commit") != source_commit(require_tag=True)
        or manifest.get("split") != "test"
        or manifest.get("training_performed") is not False
        or manifest.get("limited") is not False
        or manifest.get("complete") is not True
        or manifest.get("pair_count") != 48
        or manifest.get("expected_pairs") != 48
        or manifest.get("completed_pairs") != 48
        or manifest.get("preference_pairs_sha256")
        != TEST_PREFERENCE_PAIRS_SHA256
        or manifest.get("checkpoint_fingerprint") != expected_fingerprint
    ):
        raise RuntimeError(f"{arm} pair-score manifest contract failed")
    if (
        len(scores) != 48
        or metrics.get("protocol")
        != "qlora_v4_preference_continuation_pair_scoring"
        or metrics.get("split") != "test"
        or metrics.get("pair_count") != 48
        or metrics.get("completion_eos_included") is not True
        or (manifest.get("outputs") or {}).get("pair_scores_rows") != 48
    ):
        raise RuntimeError(f"{arm} pair-score rows incomplete")
    metrics_hash = sha256_file(result_dir / "metrics.json")
    if (
        manifest.get("pair_scores_sha256") != sha256_file(scores_path)
        or manifest.get("metrics_sha256") != metrics_hash
        or (manifest.get("outputs") or {}).get("metrics_sha256")
        != metrics_hash
    ):
        raise RuntimeError(f"{arm} pair-score output hash drift")
    expected_pairs_path = (
        V4_PROCESSED / "evaluation" / "test_preference_pairs.jsonl"
    )
    if sha256_file(expected_pairs_path) != TEST_PREFERENCE_PAIRS_SHA256:
        raise RuntimeError("frozen test preference-pair hash drift")
    expected_pairs = load_jsonl(expected_pairs_path)
    if len(expected_pairs) != 48:
        raise RuntimeError("frozen test preference-pair count drift")
    summed_margins: list[float] = []
    normalized_margins: list[float] = []
    summed_correct: list[bool] = []
    normalized_correct: list[bool] = []
    for index, (row, expected) in enumerate(zip(scores, expected_pairs)):
        expected_content = hashlib.sha256(
            json.dumps(
                {
                    "prompt": expected.get("prompt"),
                    "chosen": expected.get("chosen"),
                    "rejected": expected.get("rejected"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            row.get("pair_index") != index
            or row.get("pair_id") != expected.get("pair_id")
            or row.get("pair_content_sha256") != expected_content
            or row.get("split") != "test"
            or row.get("task_key") != expected.get("task_key")
            or not isinstance(row.get("summed_logp_correct"), bool)
            or not isinstance(
                row.get("per_token_normalized_correct"), bool
            )
        ):
            raise RuntimeError(
                f"{arm} pair-score row schema/provenance drift at {index}"
            )
        chosen_tokens = row.get("chosen_tokens_including_eos")
        rejected_tokens = row.get("rejected_tokens_including_eos")
        if (
            isinstance(chosen_tokens, bool)
            or not isinstance(chosen_tokens, int)
            or chosen_tokens <= 0
            or isinstance(rejected_tokens, bool)
            or not isinstance(rejected_tokens, int)
            or rejected_tokens <= 0
        ):
            raise RuntimeError(
                f"{arm} pair-score token-count drift at {index}"
            )
        numeric_keys = (
            "chosen_summed_logp_including_eos",
            "rejected_summed_logp_including_eos",
            "summed_logp_margin_chosen_minus_rejected",
            "chosen_per_token_logp_including_eos",
            "rejected_per_token_logp_including_eos",
            "per_token_normalized_margin_chosen_minus_rejected",
        )
        numeric: dict[str, float] = {}
        for key in numeric_keys:
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"{arm} pair-score {key} is invalid at {index}: "
                    f"{value!r}"
                )
            numeric[key] = float(value)
        chosen_sum = numeric["chosen_summed_logp_including_eos"]
        rejected_sum = numeric["rejected_summed_logp_including_eos"]
        summed_margin = numeric[
            "summed_logp_margin_chosen_minus_rejected"
        ]
        chosen_mean = numeric["chosen_per_token_logp_including_eos"]
        rejected_mean = numeric["rejected_per_token_logp_including_eos"]
        normalized_margin = numeric[
            "per_token_normalized_margin_chosen_minus_rejected"
        ]
        comparisons = (
            (summed_margin, chosen_sum - rejected_sum),
            (chosen_mean, chosen_sum / chosen_tokens),
            (rejected_mean, rejected_sum / rejected_tokens),
            (normalized_margin, chosen_mean - rejected_mean),
        )
        if any(
            not math.isclose(
                observed,
                recomputed,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            for observed, recomputed in comparisons
        ):
            raise RuntimeError(
                f"{arm} pair-score arithmetic drift at {index}"
            )
        if (
            row["summed_logp_correct"] is not (summed_margin > 0.0)
            or row["per_token_normalized_correct"]
            is not (normalized_margin > 0.0)
        ):
            raise RuntimeError(
                f"{arm} pair-score correctness drift at {index}"
            )
        summed_margins.append(summed_margin)
        normalized_margins.append(normalized_margin)
        summed_correct.append(row["summed_logp_correct"])
        normalized_correct.append(
            row["per_token_normalized_correct"]
        )
    expected_metrics: dict[str, int | float] = {
        "pair_accuracy_summed_logp": sum(summed_correct) / 48,
        "summed_logp_correct_count": sum(summed_correct),
        "summed_logp_tie_count": sum(
            margin == 0.0 for margin in summed_margins
        ),
        "mean_summed_logp_margin": statistics.fmean(summed_margins),
        "median_summed_logp_margin": statistics.median(summed_margins),
        "per_token_normalized_pair_accuracy":
            sum(normalized_correct) / 48,
        "per_token_normalized_correct_count": sum(normalized_correct),
        "mean_per_token_normalized_margin":
            statistics.fmean(normalized_margins),
        "median_per_token_normalized_margin":
            statistics.median(normalized_margins),
    }
    for key, expected in expected_metrics.items():
        observed = metrics.get(key)
        if isinstance(expected, int):
            valid = type(observed) is int and observed == expected
        else:
            valid = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and math.isfinite(float(observed))
                and math.isclose(
                    float(observed),
                    expected,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
            )
        if not valid:
            raise RuntimeError(
                f"{arm} pair-score metric {key} drift: "
                f"{observed!r} != {expected!r}"
            )
    for key in (
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"{arm} pair-score memory {key} is invalid: {value!r}"
            )


def prepare(data_dir: Path, local_files_only: bool) -> None:
    v3_command = [
        sys.executable,
        "scripts/prepare_qlora_v3.py",
        "--data-dir",
        str(data_dir),
    ]
    v4_command = [
        sys.executable,
        "scripts/prepare_qlora_v4.py",
        "--data-dir",
        str(data_dir),
    ]
    if local_files_only:
        v3_command.append("--local-files-only")
        v4_command.append("--local-files-only")
    run_command(v3_command, "prepare-v3-frozen-input")
    run_command(v4_command, "prepare-v4")
    validate_prepare_audit()


def smoke_clean(*, archive_partial: bool) -> None:
    validate_prepare_audit()
    output = V4_RESULTS / "smoke_clean"
    if output.is_dir() and any(output.iterdir()):
        try:
            validate_clean_training_result(output, smoke=True)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            try:
                run_longest_prompt_smoke(
                    output / "checkpoint_final",
                    output / "longest_prompt_metrics.json",
                    "smoke-clean-sft-generation",
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ):
                if not archive_partial:
                    raise
            else:
                print(
                    "clean-SFT smoke artifacts already validate; skipping",
                    flush=True,
                )
                return
    prepare_output_directory(
        output,
        "clean-SFT smoke output",
        archive_partial=archive_partial,
    )
    run_command(
        [
            sys.executable,
            "scripts/train_qlora_v4_sft.py",
            "--train-file",
            str(V4_PROCESSED / "clean_sft" / "train_schedule.jsonl"),
            "--validation-file",
            str(V4_PROCESSED / "clean_sft" / "validation.jsonl"),
            "--output-dir",
            str(output),
            "--model-revision",
            MODEL_REVISION,
            "--smoke-test",
        ],
        "smoke-clean-sft-train",
    )
    validate_clean_training_result(output, smoke=True)
    run_longest_prompt_smoke(
        output / "checkpoint_final",
        output / "longest_prompt_metrics.json",
        "smoke-clean-sft-generation",
    )


def train_clean(*, archive_partial: bool) -> None:
    validate_prepare_audit()
    validate_clean_training_result(V4_RESULTS / "smoke_clean", smoke=True)
    validate_longest_prompt_smoke(
        V4_RESULTS / "smoke_clean" / "longest_prompt_metrics.json"
    )
    output = ARM_RESULT_DIRS["clean_sft"]
    if output.is_dir() and any(output.iterdir()):
        try:
            validate_clean_training_result(output, smoke=False)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            record_validated_completion("train-clean-sft")
            print("formal clean-SFT artifacts already validate; skipping", flush=True)
            return
    prepare_output_directory(
        output,
        "formal clean-SFT output",
        archive_partial=archive_partial,
    )
    run_command(
        [
            sys.executable,
            "scripts/train_qlora_v4_sft.py",
            "--train-file",
            str(V4_PROCESSED / "clean_sft" / "train_schedule.jsonl"),
            "--validation-file",
            str(V4_PROCESSED / "clean_sft" / "validation.jsonl"),
            "--output-dir",
            str(output),
            "--model-revision",
            MODEL_REVISION,
        ],
        "train-clean-sft",
    )
    validate_clean_training_result(output, smoke=False)


def preference_command(
    arm: str,
    pair_file: Path,
    expected_hash: str,
    output: Path,
    *,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_preference_v4.py",
        "--arm",
        arm,
        "--pair-file",
        str(pair_file),
        "--expected-pair-file-sha256",
        expected_hash,
        "--clean-sft-adapter",
        str(ARM_RESULT_DIRS["clean_sft"] / "checkpoint_final"),
        "--output-dir",
        str(output),
    ]
    if smoke:
        command.extend(
            [
                "--expected-longest-pair-id",
                LONGEST_PAIR_ID,
                "--smoke-test",
            ]
        )
    return command


def run_longest_prompt_smoke(
    checkpoint: Path,
    output: Path,
    stage: str,
) -> None:
    require_checkpoint(checkpoint)
    if output.is_file():
        try:
            validate_longest_prompt_smoke(output)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            return
    command = [
        sys.executable,
        "scripts/evaluate_tool_actions_v4.py",
        "--test-file",
        str(TEST_FILE),
        "--adapter",
        str(checkpoint),
        "--output",
        str(output),
        "--limit",
        "1",
        "--smoke-longest-prompt",
    ]
    predictions = output.with_name(
        output.stem + ".predictions.jsonl"
    )
    contract = output.with_name(output.stem + ".contract.json")
    if predictions.exists() or contract.exists():
        command.append("--resume")
    run_command(command, stage)
    validate_longest_prompt_smoke(output)


def smoke_preference(*, archive_partial: bool) -> None:
    validate_prepare_audit()
    clean_checkpoint = ARM_RESULT_DIRS["clean_sft"] / "checkpoint_final"
    validate_clean_training_result(ARM_RESULT_DIRS["clean_sft"], smoke=False)
    pair_file = V4_PROCESSED / "preference" / "smoke_pairs.jsonl"
    for arm in ("continued_sft", "dpo"):
        output = V4_RESULTS / "smoke_preference" / arm
        if output.is_dir() and any(output.iterdir()):
            try:
                validate_preference_training_result(
                    output,
                    arm,
                    smoke=True,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ):
                pass
            else:
                try:
                    run_longest_prompt_smoke(
                        output / "checkpoint_final",
                        output / "longest_prompt_metrics.json",
                        f"smoke-{arm}-generation",
                    )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ):
                    if not archive_partial:
                        raise
                else:
                    print(
                        f"{arm} smoke artifacts already validate; skipping",
                        flush=True,
                    )
                    continue
        prepare_output_directory(
            output,
            f"{arm} preference smoke output",
            archive_partial=archive_partial,
        )
        run_command(
            preference_command(
                arm,
                pair_file,
                PREFERENCE_SMOKE_SHA256,
                output,
                smoke=True,
            ),
            f"smoke-{arm}-train",
        )
        validate_preference_training_result(output, arm, smoke=True)
        run_longest_prompt_smoke(
            output / "checkpoint_final",
            output / "longest_prompt_metrics.json",
            f"smoke-{arm}-generation",
        )


def train_preference(arm: str, *, archive_partial: bool) -> None:
    validate_prepare_audit()
    validate_clean_training_result(ARM_RESULT_DIRS["clean_sft"], smoke=False)
    smoke_ids = set()
    for smoke_arm in ("continued_sft", "dpo"):
        smoke_dir = V4_RESULTS / "smoke_preference" / smoke_arm
        _, comparison_id = validate_preference_training_result(
            smoke_dir,
            smoke_arm,
            smoke=True,
        )
        smoke_ids.add(comparison_id)
        validate_longest_prompt_smoke(
            smoke_dir / "longest_prompt_metrics.json"
        )
    if len(smoke_ids) != 1:
        raise RuntimeError("preference smoke arms used different contracts")
    output = ARM_RESULT_DIRS[arm]
    if output.is_dir() and any(output.iterdir()):
        try:
            validate_preference_training_result(
                output,
                arm,
                smoke=False,
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            record_validated_completion(f"train-{arm}")
            print(
                f"formal {arm} artifacts already validate; skipping",
                flush=True,
            )
            return
    prepare_output_directory(
        output,
        f"formal {arm} output",
        archive_partial=archive_partial,
    )
    run_command(
        preference_command(
            arm,
            V4_PROCESSED / "preference" / "train_schedule.jsonl",
            PREFERENCE_SCHEDULE_SHA256,
            output,
            smoke=False,
        ),
        f"train-{arm}",
    )
    validate_preference_training_result(output, arm, smoke=False)


def evaluate_all() -> None:
    validate_prepare_audit()
    validate_clean_training_result(ARM_RESULT_DIRS["clean_sft"], smoke=False)
    formal_contract_ids = set()
    for arm in ("continued_sft", "dpo"):
        _, comparison_id = validate_preference_training_result(
            ARM_RESULT_DIRS[arm],
            arm,
            smoke=False,
        )
        formal_contract_ids.add(comparison_id)
    if len(formal_contract_ids) != 1:
        raise RuntimeError("formal continuation arms used different contracts")
    for arm in ("clean_sft", "continued_sft", "dpo"):
        result_dir = ARM_RESULT_DIRS[arm]
        require_checkpoint(result_dir / "checkpoint_final")
        metrics_path = result_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                validate_formal_metrics(result_dir)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                # A killed wrapper can leave complete resumable predictions
                # but no final receipt.  Re-enter the frozen evaluator below;
                # it validates every retained prediction before recomputing.
                pass
            else:
                record_validated_completion(f"evaluate-{arm}")
                continue
        command = [
            sys.executable,
            "scripts/evaluate_tool_actions_v4.py",
            "--test-file",
            str(TEST_FILE),
            "--adapter",
            str(result_dir / "checkpoint_final"),
            "--output",
            str(metrics_path),
        ]
        if (
            result_dir / "metrics.predictions.jsonl"
        ).exists() or (
            result_dir / "metrics.contract.json"
        ).exists():
            command.append("--resume")
        run_command(command, f"evaluate-{arm}")
        validate_formal_metrics(result_dir)


def score_all(*, archive_partial: bool) -> None:
    validate_prepare_audit()
    for arm in ("clean_sft", "continued_sft", "dpo"):
        validate_formal_metrics(ARM_RESULT_DIRS[arm])
    pair_file = V4_PROCESSED / "evaluation" / "test_preference_pairs.jsonl"
    for arm in ("clean_sft", "continued_sft", "dpo"):
        checkpoint = ARM_RESULT_DIRS[arm] / "checkpoint_final"
        require_checkpoint(checkpoint)
        output = V4_RESULTS / "pair_scores" / arm
        if (output / "metrics.json").is_file():
            try:
                validate_pair_score_result(output, arm)
            except Exception:
                if not archive_partial:
                    raise
            else:
                record_validated_completion(f"score-{arm}")
                continue
        prepare_output_directory(
            output,
            f"{arm} pair-score output",
            archive_partial=archive_partial,
        )
        run_command(
            [
                sys.executable,
                "scripts/train_preference_v4.py",
                "--mode",
                "score",
                "--score-adapter",
                str(checkpoint),
                "--pair-file",
                str(pair_file),
                "--expected-pair-file-sha256",
                TEST_PREFERENCE_PAIRS_SHA256,
                "--score-split",
                "test",
                "--output-dir",
                str(output),
            ],
            f"score-{arm}",
        )
        validate_pair_score_result(output, arm)


def ensure_standard_v3_reference() -> Path:
    destination = ROOT / "results" / "qlora_v3" / "constrained_recovery"
    filenames = (
        "metrics.json",
        "metrics.contract.json",
        "metrics.predictions.jsonl",
    )
    if all((destination / filename).is_file() for filename in filenames):
        for filename in filenames:
            if sha256_file(destination / filename) != V3_RESULT_SHA256[filename]:
                raise RuntimeError(
                    f"local Standard V3 reference hash drift: {filename}"
                )
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"partial V3 reference directory; do not mix artifacts: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "fetch", "origin", V3_RESULT_BRANCH],
        cwd=ROOT,
        check=True,
    )
    remote_commit = subprocess.run(
        ["git", "rev-parse", f"origin/{V3_RESULT_BRANCH}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote_commit != V3_RESULT_COMMIT:
        raise RuntimeError(
            f"Standard V3 result branch moved: {remote_commit} != "
            f"{V3_RESULT_COMMIT}"
        )
    for filename in filenames:
        spec = (
            f"{V3_RESULT_COMMIT}:{V3_ARTIFACT_PREFIX}/{filename}"
        )
        payload = subprocess.run(
            ["git", "show", spec],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        (destination / filename).write_bytes(payload)
        if sha256_file(destination / filename) != V3_RESULT_SHA256[filename]:
            raise RuntimeError(
                f"fetched Standard V3 reference hash drift: {filename}"
            )
    metrics = load_json(destination / "metrics.json")
    if (
        metrics.get("protocol") != "qlora_v3"
        or metrics.get("checkpoint_fingerprint") != V3_CHECKPOINT_FINGERPRINT
    ):
        raise RuntimeError("Standard V3 identity contract drift")
    return destination


def aggregate() -> None:
    validate_prepare_audit()
    for arm in ("clean_sft", "continued_sft", "dpo"):
        validate_formal_metrics(ARM_RESULT_DIRS[arm])
        validate_pair_score_result(V4_RESULTS / "pair_scores" / arm, arm)
    standard = ensure_standard_v3_reference()
    command = [
        sys.executable,
        "scripts/aggregate_qlora_v4.py",
        "--standard-v3-result-dir",
        str(standard),
        "--clean-sft-result-dir",
        str(ARM_RESULT_DIRS["clean_sft"]),
        "--continued-sft-result-dir",
        str(ARM_RESULT_DIRS["continued_sft"]),
        "--dpo-result-dir",
        str(ARM_RESULT_DIRS["dpo"]),
        "--test-outcomes",
        str(V4_PROCESSED / "evaluation" / "test_outcomes.jsonl"),
        "--test-preference-pairs",
        str(V4_PROCESSED / "evaluation" / "test_preference_pairs.jsonl"),
        "--output-dir",
        str(ROOT / "results" / "analysis_v4"),
    ]
    score_root = V4_RESULTS / "pair_scores"
    command.extend(
        [
            "--clean-sft-pair-score-dir",
            str(score_root / "clean_sft"),
            "--continued-sft-pair-score-dir",
            str(score_root / "continued_sft"),
            "--dpo-pair-score-dir",
            str(score_root / "dpo"),
        ]
    )
    run_command(command, "aggregate-v4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "audit",
            "smoke-clean",
            "train-clean-sft",
            "smoke-preference",
            "train-sft-long",
            "train-dpo",
            "evaluate",
            "score",
            "aggregate",
            "package",
        ),
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--archive-partial-output",
        action="store_true",
        help=(
            "After confirming no matching process is alive, preserve a "
            "non-empty failed train/smoke/score output under "
            "results/qlora_v4/failed_attempts and restart the stage."
        ),
    )
    args = parser.parse_args()

    require_tag = args.stage not in ("prepare", "audit")
    source_commit(require_tag=require_tag)

    if args.stage == "prepare":
        if args.data_dir is None:
            raise RuntimeError("--data-dir is required for prepare")
        prepare(args.data_dir, args.local_files_only)
    elif args.stage == "audit":
        print(json.dumps(validate_prepare_audit(), indent=2))
    elif args.stage == "smoke-clean":
        smoke_clean(archive_partial=args.archive_partial_output)
    elif args.stage == "train-clean-sft":
        train_clean(archive_partial=args.archive_partial_output)
    elif args.stage == "smoke-preference":
        smoke_preference(archive_partial=args.archive_partial_output)
    elif args.stage == "train-sft-long":
        train_preference(
            "continued_sft",
            archive_partial=args.archive_partial_output,
        )
    elif args.stage == "train-dpo":
        train_preference(
            "dpo",
            archive_partial=args.archive_partial_output,
        )
    elif args.stage == "evaluate":
        evaluate_all()
    elif args.stage == "score":
        score_all(archive_partial=args.archive_partial_output)
    elif args.stage == "aggregate":
        aggregate()
    elif args.stage == "package":
        run_command(
            [sys.executable, "scripts/package_qlora_v4_results.py"],
            "package-v4",
        )


if __name__ == "__main__":
    main()
