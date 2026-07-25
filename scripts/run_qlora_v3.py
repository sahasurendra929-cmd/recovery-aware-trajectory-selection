#!/usr/bin/env python3
"""Fail-closed single-GPU runner for the overnight V3 experiment.

V3 trains one ``constrained_recovery`` selection arm.  It deliberately calls
the frozen V2 trainer so that selection is the only changed scientific
variable, then uses the semantically identical V3 evaluator with safer resume
handling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUILD_PROTOCOL = "qlora_v3_constrained_recovery"
EVALUATION_PROTOCOL = "qlora_v3"
FROZEN_TRAIN_PROTOCOL = "qlora_v2"
ARM = "constrained_recovery"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20260722
MAX_PROMPT_TOKENS = 1664
MAX_COMPLETION_TOKENS = 384
MAX_SEQUENCE_TOKENS = 2048
VALIDATION_EXAMPLES = 392
TEST_EXAMPLES = 959
MICROBATCHES = 1088
GRADIENT_ACCUMULATION = 16
OPTIMIZER_STEPS = 68
EXPECTED_SOURCE_BUDGETS = {
    "gpt-4o-retail": 523_182,
    "sonnet-35-new-retail": 1_167_747,
}
EXPECTED_RAW_SHA256 = {
    "gpt-4o-retail.json": "df01707894836168ff0ec9616b0bf08f66c7e5afcf313e5fe4f7a2f5c2ec938b",
    "sonnet-35-new-retail.json": "0df526398e9d2720c32d340815cffb04fe8c4f8a61b1f4f84bf3bb558f760131",
}
EXPECTED_VALIDATION_SHA256 = "71f66c41394a50e3d992b7b860bb444774e7215ada3ad95e7d749b911916057a"
EXPECTED_TEST_SHA256 = "0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96"
EXPECTED_V3_TRACE_SHA256 = "a65bba64baf7c9a6e816e721b382511211aa9df6f5204e7c4cce74f78b992cc5"
EXPECTED_V3_TRAIN_SHA256 = "6b991fe03c7b79132438f8681dccee9e4fab2003a5859bd1abce26ba32ed046d"
EXPECTED_V3_SCHEDULE_SHA256 = "46acdd204d3dc213389af9b44ed6884031899a82615f8a9be47e024c30e2ea38"
EXPECTED_V3_STATS = {
    "trajectories": 167,
    "examples": 1069,
    "unique_tasks": 76,
    "selected_sft_tokens": 1_690_929,
    "recovery_targets": 102,
    "agent_initiated_targets": 20,
    "non_recovery_targets": 967,
    "failed_action_targets": 104,
    "unique_recovery_signatures": 26,
    "unique_agent_recovery_signatures": 15,
    "anchor_task_overlap": 63,
    "scheduled_loss_tokens": 36_599,
    "scheduled_target_tool_tvd_from_v2_random": 0.07444852941176469,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint_dir: Path) -> str:
    paths = (
        checkpoint_dir / "adapter_config.json",
        checkpoint_dir / "adapter_model.safetensors",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"incomplete adapter checkpoint: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def trace_set_fingerprint(trace_ids: list[str]) -> str:
    payload = "\n".join(sorted(trace_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing required JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing required JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
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
        raise RuntimeError(f"V3 contract drift for {label}: {observed!r} != {expected!r}")


def require_empty_directory(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"{label} directory is non-empty: {path}. Refusing to mix a new "
            "formal run with old or partial artifacts; choose a new directory."
        )


def validate_example_rows(rows: list[dict[str, Any]], label: str) -> set[str]:
    ids: set[str] = set()
    for index, row in enumerate(rows):
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            raise RuntimeError(f"{label}[{index}] lacks a string example_id")
        if example_id in ids:
            raise RuntimeError(f"duplicate {label} example_id: {example_id}")
        ids.add(example_id)
        prompt_tokens = row.get("prompt_tokens")
        completion_tokens = row.get("completion_tokens")
        sequence_tokens = row.get("sequence_tokens")
        if not all(isinstance(value, int) and value > 0 for value in (prompt_tokens, completion_tokens, sequence_tokens)):
            raise RuntimeError(f"invalid token counts for {label} example {example_id}")
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise RuntimeError(f"{label} prompt cap drift for {example_id}")
        if completion_tokens > MAX_COMPLETION_TOKENS:
            raise RuntimeError(f"{label} completion cap drift for {example_id}")
        if sequence_tokens != prompt_tokens + completion_tokens or sequence_tokens > MAX_SEQUENCE_TOKENS:
            raise RuntimeError(f"{label} sequence-token contract drift for {example_id}")
    return ids


def _tier_satisfied(stats: dict[str, Any], anchor: dict[str, Any], tier: dict[str, Any]) -> bool:
    if "min_examples" in tier and stats["examples"] < tier["min_examples"]:
        return False
    if "max_examples" in tier and stats["examples"] > tier["max_examples"]:
        return False
    if "min_unique_tasks" in tier and stats["unique_tasks"] < tier["min_unique_tasks"]:
        return False
    if "min_anchor_task_overlap" in tier and stats["anchor_task_overlap"] < tier["min_anchor_task_overlap"]:
        return False
    if "max_task_loss" in tier and stats["unique_tasks"] < anchor["unique_tasks"] - tier["max_task_loss"]:
        return False
    if "min_clean_ratio" in tier and stats["clean_target_ratio"] < tier["min_clean_ratio"]:
        return False
    if "min_clean_targets" in tier and stats["clean_targets"] < tier["min_clean_targets"]:
        return False
    if (
        "min_ordinary_targets" in tier
        and stats.get("ordinary_targets", stats.get("clean_targets", -1))
        < tier["min_ordinary_targets"]
    ):
        return False
    if "min_non_recovery_targets" in tier and stats["non_recovery_targets"] < tier["min_non_recovery_targets"]:
        return False
    if "max_tool_tvd" in tier and stats["target_tool_tvd_from_v2_random"] > tier["max_tool_tvd"]:
        return False
    if "max_target_tool_tvd" in tier and stats["target_tool_tvd_from_v2_random"] > tier["max_target_tool_tvd"]:
        return False
    if (
        "max_scheduled_target_tool_tvd" in tier
        and stats.get("scheduled_target_tool_tvd_from_v2_random", float("inf"))
        > tier["max_scheduled_target_tool_tvd"]
    ):
        return False
    if "min_recovery_targets" in tier and stats["recovery_targets"] < tier["min_recovery_targets"]:
        return False
    if "min_recovery_gain" in tier and stats["recovery_targets"] < anchor["recovery_targets"] + tier["min_recovery_gain"]:
        return False
    if "max_recovery_targets" in tier and stats["recovery_targets"] > tier["max_recovery_targets"]:
        return False
    if "min_agent_initiated_targets" in tier and stats["agent_initiated_targets"] < tier["min_agent_initiated_targets"]:
        return False
    if "min_agent_gain" in tier and stats["agent_initiated_targets"] < anchor["agent_initiated_targets"] + tier["min_agent_gain"]:
        return False
    if "min_anchor_trace_overlap_ratio" in tier:
        denominator = max(1, anchor["trajectories"])
        if stats["anchor_trace_overlap"] / denominator < tier["min_anchor_trace_overlap_ratio"]:
            return False
    if "min_anchor_token_overlap_ratio" in tier and stats.get("anchor_token_overlap_ratio", -1.0) < tier["min_anchor_token_overlap_ratio"]:
        return False
    if "min_anchor_token_overlap" in tier and stats.get("anchor_token_overlap_ratio", -1.0) < tier["min_anchor_token_overlap"]:
        return False
    if "max_loss_token_relative_delta" in tier:
        anchor_loss = anchor.get("scheduled_loss_tokens")
        selected_loss = stats.get("scheduled_loss_tokens")
        if not isinstance(anchor_loss, (int, float)) or not anchor_loss:
            return False
        if not isinstance(selected_loss, (int, float)):
            return False
        if abs(selected_loss - anchor_loss) / anchor_loss > tier["max_loss_token_relative_delta"]:
            return False
    if "max_unpaired_failed_action_excess" in tier and stats.get("unpaired_failed_action_excess", float("inf")) > tier["max_unpaired_failed_action_excess"]:
        return False
    return True


def load_and_validate_contract(
    processed_dir: Path,
    selection_dir: Path,
) -> dict[str, Any]:
    build_path = processed_dir / "build_summary.json"
    build = load_json(build_path)
    require_exact("build protocol", build.get("protocol"), BUILD_PROTOCOL)
    require_exact(
        "frozen training/evaluation protocol",
        build.get("training_and_evaluation_protocol"),
        "qlora_v2_frozen",
    )
    require_exact("seed", build.get("seed"), SEED)
    require_exact("model name", build.get("model", {}).get("name"), MODEL)
    require_exact(
        "resolved model revision",
        build.get("model", {}).get("resolved_revision"),
        MODEL_REVISION,
    )
    require_exact("max prompt tokens", build.get("context", {}).get("max_prompt_tokens"), MAX_PROMPT_TOKENS)
    require_exact(
        "max completion tokens",
        build.get("context", {}).get("max_completion_tokens"),
        MAX_COMPLETION_TOKENS,
    )
    require_exact(
        "max sequence tokens",
        build.get("context", {}).get("max_sequence_tokens"),
        MAX_SEQUENCE_TOKENS,
    )
    require_exact("selection arm", build.get("selection", {}).get("arm"), ARM)
    require_exact(
        "selection source budgets",
        build.get("selection", {}).get("source_token_budgets"),
        EXPECTED_SOURCE_BUDGETS,
    )
    require_exact(
        "selection total budget",
        build.get("selection", {}).get("total_token_budget"),
        sum(EXPECTED_SOURCE_BUDGETS.values()),
    )
    require_exact(
        "uses validation outcomes",
        build.get("selection", {}).get("uses_validation_outcomes"),
        False,
    )
    require_exact("uses test outcomes", build.get("selection", {}).get("uses_test_outcomes"), False)
    require_exact(
        "raw trajectory hashes",
        build.get("data", {}).get("raw_file_sha256"),
        EXPECTED_RAW_SHA256,
    )
    require_exact(
        "failed-action label retention",
        build.get("data", {}).get("failed_action_labels_removed"),
        False,
    )

    training = build.get("training", {})
    require_exact("training microbatches", training.get("microbatches"), MICROBATCHES)
    require_exact(
        "training gradient accumulation",
        training.get("gradient_accumulation"),
        GRADIENT_ACCUMULATION,
    )
    require_exact("training optimizer steps", training.get("optimizer_steps"), OPTIMIZER_STEPS)
    require_exact(
        "training padded length",
        training.get("pad_to_max_sequence_tokens"),
        MAX_SEQUENCE_TOKENS,
    )
    require_exact(
        "training padded tokens",
        training.get("padded_tokens"),
        MICROBATCHES * MAX_SEQUENCE_TOKENS,
    )
    require_exact(
        "validation examples",
        build.get("shared_splits", {}).get("validation_examples"),
        VALIDATION_EXAMPLES,
    )
    require_exact("test examples", build.get("shared_splits", {}).get("test_examples"), TEST_EXAMPLES)
    require_exact(
        "validation hash",
        build.get("hashes", {}).get("validation_jsonl"),
        EXPECTED_VALIDATION_SHA256,
    )
    require_exact("test hash", build.get("hashes", {}).get("test_jsonl"), EXPECTED_TEST_SHA256)

    validation_path = processed_dir / "shared" / "validation.jsonl"
    test_path = processed_dir / "shared" / "test.jsonl"
    train_path = processed_dir / ARM / "train.jsonl"
    schedule_path = processed_dir / ARM / "train_schedule.jsonl"
    require_exact("validation file hash", sha256_file(validation_path), EXPECTED_VALIDATION_SHA256)
    require_exact("test file hash", sha256_file(test_path), EXPECTED_TEST_SHA256)
    validation_rows = read_jsonl(validation_path)
    test_rows = read_jsonl(test_path)
    train_rows = read_jsonl(train_path)
    schedule_rows = read_jsonl(schedule_path)
    require_exact("validation row count", len(validation_rows), VALIDATION_EXAMPLES)
    require_exact("test row count", len(test_rows), TEST_EXAMPLES)
    validate_example_rows(validation_rows, "validation")
    validate_example_rows(test_rows, "test")
    train_ids = validate_example_rows(train_rows, "train")
    train_by_id = {row["example_id"]: row for row in train_rows}
    if not 1 <= len(train_rows) <= MICROBATCHES:
        raise RuntimeError(f"V3 train examples must be in [1, {MICROBATCHES}]; found {len(train_rows)}")
    require_exact("schedule row count", len(schedule_rows), MICROBATCHES)
    scheduled_ids: set[str] = set()
    for index, row in enumerate(schedule_rows):
        require_exact(f"schedule index {index}", row.get("schedule_index"), index)
        example_id = row.get("example_id")
        if example_id not in train_ids:
            raise RuntimeError(f"schedule contains foreign train example_id: {example_id}")
        scheduled_ids.add(example_id)
        matching = train_by_id[example_id]
        for key, value in matching.items():
            if row.get(key) != value:
                raise RuntimeError(f"schedule drift for {example_id}: field {key}")
    require_exact("scheduled unique example set", scheduled_ids, train_ids)
    train_tasks = {row["task_key"] for row in train_rows}
    validation_tasks = {row["task_key"] for row in validation_rows}
    test_tasks = {row["task_key"] for row in test_rows}
    if train_tasks & validation_tasks or train_tasks & test_tasks or validation_tasks & test_tasks:
        raise RuntimeError("task-group split leakage detected across train/validation/test")

    arm = build.get("arm", {})
    require_exact("arm name", arm.get("arm"), ARM)
    require_exact("arm examples", arm.get("examples"), len(train_rows))
    require_exact("scheduled microbatches", arm.get("scheduled_microbatches"), MICROBATCHES)
    require_exact("scheduled unique examples", arm.get("scheduled_unique_examples"), len(train_rows))
    require_exact(
        "selected SFT tokens",
        arm.get("selected_sft_tokens"),
        sum(EXPECTED_SOURCE_BUDGETS.values()),
    )
    require_exact("arm source token counts", arm.get("source_token_counts"), EXPECTED_SOURCE_BUDGETS)
    require_exact("train file hash", arm.get("train_file_sha256"), sha256_file(train_path))
    require_exact("schedule file hash", arm.get("train_schedule_sha256"), sha256_file(schedule_path))
    require_exact("frozen V3 train file hash", arm.get("train_file_sha256"), EXPECTED_V3_TRAIN_SHA256)
    require_exact(
        "frozen V3 schedule file hash",
        arm.get("train_schedule_sha256"),
        EXPECTED_V3_SCHEDULE_SHA256,
    )

    manifest_path = selection_dir / f"{ARM}_manifest.json"
    manifest = load_json(manifest_path)
    require_exact("manifest protocol", manifest.get("protocol"), BUILD_PROTOCOL)
    require_exact("manifest frozen protocol", manifest.get("training_and_evaluation_protocol"), "qlora_v2_frozen")
    require_exact("manifest seed", manifest.get("seed"), SEED)
    require_exact("manifest model", manifest.get("model"), MODEL)
    require_exact("manifest revision", manifest.get("resolved_model_revision"), MODEL_REVISION)
    require_exact("manifest source budgets", manifest.get("source_token_budgets"), EXPECTED_SOURCE_BUDGETS)
    require_exact("manifest raw trajectory hashes", manifest.get("raw_file_sha256"), EXPECTED_RAW_SHA256)
    require_exact(
        "manifest selected tokens",
        manifest.get("selected_sft_tokens"),
        sum(EXPECTED_SOURCE_BUDGETS.values()),
    )
    trace_ids = manifest.get("trace_ids")
    traces = manifest.get("traces")
    if not isinstance(trace_ids, list) or len(trace_ids) != len(set(trace_ids)):
        raise RuntimeError("manifest trace_ids must be a unique list")
    if not all(isinstance(trace_id, str) for trace_id in trace_ids):
        raise RuntimeError("manifest trace_ids must contain strings only")
    require_exact(
        "frozen V3 trace-set fingerprint",
        trace_set_fingerprint(trace_ids),
        EXPECTED_V3_TRACE_SHA256,
    )
    if not isinstance(traces, list) or {item.get("trace_id") for item in traces} != set(trace_ids):
        raise RuntimeError("manifest trace metadata does not match trace_ids")
    observed_source_tokens: dict[str, int] = {}
    for trace in traces:
        source = trace.get("source")
        cost = trace.get("sft_token_cost")
        if not isinstance(source, str) or not isinstance(cost, int) or cost <= 0:
            raise RuntimeError("invalid trace source/token metadata in selection manifest")
        observed_source_tokens[source] = observed_source_tokens.get(source, 0) + cost
    require_exact("manifest trace source quotas", observed_source_tokens, EXPECTED_SOURCE_BUDGETS)
    require_exact("manifest hash", build.get("hashes", {}).get("manifest_json"), sha256_file(manifest_path))

    audit = build.get("selection", {}).get("audit", {})
    require_exact("manifest/build selection audit", manifest.get("selection_audit"), audit)
    selected_tier = audit.get("selected_tier")
    selected_stats = audit.get("selected_stats")
    anchor_stats = audit.get("anchor_stats")
    tiers = build.get("selection", {}).get("constraint_tiers")
    if not isinstance(selected_tier, dict) or not isinstance(selected_stats, dict) or not isinstance(anchor_stats, dict):
        raise RuntimeError("V3 selection audit is incomplete")
    if not isinstance(tiers, list) or selected_tier not in tiers:
        raise RuntimeError("selected V3 constraint tier is not one of the predeclared tiers")
    require_exact("selected V3 tier", selected_tier.get("name"), "strict")
    require_exact("V3 fallback use", audit.get("fallback_used"), False)
    if not _tier_satisfied(selected_stats, anchor_stats, selected_tier):
        raise RuntimeError("selected V3 data no longer satisfies its recorded constraint tier")
    for key, expected in EXPECTED_V3_STATS.items():
        require_exact(f"frozen selected statistic {key}", selected_stats.get(key), expected)
        require_exact(f"frozen arm statistic {key}", arm.get(key), expected)
    require_exact(
        "frozen selected trace fingerprint",
        selected_stats.get("trace_set_sha256"),
        EXPECTED_V3_TRACE_SHA256,
    )
    require_exact(
        "frozen arm trace fingerprint",
        arm.get("trace_set_sha256"),
        EXPECTED_V3_TRACE_SHA256,
    )
    for label, observed, expected in (
        (
            "target-tool TVD",
            selected_stats.get("target_tool_tvd_from_v2_random"),
            0.06992296986364839,
        ),
        (
            "V2 Random token overlap",
            selected_stats.get("anchor_token_overlap_ratio"),
            0.6721890747630445,
        ),
    ):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(
                f"V3 contract drift for {label}: {observed!r} != {expected!r}"
            )
    require_exact(
        "selection recovery partition",
        selected_stats.get("agent_initiated_targets", 0)
        + selected_stats.get("user_assisted_targets", 0),
        selected_stats.get("recovery_targets"),
    )
    require_exact(
        "selection recovery/non-recovery partition",
        selected_stats.get("recovery_targets", 0)
        + selected_stats.get("non_recovery_targets", 0),
        selected_stats.get("examples"),
    )
    require_exact(
        "selection schedule exposure",
        selected_stats.get("scheduled_unique_examples"),
        selected_stats.get("examples"),
    )
    # These are the frozen V3 safety rails, independent of which predeclared
    # feasibility tier was selected.  Missing audit fields fail closed.
    frozen_selection_bounds = (
        ("examples", 1045, 1088),
        ("unique_tasks", 60, None),
        ("recovery_targets", 100, 139),
        ("agent_initiated_targets", 17, None),
        ("non_recovery_targets", 939, None),
    )
    for key, minimum, maximum in frozen_selection_bounds:
        value = selected_stats.get(key)
        if not isinstance(value, (int, float)) or value < minimum or (maximum is not None and value > maximum):
            raise RuntimeError(
                f"V3 frozen selection bound failed for {key}: "
                f"observed={value!r}, required=[{minimum}, {maximum}]"
            )
    tool_tvd = selected_stats.get("target_tool_tvd_from_v2_random")
    if not isinstance(tool_tvd, (int, float)) or tool_tvd > 0.10:
        raise RuntimeError(f"V3 target-tool TVD exceeds 0.10: {tool_tvd!r}")
    scheduled_tool_tvd = selected_stats.get(
        "scheduled_target_tool_tvd_from_v2_random"
    )
    if not isinstance(scheduled_tool_tvd, (int, float)) or scheduled_tool_tvd > 0.10:
        raise RuntimeError(
            "V3 scheduled target-tool TVD exceeds 0.10: "
            f"{scheduled_tool_tvd!r}"
        )
    token_overlap = selected_stats.get("anchor_token_overlap_ratio")
    if not isinstance(token_overlap, (int, float)) or token_overlap < 0.65:
        raise RuntimeError(
            f"V3 random-anchor token overlap is below 0.65 or missing: {token_overlap!r}"
        )
    require_exact("audit examples", selected_stats.get("examples"), len(train_rows))
    require_exact("audit selected tokens", selected_stats.get("selected_sft_tokens"), sum(EXPECTED_SOURCE_BUDGETS.values()))

    audit_result = {
        "protocol": BUILD_PROTOCOL,
        "validated_at_utc": utc_now(),
        "build_summary_sha256": sha256_file(build_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_sha256": sha256_file(train_path),
        "train_schedule_sha256": sha256_file(schedule_path),
        "validation_sha256": sha256_file(validation_path),
        "test_sha256": sha256_file(test_path),
        "train_examples": len(train_rows),
        "scheduled_microbatches": len(schedule_rows),
        "selected_tier": selected_tier.get("name"),
        "status": "PASS",
    }
    (processed_dir / "contract_audit.json").write_text(
        json.dumps(audit_result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"v3_contract_audit": audit_result}, indent=2), flush=True)
    return build


def cuda_preflight(output_root: Path) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"frozen V3 requires Python 3.11; found {sys.version.split()[0]}"
        )
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is missing. Install a CUDA build before starting V3."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Formal V3 cannot run on CPU or MPS.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen V3 contract requires CUDA bfloat16 support.")
    if torch.__version__.split("+", 1)[0] != "2.7.1" or torch.version.cuda != "12.8":
        raise RuntimeError(
            "frozen V3 requires torch 2.7.1 with CUDA 12.8; "
            f"found torch={torch.__version__}, cuda={torch.version.cuda}"
        )
    expected_packages = {
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "bitsandbytes": "0.46.0",
        "datasets": "3.6.0",
        "accelerate": "1.7.0",
        "sentencepiece": "0.2.0",
        "safetensors": "0.5.3",
        "numpy": "1.26.4",
    }
    observed_packages = {}
    for package, expected_version in expected_packages.items():
        try:
            observed_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"frozen V3 dependency is missing: {package}") from exc
        if observed_version != expected_version:
            raise RuntimeError(
                f"frozen V3 requires {package}=={expected_version}; "
                f"found {observed_version}"
            )
        observed_packages[package] = observed_version
    torch.cuda.empty_cache()
    free_bytes, total_visible_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    warning = None
    if free_bytes < 6 * 1024**3:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.2f} GiB CUDA memory is free. "
            "Close other GPU processes before the overnight run."
        )
    if free_bytes < 7 * 1024**3:
        warning = (
            f"only {free_bytes / 1024**3:.2f} GiB is free; an 8 GiB RTX 5060 "
            "run is safer after closing browsers and other GPU applications"
        )
    environment = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": properties.total_memory,
        "cuda_visible_total_bytes": total_visible_bytes,
        "cuda_free_bytes": free_bytes,
        "bf16_supported": True,
        "python": sys.version,
        "packages": observed_packages,
        "warning": warning,
        "checked_at_utc": utc_now(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight_environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"cuda_preflight": environment}, indent=2), flush=True)
    return environment


def _walk_nonfinite(value: Any, prefix: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(prefix or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            problems.extend(_walk_nonfinite(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            problems.extend(_walk_nonfinite(item, f"{prefix}[{index}]"))
    return problems


def validate_training_artifacts(
    formal_dir: Path,
    processed_dir: Path,
    build: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = formal_dir / "checkpoint_final"
    checkpoint_files = (checkpoint / "adapter_config.json", checkpoint / "adapter_model.safetensors")
    missing = [str(path) for path in checkpoint_files if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete V3 adapter checkpoint: {missing}")
    manifest = load_json(formal_dir / "run_manifest.json")
    expected_manifest = {
        "protocol": FROZEN_TRAIN_PROTOCOL,
        "objective": "completion_only_causal_language_model_cross_entropy",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_seq_len": MAX_SEQUENCE_TOKENS,
        "max_steps": OPTIMIZER_STEPS,
        "batch_size": 1,
        "grad_accum": GRADIENT_ACCUMULATION,
        "learning_rate": 1e-4,
        "smoke_test": False,
    }
    for key, value in expected_manifest.items():
        require_exact(f"formal training manifest {key}", manifest.get(key), value)
    require_exact(
        "formal train schedule hash",
        manifest.get("train_file_sha256"),
        build["arm"]["train_schedule_sha256"],
    )
    require_exact(
        "formal validation hash",
        manifest.get("validation_file_sha256"),
        EXPECTED_VALIDATION_SHA256,
    )
    metrics = load_json(formal_dir / "training_metrics.json")
    logs_value = json.loads((formal_dir / "training_log.json").read_text(encoding="utf-8"))
    if not isinstance(logs_value, list) or not logs_value:
        raise RuntimeError("training_log.json must contain a non-empty list")
    metrics_nonfinite = _walk_nonfinite(metrics)
    if metrics_nonfinite:
        raise RuntimeError(f"non-finite formal training metrics: {metrics_nonfinite}")
    objective_nonfinite = []
    diagnostic_nonfinite = []
    for index, entry in enumerate(logs_value):
        if not isinstance(entry, dict):
            raise RuntimeError(f"training log entry {index} is not an object")
        for key, value in entry.items():
            if isinstance(value, float) and not math.isfinite(value):
                target = objective_nonfinite if "loss" in key.lower() else diagnostic_nonfinite
                target.append(f"[{index}].{key}")
    if objective_nonfinite:
        raise RuntimeError(f"non-finite loss values in formal training log: {objective_nonfinite}")
    if not any("eval_loss" in entry for entry in logs_value):
        raise RuntimeError("formal training log lacks the frozen final validation loss")
    audit = {
        "status": "PASS",
        "validated_at_utc": utc_now(),
        "checkpoint_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in checkpoint_files
        },
        "run_manifest_sha256": sha256_file(formal_dir / "run_manifest.json"),
        "training_metrics_sha256": sha256_file(formal_dir / "training_metrics.json"),
        "training_log_sha256": sha256_file(formal_dir / "training_log.json"),
        "finite_training_metrics": True,
        "finite_loss_history": True,
        "nonfinite_nonobjective_diagnostics": diagnostic_nonfinite,
        "warning": (
            "non-finite diagnostic fields were recorded but all optimization "
            "and validation losses are finite"
            if diagnostic_nonfinite
            else None
        ),
    }
    (formal_dir / "training_artifact_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"training_artifact_audit": audit}, indent=2), flush=True)
    return audit


def validate_smoke_artifacts(
    smoke_dir: Path,
    build: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_files = (
        smoke_dir / "checkpoint_final" / "adapter_config.json",
        smoke_dir / "checkpoint_final" / "adapter_model.safetensors",
    )
    missing = [str(path) for path in checkpoint_files if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete V3 smoke-test checkpoint: {missing}")
    manifest = load_json(smoke_dir / "run_manifest.json")
    expected = {
        "protocol": FROZEN_TRAIN_PROTOCOL,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_seq_len": MAX_SEQUENCE_TOKENS,
        "max_steps": 1,
        "batch_size": 1,
        "grad_accum": 16,
        "smoke_test": True,
    }
    for key, value in expected.items():
        require_exact(f"smoke-test manifest {key}", manifest.get(key), value)
    require_exact(
        "smoke-test train schedule hash",
        manifest.get("train_file_sha256"),
        build["arm"]["train_schedule_sha256"],
    )
    require_exact(
        "smoke-test validation hash",
        manifest.get("validation_file_sha256"),
        EXPECTED_VALIDATION_SHA256,
    )
    metrics = load_json(smoke_dir / "training_metrics.json")
    if _walk_nonfinite(metrics):
        raise RuntimeError(f"non-finite smoke-test training metrics: {_walk_nonfinite(metrics)}")
    log_path = smoke_dir / "training_log.json"
    if not log_path.exists():
        raise RuntimeError(f"missing smoke-test training log: {log_path}")
    logs = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(logs, list) or not logs:
        raise RuntimeError("smoke-test training log is empty or invalid")
    nonfinite_loss = [
        f"[{index}].{key}"
        for index, entry in enumerate(logs)
        if isinstance(entry, dict)
        for key, value in entry.items()
        if "loss" in key.lower()
        and isinstance(value, (int, float))
        and not math.isfinite(float(value))
    ]
    if nonfinite_loss or not any("eval_loss" in entry for entry in logs if isinstance(entry, dict)):
        raise RuntimeError(
            f"smoke-test requires finite train/eval loss; nonfinite={nonfinite_loss}"
        )
    audit = {
        "status": "PASS",
        "checkpoint_nonempty": True,
        "finite_training_metrics": True,
        "finite_loss_history": True,
        "validated_at_utc": utc_now(),
    }
    (smoke_dir / "smoke_artifact_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"smoke_artifact_audit": audit}, indent=2), flush=True)
    return audit


def validate_longest_prompt_smoke(
    smoke_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    metrics_path = smoke_dir / "longest_prompt_metrics.json"
    contract_path = smoke_dir / "longest_prompt_metrics.contract.json"
    predictions_path = smoke_dir / "longest_prompt_metrics.predictions.jsonl"
    metrics = load_json(metrics_path)
    contract = load_json(contract_path)
    predictions = read_jsonl(predictions_path)
    test_rows = read_jsonl(processed_dir / "shared" / "test.jsonl")
    require_exact("longest-prompt smoke test rows", len(test_rows), TEST_EXAMPLES)
    longest = max(
        test_rows,
        key=lambda row: (
            row.get("prompt_tokens", -1),
            str(row.get("example_id", "")),
        ),
    )
    expected_contract = {
        "protocol": EVALUATION_PROTOCOL,
        "training_and_evaluation_protocol": "qlora_v2_frozen",
        "test_file_sha256": EXPECTED_TEST_SHA256,
        "formal_test_examples": TEST_EXAMPLES,
        "evaluated_examples": 1,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "base_model_loading": "nf4_4bit",
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "generation": {
            "do_sample": False,
            "max_new_tokens": 128,
            "batch_size": 1,
        },
        "limited": True,
        "smoke_selection": "longest_prompt",
        "smoke_forced_new_tokens": 128,
    }
    for key, value in expected_contract.items():
        require_exact(f"longest-prompt smoke contract {key}", contract.get(key), value)
        require_exact(f"longest-prompt smoke metrics {key}", metrics.get(key), value)
    expected_checkpoint = checkpoint_fingerprint(smoke_dir / "checkpoint_final")
    require_exact(
        "longest-prompt smoke contract checkpoint",
        contract.get("checkpoint_fingerprint"),
        expected_checkpoint,
    )
    require_exact(
        "longest-prompt smoke metrics checkpoint",
        metrics.get("checkpoint_fingerprint"),
        expected_checkpoint,
    )
    require_exact("longest-prompt smoke predictions", len(predictions), 1)
    require_exact(
        "longest-prompt smoke prediction example",
        predictions[0].get("example_id"),
        longest.get("example_id"),
    )
    require_exact(
        "longest-prompt smoke generated tokens",
        predictions[0].get("generated_token_count"),
        128,
    )
    require_exact(
        "longest-prompt smoke group size",
        metrics.get("groups", {}).get("overall", {}).get("examples"),
        1,
    )
    require_exact(
        "longest-prompt smoke predictions hash",
        metrics.get("predictions_sha256"),
        sha256_file(predictions_path),
    )
    if longest.get("prompt_tokens") != MAX_PROMPT_TOKENS:
        raise RuntimeError(
            "frozen V3 longest-prompt smoke no longer reaches the declared "
            f"{MAX_PROMPT_TOKENS}-token cap: {longest.get('prompt_tokens')!r}"
        )
    audit_path = smoke_dir / "smoke_artifact_audit.json"
    audit = load_json(audit_path)
    audit.update({
        "longest_prompt_evaluation": "PASS",
        "longest_prompt_example_id": longest["example_id"],
        "longest_prompt_tokens": longest["prompt_tokens"],
        "evaluation_loading": "nf4_4bit",
        "evaluation_max_new_tokens": 128,
        "evaluation_metrics_sha256": sha256_file(metrics_path),
        "evaluation_predictions_sha256": sha256_file(predictions_path),
    })
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"smoke_artifact_audit": audit}, indent=2), flush=True)
    return audit


class CommandRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict[str, Any]] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                self.records = existing
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def run(self, *parts: object) -> None:
        argv = list(map(str, parts))
        record = {
            "argv": argv,
            "shell_display": shlex.join(argv),
            "started_at_utc": utc_now(),
            "status": "running",
        }
        self.records.append(record)
        self._save()
        print("+", record["shell_display"], flush=True)
        try:
            completed = subprocess.run(argv, check=False)
            record["returncode"] = completed.returncode
            record["status"] = "complete" if completed.returncode == 0 else "failed"
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, argv)
        finally:
            record["finished_at_utc"] = utc_now()
            self._save()

    def note_invocation(self, argv: list[str]) -> None:
        self.records.append({
            "kind": "runner_invocation",
            "argv": argv,
            "shell_display": shlex.join(argv),
            "recorded_at_utc": utc_now(),
        })
        self._save()


def common_train_command(
    py: str,
    scripts: Path,
    processed_dir: Path,
    output_dir: Path,
) -> tuple[object, ...]:
    return (
        py,
        scripts / "train_qlora_v2.py",
        "--train-file",
        processed_dir / ARM / "train_schedule.jsonl",
        "--validation-file",
        processed_dir / "shared" / "validation.jsonl",
        "--output-dir",
        output_dir,
        "--model",
        MODEL,
        "--model-revision",
        MODEL_REVISION,
        "--seed",
        SEED,
        "--max-seq-len",
        MAX_SEQUENCE_TOKENS,
        "--max-steps",
        OPTIMIZER_STEPS,
        "--batch-size",
        1,
        "--grad-accum",
        GRADIENT_ACCUMULATION,
        "--learning-rate",
        1e-4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=(ARM,),
        default=ARM,
        help="V3 freezes a single new selection arm.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "audit", "smoke", "train", "evaluate", "aggregate", "overnight", "all"),
        default="overnight",
    )
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v3"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v3"))
    parser.add_argument("--output-root", type=Path, default=Path("results/qlora_v3"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis_v3"))
    parser.add_argument(
        "--v2-results-root",
        type=Path,
        default=Path("results/qlora_v2"),
        help="Directory containing random_success; absence produces standalone V3 analysis.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    require_exact("requested model", args.model, MODEL)
    require_exact("requested model revision", args.model_revision, MODEL_REVISION)

    scripts = Path(__file__).resolve().parent
    py = sys.executable
    recorder = CommandRecorder(args.output_root / "commands.json")
    recorder.note_invocation([py, str(Path(__file__).resolve()), *sys.argv[1:]])
    sequence = (
        ("prepare", "audit", "smoke", "train", "evaluate", "aggregate")
        if args.stage in ("overnight", "all")
        else (args.stage,)
    )
    build: dict[str, Any] | None = None
    formal_dir = args.output_root / ARM

    for stage in sequence:
        if stage == "prepare":
            require_empty_directory(args.selection_dir, "V3 selection")
            require_empty_directory(args.processed_dir, "V3 processed-data")
            prepare = [
                py,
                scripts / "prepare_qlora_v3.py",
                "--data-dir",
                args.data_dir,
                "--selection-dir",
                args.selection_dir,
                "--processed-dir",
                args.processed_dir,
                "--model",
                MODEL,
                "--model-revision",
                MODEL_REVISION,
            ]
            if args.local_files_only:
                prepare.append("--local-files-only")
            recorder.run(*prepare)
            build = load_and_validate_contract(args.processed_dir, args.selection_dir)
        elif stage == "audit":
            build = load_and_validate_contract(args.processed_dir, args.selection_dir)
        elif stage == "smoke":
            build = build or load_and_validate_contract(args.processed_dir, args.selection_dir)
            cuda_preflight(args.output_root)
            smoke_dir = args.output_root / "smoke" / ARM
            require_empty_directory(smoke_dir, "V3 smoke-test")
            recorder.run(
                *common_train_command(py, scripts, args.processed_dir, smoke_dir),
                "--smoke-test",
            )
            validate_smoke_artifacts(smoke_dir, build)
            recorder.run(
                py,
                scripts / "evaluate_tool_actions_v3.py",
                "--test-file",
                args.processed_dir / "shared" / "test.jsonl",
                "--adapter",
                smoke_dir / "checkpoint_final",
                "--output",
                smoke_dir / "longest_prompt_metrics.json",
                "--model",
                MODEL,
                "--model-revision",
                MODEL_REVISION,
                "--max-prompt-tokens",
                MAX_PROMPT_TOKENS,
                "--max-new-tokens",
                128,
                "--seed",
                SEED,
                "--limit",
                1,
                "--smoke-longest-prompt",
                "--resume",
            )
            validate_longest_prompt_smoke(smoke_dir, args.processed_dir)
        elif stage == "train":
            build = build or load_and_validate_contract(args.processed_dir, args.selection_dir)
            cuda_preflight(args.output_root)
            require_empty_directory(formal_dir, "V3 formal arm")
            recorder.run(*common_train_command(py, scripts, args.processed_dir, formal_dir))
            validate_training_artifacts(formal_dir, args.processed_dir, build)
        elif stage == "evaluate":
            build = build or load_and_validate_contract(args.processed_dir, args.selection_dir)
            cuda_preflight(args.output_root)
            validate_training_artifacts(formal_dir, args.processed_dir, build)
            recorder.run(
                py,
                scripts / "evaluate_tool_actions_v3.py",
                "--test-file",
                args.processed_dir / "shared" / "test.jsonl",
                "--adapter",
                formal_dir / "checkpoint_final",
                "--output",
                formal_dir / "metrics.json",
                "--model",
                MODEL,
                "--model-revision",
                MODEL_REVISION,
                "--max-prompt-tokens",
                MAX_PROMPT_TOKENS,
                "--max-new-tokens",
                128,
                "--seed",
                SEED,
                "--resume",
            )
        elif stage == "aggregate":
            build = build or load_and_validate_contract(args.processed_dir, args.selection_dir)
            aggregate = [
                py,
                scripts / "aggregate_qlora_v3.py",
                "--results-root",
                args.output_root,
                "--selection-dir",
                args.selection_dir,
                "--processed-dir",
                args.processed_dir,
                "--output-dir",
                args.analysis_dir,
            ]
            if args.v2_results_root:
                aggregate.extend(("--v2-results-root", args.v2_results_root))
            recorder.run(*aggregate)
        else:
            raise AssertionError(stage)

    result = {
        "experiment": EVALUATION_PROTOCOL,
        "selection_protocol": BUILD_PROTOCOL,
        "arm": ARM,
        "requested_stage": args.stage,
        "completed_stages": list(sequence),
        "output_root": str(args.output_root),
        "analysis_dir": str(args.analysis_dir),
        "commands_json": str(recorder.path),
        "completed_at_utc": utc_now(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
