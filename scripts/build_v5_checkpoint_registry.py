#!/usr/bin/env python3
"""Build an immutable V5 Stage-1 checkpoint registry.

The legacy, V5.3, and V5.3-12h profiles require the four canonical SFT arms.
The isolated low-support diagnostic profile deliberately requires only
``perfect_success`` and ``repair_50``.  Every profile remains fail-closed:
subsets and supersets of its exact arm set are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    import v5_3_low_support_protocol as low_support
except ModuleNotFoundError:
    from scripts import v5_3_low_support_protocol as low_support


PROTOCOL = "v5_stage1_checkpoint_registry"
TRAIN_PROTOCOL = "v5_stage1_message_masked_sft_7b"
DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ARMS = ("perfect_success", "failure_raw", "repair_50", "repair_100")
MODEL_IDS = {
    "base_model": "openai/v5-base",
    "perfect_success": "openai/v5-perfect-success",
    "failure_raw": "openai/v5-failure-raw",
    "repair_50": "openai/v5-repair-50",
    "repair_100": "openai/v5-repair-100",
}
V5_3_MODEL_IDS = {
    "base_model": "openai/v5-3-base",
    "perfect_success": "openai/v5-3-perfect-success",
    "failure_raw": "openai/v5-3-failure-raw",
    "repair_50": "openai/v5-3-repair-50",
    "repair_100": "openai/v5-3-repair-100",
}
V5_3_12H_MODEL_IDS = {
    "base_model": "openai/v5-3-12h-base",
    "perfect_success": "openai/v5-3-12h-perfect-success",
    "failure_raw": "openai/v5-3-12h-failure-raw",
    "repair_50": "openai/v5-3-12h-repair-50",
    "repair_100": "openai/v5-3-12h-repair-100",
}
V5_3_LOW_SUPPORT_MODEL_IDS = dict(low_support.MODEL_IDS)
V5_3_LOW_SUPPORT_PROFILE = low_support.REGISTRY_PROFILE
V5_3_LOW_SUPPORT_DESIGN_VERSION = low_support.DESIGN_VERSION
V5_3_LOW_SUPPORT_DESIGN_PROTOCOL = low_support.DATA_PROTOCOL
V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL = low_support.PROTOCOL
V5_3_LOW_SUPPORT_TRAINED_ARMS = tuple(low_support.TRAINED_ARMS)
V5_3_LOW_SUPPORT_USER_JUDGE = {
    "model": low_support.USER_JUDGE_MODEL,
    "model_id": low_support.USER_JUDGE_MODEL_ID,
    "revision": low_support.USER_JUDGE_REVISION,
    "roles": ["user_simulator", "strict_nl_judge"],
    "official_test_used": False,
}
V5_3_LOW_SUPPORT_CLAIM_BOUNDARY = dict(low_support.CLAIM_BOUNDARY)
ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PROFILES = (
    "legacy",
    "v5_3",
    "v5_3_12h_screen",
    V5_3_LOW_SUPPORT_PROFILE,
)
V5_3_DESIGN_VERSION = "5.3"
V5_3_DESIGN_PROTOCOL = "v5_3_task_level_cross_seed_sft_screen"
V5_3_12H_DESIGN_VERSION = "5.3-12h-screen"
V5_3_12H_DESIGN_PROTOCOL = "v5_3_12h_screen_sft_data_v1"
V5_3_12H_SCREEN_PROTOCOL = "v5_3_12h_exploratory_screen_v1"
V5_3_12H_TRAINING_SEED = 20260731
SCREEN_TARGET_RECOVERY_ROW_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_50": 0.5,
    "repair_100": 1.0,
}
SCREEN_LORA_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
PROFILE_GENERATION_INJECTIONS = {
    "legacy": 83,
    "v5_3": 78,
    "v5_3_12h_screen": 78,
    V5_3_LOW_SUPPORT_PROFILE: 78,
}
PROFILE_MODEL_IDS = {
    "legacy": MODEL_IDS,
    "v5_3": V5_3_MODEL_IDS,
    "v5_3_12h_screen": V5_3_12H_MODEL_IDS,
    V5_3_LOW_SUPPORT_PROFILE: V5_3_LOW_SUPPORT_MODEL_IDS,
}
PROFILE_TRAINED_ARMS = {
    "legacy": ARMS,
    "v5_3": ARMS,
    "v5_3_12h_screen": ARMS,
    V5_3_LOW_SUPPORT_PROFILE: V5_3_LOW_SUPPORT_TRAINED_ARMS,
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def trained_arms_for_profile(provenance_profile: str) -> tuple[str, ...]:
    try:
        return PROFILE_TRAINED_ARMS[provenance_profile]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported provenance profile {provenance_profile!r}"
        ) from error


def parse_arm_bindings(
    values: list[str],
    *,
    provenance_profile: str | None = None,
) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    resolved_paths: set[Path] = set()
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"--arm must be ARM=RUN_DIR, got {value!r}")
        arm, raw_path = value.split("=", 1)
        if arm not in ARMS:
            raise RuntimeError(f"unsupported Stage-1 arm {arm!r}")
        if arm in bindings:
            raise RuntimeError(f"duplicate arm binding: {arm}")
        if not raw_path:
            raise RuntimeError(f"{arm}: empty run directory")
        path = Path(raw_path).expanduser().resolve()
        if path in resolved_paths:
            raise RuntimeError(f"duplicate run directory: {path}")
        resolved_paths.add(path)
        bindings[arm] = path
    observed = set(bindings)
    if provenance_profile is None:
        registered_sets = {
            frozenset(ARMS),
            frozenset(V5_3_LOW_SUPPORT_TRAINED_ARMS),
        }
        if frozenset(observed) in registered_sets:
            return bindings
        expected_text = (
            f"{ARMS} or {V5_3_LOW_SUPPORT_TRAINED_ARMS}"
        )
        raise RuntimeError(
            "arm bindings must be exactly one registered trained-arm set; "
            f"expected={expected_text}, observed={sorted(observed)}"
        )
    expected = set(trained_arms_for_profile(provenance_profile))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"arm bindings for {provenance_profile} must be exactly "
            f"{tuple(sorted(expected))}; missing={missing}, extra={extra}"
        )
    return bindings


def _manifest_design_version(run_dir: Path) -> Any:
    manifest = load_json(run_dir / "run_manifest.json")
    provenance = manifest.get("data_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(
            f"{run_dir}: run manifest lacks data provenance"
        )
    return provenance.get("design_version")


def infer_provenance_profile(arm_dirs: dict[str, Path]) -> str:
    """Infer a profile only from an exact registered arm set and provenance."""

    observed_arms = set(arm_dirs)
    if observed_arms == set(V5_3_LOW_SUPPORT_TRAINED_ARMS):
        versions = {
            _manifest_design_version(Path(run_dir).expanduser().resolve())
            for run_dir in arm_dirs.values()
        }
        if versions == {V5_3_LOW_SUPPORT_DESIGN_VERSION}:
            return V5_3_LOW_SUPPORT_PROFILE
        raise RuntimeError(
            "two-arm registry inference requires only the frozen low-support "
            f"design_version; found={sorted(repr(value) for value in versions)}"
        )
    if observed_arms != set(ARMS):
        raise RuntimeError(
            "arm directories must be exactly one registered trained-arm set"
        )
    versions = {
        _manifest_design_version(Path(arm_dirs[arm]).expanduser().resolve())
        for arm in ARMS
    }
    if versions == {V5_3_DESIGN_VERSION}:
        return "v5_3"
    if versions == {V5_3_12H_DESIGN_VERSION}:
        return "v5_3_12h_screen"
    if versions <= {None, "5.2"}:
        return "legacy"
    raise RuntimeError(
        "cannot infer a registered provenance profile from design versions "
        f"{sorted(repr(value) for value in versions)}"
    )


def _validate_profile_provenance(
    provenance: dict[str, Any],
    *,
    provenance_profile: str,
) -> None:
    if provenance_profile not in PROVENANCE_PROFILES:
        raise RuntimeError(
            f"unsupported provenance profile {provenance_profile!r}"
        )
    design_version = provenance.get("design_version")
    if provenance_profile == "legacy":
        if design_version not in (None, "5.2"):
            raise RuntimeError(
                "legacy provenance profile requires absent or 5.2 "
                f"design_version, got {design_version!r}"
            )
        return
    if provenance_profile == "v5_3_12h_screen":
        if design_version != V5_3_12H_DESIGN_VERSION:
            raise RuntimeError(
                "V5.3-12h provenance profile requires screen design_version"
            )
        design = provenance.get("design_provenance")
        expected = {
            "design_version": V5_3_12H_DESIGN_VERSION,
            "design_protocol": V5_3_12H_DESIGN_PROTOCOL,
            "screen_protocol": V5_3_12H_SCREEN_PROTOCOL,
            "screen_tasks": 24,
            "screen_rollouts": 288,
            "validation_tasks": 21,
            "shared_outcome_free_protocol_inputs_read_only": True,
            "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused": False,
            "screen_outputs_may_enter_formal_v5_3": False,
        }
        if not isinstance(design, dict) or any(
            design.get(field) != value for field, value in expected.items()
        ):
            raise RuntimeError(
                "V5.3-12h provenance lacks the frozen isolated design identity"
            )
        return
    if provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        if design_version != V5_3_LOW_SUPPORT_DESIGN_VERSION:
            raise RuntimeError(
                "low-support diagnostic provenance requires its frozen "
                "design_version"
            )
        design = provenance.get("design_provenance")
        expected = {
            "design_version": V5_3_LOW_SUPPORT_DESIGN_VERSION,
            "design_protocol": V5_3_LOW_SUPPORT_DESIGN_PROTOCOL,
            "diagnostic_protocol": V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL,
            "validation_tasks": 21,
            "trained_arms": list(V5_3_LOW_SUPPORT_TRAINED_ARMS),
            "maximum_pairs_per_task": low_support.MAX_PAIRS_PER_TASK,
            "schedule_rows_per_arm": low_support.SCHEDULE_ROWS,
            "repeated_schedule_rows_are_independent_examples": False,
            "formal_v5_3_result": False,
            "official_test_used": False,
            "official_test_sealed": True,
        }
        if not isinstance(design, dict) or any(
            design.get(field) != value for field, value in expected.items()
        ):
            raise RuntimeError(
                "low-support diagnostic provenance lacks its frozen "
                "post-yield identity"
            )
        processing_commit = design.get("processing_source_commit")
        generation_commit = design.get("source_generation_commit")
        if (
            not isinstance(processing_commit, str)
            or COMMIT_RE.fullmatch(processing_commit) is None
            or not isinstance(generation_commit, str)
            or COMMIT_RE.fullmatch(generation_commit) is None
            or processing_commit == generation_commit
        ):
            raise RuntimeError(
                "low-support diagnostic source-commit provenance drift"
            )
        return
    if design_version != V5_3_DESIGN_VERSION:
        raise RuntimeError(
            "V5.3 provenance profile requires design_version '5.3'"
        )
    design = provenance.get("design_provenance")
    expected = {
        "design_version": V5_3_DESIGN_VERSION,
        "design_protocol": V5_3_DESIGN_PROTOCOL,
        "effective_generation_tasks": 78,
        "validation_tasks": 21,
    }
    if not isinstance(design, dict) or any(
        design.get(field) != value for field, value in expected.items()
    ):
        raise RuntimeError(
            "V5.3 provenance profile lacks the frozen 78/21 design identity"
        )


def _validate_screen_training_manifest(
    *,
    manifest: dict[str, Any],
    adapter_config: dict[str, Any],
    arm: str,
) -> None:
    """Enforce the exact fixed-64 screen training contract at registry time."""

    quantization = manifest.get("quantization")
    lora = manifest.get("lora")
    schedule = manifest.get("formal_schedule")
    loss = manifest.get("loss_audit")
    fit = manifest.get("fit_partition")
    expected_ratio = SCREEN_TARGET_RECOVERY_ROW_RATIOS[arm]
    expected_recovery_rows = int(512 * expected_ratio)
    scalar_contract = {
        "objective": "message_masked_causal_language_model_cross_entropy",
        "seed": V5_3_12H_TRAINING_SEED,
        "max_sequence_tokens": 8192,
        "truncation": False,
        "formal_steps": 64,
        "formal_batch_size": 1,
        "formal_grad_accum": 8,
        "effective_steps": 64,
        "effective_grad_accum": 8,
        "learning_rate": 1.0e-4,
        "effective_rows": 512,
        "effective_validation_rows": 0,
        "validation_disabled_reason": "fixed_steps_exploratory",
    }
    if any(manifest.get(key) != value for key, value in scalar_contract.items()):
        raise RuntimeError(f"{arm}: V5.3-12h fixed-64 training contract drift")
    if quantization != {
        "bits": 4,
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    }:
        raise RuntimeError(f"{arm}: V5.3-12h quantization contract drift")
    if (
        not isinstance(lora, dict)
        or lora.get("r") != 16
        or lora.get("alpha") != 32
        or lora.get("dropout") != 0.0
        or set(lora.get("target_modules") or [])
        != SCREEN_LORA_TARGET_MODULES
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("base_model_name_or_path") != BASE_MODEL
        or adapter_config.get("r") != 16
        or adapter_config.get("lora_alpha") != 32
        or adapter_config.get("lora_dropout") not in {0, 0.0}
        or set(adapter_config.get("target_modules") or [])
        != SCREEN_LORA_TARGET_MODULES
    ):
        raise RuntimeError(f"{arm}: V5.3-12h LoRA contract drift")
    if (
        not isinstance(schedule, dict)
        or schedule.get("rows") != 512
        or schedule.get("expected_recovery_supervised_token_ratio")
        is not None
        or schedule.get("recovery_mixture_basis")
        != "row_mean_microbatch_equal_weight"
        or schedule.get("recovery_rows") != expected_recovery_rows
        or schedule.get("expected_recovery_row_ratio") != expected_ratio
        or schedule.get("realized_recovery_row_ratio") != expected_ratio
        or isinstance(
            schedule.get("realized_recovery_supervised_token_ratio"), bool
        )
        or not isinstance(
            schedule.get("realized_recovery_supervised_token_ratio"),
            (int, float),
        )
        or not math.isfinite(
            float(schedule["realized_recovery_supervised_token_ratio"])
        )
        or not 0.0
        <= float(schedule["realized_recovery_supervised_token_ratio"])
        <= 1.0
        or not isinstance(schedule.get("supervised_tokens"), int)
        or schedule["supervised_tokens"] <= 0
        or not isinstance(schedule.get("nonpad_tokens"), int)
        or schedule["nonpad_tokens"] <= 0
        or not isinstance(schedule.get("max_sequence_tokens"), int)
        or not 0 < schedule["max_sequence_tokens"] <= 8192
        or (
            arm == "failure_raw"
            and (
                not isinstance(
                    schedule.get("failed_action_label_messages"), int
                )
                or schedule["failed_action_label_messages"] <= 0
            )
        )
        or (
            arm != "failure_raw"
            and schedule.get("failed_action_label_messages") != 0
        )
    ):
        raise RuntimeError(
            f"{arm}: V5.3-12h formal row-weighted schedule/ratio drift"
        )
    if (
        not isinstance(loss, dict)
        or loss.get("finite") is not True
        or not isinstance(loss.get("numeric_values_checked"), int)
        or loss["numeric_values_checked"] <= 0
        or not isinstance(loss.get("loss_values_checked"), int)
        or loss["loss_values_checked"] <= 0
        or not isinstance(loss.get("grad_norm_values_checked"), int)
        or loss["grad_norm_values_checked"] <= 0
        or loss.get("validation_loss_values_checked") != 0
        or loss.get("final_validation_loss") is not None
        or isinstance(loss.get("final_train_loss"), bool)
        or not isinstance(loss.get("final_train_loss"), (int, float))
        or not math.isfinite(float(loss["final_train_loss"]))
    ):
        raise RuntimeError(f"{arm}: V5.3-12h finite loss audit drift")
    if (
        not isinstance(fit, dict)
        or fit.get("validation_loss_source_examples") != 0
        or fit.get("overlap") != 0
        or fit.get("validation_disabled_reason")
        != "fixed_steps_exploratory"
    ):
        raise RuntimeError(f"{arm}: V5.3-12h no-validation partition drift")


def validate_run(
    *,
    arm: str,
    run_dir: Path,
    source_commit: str,
    base_revision: str,
    provenance_profile: str = "legacy",
) -> tuple[dict[str, str], dict[str, Any]]:
    if not run_dir.is_dir():
        raise RuntimeError(f"{arm}: formal run directory missing: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_dir = run_dir / "checkpoint_final"
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"{arm}: checkpoint_final directory missing")
    if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
        raise RuntimeError(f"{arm}: adapter_model.safetensors missing or empty")
    if (
        not adapter_config_path.is_file()
        or adapter_config_path.stat().st_size <= 0
    ):
        raise RuntimeError(f"{arm}: adapter_config.json missing or empty")
    # Parsing here prevents a byte-bound but unusable config from entering the
    # serving registry.
    adapter_config = load_json(adapter_config_path)
    if adapter_config.get("peft_type") != "LORA":
        raise RuntimeError(f"{arm}: adapter_config.json is not a LoRA adapter")
    manifest = load_json(manifest_path)
    expected = {
        "protocol": TRAIN_PROTOCOL,
        "mode": "formal",
        "arm": arm,
        "source_commit": source_commit,
        "model": BASE_MODEL,
        "model_revision": base_revision,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"{arm}: run_manifest {field} drift: "
                f"expected {value!r}, got {manifest.get(field)!r}"
            )
    if manifest.get("held_out_test_accessed") is not False:
        raise RuntimeError(f"{arm}: run manifest does not certify held-out seal")
    provenance = manifest.get("data_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{arm}: run manifest lacks data provenance")
    _validate_profile_provenance(
        provenance,
        provenance_profile=provenance_profile,
    )
    if (
        provenance_profile == V5_3_LOW_SUPPORT_PROFILE
        and provenance["design_provenance"].get(
            "processing_source_commit"
        )
        != source_commit
    ):
        raise RuntimeError(
            f"{arm}: low-support processing commit differs from run source"
        )
    if provenance_profile in {
        "v5_3_12h_screen",
        V5_3_LOW_SUPPORT_PROFILE,
    }:
        _validate_screen_training_manifest(
            manifest=manifest,
            adapter_config=adapter_config,
            arm=arm,
        )
    if (
        provenance.get("official_test_used") is not False
        or provenance.get("official_test_sealed") is not True
    ):
        raise RuntimeError(f"{arm}: data provenance lacks held-out seal")
    common_provenance: dict[str, Any] = {
        "data_audit_sha256": provenance.get("data_audit_sha256"),
        "data_hashes_sha256": provenance.get("data_hashes_sha256"),
        "dynamic_audits": provenance.get("dynamic_audits"),
        "official_test_used": False,
        "official_test_sealed": True,
    }
    if "design_version" in provenance:
        common_provenance["design_version"] = provenance["design_version"]
    if "design_provenance" in provenance:
        common_provenance["design_provenance"] = provenance[
            "design_provenance"
        ]
    if any(
        not isinstance(common_provenance[field], str)
        or SHA256_RE.fullmatch(common_provenance[field]) is None
        for field in ("data_audit_sha256", "data_hashes_sha256")
    ):
        raise RuntimeError(f"{arm}: data provenance SHA drift")
    dynamic = common_provenance["dynamic_audits"]
    if not isinstance(dynamic, dict) or set(dynamic) != {
        "generation",
        "validation",
    }:
        raise RuntimeError(f"{arm}: data provenance dynamic audits drift")
    for name, expected_count in (
        (
            "generation",
            PROFILE_GENERATION_INJECTIONS[provenance_profile],
        ),
        ("validation", 21),
    ):
        identity = dynamic.get(name)
        if (
            not isinstance(identity, dict)
            or identity.get("protocol") != DYNAMIC_AUDIT_PROTOCOL
            or identity.get("verified_injections") != expected_count
            or identity.get("official_test_used") is not False
            or identity.get("official_test_sealed") is not True
        ):
            raise RuntimeError(f"{arm}: {name} dynamic audit identity drift")
        for field in ("sha256", "manifest_sha256", "split_manifest_sha256"):
            value = identity.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RuntimeError(
                    f"{arm}: {name} dynamic audit {field} drift"
                )
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{arm}: run manifest lacks checkpoint metadata")
    recorded_files = checkpoint.get("file_sha256")
    if not isinstance(recorded_files, dict):
        raise RuntimeError(f"{arm}: run manifest lacks checkpoint file hashes")
    adapter_sha = sha256_file(adapter_path)
    adapter_config_sha = sha256_file(adapter_config_path)
    if recorded_files.get("adapter_model.safetensors") != adapter_sha:
        raise RuntimeError(f"{arm}: adapter SHA differs from run manifest")
    if recorded_files.get("adapter_config.json") != adapter_config_sha:
        raise RuntimeError(
            f"{arm}: adapter config SHA differs from run manifest"
        )
    manifest_sha = sha256_file(manifest_path)
    if any(
        SHA256_RE.fullmatch(value) is None
        for value in (adapter_sha, adapter_config_sha, manifest_sha)
    ):
        raise RuntimeError(f"{arm}: invalid computed SHA-256")
    return (
        {
            "model_id": PROFILE_MODEL_IDS[provenance_profile][arm],
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha,
            "training_run_manifest_sha256": manifest_sha,
        },
        common_provenance,
    )


def build_registry(
    *,
    source_commit: str,
    base_revision: str,
    arm_dirs: dict[str, Path],
    provenance_profile: str | None = None,
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("source commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(base_revision) is None:
        raise RuntimeError("base revision must be a full lowercase revision")
    if provenance_profile is None:
        provenance_profile = infer_provenance_profile(arm_dirs)
    elif provenance_profile not in PROVENANCE_PROFILES:
        raise RuntimeError(
            f"unsupported provenance profile {provenance_profile!r}"
        )
    trained_arms = trained_arms_for_profile(provenance_profile)
    if set(arm_dirs) != set(trained_arms):
        missing = sorted(set(trained_arms) - set(arm_dirs))
        extra = sorted(set(arm_dirs) - set(trained_arms))
        raise RuntimeError(
            f"arm directories for {provenance_profile} must be exactly "
            f"{trained_arms}; missing={missing}, extra={extra}"
        )
    model_ids_for_profile = PROFILE_MODEL_IDS[provenance_profile]
    entries: dict[str, Any] = {
        "base_model": {
            "model_id": model_ids_for_profile["base_model"],
            "adapter_sha256": None,
            "adapter_config_sha256": None,
            "training_run_manifest_sha256": None,
        }
    }
    resolved = [
        Path(arm_dirs[arm]).expanduser().resolve() for arm in trained_arms
    ]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("trained arms contain duplicate run directories")
    provenance_by_arm: dict[str, dict[str, Any]] = {}
    for arm, run_dir in zip(trained_arms, resolved):
        entries[arm], provenance_by_arm[arm] = validate_run(
            arm=arm,
            run_dir=run_dir,
            source_commit=source_commit,
            base_revision=base_revision,
            provenance_profile=provenance_profile,
        )
    serialized_provenance = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in provenance_by_arm.values()
    }
    if len(serialized_provenance) != 1:
        raise RuntimeError(
            "trained arms do not share one data/dynamic-audit provenance"
        )
    common_provenance = provenance_by_arm[trained_arms[0]]
    model_ids = [entry["model_id"] for entry in entries.values()]
    if len(set(model_ids)) != len(entries):
        raise RuntimeError("all registry model_id aliases must be unique")
    adapter_hashes = [
        entries[arm]["adapter_sha256"] for arm in trained_arms
    ]
    manifest_hashes = [
        entries[arm]["training_run_manifest_sha256"]
        for arm in trained_arms
    ]
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise RuntimeError("trained arms contain duplicate adapter bytes")
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise RuntimeError("trained arms contain duplicate run manifests")
    registry = {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "base_model_revision": base_revision,
        "provenance_profile": provenance_profile,
        "training_data_provenance": common_provenance,
        "entries": entries,
    }
    if provenance_profile == "v5_3_12h_screen":
        registry["screen_claim_boundary"] = {
            "exploratory_screen_only": True,
            "formal_v5_3_result": False,
            "screen_outputs_may_enter_formal_v5_3": False,
            "official_test_used": False,
            "official_test_sealed": True,
        }
        registry["screen_evaluator"] = {
            "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "model_id": "openai/v5-3-12h-user-judge",
            "revision": "539535859b135b0244c91f3e59816150c8056698",
            "roles": ["user_simulator", "strict_nl_judge"],
            "official_test_used": False,
        }
    elif provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        registry["diagnostic_claim_boundary"] = dict(
            V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
        )
        registry["diagnostic_evaluator"] = dict(
            V5_3_LOW_SUPPORT_USER_JUDGE
        )
    return registry


def atomic_write_registry(path: Path, registry: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument(
        "--provenance-profile",
        choices=PROVENANCE_PROFILES,
        help=(
            "Explicitly require legacy (83 generation injections, openai/v5-*) "
            "or V5.3 (78 generation injections, openai/v5-3-*), or the "
            "isolated V5.3-12h or low-support diagnostic profile. If omitted, "
            "derive only from the registered data_provenance.design_version."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help=(
            "Canonical ARM=FORMAL_RUN_DIR binding; provide the profile's "
            "exact registered trained-arm set."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bindings = parse_arm_bindings(
        args.arm,
        provenance_profile=args.provenance_profile,
    )
    registry = build_registry(
        source_commit=args.source_commit,
        base_revision=args.base_revision,
        arm_dirs=bindings,
        provenance_profile=args.provenance_profile,
    )
    output = args.output.expanduser().resolve()
    if (
        registry["provenance_profile"] == V5_3_LOW_SUPPORT_PROFILE
        and output
        != low_support.artifact_root(ROOT, "results_root")
        / "checkpoint_registry.json"
    ):
        raise RuntimeError(
            "low-support registry output is outside its isolated root"
        )
    if registry["provenance_profile"] == V5_3_LOW_SUPPORT_PROFILE:
        low_support.require_whole_run_source_lock(ROOT)
    atomic_write_registry(output, registry)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256_file(output),
                "model_ids": {
                    arm: entry["model_id"]
                    for arm, entry in registry["entries"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
