#!/usr/bin/env python3
"""Train one frozen V5 Stage-1 message-level SFT arm with 7B QLoRA.

The input contract is deliberately message structured.  Each JSONL row has
``messages`` and a parallel boolean ``label_mask``.  A true mask value is
allowed only for an assistant message.  The Qwen chat template is rendered
without truncation, and only the content/tool-call tokens (plus the assistant
end marker) of selected assistant messages enter causal-LM cross entropy.

This module keeps all GPU/ML imports inside ``main`` so its contract helpers
can be unit tested on a CPU-only coordinator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import v5_3_low_support_protocol as low_support
except ModuleNotFoundError:
    from scripts import v5_3_low_support_protocol as low_support


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260722
MAX_SEQUENCE_TOKENS = 8192
V5_5_MAX_SEQUENCE_TOKENS = 10240
V5_6_MAX_SEQUENCE_TOKENS = 10240
FORMAL_STEPS = 64
FORMAL_BATCH_SIZE = 1
FORMAL_GRAD_ACCUM = 8
FORMAL_SCHEDULE_ROWS = FORMAL_STEPS * FORMAL_BATCH_SIZE * FORMAL_GRAD_ACCUM
FORMAL_LEARNING_RATE = 1.0e-4
SMOKE_STEPS = 2
SMOKE_GRAD_ACCUM = 8
SMOKE_ROWS = SMOKE_STEPS * FORMAL_BATCH_SIZE * SMOKE_GRAD_ACCUM
RATIO_TOLERANCE = 0.01
MIN_GPU_MEMORY_BYTES = 20 * 1024**3
ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DATA_AUDIT_PROTOCOL = "v5_stage1_sft_causal"
DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
EXPECTED_DYNAMIC_AUDITS = {
    "generation": {
        "source_split": "inner_train",
        "verified_injections": 83,
    },
    "validation": {
        "source_split": "derived_validation",
        "verified_injections": 21,
    },
}
V5_3_DESIGN_VERSION = "5.3"
V5_3_DESIGN_PROTOCOL = "v5_3_task_level_cross_seed_sft_screen"
V5_3_12H_DESIGN_VERSION = "5.3-12h-screen"
V5_3_12H_DESIGN_PROTOCOL = "v5_3_12h_screen_sft_data_v1"
V5_3_12H_SCREEN_PROTOCOL = "v5_3_12h_exploratory_screen_v1"
V5_3_12H_TRAINING_SEED = 20260731
V5_3_12H_RECOVERY_ROW_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_50": 0.5,
    "repair_100": 1.0,
}
V5_3_LOW_SUPPORT_DESIGN_VERSION = low_support.DESIGN_VERSION
V5_3_LOW_SUPPORT_DESIGN_PROTOCOL = low_support.DATA_PROTOCOL
V5_3_LOW_SUPPORT_PROTOCOL = low_support.PROTOCOL
V5_3_LOW_SUPPORT_TRAINING_SEED = low_support.BASE_SEED
V5_3_LOW_SUPPORT_TRAINED_ARMS = tuple(low_support.TRAINED_ARMS)
V5_3_LOW_SUPPORT_RECOVERY_ROW_RATIOS = dict(
    low_support.ARM_RECOVERY_ROW_RATIOS
)
V5_3_LOW_SUPPORT_GATE_THRESHOLDS = {
    "minimum_distinct_tasks": low_support.MIN_DISTINCT_TASKS,
    "minimum_capped_pairs": low_support.MIN_CAPPED_PAIRS,
    "maximum_pairs_per_task": low_support.MAX_PAIRS_PER_TASK,
}
V5_3_LOW_SUPPORT_ARTIFACT_NAMESPACE = dict(
    low_support.ARTIFACT_NAMESPACE
)
V5_5_DESIGN_VERSION = "5.5-full"
V5_5_DESIGN_PROTOCOL = "v5_5_full_recovery_dose_response_v1"
V5_5_DATA_PROTOCOL = "v5_5_full_sft_data_v1"
V5_5_TRAINING_SEEDS = (20260805, 20260806, 20260807)
V5_6_DESIGN_VERSION = "5.6-context-mechanism"
V5_6_DESIGN_PROTOCOL = "v5_6_error_context_mechanism_v1"
V5_6_DATA_PROTOCOL = "v5_6_error_context_sft_data_v1"
V5_3_12H_TASK_IDS = (
    "airline:1",
    "airline:11",
    "airline:14",
    "airline:33",
    "airline:38",
    "airline:40",
    "retail:2",
    "retail:8",
    "retail:10",
    "retail:15",
    "retail:19",
    "retail:25",
    "retail:30",
    "retail:35",
    "retail:54",
    "retail:67",
    "retail:69",
    "retail:72",
    "retail:85",
    "retail:92",
    "retail:93",
    "retail:104",
    "retail:106",
    "retail:110",
)
V5_3_GENERATION_MANIFEST_PROTOCOL = "v5_3_multifault_data_construction"
V5_3_VALIDATION_MANIFEST_PROTOCOL = "v5_stage1_sft_causal_validation"
V5_3_GENERATION_SHARDS = 3
V5_3_GENERATION_TRIALS = 12
V5_3_GT_FILTER_PROTOCOL = "v5_3_gt_compatibility_filter_v1"
V5_3_GT_INCOMPATIBLE_TASK_IDS = (
    "airline:0",
    "airline:10",
    "airline:28",
    "airline:34",
    "retail:24",
)
V5_3_EXPECTED_DYNAMIC_AUDITS = {
    "generation": {
        "source_split": "derived_inner_train",
        "verified_injections": 78,
    },
    "validation": {
        "source_split": "derived_validation",
        "verified_injections": 21,
    },
}
V5_3_GT_FILTER = {
    "protocol": V5_3_GT_FILTER_PROTOCOL,
    "policy": "exclude_before_sharding",
    "teacher_mode": "ground_truth",
    "source_task_count": 83,
    "included_task_count": 78,
    "excluded_task_ids": list(V5_3_GT_INCOMPATIBLE_TASK_IDS),
    "exclusion_reason": "no_expected_tool_actions",
    "selection_uses_rollouts_rewards_validation_or_test": False,
    "official_test_used": False,
}

ARM_TARGET_RECOVERY_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_25": 0.25,
    "repair_25_true": 0.25,
    "repair_25_shuffled": 0.25,
    "repair_50": 0.50,
    "repair_75": 0.75,
    "repair_100": 1.0,
}
REPAIR_ARMS = {"repair_25", "repair_25_true", "repair_25_shuffled", "repair_50", "repair_75", "repair_100"}
ROW_KEYS = {"id", "messages", "label_mask", "metadata", "token_contract"}
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def validate_v5_3_design_provenance(
    *,
    audit: dict[str, Any],
    hashes: dict[str, Any],
    data_root: Path,
    dynamic_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind a V5.3 training run to its filtered manifests and raw contracts."""

    if audit.get("design_protocol") != V5_3_DESIGN_PROTOCOL:
        raise RuntimeError("V5.3 data audit design protocol drift")
    incompatible = audit.get("ground_truth_incompatible_task_ids")
    if incompatible != list(V5_3_GT_INCOMPATIBLE_TASK_IDS):
        raise RuntimeError("V5.3 data audit GT-incompatible task set drift")

    arm_train = audit.get("arm_train_task_ids")
    loss_validation = audit.get("loss_validation_task_ids")
    if (
        not isinstance(arm_train, list)
        or len(arm_train) != 70
        or len(set(arm_train)) != 70
        or any(not isinstance(value, str) for value in arm_train)
        or not isinstance(loss_validation, list)
        or len(loss_validation) != 8
        or len(set(loss_validation)) != 8
        or any(not isinstance(value, str) for value in loss_validation)
    ):
        raise RuntimeError("V5.3 70/8 train/loss-validation partition drift")
    included_tasks = set(arm_train) | set(loss_validation)
    if (
        set(arm_train) & set(loss_validation)
        or len(included_tasks) != 78
        or included_tasks & set(V5_3_GT_INCOMPATIBLE_TASK_IDS)
    ):
        raise RuntimeError("V5.3 effective 78-task universe drift")

    generation_sha = audit.get("generation_manifest_sha256")
    if not isinstance(generation_sha, str) or SHA256_RE.fullmatch(
        generation_sha
    ) is None:
        raise RuntimeError("V5.3 generation manifest SHA is invalid")
    generation_dynamic = dynamic_identities["generation"]
    if generation_dynamic.get("manifest_sha256") != generation_sha:
        raise RuntimeError(
            "V5.3 generation dynamic audit is not bound to generation manifest"
        )

    generation_contracts = audit.get("generation_contracts")
    if not isinstance(generation_contracts, dict):
        raise RuntimeError("V5.3 data audit lacks formal generation contracts")
    if (
        generation_contracts.get("generation_manifest_sha256") != generation_sha
        or generation_contracts.get("dynamic_audit_identity")
        != generation_dynamic
        or generation_contracts.get("task_union") != 78
        or generation_contracts.get("shards") != V5_3_GENERATION_SHARDS
    ):
        raise RuntimeError("V5.3 generation contract/manifest identity drift")
    contract_files = generation_contracts.get("contract_files")
    if (
        not isinstance(contract_files, dict)
        or len(contract_files) != V5_3_GENERATION_SHARDS
    ):
        raise RuntimeError("V5.3 requires exactly three generation contracts")

    observed_indices: set[int] = set()
    contract_tasks: set[str] = set()
    contract_hashes: dict[str, str] = {}
    for raw_path, declared_sha in sorted(contract_files.items()):
        if (
            not isinstance(raw_path, str)
            or not isinstance(declared_sha, str)
            or SHA256_RE.fullmatch(declared_sha) is None
        ):
            raise RuntimeError("V5.3 generation contract file identity is invalid")
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != declared_sha:
            raise RuntimeError(
                f"V5.3 generation contract file SHA drift: {path}"
            )
        contract = read_json_object(path, label="V5.3 generation contract")
        index = contract.get("shard_index")
        if (
            contract.get("protocol")
            != "v5_stage1_inner_train_generation_run"
            or contract.get("status") != "COMPLETE"
            or contract.get("generation_manifest_protocol")
            != V5_3_GENERATION_MANIFEST_PROTOCOL
            or contract.get("gt_compatibility_filter") != V5_3_GT_FILTER
            or contract.get("manifest_sha256") != generation_sha
            or contract.get("dynamic_audit_identity") != generation_dynamic
            or contract.get("source_split") != "derived_inner_train"
            or contract.get("official_test_used") is not False
            or contract.get("num_shards") != V5_3_GENERATION_SHARDS
            or contract.get("num_trials") != V5_3_GENERATION_TRIALS
            or type(index) is not int
            or index in observed_indices
        ):
            raise RuntimeError(
                f"V5.3 generation manifest protocol/contract drift: {path}"
            )
        observed_indices.add(index)
        task_ids = contract.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or any(not isinstance(value, str) for value in task_ids)
            or len(task_ids) != len(set(task_ids))
            or contract_tasks & set(task_ids)
        ):
            raise RuntimeError(f"V5.3 generation shard task identity drift: {path}")
        contract_tasks.update(task_ids)
        contract_hashes[str(path)] = declared_sha
    if (
        observed_indices != set(range(V5_3_GENERATION_SHARDS))
        or contract_tasks != included_tasks
    ):
        raise RuntimeError(
            "V5.3 generation contracts do not partition the frozen 78 tasks"
        )

    validation_path = data_root / "validation_manifest.json"
    declared_validation_sha = hashes.get("validation_manifest.json")
    if (
        not isinstance(declared_validation_sha, str)
        or SHA256_RE.fullmatch(declared_validation_sha) is None
        or not validation_path.is_file()
        or sha256_file(validation_path) != declared_validation_sha
    ):
        raise RuntimeError("V5.3 validation manifest/hash binding drift")
    validation_dynamic = dynamic_identities["validation"]
    if validation_dynamic.get("manifest_sha256") != declared_validation_sha:
        raise RuntimeError(
            "V5.3 validation dynamic audit is not bound to validation manifest"
        )
    validation_manifest = read_json_object(
        validation_path,
        label="V5.3 validation manifest",
    )
    validation_rows = validation_manifest.get("rows")
    if (
        validation_manifest.get("protocol")
        != V5_3_VALIDATION_MANIFEST_PROTOCOL
        or validation_manifest.get("paired_task_count") != 21
        or not isinstance(validation_rows, list)
        or len(validation_rows) != 21
        or validation_manifest.get("official_test_used") is not False
        or validation_manifest.get("official_test_sealed") is not True
    ):
        raise RuntimeError("V5.3 validation manifest protocol/coverage drift")
    validation_ids: set[str] = set()
    for row in validation_rows:
        if (
            not isinstance(row, dict)
            or row.get("source_split") != "derived_validation"
            or row.get("domain") not in {"retail", "airline"}
        ):
            raise RuntimeError("V5.3 validation manifest source split drift")
        identity = f"{row['domain']}:{row.get('task_id')}"
        if identity in validation_ids:
            raise RuntimeError("V5.3 validation manifest has duplicate tasks")
        validation_ids.add(identity)
    if len(validation_ids) != 21:
        raise RuntimeError("V5.3 validation manifest task coverage drift")

    return {
        "design_version": V5_3_DESIGN_VERSION,
        "design_protocol": V5_3_DESIGN_PROTOCOL,
        "generation_manifest_protocol": V5_3_GENERATION_MANIFEST_PROTOCOL,
        "generation_manifest_sha256": generation_sha,
        "generation_contract_sha256": contract_hashes,
        "ground_truth_incompatible_task_ids": list(
            V5_3_GT_INCOMPATIBLE_TASK_IDS
        ),
        "effective_generation_tasks": 78,
        "validation_manifest_protocol": V5_3_VALIDATION_MANIFEST_PROTOCOL,
        "validation_manifest_sha256": declared_validation_sha,
        "validation_tasks": 21,
    }


def validate_v5_3_12h_design_provenance(
    *,
    audit: dict[str, Any],
    hashes: dict[str, Any],
    data_root: Path,
    dynamic_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind training to the isolated 24-task/288-rollout screen only."""

    if (
        audit.get("design_protocol") != V5_3_12H_DESIGN_PROTOCOL
        or audit.get("screen_protocol") != V5_3_12H_SCREEN_PROTOCOL
        or audit.get("screen_task_ids") != list(V5_3_12H_TASK_IDS)
        or audit.get("attempts_per_task_per_condition") != 6
        or audit.get("expected_rollouts") != 288
        or audit.get("shared_outcome_free_protocol_inputs_read_only")
        is not True
        or audit.get(
            "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused"
        )
        is not False
        or audit.get("screen_outputs_may_enter_formal_v5_3") is not False
    ):
        raise RuntimeError("V5.3-12h screen design identity drift")
    gate = audit.get("data_gate")
    observed = gate.get("observed") if isinstance(gate, dict) else None
    thresholds = gate.get("thresholds") if isinstance(gate, dict) else None
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "PASS"
        or gate.get("training_authorized") is not True
        or not isinstance(observed, dict)
        or observed.get("tasks") != 24
        or observed.get("tasks_with_pair", 0) < 14
        or observed.get("capped_pairs", 0) < 17
        or observed.get("maximum_pairs_per_task") != 2
        or thresholds
        != {
            "minimum_tasks_with_pair": 14,
            "minimum_capped_pairs": 17,
            "maximum_pairs_per_task": 2,
        }
    ):
        raise RuntimeError("V5.3-12h screen data gate drift")
    arms = audit.get("arms")
    mixture = audit.get("arm_mixture_contract")
    if (
        not isinstance(arms, dict)
        or set(arms) != set(V5_3_12H_RECOVERY_ROW_RATIOS)
        or not isinstance(mixture, dict)
        or mixture.get("weight_unit") != "training_row"
        or mixture.get("target_recovery_row_ratios")
        != V5_3_12H_RECOVERY_ROW_RATIOS
        or mixture.get("target_miss_blocks_training") is not False
        or audit.get("cross_arm_token_budgets_gate_training") is not False
    ):
        raise RuntimeError("V5.3-12h row-weighted arm mixture drift")
    for arm, expected_ratio in V5_3_12H_RECOVERY_ROW_RATIOS.items():
        arm_row = arms.get(arm)
        expected_rows = int(FORMAL_SCHEDULE_ROWS * expected_ratio)
        if (
            not isinstance(arm_row, dict)
            or arm_row.get("rows") != FORMAL_SCHEDULE_ROWS
            or arm_row.get("recovery_rows") != expected_rows
            or arm_row.get("recovery_row_ratio") != expected_ratio
            or arm_row.get("expected_recovery_row_ratio") != expected_ratio
            or arm_row.get("recovery_mixture_basis")
            != "row_mean_microbatch_equal_weight"
        ):
            raise RuntimeError(
                f"V5.3-12h {arm} row-weighted arm mixture drift"
            )
    generation_contracts = audit.get("generation_contracts")
    if (
        not isinstance(generation_contracts, dict)
        or generation_contracts.get("protocol")
        != f"{V5_3_12H_SCREEN_PROTOCOL}:generation_contract_audit_v1"
        or generation_contracts.get("task_union") != 24
        or generation_contracts.get("shards") != 3
        or generation_contracts.get("official_test_used") is not False
    ):
        raise RuntimeError("V5.3-12h generation provenance drift")
    mapping_sha = generation_contracts.get(
        "strict_judge_evidence_mapping_sha256"
    )
    if (
        not isinstance(mapping_sha, str)
        or SHA256_RE.fullmatch(mapping_sha) is None
    ):
        raise RuntimeError("V5.3-12h strict-judge evidence is unbound")
    validation_path = data_root / "validation_manifest.json"
    validation_sha = hashes.get("validation_manifest.json")
    if (
        not isinstance(validation_sha, str)
        or SHA256_RE.fullmatch(validation_sha) is None
        or not validation_path.is_file()
        or sha256_file(validation_path) != validation_sha
        or audit.get("validation_manifest_sha256") != validation_sha
        or dynamic_identities["validation"].get("manifest_sha256")
        != validation_sha
    ):
        raise RuntimeError("V5.3-12h validation manifest binding drift")
    validation = read_json_object(
        validation_path,
        label="V5.3-12h validation manifest",
    )
    rows = validation.get("rows")
    if (
        validation.get("protocol") != V5_3_VALIDATION_MANIFEST_PROTOCOL
        or validation.get("paired_task_count") != 21
        or not isinstance(rows, list)
        or len(rows) != 21
        or validation.get("official_test_used") is not False
        or validation.get("official_test_sealed") is not True
        or any(
            not isinstance(row, dict)
            or row.get("source_split") != "derived_validation"
            for row in rows
        )
    ):
        raise RuntimeError("V5.3-12h derived-validation coverage drift")
    return {
        "design_version": V5_3_12H_DESIGN_VERSION,
        "design_protocol": V5_3_12H_DESIGN_PROTOCOL,
        "screen_protocol": V5_3_12H_SCREEN_PROTOCOL,
        "screen_tasks": 24,
        "screen_rollouts": 288,
        "validation_tasks": 21,
        "strict_judge_evidence_mapping_sha256": mapping_sha,
        "recovery_mixture_basis": "row_mean_microbatch_equal_weight",
        "target_recovery_row_ratios": V5_3_12H_RECOVERY_ROW_RATIOS,
        "cross_arm_token_budgets_gate_training": False,
        "shared_outcome_free_protocol_inputs_read_only": True,
        "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused": False,
        "screen_outputs_may_enter_formal_v5_3": False,
    }


def validate_v5_3_low_support_design_provenance(
    *,
    audit: dict[str, Any],
    hashes: dict[str, Any],
    data_root: Path,
    dynamic_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind training to the isolated post-yield low-support diagnostic."""

    if (
        audit.get("design_protocol")
        != V5_3_LOW_SUPPORT_DESIGN_PROTOCOL
        or audit.get("diagnostic_protocol")
        != V5_3_LOW_SUPPORT_PROTOCOL
        or audit.get("source_screen_protocol")
        != V5_3_12H_SCREEN_PROTOCOL
        or audit.get("screen_task_ids") != list(V5_3_12H_TASK_IDS)
        or audit.get("attempts_per_task_per_condition") != 6
        or audit.get("expected_rollouts") != 288
        or audit.get("post_yield_exploratory_diagnostic") is not True
        or audit.get("strict_source_gate_required_fail_closed") is not True
        or audit.get("artifact_namespace")
        != V5_3_LOW_SUPPORT_ARTIFACT_NAMESPACE
        or audit.get("diagnostic_outputs_may_enter_formal_v5_3")
        is not False
        or audit.get("screen_outputs_may_enter_formal_v5_3") is not False
        or audit.get("official_test_sealed") is not True
    ):
        raise RuntimeError("V5.3 low-support design identity drift")

    claim = audit.get("claim_boundary")
    if claim != low_support.CLAIM_BOUNDARY:
        raise RuntimeError("V5.3 low-support claim boundary drift")
    processing_source_commit = audit.get("processing_source_commit")
    source_generation_commit = audit.get("source_generation_commit")
    if (
        not isinstance(processing_source_commit, str)
        or COMMIT_RE.fullmatch(processing_source_commit) is None
        or not isinstance(source_generation_commit, str)
        or COMMIT_RE.fullmatch(source_generation_commit) is None
        or processing_source_commit == source_generation_commit
    ):
        raise RuntimeError("V5.3 low-support source-commit provenance drift")

    task_yield = audit.get("task_yield")
    if not isinstance(task_yield, dict) or set(task_yield) != set(
        low_support.PILOT_TASK_IDS
    ):
        raise RuntimeError("V5.3 low-support task-yield evidence drift")
    pair_counts: dict[str, int] = {}
    for task_id in low_support.PILOT_TASK_IDS:
        row = task_yield.get(task_id)
        count = row.get("selected_pairs") if isinstance(row, dict) else None
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= low_support.MAX_PAIRS_PER_TASK
        ):
            raise RuntimeError("V5.3 low-support task-yield evidence drift")
        pair_counts[task_id] = count
    recomputed_gate = low_support.data_gate(pair_counts)
    gate = audit.get("data_gate")
    if gate != recomputed_gate:
        raise RuntimeError("V5.3 low-support data gate drift")
    observed = gate.get("observed") if isinstance(gate, dict) else None
    strict = gate.get("source_strict_gate") if isinstance(gate, dict) else None
    strict_observed = (
        strict.get("observed") if isinstance(strict, dict) else None
    )
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "PASS_LOW_SUPPORT_DIAGNOSTIC"
        or gate.get("training_authorized") is not True
        or gate.get("thresholds") != V5_3_LOW_SUPPORT_GATE_THRESHOLDS
        or gate.get("claim_boundary") != low_support.CLAIM_BOUNDARY
        or not isinstance(observed, dict)
        or observed.get("tasks") != 24
        or observed.get("tasks_with_pair", 0) < 8
        or observed.get("capped_pairs", 0) < 10
        or observed.get("maximum_pairs_per_task") != 2
        or not isinstance(strict, dict)
        or strict.get("status") != "FAIL_CLOSED"
        or strict.get("training_authorized") is not False
        or not isinstance(strict_observed, dict)
        or strict_observed.get("tasks") != 24
        or strict_observed.get("tasks_with_pair")
        != observed.get("tasks_with_pair")
        or strict_observed.get("capped_pairs")
        != observed.get("capped_pairs")
    ):
        raise RuntimeError("V5.3 low-support data gate drift")

    arms = audit.get("arms")
    mixture = audit.get("arm_mixture_contract")
    if (
        not isinstance(arms, dict)
        or tuple(arms) != V5_3_LOW_SUPPORT_TRAINED_ARMS
        or not isinstance(mixture, dict)
        or mixture.get("weight_unit") != "training_row"
        or mixture.get("target_recovery_row_ratios")
        != V5_3_LOW_SUPPORT_RECOVERY_ROW_RATIOS
        or mixture.get("target_miss_blocks_training") is not False
        or audit.get("cross_arm_token_budgets_gate_training") is not False
        or audit.get("train_schedule_rows_per_arm") != FORMAL_SCHEDULE_ROWS
        or audit.get("core_task_id_multiset_equal") is not True
        or audit.get("common_eligible_pair_support") is not True
    ):
        raise RuntimeError("V5.3 low-support arm/matching contract drift")
    for arm, expected_ratio in V5_3_LOW_SUPPORT_RECOVERY_ROW_RATIOS.items():
        arm_row = arms.get(arm)
        expected_recovery_rows = int(FORMAL_SCHEDULE_ROWS * expected_ratio)
        if (
            not isinstance(arm_row, dict)
            or arm_row.get("rows") != FORMAL_SCHEDULE_ROWS
            or arm_row.get("recovery_rows") != expected_recovery_rows
            or arm_row.get("recovery_row_ratio") != expected_ratio
            or arm_row.get("expected_recovery_row_ratio") != expected_ratio
            or arm_row.get("recovery_mixture_basis")
            != "row_mean_microbatch_equal_weight"
            or arm_row.get("controlled_failed_action_labels") != 0
        ):
            raise RuntimeError(
                f"V5.3 low-support {arm} schedule contract drift"
            )

    generation_contracts = audit.get("generation_contracts")
    if (
        not isinstance(generation_contracts, dict)
        or generation_contracts.get("protocol")
        != f"{V5_3_12H_SCREEN_PROTOCOL}:generation_contract_audit_v1"
        or generation_contracts.get("task_union") != 24
        or generation_contracts.get("shards") != 3
        or generation_contracts.get("official_test_used") is not False
    ):
        raise RuntimeError("V5.3 low-support generation provenance drift")
    mapping_sha = generation_contracts.get(
        "strict_judge_evidence_mapping_sha256"
    )
    if (
        not isinstance(mapping_sha, str)
        or SHA256_RE.fullmatch(mapping_sha) is None
    ):
        raise RuntimeError(
            "V5.3 low-support strict-judge evidence is unbound"
        )

    validation_path = data_root / "validation_manifest.json"
    validation_sha = hashes.get("validation_manifest.json")
    if (
        not isinstance(validation_sha, str)
        or SHA256_RE.fullmatch(validation_sha) is None
        or not validation_path.is_file()
        or sha256_file(validation_path) != validation_sha
        or audit.get("validation_manifest_sha256") != validation_sha
        or dynamic_identities["validation"].get("manifest_sha256")
        != validation_sha
    ):
        raise RuntimeError(
            "V5.3 low-support validation manifest binding drift"
        )
    validation = read_json_object(
        validation_path,
        label="V5.3 low-support validation manifest",
    )
    rows = validation.get("rows")
    if (
        validation.get("protocol") != V5_3_VALIDATION_MANIFEST_PROTOCOL
        or validation.get("paired_task_count") != 21
        or not isinstance(rows, list)
        or len(rows) != 21
        or validation.get("official_test_used") is not False
        or validation.get("official_test_sealed") is not True
        or any(
            not isinstance(row, dict)
            or row.get("source_split") != "derived_validation"
            for row in rows
        )
    ):
        raise RuntimeError(
            "V5.3 low-support derived-validation coverage drift"
        )
    return {
        "design_version": V5_3_LOW_SUPPORT_DESIGN_VERSION,
        "design_protocol": V5_3_LOW_SUPPORT_DESIGN_PROTOCOL,
        "diagnostic_protocol": V5_3_LOW_SUPPORT_PROTOCOL,
        "source_screen_protocol": V5_3_12H_SCREEN_PROTOCOL,
        "screen_tasks": 24,
        "screen_rollouts": 288,
        "eligible_distinct_tasks": observed["tasks_with_pair"],
        "eligible_capped_pairs": observed["capped_pairs"],
        "maximum_pairs_per_task": low_support.MAX_PAIRS_PER_TASK,
        "schedule_rows_per_arm": FORMAL_SCHEDULE_ROWS,
        "repeated_schedule_rows_are_independent_examples": False,
        "validation_tasks": 21,
        "strict_judge_evidence_mapping_sha256": mapping_sha,
        "trained_arms": list(V5_3_LOW_SUPPORT_TRAINED_ARMS),
        "processing_source_commit": processing_source_commit,
        "source_generation_commit": source_generation_commit,
        "recovery_mixture_basis": "row_mean_microbatch_equal_weight",
        "target_recovery_row_ratios": (
            V5_3_LOW_SUPPORT_RECOVERY_ROW_RATIOS
        ),
        "formal_v5_3_result": False,
        "official_test_used": False,
        "official_test_sealed": True,
    }


def validate_training_data_provenance(
    *,
    arm: str,
    train_file: Path,
    validation_file: Path,
    data_audit_path: Path,
    data_hashes_path: Path,
    expected_train_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    """Bind one training run to the constructor's fail-closed audit bundle."""

    data_root = data_audit_path.resolve().parent
    if data_audit_path.resolve() != data_root / "audit.json":
        raise RuntimeError("--data-audit must name the constructor's audit.json")
    if data_hashes_path.resolve() != data_root / "hashes.json":
        raise RuntimeError(
            "--data-hashes must name hashes.json beside the constructor audit"
        )
    expected_train_path = data_root / "arms" / arm / "train.jsonl"
    expected_validation_path = data_root / "validation_loss.jsonl"
    if train_file.resolve() != expected_train_path:
        raise RuntimeError(
            f"--train-file must be the audited {arm} file: {expected_train_path}"
        )
    if validation_file.resolve() != expected_validation_path:
        raise RuntimeError(
            "--validation-file must be the audited validation_loss.jsonl: "
            f"{expected_validation_path}"
        )

    audit = read_json_object(data_audit_path, label="data audit")
    hashes = read_json_object(data_hashes_path, label="data hash manifest")
    if audit.get("protocol") == V5_5_DATA_PROTOCOL:
        return validate_v5_5_training_data_provenance(
            arm=arm,
            train_file=train_file,
            validation_file=validation_file,
            data_audit_path=data_audit_path,
            data_hashes_path=data_hashes_path,
            audit=audit,
            hashes=hashes,
            expected_train_sha256=expected_train_sha256,
            expected_validation_sha256=expected_validation_sha256,
        )
    if audit.get("protocol") == V5_6_DATA_PROTOCOL:
        return validate_v5_6_training_data_provenance(
            arm=arm,
            train_file=train_file,
            validation_file=validation_file,
            data_audit_path=data_audit_path,
            data_hashes_path=data_hashes_path,
            audit=audit,
            hashes=hashes,
            expected_train_sha256=expected_train_sha256,
            expected_validation_sha256=expected_validation_sha256,
        )
    if audit.get("status") != "PASS":
        raise RuntimeError("data audit status is not PASS")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL:
        raise RuntimeError("data audit protocol drift")
    if audit.get("official_test_used") is not False:
        raise RuntimeError("data audit does not seal the official test")
    if audit.get("derived_validation_used_for_supervision") is not False:
        raise RuntimeError("data audit used derived validation for supervision")
    if audit.get("train_validation_source_overlap") != 0:
        raise RuntimeError("data audit reports train/validation source leakage")
    label_guarantees = audit.get("label_guarantees")
    if (
        not isinstance(label_guarantees, dict)
        or label_guarantees.get(
            "official_test_and_derived_validation_label_leakage"
        )
        != 0
    ):
        raise RuntimeError("data audit lacks the frozen supervision-leakage seal")

    required_hash_keys = {
        "audit.json",
        f"arms/{arm}/train.jsonl",
        "validation_loss.jsonl",
    }
    design_version = audit.get("design_version")
    if design_version in {
        V5_3_DESIGN_VERSION,
        V5_3_12H_DESIGN_VERSION,
        V5_3_LOW_SUPPORT_DESIGN_VERSION,
    }:
        required_hash_keys.add("validation_manifest.json")
    for key in required_hash_keys:
        value = hashes.get(key)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"data hash manifest lacks valid {key} SHA-256")

    audit_sha = sha256_file(data_audit_path)
    hashes_sha = sha256_file(data_hashes_path)
    train_sha = sha256_file(train_file)
    validation_sha = sha256_file(validation_file)
    if hashes["audit.json"] != audit_sha:
        raise RuntimeError("data audit/hash manifest binding drift")
    if hashes[f"arms/{arm}/train.jsonl"] != train_sha:
        raise RuntimeError("audited arm/hash manifest binding drift")
    if hashes["validation_loss.jsonl"] != validation_sha:
        raise RuntimeError("validation-loss/hash manifest binding drift")
    if train_sha != expected_train_sha256:
        raise RuntimeError(f"training data hash drift: {train_sha}")
    if validation_sha != expected_validation_sha256:
        raise RuntimeError(f"validation data hash drift: {validation_sha}")

    arms = audit.get("arms")
    arm_audit = arms.get(arm) if isinstance(arms, dict) else None
    if not isinstance(arm_audit, dict) or arm_audit.get("sha256") != train_sha:
        raise RuntimeError("selected arm SHA is not bound into data audit")
    validation_audit = audit.get("validation_loss")
    expected_validation_source = (
        "not_applicable_fixed_step_screen"
        if design_version
        in {
            V5_3_12H_DESIGN_VERSION,
            V5_3_LOW_SUPPORT_DESIGN_VERSION,
        }
        else "inner_train"
    )
    if (
        not isinstance(validation_audit, dict)
        or validation_audit.get("sha256") != validation_sha
        or validation_audit.get("source_split")
        != expected_validation_source
    ):
        raise RuntimeError("validation-loss SHA/source is not bound into data audit")
    if design_version in {
        V5_3_12H_DESIGN_VERSION,
        V5_3_LOW_SUPPORT_DESIGN_VERSION,
    } and (
        validation_audit.get("rows") != 0
        or validation_audit.get("used_for_checkpoint_selection") is not False
        or validation_audit.get("validation_disabled_reason")
        != "fixed_steps_exploratory"
        or validation_file.stat().st_size != 0
    ):
        raise RuntimeError("screen no-eval validation contract drift")

    dynamic = audit.get("dynamic_audits")
    if not isinstance(dynamic, dict):
        raise RuntimeError("data audit lacks dynamic audit identities")
    dynamic_identities: dict[str, dict[str, Any]] = {}
    split_sha = audit.get("split_manifest_sha256")
    if not isinstance(split_sha, str) or SHA256_RE.fullmatch(split_sha) is None:
        raise RuntimeError("data audit split-manifest SHA is invalid")
    expected_dynamic_audits = (
        V5_3_EXPECTED_DYNAMIC_AUDITS
        if design_version
        in {
            V5_3_DESIGN_VERSION,
            V5_3_12H_DESIGN_VERSION,
            V5_3_LOW_SUPPORT_DESIGN_VERSION,
        }
        else EXPECTED_DYNAMIC_AUDITS
    )
    for name, expected in expected_dynamic_audits.items():
        identity = dynamic.get(name)
        if not isinstance(identity, dict):
            raise RuntimeError(f"data audit lacks {name} dynamic audit identity")
        if identity.get("protocol") != DYNAMIC_AUDIT_PROTOCOL:
            raise RuntimeError(f"{name} dynamic audit protocol drift")
        if identity.get("source_split") != expected["source_split"]:
            raise RuntimeError(f"{name} dynamic audit source split drift")
        if identity.get("verified_injections") != expected["verified_injections"]:
            raise RuntimeError(f"{name} dynamic audit coverage drift")
        if identity.get("official_test_used") is not False:
            raise RuntimeError(f"{name} dynamic audit opened official test")
        if identity.get("official_test_sealed") is not True:
            raise RuntimeError(f"{name} dynamic audit lacks official-test seal")
        if identity.get("split_manifest_sha256") != split_sha:
            raise RuntimeError(f"{name} dynamic audit split SHA drift")
        for field in ("sha256", "manifest_sha256", "split_manifest_sha256"):
            value = identity.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RuntimeError(
                    f"{name} dynamic audit identity lacks valid {field}"
                )
        dynamic_identities[name] = dict(identity)

    design_provenance = None
    if design_version == V5_3_DESIGN_VERSION:
        design_provenance = validate_v5_3_design_provenance(
            audit=audit,
            hashes=hashes,
            data_root=data_root,
            dynamic_identities=dynamic_identities,
        )
    elif design_version == V5_3_12H_DESIGN_VERSION:
        design_provenance = validate_v5_3_12h_design_provenance(
            audit=audit,
            hashes=hashes,
            data_root=data_root,
            dynamic_identities=dynamic_identities,
        )
    elif design_version == V5_3_LOW_SUPPORT_DESIGN_VERSION:
        design_provenance = validate_v5_3_low_support_design_provenance(
            audit=audit,
            hashes=hashes,
            data_root=data_root,
            dynamic_identities=dynamic_identities,
        )

    return {
        "data_audit_path": str(data_audit_path),
        "data_audit_sha256": audit_sha,
        "data_hashes_path": str(data_hashes_path),
        "data_hashes_sha256": hashes_sha,
        "train_file_sha256": train_sha,
        "validation_file_sha256": validation_sha,
        "dynamic_audits": dynamic_identities,
        "design_version": design_version,
        "design_provenance": design_provenance,
        "official_test_used": False,
        "official_test_sealed": True,
    }


def validate_v5_5_training_data_provenance(
    *,
    arm: str,
    train_file: Path,
    validation_file: Path,
    data_audit_path: Path,
    data_hashes_path: Path,
    audit: dict[str, Any],
    hashes: dict[str, Any],
    expected_train_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    """Validate the standalone V5.5 five-dose data bundle.

    V5.5 uses independently replayed counterfactual pairs instead of the V5.2
    generation/validation dynamic-injection manifests.  It therefore has a
    separate, smaller provenance graph and must never silently fall through to
    the legacy Stage-1 validator.
    """

    if (
        audit.get("status") != "PASS"
        or audit.get("protocol") != V5_5_DATA_PROTOCOL
        or audit.get("design_protocol") != V5_5_DESIGN_PROTOCOL
        or audit.get("design_version") != V5_5_DESIGN_VERSION
    ):
        raise RuntimeError("V5.5 data audit protocol/design/status drift")
    if (
        audit.get("official_test_used") is not False
        or audit.get("official_test_sealed") is not True
        or audit.get("derived_validation_used_for_supervision") is not False
        or audit.get("train_validation_source_overlap") != 0
        or audit.get("training_mixture_basis") != "supervised_token_mass"
        or audit.get("same_source_pair_exposure_across_arms") is not True
        or audit.get("same_supervised_target_within_clean_recovery_pair")
        is not True
    ):
        raise RuntimeError("V5.5 data audit leakage/budget guarantees drift")
    labels = audit.get("label_guarantees")
    if (
        not isinstance(labels, dict)
        or labels.get("failed_action_positive_labels") != 0
        or labels.get("official_test_and_derived_validation_label_leakage") != 0
        or labels.get("failed_call_and_result_are_context_only") is not True
        or labels.get("post_error_successful_assistant_actions_only") is not True
    ):
        raise RuntimeError("V5.5 data audit label guarantees drift")
    for field in (
        "pair_manifest_sha256",
        "pair_audit_sha256",
        "pairs_jsonl_sha256",
        "source_pairs_sha256",
    ):
        if not isinstance(audit.get(field), str) or SHA256_RE.fullmatch(
            audit[field]
        ) is None:
            raise RuntimeError(f"V5.5 audit lacks valid {field}")

    audit_sha = sha256_file(data_audit_path)
    hashes_sha = sha256_file(data_hashes_path)
    train_sha = sha256_file(train_file)
    validation_sha = sha256_file(validation_file)
    required = {
        "audit.json": audit_sha,
        f"arms/{arm}/train.jsonl": train_sha,
        "validation_loss.jsonl": validation_sha,
        "source_pairs.json": audit["source_pairs_sha256"],
    }
    for name, expected in required.items():
        observed = hashes.get(name)
        if observed != expected or not isinstance(observed, str):
            raise RuntimeError(f"V5.5 hash binding drift for {name}")
    if train_sha != expected_train_sha256:
        raise RuntimeError(f"training data hash drift: {train_sha}")
    if validation_sha != expected_validation_sha256:
        raise RuntimeError(f"validation data hash drift: {validation_sha}")

    arm_payload = (audit.get("arms") or {}).get(arm)
    expected_ratios = {
        "perfect_success": 0.0,
        "repair_25": 0.25,
        "repair_50": 0.50,
        "repair_75": 0.75,
        "repair_100": 1.0,
    }
    if arm not in expected_ratios or not isinstance(arm_payload, dict):
        raise RuntimeError("V5.5 selected arm is absent from the data audit")
    if (
        arm_payload.get("sha256") != train_sha
        or arm_payload.get("rows") != FORMAL_SCHEDULE_ROWS
        or arm_payload.get("failed_action_label_messages") != 0
        or abs(
            float(
                arm_payload.get(
                    "realized_recovery_supervised_token_ratio", -1.0
                )
            )
            - expected_ratios[arm]
        )
        > 1e-12
    ):
        raise RuntimeError("V5.5 selected arm audit drift")
    validation_payload = audit.get("validation_loss")
    if (
        not isinstance(validation_payload, dict)
        or validation_payload.get("rows") != 0
        or validation_payload.get("sha256") != validation_sha
        or validation_payload.get("source_split")
        != "not_applicable_fixed_step_screen"
        or validation_payload.get("used_for_checkpoint_selection") is not False
        or validation_payload.get("validation_disabled_reason")
        != "fixed_steps_v5_5"
        or validation_file.stat().st_size != 0
    ):
        raise RuntimeError("V5.5 fixed-step validation contract drift")
    input_authorization = audit.get("input_authorization")
    if (
        not isinstance(input_authorization, dict)
        or input_authorization.get("pair_mode") not in {"reference", "natural"}
        or input_authorization.get("scientific_claim_level")
        not in {
            "diagnostic_only",
            "natural_counterfactual_training_candidate",
        }
    ):
        raise RuntimeError("V5.5 input authorization/claim boundary drift")
    return {
        "data_audit_path": str(data_audit_path),
        "data_audit_sha256": audit_sha,
        "data_hashes_path": str(data_hashes_path),
        "data_hashes_sha256": hashes_sha,
        "train_file_sha256": train_sha,
        "validation_file_sha256": validation_sha,
        "dynamic_audits": {},
        "design_version": V5_5_DESIGN_VERSION,
        "design_provenance": {
            "design_version": V5_5_DESIGN_VERSION,
            "design_protocol": V5_5_DESIGN_PROTOCOL,
            "data_protocol": V5_5_DATA_PROTOCOL,
            "pair_mode": input_authorization["pair_mode"],
            "scientific_claim_level": input_authorization[
                "scientific_claim_level"
            ],
            "source_pairs": input_authorization["pairs"],
            "source_tasks": input_authorization["tasks"],
            "schedule_rows_per_arm": FORMAL_SCHEDULE_ROWS,
            "training_mixture_basis": "supervised_token_mass",
            "failed_action_positive_labels": 0,
            "official_test_used": False,
            "official_test_sealed": True,
        },
        "official_test_used": False,
        "official_test_sealed": True,
    }


def validate_v5_6_training_data_provenance(
    *, arm: str, train_file: Path, validation_file: Path,
    data_audit_path: Path, data_hashes_path: Path, audit: dict[str, Any],
    hashes: dict[str, Any], expected_train_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    """Bind V5.6's true-versus-shuffled context screen to its audit bundle."""
    expected_arms = {"perfect_success": 0.0, "repair_25_true": 0.25, "repair_25_shuffled": 0.25}
    if (
        audit.get("status") != "PASS" or audit.get("protocol") != V5_6_DATA_PROTOCOL
        or audit.get("design_protocol") != V5_6_DESIGN_PROTOCOL
        or audit.get("design_version") != V5_6_DESIGN_VERSION
        or audit.get("official_test_used") is not False
        or audit.get("official_test_sealed") is not True
        or audit.get("derived_validation_used_for_supervision") is not False
        or audit.get("train_validation_source_overlap") != 0
        or audit.get("training_mixture_basis") != "supervised_token_mass"
        or audit.get("same_source_pair_exposure_across_arms") is not True
        or audit.get("max_sequence_tokens") != V5_6_MAX_SEQUENCE_TOKENS
        or arm not in expected_arms
    ):
        raise RuntimeError("V5.6 data audit protocol/leakage/arm drift")
    labels = audit.get("label_guarantees")
    shuffle = audit.get("shuffle_contract")
    if (
        not isinstance(labels, dict) or labels.get("failed_action_positive_labels") != 0
        or labels.get("official_test_and_derived_validation_label_leakage") != 0
        or labels.get("failed_call_and_result_are_context_only") is not True
        or not isinstance(shuffle, dict) or shuffle.get("within_domain") is not True
        or shuffle.get("different_task") is not True
        or shuffle.get("error_call_and_result_moved_together") is not True
    ):
        raise RuntimeError("V5.6 label or shuffle contract drift")
    audit_sha, hashes_sha = sha256_file(data_audit_path), sha256_file(data_hashes_path)
    train_sha, validation_sha = sha256_file(train_file), sha256_file(validation_file)
    required = {"audit.json": audit_sha, f"arms/{arm}/train.jsonl": train_sha,
                "validation_loss.jsonl": validation_sha, "source_pairs.json": audit.get("source_pairs_sha256")}
    if any(hashes.get(name) != value for name, value in required.items()):
        raise RuntimeError("V5.6 hash binding drift")
    if train_sha != expected_train_sha256 or validation_sha != expected_validation_sha256:
        raise RuntimeError("V5.6 expected data hash drift")
    arm_audit = (audit.get("arms") or {}).get(arm)
    if (
        not isinstance(arm_audit, dict) or arm_audit.get("sha256") != train_sha
        or arm_audit.get("rows") != FORMAL_SCHEDULE_ROWS
        or not isinstance(arm_audit.get("max_sequence_tokens"), int)
        or arm_audit["max_sequence_tokens"] > V5_6_MAX_SEQUENCE_TOKENS
        or arm_audit.get("failed_action_label_messages") != 0
        or abs(float(arm_audit.get("realized_recovery_supervised_token_ratio", -1)) - expected_arms[arm]) > 1e-12
    ):
        raise RuntimeError("V5.6 selected-arm audit drift")
    if validation_file.stat().st_size != 0:
        raise RuntimeError("V5.6 fixed-step screen forbids loss-validation rows")
    return {
        "data_audit_path": str(data_audit_path), "data_audit_sha256": audit_sha,
        "data_hashes_path": str(data_hashes_path), "data_hashes_sha256": hashes_sha,
        "train_file_sha256": train_sha, "validation_file_sha256": validation_sha,
        "dynamic_audits": {}, "design_version": V5_6_DESIGN_VERSION,
        "design_provenance": {"design_version": V5_6_DESIGN_VERSION, "design_protocol": V5_6_DESIGN_PROTOCOL,
                              "data_protocol": V5_6_DATA_PROTOCOL, "schedule_rows_per_arm": FORMAL_SCHEDULE_ROWS,
                              "training_mixture_basis": "supervised_token_mass", "official_test_used": False},
        "official_test_used": False, "official_test_sealed": True,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"JSONL input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(
                    f"{path}:{line_number}: every JSONL record must end in newline"
                )
            if not line.strip():
                raise RuntimeError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: row must be an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise RuntimeError(f"{path}:{line_number}: missing non-empty id")
            if identifier in identifiers:
                raise RuntimeError(f"{path}:{line_number}: duplicate id {identifier!r}")
            identifiers.add(identifier)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"JSONL input is empty: {path}")
    return rows


def validate_fit_partition(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Reject loss-validation examples reused by the optimization schedule."""
    def source_ids(rows: list[dict[str, Any]], split: str) -> set[str]:
        identifiers: set[str] = set()
        for row in rows:
            metadata = row.get("metadata")
            source_id = metadata.get("source_example_id") if isinstance(metadata, dict) else None
            if not isinstance(source_id, str) or not source_id:
                raise RuntimeError(
                    f"{row.get('id', '<unknown>')}: {split} source_example_id is required"
                )
            identifiers.add(source_id)
        return identifiers

    train_sources = source_ids(train_rows, "train")
    validation_sources = source_ids(validation_rows, "validation_loss")
    overlap = train_sources & validation_sources
    if overlap:
        sample = sorted(overlap)[:3]
        raise RuntimeError(
            "train/validation-loss source leakage: "
            f"{len(overlap)} overlapping source_example_id values, e.g. {sample}"
        )
    return {
        "train_source_examples": len(train_sources),
        "validation_loss_source_examples": len(validation_sources),
        "overlap": 0,
    }


def _content_text(value: Any, *, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return canonical(value)
    raise RuntimeError(f"{where}: content must be string, object, list, or null")


def _normalize_tool_call(call: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise RuntimeError(f"{where}: tool call must be an object")
    function = call.get("function")
    if function is not None:
        if call.get("type", "function") != "function" or not isinstance(function, dict):
            raise RuntimeError(f"{where}: malformed standard function call")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"{where}: tool call requires a non-empty name")
    if isinstance(arguments, str):
        try:
            arguments_value = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{where}: arguments are not valid JSON") from error
    else:
        arguments_value = arguments
    if not isinstance(arguments_value, dict):
        raise RuntimeError(f"{where}: tool-call arguments must encode an object")
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            # Qwen2.5's chat template applies ``tojson`` itself.  Keeping an
            # object here avoids serializing arguments as a quoted JSON string.
            "arguments": arguments_value,
        },
    }
    call_id = call.get("id")
    if call_id is not None:
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError(f"{where}: tool-call id must be a non-empty string")
        normalized["id"] = call_id
    return normalized


def normalize_message(message: Any, *, where: str) -> dict[str, Any]:
    """Convert compact τ² or OpenAI-style messages to Qwen chat-template form."""
    if not isinstance(message, dict):
        raise RuntimeError(f"{where}: message must be an object")
    role = message.get("role")
    if role not in MESSAGE_ROLES:
        raise RuntimeError(f"{where}: unsupported role {role!r}")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _content_text(message.get("content"), where=where),
    }
    calls = message.get("tool_calls") or []
    if calls:
        if role != "assistant" or not isinstance(calls, list):
            raise RuntimeError(f"{where}: only assistant messages may contain tool_calls")
        if len(calls) != 1:
            raise RuntimeError(f"{where}: Stage-1 freezes exactly one tool call per turn")
        normalized["tool_calls"] = [
            _normalize_tool_call(calls[0], where=f"{where}.tool_calls[0]")
        ]
    if role == "assistant" and not normalized["content"] and not calls:
        raise RuntimeError(f"{where}: assistant message has no content or tool call")
    if role == "tool":
        tool_call_id = message.get("tool_call_id", message.get("id"))
        if tool_call_id is not None:
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise RuntimeError(f"{where}: tool-call link id must be a string")
            normalized["tool_call_id"] = tool_call_id
    return normalized


def _call_id(message: dict[str, Any]) -> str | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    value = call.get("id")
    return value if isinstance(value, str) else None


def failed_assistant_indices(messages: list[dict[str, Any]], *, row_id: str) -> list[int]:
    """Return assistant indices whose immediately linked tool result is an error."""
    failed: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        if index == 0 or messages[index - 1].get("role") != "assistant":
            raise RuntimeError(f"{row_id}: tool message {index} lacks adjacent assistant")
        assistant = messages[index - 1]
        if not assistant.get("tool_calls"):
            raise RuntimeError(f"{row_id}: tool message {index} has no adjacent tool call")
        assistant_id = _call_id(assistant)
        tool_id = message.get("tool_call_id", message.get("id"))
        if assistant_id is not None and tool_id is not None and assistant_id != tool_id:
            raise RuntimeError(
                f"{row_id}: tool-call id mismatch at messages {index - 1}/{index}"
            )
        if not isinstance(message.get("error"), bool):
            raise RuntimeError(f"{row_id}: tool message {index} needs boolean error")
        if message["error"]:
            failed.append(index - 1)
    return failed


def validate_row(
    row: dict[str, Any],
    *,
    arm: str | None,
    split: str,
) -> dict[str, Any]:
    """Validate the message/mask and arm semantics before tokenization."""
    unknown = set(row) - ROW_KEYS
    missing = {"id", "messages", "label_mask", "metadata"} - set(row)
    if unknown:
        raise RuntimeError(f"{row.get('id', '<unknown>')}: unknown row keys {sorted(unknown)}")
    if missing:
        raise RuntimeError(f"{row.get('id', '<unknown>')}: missing row keys {sorted(missing)}")
    row_id = row["id"]
    if not isinstance(row_id, str) or not row_id:
        raise RuntimeError(f"{split}: row id must be non-empty")
    messages = row["messages"]
    mask = row["label_mask"]
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"{row_id}: messages must be a non-empty list")
    if (
        not isinstance(mask, list)
        or len(mask) != len(messages)
        or any(type(value) is not bool for value in mask)
    ):
        raise RuntimeError(
            f"{row_id}: label_mask must contain one boolean per message"
        )
    if not isinstance(row["metadata"], dict):
        raise RuntimeError(f"{row_id}: metadata must be an object")
    normalized = [
        normalize_message(message, where=f"{row_id}.messages[{index}]")
        for index, message in enumerate(messages)
    ]
    selected = [index for index, value in enumerate(mask) if value]
    if not selected:
        raise RuntimeError(f"{row_id}: no supervised assistant message")
    for index in selected:
        if messages[index].get("role") != "assistant":
            raise RuntimeError(
                f"{row_id}: label_mask[{index}]=true for non-assistant role"
            )
        if not messages[index].get("tool_calls"):
            raise RuntimeError(
                f"{row_id}: Stage-1 supervises assistant tool calls only"
            )
    if selected[-1] + 1 != len(messages) - 1:
        raise RuntimeError(
            f"{row_id}: final supervised assistant must have exactly one "
            "adjacent evidence tool result and no later messages"
        )
    for index in selected:
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise RuntimeError(
                f"{row_id}: selected assistant tool call {index} lacks its "
                "adjacent evidence tool result"
            )
        assistant_id = _call_id(messages[index])
        tool_message = messages[index + 1]
        tool_id = tool_message.get("tool_call_id", tool_message.get("id"))
        if assistant_id is not None and tool_id is not None and assistant_id != tool_id:
            raise RuntimeError(
                f"{row_id}: selected tool-call/result id mismatch at "
                f"messages {index}/{index + 1}"
            )
        if not isinstance(tool_message.get("error"), bool):
            raise RuntimeError(
                f"{row_id}: selected tool result {index + 1} needs boolean error"
            )
    failed = failed_assistant_indices(messages, row_id=row_id)
    failed_set = set(failed)
    selected_set = set(selected)
    metadata = row["metadata"]
    if metadata.get("source_split") != "inner_train":
        raise RuntimeError(
            f"{row_id}: both training and loss-validation data must come from "
            "the inner_train split"
        )
    expected_fit_split = "train_schedule" if split == "train" else "validation_loss"
    if metadata.get("fit_split") != expected_fit_split:
        raise RuntimeError(
            f"{row_id}: metadata.fit_split must be {expected_fit_split!r}"
        )
    source_example_id = metadata.get("source_example_id")
    if not isinstance(source_example_id, str) or not source_example_id:
        raise RuntimeError(f"{row_id}: metadata.source_example_id is required")
    reward = metadata.get("full_trajectory_reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or float(reward) != 1.0
    ):
        raise RuntimeError(f"{row_id}: full_trajectory_reward must equal 1.0")
    trajectory_sha = metadata.get("source_trajectory_sha256")
    if not isinstance(trajectory_sha, str) or SHA256_RE.fullmatch(trajectory_sha) is None:
        raise RuntimeError(
            f"{row_id}: source_trajectory_sha256 must be lowercase SHA-256"
        )
    if metadata.get("official_test_used") is not False:
        raise RuntimeError(f"{row_id}: official_test_used must be false")
    metadata_arm = metadata.get("arm")
    if split == "train" and metadata_arm != arm:
        raise RuntimeError(
            f"{row_id}: metadata.arm={metadata_arm!r} does not match {arm!r}"
        )
    source = metadata.get("source")
    if source not in {"perfect_success", "failure_rich"}:
        raise RuntimeError(
            f"{row_id}: metadata.source must be perfect_success or failure_rich"
        )
    recorded_failed = metadata.get("failed_assistant_message_indices")
    if recorded_failed != failed:
        raise RuntimeError(
            f"{row_id}: failed_assistant_message_indices disagree with messages"
        )
    injected_failed = metadata.get("injected_failed_assistant_message_index")
    if failed:
        if len(failed) != 1 or injected_failed != failed[0]:
            raise RuntimeError(
                f"{row_id}: recovery row must contain exactly its one controlled "
                "injected failure"
            )
        if source != "failure_rich":
            raise RuntimeError(f"{row_id}: recovery messages need failure_rich source")
    else:
        if injected_failed is not None:
            raise RuntimeError(f"{row_id}: clean row records an injected failure")
        if source != "perfect_success":
            raise RuntimeError(f"{row_id}: clean messages need perfect_success source")
    tool_schemas = metadata.get("tool_schemas")
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise RuntimeError(f"{row_id}: metadata.tool_schemas must be a non-empty list")
    if any(not isinstance(schema, dict) for schema in tool_schemas):
        raise RuntimeError(f"{row_id}: every tool schema must be an object")
    tool_schema_hash = metadata.get("tool_schemas_sha256")
    observed_schema_hash = hashlib.sha256(
        canonical(tool_schemas).encode("utf-8")
    ).hexdigest()
    if tool_schema_hash != observed_schema_hash:
        raise RuntimeError(f"{row_id}: tool_schemas_sha256 mismatch")
    if messages[0].get("role") != "system":
        raise RuntimeError(f"{row_id}: first message must be the agent system policy")

    if split == "validation":
        if failed_set & selected_set:
            raise RuntimeError(
                f"{row_id}: validation may not positively supervise a failed action"
            )
        if failed and not any(index > failed[0] for index in selected):
            raise RuntimeError(f"{row_id}: loss-validation recovery has no repair label")
    elif arm == "perfect_success":
        if failed:
            raise RuntimeError(f"{row_id}: perfect_success contains a tool error")
    elif arm == "failure_raw":
        if not failed:
            raise RuntimeError(f"{row_id}: failure_raw row has no failed action")
        if not failed_set <= selected_set:
            raise RuntimeError(
                f"{row_id}: failure_raw must supervise every failed action"
            )
        if not any(index > failed[0] for index in selected):
            raise RuntimeError(f"{row_id}: failure_raw has no successful repair label")
    elif arm in REPAIR_ARMS:
        if failed_set & selected_set:
            raise RuntimeError(
                f"{row_id}: repair arm must mask every failed action"
            )
        if failed and not any(
            index > max(failed) and index in selected_set for index in range(len(messages))
        ):
            raise RuntimeError(
                f"{row_id}: recovery context lacks a supervised post-error repair"
            )
        if arm == "repair_100" and not failed:
            raise RuntimeError(f"{row_id}: repair_100 requires a recovery context")
    elif split == "train":
        raise RuntimeError(f"unsupported training arm: {arm!r}")
    selected_errors = {
        index for index in selected if messages[index + 1]["error"]
    }
    allowed_selected_errors = {failed[0]} if arm == "failure_raw" and failed else set()
    if selected_errors != allowed_selected_errors:
        raise RuntimeError(
            f"{row_id}: selected error labels {sorted(selected_errors)} differ "
            f"from allowed controlled failure {sorted(allowed_selected_errors)}"
        )

    token_contract = row.get("token_contract")
    if not isinstance(token_contract, dict):
        raise RuntimeError(f"{row_id}: token_contract must be an object")
    return {
        "id": row_id,
        "messages": normalized,
        "label_mask": list(mask),
        "metadata": dict(metadata),
        "failed_message_indices": failed,
        "is_recovery": bool(failed),
        "token_contract": token_contract,
    }


def _token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    generation: bool,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=generation,
    )
    if isinstance(result, dict):
        result = result.get("input_ids")
    if not isinstance(result, list) or any(type(value) is not int for value in result):
        raise RuntimeError("tokenizer chat template did not return a flat integer list")
    return result


def encode_row(
    tokenizer: Any,
    validated: dict[str, Any],
    *,
    max_seq_len: int = MAX_SEQUENCE_TOKENS,
) -> dict[str, Any]:
    """Render a row and map selected assistant messages to exact token spans."""
    row_id = validated["id"]
    messages = validated["messages"]
    mask = validated["label_mask"]
    tools = validated["metadata"]["tool_schemas"]
    full_ids = _token_ids(
        tokenizer,
        messages,
        tools=tools,
        generation=False,
    )
    if not full_ids:
        raise RuntimeError(f"{row_id}: chat template produced no tokens")
    if len(full_ids) > max_seq_len:
        raise RuntimeError(
            f"{row_id}: {len(full_ids)} tokens exceed {max_seq_len}; "
            "Stage-1 forbids truncation"
        )
    labels = [IGNORE_INDEX] * len(full_ids)
    spans: list[dict[str, int]] = []
    for message_index, selected in enumerate(mask):
        if not selected:
            continue
        before = _token_ids(
            tokenizer,
            messages[:message_index],
            tools=tools,
            generation=True,
        )
        through = _token_ids(
            tokenizer,
            messages[: message_index + 1],
            tools=tools,
            generation=False,
        )
        if len(before) >= len(through):
            raise RuntimeError(f"{row_id}: empty label span at message {message_index}")
        if through[: len(before)] != before:
            raise RuntimeError(
                f"{row_id}: chat template is not prefix-stable at message {message_index}"
            )
        if full_ids[: len(through)] != through:
            raise RuntimeError(
                f"{row_id}: future messages altered prefix tokenization at "
                f"message {message_index}"
            )
        start, end = len(before), len(through)
        if any(value != IGNORE_INDEX for value in labels[start:end]):
            raise RuntimeError(f"{row_id}: overlapping label spans")
        labels[start:end] = full_ids[start:end]
        spans.append(
            {
                "message_index": message_index,
                "token_start": start,
                "token_end": end,
            }
        )
    supervised_tokens = sum(value != IGNORE_INDEX for value in labels)
    if supervised_tokens <= 0:
        raise RuntimeError(f"{row_id}: tokenized row has no supervised tokens")
    contract = validated["token_contract"]
    required_contract = {"sequence_tokens", "supervised_tokens", "label_spans"}
    if set(contract) != required_contract:
        raise RuntimeError(
            f"{row_id}: token_contract keys must be {sorted(required_contract)}"
        )
    if contract["sequence_tokens"] != len(full_ids):
        raise RuntimeError(f"{row_id}: sequence token contract drift")
    if contract["supervised_tokens"] != supervised_tokens:
        raise RuntimeError(f"{row_id}: supervised token contract drift")
    if contract["label_spans"] != spans:
        raise RuntimeError(f"{row_id}: label-span contract drift")
    return {
        "id": row_id,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "label_spans": spans,
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised_tokens,
        "is_recovery": validated["is_recovery"],
        "failed_message_count": len(validated["failed_message_indices"]),
        "failed_label_message_count": len(
            set(validated["failed_message_indices"])
            & {index for index, value in enumerate(mask) if value}
        ),
        "failed_label_tokens": sum(
            span["token_end"] - span["token_start"]
            for span in spans
            if span["message_index"] in set(validated["failed_message_indices"])
        ),
    }


def encode_rows(
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    *,
    arm: str | None,
    split: str,
    max_seq_len: int = MAX_SEQUENCE_TOKENS,
) -> list[dict[str, Any]]:
    return [
        encode_row(
            tokenizer,
            validate_row(row, arm=arm, split=split),
            max_seq_len=max_seq_len,
        )
        for row in rows
    ]


def arm_audit(
    encoded_rows: list[dict[str, Any]],
    arm: str,
    *,
    recovery_mixture_basis: str = "supervised_token_mass",
) -> dict[str, Any]:
    loss_tokens = sum(row["supervised_tokens"] for row in encoded_rows)
    recovery_loss_tokens = sum(
        row["supervised_tokens"] for row in encoded_rows if row["is_recovery"]
    )
    if loss_tokens <= 0:
        raise RuntimeError("training schedule has zero supervised tokens")
    recovery_rows = sum(row["is_recovery"] for row in encoded_rows)
    observed_token_ratio = recovery_loss_tokens / loss_tokens
    observed_row_ratio = recovery_rows / len(encoded_rows)
    expected_ratio = ARM_TARGET_RECOVERY_RATIOS[arm]
    if recovery_mixture_basis == "supervised_token_mass":
        if abs(observed_token_ratio - expected_ratio) > RATIO_TOLERANCE:
            raise RuntimeError(
                f"{arm} recovery supervised-token ratio is "
                f"{observed_token_ratio:.6f}; expected {expected_ratio:.2f} "
                f"+/- {RATIO_TOLERANCE:.2f}"
            )
        expected_token_ratio: float | None = expected_ratio
        expected_row_ratio: float | None = None
    elif recovery_mixture_basis == "row_mean_microbatch_equal_weight":
        # With batch size one, mean-reduced causal loss, and fixed gradient
        # accumulation, every schedule row contributes one equally weighted
        # microbatch loss.  The arm dose must therefore be exact in rows.
        expected_rows = int(len(encoded_rows) * expected_ratio)
        if (
            len(encoded_rows) * expected_ratio != expected_rows
            or recovery_rows != expected_rows
        ):
            raise RuntimeError(
                f"{arm} recovery row ratio is {observed_row_ratio:.6f}; "
                f"expected exactly {expected_ratio:.2f} "
                f"({expected_rows}/{len(encoded_rows)} rows)"
            )
        expected_token_ratio = None
        expected_row_ratio = expected_ratio
    else:
        raise RuntimeError(
            f"unsupported recovery mixture basis: {recovery_mixture_basis}"
        )
    failed_labels = sum(row["failed_label_message_count"] for row in encoded_rows)
    failed_label_tokens = sum(row["failed_label_tokens"] for row in encoded_rows)
    if arm == "failure_raw" and failed_labels <= 0:
        raise RuntimeError("failure_raw produced no supervised failed-action label")
    if arm != "failure_raw" and failed_labels != 0:
        raise RuntimeError(f"{arm} unexpectedly supervises failed actions")
    return {
        "rows": len(encoded_rows),
        "nonpad_tokens": sum(row["sequence_tokens"] for row in encoded_rows),
        "supervised_tokens": loss_tokens,
        "recovery_rows": recovery_rows,
        "recovery_supervised_tokens": recovery_loss_tokens,
        "realized_recovery_supervised_token_ratio": observed_token_ratio,
        "expected_recovery_supervised_token_ratio": expected_token_ratio,
        "realized_recovery_row_ratio": observed_row_ratio,
        "expected_recovery_row_ratio": expected_row_ratio,
        "recovery_mixture_basis": recovery_mixture_basis,
        "failed_action_context_messages": sum(
            row["failed_message_count"] for row in encoded_rows
        ),
        "failed_action_label_messages": failed_labels,
        "failed_action_label_tokens": failed_label_tokens,
        "max_sequence_tokens": max(row["sequence_tokens"] for row in encoded_rows),
        "max_supervised_tokens": max(row["supervised_tokens"] for row in encoded_rows),
    }


def resolve_train_sampler_dataset(trainer: Any, train_dataset: Any = None) -> Any:
    return trainer.train_dataset if train_dataset is None else train_dataset


def tail_logit_contract(sequence_length: int, first_supervised: int) -> tuple[int, int]:
    """Return Qwen's tail-logit count and matching target start.

    To predict the label at position ``s``, causal LM needs the logit at
    position ``s - 1``.  Retaining ``L - s + 1`` final logits and dropping
    their last element therefore aligns exactly with ``labels[:, s:]``.
    """
    if sequence_length <= 1:
        raise ValueError("causal sequence must contain at least two tokens")
    if not 1 <= first_supervised < sequence_length:
        raise ValueError("first supervised token must follow context")
    return sequence_length - first_supervised + 1, first_supervised


def initialize_cuda_peak_tracking(torch_module: Any) -> int:
    device = 0
    torch_module.cuda.init()
    torch_module.cuda.set_device(device)
    torch_module.empty(1, device=f"cuda:{device}")
    torch_module.cuda.synchronize(device)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device)
    return device


def finite_training_audit(
    log_history: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    require_validation_loss: bool = True,
) -> dict[str, Any]:
    loss_values: list[float] = []
    grad_norms: list[float] = []
    eval_losses: list[float] = []
    checked = 0
    for record in [*log_history, metrics]:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"non-finite training metric {key}: {value!r}")
                checked += 1
                if "loss" in key.lower():
                    loss_values.append(numeric)
                if key == "grad_norm":
                    grad_norms.append(numeric)
                if key == "eval_loss":
                    eval_losses.append(numeric)
    if not loss_values:
        raise RuntimeError("training emitted no finite loss")
    if not grad_norms:
        raise RuntimeError("training emitted no finite grad_norm")
    if require_validation_loss and not eval_losses:
        raise RuntimeError("training emitted no finite validation loss")
    if not require_validation_loss and eval_losses:
        raise RuntimeError(
            "fixed-step screen unexpectedly emitted validation loss"
        )
    return {
        "finite": True,
        "numeric_values_checked": checked,
        "loss_values_checked": len(loss_values),
        "grad_norm_values_checked": len(grad_norms),
        "validation_loss_values_checked": len(eval_losses),
        "final_train_loss": float(metrics["train_loss"]),
        "final_validation_loss": eval_losses[-1] if eval_losses else None,
    }


def checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        target = path / filename
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"incomplete adapter checkpoint: {target}")
        file_hash = sha256_file(target)
        file_hashes[filename] = file_hash
        digest.update(filename.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), file_hashes


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_tracked_source() -> None:
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"repository contains {label} tracked source drift")


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_TARGET_RECOVERY_RATIOS), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--data-hashes", type=Path, required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        help=(
            "Required for V5.5 formal replications; must be one of "
            "20260805/20260806/20260807. Other designs retain their frozen seed."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=FORMAL_LEARNING_RATE,
        help=(
            "Frozen V5.5 rate by default; V5.5.2 preregisters 1.25e-5."
        ),
    )
    parser.add_argument(
        "--formal-steps",
        type=int,
        choices=(32, FORMAL_STEPS),
        default=FORMAL_STEPS,
        help=(
            "Frozen 64-step budget by default; V5.5.3 preregisters 32 steps "
            "with the V5.5.2 learning rate."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.expected_source_commit = args.expected_source_commit.lower()
    if COMMIT_RE.fullmatch(args.expected_source_commit) is None:
        raise RuntimeError("--expected-source-commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(args.model_revision) is None:
        raise RuntimeError("--model-revision must be a full lowercase model commit")
    if args.learning_rate not in {FORMAL_LEARNING_RATE, 1.25e-5}:
        raise RuntimeError(
            "--learning-rate must be the frozen V5.5 value or the "
            "preregistered V5.5.2 value"
        )
    if args.formal_steps == 32 and args.learning_rate != 1.25e-5:
        raise RuntimeError(
            "the preregistered 32-step V5.5.3 budget requires learning rate 1.25e-5"
        )
    expected_train_sha = validate_sha256(
        args.expected_train_sha256, "--expected-train-sha256"
    )
    expected_validation_sha = validate_sha256(
        args.expected_validation_sha256, "--expected-validation-sha256"
    )
    require_clean_tracked_source()
    source_commit = git_commit()
    if source_commit != args.expected_source_commit:
        raise RuntimeError(
            f"source commit drift: HEAD={source_commit}, expected={args.expected_source_commit}"
        )
    data_provenance = validate_training_data_provenance(
        arm=args.arm,
        train_file=args.train_file,
        validation_file=args.validation_file,
        data_audit_path=args.data_audit,
        data_hashes_path=args.data_hashes,
        expected_train_sha256=expected_train_sha,
        expected_validation_sha256=expected_validation_sha,
    )
    if (
        data_provenance.get("design_version")
        == V5_3_LOW_SUPPORT_DESIGN_VERSION
        and (data_provenance.get("design_provenance") or {}).get(
            "processing_source_commit"
        )
        != source_commit
    ):
        raise RuntimeError(
            "low-support data was processed by a different source commit"
        )
    if (
        data_provenance.get("design_version")
        == V5_3_LOW_SUPPORT_DESIGN_VERSION
    ):
        expected_data_root = low_support.artifact_root(
            ROOT, "processed_root"
        )
        expected_output_dir = (
            low_support.artifact_root(ROOT, "results_root")
            / "training"
            / args.arm
            / args.mode
        )
        if args.data_audit.resolve().parent != expected_data_root:
            raise RuntimeError(
                "low-support data audit is outside its isolated root"
            )
        if args.output_dir.resolve() != expected_output_dir:
            raise RuntimeError(
                "low-support training output is outside its isolated root"
            )
        low_support.require_whole_run_source_lock(ROOT)
    if data_provenance.get("design_version") in {
        V5_5_DESIGN_VERSION,
        V5_6_DESIGN_VERSION,
    }:
        if args.training_seed not in V5_5_TRAINING_SEEDS:
            raise RuntimeError(
                "V5.5/V5.6 requires --training-seed in "
                f"{V5_5_TRAINING_SEEDS}"
            )
        effective_seed = args.training_seed
    else:
        if args.training_seed is not None:
            raise RuntimeError(
                "--training-seed is reserved for the V5.5 replication grid"
            )
        effective_seed = (
            V5_3_LOW_SUPPORT_TRAINING_SEED
            if data_provenance.get("design_version")
            == V5_3_LOW_SUPPORT_DESIGN_VERSION
            else V5_3_12H_TRAINING_SEED
            if data_provenance.get("design_version")
            == V5_3_12H_DESIGN_VERSION
            else SEED
        )
    train_sha = data_provenance["train_file_sha256"]
    validation_sha = data_provenance["validation_file_sha256"]
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise RuntimeError(f"output directory must be absent or empty: {args.output_dir}")

    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or ':4096:8' before importing torch"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import torch.nn.functional as functional
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import SequentialSampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Stage-1 requires Python 3.12; found {sys.version.split()[0]}"
        )
    frozen_packages = {
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "bitsandbytes": "0.46.0",
        "datasets": "3.6.0",
        "accelerate": "1.7.0",
    }
    for package, expected in frozen_packages.items():
        observed = package_version(package)
        if observed != expected:
            raise RuntimeError(f"Stage-1 requires {package}=={expected}; found {observed}")
    if not str(torch.__version__).startswith("2.7.1"):
        raise RuntimeError(f"Stage-1 requires torch 2.7.1+cu128; found {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Stage-1 requires CUDA runtime 12.8; found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("V5 Stage-1 7B QLoRA requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V5 Stage-1 freezes BF16 compute, but BF16 is unavailable")
    if torch.cuda.get_device_properties(0).total_memory < MIN_GPU_MEMORY_BYTES:
        raise RuntimeError(
            "V5 Stage-1 7B/8192 contract requires a GPU with at least 20 GiB"
        )

    train_rows = read_jsonl(args.train_file)
    screen_no_eval = (
        data_provenance.get("design_version")
        in {
            V5_3_12H_DESIGN_VERSION,
            V5_3_LOW_SUPPORT_DESIGN_VERSION,
            V5_5_DESIGN_VERSION,
            V5_6_DESIGN_VERSION,
        }
    )
    no_eval_reason = (
        "fixed_steps_v5_6"
        if data_provenance.get("design_version") == V5_6_DESIGN_VERSION
        else "fixed_steps_v5_5"
        if data_provenance.get("design_version") == V5_5_DESIGN_VERSION
        else "fixed_steps_exploratory"
    )
    if screen_no_eval:
        if args.validation_file.read_bytes() != b"":
            raise RuntimeError("screen validation artifact must be zero bytes")
        validation_rows = []
        fit_partition_audit = {
            "train_source_examples": len(
                {
                    row["metadata"]["source_example_id"]
                    for row in train_rows
                }
            ),
            "validation_loss_source_examples": 0,
            "overlap": 0,
            "validation_disabled_reason": no_eval_reason,
        }
    else:
        validation_rows = read_jsonl(args.validation_file)
        fit_partition_audit = validate_fit_partition(
            train_rows, validation_rows
        )
    if len(train_rows) != FORMAL_SCHEDULE_ROWS:
        raise RuntimeError(
            f"formal schedule must contain exactly {FORMAL_SCHEDULE_ROWS} rows; "
            f"found {len(train_rows)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no usable pad token")
    tokenizer.padding_side = "right"

    effective_max_sequence_tokens = (
        V5_5_MAX_SEQUENCE_TOKENS
        if data_provenance.get("design_version") == V5_5_DESIGN_VERSION
        else V5_6_MAX_SEQUENCE_TOKENS
        if data_provenance.get("design_version") == V5_6_DESIGN_VERSION
        else MAX_SEQUENCE_TOKENS
    )
    formal_encoded = encode_rows(
        tokenizer,
        train_rows,
        arm=args.arm,
        split="train",
        max_seq_len=effective_max_sequence_tokens,
    )
    validation_encoded = (
        []
        if screen_no_eval
        else encode_rows(
            tokenizer,
            validation_rows,
            arm=None,
            split="validation",
            max_seq_len=effective_max_sequence_tokens,
        )
    )
    formal_arm_audit = arm_audit(
        formal_encoded,
        args.arm,
        recovery_mixture_basis=(
            "supervised_token_mass"
            if data_provenance.get("design_version") == V5_5_DESIGN_VERSION
            else "row_mean_microbatch_equal_weight"
            if screen_no_eval
            else "supervised_token_mass"
        ),
    )
    if args.mode == "smoke":
        encoded_rows = sorted(
            formal_encoded,
            key=lambda row: (row["sequence_tokens"], row["id"]),
            reverse=True,
        )[:SMOKE_ROWS]
        effective_steps = SMOKE_STEPS
        effective_grad_accum = SMOKE_GRAD_ACCUM
        effective_validation = (
            []
            if screen_no_eval
            else sorted(
                validation_encoded,
                key=lambda row: (row["sequence_tokens"], row["id"]),
                reverse=True,
            )[: min(16, len(validation_encoded))]
        )
    else:
        encoded_rows = formal_encoded
        effective_steps = args.formal_steps
        effective_grad_accum = FORMAL_GRAD_ACCUM
        effective_validation = validation_encoded

    set_seed(effective_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = initialize_cuda_peak_tracking(torch)

    def dataset_features(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                    "labels": row["labels"],
                }
                for row in rows
            ]
        )

    class DynamicCompletionCollator:
        def __call__(self, features):
            maximum = max(len(feature["input_ids"]) for feature in features)
            ids, attention, labels = [], [], []
            for feature in features:
                pad = maximum - len(feature["input_ids"])
                ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * pad)
                attention.append(feature["attention_mask"] + [0] * pad)
                labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(
                resolve_train_sampler_dataset(self, train_dataset)
            )

    class MaskedCausalTrainer(SequentialTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs["labels"]
            supervised = labels.ne(IGNORE_INDEX)
            if not bool(supervised.any()):
                raise RuntimeError("SFT microbatch has no supervised tokens")
            first_supervised = int(supervised.nonzero(as_tuple=False)[:, 1].min().item())
            sequence_length = labels.shape[1]
            logits_to_keep, target_start = tail_logit_contract(
                sequence_length,
                first_supervised,
            )
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
            prediction_logits = outputs.logits[:, :-1, :]
            targets = labels[:, target_start:]
            if prediction_logits.shape[:2] != targets.shape:
                raise RuntimeError(
                    "tail-logit alignment failure: "
                    f"{tuple(prediction_logits.shape[:2])} != {tuple(targets.shape)}"
                )
            loss = functional.cross_entropy(
                prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite SFT loss: {loss.detach().item()!r}")
            return (loss, outputs) if return_outputs else loss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        max_steps=effective_steps,
        per_device_train_batch_size=FORMAL_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no" if screen_no_eval else "steps",
        eval_steps=effective_steps,
        save_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=effective_seed,
        data_seed=effective_seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        label_names=["labels"],
    )
    trainer = MaskedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_features(encoded_rows),
        eval_dataset=(
            None
            if screen_no_eval
            else dataset_features(effective_validation)
        ),
        data_collator=DynamicCompletionCollator(),
    )
    result = trainer.train()
    loss_audit = finite_training_audit(
        trainer.state.log_history,
        result.metrics,
        require_validation_loss=not screen_no_eval,
    )
    checkpoint = args.output_dir / "checkpoint_final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    checkpoint_hash, checkpoint_files = checkpoint_fingerprint(checkpoint)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "bf16": True,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    manifest = {
        "protocol": "v5_stage1_message_masked_sft_7b",
        "source_commit": source_commit,
        "arm": args.arm,
        "mode": args.mode,
        "objective": "message_masked_causal_language_model_cross_entropy",
        "model": MODEL,
        "model_revision": args.model_revision,
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "seed": effective_seed,
        "max_sequence_tokens": effective_max_sequence_tokens,
        "truncation": False,
        "formal_steps": args.formal_steps,
        "formal_batch_size": FORMAL_BATCH_SIZE,
        "formal_grad_accum": FORMAL_GRAD_ACCUM,
        "effective_steps": effective_steps,
        "effective_grad_accum": effective_grad_accum,
        "learning_rate": args.learning_rate,
        "train_file": str(args.train_file),
        "train_file_sha256": train_sha,
        "validation_file": str(args.validation_file),
        "validation_file_sha256": validation_sha,
        "data_provenance": data_provenance,
        "formal_schedule": formal_arm_audit,
        "fit_partition": fit_partition_audit,
        "effective_rows": len(encoded_rows),
        "effective_validation_rows": len(effective_validation),
        "loss_audit": loss_audit,
        "held_out_test_accessed": False,
        "checkpoint": {
            "path": str(checkpoint),
            "fingerprint": checkpoint_hash,
            "file_sha256": checkpoint_files,
        },
        "environment": environment,
    }
    if screen_no_eval:
        manifest["validation_disabled_reason"] = no_eval_reason
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "mode": args.mode,
                "checkpoint": str(checkpoint),
                "metrics": result.metrics,
                "environment": environment,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
