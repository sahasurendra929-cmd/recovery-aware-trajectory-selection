#!/usr/bin/env python3
"""Fail-closed V5 Stage-1 SFT end-to-end validation aggregation.

The independent unit is a tau2 task.  This screen requires exactly one
observation per ``(evaluation model, condition, task_id)``; a second rollout or
seed is rejected instead of being counted as another task.  Official-test IDs
are used solely as a seal: seeing one in the validation manifest or a result
file is a hard error.

The zero-shot ``base_model`` control and all four trained-arm directories are
required.  Each directory may contain merged tau2 JSON results or
runner-native ``*.shard-NNN-of-NNN.json`` files.  Every JSON file must contain
a top-level ``simulations`` list, and the union must cover exactly the 21
derived-validation tasks.  The validation manifest uses the paired Stage-1
schema and binds itself to the split-manifest SHA-256.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any, Iterable

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit
try:
    import v5_judge_audit_contract as judge_audit_contract
except ModuleNotFoundError:
    from scripts import v5_judge_audit_contract as judge_audit_contract
try:
    from run_v5_sft_causal_eval import (
        MAX_MODEL_LEN,
        PROFILE_GENERATION_INJECTIONS,
        PROFILE_MODEL_IDS,
        PROVENANCE_PROFILES,
        STRICT_NL_JUDGE_ENTRYPOINT,
        STRICT_NL_JUDGE_MODULE,
        TOOL_ACTION_INTERFACE,
        V5_3_DESIGN_PROTOCOL,
        V5_3_DESIGN_VERSION,
        audit_result_interface,
        checkpoint_registry_provenance_profile,
        contract_core_sha256,
    )
except ModuleNotFoundError:
    from scripts.run_v5_sft_causal_eval import (
        MAX_MODEL_LEN,
        PROFILE_GENERATION_INJECTIONS,
        PROFILE_MODEL_IDS,
        PROVENANCE_PROFILES,
        STRICT_NL_JUDGE_ENTRYPOINT,
        STRICT_NL_JUDGE_MODULE,
        TOOL_ACTION_INTERFACE,
        V5_3_DESIGN_PROTOCOL,
        V5_3_DESIGN_VERSION,
        audit_result_interface,
        checkpoint_registry_provenance_profile,
        contract_core_sha256,
    )


PROTOCOL = "v5_stage1_sft_causal_validation"
SUMMARY_CONTRACT_PROTOCOL = "v5_stage1_sft_causal_validation_summary_v2"
EVALUATION_MANIFEST_PROTOCOL = PROTOCOL
FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
PUBLISHED_STAGE0_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
CHECKPOINT_REGISTRY_PROTOCOL = "v5_stage1_checkpoint_registry"
RUN_CONTRACT_PROTOCOL = "v5_stage1_sft_causal_validation_run"
TRAINED_ARMS = (
    "perfect_success",
    "failure_raw",
    "repair_50",
    "repair_100",
)
EXPECTED_ARMS = ("base_model", *TRAINED_ARMS)
DEFAULT_CONTROL_ARM = "perfect_success"
DIRECTIONAL_CANDIDATE_ARM = "repair_50"
DOMAINS = ("retail", "airline")
CONDITIONS = ("clean", "error")
EXPECTED_VALIDATION_TASKS = 21
EXPECTED_SEALED_TEST_TASKS = 60
INFRASTRUCTURE_TERMINATIONS = {
    "infrastructure_error",
    "unexpected_error",
    "context_window_exceeded",
}
GATE_MIN_INJECTED_COUNT_EQUIVALENT = 2.0
GATE_MIN_CLEAN_COUNT_EQUIVALENT = -1.0
REPLICATE_FIELDS = (
    "replicate_id",
    "rollout_id",
    "rollout_seed",
    "evaluation_seed",
    "trial",
    "seed",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FROZEN_DECODING = {
    "temperature": 0,
    "max_tokens": 512,
    "max_steps": 60,
    "task_timeout_seconds": 900.0,
    "seed": 20260722,
    "num_trials": 1,
}
EXPECTED_RUNTIME_PREFLIGHT = {
    "served_context_window_tokens": MAX_MODEL_LEN,
    "request_max_tokens": FROZEN_DECODING["max_tokens"],
    "maximum_nonoverflow_prompt_tokens": (
        MAX_MODEL_LEN - FROZEN_DECODING["max_tokens"]
    ),
    "max_steps": FROZEN_DECODING["max_steps"],
    "request_token_overflow_policy": "fail_closed",
    "longest_prompt_observation": (
        "completion_audit.max_observed_prompt_tokens"
    ),
}
EXPECTED_STRICT_NL_JUDGE = {
    "module": STRICT_NL_JUDGE_MODULE,
    "entrypoint": STRICT_NL_JUDGE_ENTRYPOINT,
    "mode": "strict_json_schema_fail_closed",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def validate_training_data_provenance(
    payload: dict[str, Any],
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    provenance_profile = checkpoint_registry_provenance_profile(
        payload,
        expected_profile=expected_profile,
    )
    provenance = payload.get("training_data_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Checkpoint registry lacks training_data_provenance")
    design_version = provenance.get("design_version")
    if provenance_profile == "legacy":
        if design_version not in (None, "5.2"):
            raise RuntimeError(
                "Legacy checkpoint provenance requires absent or 5.2 "
                f"design_version, got {design_version!r}"
            )
    else:
        design = provenance.get("design_provenance")
        expected_design = {
            "design_version": V5_3_DESIGN_VERSION,
            "design_protocol": V5_3_DESIGN_PROTOCOL,
            "effective_generation_tasks": 78,
            "validation_tasks": 21,
        }
        if design_version != V5_3_DESIGN_VERSION:
            raise RuntimeError(
                "V5.3 checkpoint provenance requires design_version '5.3'"
            )
        if not isinstance(design, dict) or any(
            design.get(field) != value
            for field, value in expected_design.items()
        ):
            raise RuntimeError(
                "V5.3 checkpoint provenance lacks the frozen 78/21 "
                "design identity"
            )
    if (
        provenance.get("official_test_used") is not False
        or provenance.get("official_test_sealed") is not True
    ):
        raise RuntimeError("Checkpoint registry data provenance lacks test seal")
    for field in ("data_audit_sha256", "data_hashes_sha256"):
        if not _valid_sha256(provenance.get(field)):
            raise RuntimeError(
                f"Checkpoint registry data provenance has invalid {field}"
            )
    dynamic = provenance.get("dynamic_audits")
    if not isinstance(dynamic, dict) or set(dynamic) != {
        "generation",
        "validation",
    }:
        raise RuntimeError("Checkpoint registry dynamic-audit provenance drift")
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
            or identity.get("protocol")
            != "v5_stage1_dynamic_injection_audit"
            or identity.get("verified_injections") != expected_count
            or identity.get("official_test_used") is not False
            or identity.get("official_test_sealed") is not True
        ):
            raise RuntimeError(
                f"Checkpoint registry {name} dynamic audit identity drift"
            )
        for field in ("sha256", "manifest_sha256", "split_manifest_sha256"):
            if not _valid_sha256(identity.get(field)):
                raise RuntimeError(
                    f"Checkpoint registry {name} dynamic audit {field} drift"
                )
    return provenance


def load_checkpoint_registry(
    path: Path,
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != CHECKPOINT_REGISTRY_PROTOCOL
    ):
        raise RuntimeError(
            f"Expected checkpoint registry protocol "
            f"{CHECKPOINT_REGISTRY_PROTOCOL!r}"
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("Checkpoint registry source_commit must be a full commit")
    base_revision = payload.get("base_model_revision")
    if (
        not isinstance(base_revision, str)
        or COMMIT_RE.fullmatch(base_revision) is None
    ):
        raise RuntimeError(
            "Checkpoint registry base_model_revision must be a full revision"
        )
    entries = payload.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(EXPECTED_ARMS):
        raise RuntimeError(
            f"Checkpoint registry entries must be exactly {EXPECTED_ARMS}"
        )
    provenance_profile = checkpoint_registry_provenance_profile(
        payload,
        expected_profile=expected_profile,
    )
    expected_model_ids = PROFILE_MODEL_IDS[provenance_profile]
    model_ids: list[str] = []
    for arm in EXPECTED_ARMS:
        entry = entries[arm]
        if not isinstance(entry, dict):
            raise RuntimeError(f"Checkpoint registry entry {arm} must be an object")
        model_id = entry.get("model_id")
        if (
            not isinstance(model_id, str)
            or model_id != model_id.strip()
            or not model_id.startswith("openai/")
            or len(model_id) == len("openai/")
        ):
            raise RuntimeError(
                f"Checkpoint registry entry {arm} model_id must use an explicit "
                "openai/ LiteLLM provider prefix"
            )
        model_ids.append(model_id)
        if arm == "base_model":
            if entry.get("adapter_sha256") is not None:
                raise RuntimeError("base_model adapter_sha256 must be null")
            if entry.get("adapter_config_sha256") is not None:
                raise RuntimeError(
                    "base_model adapter_config_sha256 must be null"
                )
            if entry.get("training_run_manifest_sha256") is not None:
                raise RuntimeError(
                    "base_model training_run_manifest_sha256 must be null"
                )
        else:
            if not _valid_sha256(entry.get("adapter_sha256")):
                raise RuntimeError(f"{arm} has invalid adapter_sha256")
            if not _valid_sha256(entry.get("adapter_config_sha256")):
                raise RuntimeError(
                    f"{arm} has invalid adapter_config_sha256"
                )
            if not _valid_sha256(entry.get("training_run_manifest_sha256")):
                raise RuntimeError(
                    f"{arm} has invalid training_run_manifest_sha256"
                )
    if len(set(model_ids)) != len(model_ids):
        raise RuntimeError("All five checkpoint registry model_id aliases must be unique")
    for arm in EXPECTED_ARMS:
        model_id = entries[arm]["model_id"]
        if model_id != expected_model_ids[arm]:
            raise RuntimeError(
                f"Checkpoint registry entry {arm} model_id drift for "
                f"{provenance_profile}: expected {expected_model_ids[arm]!r}, "
                f"got {model_id!r}"
            )
    validate_training_data_provenance(
        payload,
        expected_profile=provenance_profile,
    )
    return payload


def load_simulations(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    simulations = payload.get("simulations") if isinstance(payload, dict) else None
    if not isinstance(simulations, list):
        raise RuntimeError(f"Missing simulations list in {path}")
    if not all(isinstance(row, dict) for row in simulations):
        raise RuntimeError(f"Every simulation must be an object in {path}")
    return simulations


def result_files(arm_dir: Path, domain: str, condition: str) -> list[Path]:
    """Resolve either one merged result or one complete set of GPU shards."""

    stem = f"{domain}_{condition}"
    merged = arm_dir / f"{stem}.json"
    shards = sorted(arm_dir.glob(f"{stem}.shard-*-of-*.json"))
    if merged.is_file() and shards:
        raise RuntimeError(
            f"Ambiguous merged and sharded results both exist for {stem} in {arm_dir}"
        )
    if merged.is_file():
        return [merged]
    if not shards:
        raise RuntimeError(
            f"Missing merged or sharded result files for {stem} in {arm_dir}"
        )

    pattern = re.compile(
        rf"^{re.escape(stem)}\.shard-(\d+)-of-(\d+)\.json$"
    )
    parsed: list[tuple[int, int, Path]] = []
    for path in shards:
        match = pattern.match(path.name)
        if match is None:
            raise RuntimeError(f"Malformed result shard name: {path}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        raise RuntimeError(f"Inconsistent shard totals for {stem}: {sorted(totals)}")
    total = totals.pop()
    indices = [index for index, _, _ in parsed]
    if total <= 0 or len(indices) != len(set(indices)) or any(
        index < 0 or index >= total for index in indices
    ):
        raise RuntimeError(
            f"Invalid shard set for {stem}: indices={sorted(indices)}, total={total}"
        )
    # The runner deliberately omits a domain/condition file when a global
    # task shard contains no task from that domain.  Missing *non-empty*
    # shards are still detected by the exact 21-task coverage check below.
    return [path for _, _, path in sorted(parsed)]


def contract_files(arm_dir: Path) -> list[Path]:
    merged = arm_dir / "run_contract.json"
    shards = sorted(arm_dir.glob("run_contract.shard-*-of-*.json"))
    if merged.is_file() and shards:
        raise RuntimeError(
            f"Ambiguous merged and sharded run contracts in {arm_dir}"
        )
    if merged.is_file():
        return [merged]
    if not shards:
        raise RuntimeError(f"Missing run contract collection in {arm_dir}")
    pattern = re.compile(r"^run_contract\.shard-(\d+)-of-(\d+)\.json$")
    parsed: list[tuple[int, int, Path]] = []
    for path in shards:
        match = pattern.match(path.name)
        if match is None:
            raise RuntimeError(f"Malformed run contract filename: {path}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        raise RuntimeError(
            f"Inconsistent run contract shard totals in {arm_dir}: {sorted(totals)}"
        )
    total = totals.pop()
    indices = sorted(index for index, _, _ in parsed)
    if total <= 0 or indices != list(range(total)):
        raise RuntimeError(
            f"Incomplete run contract collection in {arm_dir}: "
            f"indices={indices}, total={total}"
        )
    return [path for _, _, path in sorted(parsed)]


def expected_contract_result_tasks(
    payload: dict[str, Any],
) -> dict[str, list[str]]:
    task_ids = payload.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise RuntimeError("Run contract task_ids are invalid")
    if payload.get("conditions") != ["clean", "error"]:
        raise RuntimeError("Run contract must cover clean and error")
    try:
        shard_index = int(payload["shard_index"])
        num_shards = int(payload["num_shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Run contract shard metadata is invalid") from exc
    by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for identity in task_ids:
        if ":" not in identity:
            raise RuntimeError(f"Malformed contract task identity {identity!r}")
        domain, task_id = identity.split(":", 1)
        if domain not in by_domain or not task_id:
            raise RuntimeError(f"Malformed contract task identity {identity!r}")
        by_domain[domain].append(task_id)
    suffix = (
        ""
        if num_shards == 1
        else f".shard-{shard_index:03d}-of-{num_shards:03d}"
    )
    return {
        f"{domain}_{condition}{suffix}.json": sorted(by_domain[domain])
        for domain in DOMAINS
        if by_domain[domain]
        for condition in CONDITIONS
    }


def expected_local_adapter_identity(
    registry: dict[str, Any],
) -> dict[str, dict[str, str]]:
    return {
        arm: {
            "adapter_sha256": registry["entries"][arm]["adapter_sha256"],
            "adapter_config_sha256": registry["entries"][arm][
                "adapter_config_sha256"
            ],
        }
        for arm in sorted(TRAINED_ARMS)
    }


def expected_served_aliases(registry: dict[str, Any]) -> list[str]:
    return sorted(
        registry["entries"][arm]["model_id"].removeprefix("openai/")
        for arm in EXPECTED_ARMS
    )


def validate_completion_audit(
    payload: dict[str, Any],
    *,
    path: Path,
    arm_dir: Path,
    expected_results: dict[str, list[str]],
) -> dict[str, Any]:
    """Recompute evaluator completion evidence from every bound result."""

    file_audits = {
        filename: audit_result_interface(
            arm_dir / filename,
            expected_task_ids=task_ids,
            num_trials=int(payload["decoding"]["num_trials"]),
            max_tokens=int(payload["decoding"]["max_tokens"]),
        )
        for filename, task_ids in sorted(expected_results.items())
    }
    expected = {
        "protocol": "v5_stage1_sft_causal_interface_completion_audit",
        "tool_action_interface": dict(TOOL_ACTION_INTERFACE),
        "result_files": file_audits,
        "max_observed_prompt_tokens": max(
            item["max_observed_prompt_tokens"]
            for item in file_audits.values()
        ),
        "max_observed_total_request_tokens": max(
            item["max_observed_total_request_tokens"]
            for item in file_audits.values()
        ),
        "all_expected_tasks_observed_once_per_trial": True,
        "request_token_overflow_detected": False,
        "status": "PASS",
    }
    if payload.get("completion_audit") != expected:
        raise RuntimeError(
            f"Run contract completion audit drift or stale evidence in {path}"
        )
    return expected


def load_arm_contracts(
    *,
    arm: str,
    arm_dir: Path,
    registry: dict[str, Any],
    registry_sha256: str,
    evaluation_manifest_sha256: str,
    split_manifest_sha256: str,
    expected_task_ids: set[str],
    expected_fault_protocol: dict[str, Any],
    expected_tau2_commit: str,
    expected_source_files: dict[str, Any],
    expected_dynamic_audit_identity: dict[str, Any],
) -> dict[str, Any]:
    registry_profile = checkpoint_registry_provenance_profile(registry)
    expected_entry = registry["entries"][arm]
    frozen_base_alias = registry["entries"]["base_model"]["model_id"]
    seen_tasks: set[str] = set()
    schedule: list[dict[str, Any]] = []
    contract_hashes: dict[str, str] = {}
    decoding_contract: str | None = None
    role_contract: str | None = None
    declared_result_hashes: dict[str, str] = {}
    strict_judge_evidence_hashes: dict[str, str] = {}

    for path in contract_files(arm_dir):
        payload = load_json(path)
        if not isinstance(payload, dict) or payload.get("protocol") != RUN_CONTRACT_PROTOCOL:
            raise RuntimeError(f"Invalid validation run contract: {path}")
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"Run contract is not COMPLETE: {path}")
        if payload.get("arm") != arm:
            raise RuntimeError(f"Run contract arm mismatch in {path}")
        if payload.get("source_commit") != registry["source_commit"]:
            raise RuntimeError(f"Run contract source commit drift in {path}")
        if payload.get("base_model_revision") != registry["base_model_revision"]:
            raise RuntimeError(f"Run contract base revision drift in {path}")
        if (
            payload.get("evaluation_manifest_protocol")
            != EVALUATION_MANIFEST_PROTOCOL
        ):
            raise RuntimeError(
                f"Run contract evaluation manifest protocol drift in {path}"
            )
        if payload.get("checkpoint_registry_protocol") != registry["protocol"]:
            raise RuntimeError(f"Run contract registry protocol drift in {path}")
        contract_profile = payload.get(
            "checkpoint_registry_provenance_profile",
            "legacy",
        )
        if contract_profile != registry_profile:
            raise RuntimeError(
                f"Run contract registry provenance profile drift in {path}"
            )
        if payload.get("checkpoint_registry_sha256") != registry_sha256:
            raise RuntimeError(f"Run contract registry SHA drift in {path}")
        if payload.get("evaluation_manifest_sha256") != evaluation_manifest_sha256:
            raise RuntimeError(f"Run contract evaluation manifest SHA drift in {path}")
        if payload.get("split_manifest_sha256") != split_manifest_sha256:
            raise RuntimeError(f"Run contract split manifest SHA drift in {path}")
        if payload.get("dynamic_audit_identity") != expected_dynamic_audit_identity:
            raise RuntimeError(f"Run contract dynamic audit identity drift in {path}")
        if payload.get("fault_protocol") != expected_fault_protocol:
            raise RuntimeError(f"Run contract fault protocol drift in {path}")
        if payload.get("tau2_commit") != expected_tau2_commit:
            raise RuntimeError(f"Run contract tau2 commit drift in {path}")
        if payload.get("source_files") != expected_source_files:
            raise RuntimeError(f"Run contract source-file provenance drift in {path}")
        if payload.get("checkpoint_entry") != expected_entry:
            raise RuntimeError(f"Run contract checkpoint entry drift in {path}")
        if payload.get(
            "locally_verified_adapter_identity"
        ) != expected_local_adapter_identity(registry):
            raise RuntimeError(
                f"Run contract local adapter identity drift in {path}"
            )
        if payload.get("served_registry_aliases") != expected_served_aliases(
            registry
        ):
            raise RuntimeError(
                f"Run contract served model alias audit drift in {path}"
            )
        agent = payload.get("agent")
        user = payload.get("user")
        judge = payload.get("judge")
        if not all(isinstance(role, dict) for role in (agent, user, judge)):
            raise RuntimeError(f"Run contract role metadata missing in {path}")
        if agent.get("model") != expected_entry["model_id"]:
            raise RuntimeError(f"Run contract agent model drift in {path}")
        if agent.get("revision") != registry["base_model_revision"]:
            raise RuntimeError(f"Run contract agent revision drift in {path}")
        if (
            user.get("model") != frozen_base_alias
            or judge.get("model") != frozen_base_alias
        ):
            raise RuntimeError(
                f"Run contract user/judge must both equal the registry "
                f"base_model alias in {path}"
            )
        if (
            user.get("revision") != registry["base_model_revision"]
            or judge.get("revision") != registry["base_model_revision"]
        ):
            raise RuntimeError(
                f"Run contract user/judge revision drift in {path}"
            )
        if judge.get("strict_backend") != EXPECTED_STRICT_NL_JUDGE:
            raise RuntimeError(
                f"Run contract strict NL judge drift in {path}"
            )
        api_bases = [role.get("api_base") for role in (agent, user, judge)]
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in api_bases
            )
            or len(
                {value.strip().rstrip("/") for value in api_bases}
            )
            != 1
        ):
            raise RuntimeError(
                f"Run contract agent/user/judge API interface drift in {path}"
            )
        roles = canonical_json(
            {
                "user_model": user["model"],
                "judge_model": judge["model"],
                "user_revision": user["revision"],
                "judge_revision": judge["revision"],
                "strict_backend": judge["strict_backend"],
            }
        )
        if role_contract is None:
            role_contract = roles
        elif roles != role_contract:
            raise RuntimeError(f"Run contract user/judge role drift in {path}")

        decoding = payload.get("decoding")
        required_decoding = {
            "temperature",
            "max_tokens",
            "max_steps",
            "task_timeout_seconds",
            "seed",
            "num_trials",
        }
        if not isinstance(decoding, dict) or not required_decoding <= set(decoding):
            raise RuntimeError(f"Run contract decoding metadata missing in {path}")
        if decoding != FROZEN_DECODING:
            raise RuntimeError(
                f"Run contract decoding protocol drift in {path}; expected "
                f"{FROZEN_DECODING}, got {decoding}"
            )
        serialized_decoding = canonical_json(decoding)
        if decoding_contract is None:
            decoding_contract = serialized_decoding
        elif serialized_decoding != decoding_contract:
            raise RuntimeError(f"Run contract decoding drift in {path}")
        if payload.get("tool_action_interface") != TOOL_ACTION_INTERFACE:
            raise RuntimeError(
                f"Run contract tool-action interface drift in {path}"
            )
        if payload.get("runtime_preflight") != EXPECTED_RUNTIME_PREFLIGHT:
            raise RuntimeError(
                f"Run contract 32k/60-step runtime preflight drift in {path}"
            )
        if payload.get("conditions") != ["clean", "error"]:
            raise RuntimeError(f"Run contract conditions drift in {path}")
        if payload.get("official_test_used") is not False:
            raise RuntimeError(f"Run contract opened official test in {path}")
        observed_core_hash = payload.get("contract_core_sha256")
        if (
            not _valid_sha256(observed_core_hash)
            or observed_core_hash != contract_core_sha256(payload)
        ):
            raise RuntimeError(
                f"Run contract immutable core hash drift in {path}"
            )

        try:
            shard_index = int(payload["shard_index"])
            num_shards = int(payload["num_shards"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Run contract shard metadata missing in {path}") from exc
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise RuntimeError(f"Run contract shard metadata invalid in {path}")
        if path.name == "run_contract.json":
            if (shard_index, num_shards) != (0, 1):
                raise RuntimeError("Merged run_contract.json must be shard 0 of 1")
        else:
            expected_name = (
                f"run_contract.shard-{shard_index:03d}-of-"
                f"{num_shards:03d}.json"
            )
            if path.name != expected_name:
                raise RuntimeError(f"Run contract filename/metadata drift in {path}")

        result_hashes = payload.get("result_sha256")
        if (
            not isinstance(result_hashes, dict)
            or not result_hashes
            or any(
                not isinstance(name, str) or not _valid_sha256(digest)
                for name, digest in result_hashes.items()
            )
        ):
            raise RuntimeError(f"Run contract lacks valid result_sha256 in {path}")
        expected_contract_results = expected_contract_result_tasks(payload)
        if set(result_hashes) != set(expected_contract_results):
            raise RuntimeError(
                f"Run contract result set is not the exact task/domain/condition "
                f"coverage in {path}; expected={sorted(expected_contract_results)}, "
                f"observed={sorted(result_hashes)}"
            )
        result_pattern = re.compile(
            r"^(retail|airline)_(clean|error)"
            r"(?:\.shard-(\d+)-of-(\d+))?\.json$"
        )
        contract_result_paths: list[Path] = []
        for filename, expected_hash in result_hashes.items():
            match = result_pattern.fullmatch(filename)
            if match is None:
                raise RuntimeError(
                    f"Run contract declares malformed result filename: {filename}"
                )
            result_shard = match.group(3)
            result_total = match.group(4)
            if num_shards == 1:
                if result_shard is not None:
                    raise RuntimeError(
                        f"Merged contract declares a sharded result: {filename}"
                    )
            elif (
                result_shard is None
                or int(result_shard) != shard_index
                or int(result_total) != num_shards
            ):
                raise RuntimeError(
                    f"Run contract/result shard identity drift: {filename}"
                )
            if filename in declared_result_hashes:
                raise RuntimeError(
                    f"Result file declared by multiple contracts: {filename}"
                )
            result_path = arm_dir / filename
            if not result_path.is_file():
                raise RuntimeError(
                    f"Run contract result file is missing: {result_path}"
                )
            observed_hash = sha256_file(result_path)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Run contract result SHA drift for {result_path}"
                )
            declared_result_hashes[filename] = expected_hash
            contract_result_paths.append(result_path)
        validate_completion_audit(
            payload,
            path=path,
            arm_dir=arm_dir,
            expected_results=expected_contract_results,
        )
        if registry_profile == "v5_3":
            try:
                observed_evidence = (
                    judge_audit_contract.validate_strict_judge_evidence(
                        contract_result_paths,
                        maximum_content_attempts=2,
                    )
                )
            except judge_audit_contract.StrictJudgeEvidenceError as error:
                raise RuntimeError(
                    f"{path}: V5.3 strict-judge evidence is incomplete"
                ) from error
            if payload.get("strict_judge_audit_evidence") != observed_evidence:
                raise RuntimeError(
                    f"{path}: V5.3 strict-judge evidence contract drift"
                )
            strict_judge_evidence_hashes[path.name] = observed_evidence[
                "canonical_mapping_sha256"
            ]

        task_ids = payload.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or any(not isinstance(task_id, str) for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise RuntimeError(f"Run contract task_ids invalid in {path}")
        task_set = set(task_ids)
        unexpected = task_set - expected_task_ids
        if unexpected:
            raise RuntimeError(
                f"Run contract contains non-validation tasks in {path}: "
                f"{sorted(unexpected)}"
            )
        overlap = seen_tasks & task_set
        if overlap:
            raise RuntimeError(
                f"Run contracts duplicate validation tasks for {arm}: "
                f"{sorted(overlap)}"
            )
        seen_tasks.update(task_set)
        schedule.append(
            {
                "shard_index": shard_index,
                "num_shards": num_shards,
                "task_ids": sorted(task_set),
            }
        )
        contract_hashes[path.name] = sha256_file(path)

    if seen_tasks != expected_task_ids:
        raise RuntimeError(
            f"Run contracts for {arm} do not cover all 21 validation tasks; "
            f"missing={sorted(expected_task_ids - seen_tasks)}"
        )
    schedule.sort(key=lambda row: row["shard_index"])
    totals = {row["num_shards"] for row in schedule}
    if len(totals) != 1 or len(schedule) != next(iter(totals)):
        raise RuntimeError(f"Run contract shard collection drift for {arm}")
    loaded_result_names = {
        path.name
        for domain in DOMAINS
        for condition in CONDITIONS
        for path in result_files(arm_dir, domain, condition)
    }
    if set(declared_result_hashes) != loaded_result_names:
        raise RuntimeError(
            f"Run contract result declaration set differs from loaded files for "
            f"{arm}; undeclared={sorted(loaded_result_names - set(declared_result_hashes))}, "
            f"missing={sorted(set(declared_result_hashes) - loaded_result_names)}"
        )
    return {
        "model_id": expected_entry["model_id"],
        "checkpoint_entry": expected_entry,
        "source_commit": registry["source_commit"],
        "base_model_revision": registry["base_model_revision"],
        "dynamic_audit_identity": expected_dynamic_audit_identity,
        "contract_sha256": contract_hashes,
        "result_sha256": dict(sorted(declared_result_hashes.items())),
        "contract_collection_sha256": hashlib.sha256(
            canonical_json(dict(sorted(contract_hashes.items()))).encode("utf-8")
        ).hexdigest(),
        "result_collection_sha256": hashlib.sha256(
            canonical_json(
                dict(sorted(declared_result_hashes.items()))
            ).encode("utf-8")
        ).hexdigest(),
        "schedule": schedule,
        "decoding": json.loads(decoding_contract or "{}"),
        "roles": json.loads(role_contract or "{}"),
        "tool_action_interface": dict(TOOL_ACTION_INTERFACE),
        "runtime_preflight": dict(EXPECTED_RUNTIME_PREFLIGHT),
        "completion_audit_required": True,
        "strict_judge_evidence_mapping_sha256": dict(
            sorted(strict_judge_evidence_hashes.items())
        ),
    }


def pair_id(domain: str, task_id: Any) -> str:
    return f"{domain}:{task_id}"


def canonical_call(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise RuntimeError("Tool calls must contain a string name and object arguments")
    return canonical_json({"name": name, "arguments": arguments})


def reward(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info") or {}
    value = reward_info.get("reward")
    if value is None:
        raise RuntimeError(
            f"Missing official reward for task {simulation.get('task_id')}"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Non-numeric reward for task {simulation.get('task_id')}: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise RuntimeError(
            f"Non-finite reward for task {simulation.get('task_id')}: {numeric}"
        )
    return numeric


def is_success(simulation: dict[str, Any]) -> bool:
    """Official tau2 task success is a final reward exactly equal to one."""

    return math.isclose(reward(simulation), 1.0, rel_tol=0.0, abs_tol=1e-12)


def replicate_key(simulation: dict[str, Any]) -> str:
    """Return an auditable key without treating the key as an independent task."""

    values: dict[str, Any] = {}
    metadata = simulation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for field in REPLICATE_FIELDS:
        if field in simulation:
            values[field] = simulation[field]
        elif field in metadata:
            values[field] = metadata[field]
    return canonical_json(values) if values else "__single_replicate__"


def all_tool_calls(messages: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                yield tool_call


def analyze_error_run(
    simulation: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, bool]:
    """Verify the injected error and measure post-error behavior.

    A valid post-error result must correspond to a tool call emitted *after*
    the injected error.  A stray successful tool message therefore cannot
    satisfy the metric.
    """

    messages = simulation.get("messages") or []
    if not isinstance(messages, list):
        raise RuntimeError(
            f"Messages must be a list for task {simulation.get('task_id')}"
        )
    expected_id = expected["tool_call_id"]
    expected_call = canonical_call(
        {
            "name": expected["tool_name"],
            "arguments": expected["arguments"],
        }
    )
    injected_call: dict[str, Any] | None = None
    injected_call_index: int | None = None
    error_indices: list[int] = []

    for index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            if tool_call.get("id") == expected_id:
                if injected_call is not None:
                    raise RuntimeError(
                        f"Injected call ID appears more than once for task "
                        f"{simulation.get('task_id')}"
                    )
                injected_call = tool_call
                injected_call_index = index
        if (
            message.get("role") == "tool"
            and message.get("id") == expected_id
            and bool(message.get("error"))
        ):
            error_indices.append(index)

    if injected_call is None:
        raise RuntimeError(
            f"Task {simulation.get('task_id')} lacks the injected tool call"
        )
    if canonical_call(injected_call) != expected_call:
        raise RuntimeError(
            f"Task {simulation.get('task_id')} injected call differs from manifest"
        )
    if len(error_indices) != 1:
        raise RuntimeError(
            f"Task {simulation.get('task_id')} must contain exactly one "
            f"injected tool error; found {len(error_indices)}"
        )
    error_index = error_indices[0]
    if injected_call_index is None or injected_call_index >= error_index:
        raise RuntimeError(
            f"Task {simulation.get('task_id')} injected result precedes its call"
        )

    repeated_identical = False
    later_call_ids: set[str] = set()
    valid_post_error_tool_result = False
    for message in messages[error_index + 1 :]:
        for tool_call in message.get("tool_calls") or []:
            if canonical_call(tool_call) == expected_call:
                repeated_identical = True
            tool_call_id = tool_call.get("id")
            if tool_call_id is not None:
                later_call_ids.add(str(tool_call_id))
        if (
            message.get("role") == "tool"
            and str(message.get("id")) in later_call_ids
            and message.get("error") is False
        ):
            valid_post_error_tool_result = True

    return {
        "injected_error_observed": True,
        "repeated_identical_error": repeated_identical,
        "valid_post_error_tool_result": valid_post_error_tool_result,
    }


def verify_clean_has_no_injection(
    simulation: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    expected_id = expected["tool_call_id"]
    for call in all_tool_calls(simulation.get("messages") or []):
        if call.get("id") == expected_id:
            raise RuntimeError(
                f"Clean task {simulation.get('task_id')} contains injected call ID"
            )


def validate_split_and_evaluation_manifests(
    split_manifest_path: Path,
    evaluation_manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    if sha256_file(split_manifest_path) != PUBLISHED_STAGE0_SPLIT_SHA256:
        raise RuntimeError(
            "historical Stage-0 split differs from its published SHA-256"
        )
    split = load_json(split_manifest_path)
    evaluation = load_json(evaluation_manifest_path)
    if not isinstance(split, dict) or not isinstance(evaluation, dict):
        raise RuntimeError("Split and evaluation manifests must be JSON objects")

    guarantees = split.get("guarantees") or {}
    if guarantees.get("validation_derived_from_official_train_only") is not True:
        raise RuntimeError("Split does not certify validation-from-train")
    if guarantees.get("official_test_task_content_exported") is not False:
        raise RuntimeError("Split does not certify that official-test content is sealed")

    validation_ids: set[str] = set()
    sealed_ids: set[str] = set()
    for domain in DOMAINS:
        domain_split = (split.get("domains") or {}).get(domain)
        if not isinstance(domain_split, dict):
            raise RuntimeError(f"Split manifest lacks domain {domain}")
        inner = {
            pair_id(domain, task_id)
            for task_id in domain_split.get("inner_train_ids") or []
        }
        validation = {
            pair_id(domain, task_id)
            for task_id in domain_split.get("validation_ids") or []
        }
        sealed = {
            pair_id(domain, task_id)
            for task_id in domain_split.get("sealed_test_ids") or []
        }
        if inner & validation or inner & sealed or validation & sealed:
            raise RuntimeError(f"Split overlap detected for domain {domain}")
        validation_ids.update(validation)
        sealed_ids.update(sealed)

    if len(validation_ids) != EXPECTED_VALIDATION_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_TASKS} validation tasks, "
            f"found {len(validation_ids)}"
        )
    if len(sealed_ids) != EXPECTED_SEALED_TEST_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_SEALED_TEST_TASKS} sealed official-test tasks, "
            f"found {len(sealed_ids)}"
        )

    if evaluation.get("protocol") != EVALUATION_MANIFEST_PROTOCOL:
        raise RuntimeError(
            f"Evaluation protocol must be {EVALUATION_MANIFEST_PROTOCOL!r}, "
            f"got {evaluation.get('protocol')!r}"
        )
    if evaluation.get("source_split") != "derived_validation":
        raise RuntimeError("Stage-1 evaluation must be validation-only")
    if evaluation.get("official_test_used") is not False:
        raise RuntimeError("Evaluation manifest does not seal official test")
    if evaluation.get("official_test_sealed") is not True:
        raise RuntimeError("Evaluation manifest lacks the explicit official-test seal")
    if evaluation.get("scientific_claim_allowed") not in (None, False):
        raise RuntimeError("Stage-1 validation manifest must prohibit final claims")
    expected_split_sha = sha256_file(split_manifest_path)
    declared_split_sha = evaluation.get("split_manifest_sha256")
    if declared_split_sha != expected_split_sha:
        raise RuntimeError("Evaluation manifest is not bound to this split manifest")
    fault_protocol = evaluation.get("fault_protocol")
    if (
        not isinstance(fault_protocol, dict)
        or fault_protocol.get("protocol") != FAULT_PROTOCOL
        or fault_protocol.get("claim_scope")
        != "multi_fault_family_post_fault_robustness_screen"
        or fault_protocol.get("family_level_inference") != "descriptive_only"
    ):
        raise RuntimeError("Evaluation manifest lacks frozen multi-fault protocol")
    guarantees = fault_protocol.get("guarantees")
    required_guarantees = {
        "tool_type": "READ",
        "expected_tool_error": True,
        "expected_state_mutation": False,
        "per_task_unique_invalid_parameter": True,
        "database_absence_checked_at_pinned_tau2_commit": True,
    }
    if not isinstance(guarantees, dict) or any(
        guarantees.get(field) != value
        for field, value in required_guarantees.items()
    ):
        raise RuntimeError("Evaluation fault-protocol safety guarantees drift")
    raw_catalog = fault_protocol.get("families")
    if not isinstance(raw_catalog, dict) or set(raw_catalog) != set(DOMAINS):
        raise RuntimeError("Evaluation fault catalog domain drift")
    catalog: dict[str, dict[str, tuple[str, str]]] = {}
    for domain in DOMAINS:
        family_rows = raw_catalog.get(domain)
        if not isinstance(family_rows, list) or len(family_rows) < 2:
            raise RuntimeError(f"{domain}: insufficient evaluation fault catalog")
        catalog[domain] = {}
        for family_row in family_rows:
            if not isinstance(family_row, dict):
                raise RuntimeError(f"{domain}: malformed evaluation fault catalog")
            family = family_row.get("fault_family")
            tool = family_row.get("tool_name")
            invalid_key = family_row.get("invalid_argument_key")
            if not all(
                isinstance(value, str) and value
                for value in (family, tool, invalid_key)
            ):
                raise RuntimeError(f"{domain}: incomplete evaluation fault catalog")
            if family in catalog[domain]:
                raise RuntimeError(f"{domain}: duplicate evaluation fault family")
            catalog[domain][family] = (tool, invalid_key)
    tau2_commit = evaluation.get("tau2_commit")
    if not isinstance(tau2_commit, str) or COMMIT_RE.fullmatch(tau2_commit) is None:
        raise RuntimeError("Evaluation manifest lacks pinned tau2 commit")
    source_files = evaluation.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(DOMAINS):
        raise RuntimeError("Evaluation manifest lacks source file provenance")
    for domain in DOMAINS:
        declared = source_files.get(domain)
        if not isinstance(declared, dict) or any(
            not _valid_sha256(declared.get(field))
            for field in (
                "tasks_json_sha256",
                "db_json_sha256",
                "tools_py_sha256",
            )
        ):
            raise RuntimeError(f"{domain}: invalid source file provenance")

    rows = evaluation.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Evaluation manifest lacks rows")
    expected_rows: dict[str, dict[str, Any]] = {}
    invalid_parameters: set[str] = set()
    tool_call_ids: set[str] = set()
    families_by_domain: dict[str, set[str]] = {
        domain: set() for domain in DOMAINS
    }
    tools_by_domain: dict[str, set[str]] = {
        domain: set() for domain in DOMAINS
    }
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Evaluation rows must be objects")
        domain = row.get("domain")
        if domain not in DOMAINS:
            raise RuntimeError(f"Unexpected evaluation domain: {domain!r}")
        key = pair_id(domain, row.get("task_id"))
        if row.get("pair_id") != key:
            raise RuntimeError(f"Non-canonical pair_id in evaluation manifest: {row}")
        if key in expected_rows:
            raise RuntimeError(f"Duplicate evaluation task: {key}")
        if key in sealed_ids:
            raise RuntimeError(f"Official-test task leaked into validation: {key}")
        if key not in validation_ids:
            raise RuntimeError(f"Non-validation task in evaluation manifest: {key}")
        if row.get("source_split") != "derived_validation":
            raise RuntimeError(f"Task {key} is not marked derived_validation")
        clean = row.get("clean_condition") or {}
        error = row.get("error_condition") or {}
        if clean.get("inject_error") is not False:
            raise RuntimeError(f"Task {key} clean condition is not clean")
        required_error = {
            "inject_error": True,
            "expected_tool_error": True,
            "expected_state_mutation": False,
        }
        for field, value in required_error.items():
            if error.get(field) is not value:
                raise RuntimeError(f"Task {key} has invalid error field {field}")
        family = error.get("fault_family")
        specification = catalog[domain].get(family)
        if specification is None:
            raise RuntimeError(f"Task {key} uses unknown fault family {family!r}")
        tool_name, invalid_key = specification
        if (
            error.get("tool_name") != tool_name
            or error.get("tool_type") != "READ"
            or error.get("invalid_argument_key") != invalid_key
        ):
            raise RuntimeError(f"Task {key} fault descriptor differs from catalog")
        arguments = error.get("arguments")
        canonical_call({"name": tool_name, "arguments": arguments})
        if not isinstance(arguments.get(invalid_key), str):
            raise RuntimeError(f"Task {key} lacks its invalid parameter")
        invalid_identity = canonical_json(
            {invalid_key: arguments[invalid_key]}
        )
        if invalid_identity in invalid_parameters:
            raise RuntimeError(
                "Evaluation invalid parameters must be unique per task"
            )
        invalid_parameters.add(invalid_identity)
        tool_call_id = error.get("tool_call_id")
        if (
            not isinstance(tool_call_id, str)
            or not tool_call_id
            or tool_call_id in tool_call_ids
        ):
            raise RuntimeError(
                f"Task {key} lacks a unique injected tool_call_id"
            )
        tool_call_ids.add(tool_call_id)
        if type(error.get("on_reference_path")) is not bool:
            raise RuntimeError(f"Task {key} lacks on_reference_path boolean")
        relevance = error.get("fault_relevance")
        if relevance not in {
            "reference_path_or_operation_aligned",
            "domain_plausible_fallback",
        }:
            raise RuntimeError(f"Task {key} has invalid fault_relevance")
        if (
            error["on_reference_path"]
            and relevance != "reference_path_or_operation_aligned"
        ):
            raise RuntimeError(
                f"Task {key} reference-path fault cannot be a fallback"
            )
        families_by_domain[domain].add(family)
        tools_by_domain[domain].add(tool_name)
        expected_rows[key] = row

    if set(expected_rows) != validation_ids:
        missing = sorted(validation_ids - set(expected_rows))
        extra = sorted(set(expected_rows) - validation_ids)
        raise RuntimeError(
            f"Evaluation manifest must cover exactly all validation tasks; "
            f"missing={missing}, extra={extra}"
        )
    if evaluation.get("paired_task_count") != EXPECTED_VALIDATION_TASKS:
        raise RuntimeError("Evaluation manifest paired_task_count is not 21")
    expected_domain_counts = {
        domain: sum(row["domain"] == domain for row in expected_rows.values())
        for domain in DOMAINS
    }
    if evaluation.get("domain_counts") != expected_domain_counts:
        raise RuntimeError(
            "Evaluation manifest domain_counts differ from the frozen split"
        )
    for domain in DOMAINS:
        if (
            len(families_by_domain[domain]) < 2
            or len(tools_by_domain[domain]) < 2
        ):
            raise RuntimeError(
                f"{domain}: validation lacks multi-fault diversity"
            )
    return expected_rows, validation_ids, sealed_ids


def load_arm(
    arm: str,
    arm_dir: Path,
    expected_rows: dict[str, dict[str, Any]],
    sealed_ids: set[str],
) -> dict[str, Any]:
    """Load one arm, requiring exactly one observation per task/condition."""

    raw: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        condition: {} for condition in CONDITIONS
    }
    result_hashes: dict[str, str] = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for path in result_files(arm_dir, domain, condition):
                result_hashes[path.name] = sha256_file(path)
                for simulation in load_simulations(path):
                    key = pair_id(domain, simulation.get("task_id"))
                    if key in sealed_ids:
                        raise RuntimeError(
                            f"Official-test result leaked into arm {arm}: {key}"
                        )
                    if key not in expected_rows:
                        raise RuntimeError(f"Unexpected task in arm {arm}: {key}")
                    # A second row is not a new independent task, even if it
                    # carries a different rollout/seed identifier.
                    if key in raw[condition]:
                        raise RuntimeError(
                            f"Duplicate task_id across {arm}/{condition} "
                            f"result files: {key}"
                        )
                    if (
                        simulation.get("termination_reason")
                        in INFRASTRUCTURE_TERMINATIONS
                    ):
                        raise RuntimeError(
                            f"Infrastructure failure in {arm}/{condition}/{key}: "
                            f"{simulation.get('termination_reason')}"
                        )
                    replicate = replicate_key(simulation)
                    expected_error = expected_rows[key]["error_condition"]
                    if condition == "clean":
                        verify_clean_has_no_injection(simulation, expected_error)
                        behavior = {}
                    else:
                        behavior = analyze_error_run(simulation, expected_error)
                    raw[condition][key] = {
                        replicate: {
                            "success": float(is_success(simulation)),
                            "reward": reward(simulation),
                            "termination_reason": simulation.get(
                                "termination_reason"
                            ),
                            **behavior,
                        }
                    }

    expected_ids = set(expected_rows)
    for condition in CONDITIONS:
        actual = set(raw[condition])
        if actual != expected_ids:
            raise RuntimeError(
                f"{arm}/{condition} task IDs do not match validation manifest; "
                f"missing={sorted(expected_ids - actual)}, "
                f"extra={sorted(actual - expected_ids)}"
            )

    task_rows: list[dict[str, Any]] = []
    replicate_keys_by_task: dict[str, tuple[str, ...]] = {}
    termination_reasons = {"clean": Counter(), "error": Counter()}
    total_runs = {"clean": 0, "error": 0}
    for key in sorted(expected_ids):
        clean_replicates = raw["clean"][key]
        error_replicates = raw["error"][key]
        if set(clean_replicates) != set(error_replicates):
            raise RuntimeError(
                f"Clean/error replicate mismatch in arm {arm}, task {key}"
            )
        replicate_keys = tuple(sorted(clean_replicates))
        replicate_keys_by_task[key] = replicate_keys
        total_runs["clean"] += len(replicate_keys)
        total_runs["error"] += len(replicate_keys)
        for condition, replicates in (
            ("clean", clean_replicates),
            ("error", error_replicates),
        ):
            termination_reasons[condition].update(
                row["termination_reason"] for row in replicates.values()
            )
        clean_success = mean(
            row["success"] for row in clean_replicates.values()
        )
        error_success = mean(
            row["success"] for row in error_replicates.values()
        )
        task_rows.append(
            {
                "pair_id": key,
                "domain": expected_rows[key]["domain"],
                "task_id": str(expected_rows[key]["task_id"]),
                "fault_family": expected_rows[key]["error_condition"][
                    "fault_family"
                ],
                "fault_tool_name": expected_rows[key]["error_condition"][
                    "tool_name"
                ],
                "fault_relevance": expected_rows[key]["error_condition"][
                    "fault_relevance"
                ],
                "fault_on_reference_path": expected_rows[key][
                    "error_condition"
                ]["on_reference_path"],
                "replicate_count": len(replicate_keys),
                "clean_end_to_end_success": clean_success,
                "error_injected_end_to_end_success": error_success,
                "within_arm_injected_minus_clean": error_success - clean_success,
                "repeated_identical_error_rate": mean(
                    float(row["repeated_identical_error"])
                    for row in error_replicates.values()
                ),
                "valid_post_error_tool_result_rate": mean(
                    float(row["valid_post_error_tool_result"])
                    for row in error_replicates.values()
                ),
            }
        )

    return {
        "arm": arm,
        "arm_dir": str(arm_dir),
        "task_count": len(task_rows),
        "total_runs": total_runs,
        "replicate_keys_by_task": replicate_keys_by_task,
        "result_sha256": result_hashes,
        "termination_reasons": {
            condition: dict(sorted(counter.items()))
            for condition, counter in termination_reasons.items()
        },
        "task_rows": task_rows,
    }


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise RuntimeError("Cannot take a percentile of an empty list")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def task_bootstrap_ci(
    task_deltas: list[float],
    *,
    seed: int,
    draws: int,
) -> list[float] | None:
    if draws <= 0:
        return None
    rng = random.Random(seed)
    count = len(task_deltas)
    boot = [
        mean(task_deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def summarize_arm(loaded: dict[str, Any]) -> dict[str, Any]:
    rows = loaded["task_rows"]
    family_rows = {
        family: [row for row in rows if row["fault_family"] == family]
        for family in sorted({row["fault_family"] for row in rows})
    }
    family_descriptive = {
        family: {
            "inference": "descriptive_only",
            "task_count": len(group),
            "domains": sorted({row["domain"] for row in group}),
            "fault_tool_names": sorted(
                {row["fault_tool_name"] for row in group}
            ),
            "fault_relevance_counts": dict(
                sorted(Counter(row["fault_relevance"] for row in group).items())
            ),
            "on_reference_path_task_count": sum(
                bool(row["fault_on_reference_path"]) for row in group
            ),
            "error_injected_end_to_end_success": metric_mean(
                group, "error_injected_end_to_end_success"
            ),
            "clean_end_to_end_success": metric_mean(
                group, "clean_end_to_end_success"
            ),
            "repeated_identical_error_rate": metric_mean(
                group, "repeated_identical_error_rate"
            ),
            "valid_post_error_tool_result_rate": metric_mean(
                group, "valid_post_error_tool_result_rate"
            ),
        }
        for family, group in family_rows.items()
    }
    return {
        "task_count": len(rows),
        "total_runs": loaded["total_runs"],
        "primary_error_injected_end_to_end_success": metric_mean(
            rows, "error_injected_end_to_end_success"
        ),
        "clean_end_to_end_success": metric_mean(
            rows, "clean_end_to_end_success"
        ),
        "within_arm_injected_minus_clean": metric_mean(
            rows, "within_arm_injected_minus_clean"
        ),
        "repeated_identical_error_rate": metric_mean(
            rows, "repeated_identical_error_rate"
        ),
        "valid_post_error_tool_result_rate": metric_mean(
            rows, "valid_post_error_tool_result_rate"
        ),
        "fault_family_descriptive": family_descriptive,
        "termination_reasons": loaded["termination_reasons"],
        "result_sha256": loaded["result_sha256"],
        "task_results": rows,
    }


def comparison_seed(seed: int, treatment: str, control: str, metric: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{treatment}:{control}:{metric}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def compare_arms(
    treatment: dict[str, Any],
    control: dict[str, Any],
    *,
    seed: int,
    draws: int,
    apply_directional_gate: bool,
) -> dict[str, Any]:
    treatment_rows = {
        row["pair_id"]: row for row in treatment["task_rows"]
    }
    control_rows = {row["pair_id"]: row for row in control["task_rows"]}
    if set(treatment_rows) != set(control_rows):
        raise RuntimeError("Arm comparison task-ID mismatch")

    task_rows: list[dict[str, Any]] = []
    for key in sorted(control_rows):
        tr = treatment_rows[key]
        co = control_rows[key]
        if treatment["replicate_keys_by_task"][key] != control[
            "replicate_keys_by_task"
        ][key]:
            raise RuntimeError(
                f"Cross-arm replicate mismatch for task {key}: "
                f"{treatment['arm']} vs {control['arm']}"
            )
        task_rows.append(
            {
                "pair_id": key,
                "domain": tr["domain"],
                "fault_family": tr["fault_family"],
                "fault_tool_name": tr["fault_tool_name"],
                "fault_relevance": tr["fault_relevance"],
                "injected_success_delta": (
                    tr["error_injected_end_to_end_success"]
                    - co["error_injected_end_to_end_success"]
                ),
                "clean_success_delta": (
                    tr["clean_end_to_end_success"]
                    - co["clean_end_to_end_success"]
                ),
                "repeated_error_rate_delta": (
                    tr["repeated_identical_error_rate"]
                    - co["repeated_identical_error_rate"]
                ),
                "valid_post_error_tool_result_rate_delta": (
                    tr["valid_post_error_tool_result_rate"]
                    - co["valid_post_error_tool_result_rate"]
                ),
            }
        )

    metric_keys = (
        "injected_success_delta",
        "clean_success_delta",
        "repeated_error_rate_delta",
        "valid_post_error_tool_result_rate_delta",
    )
    aggregates: dict[str, Any] = {}
    for metric in metric_keys:
        deltas = [float(row[metric]) for row in task_rows]
        aggregates[metric] = {
            "mean_task_paired_delta": mean(deltas),
            "task_cluster_bootstrap_95": task_bootstrap_ci(
                deltas,
                seed=comparison_seed(
                    seed, treatment["arm"], control["arm"], metric
                ),
                draws=draws,
            ),
        }

    task_count = len(task_rows)
    injected_count_equivalent = sum(
        row["injected_success_delta"] for row in task_rows
    )
    clean_count_equivalent = sum(row["clean_success_delta"] for row in task_rows)
    repeated_error_delta = mean(
        row["repeated_error_rate_delta"] for row in task_rows
    )
    gate = {
        "applies": apply_directional_gate,
        "screening_only": True,
        "required_independent_validation_tasks": EXPECTED_VALIDATION_TASKS,
        "observed_independent_validation_tasks": task_count,
        "minimum_injected_success_count_equivalent_delta": (
            GATE_MIN_INJECTED_COUNT_EQUIVALENT
        ),
        "observed_injected_success_count_equivalent_delta": (
            injected_count_equivalent
        ),
        "minimum_clean_success_count_equivalent_delta": (
            GATE_MIN_CLEAN_COUNT_EQUIVALENT
        ),
        "observed_clean_success_count_equivalent_delta": clean_count_equivalent,
        "repeated_identical_failed_call_must_not_increase": True,
        "observed_repeated_error_rate_delta": repeated_error_delta,
        "pass": (
            apply_directional_gate
            and task_count == EXPECTED_VALIDATION_TASKS
            and injected_count_equivalent
            >= GATE_MIN_INJECTED_COUNT_EQUIVALENT - 1e-12
            and clean_count_equivalent
            >= GATE_MIN_CLEAN_COUNT_EQUIVALENT - 1e-12
            and repeated_error_delta <= 1e-12
        ),
    }
    return {
        "treatment_arm": treatment["arm"],
        "control_arm": control["arm"],
        "independent_unit": "task_id",
        "independent_task_count": task_count,
        "rollouts_or_seeds_counted_as_independent": False,
        "aggregates": aggregates,
        "directional_gate": gate,
        "task_paired_deltas": task_rows,
    }


def parse_arm_specs(values: list[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"Arm must be NAME=DIR, got {value!r}")
        name, raw_path = value.split("=", 1)
        if name in arms:
            raise RuntimeError(f"Duplicate arm specification: {name}")
        arms[name] = Path(raw_path).expanduser().resolve()
    if set(arms) != set(EXPECTED_ARMS):
        raise RuntimeError(
            f"Expected exactly arms {EXPECTED_ARMS}, got {tuple(sorted(arms))}"
        )
    return arms


def summarize(
    *,
    split_manifest_path: Path,
    evaluation_manifest_path: Path,
    dynamic_audit_path: Path,
    checkpoint_registry_path: Path,
    arm_dirs: dict[str, Path],
    output_path: Path,
    control_arm: str = DEFAULT_CONTROL_ARM,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20260722,
    provenance_profile: str | None = None,
) -> dict[str, Any]:
    if set(arm_dirs) != set(EXPECTED_ARMS):
        raise RuntimeError(
            "The zero-shot base_model control and all four frozen trained arms "
            f"are required: {EXPECTED_ARMS}"
        )
    if control_arm != DEFAULT_CONTROL_ARM:
        raise RuntimeError(
            f"Frozen control arm is {DEFAULT_CONTROL_ARM}, got {control_arm}"
        )

    expected_rows, validation_ids, sealed_ids = (
        validate_split_and_evaluation_manifests(
            split_manifest_path, evaluation_manifest_path
        )
    )
    split_sha = sha256_file(split_manifest_path)
    evaluation_sha = sha256_file(evaluation_manifest_path)
    evaluation_payload = load_json(evaluation_manifest_path)
    dynamic_audit_identity = load_complete_dynamic_audit(
        dynamic_audit_path,
        manifest_path=evaluation_manifest_path,
        split_manifest_path=split_manifest_path,
        expected_source_split="derived_validation",
        expected_task_ids=set(validation_ids),
    )
    registry = load_checkpoint_registry(
        checkpoint_registry_path,
        expected_profile=provenance_profile,
    )
    registry_profile = checkpoint_registry_provenance_profile(registry)
    if (
        registry["training_data_provenance"]["dynamic_audits"]["validation"]
        != dynamic_audit_identity
    ):
        raise RuntimeError(
            "Checkpoint registry validation audit differs from evaluation audit"
        )
    registry_sha = sha256_file(checkpoint_registry_path)
    contract_audits = {
        arm: load_arm_contracts(
            arm=arm,
            arm_dir=arm_dirs[arm],
            registry=registry,
            registry_sha256=registry_sha,
            evaluation_manifest_sha256=evaluation_sha,
            split_manifest_sha256=split_sha,
            expected_task_ids=validation_ids,
            expected_fault_protocol=evaluation_payload["fault_protocol"],
            expected_tau2_commit=evaluation_payload["tau2_commit"],
            expected_source_files=evaluation_payload["source_files"],
            expected_dynamic_audit_identity=dynamic_audit_identity,
        )
        for arm in EXPECTED_ARMS
    }
    reference_contract = contract_audits["base_model"]
    reference_schedule = canonical_json(reference_contract["schedule"])
    reference_decoding = canonical_json(reference_contract["decoding"])
    reference_roles = canonical_json(reference_contract["roles"])
    for arm, audit in contract_audits.items():
        if canonical_json(audit["schedule"]) != reference_schedule:
            raise RuntimeError(
                f"Run contract task-shard schedule differs for {arm}"
            )
        if canonical_json(audit["decoding"]) != reference_decoding:
            raise RuntimeError(f"Run contract decoding differs for {arm}")
        if canonical_json(audit["roles"]) != reference_roles:
            raise RuntimeError(f"Run contract user/judge roles differ for {arm}")

    loaded = {
        arm: load_arm(arm, arm_dirs[arm], expected_rows, sealed_ids)
        for arm in EXPECTED_ARMS
    }

    # The single frozen evaluation seed/trial identity must match across all
    # five models task by task.
    reference = loaded[control_arm]["replicate_keys_by_task"]
    for arm, payload in loaded.items():
        if payload["replicate_keys_by_task"] != reference:
            raise RuntimeError(
                f"Evaluation replicate schedule differs between {arm} "
                f"and {control_arm}"
            )

    comparisons: dict[str, Any] = {}
    for control_index, control_name in enumerate(EXPECTED_ARMS):
        for treatment_name in EXPECTED_ARMS[control_index + 1 :]:
            key = f"{treatment_name}_minus_{control_name}"
            comparisons[key] = compare_arms(
                loaded[treatment_name],
                loaded[control_name],
                seed=bootstrap_seed,
                draws=bootstrap_draws,
                apply_directional_gate=(
                    control_name == control_arm
                    and treatment_name == DIRECTIONAL_CANDIDATE_ARM
                ),
            )

    arm_summaries = {
        arm: {
            **summarize_arm(loaded[arm]),
            "checkpoint_identity": contract_audits[arm],
        }
        for arm in EXPECTED_ARMS
    }
    all_contract_hashes = {
        arm: dict(sorted(contract_audits[arm]["contract_sha256"].items()))
        for arm in EXPECTED_ARMS
    }
    all_result_hashes = {
        arm: dict(sorted(contract_audits[arm]["result_sha256"].items()))
        for arm in EXPECTED_ARMS
    }
    provenance = {
        "summary_contract_protocol": SUMMARY_CONTRACT_PROTOCOL,
        "source": {
            "experiment_source_commit": registry["source_commit"],
            "base_model_revision": registry["base_model_revision"],
            "checkpoint_registry_provenance_profile": registry_profile,
            "tau2_commit": evaluation_payload["tau2_commit"],
            "tau2_source_files": evaluation_payload["source_files"],
        },
        "manifests": {
            "split_sha256": split_sha,
            "evaluation_sha256": evaluation_sha,
            "dynamic_audit_sha256": dynamic_audit_identity["sha256"],
            "checkpoint_registry_sha256": registry_sha,
        },
        "evaluation_contract_sha256": all_contract_hashes,
        "evaluation_result_sha256": all_result_hashes,
        "all_evaluation_contracts_sha256": hashlib.sha256(
            canonical_json(all_contract_hashes).encode("utf-8")
        ).hexdigest(),
        "all_evaluation_results_sha256": hashlib.sha256(
            canonical_json(all_result_hashes).encode("utf-8")
        ).hexdigest(),
        "tool_action_interface": dict(TOOL_ACTION_INTERFACE),
        "runtime_preflight": dict(EXPECTED_RUNTIME_PREFLIGHT),
        "completion_audits_recomputed": True,
    }
    provenance["input_binding_sha256"] = hashlib.sha256(
        canonical_json(provenance).encode("utf-8")
    ).hexdigest()
    summary = {
        "protocol": PROTOCOL,
        "summary_contract_protocol": SUMMARY_CONTRACT_PROTOCOL,
        "status": "PASS",
        "claim_boundary": "validation_directional_screen_only",
        "claim_scope": "multi_fault_family_post_fault_robustness_screen",
        "semantic_repair_claim_allowed": False,
        "fault_family_inference": "descriptive_only",
        "primary_metric": (
            "tau2_official_composite_task_success_error_condition"
        ),
        "control_arm": control_arm,
        "directional_candidate_arm": DIRECTIONAL_CANDIDATE_ARM,
        "base_model_is_evaluation_only": True,
        "independent_unit": "task_id",
        "independent_validation_tasks": len(validation_ids),
        "rollouts_or_seeds_counted_as_independent": False,
        "official_test": {
            "status": "SEALED",
            "sealed_task_count": len(sealed_ids),
            "used_for_selection_or_summary": False,
        },
        "manifest_sha256": {
            "split": split_sha,
            "evaluation": evaluation_sha,
            "dynamic_audit": dynamic_audit_identity["sha256"],
            "checkpoint_registry": registry_sha,
        },
        "provenance": provenance,
        "checkpoint_registry": {
            "protocol": registry["protocol"],
            "provenance_profile": registry_profile,
            "source_commit": registry["source_commit"],
            "base_model_revision": registry["base_model_revision"],
            "sha256": registry_sha,
            "model_ids_unique": True,
        },
        "bootstrap": {
            "unit": "task_id",
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
        },
        "arms": arm_summaries,
        "paired_comparisons": comparisons,
        "directional_gate_definition": {
            "reference_arm": control_arm,
            "candidate_arm": DIRECTIONAL_CANDIDATE_ARM,
            "injected_success": (
                "treatment must gain at least 2 task-success equivalents "
                "across the same 21 validation task IDs"
            ),
            "clean_retention": (
                "treatment may lose at most 1 task-success equivalent "
                "across the same 21 validation task IDs"
            ),
            "behavioral_safety": (
                "repeated identical failed-call rate must not increase"
            ),
            "not_a_final_claim": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "artifacts/v5_stage0/manifests/split_manifest.json"
        ),
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument(
        "--provenance-profile",
        choices=PROVENANCE_PROFILES,
        help=(
            "Optionally require the checkpoint registry's frozen legacy or "
            "V5.3 provenance profile."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help=(
            "Specify the zero-shot base_model control and each of the four "
            "frozen trained-arm result directories"
        ),
    )
    parser.add_argument("--control-arm", default=DEFAULT_CONTROL_ARM)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v5_stage1_sft_causal/mechanism_screen_summary.json"
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 0:
        raise SystemExit("--bootstrap-draws cannot be negative")
    try:
        arm_dirs = parse_arm_specs(args.arm)
        summary = summarize(
            split_manifest_path=args.split_manifest.expanduser().resolve(),
            evaluation_manifest_path=(
                args.evaluation_manifest.expanduser().resolve()
            ),
            dynamic_audit_path=args.dynamic_audit.expanduser().resolve(),
            checkpoint_registry_path=(
                args.checkpoint_registry.expanduser().resolve()
            ),
            arm_dirs=arm_dirs,
            output_path=args.output.expanduser().resolve(),
            control_arm=args.control_arm,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
            provenance_profile=args.provenance_profile,
        )
    except RuntimeError as exc:
        raise SystemExit(f"V5 Stage-1 aggregation failed closed: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
