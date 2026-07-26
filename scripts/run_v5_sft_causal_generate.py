#!/usr/bin/env python3
"""Generate paired inner-train trajectories for the Stage-1 SFT experiment.

The script supports deterministic task sharding so four machines can generate
disjoint task IDs.  It refuses validation/test manifests and records the agent,
user-simulator, judge, benchmark manifest, and shard in a run contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit
try:
    import v5_judge_audit_contract as judge_audit_contract
except ModuleNotFoundError:
    from scripts import v5_judge_audit_contract as judge_audit_contract

try:
    from v5_strict_nl_judge import (
        DEFAULT_CONTENT_ATTEMPTS,
        STRICT_NL_JUDGE_PROTOCOL,
        install_strict_nl_judge,
    )
except ModuleNotFoundError:
    from scripts.v5_strict_nl_judge import (
        DEFAULT_CONTENT_ATTEMPTS,
        STRICT_NL_JUDGE_PROTOCOL,
        install_strict_nl_judge,
    )


GENERATION_PROTOCOL = "v5_stage1_multifault_data_construction"
V5_3_GENERATION_PROTOCOL = "v5_3_multifault_data_construction"
SCREEN_MANIFEST_PROTOCOL = "v5_3_12h_exploratory_screen_v1:manifest_v1"
SCREEN_PROTOCOL = "v5_3_12h_exploratory_screen_v1"
SCREEN_GENERATION_SUBPROTOCOL = "v5_3_12h_screen_generation_v1"
FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
GT_COMPATIBILITY_PROTOCOL = "v5_gt_compatibility_preflight_v1"
GT_FILTER_PROTOCOL = "v5_3_gt_compatibility_filter_v1"
V5_3_TEMPERATURE = 0.2
V5_3_TOP_P = 0.95
V5_3_NUM_TRIALS = 12
V5_3_SEED = 20260722
SCREEN_TEMPERATURE = 0.2
SCREEN_TOP_P = 0.95
SCREEN_NUM_TRIALS = 6
SCREEN_SEED = 20260731
SCREEN_TRIAL_SEEDS = [25987, 293840, 725284, 249400, 591103, 257709]
SCREEN_NUM_SHARDS = 3
SCREEN_MAX_MODEL_LEN = 32768
SCREEN_MAX_TOKENS = 512
SCREEN_TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
SCREEN_TEACHER_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
SCREEN_USER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
SCREEN_USER_REVISION = "539535859b135b0244c91f3e59816150c8056698"
SCREEN_GPU_MODEL = "NVIDIA GeForce RTX 5090"
SCREEN_CUDA_VERSION = "12.8"
SCREEN_TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
SCREEN_SPLIT_SHA256 = (
    "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
)
SCREEN_RUNTIME_EVIDENCE_PROTOCOL = "v5_3_generation_runtime_preflight_v2"
SCREEN_TASK_IDS = {
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
}
ALLOWED_PROTOCOLS = {
    GENERATION_PROTOCOL,
    V5_3_GENERATION_PROTOCOL,
    SCREEN_MANIFEST_PROTOCOL,
}
SPLIT_PROTOCOL = "v5_stage0_tau2_end_to_end"
DOMAINS = ("retail", "airline")
EXPECTED_SPLIT_COUNTS = {
    "retail": {"inner_train": 59, "validation": 15, "sealed_test": 40},
    "airline": {"inner_train": 24, "validation": 6, "sealed_test": 20},
}
ROOT = Path(__file__).resolve().parents[1]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FAULT_INJECTIONS: dict[str, dict[str, Any]] = {}


class GTCompatibilityError(RuntimeError):
    """The frozen generation universe contains tasks a GT teacher cannot run."""

    def __init__(self, report: dict[str, Any]):
        incompatible = report["incompatible_task_ids"]
        declared = report.get("declared_incompatible_task_ids", [])
        super().__init__(
            "GT compatibility preflight failed before generation: "
            f"{len(incompatible)} task(s) require no expected tool action: "
            f"{', '.join(incompatible)}; frozen exclusions: "
            f"{', '.join(declared) if declared else '<none>'}. "
            "Revise and re-freeze the protocol task universe; do not convert "
            "these tasks into empty simulations."
        )
        self.report = report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-api-base", required=True)
    parser.add_argument("--teacher-api-key", default="stage1-teacher-local")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-revision", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--user-api-key", default="stage1-user-local")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision", required=True)
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-api-key")
    parser.add_argument(
        "--teacher-mode",
        choices=("standard", "ground_truth"),
        default="ground_truth",
    )
    parser.add_argument("--condition", choices=("clean", "error", "both"), default="both")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_tracked_source() -> None:
    """Fail closed on staged or unstaged tracked source drift."""

    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"repository contains {label} tracked source drift")


def require_clean_tracked_checkout(path: Path, *, label: str) -> None:
    for command, drift_kind in (
        (["git", "-C", str(path), "diff", "--quiet", "--"], "unstaged"),
        (
            ["git", "-C", str(path), "diff", "--cached", "--quiet", "--"],
            "staged",
        ),
    ):
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} contains {drift_kind} tracked source drift"
            )


def validate_provenance(
    *,
    expected_source_commit: str,
    teacher_revision: str,
    user_revision: str,
    judge_revision: str,
) -> str:
    values = {
        "--expected-source-commit": expected_source_commit,
        "--teacher-revision": teacher_revision,
        "--user-revision": user_revision,
        "--judge-revision": judge_revision,
    }
    for name, value in values.items():
        if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
            raise RuntimeError(f"{name} must be a full lowercase 40-character commit")
    require_clean_tracked_source()
    observed_source_commit = git_commit()
    if observed_source_commit != expected_source_commit:
        raise RuntimeError(
            "source commit drift: "
            f"HEAD={observed_source_commit}, expected={expected_source_commit}"
        )
    return observed_source_commit


def _identity_set(values: Any, *, domain: str, field: str) -> set[tuple[str, str]]:
    if not isinstance(values, list) or any(
        not isinstance(value, (str, int)) for value in values
    ):
        raise RuntimeError(f"Split {domain}.{field} must be a list of task IDs")
    normalized = {(domain, str(value)) for value in values}
    if len(normalized) != len(values):
        raise RuntimeError(f"Split {domain}.{field} contains duplicate task IDs")
    return normalized


def load_split_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != SPLIT_PROTOCOL:
        raise RuntimeError(f"Expected split protocol {SPLIT_PROTOCOL!r}")
    guarantees = payload.get("guarantees")
    if not isinstance(guarantees, dict):
        raise RuntimeError("Split manifest lacks guarantees")
    if guarantees.get("validation_derived_from_official_train_only") is not True:
        raise RuntimeError("Split does not certify validation-from-train")
    if guarantees.get("official_test_task_content_exported") is not False:
        raise RuntimeError("Split does not certify the official-test seal")

    domains = payload.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise RuntimeError(f"Split domains must be exactly {DOMAINS}")
    split_sets: dict[str, set[tuple[str, str]]] = {}
    for domain in DOMAINS:
        domain_payload = domains.get(domain)
        if not isinstance(domain_payload, dict):
            raise RuntimeError(f"Split manifest lacks domain {domain}")
        split_sets_for_domain = {
            "inner_train": _identity_set(
                domain_payload.get("inner_train_ids"),
                domain=domain,
                field="inner_train_ids",
            ),
            "validation": _identity_set(
                domain_payload.get("validation_ids"),
                domain=domain,
                field="validation_ids",
            ),
            "sealed_test": _identity_set(
                domain_payload.get("sealed_test_ids"),
                domain=domain,
                field="sealed_test_ids",
            ),
        }
        expected = EXPECTED_SPLIT_COUNTS[domain]
        observed = {
            name: len(values) for name, values in split_sets_for_domain.items()
        }
        if observed != expected:
            raise RuntimeError(
                f"Frozen split count drift for {domain}: "
                f"expected {expected}, observed {observed}"
            )
        names = tuple(split_sets_for_domain)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if split_sets_for_domain[left] & split_sets_for_domain[right]:
                    raise RuntimeError(
                        f"Split overlap in {domain}: {left} intersects {right}"
                    )
        for name, values in split_sets_for_domain.items():
            split_sets.setdefault(name, set()).update(values)
    payload["_validated_identity_sets"] = split_sets
    return payload


def load_inner_train_manifest(
    path: Path,
    *,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if protocol not in ALLOWED_PROTOCOLS:
        raise RuntimeError("Unexpected inner-train generation manifest protocol")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Generation manifest has no rows")
    if payload.get("paired_task_count") != len(rows):
        raise RuntimeError("Generation manifest count drift")
    if protocol == SCREEN_MANIFEST_PROTOCOL:
        core = {
            key: value
            for key, value in payload.items()
            if key != "canonical_sha256"
        }
        row_task_ids = [
            f"{row.get('domain')}:{row.get('task_id')}" for row in rows
        ]
        planned_task_ids = payload.get("planned_task_ids")
        source_generation_sha = payload.get(
            "source_generation_manifest_sha256"
        )
        generation = payload.get("generation")
        teacher = generation.get("teacher") if isinstance(generation, dict) else None
        user = generation.get("user") if isinstance(generation, dict) else None
        judge = generation.get("judge") if isinstance(generation, dict) else None
        if (
            payload.get("screen_protocol") != SCREEN_PROTOCOL
            or payload.get("design_version") != "5.3-12h-screen"
            or payload.get("formal_v5_3_data") is not False
            or payload.get("formal_data") is not False
            or payload.get("screen_outputs_may_enter_formal") is not False
            or payload.get("screen_outputs_may_enter_formal_v5_3") is not False
            or payload.get("official_test_used") is not False
            or payload.get("official_test_sealed") is not True
            or payload.get("source_split") != "derived_inner_train"
            or payload.get("base_seed") != SCREEN_SEED
            or payload.get("trial_seeds") != SCREEN_TRIAL_SEEDS
            or payload.get("attempts_per_task_per_condition")
            != SCREEN_NUM_TRIALS
            or payload.get("conditions") != ["clean", "error"]
            or payload.get("expected_rollouts")
            != len(SCREEN_TASK_IDS) * 2 * SCREEN_NUM_TRIALS
            or payload.get("formal_v5_3_seed_or_bytes_reused") is not False
            or payload.get("replacement_or_rescue_attempts") is not False
            or not isinstance(planned_task_ids, list)
            or row_task_ids != planned_task_ids
            or set(planned_task_ids) != SCREEN_TASK_IDS
            or len(planned_task_ids) != len(SCREEN_TASK_IDS)
            or not is_sha256(source_generation_sha)
            or payload.get("generation_manifest_sha256")
            != source_generation_sha
            or payload.get("tau2_commit") != SCREEN_TAU2_COMMIT
            or payload.get("split_manifest_sha256") != SCREEN_SPLIT_SHA256
            or not isinstance(generation, dict)
            or not isinstance(teacher, dict)
            or teacher.get("model") != SCREEN_TEACHER_MODEL
            or teacher.get("revision") != SCREEN_TEACHER_REVISION
            or not isinstance(user, dict)
            or user.get("model") != SCREEN_USER_MODEL
            or user.get("revision") != SCREEN_USER_REVISION
            or not isinstance(judge, dict)
            or judge.get("model") != SCREEN_USER_MODEL
            or judge.get("revision") != SCREEN_USER_REVISION
            or generation.get("teacher_mode") != "ground_truth"
            or generation.get("temperature") != SCREEN_TEMPERATURE
            or generation.get("top_p") != SCREEN_TOP_P
            or generation.get("max_model_len") != SCREEN_MAX_MODEL_LEN
            or generation.get("max_tokens") != SCREEN_MAX_TOKENS
            or generation.get("num_shards") != SCREEN_NUM_SHARDS
            or generation.get("num_trials") != SCREEN_NUM_TRIALS
            or generation.get("base_seed") != SCREEN_SEED
            or generation.get("trial_seeds") != SCREEN_TRIAL_SEEDS
            or payload.get("canonical_sha256") != canonical_sha256(core)
        ):
            raise RuntimeError("V5.3 12-hour screen manifest contract drift")
    fault_protocol = payload.get("fault_protocol")
    if (
        not isinstance(fault_protocol, dict)
        or fault_protocol.get("protocol") != FAULT_PROTOCOL
        or fault_protocol.get("claim_scope")
        != "multi_fault_family_post_fault_robustness_screen"
    ):
        raise RuntimeError("Generation manifest lacks frozen multi-fault protocol")
    raw_catalog = fault_protocol.get("families")
    if not isinstance(raw_catalog, dict) or set(raw_catalog) != {
        "retail",
        "airline",
    }:
        raise RuntimeError("Generation fault catalog domain drift")
    catalog: dict[str, dict[str, tuple[str, str]]] = {}
    for domain in ("retail", "airline"):
        family_rows = raw_catalog.get(domain)
        if not isinstance(family_rows, list) or len(family_rows) < 2:
            raise RuntimeError(f"{domain}: insufficient fault catalog")
        catalog[domain] = {}
        for family_row in family_rows:
            if not isinstance(family_row, dict):
                raise RuntimeError(f"{domain}: malformed fault catalog")
            family = family_row.get("fault_family")
            tool = family_row.get("tool_name")
            invalid_key = family_row.get("invalid_argument_key")
            if not all(
                isinstance(value, str) and value
                for value in (family, tool, invalid_key)
            ):
                raise RuntimeError(f"{domain}: incomplete fault catalog")
            if family in catalog[domain]:
                raise RuntimeError(f"{domain}: duplicate fault family")
            catalog[domain][family] = (tool, invalid_key)
    pair_ids: set[str] = set()
    task_keys: set[tuple[str, str]] = set()
    tool_call_ids: set[str] = set()
    invalid_parameters: set[str] = set()
    families: dict[str, set[str]] = {"retail": set(), "airline": set()}
    tools: dict[str, set[str]] = {"retail": set(), "airline": set()}
    for index, row in enumerate(rows):
        if row.get("source_split") != "derived_inner_train":
            raise RuntimeError(
                f"Generation row {index} is not from derived_inner_train"
            )
        domain = row.get("domain")
        task_id = str(row.get("task_id"))
        pair_id = row.get("pair_id")
        if domain not in {"retail", "airline"} or not pair_id:
            raise RuntimeError(f"Generation row {index} has invalid identity")
        if pair_id in pair_ids or (domain, task_id) in task_keys:
            raise RuntimeError("Generation manifest contains duplicate tasks")
        pair_ids.add(pair_id)
        task_keys.add((domain, task_id))
        clean = row.get("clean_condition") or {}
        error = row.get("error_condition") or {}
        if clean.get("inject_error") is not False:
            raise RuntimeError(f"Generation row {index} has invalid clean condition")
        if (
            error.get("inject_error") is not True
            or error.get("expected_tool_error") is not True
            or error.get("expected_state_mutation") is not False
        ):
            raise RuntimeError(f"Generation row {index} has unsafe error condition")
        if not error.get("tool_name") or not error.get("tool_call_id"):
            raise RuntimeError(f"Generation row {index} lacks injection identity")
        if not isinstance(error.get("arguments"), dict):
            raise RuntimeError(f"Generation row {index} lacks injection arguments")
        family = error.get("fault_family")
        specification = catalog[domain].get(family)
        if specification is None:
            raise RuntimeError(f"Generation row {index} has unknown fault family")
        tool_name, invalid_key = specification
        if (
            error.get("tool_name") != tool_name
            or error.get("tool_type") != "READ"
            or error.get("invalid_argument_key") != invalid_key
            or not isinstance(error["arguments"].get(invalid_key), str)
        ):
            raise RuntimeError(f"Generation row {index} fault schema drift")
        call_id = error["tool_call_id"]
        if call_id in tool_call_ids:
            raise RuntimeError("Generation fault call IDs are not task-unique")
        tool_call_ids.add(call_id)
        invalid_identity = json.dumps(
            {invalid_key: error["arguments"][invalid_key]},
            sort_keys=True,
            separators=(",", ":"),
        )
        if invalid_identity in invalid_parameters:
            raise RuntimeError("Generation invalid parameters are not task-unique")
        invalid_parameters.add(invalid_identity)
        if type(error.get("on_reference_path")) is not bool:
            raise RuntimeError(f"Generation row {index} lacks relevance evidence")
        if error.get("fault_relevance") not in {
            "reference_path_or_operation_aligned",
            "domain_plausible_fallback",
        }:
            raise RuntimeError(f"Generation row {index} relevance drift")
        families[domain].add(family)
        tools[domain].add(tool_name)
    expected_inner = split_manifest["_validated_identity_sets"]["inner_train"]
    expected_validation = split_manifest["_validated_identity_sets"]["validation"]
    expected_sealed = split_manifest["_validated_identity_sets"]["sealed_test"]
    observed = {(str(row["domain"]), str(row["task_id"])) for row in rows}
    leaked = observed & (expected_validation | expected_sealed)
    if leaked:
        raise RuntimeError(
            "Generation manifest leaks validation/test task IDs: "
            f"{sorted(leaked)[:5]}"
        )
    if protocol == GENERATION_PROTOCOL:
        if observed != expected_inner:
            raise RuntimeError(
                "Legacy generation manifest must cover exactly the frozen "
                "inner-train IDs; "
                f"missing={sorted(expected_inner - observed)[:5]}, "
                f"extra={sorted(observed - expected_inner)[:5]}"
            )
        if payload.get("gt_compatibility_filter") is not None:
            raise RuntimeError(
                "Legacy generation manifest cannot declare a V5.3 GT filter"
            )
        payload["_validated_gt_compatibility_filter"] = None
    elif protocol == V5_3_GENERATION_PROTOCOL:
        gt_filter = payload.get("gt_compatibility_filter")
        if not isinstance(gt_filter, dict):
            raise RuntimeError(
                "V5.3 generation manifest lacks frozen gt_compatibility_filter"
            )
        excluded_values = gt_filter.get("excluded_task_ids")
        if not isinstance(excluded_values, list) or any(
            not isinstance(value, str) or ":" not in value
            for value in excluded_values
        ):
            raise RuntimeError(
                "V5.3 gt_compatibility_filter.excluded_task_ids is malformed"
            )
        if len(set(excluded_values)) != len(excluded_values):
            raise RuntimeError(
                "V5.3 gt_compatibility_filter contains duplicate exclusions"
            )
        excluded: set[tuple[str, str]] = set()
        for value in excluded_values:
            domain, task_id = value.split(":", 1)
            if domain not in DOMAINS or not task_id:
                raise RuntimeError(
                    f"V5.3 GT compatibility exclusion is invalid: {value!r}"
                )
            excluded.add((domain, task_id))
        required_metadata = {
            "protocol": GT_FILTER_PROTOCOL,
            "policy": "exclude_before_sharding",
            "teacher_mode": "ground_truth",
            "source_task_count": len(expected_inner),
            "included_task_count": len(observed),
            "exclusion_reason": "no_expected_tool_actions",
            "selection_uses_rollouts_rewards_validation_or_test": False,
            "official_test_used": False,
        }
        for field, expected_value in required_metadata.items():
            if gt_filter.get(field) != expected_value:
                raise RuntimeError(
                    "V5.3 gt_compatibility_filter metadata drift for "
                    f"{field}: expected {expected_value!r}, "
                    f"observed {gt_filter.get(field)!r}"
                )
        if observed & excluded:
            raise RuntimeError(
                "V5.3 included rows overlap frozen GT-incompatible task IDs"
            )
        if observed | excluded != expected_inner:
            raise RuntimeError(
                "V5.3 included and excluded tasks must partition the complete "
                "frozen inner-train universe; "
                f"missing={sorted(expected_inner - (observed | excluded))[:5]}, "
                f"extra={sorted((observed | excluded) - expected_inner)[:5]}"
            )
        if not excluded:
            raise RuntimeError(
                "V5.3 gt_compatibility_filter must freeze a nonempty exclusion set"
            )
        payload["_validated_gt_compatibility_filter"] = {
            **gt_filter,
            "excluded_task_ids": sorted(excluded_values),
        }
    elif protocol == SCREEN_MANIFEST_PROTOCOL:
        gt_filter = payload.get("gt_compatibility_filter")
        if not isinstance(gt_filter, dict):
            raise RuntimeError(
                "V5.3 12-hour screen manifest lacks frozen "
                "gt_compatibility_filter"
            )
        excluded_values = gt_filter.get("excluded_task_ids")
        if not isinstance(excluded_values, list) or any(
            not isinstance(value, str) or ":" not in value
            for value in excluded_values
        ):
            raise RuntimeError(
                "V5.3 12-hour screen GT exclusions are malformed"
            )
        if len(set(excluded_values)) != len(excluded_values):
            raise RuntimeError(
                "V5.3 12-hour screen GT exclusions contain duplicates"
            )
        excluded: set[tuple[str, str]] = set()
        for value in excluded_values:
            domain, task_id = value.split(":", 1)
            if domain not in DOMAINS or not task_id:
                raise RuntimeError(
                    f"V5.3 12-hour screen GT exclusion is invalid: {value!r}"
                )
            excluded.add((domain, task_id))
        required_metadata = {
            "protocol": GT_FILTER_PROTOCOL,
            "policy": "exclude_before_sharding",
            "teacher_mode": "ground_truth",
            "source_task_count": len(expected_inner),
            "included_task_count": len(expected_inner) - len(excluded),
            "exclusion_reason": "no_expected_tool_actions",
            "selection_uses_rollouts_rewards_validation_or_test": False,
            "official_test_used": False,
        }
        for field, expected_value in required_metadata.items():
            if gt_filter.get(field) != expected_value:
                raise RuntimeError(
                    "V5.3 12-hour screen gt_compatibility_filter drift for "
                    f"{field}: expected {expected_value!r}, "
                    f"observed {gt_filter.get(field)!r}"
                )
        if (
            observed != {
                tuple(value.split(":", 1)) for value in SCREEN_TASK_IDS
            }
            or not observed <= expected_inner
            or not excluded <= expected_inner
            or observed & excluded
        ):
            raise RuntimeError(
                "V5.3 12-hour screen task/filter universe drift"
            )
        payload["_validated_gt_compatibility_filter"] = {
            **gt_filter,
            "excluded_task_ids": sorted(excluded_values),
        }
    else:  # guarded above; keeps static analyzers and future edits fail-closed.
        raise RuntimeError("Unexpected generation manifest protocol")
    declared_split_sha = payload.get("split_manifest_sha256")
    if declared_split_sha != split_manifest_sha256:
        raise RuntimeError("Generation manifest split_manifest_sha256 drift")
    for domain in ("retail", "airline"):
        if len(families[domain]) < 2 or len(tools[domain]) < 2:
            raise RuntimeError(f"{domain}: generation lacks multi-fault diversity")
    return payload


def validate_local_manifest_sources(
    payload: dict[str, Any], tau2_root: Path
) -> None:
    require_clean_tracked_checkout(tau2_root, label="tau2 checkout")
    observed_commit = subprocess.run(
        ["git", "-C", str(tau2_root), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if payload.get("tau2_commit") != observed_commit:
        raise RuntimeError(
            f"Local tau2 commit {observed_commit} differs from manifest "
            f"{payload.get('tau2_commit')}"
        )
    sources = payload.get("source_files")
    if not isinstance(sources, dict):
        raise RuntimeError("Generation manifest lacks source file hashes")
    for domain in ("retail", "airline"):
        declared = sources.get(domain)
        if not isinstance(declared, dict):
            raise RuntimeError(f"Generation manifest lacks {domain} source hashes")
        paths = {
            "tasks_json_sha256": (
                tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
            ),
            "db_json_sha256": (
                tau2_root / "data" / "tau2" / "domains" / domain / "db.json"
            ),
            "tools_py_sha256": (
                tau2_root / "src" / "tau2" / "domains" / domain / "tools.py"
            ),
        }
        for field, source_path in paths.items():
            if not source_path.is_file() or declared.get(field) != sha256_file(
                source_path
            ):
                raise RuntimeError(
                    f"Local {domain} {field} differs from multi-fault manifest"
                )


def load_screen_source_dynamic_audit(
    path: Path,
    *,
    manifest: dict[str, Any],
    split_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the full source-generation audit bound by a screen manifest."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("screen source dynamic audit is unreadable") from error
    rows = payload.get("rows") if isinstance(payload, dict) else None
    gt_filter = manifest.get("gt_compatibility_filter") or {}
    expected_count = gt_filter.get("included_task_count")
    source_generation_sha = manifest.get(
        "source_generation_manifest_sha256"
    )
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != "v5_stage1_dynamic_injection_audit"
        or payload.get("status") != "COMPLETE"
        or payload.get("manifest_sha256") != source_generation_sha
        or payload.get("manifest_protocol") != V5_3_GENERATION_PROTOCOL
        or payload.get("fault_protocol") != FAULT_PROTOCOL
        or payload.get("split_manifest_sha256")
        != sha256_file(split_manifest_path)
        or payload.get("source_split") != "derived_inner_train"
        or payload.get("official_test_used") is not False
        or payload.get("official_test_sealed") is not True
        or not isinstance(rows, list)
        or payload.get("verified_injections") != len(rows)
        or len(rows) != expected_count
    ):
        raise RuntimeError("screen source dynamic audit identity drift")
    observed: set[str] = set()
    for index, row in enumerate(rows):
        pair_id = row.get("pair_id") if isinstance(row, dict) else None
        if (
            not isinstance(pair_id, str)
            or pair_id in observed
            or row.get("runtime_tool_type") != "read"
            or row.get("runtime_tool_mutates_state") is not False
            or row.get("tool_error_observed") is not True
            or row.get("agent_database_before") is None
            or row.get("agent_database_before")
            != row.get("agent_database_after")
            or "user_database_before" not in row
            or row.get("user_database_before")
            != row.get("user_database_after")
            or row.get("agent_database_unchanged") is not True
            or row.get("user_database_unchanged") is not True
        ):
            raise RuntimeError(
                f"screen source dynamic audit row {index} failed safety"
            )
        observed.add(pair_id)
    if not SCREEN_TASK_IDS <= observed:
        raise RuntimeError(
            "screen tasks are absent from the source dynamic audit"
        )
    if any(
        payload.get(field) is not True
        for field in (
            "all_runtime_tools_read_only",
            "all_tool_errors_observed",
            "all_agent_databases_unchanged",
            "all_user_databases_unchanged",
        )
    ):
        raise RuntimeError("screen source dynamic audit aggregate drift")
    return {
        "protocol": "v5_stage1_dynamic_injection_audit",
        "sha256": sha256_file(path),
        "manifest_sha256": source_generation_sha,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "source_split": "derived_inner_train",
        "verified_injections": len(rows),
        "official_test_used": False,
        "official_test_sealed": True,
    }


def shard_rows(
    rows: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row["pair_id"]).encode("utf-8")).hexdigest(),
            str(row["pair_id"]),
        ),
    )
    return ordered[shard_index::num_shards]


def configure_tau2_path(tau2_root: Path) -> None:
    source = tau2_root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"tau2 source directory not found: {source}")
    sys.path.insert(0, str(source))


def normalize_tool_only_message(message: Any) -> Any:
    """Enforce one tool-only action before the environment executes it.

    vLLM can preserve pre-tool prose and, for some Qwen responses, ignore the
    OpenAI ``parallel_tool_calls=false`` hint. The ordered first call is
    executed and the model replans after its result. Hashes of deferred calls
    and removed text are retained in raw_data.
    """

    if message.tool_calls:
        raw_data = dict(message.raw_data or {})
        if len(message.tool_calls) > 1:
            deferred = message.tool_calls[1:]
            raw_data["v5_stage1_parallel_calls_serialized"] = {
                "original_count": len(message.tool_calls),
                "deferred_call_sha256": [
                    hashlib.sha256(
                        json.dumps(
                            call.model_dump(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    for call in deferred
                ],
            }
            message.tool_calls = message.tool_calls[:1]
        if message.content in (None, ""):
            message.raw_data = raw_data
            return message
        content = message.content
        raw_data["v5_stage1_mixed_content_normalized"] = {
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "utf8_bytes": len(content.encode("utf-8")),
        }
        message.raw_data = raw_data
        message.content = None
    return message


def register_fault_agents() -> None:
    from tau2.agent.llm_agent import LLMAgent, LLMGTAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    class Stage1Agent(LLMAgent):
        def _generate_next_message(self, message, state):
            return normalize_tool_only_message(
                super()._generate_next_message(message, state)
            )

    class Stage1GTAgent(LLMGTAgent):
        def generate_next_message(self, message, state):
            assistant, state = super().generate_next_message(message, state)
            return normalize_tool_only_message(assistant), state

    def injected_message(injection: dict[str, Any], *, ground_truth: bool):
        raw_data = {
            "v5_stage1_injected_fault": True,
            "expected_tool_error": True,
        }
        if ground_truth:
            raw_data["teacher_guidance"] = "ground_truth_resolution_steps"
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=injection["tool_call_id"],
                    name=injection["tool_name"],
                    arguments=injection["arguments"],
                    requestor="assistant",
                )
            ],
            cost=0.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            raw_data=raw_data,
            generation_time_seconds=0.0,
        )

    class Stage1FaultAgent(Stage1Agent):
        def __init__(self, tools, domain_policy, task, llm, llm_args):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=llm_args,
            )
            self._injection = FAULT_INJECTIONS.get(str(task.id))
            if self._injection is None:
                raise RuntimeError(f"Task {task.id} lacks Stage-1 injection data")
            self._injected = False

        def _generate_next_message(self, message, state):
            if not self._injected and isinstance(message, UserMessage):
                state.messages.append(message)
                self._injected = True
                return injected_message(self._injection, ground_truth=False)
            return super()._generate_next_message(message, state)

    class Stage1FaultGTAgent(Stage1GTAgent):
        def __init__(self, tools, domain_policy, task, llm, llm_args):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                task=task,
                llm=llm,
                llm_args=llm_args,
            )
            self._injection = FAULT_INJECTIONS.get(str(task.id))
            if self._injection is None:
                raise RuntimeError(f"Task {task.id} lacks Stage-1 injection data")
            self._injected = False

        def generate_next_message(self, message, state):
            if not self._injected and isinstance(message, UserMessage):
                state.messages.append(message)
                self._injected = True
                assistant = injected_message(self._injection, ground_truth=True)
                state.messages.append(assistant)
                return assistant, state
            return super().generate_next_message(message, state)

    def create_fault_agent(tools, domain_policy, **kwargs):
        return Stage1FaultAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    def create_gt_agent(tools, domain_policy, **kwargs):
        return Stage1GTAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    def create_fault_gt_agent(tools, domain_policy, **kwargs):
        return Stage1FaultGTAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory("v5_stage1_generation_fault_agent") is None:
        registry.register_agent_factory(
            create_fault_agent, "v5_stage1_generation_fault_agent"
        )
    if registry.get_agent_factory("v5_stage1_generation_gt_agent") is None:
        registry.register_agent_factory(
            create_gt_agent,
            "v5_stage1_generation_gt_agent",
            task_filter=LLMGTAgent.check_valid_task,
        )
    if registry.get_agent_factory("v5_stage1_generation_fault_gt_agent") is None:
        registry.register_agent_factory(
            create_fault_gt_agent,
            "v5_stage1_generation_fault_gt_agent",
            task_filter=LLMGTAgent.check_valid_task,
        )


def patch_local_nl_judge(model: str, llm_args: dict[str, Any]) -> None:
    install_strict_nl_judge(
        model=model,
        llm_args=llm_args,
        max_content_attempts=DEFAULT_CONTENT_ATTEMPTS,
    )


def select_tasks(domain: str, rows: list[dict[str, Any]]):
    from tau2.registry import registry

    all_tasks = {
        str(task.id): task for task in registry.get_tasks_loader(domain)(None)
    }
    tasks = []
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in all_tasks:
            raise RuntimeError(f"Missing {domain} inner-train task {task_id}")
        tasks.append(all_tasks[task_id])
    return tasks


def preflight_gt_compatibility(
    rows: list[dict[str, Any]],
    *,
    teacher_mode: str,
    gt_compatibility_filter: dict[str, Any] | None = None,
    task_selector: Any | None = None,
    compatibility_check: Any | None = None,
) -> dict[str, Any]:
    """Fail before any rollout if a GT teacher cannot run the frozen universe.

    This validates the complete manifest, not just the current shard, so all
    workers make the same decision and no subset starts expensive generation
    while another subset discovers an actionless task.
    """

    if teacher_mode not in {"standard", "ground_truth"}:
        raise ValueError(f"Unknown teacher mode {teacher_mode!r}")
    if gt_compatibility_filter is not None and teacher_mode != "ground_truth":
        raise RuntimeError(
            "A V5.3 ground-truth compatibility filter cannot be used with "
            f"teacher_mode={teacher_mode!r}"
        )
    declared_incompatible = sorted(
        (gt_compatibility_filter or {}).get("excluded_task_ids", [])
    )
    report: dict[str, Any] = {
        "protocol": GT_COMPATIBILITY_PROTOCOL,
        "teacher_mode": teacher_mode,
        "checked_scope": "complete_generation_manifest",
        "included_manifest_task_count": len(rows),
        "checked_task_count": len(rows) + len(declared_incompatible),
        "compatible_task_count": len(rows),
        "incompatible_task_ids": [],
        "declared_incompatible_task_ids": declared_incompatible,
        "filter_protocol": (gt_compatibility_filter or {}).get("protocol"),
        "filter_verified": False,
        "policy": "fail_fast_and_require_preregistered_manifest_revision",
        "status": "NOT_APPLICABLE" if teacher_mode == "standard" else "PASS",
    }
    if teacher_mode == "standard":
        return report

    if task_selector is None:
        task_selector = select_tasks
    if compatibility_check is None:
        from tau2.agent.llm_agent import LLMGTAgent

        compatibility_check = LLMGTAgent.check_valid_task

    complete_rows = list(rows)
    for value in declared_incompatible:
        domain, task_id = value.split(":", 1)
        complete_rows.append({"domain": domain, "task_id": task_id})
    identities = [
        f"{row.get('domain')}:{row.get('task_id')}" for row in complete_rows
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError(
            "GT compatibility preflight received overlapping included/excluded tasks"
        )

    incompatible: list[str] = []
    checked = 0
    for domain in DOMAINS:
        domain_rows = [
            row for row in complete_rows if row.get("domain") == domain
        ]
        if not domain_rows:
            continue
        tasks = list(task_selector(domain, domain_rows))
        if len(tasks) != len(domain_rows):
            raise RuntimeError(
                f"GT compatibility selector count drift for {domain}: "
                f"expected {len(domain_rows)}, observed {len(tasks)}"
            )
        for row, task in zip(domain_rows, tasks, strict=True):
            row_task_id = str(row.get("task_id"))
            if str(task.id) != row_task_id:
                raise RuntimeError(
                    f"GT compatibility selector order drift for {domain}: "
                    f"expected task {row_task_id}, observed {task.id}"
                )
            checked += 1
            if compatibility_check(task) is not True:
                incompatible.append(f"{domain}:{row_task_id}")

    report["checked_task_count"] = checked
    report["compatible_task_count"] = checked - len(incompatible)
    report["incompatible_task_ids"] = sorted(incompatible)
    if sorted(incompatible) != declared_incompatible:
        report["status"] = "FAIL"
        raise GTCompatibilityError(report)
    report["filter_verified"] = bool(gt_compatibility_filter)
    return report


def derived_trial_seeds(seed: int, num_trials: int) -> list[int]:
    """Mirror tau2.run_tasks' deterministic per-trial seed schedule."""

    if type(seed) is not int or type(num_trials) is not int or num_trials < 1:
        raise ValueError("seed and positive integer num_trials are required")
    generator = random.Random(seed)
    seeds = [generator.randint(0, 1_000_000) for _ in range(num_trials)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("derived per-trial seeds are not distinct")
    return seeds


def validate_sampling_contract(
    *,
    manifest_protocol: str,
    temperature: float,
    top_p: float,
    num_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Freeze stochastic V5.3 attempts while preserving legacy V5 defaults."""

    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be numeric")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError("top_p must be numeric")
    temperature = float(temperature)
    top_p = float(top_p)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2]")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    seeds = derived_trial_seeds(seed, num_trials)

    if manifest_protocol == V5_3_GENERATION_PROTOCOL:
        expected = {
            "temperature": V5_3_TEMPERATURE,
            "top_p": V5_3_TOP_P,
            "num_trials": V5_3_NUM_TRIALS,
            "seed": V5_3_SEED,
        }
        observed = {
            "temperature": temperature,
            "top_p": top_p,
            "num_trials": num_trials,
            "seed": seed,
        }
        if observed != expected:
            raise RuntimeError(
                "V5.3 generation sampling contract drift: "
                f"expected {expected}, observed {observed}"
            )
        if temperature <= 0:
            raise RuntimeError(
                "V5.3 requires nonzero temperature for independent attempts"
            )
    elif manifest_protocol == SCREEN_MANIFEST_PROTOCOL:
        expected = {
            "temperature": SCREEN_TEMPERATURE,
            "top_p": SCREEN_TOP_P,
            "num_trials": SCREEN_NUM_TRIALS,
            "seed": SCREEN_SEED,
        }
        observed = {
            "temperature": temperature,
            "top_p": top_p,
            "num_trials": num_trials,
            "seed": seed,
        }
        if observed != expected or seeds != SCREEN_TRIAL_SEEDS:
            raise RuntimeError(
                "V5.3 12-hour screen sampling contract drift: "
                f"expected {expected} with trial seeds "
                f"{SCREEN_TRIAL_SEEDS}, observed {observed} with "
                f"trial seeds {seeds}"
            )
    elif manifest_protocol != GENERATION_PROTOCOL:
        raise RuntimeError(
            f"Unsupported generation sampling protocol {manifest_protocol!r}"
        )

    return {
        "protocol": (
            "v5_3_frozen_stochastic_attempts_v1"
            if manifest_protocol == V5_3_GENERATION_PROTOCOL
            else (
                "v5_3_12h_screen_stochastic_attempts_v1"
                if manifest_protocol == SCREEN_MANIFEST_PROTOCOL
                else "v5_legacy_generation_sampling_v1"
            )
        ),
        "temperature": temperature,
        "top_p": top_p,
        "base_seed": seed,
        "num_trials": num_trials,
        "derived_trial_seeds": seeds,
        "derived_trial_seeds_are_distinct": True,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
    }


def endpoint_args(
    api_base: str,
    api_key: str,
    *,
    max_tokens: int,
    seed: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    return {
        "api_base": api_base,
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "parallel_tool_calls": False,
    }


def litellm_openai_model(model: str) -> str:
    """Route a frozen vLLM model name through LiteLLM's OpenAI provider.

    The scientific contract records the underlying Hugging Face/vLLM model
    name without a provider prefix.  tau2 calls LiteLLM, however, and a custom
    OpenAI-compatible ``api_base`` must use the explicit ``openai/`` provider
    prefix.  Keeping that translation here prevents a local vLLM alias from
    being mistaken for a hosted provider model.
    """

    if not isinstance(model, str) or not model or model.startswith("openai/"):
        raise RuntimeError(
            "Generation model contracts must use the raw served model name; "
            "the runner adds LiteLLM's openai/ prefix"
        )
    return f"openai/{model}"


def output_filename(
    domain: str, condition: str, shard_index: int, num_shards: int
) -> str:
    if num_shards == 1:
        return f"{domain}_{condition}.json"
    return (
        f"{domain}_{condition}.shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )


def run_condition(
    *,
    domain: str,
    rows: list[dict[str, Any]],
    condition: str,
    args: argparse.Namespace,
    teacher_args: dict[str, Any],
    user_args: dict[str, Any],
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    if args.teacher_mode == "ground_truth":
        agent_name = (
            "v5_stage1_generation_gt_agent"
            if condition == "clean"
            else "v5_stage1_generation_fault_gt_agent"
        )
    else:
        agent_name = (
            "llm_agent"
            if condition == "clean"
            else "v5_stage1_generation_fault_agent"
        )
    if condition == "error":
        FAULT_INJECTIONS.clear()
        FAULT_INJECTIONS.update(
            {
                str(row["task_id"]): dict(row["error_condition"])
                for row in rows
            }
        )

    output_path = args.output_dir / output_filename(
        domain, condition, args.shard_index, args.num_shards
    )
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result: {output_path}")
    save_dir = args.output_dir / "logs" / output_path.stem
    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=litellm_openai_model(args.teacher_model),
        llm_args_agent=teacher_args,
        llm_user=litellm_openai_model(args.user_model),
        llm_args_user=user_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=10,
        timeout=args.timeout,
        max_concurrency=1,
        seed=args.seed,
        log_level="INFO",
        max_retries=1,
        retry_delay=1.0,
        auto_resume=False,
        hallucination_retries=0,
        enforce_communication_protocol=False,
        verbose_logs=True,
    )
    run_tasks(
        config,
        select_tasks(domain, rows),
        save_path=output_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=True,
        results_format="json",
    )
    if not output_path.is_file():
        raise RuntimeError(f"tau2-bench did not write {output_path}")
    return output_path


def validate_screen_runtime_evidence(
    path: Path | None,
    *,
    expected_source_commit: str,
    max_model_len: int | None,
) -> dict[str, Any]:
    """Validate and hash-bind the live 32K-context screen service receipt."""

    if max_model_len != SCREEN_MAX_MODEL_LEN:
        raise RuntimeError(
            "V5.3 12-hour screen requires --max-model-len "
            f"{SCREEN_MAX_MODEL_LEN}"
        )
    if path is None:
        raise RuntimeError(
            "V5.3 12-hour screen requires --runtime-evidence"
        )
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"V5.3 12-hour screen runtime evidence is unreadable: {resolved}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("V5.3 12-hour screen runtime evidence is not an object")
    receipt_core = {
        key: value
        for key, value in payload.items()
        if key not in {"canonical_receipt_sha256", "receipt_pointer"}
    }
    roles = payload.get("roles")
    invocation = payload.get("invocation")
    services = invocation.get("services") if isinstance(invocation, dict) else None
    inventory = payload.get("gpu_inventory")
    host_pointer = payload.get("host_preflight")
    host_path = (
        Path(str(host_pointer.get("path"))).resolve()
        if isinstance(host_pointer, dict)
        else None
    )
    try:
        host_payload = (
            json.loads(host_path.read_text(encoding="utf-8"))
            if host_path is not None
            else None
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "V5.3 12-hour screen host/CUDA preflight is unreadable"
        ) from error
    train_environment = (
        (host_payload.get("environments") or {}).get("train")
        if isinstance(host_payload, dict)
        else None
    )
    expected_roles = {
        "teacher": {
            "model": SCREEN_TEACHER_MODEL,
            "revision": SCREEN_TEACHER_REVISION,
            "visible_gpu_indices": [1, 2],
            "tensor_parallel_size": 2,
        },
        "user_and_judge": {
            "model": SCREEN_USER_MODEL,
            "revision": SCREEN_USER_REVISION,
            "visible_gpu_indices": [0],
            "tensor_parallel_size": 1,
        },
    }
    if (
        payload.get("protocol") != SCREEN_RUNTIME_EVIDENCE_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("phase") != "screen"
        or payload.get("source_commit") != expected_source_commit
        or payload.get("max_model_len") != SCREEN_MAX_MODEL_LEN
        or payload.get("expected_gpu_model") != SCREEN_GPU_MODEL
        or payload.get("official_test_used") is not False
        or payload.get("canonical_receipt_sha256")
        != canonical_sha256(receipt_core)
        or not isinstance(roles, dict)
        or set(roles) != set(expected_roles)
        or not isinstance(invocation, dict)
        or not isinstance(services, dict)
        or set(services) != set(expected_roles)
        or not isinstance(inventory, list)
        or len(inventory) != 4
        or not isinstance(host_pointer, dict)
        or host_pointer.get("path") != str(host_path)
        or not host_path.is_file()
        or host_pointer.get("sha256") != sha256_file(host_path)
        or not isinstance(train_environment, dict)
        or train_environment.get("cuda") != SCREEN_CUDA_VERSION
        or train_environment.get("cuda_available") is not True
    ):
        raise RuntimeError("V5.3 12-hour screen runtime receipt drift")
    inventory_by_index: dict[int, dict[str, Any]] = {}
    for row in inventory:
        index = row.get("index") if isinstance(row, dict) else None
        name = row.get("name") if isinstance(row, dict) else None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in inventory_by_index
            or not isinstance(name, str)
            or name != SCREEN_GPU_MODEL
            or not isinstance(row.get("driver"), str)
            or not row["driver"]
        ):
            raise RuntimeError(
                "V5.3 12-hour screen runtime GPU identity drift"
            )
        inventory_by_index[index] = row
    if set(inventory_by_index) != {0, 1, 2, 3}:
        raise RuntimeError("V5.3 12-hour screen requires four indexed RTX 5090s")
    for role, expected in expected_roles.items():
        role_receipt = roles.get(role)
        service = services.get(role)
        snapshot = service.get("model_snapshot") if isinstance(service, dict) else None
        long_probe = (
            role_receipt.get("long_context_probe")
            if isinstance(role_receipt, dict)
            else None
        )
        prompt_tokens = (
            long_probe.get("prompt_tokens")
            if isinstance(long_probe, dict)
            else None
        )
        completion_tokens = (
            long_probe.get("completion_tokens")
            if isinstance(long_probe, dict)
            else None
        )
        indices = expected["visible_gpu_indices"]
        if (
            not isinstance(role_receipt, dict)
            or role_receipt.get("model") != expected["model"]
            or role_receipt.get("revision") != expected["revision"]
            or role_receipt.get("tensor_parallel_size")
            != expected["tensor_parallel_size"]
            or not isinstance(role_receipt.get("api_base"), str)
            or not isinstance(long_probe, dict)
            or long_probe.get("status") != "PASS"
            or long_probe.get("parallel_tool_calls") is not False
            or long_probe.get("max_tokens") != SCREEN_MAX_TOKENS
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or not 27_000
            <= prompt_tokens
            <= SCREEN_MAX_MODEL_LEN - SCREEN_MAX_TOKENS
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or not 0 < completion_tokens <= SCREEN_MAX_TOKENS
            or long_probe.get("total_tokens")
            != prompt_tokens + completion_tokens
            or not isinstance(service, dict)
            or service.get("visible_gpu_indices") != indices
            or service.get("gpu_identity")
            != [inventory_by_index[index] for index in indices]
            or not isinstance(snapshot, dict)
            or snapshot.get("model") != expected["model"]
            or snapshot.get("requested_revision") != expected["revision"]
            or snapshot.get("resolved_revision") != expected["revision"]
        ):
            raise RuntimeError(
                f"V5.3 12-hour screen runtime {role} evidence drift"
            )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "protocol": payload["protocol"],
        "phase": payload["phase"],
        "canonical_receipt_sha256": payload["canonical_receipt_sha256"],
        "max_model_len": payload["max_model_len"],
        "expected_gpu_model": SCREEN_GPU_MODEL,
        "gpu_inventory": inventory,
        "host_preflight": dict(host_pointer),
        "cuda_version": SCREEN_CUDA_VERSION,
        "teacher_visible_gpu_indices": [1, 2],
        "user_and_judge_visible_gpu_indices": [0],
    }


def validate_screen_execution_contract(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Fail closed on screen-only CLI/model/runtime drift."""

    if manifest.get("protocol") != SCREEN_MANIFEST_PROTOCOL:
        return None
    observed = {
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_api_base": args.teacher_api_base,
        "user_model": args.user_model,
        "user_revision": args.user_revision,
        "user_api_base": args.user_api_base,
        "judge_model": args.judge_model or args.user_model,
        "judge_revision": args.judge_revision,
        "judge_api_base": args.judge_api_base or args.user_api_base,
        "teacher_mode": args.teacher_mode,
        "num_shards": args.num_shards,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
    }
    expected = {
        "teacher_model": SCREEN_TEACHER_MODEL,
        "teacher_revision": SCREEN_TEACHER_REVISION,
        "teacher_api_base": "http://127.0.0.1:8011/v1",
        "user_model": SCREEN_USER_MODEL,
        "user_revision": SCREEN_USER_REVISION,
        "user_api_base": "http://127.0.0.1:8001/v1",
        "judge_model": SCREEN_USER_MODEL,
        "judge_revision": SCREEN_USER_REVISION,
        "judge_api_base": "http://127.0.0.1:8001/v1",
        "teacher_mode": "ground_truth",
        "num_shards": SCREEN_NUM_SHARDS,
        "num_trials": SCREEN_NUM_TRIALS,
        "seed": SCREEN_SEED,
        "temperature": SCREEN_TEMPERATURE,
        "top_p": SCREEN_TOP_P,
        "max_tokens": SCREEN_MAX_TOKENS,
        "max_model_len": SCREEN_MAX_MODEL_LEN,
    }
    if observed != expected:
        raise RuntimeError(
            "V5.3 12-hour screen execution contract drift: "
            f"expected {expected}, observed {observed}"
        )
    return validate_screen_runtime_evidence(
        args.runtime_evidence,
        expected_source_commit=args.expected_source_commit,
        max_model_len=args.max_model_len,
    )


def write_contract(
    args: argparse.Namespace,
    manifest_path: Path,
    split_manifest_path: Path,
    rows: list[dict[str, Any]],
    *,
    dynamic_audit_path: Path,
    dynamic_audit_identity: dict[str, Any],
    gt_compatibility_preflight: dict[str, Any] | None = None,
    sampling_contract: dict[str, Any] | None = None,
    runtime_evidence: dict[str, Any] | None = None,
) -> Path:
    path = args.output_dir / (
        "run_contract.json"
        if args.num_shards == 1
        else f"run_contract.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    )
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite contract: {path}")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "protocol": "v5_stage1_inner_train_generation_run",
        "status": "INCOMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.expected_source_commit,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generation_manifest_protocol": manifest_payload.get("protocol"),
        "gt_compatibility_filter": manifest_payload.get(
            "gt_compatibility_filter"
        ),
        "fault_protocol": manifest_payload.get("fault_protocol"),
        "tau2_commit": manifest_payload.get("tau2_commit"),
        "source_files": manifest_payload.get("source_files"),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "dynamic_audit": str(dynamic_audit_path),
        "dynamic_audit_identity": dynamic_audit_identity,
        "gt_compatibility_preflight": gt_compatibility_preflight,
        "sampling_contract": sampling_contract,
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "task_ids": [f"{row['domain']}:{row['task_id']}" for row in rows],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_trials": args.num_trials,
        "teacher": {
            "model": args.teacher_model,
            "revision": args.teacher_revision,
            "api_base": args.teacher_api_base,
            "mode": args.teacher_mode,
        },
        "user": {
            "model": args.user_model,
            "revision": args.user_revision,
            "api_base": args.user_api_base,
        },
        "judge": {
            "model": args.judge_model or args.user_model,
            "revision": args.judge_revision,
            "api_base": args.judge_api_base or args.user_api_base,
            "protocol": STRICT_NL_JUDGE_PROTOCOL,
            "content_attempts": DEFAULT_CONTENT_ATTEMPTS,
            "schema_failure": "fail_closed",
            "raw_response_audit": True,
        },
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "parallel_tool_calls": False,
            "parallel_tool_call_normalization": "execute_first_then_replan",
            "mixed_tool_call_content_normalization": "drop_text_preserve_sha256",
            "max_steps": args.max_steps,
            "task_timeout_seconds": args.timeout,
            "seed": args.seed,
            "derived_trial_seeds": (
                sampling_contract or {}
            ).get("derived_trial_seeds"),
        },
        "result_sha256": {},
        "strict_judge_audit_evidence": {},
    }
    if manifest_payload.get("protocol") == SCREEN_MANIFEST_PROTOCOL:
        if runtime_evidence is None:
            raise RuntimeError(
                "V5.3 12-hour screen contract lacks runtime evidence"
            )
        source_generation_sha = manifest_payload.get(
            "source_generation_manifest_sha256"
        )
        payload.update(
            {
                "subprotocol": SCREEN_GENERATION_SUBPROTOCOL,
                "screen_protocol": SCREEN_PROTOCOL,
                "formal_data": False,
                "screen_outputs_may_enter_formal": False,
                "screen_outputs_may_enter_formal_v5_3": False,
                "screen_manifest": str(manifest_path),
                "screen_manifest_sha256": sha256_file(manifest_path),
                "source_generation_manifest_sha256": source_generation_sha,
                "generation_manifest_sha256": source_generation_sha,
                "base_seed": SCREEN_SEED,
                "trial_seeds": list(SCREEN_TRIAL_SEEDS),
                "task_universe_complete": False,
                "max_model_len": SCREEN_MAX_MODEL_LEN,
                "runtime_evidence": runtime_evidence,
            }
        )
        payload["teacher"]["tensor_parallel_size"] = 2
        payload["user"]["tensor_parallel_size"] = 1
        payload["judge"]["tensor_parallel_size"] = 1
        payload["decoding"]["max_model_len"] = SCREEN_MAX_MODEL_LEN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def finalize_contract(contract_path: Path, output_files: list[Path]) -> dict[str, str]:
    """Atomically bind a successful generation contract to its raw shards."""

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("status") != "INCOMPLETE":
        raise RuntimeError(f"Generation contract is not INCOMPLETE: {contract_path}")
    result_hashes: dict[str, str] = {}
    contract_dir = contract_path.parent.resolve()
    for path in output_files:
        resolved = path.resolve()
        if resolved.parent != contract_dir or not resolved.is_file():
            raise RuntimeError(
                f"Generation result is absent or outside contract directory: {path}"
            )
        if resolved.name in result_hashes:
            raise RuntimeError(f"Duplicate generation result filename: {resolved.name}")
        result_hashes[resolved.name] = sha256_file(resolved)
    if not result_hashes:
        raise RuntimeError("Cannot finalize generation without result files")
    payload["result_sha256"] = dict(sorted(result_hashes.items()))
    generation_protocol = payload.get("generation_manifest_protocol")
    if generation_protocol in {
        V5_3_GENERATION_PROTOCOL,
        SCREEN_MANIFEST_PROTOCOL,
    }:
        try:
            strict_evidence = (
                judge_audit_contract.validate_strict_judge_evidence(
                    output_files,
                    maximum_content_attempts=DEFAULT_CONTENT_ATTEMPTS,
                )
            )
        except judge_audit_contract.StrictJudgeEvidenceError as error:
            raise RuntimeError(
                "V5.3 generation strict-judge evidence is incomplete"
            ) from error
        if generation_protocol == SCREEN_MANIFEST_PROTOCOL and (
            strict_evidence.get("status") != "PASS"
            or isinstance(strict_evidence.get("expected_calls"), bool)
            or not isinstance(strict_evidence.get("expected_calls"), int)
            or strict_evidence["expected_calls"] <= 0
            or strict_evidence.get("observed_unique_pass_audits")
            != strict_evidence["expected_calls"]
        ):
            raise RuntimeError(
                "V5.3 12-hour screen strict-judge evidence is zero "
                "or incomplete"
            )
        payload["strict_judge_audit_evidence"] = strict_evidence
    payload["status"] = "COMPLETE"
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, contract_path)
    return result_hashes


def main() -> None:
    args = parse_args()
    args.expected_source_commit = args.expected_source_commit.lower()
    validate_provenance(
        expected_source_commit=args.expected_source_commit,
        teacher_revision=args.teacher_revision,
        user_revision=args.user_revision,
        judge_revision=args.judge_revision,
    )
    split_manifest_path = args.split_manifest.resolve()
    manifest_path = args.manifest.resolve()
    split_manifest = load_split_manifest(split_manifest_path)
    manifest = load_inner_train_manifest(
        manifest_path,
        split_manifest=split_manifest,
        split_manifest_sha256=sha256_file(split_manifest_path),
    )
    sampling_contract = validate_sampling_contract(
        manifest_protocol=manifest["protocol"],
        temperature=args.temperature,
        top_p=args.top_p,
        num_trials=args.num_trials,
        seed=args.seed,
    )
    runtime_evidence = validate_screen_execution_contract(args, manifest)
    dynamic_audit_path = args.dynamic_audit.resolve()
    if manifest["protocol"] == SCREEN_MANIFEST_PROTOCOL:
        dynamic_audit_identity = load_screen_source_dynamic_audit(
            dynamic_audit_path,
            manifest=manifest,
            split_manifest_path=split_manifest_path,
        )
    else:
        dynamic_audit_identity = load_complete_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest_path,
            split_manifest_path=split_manifest_path,
            expected_source_split="derived_inner_train",
            expected_task_ids={
                f"{row['domain']}:{row['task_id']}"
                for row in manifest["rows"]
            },
        )
    validate_local_manifest_sources(manifest, args.tau2_root.resolve())
    rows = shard_rows(
        list(manifest["rows"]),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not rows:
        raise RuntimeError("Selected generation shard is empty")
    if (args.teacher_model, args.teacher_api_base) == (
        args.user_model,
        args.user_api_base,
    ):
        raise RuntimeError("Teacher and user simulator must be separately served roles")

    configure_tau2_path(args.tau2_root)
    gt_compatibility = preflight_gt_compatibility(
        list(manifest["rows"]),
        teacher_mode=args.teacher_mode,
        gt_compatibility_filter=manifest["_validated_gt_compatibility_filter"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = write_contract(
        args,
        manifest_path,
        split_manifest_path,
        rows,
        dynamic_audit_path=dynamic_audit_path,
        dynamic_audit_identity=dynamic_audit_identity,
        gt_compatibility_preflight=gt_compatibility,
        sampling_contract=sampling_contract,
        runtime_evidence=runtime_evidence,
    )
    os.environ.setdefault("OPENAI_API_KEY", args.teacher_api_key)
    register_fault_agents()

    teacher_args = endpoint_args(
        args.teacher_api_base,
        args.teacher_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    user_args = endpoint_args(
        args.user_api_base,
        args.user_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    judge_model = args.judge_model or args.user_model
    judge_args = endpoint_args(
        args.judge_api_base or args.user_api_base,
        args.judge_api_key or args.user_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=0.0,
        top_p=1.0,
    )
    patch_local_nl_judge(litellm_openai_model(judge_model), judge_args)

    output_files: list[Path] = []
    conditions = ("clean", "error") if args.condition == "both" else (args.condition,)
    for domain in ("retail", "airline"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if not domain_rows:
            continue
        for condition in conditions:
            output_files.append(
                run_condition(
                    domain=domain,
                    rows=domain_rows,
                    condition=condition,
                    args=args,
                    teacher_args=teacher_args,
                    user_args=user_args,
                )
            )
    # A failed generation leaves an auditable INCOMPLETE contract. Only a
    # fully successful shard collection is atomically bound to result bytes.
    result_hashes = finalize_contract(contract_path, output_files)
    print(
        json.dumps(
            {
                "status": "PASS",
                "result_files": [str(path) for path in output_files],
                "contract_result_sha256": result_hashes,
                "source_split": "derived_inner_train",
                "official_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
