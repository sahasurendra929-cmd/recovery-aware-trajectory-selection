#!/usr/bin/env python3
"""Run paired Stage-1 validation tasks with frozen, role-separated models.

The trained arm is used only as the agent.  A separately served, frozen model
acts as both the user simulator and natural-language judge.  The official test
split is deliberately unsupported by this validation-screen runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit
try:
    import v5_judge_audit_contract as judge_audit_contract
except ModuleNotFoundError:
    from scripts import v5_judge_audit_contract as judge_audit_contract
try:
    import v5_3_low_support_protocol as low_support
except ModuleNotFoundError:
    from scripts import v5_3_low_support_protocol as low_support


PROTOCOL = "v5_stage1_sft_causal_validation"
FAULT_PROTOCOL = "v5_multifamily_readonly_faults_v1"
CHECKPOINT_REGISTRY_PROTOCOL = "v5_stage1_checkpoint_registry"
RUN_CONTRACT_PROTOCOL = "v5_stage1_sft_causal_validation_run"
SPLIT_PROTOCOL = "v5_stage0_tau2_end_to_end"
DOMAINS = ("retail", "airline")
EXPECTED_SPLIT_COUNTS = {
    "retail": {"inner_train": 59, "validation": 15, "sealed_test": 40},
    "airline": {"inner_train": 24, "validation": 6, "sealed_test": 20},
}
ARMS = {
    "base_model",
    "perfect_success",
    "failure_raw",
    "repair_50",
    "repair_100",
}
TRAINED_ARMS = ARMS - {"base_model"}
V5_3_LOW_SUPPORT_PROFILE = low_support.REGISTRY_PROFILE
V5_3_LOW_SUPPORT_DESIGN_VERSION = low_support.DESIGN_VERSION
V5_3_LOW_SUPPORT_DESIGN_PROTOCOL = low_support.DATA_PROTOCOL
V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL = low_support.PROTOCOL
V5_3_LOW_SUPPORT_ARMS = set(low_support.EVAL_ARMS)
V5_3_LOW_SUPPORT_TRAINED_ARMS = (
    V5_3_LOW_SUPPORT_ARMS - {"base_model"}
)
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
V5_3_12H_USER_JUDGE = {
    "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "model_id": "openai/v5-3-12h-user-judge",
    "revision": "539535859b135b0244c91f3e59816150c8056698",
    "roles": ["user_simulator", "strict_nl_judge"],
    "official_test_used": False,
}
V5_3_LOW_SUPPORT_USER_JUDGE = {
    "model": low_support.USER_JUDGE_MODEL,
    "model_id": low_support.USER_JUDGE_MODEL_ID,
    "revision": low_support.USER_JUDGE_REVISION,
    "roles": ["user_simulator", "strict_nl_judge"],
    "official_test_used": False,
}
V5_3_LOW_SUPPORT_CLAIM_BOUNDARY = dict(low_support.CLAIM_BOUNDARY)
ROOT = Path(__file__).resolve().parents[1]
PROFILE_GENERATION_INJECTIONS = {
    "legacy": 83,
    "v5_3": 78,
    "v5_3_12h_screen": 78,
    V5_3_LOW_SUPPORT_PROFILE: 78,
}
PROFILE_MODEL_IDS = {
    "legacy": {
        "base_model": "openai/v5-base",
        "perfect_success": "openai/v5-perfect-success",
        "failure_raw": "openai/v5-failure-raw",
        "repair_50": "openai/v5-repair-50",
        "repair_100": "openai/v5-repair-100",
    },
    "v5_3": {
        "base_model": "openai/v5-3-base",
        "perfect_success": "openai/v5-3-perfect-success",
        "failure_raw": "openai/v5-3-failure-raw",
        "repair_50": "openai/v5-3-repair-50",
        "repair_100": "openai/v5-3-repair-100",
    },
    "v5_3_12h_screen": {
        "base_model": "openai/v5-3-12h-base",
        "perfect_success": "openai/v5-3-12h-perfect-success",
        "failure_raw": "openai/v5-3-12h-failure-raw",
        "repair_50": "openai/v5-3-12h-repair-50",
        "repair_100": "openai/v5-3-12h-repair-100",
    },
    V5_3_LOW_SUPPORT_PROFILE: {
        "base_model": "openai/v5-3-low-support-base",
        "perfect_success": "openai/v5-3-low-support-perfect-success",
        "repair_50": "openai/v5-3-low-support-repair-50",
    },
}
PROFILE_ARMS = {
    "legacy": ARMS,
    "v5_3": ARMS,
    "v5_3_12h_screen": ARMS,
    V5_3_LOW_SUPPORT_PROFILE: V5_3_LOW_SUPPORT_ARMS,
}
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
V5_3_12H_FROZEN_DECODING = {
    **FROZEN_DECODING,
    "seed": 20260731,
}
V5_3_LOW_SUPPORT_FROZEN_DECODING = dict(
    V5_3_12H_FROZEN_DECODING
)
FAULT_INJECTIONS: dict[str, dict[str, Any]] = {}
MAX_MODEL_LEN = 32768
STRICT_NL_JUDGE_MODULE = "v5_strict_nl_judge"
STRICT_NL_JUDGE_ENTRYPOINT = "install_strict_nl_judge"
TOOL_ACTION_INTERFACE = {
    "parallel_tool_calls": False,
    "parallel_tool_call_normalization": "execute_first_then_replan",
    "mixed_tool_call_content_normalization": "drop_text_preserve_sha256",
    "deferred_call_audit_field": "v5_stage1_parallel_calls_serialized",
    "mixed_content_audit_field": "v5_stage1_mixed_content_normalized",
    "audit_hash": "sha256",
    "clean_agent_factory": "v5_stage1_single_tool_agent",
    "error_agent_factory": "v5_stage1_single_tool_fault_agent",
    "post_injection_policy": "execute_injected_call_then_replan",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument(
        "--runtime-service-evidence",
        type=Path,
        help=(
            "Required only for the V5.3 low-support diagnostic. This immutable "
            "receipt binds exact /v1/models sets and all ten endpoint×alias "
            "one-token smokes to every evaluation contract."
        ),
    )
    parser.add_argument(
        "--provenance-profile",
        choices=PROVENANCE_PROFILES,
        help=(
            "Optionally require the checkpoint registry's frozen legacy or "
            "V5.3 provenance profile, including the isolated low-support "
            "diagnostic. The registry profile is otherwise validated and "
            "used directly."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        action="append",
        required=True,
        metavar="ARM=DIR",
        help=(
            "Local final adapter directory for a trained arm; provide exactly "
            "the selected registry profile's trained-arm set."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-api-base", required=True)
    parser.add_argument("--agent-api-key", default="stage1-agent-local")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--user-api-key", default="stage1-user-local")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--condition", choices=("clean", "error", "both"), default="both")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--num-trials", type=int, default=1)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(root: Path | None = None) -> str:
    working_tree = root or Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=working_tree,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_tracked_source(root: Path | None = None) -> None:
    working_tree = root or Path(__file__).resolve().parents[1]
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=working_tree, check=False)
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


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def checkpoint_registry_provenance_profile(
    registry: dict[str, Any],
    *,
    expected_profile: str | None = None,
) -> str:
    """Resolve a registered profile; missing means legacy for old registries."""

    profile = registry.get("provenance_profile", "legacy")
    if profile not in PROVENANCE_PROFILES:
        raise RuntimeError(
            f"Checkpoint registry has unsupported provenance_profile {profile!r}"
        )
    if expected_profile is not None and profile != expected_profile:
        raise RuntimeError(
            "Checkpoint registry provenance profile drift: "
            f"expected {expected_profile!r}, got {profile!r}"
        )
    return profile


def arms_for_profile(provenance_profile: str) -> set[str]:
    try:
        return set(PROFILE_ARMS[provenance_profile])
    except KeyError as error:
        raise RuntimeError(
            f"unsupported provenance profile {provenance_profile!r}"
        ) from error


def trained_arms_for_profile(provenance_profile: str) -> set[str]:
    return arms_for_profile(provenance_profile) - {"base_model"}


def _validate_profile_design_provenance(
    provenance: dict[str, Any],
    *,
    provenance_profile: str,
) -> None:
    design_version = provenance.get("design_version")
    if provenance_profile == "legacy":
        if design_version not in (None, "5.2"):
            raise RuntimeError(
                "Legacy checkpoint provenance requires absent or 5.2 "
                f"design_version, got {design_version!r}"
            )
        return
    if provenance_profile == "v5_3_12h_screen":
        if design_version != V5_3_12H_DESIGN_VERSION:
            raise RuntimeError(
                "V5.3-12h checkpoint provenance requires screen design_version"
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
                "V5.3-12h checkpoint provenance lacks isolated screen identity"
            )
        return
    if provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        if design_version != V5_3_LOW_SUPPORT_DESIGN_VERSION:
            raise RuntimeError(
                "low-support checkpoint provenance requires its frozen "
                "design_version"
            )
        design = provenance.get("design_provenance")
        expected = {
            "design_version": V5_3_LOW_SUPPORT_DESIGN_VERSION,
            "design_protocol": V5_3_LOW_SUPPORT_DESIGN_PROTOCOL,
            "diagnostic_protocol": V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL,
            "validation_tasks": 21,
            "trained_arms": sorted(V5_3_LOW_SUPPORT_TRAINED_ARMS),
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
                "low-support checkpoint provenance lacks its frozen "
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
                "low-support checkpoint source-commit provenance drift"
            )
        return
    if design_version != V5_3_DESIGN_VERSION:
        raise RuntimeError(
            "V5.3 checkpoint provenance requires design_version '5.3'"
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
            "V5.3 checkpoint provenance lacks the frozen 78/21 design identity"
        )


def validate_registry_training_data_provenance(
    registry: dict[str, Any],
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    provenance_profile = checkpoint_registry_provenance_profile(
        registry,
        expected_profile=expected_profile,
    )
    provenance = registry.get("training_data_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Checkpoint registry lacks training_data_provenance")
    _validate_profile_design_provenance(
        provenance,
        provenance_profile=provenance_profile,
    )
    if (
        provenance_profile == V5_3_LOW_SUPPORT_PROFILE
        and provenance["design_provenance"].get(
            "processing_source_commit"
        )
        != registry.get("source_commit")
    ):
        raise RuntimeError(
            "low-support registry source/processing commit drift"
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


def parse_adapter_dir_bindings(
    values: list[str],
    *,
    expected_arms: set[str] | None = None,
) -> dict[str, Path]:
    """Parse one profile's exact local adapter directories fail-closed."""

    expected = set(TRAINED_ARMS if expected_arms is None else expected_arms)
    if not expected or not expected <= TRAINED_ARMS:
        raise RuntimeError(
            f"invalid expected trained adapter set: {sorted(expected)}"
        )
    bindings: dict[str, Path] = {}
    resolved_paths: set[Path] = set()
    for value in values:
        if "=" not in value:
            raise RuntimeError(
                f"--adapter-dir must be ARM=DIR, got {value!r}"
            )
        arm, raw_path = value.split("=", 1)
        if arm not in expected:
            raise RuntimeError(f"unsupported trained adapter arm {arm!r}")
        if arm in bindings:
            raise RuntimeError(f"duplicate --adapter-dir binding for {arm}")
        if not raw_path:
            raise RuntimeError(f"{arm}: empty adapter directory")
        path = Path(raw_path).expanduser().resolve()
        if path in resolved_paths:
            raise RuntimeError(f"duplicate local adapter directory: {path}")
        bindings[arm] = path
        resolved_paths.add(path)
    if set(bindings) != expected:
        missing = sorted(expected - set(bindings))
        extra = sorted(set(bindings) - expected)
        raise RuntimeError(
            "adapter bindings must contain exactly the profile's trained arms; "
            f"missing={missing}, extra={extra}"
        )
    return bindings


def validate_local_adapter_identity(
    registry: dict[str, Any],
    adapter_dirs: dict[str, Path],
) -> dict[str, dict[str, str]]:
    """Bind locally served adapter bytes to the immutable registry.

    The returned object contains only stable content hashes.  Deliberately
    omitting machine-local paths keeps run contracts comparable across workers.
    """

    provenance_profile = checkpoint_registry_provenance_profile(registry)
    expected_arms = trained_arms_for_profile(provenance_profile)
    if set(adapter_dirs) != expected_arms:
        raise RuntimeError(
            "local adapter directories must be exactly the profile's "
            "trained arms"
        )
    resolved_paths = [path.expanduser().resolve() for path in adapter_dirs.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise RuntimeError("local trained arms contain duplicate adapter directories")

    identity: dict[str, dict[str, str]] = {}
    for arm in sorted(expected_arms):
        adapter_dir = adapter_dirs[arm].expanduser().resolve()
        if not adapter_dir.is_dir():
            raise RuntimeError(f"{arm}: local adapter directory missing")
        adapter_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
            raise RuntimeError(
                f"{arm}: adapter_model.safetensors missing or empty"
            )
        if not config_path.is_file() or config_path.stat().st_size <= 0:
            raise RuntimeError(f"{arm}: adapter_config.json missing or empty")
        observed = {
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_config_sha256": sha256_file(config_path),
        }
        expected = registry["entries"][arm]
        for field, digest in observed.items():
            if digest != expected.get(field):
                raise RuntimeError(
                    f"{arm}: local {field} differs from checkpoint registry"
                )
        identity[arm] = observed
    return identity


def require_shared_api_base(
    agent_api_base: str,
    user_api_base: str,
    judge_api_base: str,
) -> str:
    """Require all roles to address the same OpenAI-compatible server."""

    normalized = [
        value.strip().rstrip("/")
        for value in (agent_api_base, user_api_base, judge_api_base)
    ]
    if any(not value for value in normalized):
        raise RuntimeError("agent, user, and judge API bases must be non-empty")
    if len(set(normalized)) != 1:
        raise RuntimeError(
            "agent, user, and judge must use the same API base"
        )
    return normalized[0]


def require_screen_api_bases(
    agent_api_base: str,
    user_api_base: str,
    judge_api_base: str,
) -> tuple[str, str]:
    """Keep the screen agent separate from its fixed 14B user/judge."""

    agent = agent_api_base.strip().rstrip("/")
    user = user_api_base.strip().rstrip("/")
    judge = judge_api_base.strip().rstrip("/")
    if not agent or not user or not judge:
        raise RuntimeError("screen role API bases must be non-empty")
    if user != judge:
        raise RuntimeError("screen user and strict judge must share one endpoint")
    if agent == user:
        raise RuntimeError(
            "screen agent must not share the fixed 14B user/judge endpoint"
        )
    return agent, user


def registry_served_aliases(registry: dict[str, Any]) -> list[str]:
    """Translate LiteLLM ``openai/`` routes to vLLM served model IDs."""

    aliases = []
    profile = checkpoint_registry_provenance_profile(registry)
    expected_arms = arms_for_profile(profile)
    entries = registry.get("entries")
    if not isinstance(entries, dict) or set(entries) != expected_arms:
        raise RuntimeError(
            "checkpoint registry entries do not match the profile arm set"
        )
    for arm in sorted(expected_arms):
        model_id = registry["entries"][arm]["model_id"]
        if not model_id.startswith("openai/"):
            raise RuntimeError(f"{arm}: registry model ID lacks openai/ route")
        aliases.append(model_id.removeprefix("openai/"))
    if len(set(aliases)) != len(expected_arms):
        raise RuntimeError("registry served aliases must be unique")
    return sorted(aliases)


def verify_served_registry_aliases(
    api_base: str,
    api_key: str,
    registry: dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    require_exact: bool = False,
) -> list[str]:
    """Query ``/v1/models`` and require registered aliases."""

    url = f"{api_base.rstrip('/')}/models"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot verify served model aliases at {url}: {error}") from error
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("/v1/models response lacks a data list")
    served_rows = [
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    served = set(served_rows)
    expected = set(registry_served_aliases(registry))
    missing = sorted(expected - served)
    if missing:
        raise RuntimeError(
            f"/v1/models is missing checkpoint-registry aliases: {missing}"
        )
    if require_exact and (
        len(served_rows) != len(served)
        or served != expected
    ):
        raise RuntimeError(
            "/v1/models must contain exactly the checkpoint-registry "
            f"aliases: expected={sorted(expected)}, observed={sorted(served)}"
        )
    return sorted(expected)


def verify_served_model_id(
    api_base: str,
    api_key: str,
    expected_model_id: str,
    *,
    timeout_seconds: float = 15.0,
    require_exact: bool = False,
) -> str:
    """Require one independently served, pinned model alias."""

    url = f"{api_base.rstrip('/')}/models"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot verify served model at {url}: {error}") from error
    rows = payload.get("data") if isinstance(payload, dict) else None
    observed_rows = [
        row.get("id")
        for row in rows or []
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    observed = set(observed_rows)
    expected = expected_model_id.removeprefix("openai/")
    if expected not in observed:
        raise RuntimeError(
            f"/v1/models is missing {expected_model_id!r}: {sorted(observed)}"
        )
    if require_exact and (
        len(observed_rows) != len(observed)
        or observed != {expected}
    ):
        raise RuntimeError(
            "/v1/models must contain exactly the independently served model: "
            f"expected={[expected]}, observed={sorted(observed)}"
        )
    return expected


def _linux_process_identity(pid: int) -> dict[str, Any]:
    if not isinstance(pid, int) or pid <= 1:
        raise RuntimeError("runtime process identity has unsafe PID")
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        cmdline = (proc / "cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as error:
        raise RuntimeError(
            f"runtime service PID {pid} is not live"
        ) from error
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20 or not cmdline or not boot_id:
        raise RuntimeError(f"runtime service PID {pid} identity is malformed")
    try:
        process_group_id = int(fields[2])
        start_time_ticks = int(fields[19])
    except ValueError as error:
        raise RuntimeError(
            f"runtime service PID {pid} identity has non-integer fields"
        ) from error
    return {
        "protocol": low_support.PROCESS_IDENTITY_PROTOCOL,
        "pid": pid,
        "process_group_id": process_group_id,
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def _validate_process_identity(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "protocol",
            "pid",
            "process_group_id",
            "start_time_ticks",
            "boot_id",
            "cmdline_sha256",
        }
        or value.get("protocol") != low_support.PROCESS_IDENTITY_PROTOCOL
        or not isinstance(value.get("pid"), int)
        or value["pid"] <= 1
        or value.get("process_group_id") != value["pid"]
        or not isinstance(value.get("start_time_ticks"), int)
        or value["start_time_ticks"] <= 0
        or not isinstance(value.get("boot_id"), str)
        or not value["boot_id"]
        or not _valid_sha256(value.get("cmdline_sha256"))
        or value.get("cmdline_sha256")
        == hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError("runtime service process identity drift")
    return dict(value)


def _same_live_process(
    expected: dict[str, Any], observed: dict[str, Any]
) -> bool:
    return all(
        expected.get(field) == observed.get(field)
        for field in (
            "protocol",
            "pid",
            "process_group_id",
            "start_time_ticks",
            "boot_id",
            "cmdline_sha256",
        )
    )


def load_low_support_runtime_service_evidence(
    path: Path,
    *,
    checkpoint_registry_path: Path,
    checkpoint_registry: dict[str, Any],
    expected_source_commit: str,
    require_live_services: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate and hash-bind the controller's exact live service receipt."""

    resolved = path.expanduser().resolve()
    expected_path = (
        low_support.artifact_root(ROOT, "results_root")
        / low_support.RUNTIME_SERVICE_EVIDENCE_NAME
    ).resolve()
    if resolved != expected_path:
        raise RuntimeError(
            "low-support runtime service evidence must use its isolated "
            f"canonical path: {expected_path}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid low-support runtime service evidence: {resolved}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("runtime service evidence must be a JSON object")
    canonical = payload.get("canonical_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "canonical_sha256"
    }
    expected_endpoint_aliases = {
        endpoint: list(aliases)
        for endpoint, aliases in (
            low_support.RUNTIME_ENDPOINT_MODEL_ALIASES.items()
        )
    }
    services = payload.get("services")
    if not isinstance(services, list) or len(services) != 4:
        raise RuntimeError("runtime service evidence requires exactly 4 services")
    exact_service_identity = {
        low_support.RUNTIME_ENDPOINTS[0]: {
            "role": "user_and_strict_judge",
            "gpu": 0,
            "model": low_support.USER_JUDGE_MODEL,
            "revision": low_support.USER_JUDGE_REVISION,
            "dtype": "float16",
            "quantization": "awq",
        },
        **{
            endpoint: {
                "role": f"agent_shard_{gpu - 1}",
                "gpu": gpu,
                "model": low_support.STUDENT_MODEL,
                "revision": low_support.STUDENT_REVISION,
                "dtype": "bfloat16",
                "quantization": None,
            }
            for gpu, endpoint in enumerate(
                low_support.RUNTIME_ENDPOINTS[1:], start=1
            )
        },
    }
    observed_service_endpoints: set[str] = set()
    for row in services:
        if not isinstance(row, dict):
            raise RuntimeError("runtime service row must be an object")
        endpoint = row.get("api_base")
        if (
            endpoint not in expected_endpoint_aliases
            or endpoint in observed_service_endpoints
        ):
            raise RuntimeError("runtime service endpoint set drift")
        expected_served_name = expected_endpoint_aliases[endpoint][0]
        expected_port = endpoint.rsplit(":", 1)[1].removesuffix("/v1")
        command = row.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) for value in command)
            or not _valid_sha256(row.get("command_sha256"))
            or row.get("command_sha256")
            != low_support.canonical_sha256(command)
        ):
            raise RuntimeError("runtime service command evidence drift")
        required_command_values = {
            "--revision": row.get("revision"),
            "--tokenizer-revision": row.get("tokenizer_revision"),
            "--host": "127.0.0.1",
            "--dtype": row.get("dtype"),
            "--max-model-len": str(row.get("max_model_len")),
            "--max-num-seqs": str(row.get("max_num_seqs")),
            "--gpu-memory-utilization": "0.90",
            "--tool-call-parser": "hermes",
            "--generation-config": "vllm",
            "--served-model-name": expected_served_name,
            "--port": expected_port,
        }
        for flag, expected_value in required_command_values.items():
            if command.count(flag) != 1:
                raise RuntimeError(
                    f"runtime service command {flag} count drift"
                )
            index = command.index(flag)
            if (
                index + 1 >= len(command)
                or command[index + 1] != expected_value
            ):
                raise RuntimeError(
                    f"runtime service command {flag} value drift"
                )
        if command.count("--enable-auto-tool-choice") != 1:
            raise RuntimeError(
                "runtime service auto-tool-choice command drift"
            )
        if (
            len(command) < 3
            or command[1:3] != ["serve", row.get("model")]
        ):
            raise RuntimeError("runtime service serve/model/port command drift")
        models_response = row.get("models_response")
        models_rows = (
            models_response.get("data")
            if isinstance(models_response, dict)
            else None
        )
        if not isinstance(models_rows, list):
            raise RuntimeError("runtime raw /models response is absent")
        raw_model_ids = [
            item.get("id")
            for item in models_rows
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
        ]
        if (
            len(raw_model_ids) != len(models_rows)
            or len(raw_model_ids) != len(set(raw_model_ids))
            or sorted(raw_model_ids) != expected_endpoint_aliases[endpoint]
            or row.get("models_response_sha256")
            != low_support.canonical_sha256(models_response)
        ):
            raise RuntimeError(
                "runtime raw /models response/hash binding drift"
            )
        process_identity = _validate_process_identity(
            row.get("process_identity")
        )
        expected_pid_receipt_path = (
            resolved.parent
            / "pids"
            / f"eval-server-gpu{row.get('gpu')}.json"
        ).resolve()
        try:
            pid_receipt_path = Path(
                row.get("pid_receipt_path")
            ).expanduser().resolve()
        except TypeError as error:
            raise RuntimeError(
                "runtime service PID receipt path drift"
            ) from error
        if (
            pid_receipt_path != expected_pid_receipt_path
            or not pid_receipt_path.is_file()
            or not _valid_sha256(row.get("pid_receipt_sha256"))
            or row.get("pid_receipt_sha256")
            != sha256_file(pid_receipt_path)
        ):
            raise RuntimeError("runtime service PID receipt binding drift")
        try:
            pid_receipt = json.loads(
                pid_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "runtime service PID receipt is invalid"
            ) from error
        if not isinstance(pid_receipt, dict):
            raise RuntimeError("runtime service PID receipt is not an object")
        pid_receipt_canonical = pid_receipt.get("canonical_sha256")
        pid_receipt_core = {
            key: value
            for key, value in pid_receipt.items()
            if key != "canonical_sha256"
        }
        if (
            pid_receipt.get("protocol")
            != low_support.SERVICE_PID_RECEIPT_PROTOCOL
            or pid_receipt.get("service_role") != row.get("role")
            or pid_receipt.get("gpu") != row.get("gpu")
            or pid_receipt.get("api_base") != endpoint
            or pid_receipt.get("command_sha256")
            != row.get("command_sha256")
            or pid_receipt.get("process_identity") != process_identity
            or not _valid_sha256(pid_receipt_canonical)
            or pid_receipt_canonical
            != low_support.canonical_sha256(pid_receipt_core)
        ):
            raise RuntimeError("runtime service PID receipt content drift")
        if (
            require_live_services
            and not _same_live_process(
                process_identity,
                _linux_process_identity(process_identity["pid"]),
            )
        ):
            raise RuntimeError(
                "runtime evidence is not bound to the same live service"
            )
        adapter_aliases = {
            low_support.MODEL_IDS[arm].removeprefix("openai/")
            for arm in low_support.TRAINED_ARMS
        }
        lora_module_values = {
            value
            for value in command
            if "=" in value and value.split("=", 1)[0] in adapter_aliases
        }
        expected_lora_module_values = {
            (
                f"{low_support.MODEL_IDS[arm].removeprefix('openai/')}="
                f"{resolved.parent / 'training' / arm / 'formal' / 'checkpoint_final'}"
            )
            for arm in low_support.TRAINED_ARMS
        }
        expected_lora_flag_values = {
            "--max-lora-rank": "16",
            "--max-loras": "1",
            "--max-cpu-loras": "4",
        }
        lora_flag_drift = False
        for flag, expected_value in expected_lora_flag_values.items():
            if endpoint == low_support.RUNTIME_ENDPOINTS[0]:
                lora_flag_drift = lora_flag_drift or flag in command
                continue
            if command.count(flag) != 1:
                lora_flag_drift = True
                continue
            index = command.index(flag)
            lora_flag_drift = lora_flag_drift or (
                index + 1 >= len(command)
                or command[index + 1] != expected_value
            )
        lora_command_drift = (
            (
                "--enable-lora" in command
                or "--lora-modules" in command
                or lora_module_values
            )
            if endpoint == low_support.RUNTIME_ENDPOINTS[0]
            else (
                "--enable-lora" not in command
                or command.count("--lora-modules") != 1
                or lora_module_values != expected_lora_module_values
            )
        ) or lora_flag_drift
        if (
            endpoint not in expected_endpoint_aliases
            or endpoint in observed_service_endpoints
            or row.get("expected_models")
            != expected_endpoint_aliases[endpoint]
            or row.get("observed_models")
            != expected_endpoint_aliases[endpoint]
            or row.get("models_exact") is not True
            or row.get("tokenizer_revision") != row.get("revision")
            or row.get("max_model_len") != MAX_MODEL_LEN
            or row.get("max_num_seqs") != 1
            or row.get("gpu_memory_utilization") != 0.90
            or lora_command_drift
            or any(
                row.get(key) != value
                for key, value in exact_service_identity[endpoint].items()
            )
            or (
                row.get("quantization") == "awq"
                and (
                    command.count("--quantization") != 1
                    or command.index("--quantization") + 1 >= len(command)
                    or command[command.index("--quantization") + 1] != "awq"
                )
            )
            or (
                row.get("quantization") is None
                and "--quantization" in command
            )
        ):
            raise RuntimeError(
                "runtime service exact endpoint/model-set evidence drift"
            )
        observed_service_endpoints.add(endpoint)
    expected_pairs = {
        (endpoint, alias)
        for endpoint, aliases in expected_endpoint_aliases.items()
        for alias in aliases
    }
    smokes = payload.get("alias_smokes")
    if (
        not isinstance(smokes, list)
        or len(smokes) != low_support.RUNTIME_ALIAS_SMOKE_COUNT
    ):
        raise RuntimeError(
            "runtime service evidence must contain exactly ten alias smokes"
        )
    observed_pairs: set[tuple[str, str]] = set()
    for row in smokes:
        if not isinstance(row, dict):
            raise RuntimeError("runtime alias smoke must be an object")
        pair = (row.get("api_base"), row.get("model_alias"))
        response_sha = row.get("response_sha256")
        response = row.get("response")
        choices = (
            response.get("choices")
            if isinstance(response, dict)
            else None
        )
        choice = (
            choices[0]
            if isinstance(choices, list) and len(choices) == 1
            else None
        )
        if (
            pair not in expected_pairs
            or pair in observed_pairs
            or row.get("status") != "PASS"
            or row.get("max_tokens") != 1
            or row.get("response_model") != pair[1]
            or row.get("finish_reason") not in {"stop", "length"}
            or not _valid_sha256(response_sha)
            or not isinstance(response, dict)
            or response_sha != low_support.canonical_sha256(response)
            or response.get("model") != pair[1]
            or not isinstance(choice, dict)
            or choice.get("finish_reason") != row.get("finish_reason")
        ):
            raise RuntimeError("runtime endpoint×alias smoke evidence drift")
        observed_pairs.add(pair)
    registry_path = checkpoint_registry_path.resolve()
    source_admission_path = (
        low_support.artifact_root(ROOT, "runtime_root")
        / "source_admission_receipt.json"
    ).resolve()
    if (
        payload.get("protocol")
        != low_support.RUNTIME_SERVICE_EVIDENCE_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("processing_source_commit")
        != expected_source_commit
        or payload.get("source_generation_commit")
        != low_support.SOURCE_GENERATION_COMMIT
        or payload.get("checkpoint_registry_path") != str(registry_path)
        or payload.get("checkpoint_registry_sha256")
        != sha256_file(registry_path)
        or checkpoint_registry.get("source_commit")
        != expected_source_commit
        or payload.get("registry_model_ids")
        != {
            arm: checkpoint_registry["entries"][arm]["model_id"]
            for arm in sorted(low_support.EVAL_ARMS)
        }
        or observed_service_endpoints != set(expected_endpoint_aliases)
        or observed_pairs != expected_pairs
        or payload.get("alias_smoke_count")
        != low_support.RUNTIME_ALIAS_SMOKE_COUNT
        or payload.get("all_model_sets_exact") is not True
        or not source_admission_path.is_file()
        or payload.get("source_admission_receipt_sha256")
        != sha256_file(source_admission_path)
        or payload.get("official_test_used") is not False
        or payload.get("official_test_sealed") is not True
        or not _valid_sha256(canonical)
        or canonical != low_support.canonical_sha256(core)
    ):
        raise RuntimeError("runtime service evidence provenance/identity drift")
    return payload, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "canonical_sha256": canonical,
    }


def load_checkpoint_registry(
    path: Path,
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    provenance_profile = checkpoint_registry_provenance_profile(
        payload,
        expected_profile=expected_profile,
    )
    expected_arms = arms_for_profile(provenance_profile)
    entries = payload.get("entries")
    if not isinstance(entries, dict) or set(entries) != expected_arms:
        raise RuntimeError(
            "Checkpoint registry entries for "
            f"{provenance_profile} must be exactly "
            f"{tuple(sorted(expected_arms))}"
        )

    expected_model_ids = PROFILE_MODEL_IDS[provenance_profile]
    model_ids: list[str] = []
    for arm in sorted(expected_arms):
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
        adapter_sha = entry.get("adapter_sha256")
        adapter_config_sha = entry.get("adapter_config_sha256")
        run_manifest_sha = entry.get("training_run_manifest_sha256")
        if arm == "base_model":
            if adapter_sha is not None:
                raise RuntimeError("base_model adapter_sha256 must be null")
            if adapter_config_sha is not None:
                raise RuntimeError(
                    "base_model adapter_config_sha256 must be null"
                )
            if run_manifest_sha not in (None,):
                raise RuntimeError(
                    "base_model training_run_manifest_sha256 must be null"
                )
        else:
            if not _valid_sha256(adapter_sha):
                raise RuntimeError(f"{arm} has invalid adapter_sha256")
            if not _valid_sha256(adapter_config_sha):
                raise RuntimeError(
                    f"{arm} has invalid adapter_config_sha256"
                )
            if not _valid_sha256(run_manifest_sha):
                raise RuntimeError(
                    f"{arm} has invalid training_run_manifest_sha256"
                )
    if len(set(model_ids)) != len(model_ids):
        raise RuntimeError(
            "All checkpoint registry model_id aliases must be unique"
        )
    for arm in sorted(expected_arms):
        model_id = entries[arm]["model_id"]
        if model_id != expected_model_ids[arm]:
            raise RuntimeError(
                f"Checkpoint registry entry {arm} model_id drift for "
                f"{provenance_profile}: expected {expected_model_ids[arm]!r}, "
                f"got {model_id!r}"
            )
    validate_registry_training_data_provenance(
        payload,
        expected_profile=provenance_profile,
    )
    if (
        provenance_profile == "v5_3_12h_screen"
        and payload.get("screen_evaluator") != V5_3_12H_USER_JUDGE
    ):
        raise RuntimeError(
            "V5.3-12h registry lacks the pinned 14B evaluator identity"
        )
    if provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        if (
            payload.get("diagnostic_evaluator")
            != V5_3_LOW_SUPPORT_USER_JUDGE
            or payload.get("diagnostic_claim_boundary")
            != V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
        ):
            raise RuntimeError(
                "low-support registry lacks its pinned diagnostic identity"
            )
    return payload


def validate_checkpoint_identity(
    registry: dict[str, Any],
    *,
    arm: str,
    agent_model: str,
    local_source_commit: str,
    user_model: str,
    judge_model: str,
) -> dict[str, Any]:
    if local_source_commit != registry["source_commit"]:
        raise RuntimeError(
            f"Local source commit {local_source_commit} differs from registry "
            f"{registry['source_commit']}"
        )
    entry = registry["entries"].get(arm)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Checkpoint registry lacks arm {arm}")
    if entry.get("model_id") != agent_model:
        raise RuntimeError(
            f"--agent-model {agent_model!r} does not match registry entry "
            f"{entry.get('model_id')!r} for {arm}"
        )
    profile = checkpoint_registry_provenance_profile(registry)
    if profile == "v5_3_12h_screen":
        if (
            registry.get("screen_evaluator") != V5_3_12H_USER_JUDGE
            or user_model != V5_3_12H_USER_JUDGE["model_id"]
            or judge_model != V5_3_12H_USER_JUDGE["model_id"]
        ):
            raise RuntimeError(
                "V5.3-12h user/judge must equal the pinned 14B evaluator"
            )
    elif profile == V5_3_LOW_SUPPORT_PROFILE:
        if (
            registry.get("diagnostic_evaluator")
            != V5_3_LOW_SUPPORT_USER_JUDGE
            or registry.get("diagnostic_claim_boundary")
            != V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
            or user_model != V5_3_LOW_SUPPORT_USER_JUDGE["model_id"]
            or judge_model != V5_3_LOW_SUPPORT_USER_JUDGE["model_id"]
        ):
            raise RuntimeError(
                "low-support user/judge must equal the pinned 14B "
                "diagnostic evaluator"
            )
    else:
        frozen_base_alias = registry["entries"]["base_model"]["model_id"]
        if user_model != frozen_base_alias or judge_model != frozen_base_alias:
            raise RuntimeError(
                "User and judge model IDs must both equal the registry "
                "base_model alias"
            )
    return dict(entry)


def validate_evaluation_protocol(args: argparse.Namespace) -> None:
    observed = {
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "task_timeout_seconds": float(args.timeout),
        "seed": args.seed,
        "num_trials": args.num_trials,
    }
    profile = getattr(args, "provenance_profile", None)
    if profile == "v5_3_12h_screen":
        expected = V5_3_12H_FROZEN_DECODING
    elif profile == V5_3_LOW_SUPPORT_PROFILE:
        expected = V5_3_LOW_SUPPORT_FROZEN_DECODING
        if args.num_shards != low_support.EVALUATION_SHARDS:
            raise RuntimeError(
                "low-support evaluation requires exactly "
                f"{low_support.EVALUATION_SHARDS} shards"
            )
    else:
        expected = FROZEN_DECODING
    if observed != expected:
        raise RuntimeError(
            f"Evaluation decoding must equal frozen protocol {expected}, "
            f"got {observed}"
        )
    if args.condition != "both":
        raise RuntimeError("Official Stage-1 validation requires both conditions")


def configure_tau2_path(tau2_root: Path) -> None:
    source = tau2_root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"tau2 source directory not found: {source}")
    sys.path.insert(0, str(source))


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


def load_manifest(
    path: Path,
    *,
    split_manifest: dict[str, Any],
    split_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Expected manifest protocol {PROTOCOL!r}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Validation manifest must contain non-empty rows")
    if payload.get("paired_task_count") != len(rows):
        raise RuntimeError("Validation manifest paired_task_count drift")
    if payload.get("source_split") != "derived_validation":
        raise RuntimeError("Stage-1 screen may use only derived validation tasks")
    if payload.get("official_test_used") is not False:
        raise RuntimeError("Validation manifest must explicitly seal official test")
    if payload.get("official_test_sealed") is not True:
        raise RuntimeError("Validation manifest lacks the official-test seal")
    if payload.get("scientific_claim_allowed") is not False:
        raise RuntimeError("Validation screen must prohibit final scientific claims")
    if payload.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError("Validation manifest split_manifest_sha256 drift")
    fault_protocol = payload.get("fault_protocol")
    if (
        not isinstance(fault_protocol, dict)
        or fault_protocol.get("protocol") != FAULT_PROTOCOL
        or fault_protocol.get("claim_scope")
        != "multi_fault_family_post_fault_robustness_screen"
    ):
        raise RuntimeError("Validation manifest lacks frozen multi-fault protocol")
    raw_catalog = fault_protocol.get("families")
    if not isinstance(raw_catalog, dict) or set(raw_catalog) != {
        "retail",
        "airline",
    }:
        raise RuntimeError("Validation fault catalog domain drift")
    catalog: dict[str, dict[str, tuple[str, str]]] = {}
    for domain in ("retail", "airline"):
        family_rows = raw_catalog.get(domain)
        if not isinstance(family_rows, list) or len(family_rows) < 2:
            raise RuntimeError(f"{domain}: insufficient validation fault catalog")
        catalog[domain] = {}
        for family_row in family_rows:
            if not isinstance(family_row, dict):
                raise RuntimeError(f"{domain}: malformed validation fault catalog")
            family = family_row.get("fault_family")
            tool = family_row.get("tool_name")
            invalid_key = family_row.get("invalid_argument_key")
            if not all(
                isinstance(value, str) and value
                for value in (family, tool, invalid_key)
            ):
                raise RuntimeError(f"{domain}: incomplete validation fault catalog")
            if family in catalog[domain]:
                raise RuntimeError(f"{domain}: duplicate validation fault family")
            catalog[domain][family] = (tool, invalid_key)

    pair_ids: set[str] = set()
    task_keys: set[tuple[str, str]] = set()
    tool_call_ids: set[str] = set()
    invalid_parameters: set[str] = set()
    families: dict[str, set[str]] = {"retail": set(), "airline": set()}
    tools: dict[str, set[str]] = {"retail": set(), "airline": set()}
    for index, row in enumerate(rows):
        pair_id = row.get("pair_id")
        domain = row.get("domain")
        task_id = str(row.get("task_id"))
        if not isinstance(pair_id, str) or not pair_id:
            raise RuntimeError(f"Manifest row {index} lacks pair_id")
        if domain not in {"retail", "airline"}:
            raise RuntimeError(f"Manifest row {index} has invalid domain")
        if row.get("source_split") != "derived_validation":
            raise RuntimeError(f"Manifest row {index} split drift")
        if pair_id in pair_ids or (domain, task_id) in task_keys:
            raise RuntimeError("Validation manifest contains duplicate tasks")
        pair_ids.add(pair_id)
        task_keys.add((domain, task_id))

        clean = row.get("clean_condition") or {}
        error = row.get("error_condition") or {}
        if clean.get("inject_error") is not False:
            raise RuntimeError(f"Manifest row {index} has invalid clean condition")
        required_error = {
            "inject_error": True,
            "expected_tool_error": True,
            "expected_state_mutation": False,
        }
        if any(error.get(key) != value for key, value in required_error.items()):
            raise RuntimeError(f"Manifest row {index} has unsafe error condition")
        if not error.get("tool_name") or not error.get("tool_call_id"):
            raise RuntimeError(f"Manifest row {index} lacks error injection identity")
        if not isinstance(error.get("arguments"), dict):
            raise RuntimeError(f"Manifest row {index} lacks error injection arguments")
        family = error.get("fault_family")
        specification = catalog[domain].get(family)
        if specification is None:
            raise RuntimeError(f"Manifest row {index} has unknown fault family")
        tool_name, invalid_key = specification
        if (
            error.get("tool_name") != tool_name
            or error.get("tool_type") != "READ"
            or error.get("invalid_argument_key") != invalid_key
            or not isinstance(error["arguments"].get(invalid_key), str)
        ):
            raise RuntimeError(f"Manifest row {index} fault schema drift")
        call_id = error["tool_call_id"]
        if call_id in tool_call_ids:
            raise RuntimeError("Validation fault call IDs are not task-unique")
        tool_call_ids.add(call_id)
        invalid_identity = canonical_json(
            {invalid_key: error["arguments"][invalid_key]}
        )
        if invalid_identity in invalid_parameters:
            raise RuntimeError("Validation invalid parameters are not task-unique")
        invalid_parameters.add(invalid_identity)
        if type(error.get("on_reference_path")) is not bool:
            raise RuntimeError(f"Manifest row {index} lacks relevance evidence")
        if error.get("fault_relevance") not in {
            "reference_path_or_operation_aligned",
            "domain_plausible_fallback",
        }:
            raise RuntimeError(f"Manifest row {index} relevance drift")
        families[domain].add(family)
        tools[domain].add(tool_name)

    expected_counts = payload.get("domain_counts")
    observed_counts = {
        domain: sum(row["domain"] == domain for row in rows)
        for domain in ("retail", "airline")
    }
    if expected_counts != observed_counts:
        raise RuntimeError(
            f"Validation domain count drift: expected {expected_counts}, "
            f"observed {observed_counts}"
        )
    expected_validation = split_manifest["_validated_identity_sets"]["validation"]
    expected_sealed = split_manifest["_validated_identity_sets"]["sealed_test"]
    observed = {(str(row["domain"]), str(row["task_id"])) for row in rows}
    leaked = observed & expected_sealed
    if leaked:
        raise RuntimeError(
            f"Official-test task IDs leaked into validation: {sorted(leaked)[:5]}"
        )
    if observed != expected_validation:
        raise RuntimeError(
            "Validation manifest must cover exactly the 21 frozen validation IDs; "
            f"missing={sorted(expected_validation - observed)[:5]}, "
            f"extra={sorted(observed - expected_validation)[:5]}"
        )
    for domain in ("retail", "airline"):
        if len(families[domain]) < 2 or len(tools[domain]) < 2:
            raise RuntimeError(f"{domain}: validation lacks multi-fault diversity")
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
        raise RuntimeError("Validation manifest lacks source file hashes")
    for domain in ("retail", "airline"):
        declared = sources.get(domain)
        if not isinstance(declared, dict):
            raise RuntimeError(f"Validation manifest lacks {domain} source hashes")
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


def _tool_call_payload(call: Any) -> dict[str, Any]:
    if hasattr(call, "model_dump"):
        payload = call.model_dump()
    elif isinstance(call, dict):
        payload = dict(call)
    else:
        raise RuntimeError(
            f"Cannot audit tool call of type {type(call).__name__}"
        )
    if not isinstance(payload, dict):
        raise RuntimeError("Tool-call serialization must produce an object")
    return payload


def _sha256_canonical_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def normalize_tool_only_message(message: Any) -> Any:
    """Match the generation interface before any agent action is executed.

    The OpenAI ``parallel_tool_calls=false`` request field is a hint that some
    local model/tool parsers do not honor.  Consequently the evaluator also
    enforces the contract on the response: execute the first ordered call,
    preserve hashes of all deferred calls, discard pre-tool prose with an
    audit hash, then let the model replan after the first tool result.
    """

    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return message
    raw_data = dict(getattr(message, "raw_data", None) or {})
    if len(tool_calls) > 1:
        deferred = tool_calls[1:]
        raw_data["v5_stage1_parallel_calls_serialized"] = {
            "original_count": len(tool_calls),
            "deferred_call_sha256": [
                _sha256_canonical_payload(_tool_call_payload(call))
                for call in deferred
            ],
        }
        message.tool_calls = tool_calls[:1]
    content = getattr(message, "content", None)
    if content not in (None, ""):
        if not isinstance(content, str):
            raise RuntimeError("Mixed tool-call content must be text")
        encoded = content.encode("utf-8")
        raw_data["v5_stage1_mixed_content_normalized"] = {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "utf8_bytes": len(encoded),
        }
        message.content = None
    message.raw_data = raw_data
    return message


def _is_context_window_error(exc: BaseException) -> bool:
    """Recognize provider context errors without pinning a LiteLLM class."""

    names = {cls.__name__ for cls in type(exc).__mro__}
    text = str(exc)
    return (
        "ContextWindowExceededError" in names
        or "ContextWindowExceededError" in text
        or (
            "maximum context length" in text
            and "input tokens" in text
        )
    )


def _message_audit_payload(message: Any) -> Any:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if hasattr(message, "dict"):
        return message.dict()
    return repr(message)


def compact_oldest_complete_turn(messages: list[Any]) -> dict[str, Any]:
    """Drop one oldest complete turn while retaining a user-led suffix."""

    boundary = next(
        (
            index
            for index, candidate in enumerate(messages[1:], start=1)
            if getattr(candidate, "role", None) == "user"
        ),
        None,
    )
    if boundary is None:
        raise RuntimeError(
            "Cannot compact context without a second user-turn boundary"
        )
    removed = messages[:boundary]
    del messages[:boundary]
    return {
        "strategy": "drop_oldest_complete_turn",
        "removed_messages": len(removed),
        "removed_sha256": [
            _sha256_canonical_payload(_message_audit_payload(item))
            for item in removed
        ],
    }


def register_fault_agent() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    class Stage1SingleToolAgent(LLMAgent):
        def _generate_next_message(self, message, state):
            history_length = len(state.messages)
            try:
                generated = super()._generate_next_message(message, state)
            except Exception as exc:
                if not _is_context_window_error(exc):
                    raise
                # LLMAgent appends the current input before calling the
                # provider. Undo only that mutation, compact one old complete
                # turn, then let LLMAgent append the current input once.
                del state.messages[history_length:]
                audit = compact_oldest_complete_turn(state.messages)
                generated = super()._generate_next_message(message, state)
                raw_data = dict(getattr(generated, "raw_data", None) or {})
                raw_data["v5_context_compaction"] = audit
                generated.raw_data = raw_data
            return normalize_tool_only_message(generated)

    class Stage1FaultAgent(Stage1SingleToolAgent):
        def __init__(self, tools, domain_policy, task, llm, llm_args):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=llm_args,
            )
            injection = FAULT_INJECTIONS.get(str(task.id))
            if injection is None:
                raise RuntimeError(f"Task {task.id} lacks Stage-1 injection data")
            self._stage1_injection = injection
            self._stage1_injected = False

        def _generate_next_message(self, message, state):
            if not self._stage1_injected and isinstance(message, UserMessage):
                state.messages.append(message)
                injection = self._stage1_injection
                self._stage1_injected = True
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
                    raw_data={
                        "v5_stage1_injected_fault": True,
                        "expected_tool_error": True,
                    },
                    generation_time_seconds=0.0,
                )
            return super()._generate_next_message(message, state)

    def create_single_tool_agent(tools, domain_policy, **kwargs):
        return Stage1SingleToolAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    def create_fault_agent(tools, domain_policy, **kwargs):
        return Stage1FaultAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory("v5_stage1_single_tool_agent") is None:
        registry.register_agent_factory(
            create_single_tool_agent,
            "v5_stage1_single_tool_agent",
        )
    if registry.get_agent_factory("v5_stage1_single_tool_fault_agent") is None:
        registry.register_agent_factory(
            create_fault_agent,
            "v5_stage1_single_tool_fault_agent",
        )


def patch_local_nl_judge(model: str, llm_args: dict[str, Any]) -> dict[str, str]:
    """Install the strict judge lazily, after the pinned tau2 path is active.

    ``v5_strict_nl_judge`` is intentionally imported here rather than at module
    import time so this evaluator and its unit tests can land before the strict
    judge module.  A formal evaluation still fails closed if the entry point is
    absent; silently falling back to tau2's bare ``json.loads`` path would make
    Markdown-fenced, otherwise valid judge responses infrastructure failures.
    """

    try:
        strict_module = importlib.import_module(STRICT_NL_JUDGE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != STRICT_NL_JUDGE_MODULE:
            raise
        raise RuntimeError(
            "Strict NL judge is not installed yet; evaluation is deferred "
            f"until {STRICT_NL_JUDGE_MODULE}.{STRICT_NL_JUDGE_ENTRYPOINT} "
            "is available"
        ) from exc
    installer = getattr(strict_module, STRICT_NL_JUDGE_ENTRYPOINT, None)
    if not callable(installer):
        raise RuntimeError(
            "Strict NL judge module lacks callable "
            f"{STRICT_NL_JUDGE_ENTRYPOINT}"
        )
    installer(model=model, llm_args=dict(llm_args))
    return {
        "module": STRICT_NL_JUDGE_MODULE,
        "entrypoint": STRICT_NL_JUDGE_ENTRYPOINT,
        "mode": "strict_json_schema_fail_closed",
    }


def select_tasks(domain: str, rows: list[dict[str, Any]]):
    from tau2.registry import registry

    all_tasks = {
        str(task.id): task for task in registry.get_tasks_loader(domain)(None)
    }
    selected = []
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in all_tasks:
            raise RuntimeError(f"Missing {domain} task {task_id}")
        selected.append(all_tasks[task_id])
    return selected


def endpoint_args(
    api_base: str, api_key: str, *, max_tokens: int, seed: int
) -> dict[str, Any]:
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
        or max_tokens > MAX_MODEL_LEN
    ):
        raise RuntimeError(
            f"max_tokens must be in [1, {MAX_MODEL_LEN}], got {max_tokens!r}"
        )
    return {
        "api_base": api_base,
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
        "parallel_tool_calls": False,
    }


def validate_request_token_budget(
    prompt_tokens: Any,
    *,
    max_tokens: int,
    max_model_len: int = MAX_MODEL_LEN,
) -> int:
    """Fail closed if an observed request could exceed the served context."""

    for value, label in (
        (prompt_tokens, "prompt_tokens"),
        (max_tokens, "max_tokens"),
        (max_model_len, "max_model_len"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise RuntimeError(f"{label} must be a non-negative integer")
    if max_tokens <= 0 or max_model_len <= 0:
        raise RuntimeError("max_tokens and max_model_len must be positive")
    total = prompt_tokens + max_tokens
    if total > max_model_len:
        raise RuntimeError(
            "Observed request token budget exceeds the frozen context window: "
            f"{prompt_tokens} prompt + {max_tokens} completion > "
            f"{max_model_len}"
        )
    return total


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
    agent_args: dict[str, Any],
    user_args: dict[str, Any],
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    if condition == "error":
        FAULT_INJECTIONS.clear()
        FAULT_INJECTIONS.update(
            {
                str(row["task_id"]): dict(row["error_condition"])
                for row in rows
            }
        )
    agent_name = (
        "v5_stage1_single_tool_agent"
        if condition == "clean"
        else "v5_stage1_single_tool_fault_agent"
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
        llm_agent=args.agent_model,
        llm_args_agent=agent_args,
        llm_user=args.user_model,
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


_MUTABLE_CONTRACT_FIELDS = {
    "status",
    "completed_at",
    "result_sha256",
    "completion_audit",
    "strict_judge_audit_evidence",
    "contract_core_sha256",
}


def contract_core_sha256(payload: dict[str, Any]) -> str:
    """Hash every immutable run-contract field as one canonical object."""

    core = {
        key: value
        for key, value in payload.items()
        if key not in _MUTABLE_CONTRACT_FIELDS
    }
    return hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()


def validate_contract_core(payload: dict[str, Any]) -> None:
    observed = payload.get("contract_core_sha256")
    expected = contract_core_sha256(payload)
    if not _valid_sha256(observed) or observed != expected:
        raise RuntimeError(
            "Run contract immutable core hash drift; refusing completion"
        )
    if payload.get("tool_action_interface") != TOOL_ACTION_INTERFACE:
        raise RuntimeError(
            "Run contract tool-action interface drift; refusing completion"
        )
    preflight = payload.get("runtime_preflight")
    provenance_profile = payload.get(
        "checkpoint_registry_provenance_profile"
    )
    if provenance_profile == "v5_3_12h_screen":
        expected_decoding = V5_3_12H_FROZEN_DECODING
    elif provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        expected_decoding = V5_3_LOW_SUPPORT_FROZEN_DECODING
        runtime_service_evidence = payload.get(
            "runtime_service_evidence"
        )
        if (
            payload.get("diagnostic_protocol")
            != V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL
            or payload.get("diagnostic_claim_boundary")
            != V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
            or not isinstance(runtime_service_evidence, dict)
            or set(runtime_service_evidence)
            != {"path", "sha256", "canonical_sha256"}
            or not isinstance(runtime_service_evidence.get("path"), str)
            or not _valid_sha256(runtime_service_evidence.get("sha256"))
            or not _valid_sha256(
                runtime_service_evidence.get("canonical_sha256")
            )
        ):
            raise RuntimeError(
                "Run contract low-support diagnostic identity drift"
            )
    else:
        expected_decoding = FROZEN_DECODING
    expected_preflight = {
        "served_context_window_tokens": MAX_MODEL_LEN,
        "request_max_tokens": expected_decoding["max_tokens"],
        "maximum_nonoverflow_prompt_tokens": (
            MAX_MODEL_LEN - expected_decoding["max_tokens"]
        ),
        "max_steps": expected_decoding["max_steps"],
        "request_token_overflow_policy": "fail_closed",
        "longest_prompt_observation": (
            "completion_audit.max_observed_prompt_tokens"
        ),
    }
    if preflight != expected_preflight:
        raise RuntimeError(
            "Run contract 32k/60-step preflight metadata drift"
        )
    decoding = payload.get("decoding")
    if decoding != expected_decoding:
        raise RuntimeError("Run contract frozen decoding drift")
    judge = payload.get("judge")
    expected_strict_judge = {
        "module": STRICT_NL_JUDGE_MODULE,
        "entrypoint": STRICT_NL_JUDGE_ENTRYPOINT,
        "mode": "strict_json_schema_fail_closed",
    }
    if (
        not isinstance(judge, dict)
        or judge.get("strict_backend") != expected_strict_judge
    ):
        raise RuntimeError("Run contract strict NL judge entrypoint drift")


def _expected_result_task_ids(
    payload: dict[str, Any],
) -> dict[str, list[str]]:
    task_ids = payload.get("task_ids")
    conditions = payload.get("conditions")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or len(set(task_ids)) != len(task_ids)
    ):
        raise RuntimeError("Run contract task_ids are missing or duplicated")
    if conditions != ["clean", "error"]:
        raise RuntimeError("Run contract must cover both clean and error")
    by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for identity in task_ids:
        if not isinstance(identity, str) or ":" not in identity:
            raise RuntimeError(f"Malformed task identity in contract: {identity!r}")
        domain, task_id = identity.split(":", 1)
        if domain not in by_domain or not task_id:
            raise RuntimeError(f"Malformed task identity in contract: {identity!r}")
        by_domain[domain].append(task_id)
    expected: dict[str, list[str]] = {}
    for domain in DOMAINS:
        if not by_domain[domain]:
            continue
        for condition in conditions:
            name = output_filename(
                domain,
                condition,
                int(payload["shard_index"]),
                int(payload["num_shards"]),
            )
            expected[name] = sorted(by_domain[domain])
    return expected


def _validate_normalization_audit(
    message: dict[str, Any],
    *,
    path: Path,
    simulation_index: int,
    message_index: int,
) -> tuple[int, int]:
    location = (
        f"{path.name} simulation[{simulation_index}] message[{message_index}]"
    )
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        raise RuntimeError(f"{location}: tool_calls is not a list")
    if len(tool_calls) > 1:
        raise RuntimeError(
            f"{location}: response retained multiple parallel tool calls"
        )
    if tool_calls and message.get("content") not in (None, ""):
        raise RuntimeError(
            f"{location}: response retained mixed text and tool call"
        )
    raw_data = message.get("raw_data")
    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, dict):
        raise RuntimeError(f"{location}: raw_data is not an object")

    parallel_count = 0
    parallel = raw_data.get("v5_stage1_parallel_calls_serialized")
    if parallel is not None:
        if not tool_calls or not isinstance(parallel, dict):
            raise RuntimeError(f"{location}: malformed parallel-call audit")
        original_count = parallel.get("original_count")
        hashes = parallel.get("deferred_call_sha256")
        if (
            isinstance(original_count, bool)
            or not isinstance(original_count, int)
            or original_count < 2
            or not isinstance(hashes, list)
            or len(hashes) != original_count - 1
            or any(not _valid_sha256(value) for value in hashes)
        ):
            raise RuntimeError(f"{location}: malformed deferred-call hashes")
        parallel_count = 1

    mixed_count = 0
    mixed = raw_data.get("v5_stage1_mixed_content_normalized")
    if mixed is not None:
        if (
            not tool_calls
            or message.get("content") is not None
            or not isinstance(mixed, dict)
            or not _valid_sha256(mixed.get("sha256"))
            or isinstance(mixed.get("utf8_bytes"), bool)
            or not isinstance(mixed.get("utf8_bytes"), int)
            or mixed.get("utf8_bytes") <= 0
        ):
            raise RuntimeError(f"{location}: malformed mixed-content audit")
        mixed_count = 1
    return parallel_count, mixed_count


def audit_result_interface(
    path: Path,
    *,
    expected_task_ids: list[str],
    num_trials: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Validate task coverage, single-action normalization, and token limits."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse evaluation result {path}") from exc
    simulations = payload.get("simulations") if isinstance(payload, dict) else None
    if not isinstance(simulations, list) or not simulations:
        raise RuntimeError(f"{path.name}: simulations must be a non-empty list")
    if (
        isinstance(num_trials, bool)
        or not isinstance(num_trials, int)
        or num_trials <= 0
    ):
        raise RuntimeError("Run contract num_trials must be positive")
    observed_tasks = Counter()
    assistant_messages = 0
    tool_call_messages = 0
    parallel_normalizations = 0
    mixed_normalizations = 0
    usage_records = 0
    max_prompt_tokens = 0
    max_total_request_tokens = 0
    for simulation_index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            raise RuntimeError(
                f"{path.name} simulation[{simulation_index}] is not an object"
            )
        task_id = simulation.get("task_id")
        if not isinstance(task_id, (str, int)):
            raise RuntimeError(
                f"{path.name} simulation[{simulation_index}] lacks task_id"
            )
        observed_tasks[str(task_id)] += 1
        if simulation.get("termination_reason") == "context_window_exceeded":
            raise RuntimeError(
                f"{path.name} simulation[{simulation_index}] exceeded context"
            )
        messages = simulation.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(
                f"{path.name} simulation[{simulation_index}] lacks messages"
            )
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise RuntimeError(
                    f"{path.name} simulation[{simulation_index}] "
                    f"message[{message_index}] is not an object"
                )
            usage = message.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise RuntimeError(
                        f"{path.name} simulation[{simulation_index}] "
                        f"message[{message_index}] usage is not an object"
                    )
                if "prompt_tokens" in usage:
                    total = validate_request_token_budget(
                        usage["prompt_tokens"],
                        max_tokens=max_tokens,
                    )
                    usage_records += 1
                    max_prompt_tokens = max(
                        max_prompt_tokens,
                        usage["prompt_tokens"],
                    )
                    max_total_request_tokens = max(
                        max_total_request_tokens,
                        total,
                    )
            if message.get("role") != "assistant":
                continue
            assistant_messages += 1
            if message.get("tool_calls"):
                tool_call_messages += 1
            parallel, mixed = _validate_normalization_audit(
                message,
                path=path,
                simulation_index=simulation_index,
                message_index=message_index,
            )
            parallel_normalizations += parallel
            mixed_normalizations += mixed
    expected_counts = Counter(
        {
            str(task_id): num_trials
            for task_id in expected_task_ids
        }
    )
    if observed_tasks != expected_counts:
        raise RuntimeError(
            f"{path.name}: incomplete or duplicate task coverage; "
            f"expected={dict(expected_counts)}, observed={dict(observed_tasks)}"
        )
    if usage_records <= 0:
        raise RuntimeError(
            f"{path.name}: no prompt-token usage was available for overflow audit"
        )
    return {
        "simulation_count": len(simulations),
        "task_count": len(expected_counts),
        "assistant_message_count": assistant_messages,
        "tool_call_message_count": tool_call_messages,
        "parallel_call_normalization_count": parallel_normalizations,
        "mixed_content_normalization_count": mixed_normalizations,
        "prompt_usage_records_checked": usage_records,
        "max_observed_prompt_tokens": max_prompt_tokens,
        "max_observed_total_request_tokens": max_total_request_tokens,
        "context_window_tokens": MAX_MODEL_LEN,
        "max_steps": FROZEN_DECODING["max_steps"],
        "status": "PASS",
    }


def write_contract(
    args: argparse.Namespace,
    manifest_path: Path,
    split_manifest_path: Path,
    checkpoint_registry_path: Path,
    checkpoint_registry: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    local_source_commit: str,
    rows: list[dict[str, Any]],
    *,
    dynamic_audit_path: Path,
    dynamic_audit_identity: dict[str, Any],
    local_adapter_identity: dict[str, dict[str, str]],
    served_registry_aliases: list[str],
    runtime_service_evidence: dict[str, str] | None = None,
) -> Path:
    contract_path = args.output_dir / (
        "run_contract.json"
        if args.num_shards == 1
        else f"run_contract.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    )
    if contract_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing contract: {contract_path}")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_profile = checkpoint_registry_provenance_profile(
        checkpoint_registry
    )
    if provenance_profile == "v5_3_12h_screen":
        user_judge_revision = V5_3_12H_USER_JUDGE["revision"]
    elif provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        user_judge_revision = V5_3_LOW_SUPPORT_USER_JUDGE["revision"]
    else:
        user_judge_revision = checkpoint_registry["base_model_revision"]
    contract = {
        "protocol": RUN_CONTRACT_PROTOCOL,
        "status": "INCOMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "manifest": str(manifest_path),
        "evaluation_manifest_protocol": manifest_payload.get("protocol"),
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "fault_protocol": manifest_payload.get("fault_protocol"),
        "tau2_commit": manifest_payload.get("tau2_commit"),
        "source_files": manifest_payload.get("source_files"),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "dynamic_audit": str(dynamic_audit_path),
        "dynamic_audit_identity": dynamic_audit_identity,
        "checkpoint_registry": str(checkpoint_registry_path),
        "checkpoint_registry_sha256": sha256_file(checkpoint_registry_path),
        "checkpoint_registry_protocol": checkpoint_registry["protocol"],
        "checkpoint_registry_provenance_profile": (
            provenance_profile
        ),
        "checkpoint_entry": checkpoint_entry,
        "locally_verified_adapter_identity": local_adapter_identity,
        "served_registry_aliases": served_registry_aliases,
        "source_commit": local_source_commit,
        "base_model_revision": checkpoint_registry["base_model_revision"],
        "task_ids": [f"{row['domain']}:{row['task_id']}" for row in rows],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "agent": {
            "model": args.agent_model,
            "revision": checkpoint_registry["base_model_revision"],
            "api_base": args.agent_api_base,
        },
        "user": {
            "model": args.user_model,
            "revision": user_judge_revision,
            "api_base": args.user_api_base,
        },
        "judge": {
            "model": args.judge_model or args.user_model,
            "revision": user_judge_revision,
            "api_base": args.judge_api_base or args.user_api_base,
            "strict_backend": {
                "module": STRICT_NL_JUDGE_MODULE,
                "entrypoint": STRICT_NL_JUDGE_ENTRYPOINT,
                "mode": "strict_json_schema_fail_closed",
            },
        },
        "decoding": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "max_steps": args.max_steps,
            "task_timeout_seconds": args.timeout,
            "seed": args.seed,
            "num_trials": args.num_trials,
        },
        "tool_action_interface": dict(TOOL_ACTION_INTERFACE),
        "runtime_preflight": {
            "served_context_window_tokens": MAX_MODEL_LEN,
            "request_max_tokens": args.max_tokens,
            "maximum_nonoverflow_prompt_tokens": (
                MAX_MODEL_LEN - args.max_tokens
            ),
            "max_steps": args.max_steps,
            "request_token_overflow_policy": "fail_closed",
            "longest_prompt_observation": (
                "completion_audit.max_observed_prompt_tokens"
            ),
        },
        "conditions": (
            ["clean", "error"] if args.condition == "both" else [args.condition]
        ),
        "official_test_used": False,
        "result_sha256": {},
        "completion_audit": {},
        "strict_judge_audit_evidence": {},
    }
    if provenance_profile == "v5_3_12h_screen":
        contract["screen_claim_boundary"] = {
            "exploratory_screen_only": True,
            "formal_v5_3_result": False,
            "screen_outputs_may_enter_formal_v5_3": False,
            "paper_level_confirmation": False,
            "official_test_used": False,
            "official_test_sealed": True,
        }
    elif provenance_profile == V5_3_LOW_SUPPORT_PROFILE:
        if (
            not isinstance(runtime_service_evidence, dict)
            or set(runtime_service_evidence)
            != {"path", "sha256", "canonical_sha256"}
            or not _valid_sha256(runtime_service_evidence.get("sha256"))
            or not _valid_sha256(
                runtime_service_evidence.get("canonical_sha256")
            )
        ):
            raise RuntimeError(
                "low-support contract requires hash-bound runtime service "
                "evidence"
            )
        contract["diagnostic_claim_boundary"] = dict(
            V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
        )
        contract["diagnostic_protocol"] = (
            V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL
        )
        contract["runtime_service_evidence"] = dict(
            runtime_service_evidence
        )
    contract["contract_core_sha256"] = contract_core_sha256(contract)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return contract_path


def finalize_contract(contract_path: Path, output_files: list[Path]) -> dict[str, str]:
    """Atomically bind a completed run contract to this shard's result bytes."""

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("status") != "INCOMPLETE":
        raise RuntimeError(f"Run contract is not INCOMPLETE: {contract_path}")
    if (
        payload.get("result_sha256") != {}
        or payload.get("completion_audit") != {}
        or payload.get("strict_judge_audit_evidence", {}) != {}
    ):
        raise RuntimeError(
            "INCOMPLETE run contract contains stale completion artifacts"
        )
    validate_contract_core(payload)
    expected_results = _expected_result_task_ids(payload)
    observed_names = [path.resolve().name for path in output_files]
    if (
        len(observed_names) != len(set(observed_names))
        or set(observed_names) != set(expected_results)
    ):
        raise RuntimeError(
            "Evaluation result set does not exactly cover the contract; "
            f"expected={sorted(expected_results)}, observed={sorted(observed_names)}"
        )
    result_hashes: dict[str, str] = {}
    completion_files: dict[str, dict[str, Any]] = {}
    for path in output_files:
        resolved = path.resolve()
        if resolved.parent != contract_path.parent.resolve() or not resolved.is_file():
            raise RuntimeError(
                f"Result file is absent or outside contract directory: {path}"
            )
        if resolved.name in result_hashes:
            raise RuntimeError(f"Duplicate result filename: {resolved.name}")
        completion_files[resolved.name] = audit_result_interface(
            resolved,
            expected_task_ids=expected_results[resolved.name],
            num_trials=int(payload["decoding"]["num_trials"]),
            max_tokens=int(payload["decoding"]["max_tokens"]),
        )
        result_hashes[resolved.name] = sha256_file(resolved)
    if not result_hashes:
        raise RuntimeError("Cannot finalize a contract without result files")
    payload["result_sha256"] = dict(sorted(result_hashes.items()))
    payload["completion_audit"] = {
        "protocol": "v5_stage1_sft_causal_interface_completion_audit",
        "tool_action_interface": dict(TOOL_ACTION_INTERFACE),
        "result_files": dict(sorted(completion_files.items())),
        "max_observed_prompt_tokens": max(
            item["max_observed_prompt_tokens"]
            for item in completion_files.values()
        ),
        "max_observed_total_request_tokens": max(
            item["max_observed_total_request_tokens"]
            for item in completion_files.values()
        ),
        "all_expected_tasks_observed_once_per_trial": True,
        "request_token_overflow_detected": False,
        "status": "PASS",
    }
    if payload.get("checkpoint_registry_provenance_profile") in {
        "v5_3",
        "v5_3_12h_screen",
        V5_3_LOW_SUPPORT_PROFILE,
    }:
        try:
            strict_evidence = (
                judge_audit_contract.validate_strict_judge_evidence(
                    output_files,
                    maximum_content_attempts=2,
                )
            )
        except judge_audit_contract.StrictJudgeEvidenceError as error:
            raise RuntimeError(
                "V5.3 evaluation strict-judge evidence is incomplete"
            ) from error
        if (
            payload.get("checkpoint_registry_provenance_profile")
            in {"v5_3_12h_screen", V5_3_LOW_SUPPORT_PROFILE}
            and (
                strict_evidence.get("status") != "PASS"
                or isinstance(strict_evidence.get("expected_calls"), bool)
                or not isinstance(strict_evidence.get("expected_calls"), int)
                or strict_evidence["expected_calls"] <= 0
                or strict_evidence.get("observed_unique_pass_audits")
                != strict_evidence["expected_calls"]
            )
        ):
            raise RuntimeError(
                "V5.3-12h evaluation strict-judge evidence is zero "
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
    split_manifest_path = args.split_manifest.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_registry_path = args.checkpoint_registry.resolve()
    checkpoint_registry = load_checkpoint_registry(
        checkpoint_registry_path,
        expected_profile=args.provenance_profile,
    )
    # The registry is the authoritative profile.  An omitted CLI profile must
    # still select the registered screen decoding contract rather than falling
    # back to legacy defaults.
    args.provenance_profile = checkpoint_registry_provenance_profile(
        checkpoint_registry
    )
    validate_evaluation_protocol(args)
    split_manifest = load_split_manifest(split_manifest_path)
    manifest = load_manifest(
        manifest_path,
        split_manifest=split_manifest,
        split_manifest_sha256=sha256_file(split_manifest_path),
    )
    dynamic_audit_path = args.dynamic_audit.resolve()
    dynamic_audit_identity = load_complete_dynamic_audit(
        dynamic_audit_path,
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        expected_source_split="derived_validation",
        expected_task_ids={
            f"{row['domain']}:{row['task_id']}" for row in manifest["rows"]
        },
    )
    validate_local_manifest_sources(manifest, args.tau2_root.resolve())
    rows = shard_rows(
        list(manifest["rows"]),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not rows:
        raise RuntimeError("Selected validation shard is empty")

    judge_model = args.judge_model or args.user_model
    judge_api_base = args.judge_api_base or args.user_api_base
    judge_api_key = args.judge_api_key or args.user_api_key
    profile = checkpoint_registry_provenance_profile(checkpoint_registry)
    if (
        profile == V5_3_LOW_SUPPORT_PROFILE
        and args.output_dir.resolve()
        != low_support.artifact_root(ROOT, "results_root")
        / "evaluation"
        / args.arm
    ):
        raise RuntimeError(
            "low-support evaluation output is outside its isolated root"
        )
    if profile == V5_3_LOW_SUPPORT_PROFILE:
        low_support.require_whole_run_source_lock(ROOT)
    if profile in {"v5_3_12h_screen", V5_3_LOW_SUPPORT_PROFILE}:
        agent_api_base, user_judge_api_base = require_screen_api_bases(
            args.agent_api_base,
            args.user_api_base,
            judge_api_base,
        )
        args.agent_api_base = agent_api_base
        args.user_api_base = user_judge_api_base
        args.judge_api_base = user_judge_api_base
        judge_api_base = user_judge_api_base
        if profile == V5_3_LOW_SUPPORT_PROFILE:
            if (
                args.shard_index not in range(low_support.EVALUATION_SHARDS)
                or args.agent_api_base
                != low_support.RUNTIME_ENDPOINTS[args.shard_index + 1]
                or args.user_api_base != low_support.RUNTIME_ENDPOINTS[0]
            ):
                raise RuntimeError(
                    "low-support shard/service endpoint topology drift"
                )
    else:
        shared_api_base = require_shared_api_base(
            args.agent_api_base,
            args.user_api_base,
            judge_api_base,
        )
        args.agent_api_base = shared_api_base
        args.user_api_base = shared_api_base
        args.judge_api_base = shared_api_base
        judge_api_base = shared_api_base
    if (
        checkpoint_registry["training_data_provenance"]["dynamic_audits"][
            "validation"
        ]
        != dynamic_audit_identity
    ):
        raise RuntimeError(
            "Checkpoint registry validation audit differs from evaluation audit"
        )
    expected_arms = arms_for_profile(profile)
    if args.arm not in expected_arms:
        raise RuntimeError(
            f"Arm {args.arm!r} is not registered for profile {profile!r}; "
            f"expected exactly {sorted(expected_arms)}"
        )
    adapter_dirs = parse_adapter_dir_bindings(
        args.adapter_dir,
        expected_arms=trained_arms_for_profile(profile),
    )
    local_adapter_identity = validate_local_adapter_identity(
        checkpoint_registry,
        adapter_dirs,
    )
    served_registry_aliases = verify_served_registry_aliases(
        args.agent_api_base,
        args.agent_api_key,
        checkpoint_registry,
        require_exact=(profile == V5_3_LOW_SUPPORT_PROFILE),
    )
    if profile == "v5_3_12h_screen":
        verify_served_model_id(
            args.user_api_base,
            args.user_api_key,
            V5_3_12H_USER_JUDGE["model_id"],
        )
    elif profile == V5_3_LOW_SUPPORT_PROFILE:
        verify_served_model_id(
            args.user_api_base,
            args.user_api_key,
            V5_3_LOW_SUPPORT_USER_JUDGE["model_id"],
            require_exact=True,
        )
    require_clean_tracked_source()
    local_source_commit = git_commit()
    runtime_service_binding: dict[str, str] | None = None
    if profile == V5_3_LOW_SUPPORT_PROFILE:
        if args.runtime_service_evidence is None:
            raise RuntimeError(
                "--runtime-service-evidence is required for the low-support "
                "diagnostic"
            )
        _, runtime_service_binding = (
            load_low_support_runtime_service_evidence(
                args.runtime_service_evidence,
                checkpoint_registry_path=checkpoint_registry_path,
                checkpoint_registry=checkpoint_registry,
                expected_source_commit=local_source_commit,
            )
        )
    elif args.runtime_service_evidence is not None:
        raise RuntimeError(
            "--runtime-service-evidence is reserved for the low-support "
            "diagnostic"
        )
    checkpoint_entry = validate_checkpoint_identity(
        checkpoint_registry,
        arm=args.arm,
        agent_model=args.agent_model,
        local_source_commit=local_source_commit,
        user_model=args.user_model,
        judge_model=judge_model,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = write_contract(
        args,
        manifest_path,
        split_manifest_path,
        checkpoint_registry_path,
        checkpoint_registry,
        checkpoint_entry,
        local_source_commit,
        rows,
        dynamic_audit_path=dynamic_audit_path,
        dynamic_audit_identity=dynamic_audit_identity,
        local_adapter_identity=local_adapter_identity,
        served_registry_aliases=served_registry_aliases,
        runtime_service_evidence=runtime_service_binding,
    )
    configure_tau2_path(args.tau2_root)
    os.environ.setdefault("OPENAI_API_KEY", args.agent_api_key)
    register_fault_agent()

    agent_args = endpoint_args(
        args.agent_api_base,
        args.agent_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    user_args = endpoint_args(
        args.user_api_base,
        args.user_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    judge_args = endpoint_args(
        judge_api_base,
        judge_api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    patch_local_nl_judge(judge_model, judge_args)

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
                    agent_args=agent_args,
                    user_args=user_args,
                )
            )
    # The contract remains INCOMPLETE if any preceding evaluation raises.
    # Only successful completion atomically binds it to the result bytes.
    result_hashes = finalize_contract(contract_path, output_files)
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "agent_model": args.agent_model,
                "checkpoint_registry_sha256": sha256_file(
                    checkpoint_registry_path
                ),
                "contract_result_sha256": result_hashes,
                "result_files": [str(path) for path in output_files],
                "official_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
