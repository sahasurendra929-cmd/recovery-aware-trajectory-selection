#!/usr/bin/env python3
"""Generate fresh, environment-grounded V6 recovery candidate pairs.

This executable materializes the outcome-independent registry produced by
``prepare_v6_candidate_registry.py``.  It is intentionally fail closed:

* only registry tasks in the selected phase are touched;
* all candidate pairs for a task share one observed clean prefix and one
  environment snapshot;
* an injected call must produce a real tau2 tool error while leaving both
  databases unchanged;
* the clean future is removed before a recovery rollout is generated;
* the first recovery tool action must equal the registered corrective call;
* failed calls/results are context and never positive labels;
* matched and crossed forced-first cells use the same continuation policy,
  decoding contract, seed set, and rollout budget.

The output is an unscored pool.  ``measure_v6_candidate_tokens.py`` adds exact
token costs and frozen-base first-action log probabilities before
``score_v6_candidates.py`` computes hardness and kappa ranks.

No official-test identity or content is accepted by this program.
"""
from __future__ import annotations

import argparse
import atexit
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import fcntl
except ImportError:  # pragma: no cover - V6.10 formal runtime is Linux-only
    fcntl = None

try:
    import prepare_v6_candidate_registry as registry_contract
    import v6_10_selection_protocol as v610_protocol
    import v6_reference_contract as reference_contract
    import v6_selection_protocol as protocol
    from run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import prepare_v6_candidate_registry as registry_contract
    from scripts import v6_10_selection_protocol as v610_protocol
    from scripts import v6_reference_contract as reference_contract
    from scripts import v6_selection_protocol as protocol
    from scripts.run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )


GENERATION_PROTOCOL = "v6_fresh_recovery_candidate_generation_v1"
SEMANTIC_GENERATION_CONTRACT_PROTOCOL = (
    "v6_fresh_recovery_semantic_generation_contract_v1"
)
V6_10_PIPELINE_CLOSURE_PROTOCOL = "v6_10_pipeline_closure_v1"
V610_TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
V610_TEACHER_REVISION = "698703eae6604af048a3d2f509995dc302088217"
V610_USER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
V610_USER_JUDGE_REVISION = "539535859b135b0244c91f3e59816150c8056698"
V610_COMPATIBILITY_SEEDS = (20260806,)
V610_SCIENTIFIC_SEEDS = (20260806, 20260807, 20260808)
V610_MAX_STEPS = 60
V610_MAX_TOKENS = 512
V610_MODEL_SERVER_RECEIPTS_PROTOCOL = "v6_10_model_server_receipts_v1"
V610_SOURCE_CONTAINER_RECEIPT_PROTOCOL = (
    "v6_10_source_container_provenance_v1"
)
V610_SELECTED_PHASE_PREFLIGHT_PROTOCOL = (
    "v6_10_selected_phase_pre_model_preflight_v1"
)
V610_RELEASE_MANIFEST_PROTOCOL = "v6_10_release_manifest_v1"
V610_TASK_RESTART_LEDGER_PROTOCOL = "v6_10_task_restart_ledger_v1"
V610_TASK_BUNDLE_PROTOCOL = "v6_10_atomic_task_bundle_v1"
V610_MAX_PROCESS_RESTARTS = 2
V610_ATTEMPT_DIR_RE = re.compile(
    r"^attempt-(?P<number>0*[1-9][0-9]*)(?:-(?P<suffix>.+))?$"
)
_V610_ACTIVE_RUN_LOCKS: list[Any] = []
V610_REQUIRED_RELEASE_SCRIPTS = frozenset(
    {
        "scripts/run_v6_candidate_generation.py",
        "scripts/audit_v6_candidates.py",
        "scripts/materialize_v6_sft.py",
        "scripts/train_v6_directional_sft.py",
        "scripts/measure_v6_candidate_tokens.py",
        "scripts/score_v6_candidates.py",
        "scripts/build_v6_selector_manifests.py",
        "scripts/build_v6_checkpoint_registry.py",
        "scripts/preflight_v6_reference_traces.py",
        "scripts/prepare_v6_10_registry.py",
        "scripts/build_v6_10_runtime_receipts.py",
        "scripts/build_v6_10_release_manifest.py",
        "scripts/v6_10_selection_protocol.py",
        "scripts/v6_reference_contract.py",
    }
)
V610_PARENT_ARTIFACT_HASH_KEYS = frozenset(
    {
        "config_sha256",
        "preregistration_sha256",
        "split_manifest_sha256",
        "reference_preflight_receipt_sha256",
        "registry_file_sha256",
        "source_container_provenance_sha256",
        "model_server_receipts_sha256",
        "runtime_receipt_hashes_sha256",
        "release_manifest_sha256",
    }
)
V610_MODEL_QUANTIZATION = "awq"
V610_ROLE_TENSOR_PARALLEL_SIZE = {
    "teacher": 2,
    "user": 1,
    "judge": 1,
}
AGENT_NAME = "v6_fresh_recovery_agent"
REFERENCE_GUIDED_CLEAN_AGENT_NAME = "llm_agent_gt"
CLEAN_AGENT_MODES = (
    "standard",
    "reference_guided",
    "deterministic_reference_replay",
    "single_turn_user_reference_replay",
)
RECOVERY_CONTINUATION_MODES = (
    "fresh_teacher",
    "deterministic_reference_tail",
    "deterministic_reference_completion",
)
COMPLETION_RENDERERS = (
    "legacy_assertion_echo",
    "natural_direct_v1",
    "explicit_user_direct_v2",
    "explicit_user_direct_v3",
)
SFT_SYSTEM_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
V610_ATTEMPT_ID_RE_BY_PHASE = {
    "compatibility": re.compile(
        r"^v6_10-compat-a\d{2}-\d{8}T\d{6}Z$"
    ),
    "pilot": re.compile(r"^v6_10-pilot-a\d{2}-\d{8}T\d{6}Z$"),
    "formal": re.compile(r"^v6_10-formal-a\d{2}-\d{8}T\d{6}Z$"),
}
DEFAULT_CONTINUATION_SEEDS = (20260821,)
DECODING_TEMPERATURE = 0.0
DECODING_TOP_P = 1.0
PARALLEL_TOOL_CALLS = False
RUN_NUM_TRIALS = 1
RUN_MAX_ERRORS = 10
RUN_MAX_CONCURRENCY = 1
RUN_MAX_RETRIES = 1
RUN_RETRY_DELAY_SECONDS = 1.0
RUN_HALLUCINATION_RETRIES = 0
# Tau2 treats ``max_steps`` as the largest zero-based step index.  Steps 0
# and 1 therefore retain exactly the greeting and first user response.
SINGLE_TURN_PREFIX_MAX_STEPS = 1
SINGLE_TURN_PREFIX_MESSAGE_COUNT = 2
VOLATILE_MESSAGE_FIELDS = {
    "timestamp",
    "turn_idx",
    "cost",
    "usage",
    "generation_time_seconds",
}
# Pydantic expands these tau2 participant-message transport fields even when
# compact frozen history omits them.  They are equivalent only at the exact
# defaults below; non-default audio/streaming values remain semantic drift.
DEFAULT_TAU2_MESSAGE_TRANSPORT_FIELDS = {
    "is_audio": False,
    "raw_data": None,
    "audio_format": None,
    "audio_path": None,
    "audio_script_gold": None,
    "speech_effects": None,
    "source_effects": None,
    "channel_effects": None,
    "turn_taking_action": None,
    "utterance_ids": None,
    "chunk_id": None,
    "is_final_chunk": True,
    "source": None,
    "contains_speech": True,
}


class V6GenerationError(RuntimeError):
    """A registry, environment, or fresh-recovery invariant failed."""


class V6RawProbeMalformedResponse(V6GenerationError):
    """A completed raw-probe request returned an unparseable model response."""

    def __init__(
        self,
        *,
        evidence: Mapping[str, Any],
        wall_seconds: float,
    ) -> None:
        self.evidence = deepcopy(dict(evidence))
        self.wall_seconds = float(wall_seconds)
        super().__init__("raw teacher first response was malformed")


class V6RawProbeInfrastructureError(V6GenerationError):
    """The raw-probe request failed before a scoreable response existed."""


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise V6GenerationError(
            f"cannot inspect git provenance for {root}: {' '.join(arguments)}"
        ) from error
    return result.stdout


def git_provenance(root: Path) -> dict[str, Any]:
    """Return a tracked-source proof suitable for a semantic run contract."""

    resolved = root.resolve()
    commit = _git_output(resolved, "rev-parse", "HEAD").strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise V6GenerationError(f"{resolved}: git HEAD is not a full commit")
    status = _git_output(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    tree = _git_output(resolved, "rev-parse", "HEAD^{tree}").strip()
    if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise V6GenerationError(f"{resolved}: git tree is not a full object id")
    return {
        "commit": commit,
        "tree": tree,
        "tracked_worktree_clean": status == "",
        "tracked_worktree_status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "worktree_scope": "tracked_and_untracked_files",
    }


def load_model_server_receipts(
    path: Path,
    *,
    expected_sha256: str,
    expected_roles: Mapping[str, Mapping[str, Any]],
    container_image_digest: str,
    available_gpu_uuids: Sequence[str],
    expected_python_executable_path: str,
    expected_python_executable_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact role-scoped V6.10 model-server receipt schema."""

    if (
        SHA256_RE.fullmatch(expected_sha256) is None
        or OCI_DIGEST_RE.fullmatch(container_image_digest) is None
        or not isinstance(expected_python_executable_path, str)
        or not Path(expected_python_executable_path).is_absolute()
        or SHA256_RE.fullmatch(expected_python_executable_sha256) is None
    ):
        raise V6GenerationError("expected model-server receipt SHA-256 is invalid")
    if not path.is_file():
        raise V6GenerationError(f"model-server receipt is absent: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise V6GenerationError(
            "model-server receipt file hash drift: "
            f"{observed_sha256} != {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            "model-server receipt file is not valid JSON"
        ) from error
    top_keys = {
        "protocol",
        "design_protocol",
        "status",
        "container_image_digest",
        "roles",
        "official_test_used",
    }
    role_keys = {
        "role",
        "model",
        "resolved_revision",
        "api_base",
        "container_image_digest",
        "quantization",
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "gpu_uuids",
        "launch_command",
        "launch_command_sha256",
        "server_pid",
        "process_boot_id",
        "process_start_time_ticks",
        "process_started_at_utc",
        "observed_cmdline_sha256",
        "observed_executable_path",
        "observed_executable_sha256",
        "model_snapshot_path",
        "snapshot_identity_files",
        "snapshot_identity_files_sha256",
        "observed_cuda_visible_devices",
        "observed_gpu_uuids",
        "listening_socket_inode",
        "selected_process_environment",
        "tokenizer_or_config_sha256",
        "launched_at_utc",
        "healthcheck",
        "generation_probe",
    }
    healthcheck_keys = {
        "status",
        "observed_model_ids",
        "checked_at_utc",
    }
    generation_probe_keys = {
        "status",
        "prompt_sha256",
        "input_token_count",
        "max_new_tokens",
        "generated_token_count",
        "seed",
        "temperature",
        "response_sha256",
        "latency_seconds",
        "peak_gpu_memory_bytes",
    }
    roles = payload.get("roles") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != top_keys
        or payload.get("protocol") != V610_MODEL_SERVER_RECEIPTS_PROTOCOL
        or payload.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("container_image_digest") != container_image_digest
        or payload.get("official_test_used") is not False
        or not isinstance(roles, dict)
        or set(roles) != {"teacher", "user", "judge"}
        or set(expected_roles) != {"teacher", "user", "judge"}
    ):
        raise V6GenerationError(
            "model-server receipt top-level schema/identity drift"
        )
    allowed_gpu_uuids = set(available_gpu_uuids)
    if (
        not allowed_gpu_uuids
        or len(allowed_gpu_uuids) != len(tuple(available_gpu_uuids))
        or any(not isinstance(item, str) or not item for item in allowed_gpu_uuids)
    ):
        raise V6GenerationError("source-container GPU UUID inventory is invalid")
    normalized_roles: dict[str, dict[str, Any]] = {}
    for role in ("teacher", "user", "judge"):
        observed = roles.get(role)
        expected = expected_roles[role]
        healthcheck = (
            observed.get("healthcheck")
            if isinstance(observed, Mapping)
            else None
        )
        observed_model_ids = (
            healthcheck.get("observed_model_ids")
            if isinstance(healthcheck, Mapping)
            else None
        )
        launch_command = (
            observed.get("launch_command")
            if isinstance(observed, Mapping)
            else None
        )
        gpu_uuids = (
            observed.get("gpu_uuids")
            if isinstance(observed, Mapping)
            else None
        )
        generation_probe = (
            observed.get("generation_probe")
            if isinstance(observed, Mapping)
            else None
        )
        probe_latency = (
            generation_probe.get("latency_seconds")
            if isinstance(generation_probe, Mapping)
            else None
        )
        snapshot_identity_files = (
            observed.get("snapshot_identity_files")
            if isinstance(observed, Mapping)
            else None
        )
        byte_files = (
            snapshot_identity_files.get("byte_files")
            if isinstance(snapshot_identity_files, Mapping)
            else None
        )
        weight_blob_targets = (
            snapshot_identity_files.get("weight_blob_targets")
            if isinstance(snapshot_identity_files, Mapping)
            else None
        )
        selected_environment = (
            observed.get("selected_process_environment")
            if isinstance(observed, Mapping)
            else None
        )
        observed_gpu_uuids = (
            observed.get("observed_gpu_uuids")
            if isinstance(observed, Mapping)
            else None
        )
        if (
            not isinstance(observed, dict)
            or set(observed) != role_keys
            or observed.get("role") != role
            or observed.get("model") != expected.get("model")
            or observed.get("resolved_revision") != expected.get("revision")
            or observed.get("api_base") != expected.get("api_base")
            or observed.get("container_image_digest")
            != container_image_digest
            or observed.get("quantization")
            != expected.get("quantization")
            or observed.get("dtype") != "float16"
            or observed.get("tensor_parallel_size")
            != expected.get("tensor_parallel_size")
            or observed.get("max_model_len") != 8192
            or observed.get("gpu_memory_utilization") != 0.90
            or not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != expected.get("tensor_parallel_size")
            or len(gpu_uuids) != len(set(gpu_uuids))
            or any(
                not isinstance(item, str) or item not in allowed_gpu_uuids
                for item in gpu_uuids
            )
            or not isinstance(launch_command, list)
            or not launch_command
            or any(not isinstance(item, str) or not item for item in launch_command)
            or launch_command[0] != expected_python_executable_path
            or SHA256_RE.fullmatch(
                str(observed.get("launch_command_sha256"))
            )
            is None
            or observed.get("launch_command_sha256")
            != sha256(launch_command)
            or isinstance(observed.get("server_pid"), bool)
            or not isinstance(observed.get("server_pid"), int)
            or observed["server_pid"] <= 0
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}",
                str(observed.get("process_boot_id")),
            )
            is None
            or isinstance(observed.get("process_start_time_ticks"), bool)
            or not isinstance(observed.get("process_start_time_ticks"), int)
            or observed["process_start_time_ticks"] <= 0
            or not valid_utc_timestamp(
                observed.get("process_started_at_utc")
            )
            or observed.get("launched_at_utc")
            != observed.get("process_started_at_utc")
            or SHA256_RE.fullmatch(
                str(observed.get("observed_cmdline_sha256"))
            )
            is None
            or observed.get("observed_cmdline_sha256")
            != observed.get("launch_command_sha256")
            or not isinstance(observed.get("observed_executable_path"), str)
            or not Path(observed["observed_executable_path"]).is_absolute()
            or SHA256_RE.fullmatch(
                str(observed.get("observed_executable_sha256"))
            )
            is None
            or observed.get("observed_executable_sha256")
            != expected_python_executable_sha256
            or not isinstance(observed.get("model_snapshot_path"), str)
            or not Path(observed["model_snapshot_path"]).is_absolute()
            or not isinstance(snapshot_identity_files, dict)
            or set(snapshot_identity_files)
            != {"byte_files", "weight_blob_targets"}
            or not isinstance(byte_files, dict)
            or not byte_files
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or SHA256_RE.fullmatch(str(value)) is None
                for name, value in byte_files.items()
            )
            or not isinstance(weight_blob_targets, dict)
            or not weight_blob_targets
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", str(value))
                is None
                for name, value in weight_blob_targets.items()
            )
            or SHA256_RE.fullmatch(
                str(observed.get("snapshot_identity_files_sha256"))
            )
            is None
            or observed.get("snapshot_identity_files_sha256")
            != sha256(snapshot_identity_files)
            or observed.get("tokenizer_or_config_sha256")
            != observed.get("snapshot_identity_files_sha256")
            or not isinstance(
                observed.get("observed_cuda_visible_devices"), str
            )
            or not observed["observed_cuda_visible_devices"]
            or observed_gpu_uuids != gpu_uuids
            or not isinstance(observed.get("listening_socket_inode"), str)
            or re.fullmatch(
                r"[1-9][0-9]*",
                observed["listening_socket_inode"],
            )
            is None
            or not isinstance(selected_environment, dict)
            or set(selected_environment)
            != {
                "pythonpath_override_absent",
                "pythonhome_override_absent",
                "virtual_env",
            }
            or selected_environment.get("pythonpath_override_absent")
            is not True
            or selected_environment.get("pythonhome_override_absent")
            is not True
            or (
                selected_environment.get("virtual_env") is not None
                and not isinstance(
                    selected_environment.get("virtual_env"), str
                )
            )
            or SHA256_RE.fullmatch(
                str(observed.get("tokenizer_or_config_sha256"))
            )
            is None
            or not valid_utc_timestamp(observed.get("launched_at_utc"))
            or not isinstance(healthcheck, dict)
            or set(healthcheck) != healthcheck_keys
            or healthcheck.get("status") != "PASS"
            or not valid_utc_timestamp(healthcheck.get("checked_at_utc"))
            or not isinstance(observed_model_ids, list)
            or any(not isinstance(item, str) for item in observed_model_ids)
            or observed_model_ids != [expected.get("model")]
            or not isinstance(generation_probe, dict)
            or set(generation_probe) != generation_probe_keys
            or generation_probe.get("status") != "PASS"
            or SHA256_RE.fullmatch(
                str(generation_probe.get("prompt_sha256"))
            )
            is None
            or isinstance(
                generation_probe.get("input_token_count"), bool
            )
            or not isinstance(
                generation_probe.get("input_token_count"), int
            )
            or generation_probe["input_token_count"] < 4096
            or generation_probe.get("max_new_tokens") != 128
            or isinstance(
                generation_probe.get("generated_token_count"), bool
            )
            or not isinstance(
                generation_probe.get("generated_token_count"), int
            )
            or not (
                1
                <= generation_probe["generated_token_count"]
                <= generation_probe["max_new_tokens"]
            )
            or generation_probe.get("seed") != V610_COMPATIBILITY_SEEDS[0]
            or generation_probe.get("temperature") != 0.0
            or SHA256_RE.fullmatch(
                str(generation_probe.get("response_sha256"))
            )
            is None
            or isinstance(probe_latency, bool)
            or not isinstance(probe_latency, (int, float))
            or not math.isfinite(float(probe_latency))
            or float(probe_latency) <= 0.0
            or isinstance(
                generation_probe.get("peak_gpu_memory_bytes"), bool
            )
            or not isinstance(
                generation_probe.get("peak_gpu_memory_bytes"), int
            )
            or generation_probe["peak_gpu_memory_bytes"] <= 0
        ):
            raise V6GenerationError(
                f"model-server receipt role path drift: roles.{role}"
            )
        normalized_roles[role] = deepcopy(observed)
    teacher = normalized_roles["teacher"]
    user = normalized_roles["user"]
    judge = normalized_roles["judge"]
    teacher_gpus = set(teacher["gpu_uuids"])
    user_gpus = set(user["gpu_uuids"])
    judge_gpus = set(judge["gpu_uuids"])
    shared_process_fields = {
        "server_pid",
        "process_boot_id",
        "process_start_time_ticks",
        "process_started_at_utc",
        "observed_cmdline_sha256",
        "observed_executable_path",
        "observed_executable_sha256",
        "model_snapshot_path",
        "snapshot_identity_files",
        "snapshot_identity_files_sha256",
        "observed_cuda_visible_devices",
        "observed_gpu_uuids",
        "listening_socket_inode",
        "selected_process_environment",
        "launch_command",
        "launch_command_sha256",
        "tokenizer_or_config_sha256",
        "launched_at_utc",
        "max_model_len",
        "gpu_memory_utilization",
    }
    if (
        user["model"] != judge["model"]
        or user["resolved_revision"] != judge["resolved_revision"]
        or user["api_base"] != judge["api_base"]
        or user["container_image_digest"] != judge["container_image_digest"]
        or user["quantization"] != judge["quantization"]
        or user["dtype"] != judge["dtype"]
        or user_gpus != judge_gpus
        or any(
            user[field] != judge[field] for field in shared_process_fields
        )
        or teacher["server_pid"] == user["server_pid"]
        or len(teacher_gpus) != 2
        or len(user_gpus) != 1
        or not teacher_gpus.isdisjoint(user_gpus)
        or teacher_gpus | user_gpus != allowed_gpu_uuids
    ):
        raise V6GenerationError(
            "model-server receipt cross-role 3-GPU topology drift"
        )
    return payload, {
        "file_sha256": observed_sha256,
        "semantic_sha256": sha256(payload),
        "protocol": V610_MODEL_SERVER_RECEIPTS_PROTOCOL,
        "roles": normalized_roles,
        "container_image_digest": container_image_digest,
        "python_executable_path": expected_python_executable_path,
        "python_executable_sha256": expected_python_executable_sha256,
    }


def load_source_container_provenance(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_commit: str,
    expected_generation_script_sha256: str,
    expected_tau2_commit: str,
    container_image_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        SHA256_RE.fullmatch(expected_sha256) is None
        or OCI_DIGEST_RE.fullmatch(container_image_digest) is None
    ):
        raise V6GenerationError(
            "expected source/container provenance SHA-256 is invalid"
        )
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise V6GenerationError(
            "source/container provenance file is absent or hash-drifted"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            "source/container provenance is not valid JSON"
        ) from error
    expected_keys = {
        "protocol",
        "design_protocol",
        "status",
        "source_commit",
        "generation_script_sha256",
        "tau2_commit",
        "container_image_digest",
        "dependency_lock_sha256",
        "runtime",
        "gpus",
        "created_at_utc",
        "official_test_used",
    }
    runtime_keys = {
        "python_version",
        "python_executable_path",
        "python_executable_sha256",
        "torch_version",
        "cuda_version",
        "driver_version",
        "vllm_version",
    }
    gpu_keys = {"index", "model", "uuid", "total_memory_bytes"}
    runtime = payload.get("runtime") if isinstance(payload, Mapping) else None
    gpus = payload.get("gpus") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("protocol")
        != V610_SOURCE_CONTAINER_RECEIPT_PROTOCOL
        or payload.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("source_commit") != expected_source_commit
        or payload.get("generation_script_sha256")
        != expected_generation_script_sha256
        or payload.get("tau2_commit") != expected_tau2_commit
        or payload.get("container_image_digest") != container_image_digest
        or SHA256_RE.fullmatch(
            str(payload.get("dependency_lock_sha256"))
        )
        is None
        or not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in runtime_keys
        )
        or not Path(runtime["python_executable_path"]).is_absolute()
        or SHA256_RE.fullmatch(runtime["python_executable_sha256"]) is None
        or not Path(runtime["python_executable_path"]).is_file()
        or sha256_file(Path(runtime["python_executable_path"]))
        != runtime["python_executable_sha256"]
        or not isinstance(gpus, list)
        or len(gpus) != 3
        or not valid_utc_timestamp(payload.get("created_at_utc"))
        or payload.get("official_test_used") is not False
    ):
        raise V6GenerationError(
            "source/container provenance exact schema/identity drift"
        )
    gpu_by_index: dict[int, dict[str, Any]] = {}
    for row in gpus:
        index = row.get("index") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, dict)
            or set(row) != gpu_keys
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index in gpu_by_index
            or not isinstance(row.get("model"), str)
            or not row["model"]
            or not isinstance(row.get("uuid"), str)
            or not row["uuid"]
            or isinstance(row.get("total_memory_bytes"), bool)
            or not isinstance(row.get("total_memory_bytes"), int)
            or row["total_memory_bytes"] <= 0
        ):
            raise V6GenerationError(
                "source/container provenance GPU inventory drift"
            )
        gpu_by_index[index] = deepcopy(row)
    if (
        set(gpu_by_index) != {0, 1, 2}
        or len({row["uuid"] for row in gpu_by_index.values()}) != 3
    ):
        raise V6GenerationError(
            "source/container provenance requires three distinct indexed GPUs"
        )
    return payload, {
        "file_sha256": expected_sha256,
        "semantic_sha256": sha256(payload),
        "protocol": V610_SOURCE_CONTAINER_RECEIPT_PROTOCOL,
        "source_commit": expected_source_commit,
        "generation_script_sha256": expected_generation_script_sha256,
        "tau2_commit": expected_tau2_commit,
        "container_image_digest": container_image_digest,
        "dependency_lock_sha256": payload["dependency_lock_sha256"],
        "runtime": deepcopy(runtime),
        "gpus": [gpu_by_index[index] for index in sorted(gpu_by_index)],
        "created_at_utc": payload["created_at_utc"],
    }


def verify_live_model_endpoint(
    *,
    role: str,
    api_base: str,
    expected_model: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Fail closed unless the live OpenAI-compatible endpoint serves the model."""

    url = f"{api_base.rstrip('/')}/models"
    request = urllib_request.Request(
        url,
        headers={"Authorization": "Bearer v6-local"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            status_code = int(getattr(response, "status", 0))
            body = response.read()
    except (OSError, urllib_error.URLError) as error:
        raise V6RawProbeInfrastructureError(
            f"live model endpoint is unavailable for role {role}: {url}"
        ) from error
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            f"live model endpoint returned malformed JSON for role {role}"
        ) from error
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    model_ids = sorted(
        {
            str(row["id"])
            for row in (rows or [])
            if isinstance(row, Mapping)
            and isinstance(row.get("id"), str)
            and row["id"]
        }
    )
    if status_code != 200 or model_ids != [expected_model]:
        raise V6GenerationError(
            f"live model identity mismatch for role {role}: "
            f"expected={expected_model!r}, observed={model_ids!r}"
        )
    # vLLM's ModelCard/ModelPermission response contains request-time
    # timestamps and generated permission IDs.  Hashing the raw payload would
    # therefore make a same-endpoint resume drift even though the served model
    # identity is unchanged.  Bind only the stable, experimentally relevant
    # projection; the immutable launch/model receipts already bind revision,
    # topology, quantization, container, and the long-context probe.
    stable_identity = {
        "status_code": status_code,
        "observed_model_ids": model_ids,
    }
    return {
        "role": role,
        "api_base": api_base,
        "models_url": url,
        "status": "PASS",
        "status_code": status_code,
        "expected_model": expected_model,
        "observed_model_ids": model_ids,
        "stable_identity_sha256": sha256(stable_identity),
        "raw_response_excluded_from_semantic_contract": True,
    }


def verify_live_model_endpoints(
    model_receipt_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify every role receipt against its live endpoint before generation."""

    roles = model_receipt_binding.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "teacher",
        "user",
        "judge",
    }:
        raise V6GenerationError("model receipt binding lacks exact live roles")
    checks: dict[str, dict[str, Any]] = {}
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for role in ("teacher", "user", "judge"):
        receipt = roles[role]
        if not isinstance(receipt, Mapping):
            raise V6GenerationError(f"model receipt role is malformed: {role}")
        api_base = receipt.get("api_base")
        model = receipt.get("model")
        if not isinstance(api_base, str) or not isinstance(model, str):
            raise V6GenerationError(
                f"model receipt endpoint identity is malformed: {role}"
            )
        key = (api_base, model)
        if key not in cached:
            cached[key] = verify_live_model_endpoint(
                role=role,
                api_base=api_base,
                expected_model=model,
            )
        checks[role] = {
            **deepcopy(cached[key]),
            "role": role,
        }
    return checks


V610_LIVE_PROCESS_EVIDENCE_KEYS = frozenset(
    {
        "server_pid",
        "process_boot_id",
        "process_start_time_ticks",
        "process_started_at_utc",
        "observed_cmdline_sha256",
        "observed_executable_path",
        "observed_executable_sha256",
        "model_snapshot_path",
        "snapshot_identity_files",
        "snapshot_identity_files_sha256",
        "observed_cuda_visible_devices",
        "observed_gpu_uuids",
        "listening_socket_inode",
        "selected_process_environment",
    }
)


def verify_live_model_processes(
    model_receipt_binding: Mapping[str, Any],
    source_container_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reinspect the exact PID/snapshot/GPU/socket proof behind each endpoint."""

    try:
        from scripts import build_v6_10_runtime_receipts as runtime_receipts
    except ModuleNotFoundError:  # pragma: no cover - script import path
        import build_v6_10_runtime_receipts as runtime_receipts

    roles = model_receipt_binding.get("roles")
    runtime = source_container_binding.get("runtime")
    inventory = source_container_binding.get("gpus")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != {"teacher", "user", "judge"}
        or not isinstance(runtime, Mapping)
        or not isinstance(runtime.get("python_executable_path"), str)
        or not isinstance(inventory, list)
    ):
        raise V6GenerationError(
            "live model-process reinspection lacks source/runtime binding"
        )
    runtime_python = Path(runtime["python_executable_path"])
    checks: dict[str, dict[str, Any]] = {}
    cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    for role in ("teacher", "user", "judge"):
        receipt = roles.get(role)
        if not isinstance(receipt, Mapping):
            raise V6GenerationError(
                f"live model-process role is malformed: {role}"
            )
        spec = {
            key: deepcopy(receipt[key])
            for key in (
                "model",
                "resolved_revision",
                "api_base",
                "quantization",
                "dtype",
                "tensor_parallel_size",
                "max_model_len",
                "gpu_memory_utilization",
                "gpu_uuids",
                "launch_command",
                "server_pid",
                "tokenizer_or_config_sha256",
            )
        }
        cache_key = (
            int(spec["server_pid"]),
            str(spec["api_base"]),
            str(spec["model"]),
        )
        try:
            if cache_key not in cache:
                cache[cache_key] = runtime_receipts.inspect_vllm_process(
                    spec,
                    inventory=inventory,
                    expected_runtime_python=runtime_python,
                )
        except Exception as error:
            raise V6GenerationError(
                f"live model-process reinspection failed for role {role}: "
                f"{error}"
            ) from error
        observed = cache[cache_key]
        expected = {
            key: deepcopy(receipt[key])
            for key in V610_LIVE_PROCESS_EVIDENCE_KEYS
        }
        if set(observed) != V610_LIVE_PROCESS_EVIDENCE_KEYS or observed != expected:
            raise V6GenerationError(
                f"live model process restarted or drifted for role {role}"
            )
        checks[role] = {
            "role": role,
            "status": "PASS",
            "process_evidence": deepcopy(observed),
            "process_evidence_sha256": sha256(observed),
        }
    return checks


def revalidate_v610_live_model_runtime(
    runtime_provenance: Mapping[str, Any],
) -> None:
    """Repeat PID and endpoint checks immediately before any generation."""

    model_binding = runtime_provenance.get("model_server_receipts")
    source_binding = runtime_provenance.get("source_container_provenance")
    if not isinstance(model_binding, Mapping) or not isinstance(
        source_binding, Mapping
    ):
        raise V6GenerationError(
            "V6.10 runtime provenance lacks live-model bindings"
        )
    observed_process = verify_live_model_processes(
        model_binding,
        source_binding,
    )
    if observed_process != model_binding.get("live_process_checks"):
        raise V6GenerationError(
            "live model process proof changed after startup validation"
        )
    observed_endpoints = verify_live_model_endpoints(model_binding)
    if observed_endpoints != model_binding.get("live_endpoint_checks"):
        raise V6GenerationError(
            "live model endpoint proof changed after startup validation"
        )


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or UTC_TIMESTAMP_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(
        parsed
    )


def build_execution_provenance(
    *,
    run_started_at_utc: str,
    exact_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not valid_utc_timestamp(run_started_at_utc):
        raise V6GenerationError("run start is not a canonical UTC timestamp")
    argv = [
        str(value)
        for value in (
            exact_argv
            if exact_argv is not None
            else getattr(sys, "orig_argv", [sys.executable, *sys.argv])
        )
    ]
    if not argv or any(not value for value in argv):
        raise V6GenerationError("exact process argv is empty or malformed")
    return {
        "exact_argv": argv,
        "exact_argv_sha256": sha256(argv),
        "python_executable": str(Path(sys.executable).resolve()),
        "run_started_at_utc": run_started_at_utc,
        "run_end_recording_policy": (
            "generation_receipt_records_final_utc_after_terminal_merge"
        ),
    }


def resolve_v610_run_started_at_utc(
    existing_run_contract: Mapping[str, Any] | None,
) -> str:
    """Create a run start once, then preserve it across same-attempt resumes."""

    if existing_run_contract is None:
        return utc_now()
    existing_execution = existing_run_contract.get("execution_provenance")
    run_started_at_utc = (
        existing_execution.get("run_started_at_utc")
        if isinstance(existing_execution, Mapping)
        else None
    )
    if not valid_utc_timestamp(run_started_at_utc):
        raise V6GenerationError(
            "existing V6.10 run contract lacks a valid immutable run start"
        )
    return str(run_started_at_utc)


def validate_v610_attempt_id(phase: str, attempt_id: str) -> None:
    """Validate both the frozen phase prefix and its real UTC calendar stamp."""

    pattern = V610_ATTEMPT_ID_RE_BY_PHASE.get(phase)
    if pattern is None or pattern.fullmatch(attempt_id) is None:
        raise V6GenerationError(
            "attempt id does not match the frozen phase-specific V6.10 format"
        )
    timestamp = attempt_id.rsplit("-", 1)[-1]
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise V6GenerationError(
            "attempt id contains an invalid UTC calendar timestamp"
        ) from error
    if parsed.strftime("%Y%m%dT%H%M%SZ") != timestamp:
        raise V6GenerationError(
            "attempt id contains a non-canonical UTC timestamp"
        )


def verify_committed_source_file(
    *,
    source_root: Path,
    source_commit: str,
    relative: str,
    expected_sha256: str,
    label: str,
) -> None:
    """Require a live release input to be the exact blob in the frozen commit."""

    root = source_root.resolve()
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise V6GenerationError(f"{label} path/hash is invalid: {relative!r}")
    live_path = (root / relative_path).resolve()
    try:
        live_path.relative_to(root)
    except ValueError as error:
        raise V6GenerationError(f"{label} escapes source root: {relative}") from error
    if not live_path.is_file() or sha256_file(live_path) != expected_sha256:
        raise V6GenerationError(f"{label} byte hash drift: {relative}")
    try:
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{source_commit}:{relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise V6GenerationError(
            f"{label} is not tracked by the frozen source commit: {relative}"
        ) from error
    if (
        tracked != [relative]
        or hashlib.sha256(committed).hexdigest() != expected_sha256
    ):
        raise V6GenerationError(
            f"{label} bytes differ from frozen source commit: {relative}"
        )


def load_release_manifest(
    path: Path,
    *,
    expected_sha256: str,
    source_root: Path,
    expected_fields: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact V6.10 immutable release/cross-artifact manifest."""

    identity_keys = {
        "source_commit",
        "source_tree",
        "container_image_digest",
        "tau2_commit",
        "config_sha256",
        "preregistration_sha256",
        "split_manifest_sha256",
        "reference_preflight_receipt_sha256",
        "registry_file_sha256",
        "source_container_provenance_sha256",
        "model_server_receipts_sha256",
        "runtime_receipt_hashes_sha256",
    }
    top_keys = {
        "protocol",
        "design_protocol",
        "status",
        *identity_keys,
        "relevant_scripts",
        "created_at_utc",
        "official_test_used",
    }
    if (
        SHA256_RE.fullmatch(expected_sha256) is None
        or set(expected_fields) != identity_keys
        or any(
            not isinstance(value, str) or not value
            for value in expected_fields.values()
        )
        or any(
            re.fullmatch(r"[0-9a-f]{40}", expected_fields[key]) is None
            for key in (
                "source_commit",
                "source_tree",
                "tau2_commit",
            )
        )
        or OCI_DIGEST_RE.fullmatch(
            expected_fields["container_image_digest"]
        )
        is None
        or any(
            SHA256_RE.fullmatch(expected_fields[key]) is None
            for key in identity_keys
            - {
                "source_commit",
                "source_tree",
                "container_image_digest",
                "tau2_commit",
            }
        )
    ):
        raise V6GenerationError(
            "release-manifest expected identity contract is invalid"
        )
    if not path.is_file():
        raise V6GenerationError(f"release manifest is absent: {path}")
    observed_file_sha256 = sha256_file(path)
    if observed_file_sha256 != expected_sha256:
        raise V6GenerationError(
            "release-manifest file hash drift: "
            f"{observed_file_sha256} != {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6GenerationError("release manifest is not valid JSON") from error
    scripts = (
        payload.get("relevant_scripts")
        if isinstance(payload, Mapping)
        else None
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != top_keys
        or payload.get("protocol") != V610_RELEASE_MANIFEST_PROTOCOL
        or payload.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        or payload.get("status") != "PASS"
        or any(payload.get(key) != expected_fields[key] for key in identity_keys)
        or not isinstance(scripts, dict)
        or set(scripts) != V610_REQUIRED_RELEASE_SCRIPTS
        or not valid_utc_timestamp(payload.get("created_at_utc"))
        or payload.get("official_test_used") is not False
    ):
        raise V6GenerationError(
            "release-manifest exact schema/identity drift"
        )
    resolved_root = source_root.resolve()
    observed_source_tree = _git_output(
        resolved_root,
        "rev-parse",
        f"{expected_fields['source_commit']}^{{tree}}",
    ).strip()
    if observed_source_tree != expected_fields["source_tree"]:
        raise V6GenerationError(
            "release-manifest source commit/tree binding drift"
        )
    verify_committed_source_file(
        source_root=resolved_root,
        source_commit=expected_fields["source_commit"],
        relative="configs/v6_10_closure.yaml",
        expected_sha256=expected_fields["config_sha256"],
        label="release-manifest frozen config",
    )
    verify_committed_source_file(
        source_root=resolved_root,
        source_commit=expected_fields["source_commit"],
        relative="V6_10_CLOSURE_PREREGISTRATION.md",
        expected_sha256=expected_fields["preregistration_sha256"],
        label="release-manifest frozen preregistration",
    )
    normalized_scripts: dict[str, str] = {}
    for relative, expected_script_sha256 in sorted(scripts.items()):
        if not isinstance(relative, str):
            raise V6GenerationError(
                f"release-manifest script path/hash is invalid: {relative!r}"
            )
        verify_committed_source_file(
            source_root=resolved_root,
            source_commit=expected_fields["source_commit"],
            relative=relative,
            expected_sha256=str(expected_script_sha256),
            label="release-manifest script",
        )
        normalized_scripts[relative] = str(expected_script_sha256)
    return payload, {
        "protocol": V610_RELEASE_MANIFEST_PROTOCOL,
        "file_sha256": observed_file_sha256,
        "semantic_sha256": sha256(payload),
        "created_at_utc": payload["created_at_utc"],
        "identities": {
            key: expected_fields[key] for key in sorted(identity_keys)
        },
        "relevant_scripts": normalized_scripts,
        "relevant_scripts_sha256": sha256(normalized_scripts),
        "official_test_used": False,
    }


def matched_positive_mode(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "matched_positive_continuation_mode", None)
        or getattr(args, "recovery_continuation_mode", "fresh_teacher")
    )


def causal_cell_mode(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "causal_cell_continuation_mode", None)
        or getattr(args, "recovery_continuation_mode", "fresh_teacher")
    )


def first_action_measurement_mode(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "first_action_measurement_mode", None) or "disabled"
    )


def load_reference_preflight(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fully verify the canonical 50-task reference preflight."""

    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise V6GenerationError(
            "expected reference-preflight receipt SHA-256 is invalid"
        )
    if not path.is_file():
        raise V6GenerationError(f"reference-preflight receipt is absent: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise V6GenerationError(
            "reference-preflight receipt hash drift: "
            f"{observed_sha256} != {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            "reference-preflight receipt is not valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise V6GenerationError("reference-preflight receipt is not an object")
    source_root = Path(__file__).resolve().parents[1]
    try:
        by_task = reference_contract.verify_preflight_receipt(
            payload,
            expected_task_ids=v610_protocol.FORMAL_TASK_IDS,
            expected_source_commit=expected_source_commit,
            expected_tau2_commit=protocol.TAU2_COMMIT,
            expected_preflight_script_sha256=sha256_file(
                source_root / "scripts" / "preflight_v6_reference_traces.py"
            ),
            expected_contract_module_sha256=sha256_file(
                source_root / "scripts" / "v6_reference_contract.py"
            ),
            expected_config_sha256=sha256_file(
                source_root / "configs" / "v6_10_closure.yaml"
            ),
            expected_split_manifest_sha256=sha256_file(
                source_root
                / "artifacts"
                / "v5_stage0"
                / "manifests"
                / "split_manifest.json"
            ),
            require_pass=True,
        )
    except (OSError, reference_contract.ReferenceContractError) as error:
        raise V6GenerationError(
            f"reference-preflight receipt failed canonical verification: {error}"
        ) from error
    plan_hashes = {
        task_identity: row["sanitized_successful_plan_sha256"]
        for task_identity, row in by_task.items()
    }
    task_preflight_hashes = {
        task_identity: row["task_preflight_sha256"]
        for task_identity, row in by_task.items()
    }
    binding = {
        "protocol": str(payload["protocol"]),
        "artifact_type": str(payload["artifact_type"]),
        "file_sha256": observed_sha256,
        "receipt_sha256": str(payload["receipt_sha256"]),
        "status": "PASS",
        "design_protocol": V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "tau2_commit": protocol.TAU2_COMMIT,
        "source_commit": expected_source_commit,
        "ordered_task_ids_sha256": payload["ordered_task_ids_sha256"],
        "sanitized_plan_hashes_sha256": sha256(plan_hashes),
        "task_preflight_hashes_sha256": sha256(task_preflight_hashes),
        "official_test_used": False,
    }
    return payload, binding


def verified_reference_plan_for_task(
    receipt: Mapping[str, Any],
    task_identity: str,
    *,
    preflight_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize the canonical accessor row for positive/cell construction."""

    try:
        row = reference_contract.task_contract_from_receipt(
            receipt,
            task_identity,
            require_task_pass=True,
        )
    except reference_contract.ReferenceContractError as error:
        raise V6GenerationError(
            f"{task_identity}: no verified PASS reference plan: {error}"
        ) from error
    slots = row.get("reference_slots")
    if not isinstance(slots, list):
        raise V6GenerationError(
            f"{task_identity}: verified reference contract lacks slots"
        )
    slots_by_index: dict[str, dict[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise V6GenerationError(
                f"{task_identity}: malformed verified reference slot"
            )
        index = slot.get("reference_action_index")
        slot_id = slot.get("reference_slot_id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(slot_id, str)
            or not slot_id
            or str(index) in slots_by_index
        ):
            raise V6GenerationError(
                f"{task_identity}: invalid/duplicate reference slot identity"
            )
        slots_by_index[str(index)] = {
            "reference_action_index": index,
            "reference_slot_id": slot_id,
            "reference_slot_sha256": slot.get("reference_slot_sha256"),
            "call_semantics_sha256": slot.get("call_semantics_sha256"),
        }
        if any(
            SHA256_RE.fullmatch(str(slots_by_index[str(index)][field])) is None
            for field in ("reference_slot_sha256", "call_semantics_sha256")
        ):
            raise V6GenerationError(
                f"{task_identity}: reference slot hashes are invalid"
            )
    file_sha256 = (
        preflight_binding.get("file_sha256")
        if isinstance(preflight_binding, Mapping)
        else None
    )
    if SHA256_RE.fullmatch(str(file_sha256)) is None:
        raise V6GenerationError(
            f"{task_identity}: reference-preflight file hash binding is absent"
        )
    evaluation_prefix = row.get("evaluation_prefix")
    evaluation_prefix_sha256 = row.get(
        "evaluation_prefix_semantic_sha256"
    )
    if (
        not isinstance(evaluation_prefix, list)
        or any(not isinstance(message, Mapping) for message in evaluation_prefix)
        or evaluation_prefix_sha256 != semantic_sha256(evaluation_prefix)
    ):
        raise V6GenerationError(
            f"{task_identity}: verified evaluation prefix is invalid"
        )
    return {
        "task_identity": task_identity,
        "reference_preflight_receipt_sha256": receipt["receipt_sha256"],
        "reference_preflight_file_sha256": file_sha256,
        "task_preflight_sha256": row["task_preflight_sha256"],
        "reference_slots_sha256": row["reference_slots_sha256"],
        "reference_slots_by_index": slots_by_index,
        "reference_slot_ids_by_index": {
            index: slot["reference_slot_id"]
            for index, slot in slots_by_index.items()
        },
        "expected_error_indices": deepcopy(row["expected_error_indices"]),
        "expected_error_set_sha256": row["expected_error_set_sha256"],
        "sanitized_successful_reference_indices": deepcopy(
            row["sanitized_successful_reference_indices"]
        ),
        "sanitized_successful_plan_sha256": row[
            "sanitized_successful_plan_sha256"
        ],
        "eligible_forced_first_reference_indices": deepcopy(
            row["eligible_forced_first_reference_indices"]
        ),
        "eligible_forced_first_reference_indices_sha256": row[
            "eligible_forced_first_reference_indices_sha256"
        ],
        "raw_reference_environment_reward": row[
            "raw_reference_environment_reward"
        ],
        "sanitized_environment_reward": row[
            "sanitized_environment_reward"
        ],
        "evaluation_prefix": deepcopy(evaluation_prefix),
        "evaluation_prefix_semantic_sha256": evaluation_prefix_sha256,
        "dynamic_full_official_reward_required": True,
        "official_test_used": False,
    }


def verify_v610_registry_preflight_binding(
    registry: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
) -> None:
    checks = {
        "reference_preflight_receipt_sha256": (
            registry.get("reference_preflight_receipt_sha256")
            == preflight_binding.get("receipt_sha256")
        ),
        "reference_preflight_file_sha256": (
            registry.get("reference_preflight_file_sha256")
            == preflight_binding.get("file_sha256")
        ),
        "sanitized_plan_hashes_sha256": (
            registry.get("sanitized_plan_hashes_sha256")
            == preflight_binding.get("sanitized_plan_hashes_sha256")
        ),
    }
    if not all(checks.values()):
        raise V6GenerationError(
            "V6.10 executable registry/preflight binding drift: "
            f"{canonical(checks)}"
        )


def verify_v610_task_file_hashes(
    registry: Mapping[str, Any],
    tau2_root: Path,
) -> dict[str, str]:
    """Bind the two pinned tau2 task files used by every selected task."""

    source = registry.get("source")
    expected = (
        source.get("task_file_sha256")
        if isinstance(source, Mapping)
        else None
    )
    if (
        not isinstance(expected, Mapping)
        or set(expected) != {"retail", "airline"}
        or any(
            SHA256_RE.fullmatch(str(value)) is None
            for value in expected.values()
        )
    ):
        raise V6GenerationError(
            "V6.10 registry task-file hash contract is invalid"
        )
    observed: dict[str, str] = {}
    for domain in ("airline", "retail"):
        path = (
            tau2_root.resolve()
            / "data"
            / "tau2"
            / "domains"
            / domain
            / "tasks.json"
        )
        if not path.is_file():
            raise V6GenerationError(
                f"V6.10 pinned task file is absent: {path}"
            )
        observed[domain] = sha256_file(path)
        if observed[domain] != expected[domain]:
            raise V6GenerationError(
                f"V6.10 {domain} task-file byte hash drift"
            )
    return observed


def v610_runtime_provenance(
    args: argparse.Namespace,
    *,
    design_protocol: str,
    registry: Mapping[str, Any] | None = None,
    registry_file_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify source, benchmark, executable, attempt, and preflight binding."""

    if design_protocol != V6_10_PIPELINE_CLOSURE_PROTOCOL:
        return {}, None
    required_strings = {
        "expected_source_commit": getattr(args, "expected_source_commit", None),
        "expected_generation_script_sha256": getattr(
            args, "expected_generation_script_sha256", None
        ),
        "expected_tau2_commit": getattr(args, "expected_tau2_commit", None),
        "reference_preflight_receipt_sha256": getattr(
            args, "reference_preflight_receipt_sha256", None
        ),
        "attempt_id": getattr(args, "attempt_id", None),
        "container_image_digest": getattr(
            args, "container_image_digest", None
        ),
        "model_server_receipts_sha256": getattr(
            args, "model_server_receipts_sha256", None
        ),
        "source_container_provenance_sha256": getattr(
            args, "source_container_provenance_sha256", None
        ),
        "release_manifest_sha256": getattr(
            args, "release_manifest_sha256", None
        ),
    }
    if any(not isinstance(value, str) or not value for value in required_strings.values()):
        raise V6GenerationError(
            "V6.10 requires explicit source/script/tau2/preflight/attempt binding"
        )
    if re.fullmatch(r"[0-9a-f]{40}", required_strings["expected_source_commit"]) is None:
        raise V6GenerationError("expected source commit must be a full commit")
    if SHA256_RE.fullmatch(
        required_strings["expected_generation_script_sha256"]
    ) is None:
        raise V6GenerationError("expected generation script SHA-256 is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", required_strings["expected_tau2_commit"]) is None:
        raise V6GenerationError("expected tau2 commit must be a full commit")
    if required_strings["expected_tau2_commit"] != protocol.TAU2_COMMIT:
        raise V6GenerationError("expected tau2 commit differs from frozen protocol")
    validate_v610_attempt_id(
        str(getattr(args, "phase", "")),
        required_strings["attempt_id"],
    )
    if (
        OCI_DIGEST_RE.fullmatch(
            required_strings["container_image_digest"]
        )
        is None
    ):
        raise V6GenerationError("container image digest is invalid")
    source_root = Path(__file__).resolve().parents[1]
    source = git_provenance(source_root)
    if (
        source["commit"] != required_strings["expected_source_commit"]
        or source["tracked_worktree_clean"] is not True
    ):
        raise V6GenerationError(
            "source commit/worktree proof failed closed: "
            f"{canonical(source)}"
        )
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    if script_sha256 != required_strings["expected_generation_script_sha256"]:
        raise V6GenerationError(
            "launched generation script hash differs from explicit expectation"
        )
    tau2 = git_provenance(Path(args.tau2_root))
    if (
        tau2["commit"] != required_strings["expected_tau2_commit"]
        or tau2["tracked_worktree_clean"] is not True
    ):
        raise V6GenerationError(
            "tau2 commit/worktree proof failed closed: "
            f"{canonical(tau2)}"
        )
    if (
        not isinstance(registry, Mapping)
        or SHA256_RE.fullmatch(str(registry_file_sha256)) is None
    ):
        raise V6GenerationError(
            "V6.10 runtime provenance requires the loaded registry/file hash"
        )
    task_file_sha256 = verify_v610_task_file_hashes(
        registry,
        Path(args.tau2_root),
    )
    receipt_path = getattr(args, "reference_preflight_receipt", None)
    if not isinstance(receipt_path, Path):
        raise V6GenerationError("V6.10 requires --reference-preflight-receipt")
    preflight, preflight_binding = load_reference_preflight(
        receipt_path.resolve(),
        expected_sha256=required_strings[
            "reference_preflight_receipt_sha256"
        ],
        expected_source_commit=required_strings["expected_source_commit"],
    )
    source_container_path = getattr(
        args, "source_container_provenance", None
    )
    if not isinstance(source_container_path, Path):
        raise V6GenerationError(
            "V6.10 requires --source-container-provenance"
        )
    _, source_container_binding = load_source_container_provenance(
        source_container_path.resolve(),
        expected_sha256=required_strings[
            "source_container_provenance_sha256"
        ],
        expected_source_commit=required_strings["expected_source_commit"],
        expected_generation_script_sha256=required_strings[
            "expected_generation_script_sha256"
        ],
        expected_tau2_commit=required_strings["expected_tau2_commit"],
        container_image_digest=required_strings["container_image_digest"],
    )
    model_receipt_path = getattr(args, "model_server_receipts", None)
    if not isinstance(model_receipt_path, Path):
        raise V6GenerationError("V6.10 requires --model-server-receipts")
    _, model_receipt_binding = load_model_server_receipts(
        model_receipt_path.resolve(),
        expected_sha256=required_strings[
            "model_server_receipts_sha256"
        ],
        expected_roles={
            "teacher": {
                "model": args.teacher_model,
                "revision": args.teacher_revision,
                "api_base": args.teacher_api_base,
                "quantization": V610_MODEL_QUANTIZATION,
                "tensor_parallel_size": (
                    V610_ROLE_TENSOR_PARALLEL_SIZE["teacher"]
                ),
            },
            "user": {
                "model": args.user_model,
                "revision": args.user_revision,
                "api_base": args.user_api_base,
                "quantization": V610_MODEL_QUANTIZATION,
                "tensor_parallel_size": (
                    V610_ROLE_TENSOR_PARALLEL_SIZE["user"]
                ),
            },
            "judge": {
                "model": args.judge_model or args.user_model,
                "revision": args.judge_revision or args.user_revision,
                "api_base": args.judge_api_base or args.user_api_base,
                "quantization": V610_MODEL_QUANTIZATION,
                "tensor_parallel_size": (
                    V610_ROLE_TENSOR_PARALLEL_SIZE["judge"]
                ),
            },
        },
        container_image_digest=required_strings["container_image_digest"],
        available_gpu_uuids=[
            str(row["uuid"]) for row in source_container_binding["gpus"]
        ],
        expected_python_executable_path=source_container_binding["runtime"][
            "python_executable_path"
        ],
        expected_python_executable_sha256=source_container_binding["runtime"][
            "python_executable_sha256"
        ],
    )
    config_path = source_root / "configs" / "v6_10_closure.yaml"
    preregistration_path = (
        source_root / "V6_10_CLOSURE_PREREGISTRATION.md"
    )
    split_manifest_path = (
        source_root
        / "artifacts"
        / "v5_stage0"
        / "manifests"
        / "split_manifest.json"
    )
    runtime_receipt_hashes_path = (
        source_container_path.resolve().parent
        / "runtime_receipt_hashes.json"
    )
    required_parent_paths = (
        config_path,
        preregistration_path,
        split_manifest_path,
        runtime_receipt_hashes_path,
    )
    if any(not path.is_file() for path in required_parent_paths):
        raise V6GenerationError(
            "V6.10 release parent artifact is absent"
        )
    parent_artifact_hashes = {
        "config_sha256": sha256_file(config_path),
        "preregistration_sha256": sha256_file(preregistration_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "reference_preflight_receipt_sha256": sha256_file(
            receipt_path.resolve()
        ),
        "registry_file_sha256": str(registry_file_sha256),
        "source_container_provenance_sha256": sha256_file(
            source_container_path.resolve()
        ),
        "model_server_receipts_sha256": sha256_file(
            model_receipt_path.resolve()
        ),
        "runtime_receipt_hashes_sha256": sha256_file(
            runtime_receipt_hashes_path
        ),
    }
    release_manifest_path = getattr(args, "release_manifest", None)
    if not isinstance(release_manifest_path, Path):
        raise V6GenerationError("V6.10 requires --release-manifest")
    _, release_manifest_binding = load_release_manifest(
        release_manifest_path.resolve(),
        expected_sha256=required_strings["release_manifest_sha256"],
        source_root=source_root,
        expected_fields={
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "container_image_digest": required_strings[
                "container_image_digest"
            ],
            "tau2_commit": tau2["commit"],
            "config_sha256": parent_artifact_hashes["config_sha256"],
            "preregistration_sha256": parent_artifact_hashes[
                "preregistration_sha256"
            ],
            "split_manifest_sha256": parent_artifact_hashes[
                "split_manifest_sha256"
            ],
            "reference_preflight_receipt_sha256": (
                parent_artifact_hashes[
                    "reference_preflight_receipt_sha256"
                ]
            ),
            "registry_file_sha256": parent_artifact_hashes[
                "registry_file_sha256"
            ],
            "source_container_provenance_sha256": (
                parent_artifact_hashes[
                    "source_container_provenance_sha256"
                ]
            ),
            "model_server_receipts_sha256": parent_artifact_hashes[
                "model_server_receipts_sha256"
            ],
            "runtime_receipt_hashes_sha256": parent_artifact_hashes[
                "runtime_receipt_hashes_sha256"
            ],
        },
    )
    parent_artifact_hashes["release_manifest_sha256"] = (
        release_manifest_binding["file_sha256"]
    )
    if set(parent_artifact_hashes) != V610_PARENT_ARTIFACT_HASH_KEYS:
        raise V6GenerationError(
            "V6.10 parent artifact hash contract is incomplete"
        )
    model_receipt_binding["live_process_checks"] = verify_live_model_processes(
        model_receipt_binding,
        source_container_binding,
    )
    model_receipt_binding["live_endpoint_checks"] = (
        verify_live_model_endpoints(model_receipt_binding)
    )
    provenance = {
        "source_commit": source["commit"],
        "source_tree_clean": True,
        "expected_source_commit": required_strings["expected_source_commit"],
        "observed_source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_tracked_worktree_clean": True,
        "source_tracked_worktree_status_sha256": source[
            "tracked_worktree_status_sha256"
        ],
        "source_worktree_scope": source["worktree_scope"],
        "generation_script_sha256": script_sha256,
        "tau2_commit": tau2["commit"],
        "tau2_tree_clean": True,
        "expected_tau2_commit": required_strings["expected_tau2_commit"],
        "observed_tau2_commit": tau2["commit"],
        "tau2_tree": tau2["tree"],
        "tau2_tracked_worktree_clean": True,
        "tau2_tracked_worktree_status_sha256": tau2[
            "tracked_worktree_status_sha256"
        ],
        "tau2_worktree_scope": tau2["worktree_scope"],
        "reference_preflight": preflight_binding,
        "reference_preflight_sha256": preflight_binding[
            "receipt_sha256"
        ],
        "container_image_digest": required_strings[
            "container_image_digest"
        ],
        "source_container_provenance": source_container_binding,
        "model_server_receipts": model_receipt_binding,
        "model_server_receipts_sha256": model_receipt_binding[
            "file_sha256"
        ],
        "release_manifest": release_manifest_binding,
        "release_manifest_sha256": release_manifest_binding[
            "file_sha256"
        ],
        "parent_artifact_hashes": parent_artifact_hashes,
        "task_file_sha256": task_file_sha256,
        "attempt_id": required_strings["attempt_id"],
    }
    return provenance, preflight


def validate_v610_modes(
    args: argparse.Namespace,
    *,
    design_protocol: str,
    continuation_seeds: Sequence[int] | None = None,
) -> None:
    if design_protocol != V6_10_PIPELINE_CLOSURE_PROTOCOL:
        return
    checks = {
        "phase": getattr(args, "phase", None)
        in {"compatibility", "pilot", "formal"},
        # V6.10 currently has only a single-directory merger.  Freeze the
        # release to one shard until a cross-shard completeness auditor exists.
        "single_shard_execution": (
            getattr(args, "num_shards", None) == 1
            and getattr(args, "shard_index", None) == 0
        ),
        "clean_agent_mode": (
            getattr(args, "clean_agent_mode", None)
            == "single_turn_user_reference_replay"
        ),
        "matched_positive_mode": (
            getattr(args, "matched_positive_continuation_mode", None)
            == "deterministic_reference_completion"
        ),
        "causal_cell_mode": (
            getattr(args, "causal_cell_continuation_mode", None)
            == "fresh_teacher"
        ),
        "first_action_measurement": (
            getattr(args, "first_action_measurement_mode", None)
            == "teacher_unforced"
        ),
        "completion_renderer": (
            getattr(args, "completion_renderer", None)
            == "explicit_user_direct_v3"
        ),
        "teacher_model": (
            getattr(args, "teacher_model", None) == V610_TEACHER_MODEL
        ),
        "teacher_revision": (
            getattr(args, "teacher_revision", None)
            == V610_TEACHER_REVISION
        ),
        "user_model": (
            getattr(args, "user_model", None) == V610_USER_JUDGE_MODEL
        ),
        "user_revision": (
            getattr(args, "user_revision", None)
            == V610_USER_JUDGE_REVISION
        ),
        "judge_model": (
            (getattr(args, "judge_model", None) or args.user_model)
            == V610_USER_JUDGE_MODEL
        ),
        "judge_revision": (
            (getattr(args, "judge_revision", None) or args.user_revision)
            == V610_USER_JUDGE_REVISION
        ),
        "shared_user_judge_endpoint": (
            (getattr(args, "judge_api_base", None) or args.user_api_base)
            == args.user_api_base
        ),
        "max_steps": getattr(args, "max_steps", None) == V610_MAX_STEPS,
        "max_tokens": getattr(args, "max_tokens", None) == V610_MAX_TOKENS,
        "continuation_seeds": (
            continuation_seeds is None
            or tuple(continuation_seeds)
            == (
                V610_COMPATIBILITY_SEEDS
                if getattr(args, "phase", None) == "compatibility"
                else V610_SCIENTIFIC_SEEDS
            )
        ),
    }
    if not all(checks.values()):
        raise V6GenerationError(
            "V6.10 mode separation failed closed: " + canonical(checks)
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(canonical(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def task_safe_name(task_identity: str) -> str:
    safe = task_identity.replace(":", "_")
    if (
        not safe
        or safe in {".", ".."}
        or "/" in safe
        or "\\" in safe
    ):
        raise V6GenerationError("task identity cannot form an artifact path")
    return safe


def task_evidence_tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise V6GenerationError(f"task evidence directory is absent: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise V6GenerationError(
                f"task evidence contains a forbidden symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise V6GenerationError(
                f"task evidence contains a non-regular file: {path}"
            )
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sha256(rows)


def v610_process_execution_id(
    *,
    attempt_id: str,
    process_started_at_utc: str,
    exact_argv_sha256: str,
    pid: int | None = None,
    monotonic_nonce_ns: int | None = None,
) -> str:
    if (
        not valid_utc_timestamp(process_started_at_utc)
        or SHA256_RE.fullmatch(exact_argv_sha256) is None
    ):
        raise V6GenerationError("process execution identity inputs are invalid")
    observed_pid = os.getpid() if pid is None else pid
    nonce = time.monotonic_ns() if monotonic_nonce_ns is None else monotonic_nonce_ns
    if (
        not isinstance(observed_pid, int)
        or isinstance(observed_pid, bool)
        or observed_pid <= 0
        or not isinstance(nonce, int)
        or isinstance(nonce, bool)
        or nonce < 0
    ):
        raise V6GenerationError("process execution PID/nonce is invalid")
    return sha256(
        {
            "attempt_id": attempt_id,
            "process_started_at_utc": process_started_at_utc,
            "exact_argv_sha256": exact_argv_sha256,
            "pid": observed_pid,
            "monotonic_nonce_ns": nonce,
        }
    )


def _v610_task_ledger_path(output_dir: Path, task_identity: str) -> Path:
    return (
        output_dir
        / "task_restart_ledgers"
        / f"{task_safe_name(task_identity)}.json"
    )


def _new_v610_task_ledger(
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_generation_contract_sha256: str,
) -> dict[str, Any]:
    value = {
        "protocol": V610_TASK_RESTART_LEDGER_PROTOCOL,
        "task_identity": task_identity,
        "run_contract_sha256": run_contract_sha256,
        "semantic_generation_contract_sha256": (
            semantic_generation_contract_sha256
        ),
        "max_process_restarts": V610_MAX_PROCESS_RESTARTS,
        "attempts": [],
        "official_test_used": False,
    }
    value["ledger_sha256"] = sha256(value)
    return value


def _validate_v610_task_ledger(
    value: Mapping[str, Any],
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_generation_contract_sha256: str,
) -> dict[str, Any]:
    top_keys = {
        "protocol",
        "task_identity",
        "run_contract_sha256",
        "semantic_generation_contract_sha256",
        "max_process_restarts",
        "attempts",
        "official_test_used",
        "ledger_sha256",
    }
    attempt_keys = {
        "attempt_number",
        "process_execution_id",
        "process_started_at_utc",
        "task_started_at_utc",
        "outcome",
        "task_ended_at_utc",
        "staging_relative_path",
        "evidence_relative_path",
        "evidence_tree_sha256",
        "terminal_receipt_sha256",
        "quarantined",
        "quarantined_at_utc",
        "quarantine_relative_path",
    }
    attempts = value.get("attempts") if isinstance(value, Mapping) else None
    unhashed = {
        key: item for key, item in value.items() if key != "ledger_sha256"
    }
    if (
        not isinstance(value, dict)
        or set(value) != top_keys
        or value.get("protocol") != V610_TASK_RESTART_LEDGER_PROTOCOL
        or value.get("task_identity") != task_identity
        or value.get("run_contract_sha256") != run_contract_sha256
        or value.get("semantic_generation_contract_sha256")
        != semantic_generation_contract_sha256
        or value.get("max_process_restarts") != V610_MAX_PROCESS_RESTARTS
        or value.get("official_test_used") is not False
        or value.get("ledger_sha256") != sha256(unhashed)
        or not isinstance(attempts, list)
    ):
        raise V6GenerationError(
            f"{task_identity}: task restart ledger is stale/invalid"
        )
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or set(attempt) != attempt_keys
            or attempt.get("attempt_number") != index
            or SHA256_RE.fullmatch(
                str(attempt.get("process_execution_id"))
            )
            is None
            or not valid_utc_timestamp(
                attempt.get("process_started_at_utc")
            )
            or not valid_utc_timestamp(attempt.get("task_started_at_utc"))
            or attempt.get("outcome")
            not in {
                "RUNNING",
                "PROMOTING",
                "INTERRUPTED",
                "PASS",
                "REJECTED",
            }
            or not isinstance(attempt.get("staging_relative_path"), str)
            or not attempt["staging_relative_path"]
            or not isinstance(attempt.get("quarantined"), bool)
        ):
            raise V6GenerationError(
                f"{task_identity}: task restart attempt ledger is malformed"
            )
        promotion_or_terminal = attempt["outcome"] in {
            "PROMOTING",
            "PASS",
            "REJECTED",
        }
        ended = attempt.get("task_ended_at_utc")
        if (
            (attempt["outcome"] == "RUNNING" and ended is not None)
            or (
                attempt["outcome"] != "RUNNING"
                and not valid_utc_timestamp(ended)
            )
            or (
                promotion_or_terminal
                and (
                    not isinstance(attempt.get("evidence_relative_path"), str)
                    or not attempt.get("evidence_relative_path")
                    or SHA256_RE.fullmatch(
                        str(attempt.get("evidence_tree_sha256"))
                    )
                    is None
                    or SHA256_RE.fullmatch(
                        str(attempt.get("terminal_receipt_sha256"))
                    )
                    is None
                )
            )
            or (
                not promotion_or_terminal
                and (
                    attempt.get("terminal_receipt_sha256") is not None
                    or attempt.get("evidence_tree_sha256") is not None
                )
            )
            or (
                attempt["quarantined"]
                and (
                    not valid_utc_timestamp(
                        attempt.get("quarantined_at_utc")
                    )
                    or not isinstance(
                        attempt.get("quarantine_relative_path"), str
                    )
                    or not attempt["quarantine_relative_path"]
                )
            )
            or (
                not attempt["quarantined"]
                and (
                    attempt.get("quarantined_at_utc") is not None
                    or attempt.get("quarantine_relative_path") is not None
                )
            )
        ):
            raise V6GenerationError(
                f"{task_identity}: task restart terminal ledger is malformed"
            )
    return deepcopy(dict(value))


def load_v610_task_ledger(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_generation_contract_sha256: str,
) -> dict[str, Any]:
    path = _v610_task_ledger_path(output_dir, task_identity)
    if not path.exists():
        return _new_v610_task_ledger(
            task_identity=task_identity,
            run_contract_sha256=run_contract_sha256,
            semantic_generation_contract_sha256=(
                semantic_generation_contract_sha256
            ),
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            f"{task_identity}: task restart ledger is not valid JSON"
        ) from error
    return _validate_v610_task_ledger(
        value,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_generation_contract_sha256=(
            semantic_generation_contract_sha256
        ),
    )


def write_v610_task_ledger(
    output_dir: Path,
    ledger: Mapping[str, Any],
) -> None:
    value = deepcopy(dict(ledger))
    value.pop("ledger_sha256", None)
    value["ledger_sha256"] = sha256(value)
    write_json(
        _v610_task_ledger_path(
            output_dir,
            str(value["task_identity"]),
        ),
        value,
    )


def _resolve_output_relative(output_dir: Path, relative: str) -> Path:
    root = output_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise V6GenerationError(
            f"task artifact path escapes output directory: {relative}"
        ) from error
    return path


def _journal_begin_v610_task_attempt_obsolete(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    process_execution_id: str,
    process_started_at_utc: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Quarantine prior non-PASS evidence and begin one bounded task attempt."""

    semantic_sha = sha256(semantic_contract)
    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        or SHA256_RE.fullmatch(process_execution_id) is None
        or not valid_utc_timestamp(process_started_at_utc)
    ):
        raise V6GenerationError(
            f"{task_identity}: cannot begin non-canonical V6.10 task attempt"
        )
    ledger = load_v610_task_ledger(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_generation_contract_sha256=semantic_sha,
    )
    attempts = ledger["attempts"]
    next_attempt_number = len(attempts) + 1
    if any(
        attempt.get("process_execution_id") == process_execution_id
        for attempt in attempts
    ):
        raise V6GenerationError(
            f"{task_identity}: same process cannot restart a task attempt"
        )
    quarantine_relative = (
        f"quarantine/{task_safe_name(task_identity)}/"
        f"before-attempt-{next_attempt_number:02d}"
    )
    quarantine_root = _resolve_output_relative(
        output_dir, quarantine_relative
    )
    moved_any = False

    if receipt_path.exists():
        try:
            prior_receipt = json.loads(
                receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise V6GenerationError(
                f"{task_identity}: prior task receipt is malformed"
            ) from error
        if (
            not isinstance(prior_receipt, dict)
            or prior_receipt.get("status") != "REJECTED"
        ):
            raise V6GenerationError(
                f"{task_identity}: only a validated REJECTED receipt may "
                "enter restart quarantine"
            )
        quarantine_root.mkdir(parents=True, exist_ok=False)
        os.replace(receipt_path, quarantine_root / "prior_receipt.json")
        moved_any = True

    for attempt in attempts:
        if attempt["quarantined"]:
            continue
        artifact_paths = []
        for key in ("staging_relative_path", "evidence_relative_path"):
            relative = attempt.get(key)
            if isinstance(relative, str) and relative:
                path = _resolve_output_relative(output_dir, relative)
                if path.exists():
                    artifact_paths.append((key, path))
        if (
            not artifact_paths
            and attempt["outcome"] in {"RUNNING", "PROMOTING"}
        ):
            raise V6GenerationError(
                f"{task_identity}: interrupted attempt evidence is missing"
            )
        if artifact_paths:
            if not moved_any:
                quarantine_root.mkdir(parents=True, exist_ok=False)
                moved_any = True
            for key, path in artifact_paths:
                target = quarantine_root / (
                    "staging" if key == "staging_relative_path" else "evidence"
                )
                if target.exists():
                    target = quarantine_root / f"{target.name}-{attempt['attempt_number']:02d}"
                os.replace(path, target)
        if attempt["outcome"] in {"RUNNING", "PROMOTING"}:
            attempt["outcome"] = "INTERRUPTED"
            attempt["task_ended_at_utc"] = utc_now()
            attempt["evidence_relative_path"] = None
            attempt["evidence_tree_sha256"] = None
            attempt["terminal_receipt_sha256"] = None
        attempt["quarantined"] = True
        attempt["quarantined_at_utc"] = utc_now()
        attempt["quarantine_relative_path"] = quarantine_relative

    if len(attempts) >= V610_MAX_PROCESS_RESTARTS + 1:
        write_v610_task_ledger(output_dir, ledger)
        raise V6GenerationError(
            f"{task_identity}: refusing a third process-level task restart"
        )

    staging_relative = (
        f"task_staging/{task_safe_name(task_identity)}/"
        f"attempt-{next_attempt_number:02d}"
    )
    staging = _resolve_output_relative(output_dir, staging_relative)
    if staging.exists():
        raise V6GenerationError(
            f"{task_identity}: next task staging directory already exists"
        )
    staging.mkdir(parents=True)
    task_started_at_utc = utc_now()
    attempt = {
        "attempt_number": next_attempt_number,
        "process_execution_id": process_execution_id,
        "process_started_at_utc": process_started_at_utc,
        "task_started_at_utc": task_started_at_utc,
        "outcome": "RUNNING",
        "task_ended_at_utc": None,
        "staging_relative_path": staging_relative,
        "evidence_relative_path": None,
        "evidence_tree_sha256": None,
        "terminal_receipt_sha256": None,
        "quarantined": False,
        "quarantined_at_utc": None,
        "quarantine_relative_path": None,
    }
    attempts.append(attempt)
    write_json(
        staging / "attempt.json",
        {
            "protocol": V610_TASK_RESTART_LEDGER_PROTOCOL,
            "task_identity": task_identity,
            "attempt_number": next_attempt_number,
            "process_execution_id": process_execution_id,
            "process_started_at_utc": process_started_at_utc,
            "task_started_at_utc": task_started_at_utc,
            "status": "RUNNING",
            "official_test_used": False,
        },
    )
    write_v610_task_ledger(output_dir, ledger)
    return {
        "attempt_number": next_attempt_number,
        "process_execution_id": process_execution_id,
        "process_started_at_utc": process_started_at_utc,
        "task_started_at_utc": task_started_at_utc,
        "staging_dir": staging,
        "staging_relative_path": staging_relative,
    }


def _journal_finalize_v610_task_attempt_obsolete(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    attempt_context: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically promote evidence, ledger state, then one terminal receipt."""

    status = receipt.get("status")
    if status not in {"PASS", "REJECTED"} or receipt_path.exists():
        raise V6GenerationError(
            f"{task_identity}: task attempt is not promotable terminal output"
        )
    semantic_sha = sha256(semantic_contract)
    ledger = load_v610_task_ledger(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_generation_contract_sha256=semantic_sha,
    )
    attempt_number = attempt_context.get("attempt_number")
    if (
        not isinstance(attempt_number, int)
        or attempt_number != len(ledger["attempts"])
    ):
        raise V6GenerationError(
            f"{task_identity}: terminal task attempt number drift"
        )
    attempt = ledger["attempts"][-1]
    staging = Path(str(attempt_context.get("staging_dir", ""))).resolve()
    expected_staging = _resolve_output_relative(
        output_dir, attempt["staging_relative_path"]
    )
    if (
        staging != expected_staging
        or not staging.is_dir()
        or attempt["outcome"] != "RUNNING"
        or attempt["process_execution_id"]
        != attempt_context.get("process_execution_id")
    ):
        raise V6GenerationError(
            f"{task_identity}: terminal task staging/ledger drift"
        )
    task_ended_at_utc = str(receipt.get("task_ended_at_utc", ""))
    if (
        receipt.get("task_started_at_utc")
        != attempt["task_started_at_utc"]
        or not valid_utc_timestamp(task_ended_at_utc)
        or _utc_datetime(task_ended_at_utc)
        < _utc_datetime(attempt["task_started_at_utc"])
    ):
        raise V6GenerationError(
            f"{task_identity}: terminal receipt task timing drift"
        )
    write_json(
        staging / "attempt.json",
        {
            "protocol": V610_TASK_RESTART_LEDGER_PROTOCOL,
            "task_identity": task_identity,
            "attempt_number": attempt_number,
            "process_execution_id": attempt["process_execution_id"],
            "process_started_at_utc": attempt["process_started_at_utc"],
            "task_started_at_utc": attempt["task_started_at_utc"],
            "task_ended_at_utc": task_ended_at_utc,
            "status": status,
            "official_test_used": False,
        },
    )
    evidence_relative = (
        f"task_evidence/{task_safe_name(task_identity)}/"
        f"attempt-{attempt_number:02d}"
    )
    evidence = _resolve_output_relative(output_dir, evidence_relative)
    if evidence.exists():
        raise V6GenerationError(
            f"{task_identity}: terminal task evidence already exists"
        )
    evidence_sha = task_evidence_tree_sha256(staging)
    task_execution = {
        "protocol": V610_TASK_RESTART_LEDGER_PROTOCOL,
        "attempt_number": attempt_number,
        "restart_count": attempt_number - 1,
        "max_process_restarts": V610_MAX_PROCESS_RESTARTS,
        "process_execution_id": attempt["process_execution_id"],
        "process_started_at_utc": attempt["process_started_at_utc"],
        "evidence_relative_path": evidence_relative,
        "evidence_tree_sha256": evidence_sha,
    }
    terminal_receipt = deepcopy(dict(receipt))
    terminal_receipt["task_execution"] = task_execution
    terminal_receipt.pop("task_receipt_sha256", None)
    terminal_receipt["task_receipt_sha256"] = sha256(terminal_receipt)

    # Journal the complete intended publication before moving the evidence.
    # A process death after this write is recovered by begin_v610_task_attempt:
    # it quarantines whichever of staging/evidence exists and marks this
    # PROMOTING attempt INTERRUPTED.  Thus no rename gap can lose evidence.
    attempt.update(
        {
            "outcome": "PROMOTING",
            "task_ended_at_utc": task_ended_at_utc,
            "evidence_relative_path": evidence_relative,
            "evidence_tree_sha256": evidence_sha,
            "terminal_receipt_sha256": terminal_receipt[
                "task_receipt_sha256"
            ],
        }
    )
    write_v610_task_ledger(output_dir, ledger)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, evidence)
    attempt["outcome"] = status
    write_v610_task_ledger(output_dir, ledger)
    write_json(receipt_path, terminal_receipt)
    return terminal_receipt


def _journal_v610_task_execution_binding_matches_obsolete(
    receipt: Mapping[str, Any],
    *,
    output_dir: Path | None,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> bool:
    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        return True
    if output_dir is None:
        return False
    execution = receipt.get("task_execution")
    keys = {
        "protocol",
        "attempt_number",
        "restart_count",
        "max_process_restarts",
        "process_execution_id",
        "process_started_at_utc",
        "evidence_relative_path",
        "evidence_tree_sha256",
    }
    if (
        not isinstance(execution, Mapping)
        or set(execution) != keys
        or execution.get("protocol") != V610_TASK_RESTART_LEDGER_PROTOCOL
        or not isinstance(execution.get("attempt_number"), int)
        or isinstance(execution.get("attempt_number"), bool)
        or execution["attempt_number"] <= 0
        or execution.get("restart_count")
        != execution["attempt_number"] - 1
        or execution.get("max_process_restarts") != V610_MAX_PROCESS_RESTARTS
        or SHA256_RE.fullmatch(
            str(execution.get("process_execution_id"))
        )
        is None
        or not valid_utc_timestamp(execution.get("process_started_at_utc"))
        or not isinstance(execution.get("evidence_relative_path"), str)
        or SHA256_RE.fullmatch(
            str(execution.get("evidence_tree_sha256"))
        )
        is None
    ):
        return False
    try:
        evidence = _resolve_output_relative(
            output_dir,
            execution["evidence_relative_path"],
        )
        if (
            task_evidence_tree_sha256(evidence)
            != execution["evidence_tree_sha256"]
        ):
            return False
        ledger = load_v610_task_ledger(
            output_dir,
            task_identity=task_identity,
            run_contract_sha256=run_contract_sha256,
            semantic_generation_contract_sha256=sha256(semantic_contract),
        )
    except V6GenerationError:
        return False
    attempt_number = execution["attempt_number"]
    if attempt_number > len(ledger["attempts"]):
        return False
    attempt = ledger["attempts"][attempt_number - 1]
    return (
        attempt.get("outcome") == receipt.get("status")
        and attempt.get("process_execution_id")
        == execution["process_execution_id"]
        and attempt.get("evidence_relative_path")
        == execution["evidence_relative_path"]
        and attempt.get("evidence_tree_sha256")
        == execution["evidence_tree_sha256"]
        and attempt.get("terminal_receipt_sha256")
        == receipt.get("task_receipt_sha256")
        and attempt.get("task_started_at_utc")
        == receipt.get("task_started_at_utc")
        and attempt.get("task_ended_at_utc")
        == receipt.get("task_ended_at_utc")
        and attempt.get("quarantined") is False
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise V6GenerationError(
                f"atomic task bundle contains a forbidden symlink: {path}"
            )
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise V6GenerationError(
                f"atomic task bundle contains a special node: {path}"
            )
    for path in files:
        _fsync_file(path)
    for path in sorted(
        directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _require_real_directory_in_output(
    output_dir: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    root = output_dir.resolve()
    try:
        relative = path.absolute().relative_to(output_dir.absolute())
    except ValueError as error:
        raise V6GenerationError(f"{label} escapes output directory") from error
    cursor = output_dir.absolute()
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise V6GenerationError(f"{label} traverses a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise V6GenerationError(f"{label} resolves outside output") from error
    if not resolved.is_dir():
        raise V6GenerationError(f"{label} is not a directory")
    return resolved


def acquire_v610_run_lock(
    output_dir: Path,
    *,
    process_execution_id: str,
    process_started_at_utc: str,
) -> Any:
    """Hold one non-blocking Linux flock for the entire V6.10 process."""

    if fcntl is None:
        raise V6GenerationError(
            "V6.10 requires Linux fcntl.flock single-writer locking"
        )
    if (
        SHA256_RE.fullmatch(process_execution_id) is None
        or not valid_utc_timestamp(process_started_at_utc)
    ):
        raise V6GenerationError("V6.10 run-lock owner identity is invalid")
    lock_path = output_dir / ".v610-run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        raise V6GenerationError(
            "V6.10 output directory already has an active writer"
            + (f": {owner}" if owner else "")
        ) from error
    owner = {
        "protocol": V610_TASK_BUNDLE_PROTOCOL,
        "pid": os.getpid(),
        "process_execution_id": process_execution_id,
        "process_started_at_utc": process_started_at_utc,
    }
    handle.seek(0)
    handle.truncate()
    handle.write(canonical(owner) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    _V610_ACTIVE_RUN_LOCKS.append(handle)
    return handle


def _release_v610_run_locks() -> None:
    while _V610_ACTIVE_RUN_LOCKS:
        handle = _V610_ACTIVE_RUN_LOCKS.pop()
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


atexit.register(_release_v610_run_locks)


def _v610_bundle_marker_path(bundle: Path) -> Path:
    return bundle / "attempt.json"


def _read_v610_bundle_marker(
    bundle: Path,
    *,
    task_identity: str,
    attempt_number: int,
) -> dict[str, Any]:
    path = _v610_bundle_marker_path(bundle)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            f"{task_identity}: task bundle marker is absent/malformed: {bundle}"
        ) from error
    keys = {
        "protocol",
        "task_identity",
        "attempt_number",
        "process_execution_id",
        "process_started_at_utc",
        "task_started_at_utc",
        "task_ended_at_utc",
        "status",
        "bundle_identity_sha256",
        "evidence_tree_sha256",
        "terminal_receipt_sha256",
        "official_test_used",
    }
    status = value.get("status") if isinstance(value, Mapping) else None
    terminal = status in {"PASS", "REJECTED"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("protocol") != V610_TASK_BUNDLE_PROTOCOL
        or value.get("task_identity") != task_identity
        or value.get("attempt_number") != attempt_number
        or attempt_number > V610_MAX_PROCESS_RESTARTS + 1
        or SHA256_RE.fullmatch(
            str(value.get("process_execution_id"))
        )
        is None
        or not valid_utc_timestamp(value.get("process_started_at_utc"))
        or not valid_utc_timestamp(value.get("task_started_at_utc"))
        or status not in {"RUNNING", "INTERRUPTED", "PASS", "REJECTED"}
        or value.get("official_test_used") is not False
        or (
            status == "RUNNING"
            and any(
                value.get(key) is not None
                for key in (
                    "task_ended_at_utc",
                    "bundle_identity_sha256",
                    "evidence_tree_sha256",
                    "terminal_receipt_sha256",
                )
            )
        )
        or (
            status == "INTERRUPTED"
            and (
                not valid_utc_timestamp(value.get("task_ended_at_utc"))
                or any(
                    value.get(key) is not None
                    for key in (
                        "bundle_identity_sha256",
                        "evidence_tree_sha256",
                        "terminal_receipt_sha256",
                    )
                )
            )
        )
        or (
            terminal
            and (
                not valid_utc_timestamp(value.get("task_ended_at_utc"))
                or SHA256_RE.fullmatch(
                    str(value.get("bundle_identity_sha256"))
                )
                is None
                or SHA256_RE.fullmatch(
                    str(value.get("evidence_tree_sha256"))
                )
                is None
                or SHA256_RE.fullmatch(
                    str(value.get("terminal_receipt_sha256"))
                )
                is None
            )
        )
    ):
        raise V6GenerationError(
            f"{task_identity}: task bundle marker contract is invalid"
        )
    return value


def _v610_task_artifacts(
    output_dir: Path,
    task_identity: str,
) -> list[dict[str, Any]]:
    output_root = output_dir.resolve()
    safe = task_safe_name(task_identity)
    roots = (
        ("STAGING", output_root / "task_staging" / safe),
        ("COMPLETED", output_root / "task_bundles" / safe),
        ("QUARANTINED", output_root / "quarantine" / safe),
    )
    rows: list[dict[str, Any]] = []
    for location, root in roots:
        if not root.exists():
            continue
        root = _require_real_directory_in_output(
            output_root,
            root,
            label=f"{task_identity}: task artifact root",
        )
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.is_symlink():
                raise V6GenerationError(
                    f"{task_identity}: unexpected task artifact: {path}"
                )
            matched = V610_ATTEMPT_DIR_RE.fullmatch(path.name)
            if matched is None:
                # Creation scratch directories are never publication points.
                # They are preserved, but cannot masquerade as an attempt.
                if path.name.startswith(".creating-attempt-"):
                    continue
                raise V6GenerationError(
                    f"{task_identity}: malformed task artifact name: {path.name}"
                )
            attempt_number = int(matched.group("number"))
            if attempt_number > V610_MAX_PROCESS_RESTARTS + 1:
                raise V6GenerationError(
                    f"{task_identity}: task attempt exceeds restart budget"
                )
            marker = _read_v610_bundle_marker(
                path,
                task_identity=task_identity,
                attempt_number=attempt_number,
            )
            rows.append(
                {
                    "location": location,
                    "path": path,
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "attempt_number": attempt_number,
                    "marker": marker,
                }
            )
    active_by_attempt: dict[int, int] = {}
    for row in rows:
        if row["location"] != "QUARANTINED":
            number = int(row["attempt_number"])
            active_by_attempt[number] = active_by_attempt.get(number, 0) + 1
    if any(count != 1 for count in active_by_attempt.values()):
        raise V6GenerationError(
            f"{task_identity}: duplicate active task-attempt artifacts"
        )
    return rows


def _write_v610_derived_restart_ledger(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _v610_task_artifacts(output_dir, task_identity)
    rows = [
        {
            "attempt_number": row["attempt_number"],
            "location": row["location"],
            "relative_path": row["relative_path"],
            "status": row["marker"]["status"],
            "process_execution_id": row["marker"]["process_execution_id"],
            "process_started_at_utc": row["marker"][
                "process_started_at_utc"
            ],
            "task_started_at_utc": row["marker"]["task_started_at_utc"],
            "task_ended_at_utc": row["marker"]["task_ended_at_utc"],
            "bundle_identity_sha256": row["marker"][
                "bundle_identity_sha256"
            ],
            "terminal_receipt_sha256": row["marker"][
                "terminal_receipt_sha256"
            ],
        }
        for row in artifacts
    ]
    ledger = {
        "protocol": V610_TASK_RESTART_LEDGER_PROTOCOL,
        "authority": "DERIVED_INDEX_ONLY",
        "authoritative_publication": "ATOMIC_TASK_BUNDLE_RENAME",
        "task_identity": task_identity,
        "run_contract_sha256": run_contract_sha256,
        "semantic_generation_contract_sha256": sha256(semantic_contract),
        "max_process_restarts": V610_MAX_PROCESS_RESTARTS,
        "artifacts": rows,
        "official_test_used": False,
    }
    ledger["ledger_sha256"] = sha256(ledger)
    write_json(_v610_task_ledger_path(output_dir, task_identity), ledger)
    return ledger


def _load_completed_v610_bundle_receipt(
    bundle: Path,
    *,
    task_identity: str,
    attempt_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bundle.is_dir() or bundle.is_symlink():
        raise V6GenerationError(
            f"{task_identity}: completed bundle is not a real directory"
        )
    top_level = {path.name: path for path in bundle.iterdir()}
    if set(top_level) != {
        "attempt.json",
        "evidence",
        "terminal_receipt.json",
        "bundle_manifest.json",
    }:
        raise V6GenerationError(
            f"{task_identity}: completed bundle top-level closure drift"
        )
    for name in (
        "attempt.json",
        "terminal_receipt.json",
        "bundle_manifest.json",
    ):
        path = top_level[name]
        if not path.is_file() or path.is_symlink():
            raise V6GenerationError(
                f"{task_identity}: completed bundle metadata is not regular"
            )
    if (
        not top_level["evidence"].is_dir()
        or top_level["evidence"].is_symlink()
    ):
        raise V6GenerationError(
            f"{task_identity}: completed bundle evidence is not a directory"
        )
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise V6GenerationError(
                f"{task_identity}: completed bundle contains a symlink"
            )
        if not path.is_file() and not path.is_dir():
            raise V6GenerationError(
                f"{task_identity}: completed bundle contains a special node"
            )
    marker = _read_v610_bundle_marker(
        bundle,
        task_identity=task_identity,
        attempt_number=attempt_number,
    )
    if marker["status"] not in {"PASS", "REJECTED"}:
        raise V6GenerationError(
            f"{task_identity}: completed bundle is not terminal"
        )
    try:
        receipt = json.loads(
            (bundle / "terminal_receipt.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (bundle / "bundle_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6GenerationError(
            f"{task_identity}: completed bundle files are malformed"
        ) from error
    manifest_keys = {
        "protocol",
        "task_identity",
        "attempt_number",
        "status",
        "bundle_identity_sha256",
        "evidence_tree_sha256",
        "terminal_receipt_sha256",
        "official_test_used",
        "bundle_manifest_sha256",
    }
    unhashed_manifest = (
        {
            key: value
            for key, value in manifest.items()
            if key != "bundle_manifest_sha256"
        }
        if isinstance(manifest, Mapping)
        else {}
    )
    if (
        not isinstance(receipt, dict)
        or not isinstance(manifest, dict)
        or set(manifest) != manifest_keys
        or manifest.get("protocol") != V610_TASK_BUNDLE_PROTOCOL
        or manifest.get("task_identity") != task_identity
        or manifest.get("attempt_number") != attempt_number
        or manifest.get("status") != marker["status"]
        or manifest.get("bundle_identity_sha256")
        != marker["bundle_identity_sha256"]
        or manifest.get("evidence_tree_sha256")
        != marker["evidence_tree_sha256"]
        or manifest.get("terminal_receipt_sha256")
        != marker["terminal_receipt_sha256"]
        or manifest.get("official_test_used") is not False
        or manifest.get("bundle_manifest_sha256")
        != sha256(unhashed_manifest)
        or receipt.get("task_receipt_sha256")
        != marker["terminal_receipt_sha256"]
        or receipt.get("task_receipt_sha256")
        != sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "task_receipt_sha256"
            }
        )
        or receipt.get("status") != marker["status"]
    ):
        raise V6GenerationError(
            f"{task_identity}: completed bundle manifest/receipt drift"
        )
    evidence = bundle / "evidence"
    evidence_sha = task_evidence_tree_sha256(evidence)
    recomputed_identity = sha256(
        {
            "protocol": V610_TASK_BUNDLE_PROTOCOL,
            "task_identity": task_identity,
            "attempt_number": attempt_number,
            "process_execution_id": marker["process_execution_id"],
            "process_started_at_utc": marker["process_started_at_utc"],
            "task_started_at_utc": marker["task_started_at_utc"],
            "task_ended_at_utc": marker["task_ended_at_utc"],
            "status": marker["status"],
            "evidence_tree_sha256": evidence_sha,
        }
    )
    matched = V610_ATTEMPT_DIR_RE.fullmatch(bundle.name)
    if (
        marker["evidence_tree_sha256"] != evidence_sha
        or marker["bundle_identity_sha256"] != recomputed_identity
        or matched is None
        or int(matched.group("number")) != attempt_number
        or matched.group("suffix") != recomputed_identity
    ):
        raise V6GenerationError(
            f"{task_identity}: completed bundle identity/evidence drift"
        )
    return receipt, marker


def _quarantine_v610_creation_scratch(
    output_dir: Path,
    *,
    task_identity: str,
) -> None:
    """Preserve unpublished creation scratch without treating it as a run."""

    staging_root = (
        output_dir / "task_staging" / task_safe_name(task_identity)
    )
    if not staging_root.exists():
        return
    staging_root = _require_real_directory_in_output(
        output_dir.resolve(),
        staging_root,
        label=f"{task_identity}: creation-scratch root",
    )
    for source in sorted(staging_root.glob(".creating-attempt-*")):
        if not source.is_dir() or source.is_symlink():
            raise V6GenerationError(
                f"{task_identity}: malformed task creation scratch"
            )
        target = (
            output_dir
            / "quarantine_scratch"
            / task_safe_name(task_identity)
            / f"{source.name}-{sha256(source.name)[:16]}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise V6GenerationError(
                f"{task_identity}: creation-scratch quarantine collision"
            )
        source_parent = source.parent
        os.replace(source, target)
        _fsync_directory(source_parent)
        _fsync_directory(target.parent)


def reconcile_v610_task_index(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any] | None:
    """Rebuild mutable receipt/ledger indexes from atomic task bundles."""

    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    _quarantine_v610_creation_scratch(
        output_dir,
        task_identity=task_identity,
    )
    artifacts = _v610_task_artifacts(output_dir, task_identity)
    completed = [
        row for row in artifacts if row["location"] == "COMPLETED"
    ]
    if len(completed) > 1:
        raise V6GenerationError(
            f"{task_identity}: multiple authoritative completed bundles"
        )
    receipt: dict[str, Any] | None = None
    if completed:
        row = completed[0]
        receipt, _ = _load_completed_v610_bundle_receipt(
            row["path"],
            task_identity=task_identity,
            attempt_number=row["attempt_number"],
        )
        if receipt_path.exists():
            try:
                indexed = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                indexed = None
            if indexed != receipt:
                write_json(receipt_path, receipt)
        else:
            write_json(receipt_path, receipt)
    elif receipt_path.exists():
        # No derived index may survive without an authoritative bundle.
        archive = (
            output_dir
            / "quarantine_indexes"
            / task_safe_name(task_identity)
            / f"orphan-receipt-{sha256_file(receipt_path)[:16]}.json"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise V6GenerationError(
                f"{task_identity}: orphan receipt quarantine collision"
            )
        os.replace(receipt_path, archive)
    _write_v610_derived_restart_ledger(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=semantic_contract,
    )
    return receipt


def _quarantine_v610_artifact(
    output_dir: Path,
    *,
    task_identity: str,
    row: Mapping[str, Any],
) -> None:
    output_dir = output_dir.resolve()
    source = Path(row["path"])
    marker = deepcopy(dict(row["marker"]))
    if marker["status"] == "RUNNING":
        marker["status"] = "INTERRUPTED"
        marker["task_ended_at_utc"] = utc_now()
        write_json(_v610_bundle_marker_path(source), marker)
        _fsync_tree(source)
    target = (
        output_dir
        / "quarantine"
        / task_safe_name(task_identity)
        / (
            f"attempt-{int(row['attempt_number']):02d}-"
            f"{str(row['location']).lower()}-"
            f"{sha256(row['relative_path'])[:16]}"
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise V6GenerationError(
            f"{task_identity}: task quarantine target already exists"
        )
    source_parent = source.parent
    os.replace(source, target)
    _fsync_directory(source_parent)
    _fsync_directory(target.parent)


def begin_v610_task_attempt(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    process_execution_id: str,
    process_started_at_utc: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Quarantine non-PASS work and atomically publish a fresh staging bundle."""

    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        or SHA256_RE.fullmatch(process_execution_id) is None
        or not valid_utc_timestamp(process_started_at_utc)
    ):
        raise V6GenerationError(
            f"{task_identity}: cannot begin non-canonical V6.10 task attempt"
        )
    reconcile_v610_task_index(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=semantic_contract,
        receipt_path=receipt_path,
    )
    artifacts = _v610_task_artifacts(output_dir, task_identity)
    completed_pass = [
        row
        for row in artifacts
        if row["location"] == "COMPLETED"
        and row["marker"]["status"] == "PASS"
    ]
    if completed_pass:
        raise V6GenerationError(
            f"{task_identity}: atomic PASS bundle must be resumed, not rerun"
        )
    prior_process_ids = {
        str(row["marker"]["process_execution_id"]) for row in artifacts
    }
    if process_execution_id in prior_process_ids:
        raise V6GenerationError(
            f"{task_identity}: same process cannot restart a task attempt"
        )
    for row in artifacts:
        if row["location"] == "STAGING" or (
            row["location"] == "COMPLETED"
            and row["marker"]["status"] == "REJECTED"
        ):
            _quarantine_v610_artifact(
                output_dir,
                task_identity=task_identity,
                row=row,
            )
    if receipt_path.exists():
        archive = (
            output_dir
            / "quarantine_indexes"
            / task_safe_name(task_identity)
            / (
                f"prior-rejected-receipt-"
                f"{sha256_file(receipt_path)[:16]}.json"
            )
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise V6GenerationError(
                f"{task_identity}: rejected receipt quarantine collision"
            )
        os.replace(receipt_path, archive)
    artifacts = _v610_task_artifacts(output_dir, task_identity)
    attempt_numbers = {
        int(row["attempt_number"]) for row in artifacts
    }
    if attempt_numbers and attempt_numbers != set(
        range(1, max(attempt_numbers) + 1)
    ):
        raise V6GenerationError(
            f"{task_identity}: non-contiguous task attempt history"
        )
    completed_attempts = max(attempt_numbers, default=0)
    if completed_attempts >= V610_MAX_PROCESS_RESTARTS + 1:
        _write_v610_derived_restart_ledger(
            output_dir,
            task_identity=task_identity,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=semantic_contract,
        )
        raise V6GenerationError(
            f"{task_identity}: refusing a third process-level task restart"
        )
    attempt_number = completed_attempts + 1
    task_started_at_utc = utc_now()
    marker = {
        "protocol": V610_TASK_BUNDLE_PROTOCOL,
        "task_identity": task_identity,
        "attempt_number": attempt_number,
        "process_execution_id": process_execution_id,
        "process_started_at_utc": process_started_at_utc,
        "task_started_at_utc": task_started_at_utc,
        "task_ended_at_utc": None,
        "status": "RUNNING",
        "bundle_identity_sha256": None,
        "evidence_tree_sha256": None,
        "terminal_receipt_sha256": None,
        "official_test_used": False,
    }
    staging_parent = (
        output_dir / "task_staging" / task_safe_name(task_identity)
    )
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f"attempt-{attempt_number:02d}"
    scratch = staging_parent / (
        f".creating-attempt-{attempt_number:02d}-"
        f"{process_execution_id[:16]}"
    )
    if staging.exists() or scratch.exists():
        raise V6GenerationError(
            f"{task_identity}: next task staging path already exists"
        )
    scratch.mkdir()
    (scratch / "evidence").mkdir()
    write_json(_v610_bundle_marker_path(scratch), marker)
    _fsync_tree(scratch)
    os.replace(scratch, staging)
    _fsync_directory(staging_parent)
    _write_v610_derived_restart_ledger(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=semantic_contract,
    )
    return {
        "attempt_number": attempt_number,
        "process_execution_id": process_execution_id,
        "process_started_at_utc": process_started_at_utc,
        "task_started_at_utc": task_started_at_utc,
        "staging_dir": staging,
        "evidence_dir": staging / "evidence",
    }


def finalize_v610_task_attempt(
    output_dir: Path,
    *,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    attempt_context: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a task by one atomic directory rename; indexes are derived."""

    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    status = receipt.get("status")
    attempt_number = attempt_context.get("attempt_number")
    if (
        status not in {"PASS", "REJECTED"}
        or not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number <= 0
    ):
        raise V6GenerationError(
            f"{task_identity}: task attempt is not promotable terminal output"
        )
    staging = Path(str(attempt_context.get("staging_dir", ""))).resolve()
    expected_staging = (
        output_dir
        / "task_staging"
        / task_safe_name(task_identity)
        / f"attempt-{attempt_number:02d}"
    ).resolve()
    if staging != expected_staging or not staging.is_dir():
        raise V6GenerationError(
            f"{task_identity}: terminal staging bundle drift"
        )
    marker = _read_v610_bundle_marker(
        staging,
        task_identity=task_identity,
        attempt_number=attempt_number,
    )
    if (
        marker["status"] != "RUNNING"
        or marker["process_execution_id"]
        != attempt_context.get("process_execution_id")
        or receipt.get("task_started_at_utc")
        != marker["task_started_at_utc"]
        or not valid_utc_timestamp(receipt.get("task_ended_at_utc"))
        or _utc_datetime(receipt["task_ended_at_utc"])
        < _utc_datetime(marker["task_started_at_utc"])
    ):
        raise V6GenerationError(
            f"{task_identity}: terminal task timing/process drift"
        )
    evidence = staging / "evidence"
    evidence_sha = task_evidence_tree_sha256(evidence)
    bundle_identity = sha256(
        {
            "protocol": V610_TASK_BUNDLE_PROTOCOL,
            "task_identity": task_identity,
            "attempt_number": attempt_number,
            "process_execution_id": marker["process_execution_id"],
            "process_started_at_utc": marker["process_started_at_utc"],
            "task_started_at_utc": marker["task_started_at_utc"],
            "task_ended_at_utc": receipt["task_ended_at_utc"],
            "status": status,
            "evidence_tree_sha256": evidence_sha,
        }
    )
    bundle_relative = (
        f"task_bundles/{task_safe_name(task_identity)}/"
        f"attempt-{attempt_number:02d}-{bundle_identity}"
    )
    evidence_relative = f"{bundle_relative}/evidence"
    task_execution = {
        "protocol": V610_TASK_BUNDLE_PROTOCOL,
        "attempt_number": attempt_number,
        "restart_count": attempt_number - 1,
        "max_process_restarts": V610_MAX_PROCESS_RESTARTS,
        "process_execution_id": marker["process_execution_id"],
        "process_started_at_utc": marker["process_started_at_utc"],
        "bundle_relative_path": bundle_relative,
        "bundle_identity_sha256": bundle_identity,
        "evidence_relative_path": evidence_relative,
        "evidence_tree_sha256": evidence_sha,
    }
    terminal_receipt = deepcopy(dict(receipt))
    terminal_receipt["task_execution"] = task_execution
    terminal_receipt.pop("task_receipt_sha256", None)
    terminal_receipt["task_receipt_sha256"] = sha256(terminal_receipt)
    final_marker = {
        **marker,
        "task_ended_at_utc": receipt["task_ended_at_utc"],
        "status": status,
        "bundle_identity_sha256": bundle_identity,
        "evidence_tree_sha256": evidence_sha,
        "terminal_receipt_sha256": terminal_receipt[
            "task_receipt_sha256"
        ],
    }
    write_json(_v610_bundle_marker_path(staging), final_marker)
    write_json(staging / "terminal_receipt.json", terminal_receipt)
    manifest = {
        "protocol": V610_TASK_BUNDLE_PROTOCOL,
        "task_identity": task_identity,
        "attempt_number": attempt_number,
        "status": status,
        "bundle_identity_sha256": bundle_identity,
        "evidence_tree_sha256": evidence_sha,
        "terminal_receipt_sha256": terminal_receipt[
            "task_receipt_sha256"
        ],
        "official_test_used": False,
    }
    manifest["bundle_manifest_sha256"] = sha256(manifest)
    write_json(staging / "bundle_manifest.json", manifest)
    _fsync_tree(staging)
    completed = _resolve_output_relative(output_dir, bundle_relative)
    completed.parent.mkdir(parents=True, exist_ok=True)
    if completed.exists():
        raise V6GenerationError(
            f"{task_identity}: completed task bundle already exists"
        )
    staging_parent = staging.parent
    os.replace(staging, completed)
    _fsync_directory(staging_parent)
    _fsync_directory(completed.parent)
    indexed = reconcile_v610_task_index(
        output_dir,
        task_identity=task_identity,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=semantic_contract,
        receipt_path=receipt_path,
    )
    if indexed != terminal_receipt:
        raise V6GenerationError(
            f"{task_identity}: derived receipt index rebuild drift"
        )
    return terminal_receipt


def v610_task_execution_binding_matches(
    receipt: Mapping[str, Any],
    *,
    output_dir: Path | None,
    task_identity: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> bool:
    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        return True
    if output_dir is None:
        return False
    execution = receipt.get("task_execution")
    keys = {
        "protocol",
        "attempt_number",
        "restart_count",
        "max_process_restarts",
        "process_execution_id",
        "process_started_at_utc",
        "bundle_relative_path",
        "bundle_identity_sha256",
        "evidence_relative_path",
        "evidence_tree_sha256",
    }
    if (
        not isinstance(execution, Mapping)
        or set(execution) != keys
        or execution.get("protocol") != V610_TASK_BUNDLE_PROTOCOL
        or not isinstance(execution.get("attempt_number"), int)
        or isinstance(execution.get("attempt_number"), bool)
        or execution["attempt_number"] <= 0
        or execution["attempt_number"] > V610_MAX_PROCESS_RESTARTS + 1
        or execution.get("restart_count")
        != execution["attempt_number"] - 1
        or execution.get("max_process_restarts") != V610_MAX_PROCESS_RESTARTS
        or SHA256_RE.fullmatch(
            str(execution.get("process_execution_id"))
        )
        is None
        or not valid_utc_timestamp(execution.get("process_started_at_utc"))
        or SHA256_RE.fullmatch(
            str(execution.get("bundle_identity_sha256"))
        )
        is None
        or SHA256_RE.fullmatch(
            str(execution.get("evidence_tree_sha256"))
        )
        is None
    ):
        return False
    try:
        bundle = _resolve_output_relative(
            output_dir,
            str(execution["bundle_relative_path"]),
        )
        expected_evidence = bundle / "evidence"
        observed_evidence = _resolve_output_relative(
            output_dir,
            str(execution["evidence_relative_path"]),
        )
        if (
            observed_evidence != expected_evidence
            or task_evidence_tree_sha256(expected_evidence)
            != execution["evidence_tree_sha256"]
        ):
            return False
        matched = V610_ATTEMPT_DIR_RE.fullmatch(bundle.name)
        if (
            matched is None
            or int(matched.group("number"))
            != execution["attempt_number"]
            or matched.group("suffix")
            != execution["bundle_identity_sha256"]
        ):
            return False
        loaded, marker = _load_completed_v610_bundle_receipt(
            bundle,
            task_identity=task_identity,
            attempt_number=execution["attempt_number"],
        )
    except (KeyError, TypeError, V6GenerationError):
        return False
    return (
        loaded == receipt
        and receipt.get("status") in {"PASS", "REJECTED"}
        and receipt.get("task_started_at_utc")
        == marker.get("task_started_at_utc")
        and receipt.get("task_ended_at_utc")
        == marker.get("task_ended_at_utc")
        and execution.get("process_execution_id")
        == marker.get("process_execution_id")
        and execution.get("process_started_at_utc")
        == marker.get("process_started_at_utc")
        and execution.get("bundle_identity_sha256")
        == marker.get("bundle_identity_sha256")
    )


def semantic(value: Any) -> Any:
    """Drop transport metadata while retaining every experimental message."""
    if isinstance(value, list):
        return [semantic(item) for item in value]
    if isinstance(value, dict):
        return {
            key: semantic(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_MESSAGE_FIELDS
        }
    return value


def semantic_sha256(value: Any) -> str:
    return sha256(semantic(value))


def parse_seed_set(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise V6GenerationError("--continuation-seeds must be comma-separated integers") from error
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise V6GenerationError("continuation seeds must be non-empty, distinct, non-negative")
    return seeds


def semantic_generation_contract(
    args: argparse.Namespace,
    *,
    continuation_seeds: Sequence[int],
    judge_model: str | None = None,
    judge_revision: str | None = None,
    judge_api_base: str | None = None,
    design_protocol: str = protocol.PROTOCOL,
    runtime_provenance: Mapping[str, Any] | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every parameter that can change generated task semantics.

    Sharding and output-path details deliberately live in the enclosing run
    contract.  This smaller contract is copied into task receipts and candidate
    pairs so a deleted/recreated ``run_contract.json`` cannot make stale task
    output eligible for resume under different generation settings.
    """

    effective_judge_model = judge_model or args.judge_model or args.user_model
    effective_judge_revision = (
        judge_revision or args.judge_revision or args.user_revision
    )
    effective_judge_api_base = (
        judge_api_base or args.judge_api_base or args.user_api_base
    )
    clean_agent_mode = getattr(args, "clean_agent_mode", "standard")
    completion_renderer = getattr(
        args, "completion_renderer", "legacy_assertion_echo"
    )
    matched_mode = matched_positive_mode(args)
    causal_mode = causal_cell_mode(args)
    measurement_mode = first_action_measurement_mode(args)
    seeds = list(continuation_seeds)
    decoding = {
        "temperature": DECODING_TEMPERATURE,
        "top_p": DECODING_TOP_P,
        "max_tokens": args.max_tokens,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "applies_to": ["teacher", "user", "judge"],
        "assistant_tool_only_normalization": (
            "execute_first_tool_call_then_replan"
        ),
    }
    runner = {
        "num_trials": RUN_NUM_TRIALS,
        "max_steps": args.max_steps,
        "max_errors": RUN_MAX_ERRORS,
        "timeout_seconds": args.timeout,
        "max_concurrency": RUN_MAX_CONCURRENCY,
        "max_retries": (
            0
            if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
            else RUN_MAX_RETRIES
        ),
        "retry_delay_seconds": RUN_RETRY_DELAY_SECONDS,
        "auto_resume": False,
        "hallucination_retries": RUN_HALLUCINATION_RETRIES,
        "enforce_communication_protocol": False,
    }
    contract = {
        "protocol": SEMANTIC_GENERATION_CONTRACT_PROTOCOL,
        "design_protocol": design_protocol,
        "tau2_commit": protocol.TAU2_COMMIT,
        "agent_name": AGENT_NAME,
        "clean_agent_mode": clean_agent_mode,
        "clean_agent_name": (
            REFERENCE_GUIDED_CLEAN_AGENT_NAME
            if clean_agent_mode == "reference_guided"
            else (
                (
                    "single_turn_user_then_deterministic_tau2_reference_replay"
                    if clean_agent_mode
                    == "single_turn_user_reference_replay"
                    else "deterministic_tau2_reference_replay"
                )
                if clean_agent_mode
                in (
                    "deterministic_reference_replay",
                    "single_turn_user_reference_replay",
                )
                else AGENT_NAME
            )
        ),
        "clean_prefix_max_steps": (
            SINGLE_TURN_PREFIX_MAX_STEPS
            if clean_agent_mode == "single_turn_user_reference_replay"
            else args.max_steps
        ),
        "clean_prefix_stops_before_assistant_tool_action": (
            clean_agent_mode == "single_turn_user_reference_replay"
        ),
        "completion_renderer": completion_renderer,
        # Retain the legacy field for old receipts, but do not use it to
        # conflate positive-data construction with causal measurement.
        "recovery_continuation_mode": getattr(
            args, "recovery_continuation_mode", "fresh_teacher"
        ),
        "matched_positive_continuation_mode": matched_mode,
        "causal_cell_continuation_mode": causal_mode,
        "first_action_measurement_mode": measurement_mode,
        "system_instruction_sha256": sha256(SFT_SYSTEM_INSTRUCTION),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_api_base": args.teacher_api_base,
        "user_model": args.user_model,
        "user_revision": args.user_revision,
        "user_api_base": args.user_api_base,
        "judge_model": effective_judge_model,
        "judge_revision": effective_judge_revision,
        "judge_api_base": effective_judge_api_base,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "timeout": args.timeout,
        "clean_attempts": args.clean_attempts,
        "recovery_attempts": args.recovery_attempts,
        "continuation_seeds": seeds,
        "continuation_seed_set_sha256": sha256(seeds),
        "decoding": decoding,
        "runner": runner,
        "seed_contract": {
            "clean_seed_source": "registry.choice_seed",
            "clean_attempt_seed_derivation": (
                "choice_seed + zero_based_attempt_index"
            ),
            "recovery_seed_source": "registry.branch.recovery_seed",
            "recovery_attempt_seed_derivation": (
                "recovery_seed + zero_based_attempt_index"
            ),
            "forced_first_continuation_seeds": seeds,
            "teacher_unforced_first_action_seed_source": (
                "registry.branch.recovery_seed"
            ),
            "teacher_unforced_queries_per_error_context": (
                1 if measurement_mode == "teacher_unforced" else 0
            ),
            "judge_seed": seeds[0],
        },
        "fresh_recovery_generated": matched_mode == "fresh_teacher",
        "matched_positive_fresh_recovery_generated": (
            matched_mode == "fresh_teacher"
        ),
        "matched_positive_gold_suffix_used": matched_mode
        in {
            "deterministic_reference_tail",
            "deterministic_reference_completion",
        },
        "causal_cell_fresh_recovery_generated": causal_mode == "fresh_teacher",
        "causal_cell_gold_suffix_visible": causal_mode
        in {
            "deterministic_reference_tail",
            "deterministic_reference_completion",
        },
        "teacher_unforced_first_action_generated": (
            measurement_mode == "teacher_unforced"
        ),
        "teacher_unforced_probe_contract": {
            "raw_teacher_response": measurement_mode == "teacher_unforced",
            "assistant_generations_per_error_context": (
                1 if measurement_mode == "teacher_unforced" else 0
            ),
            "normalization_applied": False,
            "user_continuation_generated": False,
            "judge_invoked": False,
            "task_success_status": (
                "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE"
                if measurement_mode == "teacher_unforced"
                else "NOT_RUN"
            ),
            "retry_count": 0,
        },
        # V6.6 defined this legacy field as whether the single-turn model saw
        # the future, which was always false.  V6.10 explicitly separates
        # that legacy visibility claim from its deterministic positive-data
        # construction fields above.
        "gold_clean_future_visible": (
            design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
            and clean_agent_mode
            in {
                "deterministic_reference_replay",
                "single_turn_user_reference_replay",
            }
        ),
        "runtime_provenance": deepcopy(dict(runtime_provenance or {})),
    }
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        execution = deepcopy(dict(execution_provenance or {}))
        runtime = contract["runtime_provenance"]
        exact_argv = execution.get("exact_argv")
        parent_hashes = runtime.get("parent_artifact_hashes")
        task_file_hashes = runtime.get("task_file_sha256")
        release_manifest = runtime.get("release_manifest")
        if (
            not isinstance(exact_argv, list)
            or not exact_argv
            or any(not isinstance(value, str) or not value for value in exact_argv)
            or execution.get("exact_argv_sha256") != sha256(exact_argv)
            or not isinstance(execution.get("python_executable"), str)
            or not execution["python_executable"]
            or not valid_utc_timestamp(execution.get("run_started_at_utc"))
            or execution.get("run_end_recording_policy")
            != "generation_receipt_records_final_utc_after_terminal_merge"
            or not isinstance(parent_hashes, dict)
            or set(parent_hashes) != V610_PARENT_ARTIFACT_HASH_KEYS
            or any(
                SHA256_RE.fullmatch(str(value)) is None
                for value in parent_hashes.values()
            )
            or not isinstance(task_file_hashes, dict)
            or set(task_file_hashes) != {"airline", "retail"}
            or any(
                SHA256_RE.fullmatch(str(value)) is None
                for value in task_file_hashes.values()
            )
            or not isinstance(release_manifest, dict)
            or release_manifest.get("protocol")
            != V610_RELEASE_MANIFEST_PROTOCOL
            or release_manifest.get("file_sha256")
            != parent_hashes.get("release_manifest_sha256")
            or runtime.get("release_manifest_sha256")
            != parent_hashes.get("release_manifest_sha256")
        ):
            raise V6GenerationError(
                "V6.10 semantic contract lacks exact execution/release "
                "provenance"
            )
        contract.update(
            {
                "execution_provenance": execution,
                "parent_artifact_hashes": deepcopy(parent_hashes),
                "task_file_sha256": deepcopy(task_file_hashes),
                "release_manifest": deepcopy(release_manifest),
                "release_manifest_sha256": runtime[
                    "release_manifest_sha256"
                ],
            }
        )
        # Re-run the full cross-artifact check after the explicit semantic
        # fields exist; this prevents callers from constructing internally
        # inconsistent receipts outside ``main``.
        v610_explicit_receipt_fields(contract)
    return contract


def _utc_datetime(value: str) -> datetime:
    if not valid_utc_timestamp(value):
        raise V6GenerationError("timestamp is not canonical UTC")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def v610_explicit_receipt_fields(
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and revalidate immutable V6.10 fields copied into receipts."""

    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        return {}
    execution = semantic_contract.get("execution_provenance")
    runtime = semantic_contract.get("runtime_provenance")
    parent_hashes = semantic_contract.get("parent_artifact_hashes")
    task_file_hashes = semantic_contract.get("task_file_sha256")
    release_manifest = semantic_contract.get("release_manifest")
    release_identity_keys = {
        "source_commit",
        "source_tree",
        "container_image_digest",
        "tau2_commit",
        "config_sha256",
        "preregistration_sha256",
        "split_manifest_sha256",
        "reference_preflight_receipt_sha256",
        "registry_file_sha256",
        "source_container_provenance_sha256",
        "model_server_receipts_sha256",
        "runtime_receipt_hashes_sha256",
    }
    release_binding_keys = {
        "protocol",
        "file_sha256",
        "semantic_sha256",
        "created_at_utc",
        "identities",
        "relevant_scripts",
        "relevant_scripts_sha256",
        "official_test_used",
    }
    release_identities = (
        release_manifest.get("identities")
        if isinstance(release_manifest, Mapping)
        else None
    )
    release_scripts = (
        release_manifest.get("relevant_scripts")
        if isinstance(release_manifest, Mapping)
        else None
    )
    expected_release_identities = {
        "source_commit": runtime.get("source_commit"),
        "source_tree": runtime.get("source_tree"),
        "container_image_digest": runtime.get("container_image_digest"),
        "tau2_commit": runtime.get("tau2_commit"),
        **{
            key: value
            for key, value in (
                dict(parent_hashes).items()
                if isinstance(parent_hashes, Mapping)
                else ()
            )
            if key != "release_manifest_sha256"
        },
    }
    if not isinstance(execution, Mapping) or not isinstance(runtime, Mapping):
        raise V6GenerationError("V6.10 receipt provenance is absent")
    exact_argv = execution.get("exact_argv")
    if (
        not isinstance(exact_argv, list)
        or not exact_argv
        or any(not isinstance(value, str) or not value for value in exact_argv)
        or execution.get("exact_argv_sha256") != sha256(exact_argv)
        or not valid_utc_timestamp(execution.get("run_started_at_utc"))
        or execution.get("run_end_recording_policy")
        != "generation_receipt_records_final_utc_after_terminal_merge"
        or not isinstance(parent_hashes, Mapping)
        or set(parent_hashes) != V610_PARENT_ARTIFACT_HASH_KEYS
        or any(
            SHA256_RE.fullmatch(str(value)) is None
            for value in parent_hashes.values()
        )
        or not isinstance(task_file_hashes, Mapping)
        or set(task_file_hashes) != {"airline", "retail"}
        or any(
            SHA256_RE.fullmatch(str(value)) is None
            for value in task_file_hashes.values()
        )
        or not isinstance(release_manifest, Mapping)
        or set(release_manifest) != release_binding_keys
        or release_manifest.get("protocol") != V610_RELEASE_MANIFEST_PROTOCOL
        or release_manifest.get("file_sha256")
        != parent_hashes.get("release_manifest_sha256")
        or SHA256_RE.fullmatch(
            str(release_manifest.get("semantic_sha256"))
        )
        is None
        or not valid_utc_timestamp(
            release_manifest.get("created_at_utc")
        )
        or release_manifest.get("official_test_used") is not False
        or not isinstance(release_identities, Mapping)
        or set(release_identities) != release_identity_keys
        or dict(release_identities) != expected_release_identities
        or not isinstance(release_scripts, Mapping)
        or set(release_scripts) != V610_REQUIRED_RELEASE_SCRIPTS
        or any(
            SHA256_RE.fullmatch(str(value)) is None
            for value in release_scripts.values()
        )
        or release_manifest.get("relevant_scripts_sha256")
        != sha256(dict(release_scripts))
        or semantic_contract.get("release_manifest_sha256")
        != parent_hashes.get("release_manifest_sha256")
        or runtime.get("parent_artifact_hashes") != parent_hashes
        or runtime.get("task_file_sha256") != task_file_hashes
        or runtime.get("release_manifest") != release_manifest
        or runtime.get("release_manifest_sha256")
        != parent_hashes.get("release_manifest_sha256")
    ):
        raise V6GenerationError("V6.10 explicit receipt provenance drift")
    return {
        "execution_provenance": deepcopy(dict(execution)),
        "exact_argv": deepcopy(exact_argv),
        "exact_argv_sha256": execution["exact_argv_sha256"],
        "run_started_at_utc": execution["run_started_at_utc"],
        "run_ended_at_utc": None,
        "parent_artifact_hashes": deepcopy(dict(parent_hashes)),
        "task_file_sha256": deepcopy(dict(task_file_hashes)),
        "release_manifest": deepcopy(dict(release_manifest)),
        "release_manifest_sha256": semantic_contract[
            "release_manifest_sha256"
        ],
    }


def v610_task_receipt_fields(
    semantic_contract: Mapping[str, Any],
    *,
    task_identity: str,
    task_started_at_utc: str,
    task_ended_at_utc: str,
) -> dict[str, Any]:
    """Build immutable task timing and source-file bindings for V6.10."""

    fields = v610_explicit_receipt_fields(semantic_contract)
    if not fields:
        return {}
    domain = task_identity.split(":", 1)[0]
    task_files = fields["task_file_sha256"]
    if domain not in task_files:
        raise V6GenerationError(
            f"{task_identity}: no pinned task-file hash for domain"
        )
    run_started = _utc_datetime(fields["run_started_at_utc"])
    task_started = _utc_datetime(task_started_at_utc)
    task_ended = _utc_datetime(task_ended_at_utc)
    if task_started < run_started or task_ended < task_started:
        raise V6GenerationError(
            f"{task_identity}: task UTC timing is out of order"
        )
    fields.update(
        {
            "task_started_at_utc": task_started_at_utc,
            "task_ended_at_utc": task_ended_at_utc,
            "task_file": {
                "domain": domain,
                "relative_path": (
                    f"data/tau2/domains/{domain}/tasks.json"
                ),
                "sha256": task_files[domain],
            },
        }
    )
    return fields


def v610_task_receipt_fields_match(
    receipt: Mapping[str, Any],
    *,
    semantic_contract: Mapping[str, Any],
    task_identity: str,
) -> bool:
    if (
        semantic_contract.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        return True
    try:
        expected = v610_task_receipt_fields(
            semantic_contract,
            task_identity=task_identity,
            task_started_at_utc=str(receipt.get("task_started_at_utc", "")),
            task_ended_at_utc=str(receipt.get("task_ended_at_utc", "")),
        )
    except V6GenerationError:
        return False
    return all(receipt.get(key) == value for key, value in expected.items())


def build_run_contract(
    args: argparse.Namespace,
    *,
    registry_file_sha256: str,
    registry_sha256: str,
    task_ids: Sequence[str],
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shard/run envelope around the semantic generation contract."""

    semantic_contract_copy = deepcopy(dict(semantic_contract))
    if (
        semantic_contract_copy.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
        and registry_file_sha256
        != semantic_contract_copy.get("parent_artifact_hashes", {}).get(
            "registry_file_sha256"
        )
    ):
        raise V6GenerationError(
            "V6.10 run registry file hash differs from release provenance"
        )
    contract = {
        "protocol": GENERATION_PROTOCOL,
        "design_protocol": semantic_contract_copy["design_protocol"],
        "registry_file_sha256": registry_file_sha256,
        "registry_sha256": registry_sha256,
        "phase": args.phase,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "smoke_task": args.smoke_task,
        "task_ids": sorted(task_ids),
        # Retain the flat public fields for CLI/artifact compatibility while
        # binding their complete semantics below.
        "tau2_commit": semantic_contract_copy["tau2_commit"],
        "teacher_model": semantic_contract_copy["teacher_model"],
        "teacher_revision": semantic_contract_copy["teacher_revision"],
        "teacher_api_base": semantic_contract_copy["teacher_api_base"],
        "user_model": semantic_contract_copy["user_model"],
        "user_revision": semantic_contract_copy["user_revision"],
        "user_api_base": semantic_contract_copy["user_api_base"],
        "judge_model": semantic_contract_copy["judge_model"],
        "judge_revision": semantic_contract_copy["judge_revision"],
        "judge_api_base": semantic_contract_copy["judge_api_base"],
        "max_tokens": semantic_contract_copy["max_tokens"],
        "max_steps": semantic_contract_copy["max_steps"],
        "timeout": semantic_contract_copy["timeout"],
        "clean_attempts": semantic_contract_copy["clean_attempts"],
        "recovery_attempts": semantic_contract_copy["recovery_attempts"],
        "continuation_seeds": semantic_contract_copy["continuation_seeds"],
        "matched_positive_continuation_mode": semantic_contract_copy[
            "matched_positive_continuation_mode"
        ],
        "causal_cell_continuation_mode": semantic_contract_copy[
            "causal_cell_continuation_mode"
        ],
        "first_action_measurement_mode": semantic_contract_copy[
            "first_action_measurement_mode"
        ],
        "runtime_provenance": deepcopy(
            semantic_contract_copy.get("runtime_provenance", {})
        ),
        "attempt_id": (
            semantic_contract_copy.get("runtime_provenance", {})
        ).get("attempt_id"),
        "decoding": deepcopy(semantic_contract_copy["decoding"]),
        "runner": deepcopy(semantic_contract_copy["runner"]),
        "semantic_generation_contract": semantic_contract_copy,
        "semantic_generation_contract_sha256": sha256(semantic_contract_copy),
        "official_test_used": False,
    }
    contract.update(v610_explicit_receipt_fields(semantic_contract_copy))
    return contract


def validate_task_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    task_identity: str,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    expected_candidate_pair_ids: Sequence[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Fail closed unless a PASS receipt belongs to this exact generation run."""

    expected_semantic = dict(semantic_contract)
    semantic_contract_sha256 = sha256(expected_semantic)
    expected_runtime_provenance = deepcopy(
        expected_semantic.get("runtime_provenance", {})
    )
    expected_attempt_id = expected_runtime_provenance.get("attempt_id")
    pairs = receipt.get("candidate_pairs")
    expected_pair_ids = (
        tuple(str(value) for value in expected_candidate_pair_ids)
        if expected_candidate_pair_ids is not None
        else None
    )
    observed_pair_ids = (
        [str(pair.get("candidate_pair_id")) for pair in pairs]
        if isinstance(pairs, list)
        and all(isinstance(pair, dict) for pair in pairs)
        else []
    )
    exact_pair_coverage = (
        True
        if expected_pair_ids is None
        else (
            len(observed_pair_ids) == len(set(observed_pair_ids))
            and set(observed_pair_ids) == set(expected_pair_ids)
            and len(observed_pair_ids) == len(expected_pair_ids)
        )
    )
    pair_hashes_match = (
        True
        if expected_pair_ids is None
        else all(
            isinstance(pair, dict)
            and isinstance(
                pair.get("generated_candidate_pair_sha256"), str
            )
            and pair.get("generated_candidate_pair_sha256")
            == sha256(
                {
                    key: value
                    for key, value in pair.items()
                    if key != "generated_candidate_pair_sha256"
                }
            )
            and pair.get("protocol") == GENERATION_PROTOCOL
            and pair.get("official_test_used") is False
            for pair in (pairs or [])
        )
    )
    pair_contracts_match = (
        isinstance(pairs, list)
        and bool(pairs)
        and receipt.get("candidate_pair_count") == len(pairs)
        and all(
            isinstance(pair, dict)
            and pair.get("task_identity") == task_identity
            and pair.get("registry_sha256") == registry_sha256
            and pair.get("generation_contract") == expected_semantic
            and pair.get("generation_contract_sha256")
            == semantic_contract_sha256
            for pair in pairs
        )
        and exact_pair_coverage
        and pair_hashes_match
    )
    receipt_hash_matches = (
        True
        if expected_semantic.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        else (
            receipt.get("task_receipt_sha256")
            == sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "task_receipt_sha256"
                }
            )
        )
    )
    official_test_seal_matches = (
        True
        if expected_semantic.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        else receipt.get("official_test_used") is False
    )
    scientific_outcome_matches = (
        True
        if expected_semantic.get("design_protocol")
        != V6_10_PIPELINE_CLOSURE_PROTOCOL
        else (
            receipt.get("execution_status") == "PASS"
            and receipt.get("scientific_outcome") == "ACCEPTED"
        )
    )
    if (
        receipt.get("protocol") != GENERATION_PROTOCOL
        or receipt.get("task_identity") != task_identity
        or receipt.get("status") != "PASS"
        or receipt.get("registry_sha256") != registry_sha256
        or receipt.get("run_contract_sha256") != run_contract_sha256
        or receipt.get("semantic_generation_contract")
        != expected_semantic
        or receipt.get("semantic_generation_contract_sha256")
        != semantic_contract_sha256
        or receipt.get("runtime_provenance", {})
        != expected_runtime_provenance
        or receipt.get("attempt_id") != expected_attempt_id
        or not v610_task_receipt_fields_match(
            receipt,
            semantic_contract=expected_semantic,
            task_identity=task_identity,
        )
        or not v610_task_execution_binding_matches(
            receipt,
            output_dir=output_dir,
            task_identity=task_identity,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=expected_semantic,
        )
        or not official_test_seal_matches
        or not scientific_outcome_matches
        or not receipt_hash_matches
        or not pair_contracts_match
    ):
        raise V6GenerationError(
            f"{task_identity}: existing receipt is not a valid PASS resume "
            "(run/semantic generation contract drift)"
        )


TASK_LOCAL_REJECTION_PATTERNS = (
    (
        "REFERENCE_PREFLIGHT_TASK_REJECTED",
        (
            "no verified pass reference plan",
            "reference preflight task is not pass",
        ),
    ),
    (
        "FEWER_THAN_THREE_EXECUTABLE_PAIRS",
        ("fewer than three executable candidate pairs",),
    ),
    (
        "REFERENCE_ACTION_INELIGIBLE",
        (
            "not authorized by the verified sanitized preflight",
            "executable registry slot binding differs",
            "executable registry is not bound",
        ),
    ),
    (
        "INJECTED_ERROR_INVALID",
        (
            "injected call was not a real tool error",
            "failed call changed environment state",
        ),
    ),
    (
        "CLEAN_GENERATION_EXHAUSTED",
        ("no successful clean rollout",),
    ),
    (
        "MATCHED_POSITIVE_RECOVERY_FAILED",
        (
            "no fresh successful recovery",
            "did not produce a successful matched recovery",
            "frozen corrective action returned a tool error",
        ),
    ),
    (
        "INDEPENDENT_REPLAY_FAILED",
        (
            "independent environment replay produced state drift",
            "recovered final state differs",
        ),
    ),
    (
        "POSITIVE_SUFFIX_INVALID",
        (
            "successful suffix contains an additional failed tool call",
            "has no successful",
        ),
    ),
)


def classify_task_local_rejection(error: Exception) -> str | None:
    if not isinstance(error, V6GenerationError):
        return None
    message = str(error).lower()
    for reason_code, patterns in TASK_LOCAL_REJECTION_PATTERNS:
        if any(pattern in message for pattern in patterns):
            return reason_code
    return None


def rejected_task_receipt(
    *,
    task_identity: str,
    reason_code: str,
    error: Exception,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    expected_candidate_pair_ids: Sequence[str],
    task_started_at_utc: str | None = None,
    task_ended_at_utc: str | None = None,
) -> dict[str, Any]:
    semantic = deepcopy(dict(semantic_contract))
    provenance = deepcopy(semantic.get("runtime_provenance", {}))
    v610 = (
        semantic.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    )
    receipt = {
        "protocol": GENERATION_PROTOCOL,
        "status": "PASS" if v610 else "REJECTED",
        "task_identity": task_identity,
        "reason_code": reason_code,
        "error_type": type(error).__name__,
        "error": str(error),
        "registry_sha256": registry_sha256,
        "run_contract_sha256": run_contract_sha256,
        "semantic_generation_contract": semantic,
        "semantic_generation_contract_sha256": sha256(semantic),
        "runtime_provenance": provenance,
        "attempt_id": provenance.get("attempt_id"),
        "expected_candidate_pair_ids": sorted(
            str(value) for value in expected_candidate_pair_ids
        ),
        "candidate_pairs": [],
        "candidate_pair_count": 0,
        "training_started": False,
        "official_test_used": False,
    }
    if v610:
        receipt.update(
            {
                "execution_status": "PASS",
                "scientific_outcome": "REJECTED",
            }
        )
        if task_started_at_utc is None or task_ended_at_utc is None:
            raise V6GenerationError(
                f"{task_identity}: V6.10 rejection lacks task UTC timing"
            )
        receipt.update(
            v610_task_receipt_fields(
                semantic,
                task_identity=task_identity,
                task_started_at_utc=task_started_at_utc,
                task_ended_at_utc=task_ended_at_utc,
            )
        )
    receipt["task_receipt_sha256"] = sha256(receipt)
    return receipt


def validate_rejected_task_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    task_identity: str,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    expected_candidate_pair_ids: Sequence[str],
    output_dir: Path | None = None,
) -> None:
    expected_semantic = dict(semantic_contract)
    provenance = expected_semantic.get("runtime_provenance", {})
    unhashed = {
        key: value
        for key, value in receipt.items()
        if key != "task_receipt_sha256"
    }
    if (
        receipt.get("protocol") != GENERATION_PROTOCOL
        or (
            expected_semantic.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
            and (
                receipt.get("status") != "PASS"
                or receipt.get("execution_status") != "PASS"
                or receipt.get("scientific_outcome") != "REJECTED"
            )
        )
        or (
            expected_semantic.get("design_protocol")
            != V6_10_PIPELINE_CLOSURE_PROTOCOL
            and receipt.get("status") != "REJECTED"
        )
        or receipt.get("task_identity") != task_identity
        or receipt.get("registry_sha256") != registry_sha256
        or receipt.get("run_contract_sha256") != run_contract_sha256
        or receipt.get("semantic_generation_contract") != expected_semantic
        or receipt.get("semantic_generation_contract_sha256")
        != sha256(expected_semantic)
        or receipt.get("runtime_provenance") != provenance
        or receipt.get("attempt_id") != provenance.get("attempt_id")
        or not v610_task_receipt_fields_match(
            receipt,
            semantic_contract=expected_semantic,
            task_identity=task_identity,
        )
        or not v610_task_execution_binding_matches(
            receipt,
            output_dir=output_dir,
            task_identity=task_identity,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=expected_semantic,
        )
        or receipt.get("expected_candidate_pair_ids")
        != sorted(str(value) for value in expected_candidate_pair_ids)
        or receipt.get("candidate_pairs") != []
        or receipt.get("candidate_pair_count") != 0
        or not isinstance(receipt.get("reason_code"), str)
        or receipt.get("official_test_used") is not False
        or receipt.get("task_receipt_sha256") != sha256(unhashed)
    ):
        raise V6GenerationError(
            f"{task_identity}: existing REJECTED receipt is stale/invalid"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("compatibility", "pilot", "formal"),
        required=True,
    )
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-api-base", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-revision", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--clean-attempts", type=int, default=2)
    parser.add_argument(
        "--clean-agent-mode",
        choices=CLEAN_AGENT_MODES,
        default="standard",
        help=(
            "Agent used only to construct the successful clean source. "
            "reference_guided uses tau2 evaluation actions and therefore "
            "requires a separately versioned scientific protocol."
        ),
    )
    parser.add_argument("--recovery-attempts", type=int, default=2)
    parser.add_argument(
        "--recovery-continuation-mode",
        choices=RECOVERY_CONTINUATION_MODES,
        default="fresh_teacher",
        help=(
            "Legacy shared continuation mode. V6.10 requires the separate "
            "matched-positive and causal-cell modes below. "
            "deterministic reference modes require separately versioned "
            "scientific protocols."
        ),
    )
    parser.add_argument(
        "--matched-positive-continuation-mode",
        choices=RECOVERY_CONTINUATION_MODES,
        help=(
            "Continuation used only to construct successful matched positive "
            "training data."
        ),
    )
    parser.add_argument(
        "--causal-cell-continuation-mode",
        choices=RECOVERY_CONTINUATION_MODES,
        help=(
            "Continuation used only in the four forced-first causal cells. "
            "V6.10 requires fresh_teacher so no gold suffix enters q."
        ),
    )
    parser.add_argument(
        "--first-action-measurement-mode",
        choices=("disabled", "teacher_unforced"),
        help=(
            "Independent unforced teacher first-action measurement. V6.10 "
            "requires teacher_unforced."
        ),
    )
    parser.add_argument(
        "--completion-renderer",
        choices=COMPLETION_RENDERERS,
        default="legacy_assertion_echo",
        help=(
            "Renderer for deterministic clean/recovery final confirmations. "
            "natural_direct_v1 requires a separately frozen protocol."
        ),
    )
    parser.add_argument(
        "--continuation-seeds",
        default=",".join(str(seed) for seed in DEFAULT_CONTINUATION_SEEDS),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--smoke-task",
        help="Diagnostic only: materialize exactly this registered phase task.",
    )
    parser.add_argument(
        "--expected-source-commit",
        help="Full experiment-source commit required by V6.10.",
    )
    parser.add_argument(
        "--expected-generation-script-sha256",
        help="Expected byte hash of this launched generator, required by V6.10.",
    )
    parser.add_argument(
        "--expected-tau2-commit",
        help="Full tau2 commit required by V6.10.",
    )
    parser.add_argument(
        "--reference-preflight-receipt",
        type=Path,
        help="PASS reference-execution preflight receipt required by V6.10.",
    )
    parser.add_argument(
        "--reference-preflight-receipt-sha256",
        help="Expected byte hash of the V6.10 reference-preflight receipt.",
    )
    parser.add_argument(
        "--container-image-digest",
        help="Immutable container image digest required by V6.10.",
    )
    parser.add_argument(
        "--source-container-provenance",
        type=Path,
        help="Source/container release receipt required by V6.10.",
    )
    parser.add_argument(
        "--source-container-provenance-sha256",
        help="Expected byte hash of source/container release receipt.",
    )
    parser.add_argument(
        "--model-server-receipts",
        type=Path,
        help="Endpoint identity/provenance receipts required by V6.10.",
    )
    parser.add_argument(
        "--model-server-receipts-sha256",
        help="Expected byte hash of the model-server receipt bundle.",
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Immutable cross-artifact release manifest required by V6.10.",
    )
    parser.add_argument(
        "--release-manifest-sha256",
        help="Expected byte hash of the immutable V6.10 release manifest.",
    )
    parser.add_argument(
        "--attempt-id",
        help="Caller-frozen attempt identity required by V6.10 and resume.",
    )
    return parser.parse_args()


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def register_agent() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.registry import registry

    class V6Agent(LLMAgent):
        def _generate_next_message(self, message, state):
            return normalize_tool_only_message(
                super()._generate_next_message(message, state)
            )

    def factory(tools, domain_policy, **kwargs):
        return V6Agent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(factory, AGENT_NAME)


def verify_registry(payload: Mapping[str, Any]) -> str:
    design_protocol = payload.get("design_protocol")
    expected_registry_protocol = (
        v610_protocol.REGISTRY_PROTOCOL
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
        else registry_contract.REGISTRY_PROTOCOL
    )
    if payload.get("protocol") != expected_registry_protocol:
        raise V6GenerationError("candidate registry protocol drift")
    if design_protocol not in (
        protocol.PROTOCOL,
        registry_contract.V6_1_72B_TEACHER_PROTOCOL,
        registry_contract.V6_2_REFERENCE_GUIDED_CLEAN_PROTOCOL,
        registry_contract.V6_3_DETERMINISTIC_CLEAN_REPLAY_PROTOCOL,
        registry_contract.V6_4_REFERENCE_TAIL_RECOVERY_PROTOCOL,
        registry_contract.V6_5_REFERENCE_COMPLETION_PROTOCOL,
        registry_contract.V6_6_SINGLE_TURN_CLEAN_PREFIX_PROTOCOL,
        registry_contract.V6_7_NATURAL_CLEAN_COMPLETION_PROTOCOL,
        registry_contract.V6_8_EXPLICIT_ASSERTION_RENDERER_PROTOCOL,
        registry_contract.V6_9_COMPLEMENTIZER_NORMALIZATION_PROTOCOL,
        V6_10_PIPELINE_CLOSURE_PROTOCOL,
    ):
        raise V6GenerationError("candidate registry design protocol drift")
    if (
        payload.get("selection_unit") != "candidate_pair"
        or payload.get("grouping_unit") != "choice_set"
        or payload.get("official_test_used") is not False
        or payload.get("official_test_sealed") is not True
        or payload.get("official_test_task_content_exported") is not False
        or payload.get("official_test_identity_overlap_count") != 0
    ):
        raise V6GenerationError("candidate registry violates the test seal")
    structural = payload.get("structural_eligibility_sha256")
    if structural != protocol.STRUCTURAL_ELIGIBILITY_SHA256:
        raise V6GenerationError("candidate registry structural hash drift")
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        preflight_sha256 = payload.get("reference_preflight_receipt_sha256")
        preflight_file_sha256 = payload.get(
            "reference_preflight_file_sha256"
        )
        sanitized_plan_hashes_sha256 = payload.get(
            "sanitized_plan_hashes_sha256"
        )
        if (
            SHA256_RE.fullmatch(str(preflight_sha256)) is None
            or SHA256_RE.fullmatch(str(preflight_file_sha256)) is None
            or SHA256_RE.fullmatch(str(sanitized_plan_hashes_sha256)) is None
        ):
            raise V6GenerationError(
                "V6.10 executable registry lacks preflight/sanitized-plan binding"
            )
        expected_phase_tasks = {
            "compatibility": list(v610_protocol.COMPATIBILITY_TASK_IDS),
            "pilot": list(v610_protocol.PROSPECTIVE_PILOT_TASK_IDS),
            "formal": list(v610_protocol.FORMAL_TASK_IDS),
        }
        phase_registry = payload.get("phase_registry")
        if not isinstance(phase_registry, Mapping):
            raise V6GenerationError("V6.10 registry lacks phase registry")
        for phase, expected_tasks in expected_phase_tasks.items():
            row = phase_registry.get(phase)
            if (
                not isinstance(row, Mapping)
                or row.get("task_ids") != expected_tasks
                or row.get("task_ids_sha256") != sha256(expected_tasks)
                or row.get("task_count") != len(expected_tasks)
                or row.get("all_tasks_require_terminal_receipts") is not True
            ):
                raise V6GenerationError(
                    f"V6.10 {phase} task population drift"
                )
    declared = payload.get("registry_sha256")
    if not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None:
        raise V6GenerationError("candidate registry lacks a valid registry_sha256")
    unhashed = dict(payload)
    unhashed.pop("registry_sha256", None)
    observed = sha256(unhashed)
    if observed != declared:
        raise V6GenerationError(
            f"candidate registry bytes/content drift: {observed} != {declared}"
        )
    pairs = payload.get("candidate_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise V6GenerationError("candidate registry is empty")
    pair_ids: set[str] = set()
    for row in pairs:
        if not isinstance(row, dict):
            raise V6GenerationError("candidate registry pair is not an object")
        pair_id = row.get("candidate_pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise V6GenerationError("candidate_pair_id is absent or duplicated")
        pair_ids.add(pair_id)
        if (
            row.get("partition") != "arm_train"
            or row.get("official_test_used") is not False
            or not isinstance(row.get("branches"), list)
            or len(row["branches"]) != 2
        ):
            raise V6GenerationError(f"{pair_id}: malformed/forbidden registry pair")
        registered = dict(row)
        pair_hash = registered.pop("candidate_pair_sha256", None)
        if pair_hash != sha256(registered):
            raise V6GenerationError(f"{pair_id}: candidate-pair hash drift")
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
            if (
                row.get("reference_preflight_receipt_sha256")
                != payload["reference_preflight_receipt_sha256"]
                or row.get("reference_preflight_file_sha256")
                != payload["reference_preflight_file_sha256"]
                or SHA256_RE.fullmatch(
                    str(row.get("reference_task_preflight_sha256"))
                )
                is None
                or SHA256_RE.fullmatch(
                    str(row.get("sanitized_reference_plan_sha256"))
                )
                is None
            ):
                raise V6GenerationError(
                    f"{pair_id}: V6.10 pair preflight binding drift"
                )
            expected_binding_keys = {
                "reference_preflight_receipt_sha256",
                "reference_preflight_file_sha256",
                "reference_task_preflight_sha256",
                "sanitized_reference_plan_sha256",
                "reference_action_index",
                "reference_slot_id",
                "reference_slot_sha256",
                "call_semantics_sha256",
            }
            for branch in row["branches"]:
                if not isinstance(branch, Mapping):
                    raise V6GenerationError(
                        f"{pair_id}: V6.10 branch is malformed"
                    )
                branch_copy = dict(branch)
                branch_hash = branch_copy.pop("branch_slot_sha256", None)
                if branch_hash != sha256(branch_copy):
                    raise V6GenerationError(
                        f"{pair_id}: V6.10 branch hash drift"
                    )
                binding = branch.get("reference_preflight_binding")
                corrective = branch.get("corrective_action_spec")
                if (
                    not isinstance(binding, Mapping)
                    or set(binding) != expected_binding_keys
                    or not isinstance(corrective, Mapping)
                    or binding["reference_preflight_receipt_sha256"]
                    != payload["reference_preflight_receipt_sha256"]
                    or binding["reference_preflight_file_sha256"]
                    != payload["reference_preflight_file_sha256"]
                    or binding["reference_task_preflight_sha256"]
                    != row["reference_task_preflight_sha256"]
                    or binding["sanitized_reference_plan_sha256"]
                    != row["sanitized_reference_plan_sha256"]
                    or binding["reference_action_index"]
                    != corrective.get("reference_action_index")
                    or binding["reference_slot_id"]
                    != corrective.get("reference_slot_id")
                ):
                    raise V6GenerationError(
                        f"{pair_id}: V6.10 exact branch binding drift"
                    )
    return declared


def phase_pairs(
    registry: Mapping[str, Any],
    *,
    phase: str,
    shard_index: int,
    num_shards: int,
    smoke_task: str | None,
) -> list[dict[str, Any]]:
    selected_tasks = selected_phase_tasks(
        registry,
        phase=phase,
        shard_index=shard_index,
        num_shards=num_shards,
        smoke_task=smoke_task,
    )
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in selected_tasks}
    for row in registry["candidate_pairs"]:
        if row.get("phase") == phase and row.get("task_identity") in by_task:
            by_task[str(row["task_identity"])].append(deepcopy(row))
    output: list[dict[str, Any]] = []
    v610_registry = registry.get("protocol") == v610_protocol.REGISTRY_PROTOCOL
    for task in selected_tasks:
        pairs = sorted(by_task[task], key=lambda row: str(row["candidate_pair_id"]))
        if len(pairs) < protocol.MIN_PAIRS_PER_TASK and not v610_registry:
            raise V6GenerationError(
                f"{task}: registry has {len(pairs)} pairs; "
                f"need >= {protocol.MIN_PAIRS_PER_TASK}"
            )
        output.extend(pairs)
    return output


def selected_phase_tasks(
    registry: Mapping[str, Any],
    *,
    phase: str,
    shard_index: int,
    num_shards: int,
    smoke_task: str | None,
) -> list[str]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise V6GenerationError("shard index must be in [0, num_shards)")
    phase_row = (registry.get("phase_registry") or {}).get(phase)
    if not isinstance(phase_row, dict) or not isinstance(
        phase_row.get("task_ids"), list
    ):
        raise V6GenerationError(f"registry has no {phase} phase")
    registered_tasks = [str(value) for value in phase_row["task_ids"]]
    if smoke_task is not None:
        if smoke_task not in registered_tasks:
            raise V6GenerationError("--smoke-task is not registered in this phase")
        selected_tasks = [smoke_task]
    else:
        ordered = sorted(
            registered_tasks,
            key=lambda task: (hashlib.sha256(task.encode()).hexdigest(), task),
        )
        selected_tasks = ordered[shard_index::num_shards]
    return selected_tasks


def select_task(identity: str):
    from tau2.registry import registry

    domain, task_id = identity.split(":", 1)
    tasks = {str(task.id): task for task in registry.get_tasks_loader(domain)(None)}
    if task_id not in tasks:
        raise V6GenerationError(f"pinned tau2 lacks {identity}")
    return domain, tasks[task_id]


def text_run_config(
    args: argparse.Namespace,
    *,
    domain: str,
    seed: int,
    agent_name: str = AGENT_NAME,
    max_steps: int | None = None,
):
    from tau2.data_model.simulation import TextRunConfig

    teacher_args = endpoint_args(
        args.teacher_api_base,
        "v6-local",
        max_tokens=args.max_tokens,
        seed=seed,
        temperature=DECODING_TEMPERATURE,
        top_p=DECODING_TOP_P,
    )
    user_args = endpoint_args(
        args.user_api_base,
        "v6-local",
        max_tokens=args.max_tokens,
        seed=seed,
        temperature=DECODING_TEMPERATURE,
        top_p=DECODING_TOP_P,
    )
    return TextRunConfig(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=litellm_openai_model(args.teacher_model),
        llm_args_agent=teacher_args,
        llm_user=litellm_openai_model(args.user_model),
        llm_args_user=user_args,
        num_trials=RUN_NUM_TRIALS,
        max_steps=args.max_steps if max_steps is None else max_steps,
        max_errors=RUN_MAX_ERRORS,
        timeout=args.timeout,
        max_concurrency=RUN_MAX_CONCURRENCY,
        seed=seed,
        log_level="INFO",
        max_retries=(
            0
            if getattr(args, "v610_contract_active", False)
            else RUN_MAX_RETRIES
        ),
        retry_delay=RUN_RETRY_DELAY_SECONDS,
        auto_resume=False,
        hallucination_retries=RUN_HALLUCINATION_RETRIES,
        enforce_communication_protocol=False,
        verbose_logs=True,
    )


def run_one(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    seed: int,
    save_dir: Path,
    agent_name: str = AGENT_NAME,
    max_steps: int | None = None,
):
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_single_task

    started = time.monotonic()
    simulation = run_single_task(
        text_run_config(
            args,
            domain=domain,
            seed=seed,
            agent_name=agent_name,
            max_steps=max_steps,
        ),
        task,
        seed=seed,
        evaluation_type=EvaluationType.ALL,
        save_dir=save_dir,
        verbose_logs=True,
    )
    return simulation, time.monotonic() - started


def reward(simulation: Any) -> float:
    value = getattr(getattr(simulation, "reward_info", None), "reward", None)
    if value is None:
        raise V6GenerationError("tau2 simulation has no official reward")
    return float(value)


def dynamic_official_reward(
    simulation: Any,
    *,
    context: str,
    require_success: bool,
) -> tuple[float, dict[str, Any]]:
    info = getattr(simulation, "reward_info", None)
    value = reward(simulation)
    if not 0.0 <= value <= 1.0:
        raise V6GenerationError(
            f"{context}: official reward is not finite in [0,1]"
        )
    if info is None or not callable(getattr(info, "model_dump", None)):
        raise V6GenerationError(f"{context}: official reward_info is missing")
    payload = info.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise V6GenerationError(f"{context}: official reward_info is malformed")
    if require_success and value != 1.0:
        raise V6GenerationError(
            f"{context}: dynamic ALL evaluation did not reach reward 1"
        )
    return value, payload


def messages(simulation: Any) -> list[dict[str, Any]]:
    values = getattr(simulation, "messages", None)
    if not isinstance(values, list):
        raise V6GenerationError("tau2 simulation has no messages")
    return [message.model_dump(mode="json") for message in values]


def parse_messages(values: Sequence[Mapping[str, Any]]):
    from tau2.data_model.message import Message
    from pydantic import TypeAdapter

    return [
        TypeAdapter(Message).validate_python(deepcopy(dict(value)))
        for value in values
    ]


def initial_environment(domain: str, task: Any, history: Sequence[Mapping[str, Any]]):
    from tau2.runner import build_environment

    environment = build_environment(domain)
    initial = task.initial_state
    environment.set_state(
        deepcopy(initial.initialization_data) if initial else None,
        deepcopy(initial.initialization_actions) if initial else None,
        parse_messages(history),
    )
    return environment


def database_hashes(environment: Any) -> dict[str, str | None]:
    return {
        "agent_db_hash": environment.get_db_hash(),
        "user_db_hash": environment.get_user_db_hash(),
    }


def environment_tool_schemas(environment: Any) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in environment.get_tools():
        raw = deepcopy(tool.openai_schema)
        if raw.get("type") == "function" and isinstance(
            raw.get("function"), dict
        ):
            schema = raw
        elif isinstance(raw.get("name"), str):
            schema = {"type": "function", "function": raw}
        else:
            raise V6GenerationError("tau2 returned an unsupported tool schema")
        schemas.append(schema)
    schemas.sort(key=lambda row: str(row["function"]["name"]))
    if not schemas:
        raise V6GenerationError("tau2 environment exposes no assistant tools")
    return schemas


def training_system_message(policy_text: str) -> dict[str, str]:
    if not isinstance(policy_text, str) or not policy_text.strip():
        raise V6GenerationError("tau2 environment exposes no domain policy")
    return {
        "role": "system",
        "content": (
            "<instructions>\n"
            + SFT_SYSTEM_INSTRUCTION
            + "\n</instructions>\n<policy>\n"
            + policy_text
            + "\n</policy>"
        ),
    }


def snapshot_state(
    task: Any, prefix: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    initial = task.initial_state
    return {
        "initialization_data": (
            initial.initialization_data.model_dump(mode="json")
            if initial is not None and initial.initialization_data is not None
            else None
        ),
        "initialization_actions": (
            [
                action.model_dump(mode="json")
                for action in initial.initialization_actions
            ]
            if initial is not None and initial.initialization_actions
            else []
        ),
        "shared_prefix": deepcopy(list(prefix)),
    }


def execute_call(
    environment: Any, call: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tau2.data_model.message import ToolCall
    from pydantic import TypeAdapter

    parsed = TypeAdapter(ToolCall).validate_python(deepcopy(dict(call)))
    result = environment.get_response(parsed)
    call_message = {
        "role": parsed.requestor,
        "content": None,
        "tool_calls": [parsed.model_dump(mode="json")],
    }
    return call_message, result.model_dump(mode="json")


def make_recovery_task(
    task: Any, history: Sequence[Mapping[str, Any]]
):
    from tau2.data_model.tasks import InitialState

    copied = task.model_copy(deep=True)
    initial = task.initial_state
    copied.initial_state = InitialState(
        initialization_data=(
            deepcopy(initial.initialization_data) if initial else None
        ),
        initialization_actions=(
            deepcopy(initial.initialization_actions) if initial else None
        ),
        message_history=parse_messages(history),
    )
    return copied


def first_assistant_tool_index(values: Sequence[Mapping[str, Any]]) -> int | None:
    for index, message in enumerate(values):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return index
    return None


def single_turn_clean_prefix(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and freeze one assistant-greeting/user-response exchange."""

    prefix = [deepcopy(dict(value)) for value in values]
    if len(prefix) != SINGLE_TURN_PREFIX_MESSAGE_COUNT:
        raise V6GenerationError(
            "single-turn clean prefix must contain exactly two messages"
        )
    if prefix[0].get("role") != "assistant":
        raise V6GenerationError(
            "single-turn clean prefix must start with the assistant greeting"
        )
    if prefix[0].get("tool_calls"):
        raise V6GenerationError(
            "single-turn assistant greeting must not contain a tool call"
        )
    if prefix[1].get("role") != "user":
        raise V6GenerationError(
            "single-turn clean prefix must end with the first user response"
        )
    if prefix[1].get("tool_calls"):
        raise V6GenerationError(
            "single-turn user response must not contain a tool call"
        )
    if any(row.get("role") == "tool" for row in prefix):
        raise V6GenerationError(
            "single-turn clean prefix must not contain a tool result"
        )
    return prefix


def _capitalize_sentence(value: str) -> str:
    value = value.strip()
    if not value:
        raise V6GenerationError("completion assertion is empty")
    return value[0].upper() + value[1:]


def naturalize_completion_assertion(assertion: str) -> str:
    """Convert frozen evaluator phrasing into direct assistant speech."""

    value = assertion.strip()
    prefix_rules = (
        ("Agent should tell the user ", ""),
        ("Agent should provide ", "Here is "),
        ("Agent should not approve ", "I did not approve "),
        ("Agent should not cancel ", "I did not cancel "),
        ("Agent should not offer ", "I did not offer "),
        ("Agent should cancel ", "I cancelled "),
        ("Agent should book ", "I booked "),
        ("Agent should exchange ", "I exchanged "),
        ("Agent should modify ", "I modified "),
        ("Agent should realize that ", ""),
        ("Agent communicates that ", ""),
        ("Agent communicated that ", ""),
        ("Agent mentions that ", ""),
        ("Agent updates ", "I updated "),
        ("Agent assigns ", "I assigned "),
        ("Agent add ", "I added "),
        ("Agent does not allow ", "I did not allow "),
        ("Agent does not offer ", "I did not offer "),
        ("Agent cancels ", "I cancelled "),
        ("Agent books ", "I booked "),
        ("Agent charges ", "I charged "),
        ("Agent verifies ", "I verified "),
        ("Check that Agent clearly identifies that ", ""),
        ("Check that agent correctly adds ", "I added "),
    )
    if value.startswith("For this reservation Agent charges "):
        rendered = (
            "For this reservation, I charged "
            + value.removeprefix("For this reservation Agent charges ")
        )
    else:
        rendered = value
        for prefix, replacement in prefix_rules:
            if value.startswith(prefix):
                rendered = replacement + value.removeprefix(prefix)
                break
    forbidden = re.compile(r"\b(?:Agent|agent|should)\b|Check that")
    if forbidden.search(rendered):
        raise V6GenerationError(
            f"unsupported meta-level completion assertion: {assertion}"
        )
    return _capitalize_sentence(rendered)


def _ensure_terminal_punctuation(value: str) -> str:
    value = value.strip()
    if not value:
        raise V6GenerationError("completion value is empty")
    return value if value[-1] in ".!?" else value + "."


def explicit_user_direct_assertion(
    assertion: str, *, strip_leading_that: bool = False
) -> str:
    """Render one frozen assertion as an explicit statement to the user."""

    value = assertion.strip()
    tell_prefix = "Agent should tell the user "
    provide_prefix = "Agent should provide "
    if value.startswith(tell_prefix):
        fact = value.removeprefix(tell_prefix).strip()
        if strip_leading_that and fact.startswith("that "):
            fact = fact.removeprefix("that ")
        if not fact:
            raise V6GenerationError("tell-user completion assertion is empty")
        rendered = f"I am telling you directly: {fact}"
    elif value.startswith(provide_prefix):
        fact = value.removeprefix(provide_prefix).strip()
        if not fact:
            raise V6GenerationError("provide completion assertion is empty")
        rendered = f"I am providing this directly to you: {fact}"
    else:
        direct = naturalize_completion_assertion(value)
        rendered = f"I am confirming this directly to you: {direct}"
    return _ensure_terminal_punctuation(rendered)


def deterministic_completion_message(
    criteria: Any,
    *,
    renderer: str,
    fallback: str,
) -> dict[str, str]:
    communicate = [str(value) for value in (criteria.communicate_info or [])]
    assertions = [str(value) for value in (criteria.nl_assertions or [])]
    if renderer == "legacy_assertion_echo":
        facts = [*communicate, *assertions]
        content = "\n".join(facts) if facts else fallback
    elif renderer == "natural_direct_v1":
        lines = ["The requested work is complete."]
        lines.extend(f"Requested information: {value}." for value in communicate)
        lines.extend(naturalize_completion_assertion(value) for value in assertions)
        content = "\n".join(dict.fromkeys(lines))
    elif renderer == "explicit_user_direct_v2":
        lines = [
            _ensure_terminal_punctuation(
                "I am providing the requested information directly to you: "
                + value
            )
            for value in communicate
        ]
        lines.extend(explicit_user_direct_assertion(value) for value in assertions)
        if not lines:
            lines = [
                "I am confirming this directly to you: "
                + _ensure_terminal_punctuation(fallback)
            ]
        content = "\n".join(dict.fromkeys(lines))
    elif renderer == "explicit_user_direct_v3":
        lines = [
            _ensure_terminal_punctuation(
                "I am providing the requested information directly to you: "
                + value
            )
            for value in communicate
        ]
        lines.extend(
            explicit_user_direct_assertion(value, strip_leading_that=True)
            for value in assertions
        )
        if not lines:
            lines = [
                "I am confirming this directly to you: "
                + _ensure_terminal_punctuation(fallback)
            ]
        content = "\n".join(dict.fromkeys(lines))
    else:
        raise V6GenerationError(f"unsupported completion renderer: {renderer}")
    return {"role": "assistant", "content": content}


def call_semantics(call: Mapping[str, Any]) -> tuple[str, str, str]:
    requestor = str(call.get("requestor", "assistant"))
    return (
        str(call.get("name", "")),
        canonical(call.get("arguments")),
        requestor,
    )


def first_tool_call(
    values: Sequence[Mapping[str, Any]]
) -> tuple[int, dict[str, Any]] | None:
    index = first_assistant_tool_index(values)
    if index is None:
        return None
    calls = values[index].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise V6GenerationError("fresh recovery must serialize one tool call")
    return index, deepcopy(calls[0])


def successful_assistant_labels(
    values: Sequence[Mapping[str, Any]],
    *,
    include_text: bool = False,
    reject_failed_tools: bool = False,
) -> list[bool]:
    mask = [False] * len(values)
    for index, message in enumerate(values[:-1]):
        if message.get("role") != "assistant":
            continue
        if not message.get("tool_calls"):
            if include_text and message.get("content") not in (None, ""):
                mask[index] = True
            continue
        result = values[index + 1]
        if result.get("role") != "tool":
            raise V6GenerationError(
                "assistant tool call lacks its adjacent tool result"
            )
        if result.get("error") is True and reject_failed_tools:
            raise V6GenerationError(
                "fresh successful suffix contains an additional failed tool call"
            )
        if result.get("error") is False:
            mask[index] = True
    if (
        values
        and include_text
        and values[-1].get("role") == "assistant"
        and not values[-1].get("tool_calls")
        and values[-1].get("content") not in (None, "")
    ):
        mask[-1] = True
    return mask


def extract_fresh_suffix(
    observed: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if len(observed) < len(history):
        raise V6GenerationError("recovery simulation is shorter than its initial history")

    def comparable_prefix(
        values: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        normalized = semantic(list(values))
        if not isinstance(normalized, list):
            raise V6GenerationError("tau2 recovery history is malformed")
        for message in normalized:
            if not isinstance(message, dict):
                continue
            for key, default in DEFAULT_TAU2_MESSAGE_TRANSPORT_FIELDS.items():
                if key in message and message[key] == default:
                    message.pop(key)
        return normalized

    observed_prefix = comparable_prefix(observed[: len(history)])
    expected_prefix = comparable_prefix(history)
    if observed_prefix != expected_prefix:
        raise V6GenerationError("tau2 recovery history differs from frozen prompt")
    suffix = [deepcopy(dict(row)) for row in observed[len(history) :]]
    if not suffix and not allow_empty:
        raise V6GenerationError("fresh recovery suffix is empty")
    return suffix


def clean_rollout(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    task_identity: str,
    seed: int,
    log_root: Path,
    reference_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(args.clean_attempts):
        attempt_seed = seed + attempt
        single_turn_mode = (
            args.clean_agent_mode == "single_turn_user_reference_replay"
        )
        simulation, seconds = run_one(
            args,
            task=task,
            domain=domain,
            seed=attempt_seed,
            save_dir=log_root / f"clean-attempt-{attempt + 1:02d}",
            agent_name=(
                REFERENCE_GUIDED_CLEAN_AGENT_NAME
                if args.clean_agent_mode == "reference_guided"
                else AGENT_NAME
            ),
            max_steps=(
                SINGLE_TURN_PREFIX_MAX_STEPS if single_turn_mode else None
            ),
        )
        observed = messages(simulation)
        tool_index = first_assistant_tool_index(observed)
        if single_turn_mode:
            prefix = single_turn_clean_prefix(observed)
            simulation = deterministic_reference_clean_simulation(
                simulation=simulation,
                domain=domain,
                task=task,
                prefix=prefix,
                reference_action_indices=(
                    reference_plan["sanitized_successful_reference_indices"]
                    if reference_plan is not None
                    else None
                ),
                completion_renderer=getattr(
                    args, "completion_renderer", "legacy_assertion_echo"
                ),
            )
            observed = messages(simulation)
            tool_index = first_assistant_tool_index(observed)
        elif (
            args.clean_agent_mode == "deterministic_reference_replay"
            and tool_index is not None
        ):
            simulation = deterministic_reference_clean_simulation(
                simulation=simulation,
                domain=domain,
                task=task,
                prefix=observed[:tool_index],
                reference_action_indices=(
                    reference_plan["sanitized_successful_reference_indices"]
                    if reference_plan is not None
                    else None
                ),
                completion_renderer=getattr(
                    args, "completion_renderer", "legacy_assertion_echo"
                ),
            )
            observed = messages(simulation)
            tool_index = first_assistant_tool_index(observed)
        official_reward, official_reward_info = dynamic_official_reward(
            simulation,
            context=f"{task_identity}:clean-attempt-{attempt + 1}",
            require_success=False,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "official_reward": official_reward,
                "official_reward_info": official_reward_info,
                "first_assistant_tool_index": tool_index,
                "wall_seconds": seconds,
            }
        )
        if official_reward == 1.0 and tool_index is not None:
            prefix = observed[:tool_index]
            environment = initial_environment(domain, task, prefix)
            snapshot = database_hashes(environment)
            tool_schemas = environment_tool_schemas(environment)
            system = training_system_message(environment.get_policy())
            state = snapshot_state(task, prefix)
            clean_replay = independent_replay(
                domain=domain,
                task=task,
                full_messages=observed,
            )
            return {
                "task_identity": task_identity,
                "clean_seed": attempt_seed,
                "clean_messages": observed,
                "clean_messages_sha256": semantic_sha256(observed),
                "prefix": prefix,
                "prefix_sha256": sha256(prefix),
                "prefix_semantic_sha256": semantic_sha256(prefix),
                "environment_snapshot": {
                    "state": state,
                    "state_sha256": sha256(state),
                    **snapshot,
                },
                "environment_snapshot_sha256": sha256(state),
                "tool_schemas": tool_schemas,
                "tool_schemas_sha256": sha256(tool_schemas),
                "training_system_message": system,
                "training_system_message_sha256": sha256(system),
                "clean_replay": clean_replay,
                "reference_plan_binding": (
                    deepcopy(dict(reference_plan))
                    if reference_plan is not None
                    else None
                ),
                "attempts": attempts,
                "official_reward": 1.0,
                "official_reward_info": official_reward_info,
                "official_test_used": False,
            }
    raise V6GenerationError(
        f"{task_identity}: no successful clean rollout with an observed tool "
        f"call; attempts={canonical(attempts)}"
    )


def validate_error_injection(
    *,
    domain: str,
    task: Any,
    prefix: Sequence[Mapping[str, Any]],
    branch_slot: Mapping[str, Any],
) -> dict[str, Any]:
    injection = branch_slot.get("injection_spec")
    if not isinstance(injection, dict):
        raise V6GenerationError("branch lacks injection_spec")
    call = injection.get("error_call")
    if not isinstance(call, dict):
        raise V6GenerationError("branch lacks registered error_call")
    environment = initial_environment(domain, task, prefix)
    before = database_hashes(environment)
    call_message, result = execute_call(environment, call)
    after = database_hashes(environment)
    if result.get("error") is not True:
        raise V6GenerationError(
            f"{branch_slot.get('branch_id')}: injected call was not a real tool error"
        )
    if before != after:
        raise V6GenerationError(
            f"{branch_slot.get('branch_id')}: failed call changed environment state"
        )
    failed_event = [call_message, result]
    return {
        "failed_event": failed_event,
        "error_event_sha256": semantic_sha256(failed_event),
        "state_before_error": before,
        "state_after_error": after,
        "actual_tool_error": True,
        "state_unchanged_after_error": True,
    }


def independent_replay(
    *,
    domain: str,
    task: Any,
    full_messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = initial_environment(domain, task, full_messages)
    second = initial_environment(domain, task, full_messages)
    first_hashes = database_hashes(first)
    second_hashes = database_hashes(second)
    if first_hashes != second_hashes:
        raise V6GenerationError("independent environment replay produced state drift")
    return {
        "pass": True,
        "first_final_state": first_hashes,
        "second_final_state": second_hashes,
        "full_messages_sha256": semantic_sha256(full_messages),
    }


def execute_reference_actions(
    environment: Any,
    *,
    task_id: str | int,
    actions: Sequence[Any],
    action_indices: Sequence[int] | None = None,
    call_id_prefix: str = "v6-reference-clean",
) -> list[dict[str, Any]]:
    """Execute the frozen reference sequence, retaining expected tool errors.

    Tau2 reference trajectories can intentionally include a failed lookup
    followed by a corrected lookup.  Whether the complete trajectory is valid
    is decided by the frozen official evaluator, not by requiring every
    intermediate reference tool result to be successful.
    """
    if action_indices is None:
        action_indices = list(range(len(actions)))
    if len(action_indices) != len(actions):
        raise V6GenerationError("reference action/index cardinality mismatch")
    observed: list[dict[str, Any]] = []
    for index, action in zip(action_indices, actions):
        call = {
            "id": f"{call_id_prefix}-{task_id}-{index:03d}",
            "requestor": action.requestor,
            "name": action.name,
            "arguments": deepcopy(action.arguments),
        }
        call_message, result = execute_call(environment, call)
        observed.extend((call_message, result))
    return observed


def select_reference_action_plan(
    actions: Sequence[Any],
    action_indices: Sequence[int] | None,
) -> list[tuple[int, Any]]:
    """Resolve an exact-index plan without converting identity to call value."""

    if action_indices is None:
        return list(enumerate(actions))
    indices = list(action_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(actions)
        for index in indices
    ):
        raise V6GenerationError("reference plan contains an invalid action index")
    if len(indices) != len(set(indices)):
        raise V6GenerationError("reference plan contains duplicate action indices")
    if indices != sorted(indices):
        raise V6GenerationError(
            "sanitized reference plan must preserve original relative order"
        )
    if not indices:
        raise V6GenerationError("sanitized reference plan is empty")
    return [(index, actions[index]) for index in indices]


def deterministic_reference_clean_simulation(
    *,
    simulation: Any,
    domain: str,
    task: Any,
    prefix: Sequence[Mapping[str, Any]],
    reference_action_indices: Sequence[int] | None = None,
    completion_renderer: str = "legacy_assertion_echo",
) -> Any:
    """Replace the deleted clean future with a deterministic reference replay.

    The conversational prefix is retained only through the point before the
    first assistant tool action. Reference actions and evaluation text are
    confined to the clean future, which candidate construction deletes before
    generating any fresh recovery suffix.
    """
    from tau2.data_model.simulation import TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError("deterministic clean replay requires reference actions")
    environment = initial_environment(domain, task, prefix)
    observed = [deepcopy(dict(row)) for row in prefix]
    plan = select_reference_action_plan(
        criteria.actions,
        reference_action_indices,
    )
    observed.extend(
        execute_reference_actions(
            environment,
            task_id=task.id,
            actions=[action for _, action in plan],
            action_indices=[index for index, _ in plan],
            call_id_prefix="v6-reference-clean",
        )
    )
    observed.append(
        deterministic_completion_message(
            criteria,
            renderer=completion_renderer,
            fallback="The requested changes have been completed.",
        )
    )
    replayed = simulation.model_copy(
        update={
            "messages": parse_messages(observed),
            "reward_info": None,
            # A single-turn source intentionally ends at its max-step
            # boundary.  The reconstructed deterministic replay, however,
            # is a complete trajectory and must not inherit that premature
            # source termination marker.
            "termination_reason": TerminationReason.USER_STOP,
        }
    )
    replayed.reward_info = evaluate_simulation(
        simulation=replayed,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return replayed


def deterministic_reference_tail_simulation(
    *,
    domain: str,
    task: Any,
    prompt: Sequence[Mapping[str, Any]],
    forced_call: Mapping[str, Any],
    forced_reference_action_index: int | None = None,
    reference_action_indices: Sequence[int] | None = None,
    completion_renderer: str = "legacy_assertion_echo",
) -> tuple[Any, list[dict[str, Any]], float]:
    """Execute a frozen corrective action and the reference tail after it."""
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError("reference-tail recovery requires reference actions")
    forced_index = resolve_forced_reference_index(
        criteria.actions,
        forced_call,
        registered_index=forced_reference_action_index,
    )
    plan = select_reference_action_plan(
        criteria.actions,
        reference_action_indices,
    )
    plan_indices = [index for index, _ in plan]
    if forced_index not in plan_indices:
        raise V6GenerationError(
            "forced corrective action is absent from sanitized reference plan"
        )
    started = time.monotonic()
    environment = initial_environment(domain, task, prompt)
    suffix: list[dict[str, Any]] = []
    forced_message, forced_result = execute_call(environment, forced_call)
    if forced_result.get("error") is True:
        raise V6GenerationError("frozen corrective action returned a tool error")
    suffix.extend((forced_message, forced_result))
    tail = [(index, action) for index, action in plan if index > forced_index]
    suffix.extend(
        execute_reference_actions(
            environment,
            task_id=task.id,
            actions=[action for _, action in tail],
            action_indices=[index for index, _ in tail],
            call_id_prefix="v6-reference-tail",
        )
    )
    suffix.append(
        deterministic_completion_message(
            criteria,
            renderer=completion_renderer,
            fallback="The requested recovery has been completed.",
        )
    )
    full = [*deepcopy(list(prompt)), *deepcopy(suffix)]
    timestamp = "1970-01-01T00:00:00Z"
    simulation = SimulationRun(
        id=f"v6-reference-tail-{task.id}",
        task_id=str(task.id),
        timestamp=timestamp,
        start_time=timestamp,
        end_time=timestamp,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=parse_messages(full),
        seed=None,
    )
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return simulation, suffix, time.monotonic() - started


def resolve_forced_reference_index(
    actions: Sequence[Any],
    forced_call: Mapping[str, Any],
    *,
    registered_index: int | None = None,
) -> int:
    """Resolve the already-registered reference slot without value ambiguity.

    Some valid tau2 tasks repeat an identical read-only reference action.  The
    outcome-independent registry freezes the source action index precisely so
    runtime code does not need to re-identify that slot by call value.
    """
    if registered_index is not None:
        if (
            isinstance(registered_index, bool)
            or not isinstance(registered_index, int)
            or registered_index < 0
            or registered_index >= len(actions)
        ):
            raise V6GenerationError(
                "registered reference action index is out of range"
            )
        registered_call = actions[registered_index].model_dump(mode="json")
        if call_semantics(registered_call) != call_semantics(forced_call):
            raise V6GenerationError(
                "registered reference action index does not match forced call"
            )
        return registered_index
    matching_indices = [
        index
        for index, action in enumerate(actions)
        if call_semantics(action.model_dump(mode="json"))
        == call_semantics(forced_call)
    ]
    if len(matching_indices) != 1:
        raise V6GenerationError(
            "forced recovery action must match exactly one frozen reference action"
        )
    return matching_indices[0]


def deterministic_reference_completion_simulation(
    *,
    domain: str,
    task: Any,
    prompt: Sequence[Mapping[str, Any]],
    forced_call: Mapping[str, Any],
    forced_reference_action_index: int | None = None,
    reference_action_indices: Sequence[int] | None = None,
    completion_renderer: str = "legacy_assertion_echo",
) -> tuple[Any, list[dict[str, Any]], float]:
    """Force the correction, then execute every other reference action once.

    The remaining actions retain their original relative order.  This is the
    sole V6.5 scientific delta; unlike V6.4 it cannot silently omit required
    actions that precede the forced call in the frozen reference list.
    """
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError(
            "reference-completion recovery requires reference actions"
        )
    forced_index = resolve_forced_reference_index(
        criteria.actions,
        forced_call,
        registered_index=forced_reference_action_index,
    )
    plan = select_reference_action_plan(
        criteria.actions,
        reference_action_indices,
    )
    plan_indices = [index for index, _ in plan]
    if forced_index not in plan_indices:
        raise V6GenerationError(
            "forced corrective action is absent from sanitized reference plan"
        )
    started = time.monotonic()
    environment = initial_environment(domain, task, prompt)
    suffix: list[dict[str, Any]] = []
    forced_message, forced_result = execute_call(environment, forced_call)
    if forced_result.get("error") is True:
        raise V6GenerationError("frozen corrective action returned a tool error")
    suffix.extend((forced_message, forced_result))
    remaining = [
        (index, action) for index, action in plan if index != forced_index
    ]
    suffix.extend(
        execute_reference_actions(
            environment,
            task_id=task.id,
            actions=[action for _, action in remaining],
            action_indices=[index for index, _ in remaining],
            call_id_prefix="v6-reference-completion",
        )
    )
    suffix.append(
        deterministic_completion_message(
            criteria,
            renderer=completion_renderer,
            fallback="The requested recovery has been completed.",
        )
    )
    full = [*deepcopy(list(prompt)), *deepcopy(suffix)]
    timestamp = "1970-01-01T00:00:00Z"
    simulation = SimulationRun(
        id=f"v6-reference-completion-{task.id}",
        task_id=str(task.id),
        timestamp=timestamp,
        start_time=timestamp,
        end_time=timestamp,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=parse_messages(full),
        seed=None,
    )
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return simulation, suffix, time.monotonic() - started


def matched_recovery(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    prompt: list[dict[str, Any]],
    branch_slot: Mapping[str, Any],
    log_root: Path,
    reference_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    constructor = (
        branch_slot.get("corrective_action_spec") or {}
    ).get("forced_first_action_constructor")
    reference_call = (
        constructor.get("tool_call") if isinstance(constructor, dict) else None
    )
    if not isinstance(reference_call, dict):
        raise V6GenerationError("branch lacks registered corrective call")
    continuation_mode = matched_positive_mode(args)
    if continuation_mode in {
        "deterministic_reference_tail",
        "deterministic_reference_completion",
    }:
        deterministic_runner = (
            deterministic_reference_completion_simulation
            if continuation_mode == "deterministic_reference_completion"
            else deterministic_reference_tail_simulation
        )
        simulation, suffix, seconds = deterministic_runner(
            domain=domain,
            task=task,
            prompt=prompt,
            forced_call=reference_call,
            forced_reference_action_index=(
                branch_slot.get("corrective_action_spec") or {}
            ).get("reference_action_index"),
            reference_action_indices=(
                reference_plan["sanitized_successful_reference_indices"]
                if reference_plan is not None
                else None
            ),
            completion_renderer=getattr(
                args, "completion_renderer", "legacy_assertion_echo"
            ),
        )
        first = first_tool_call(suffix)
        matched = (
            first is not None
            and call_semantics(first[1]) == call_semantics(reference_call)
        )
        official_reward, official_reward_info = dynamic_official_reward(
            simulation,
            context=f"{branch_slot.get('branch_id')}:matched-positive",
            require_success=False,
        )
        attempt = {
            "attempt": 1,
            "seed": int(branch_slot["recovery_seed"]),
            "official_reward": official_reward,
            "official_reward_info": official_reward_info,
            "wall_seconds": seconds,
            "first_action": deepcopy(first[1]) if first is not None else None,
            "first_action_matched": matched,
            "continuation_mode": continuation_mode,
        }
        if official_reward != 1.0 or not matched:
            raise V6GenerationError(
                f"{branch_slot.get('branch_id')}: {continuation_mode} "
                "did not produce a successful matched recovery; "
                f"attempt={canonical(attempt)}"
            )
        # V6.10 labels only the verified sanitized successful plan.  Legacy
        # protocols retain raw expected-error slots as context-only events.
        mask = successful_assistant_labels(
            suffix,
            include_text=True,
            reject_failed_tools=reference_plan is not None,
        )
        if not any(mask):
            raise V6GenerationError(
                "deterministic matched recovery has no successful label"
            )
        full = [*deepcopy(prompt), *deepcopy(suffix)]
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=full,
        )
        return {
            "recovery_seed": int(branch_slot["recovery_seed"]),
            "prompt": deepcopy(prompt),
            "prompt_sha256": semantic_sha256(prompt),
            "fresh_recovery_suffix": suffix,
            "fresh_recovery_suffix_sha256": semantic_sha256(suffix),
            "full_recovery_messages": full,
            "label_mask": [False] * len(prompt) + mask,
            "failed_positive_labels": 0,
            "first_recovery_action": first[1],
            "first_recovery_action_key": (
                f"{first[1]['name']}::{branch_slot['identifier_key']}"
            ),
            "official_task_success": 1.0,
            "official_reward_info": official_reward_info,
            "continuation_mode": continuation_mode,
            "fresh_recovery_generated": False,
            "gold_suffix_used": True,
            "reference_plan_binding": (
                deepcopy(dict(reference_plan))
                if reference_plan is not None
                else None
            ),
            "independent_replay": replay,
            "attempts": [attempt],
        }
    attempts: list[dict[str, Any]] = []
    base_seed = int(branch_slot["recovery_seed"])
    for attempt in range(args.recovery_attempts):
        attempt_seed = base_seed + attempt
        simulation, seconds = run_one(
            args,
            task=make_recovery_task(task, prompt),
            domain=domain,
            seed=attempt_seed,
            save_dir=log_root / f"recovery-attempt-{attempt + 1:02d}",
        )
        observed = messages(simulation)
        try:
            suffix = extract_fresh_suffix(observed, prompt)
        except V6GenerationError as error:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "official_reward": reward(simulation),
                    "wall_seconds": seconds,
                    "status": "HISTORY_DRIFT",
                    "first_action": None,
                    "first_action_matched": False,
                    "error": str(error),
                }
            )
            continue
        first = first_tool_call(suffix)
        matched = (
            first is not None
            and call_semantics(first[1]) == call_semantics(reference_call)
        )
        official_reward, official_reward_info = dynamic_official_reward(
            simulation,
            context=(
                f"{branch_slot.get('branch_id')}:"
                f"matched-attempt-{attempt + 1}"
            ),
            require_success=False,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "official_reward": official_reward,
                "official_reward_info": official_reward_info,
                "wall_seconds": seconds,
                "first_action": (
                    deepcopy(first[1]) if first is not None else None
                ),
                "first_action_matched": matched,
            }
        )
        if official_reward != 1.0 or not matched:
            continue
        mask = successful_assistant_labels(
            suffix, include_text=True, reject_failed_tools=True
        )
        if not any(mask):
            raise V6GenerationError("successful recovery suffix has no successful tool label")
        full = [*deepcopy(prompt), *deepcopy(suffix)]
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=full,
        )
        return {
            "recovery_seed": attempt_seed,
            "prompt": deepcopy(prompt),
            "prompt_sha256": semantic_sha256(prompt),
            "fresh_recovery_suffix": suffix,
            "fresh_recovery_suffix_sha256": semantic_sha256(suffix),
            "full_recovery_messages": full,
            "label_mask": [False] * len(prompt) + mask,
            "failed_positive_labels": 0,
            "first_recovery_action": first[1],
            "first_recovery_action_key": (
                f"{first[1]['name']}::{branch_slot['identifier_key']}"
            ),
            "official_task_success": 1.0,
            "official_reward_info": official_reward_info,
            "continuation_mode": "fresh_teacher",
            "fresh_recovery_generated": True,
            "gold_suffix_used": False,
            "independent_replay": replay,
            "attempts": attempts,
        }
    raise V6GenerationError(
        f"{branch_slot.get('branch_id')}: no fresh successful recovery whose "
        "first action matched the registered corrective call"
    )


def raw_teacher_first_response(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    error_prompt: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Make exactly one raw teacher call, bypassing V6Agent normalization."""

    from tau2.agent.llm_agent import LLMAgent

    if not error_prompt:
        raise V6GenerationError("unforced probe error context is empty")
    environment = initial_environment(domain, task, error_prompt)
    history = parse_messages(error_prompt)
    llm_args = endpoint_args(
        args.teacher_api_base,
        "v6-local",
        max_tokens=args.max_tokens,
        seed=seed,
        temperature=DECODING_TEMPERATURE,
        top_p=DECODING_TOP_P,
    )
    # tau2 silently substitutes DEFAULT_MAX_RETRIES when this key is absent.
    # The causal measurement is one raw query, so provider-level retries must
    # be disabled in the actual request rather than merely reported as zero.
    llm_args["num_retries"] = 0
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=litellm_openai_model(args.teacher_model),
        llm_args=llm_args,
    )
    state = agent.get_init_state(message_history=history[:-1])
    started = time.monotonic()
    try:
        response, _ = agent.generate_next_message(history[-1], state)
        raw = response.model_dump(mode="json")
        if not isinstance(raw, dict):
            raise TypeError("model_dump(mode='json') did not return an object")
    except Exception as error:
        seconds = time.monotonic() - started
        # JSON/tool-argument parsing and response-shape failures are observed
        # model outcomes for this one-query probe.  Preserve them as negative
        # evidence.  Provider/transport failures remain typed global failures
        # so they cannot be silently counted as model mistakes.
        if isinstance(
            error,
            (
                json.JSONDecodeError,
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                AttributeError,
                AssertionError,
            ),
        ):
            message = str(error)
            evidence: dict[str, Any] = {
                "kind": "MALFORMED_MODEL_RESPONSE",
                "exception_module": type(error).__module__,
                "exception_type": type(error).__name__,
                "exception_message": message,
                "exception_message_sha256": sha256(message),
            }
            raw_json = getattr(error, "doc", None)
            if isinstance(raw_json, str):
                evidence.update(
                    {
                        "raw_json": raw_json,
                        "raw_json_sha256": sha256(raw_json),
                        "json_error_position": getattr(error, "pos", None),
                    }
                )
            raise V6RawProbeMalformedResponse(
                evidence=evidence,
                wall_seconds=seconds,
            ) from error
        raise V6RawProbeInfrastructureError(
            "raw teacher first-response request failed before a scoreable "
            f"response existed ({type(error).__module__}."
            f"{type(error).__name__})"
        ) from error
    seconds = time.monotonic() - started
    return raw, seconds


def teacher_unforced_first_action_measurement(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    error_prompt: list[dict[str, Any]],
    expected_corrective_call: Mapping[str, Any],
    measurement_seed: int,
    log_root: Path,
) -> dict[str, Any]:
    """Score the one raw assistant response; never normalize or continue."""

    try:
        raw_response, seconds = raw_teacher_first_response(
            args,
            task=task,
            domain=domain,
            error_prompt=error_prompt,
            seed=measurement_seed,
        )
    except V6RawProbeMalformedResponse as error:
        seconds = error.wall_seconds
        # Keep the exact parser evidence inside the hashed raw-response field
        # so the existing audit schema can recompute it without granting a
        # malformed response a valid tool-call shape.
        raw_response = {
            "role": None,
            "content": None,
            "tool_calls": None,
            "malformed_response_evidence": deepcopy(error.evidence),
        }
    calls = raw_response.get("tool_calls")
    valid_single_tool_call = (
        raw_response.get("role") == "assistant"
        and raw_response.get("content") in (None, "")
        and isinstance(calls, list)
        and len(calls) == 1
        and isinstance(calls[0], Mapping)
    )
    first_action = (
        deepcopy(dict(calls[0])) if valid_single_tool_call else None
    )
    matched = (
        first_action is not None
        and call_semantics(first_action)
        == call_semantics(expected_corrective_call)
    )
    trial = {
        "seed": measurement_seed,
        "generation_count": 1,
        "task_success": None,
        "task_success_status": "NOT_MEASURED_FIRST_RESPONSE_ONLY",
        "official_reward_info": None,
        "first_action_observed": first_action is not None,
        "first_action": first_action,
        "first_action_matched_registered_corrective": matched,
        "first_response_status": (
            "VALID_SINGLE_TOOL_CALL"
            if first_action is not None
            else "INCORRECT_OR_MALFORMED"
        ),
        "malformed_reason": (
            None
            if first_action is not None
            else "RAW_FIRST_RESPONSE_NOT_ONE_TOOL_ONLY_CALL"
        ),
        "first_action_semantics_sha256": (
            sha256(call_semantics(first_action))
            if first_action is not None
            else None
        ),
        "raw_response": deepcopy(raw_response),
        "raw_response_sha256": semantic_sha256(raw_response),
        "normalization_applied": False,
        "normalization_forbidden": True,
        "user_continuation_generated": False,
        "judge_invoked": False,
        "full_rollout_generated": False,
        "retry_count": 0,
        "wall_seconds": seconds,
    }
    result = {
        "mode": "teacher_unforced_raw_first_response",
        "forced_first": False,
        "gold_suffix_visible": False,
        "fresh_recovery_generated": False,
        "raw_first_response_generated": True,
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "measurement_seed": measurement_seed,
        "measurement_seed_sha256": sha256(measurement_seed),
        "trial_count": 1,
        "generation_count": 1,
        "matched_registered_corrective_accuracy": float(matched),
        "normalization_applied": False,
        "normalization_forbidden": True,
        "task_success_status": "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE",
        "trials": [trial],
        "official_test_used": False,
    }
    write_json(log_root / "raw_first_response_probe.json", result)
    return result


def forced_first_cell(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    error_prompt: list[dict[str, Any]],
    forced_call: Mapping[str, Any],
    forced_reference_action_index: int | None,
    continuation_seeds: Sequence[int],
    log_root: Path,
    cell_name: str,
    continuation_mode: str | None = None,
    reference_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    effective_mode = continuation_mode or causal_cell_mode(args)
    if reference_plan is not None and effective_mode != "fresh_teacher":
        raise V6GenerationError(
            f"{cell_name}: V6.10 causal continuation must be fresh_teacher"
        )
    for seed in continuation_seeds:
        if effective_mode in {
            "deterministic_reference_tail",
            "deterministic_reference_completion",
        }:
            deterministic_runner = (
                deterministic_reference_completion_simulation
                if effective_mode == "deterministic_reference_completion"
                else deterministic_reference_tail_simulation
            )
            simulation, continuation, seconds = (
                deterministic_runner(
                    domain=domain,
                    task=task,
                    prompt=error_prompt,
                    forced_call=forced_call,
                    forced_reference_action_index=(
                        forced_reference_action_index
                    ),
                    completion_renderer=getattr(
                        args, "completion_renderer", "legacy_assertion_echo"
                    ),
                )
            )
            official_reward, official_reward_info = dynamic_official_reward(
                simulation,
                context=f"{cell_name}:seed-{seed}",
                require_success=False,
            )
            full_messages = [
                *deepcopy(error_prompt),
                *deepcopy(continuation),
            ]
            replay = independent_replay(
                domain=domain,
                task=task,
                full_messages=full_messages,
            )
            trials.append(
                {
                    "seed": seed,
                    "task_success": official_reward,
                    "official_reward_info": official_reward_info,
                    "forced_call": deepcopy(dict(forced_call)),
                    "forced_result": continuation[1],
                    "continuation_messages": continuation[2:],
                    "continuation_sha256": semantic_sha256(continuation[2:]),
                    "independent_replay": replay,
                    "wall_seconds": seconds,
                }
            )
            continue
        environment = initial_environment(domain, task, error_prompt)
        call_message, tool_result = execute_call(environment, forced_call)
        if tool_result.get("error") is True:
            raise V6GenerationError(
                f"{cell_name}: forced corrective action returned a tool error"
            )
        forced_history = [
            *deepcopy(error_prompt),
            call_message,
            tool_result,
        ]
        simulation, seconds = run_one(
            args,
            task=make_recovery_task(task, forced_history),
            domain=domain,
            seed=seed,
            save_dir=log_root / cell_name / f"seed-{seed}",
        )
        official_reward, official_reward_info = dynamic_official_reward(
            simulation,
            context=f"{cell_name}:seed-{seed}",
            require_success=False,
        )
        observed = messages(simulation)
        # The forced corrective call/result already defines this causal cell.
        # A teacher that stops immediately afterwards is a valid observed
        # outcome, not missing evidence.  Positive recovery construction still
        # uses the default non-empty requirement and must contain label tokens.
        continuation = extract_fresh_suffix(
            observed,
            forced_history,
            allow_empty=True,
        )
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=observed,
        )
        trials.append(
            {
                "seed": seed,
                "task_success": official_reward,
                "official_reward_info": official_reward_info,
                "forced_call": deepcopy(dict(forced_call)),
                "forced_result": tool_result,
                "continuation_messages": continuation,
                "continuation_sha256": semantic_sha256(continuation),
                "independent_replay": replay,
                "wall_seconds": seconds,
            }
        )
    decoding = {
        "temperature": DECODING_TEMPERATURE,
        "top_p": DECODING_TOP_P,
        "max_tokens": args.max_tokens,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
    }
    policy = {
        "agent": (
            effective_mode
            if effective_mode
            in {
                "deterministic_reference_tail",
                "deterministic_reference_completion",
            }
            else AGENT_NAME
        ),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
    }
    forced_reference_slot_id = None
    task_preflight_sha256 = None
    sanitized_reference_plan_sha256 = None
    if reference_plan is not None:
        slot_ids = reference_plan.get("reference_slot_ids_by_index")
        if not isinstance(slot_ids, Mapping):
            raise V6GenerationError(
                f"{cell_name}: verified reference plan lacks slot identities"
            )
        forced_reference_slot_id = slot_ids.get(
            str(forced_reference_action_index)
        )
        task_preflight_sha256 = reference_plan.get("task_preflight_sha256")
        sanitized_reference_plan_sha256 = reference_plan.get(
            "sanitized_successful_plan_sha256"
        )
        if (
            not isinstance(forced_reference_slot_id, str)
            or not forced_reference_slot_id
            or not isinstance(task_preflight_sha256, str)
            or SHA256_RE.fullmatch(task_preflight_sha256) is None
            or SHA256_RE.fullmatch(
                str(sanitized_reference_plan_sha256)
            )
            is None
        ):
            raise V6GenerationError(
                f"{cell_name}: exact preflight slot binding is invalid"
            )
    return {
        "task_success": (
            sum(float(trial["task_success"]) for trial in trials) / len(trials)
        ),
        "forced_first_only": True,
        "forced_reference_action_index": forced_reference_action_index,
        "forced_reference_slot_id": forced_reference_slot_id,
        "reference_task_preflight_sha256": task_preflight_sha256,
        "sanitized_reference_plan_sha256": (
            sanitized_reference_plan_sha256
        ),
        "reference_preflight_receipt_sha256": (
            reference_plan.get("reference_preflight_receipt_sha256")
            if reference_plan is not None
            else None
        ),
        "reference_preflight_file_sha256": (
            reference_plan.get("reference_preflight_file_sha256")
            if reference_plan is not None
            else None
        ),
        "continuation_mode": effective_mode,
        "gold_suffix_visible": effective_mode
        in {
            "deterministic_reference_tail",
            "deterministic_reference_completion",
        },
        "fresh_recovery_generated": effective_mode == "fresh_teacher",
        "gold_access_audit": {
            "gold_reference_actions_visible_to_continuation": effective_mode
            in {
                "deterministic_reference_tail",
                "deterministic_reference_completion",
            },
            "evaluation_criteria_visible_to_continuation": effective_mode
            in {
                "deterministic_reference_tail",
                "deterministic_reference_completion",
            },
        },
        "continuation_policy_sha256": sha256(policy),
        "continuation_seed_set_sha256": sha256(list(continuation_seeds)),
        "decoding_sha256": sha256(decoding),
        "rollout_budget": args.max_steps,
        "independent_replay_pass": all(
            trial["independent_replay"]["pass"] is True for trial in trials
        ),
        "trials": trials,
    }


def corrective_call(branch_slot: Mapping[str, Any]) -> dict[str, Any]:
    constructor = (
        branch_slot.get("corrective_action_spec") or {}
    ).get("forced_first_action_constructor")
    call = constructor.get("tool_call") if isinstance(constructor, dict) else None
    if not isinstance(call, dict):
        raise V6GenerationError("branch corrective call is missing")
    return deepcopy(call)


def corrective_reference_index(branch_slot: Mapping[str, Any]) -> int:
    value = (branch_slot.get("corrective_action_spec") or {}).get(
        "reference_action_index"
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V6GenerationError(
            f"{branch_slot.get('branch_id')}: missing exact reference index"
        )
    return value


def validate_pair_against_reference_plan(
    registered_pair: Mapping[str, Any],
    reference_plan: Mapping[str, Any],
    *,
    task: Any,
) -> None:
    """Require every positive/cell action to be authorized by the preflight."""

    task_identity = str(registered_pair.get("task_identity"))
    if reference_plan.get("task_identity") != task_identity:
        raise V6GenerationError(
            f"{task_identity}: reference-plan task identity drift"
        )
    if (
        registered_pair.get("reference_task_preflight_sha256")
        != reference_plan.get("task_preflight_sha256")
        or registered_pair.get("reference_preflight_receipt_sha256")
        != reference_plan.get("reference_preflight_receipt_sha256")
        or registered_pair.get("reference_preflight_file_sha256")
        != reference_plan.get("reference_preflight_file_sha256")
        or registered_pair.get("sanitized_reference_plan_sha256")
        != reference_plan.get("sanitized_successful_plan_sha256")
    ):
        raise V6GenerationError(
            f"{task_identity}: executable registry is not bound to the "
            "verified task preflight"
        )
    sanitized = reference_plan.get("sanitized_successful_reference_indices")
    eligible = reference_plan.get("eligible_forced_first_reference_indices")
    expected_errors = reference_plan.get("expected_error_indices")
    if (
        not isinstance(sanitized, list)
        or not isinstance(eligible, list)
        or not isinstance(expected_errors, list)
    ):
        raise V6GenerationError(
            f"{task_identity}: verified reference plan lacks exact-index sets"
        )
    select_reference_action_plan(task.evaluation_criteria.actions, sanitized)
    if set(sanitized) & set(expected_errors):
        raise V6GenerationError(
            f"{task_identity}: sanitized plan contains an expected-error slot"
        )
    for branch_slot in registered_pair["branches"]:
        index = corrective_reference_index(branch_slot)
        if index not in sanitized or index not in eligible:
            raise V6GenerationError(
                f"{branch_slot.get('branch_id')}: corrective action was not "
                "authorized by the verified sanitized preflight"
            )
        binding = branch_slot.get("reference_preflight_binding")
        expected_slot = reference_plan["reference_slots_by_index"].get(
            str(index)
        )
        expected_binding = {
            "reference_preflight_receipt_sha256": reference_plan[
                "reference_preflight_receipt_sha256"
            ],
            "reference_preflight_file_sha256": reference_plan[
                "reference_preflight_file_sha256"
            ],
            "reference_task_preflight_sha256": reference_plan[
                "task_preflight_sha256"
            ],
            "sanitized_reference_plan_sha256": reference_plan[
                "sanitized_successful_plan_sha256"
            ],
            **(expected_slot or {}),
        }
        if binding != expected_binding:
            raise V6GenerationError(
                f"{branch_slot.get('branch_id')}: executable registry slot "
                "binding differs from verified preflight"
            )


def materialize_pair(
    args: argparse.Namespace,
    *,
    registered_pair: Mapping[str, Any],
    clean: Mapping[str, Any],
    task: Any,
    domain: str,
    continuation_seeds: Sequence[int],
    log_root: Path,
    registry_sha256: str,
    semantic_contract: Mapping[str, Any],
    reference_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if reference_plan is not None:
        validate_pair_against_reference_plan(
            registered_pair,
            reference_plan,
            task=task,
        )
        if causal_cell_mode(args) != "fresh_teacher":
            raise V6GenerationError(
                "V6.10 causal cells must use a gold-free fresh teacher"
            )
    observed_branches: list[dict[str, Any]] = []
    for branch_slot in registered_pair["branches"]:
        injection = validate_error_injection(
            domain=domain,
            task=task,
            prefix=clean["prefix"],
            branch_slot=branch_slot,
        )
        prompt = [*deepcopy(clean["prefix"]), *injection["failed_event"]]
        recovery = matched_recovery(
            args,
            task=task,
            domain=domain,
            prompt=prompt,
            branch_slot=branch_slot,
            log_root=log_root / str(branch_slot["branch_id"]).replace(":", "_"),
            reference_plan=reference_plan,
        )
        unforced_measurement = None
        if first_action_measurement_mode(args) == "teacher_unforced":
            unforced_measurement = teacher_unforced_first_action_measurement(
                args,
                task=task,
                domain=domain,
                error_prompt=prompt,
                expected_corrective_call=corrective_call(branch_slot),
                measurement_seed=int(branch_slot["recovery_seed"]),
                log_root=(
                    log_root
                    / str(branch_slot["branch_id"]).replace(":", "_")
                    / "teacher-unforced-first-action"
                ),
            )
        coverage = {
            "domain": domain,
            "failed_tool": str(branch_slot["tool_name"]),
            "error_family": (
                f"{branch_slot['tool_name']}::{branch_slot['identifier_key']}"
            ),
            "corrective_action": str(branch_slot["corrective_family"]),
            "recovery_length_bin": (
                "short"
                if len(recovery["fresh_recovery_suffix"]) <= 4
                else "medium"
                if len(recovery["fresh_recovery_suffix"]) <= 10
                else "long"
            ),
            "recovery_mode": (
                "user_assisted"
                if any(
                    row.get("role") == "user"
                    for row in recovery["fresh_recovery_suffix"]
                )
                else "agent_initiated"
            ),
        }
        observed_branches.append(
            {
                "branch_id": branch_slot["branch_id"],
                "registered_branch_slot_sha256": branch_slot[
                    "branch_slot_sha256"
                ],
                "reference_preflight_binding": (
                    deepcopy(
                        branch_slot.get("reference_preflight_binding")
                    )
                    if reference_plan is not None
                    else None
                ),
                "reference_action_index": (
                    corrective_reference_index(branch_slot)
                    if reference_plan is not None
                    else None
                ),
                "reference_slot_id": (
                    reference_plan["reference_slot_ids_by_index"].get(
                        str(corrective_reference_index(branch_slot))
                    )
                    if reference_plan is not None
                    else None
                ),
                "sanitized_reference_plan_sha256": (
                    reference_plan["sanitized_successful_plan_sha256"]
                    if reference_plan is not None
                    else None
                ),
                "shared_prefix_sha256": clean["prefix_sha256"],
                "environment_snapshot_sha256": clean[
                    "environment_snapshot_sha256"
                ],
                "error_event_messages": injection["failed_event"],
                "error_event_sha256": sha256(injection["failed_event"]),
                "recovery_prompt": recovery["prompt"],
                "recovery_suffix": recovery["fresh_recovery_suffix"],
                "full_trace": recovery["full_recovery_messages"],
                "full_trace_sha256": sha256(
                    recovery["full_recovery_messages"]
                ),
                "supervised_messages": recovery["fresh_recovery_suffix"],
                "label_mask": recovery["label_mask"],
                "database_hashes": {
                    "agent_before_error": injection["state_before_error"][
                        "agent_db_hash"
                    ],
                    "agent_after_error": injection["state_after_error"][
                        "agent_db_hash"
                    ],
                    "user_before_error": injection["state_before_error"][
                        "user_db_hash"
                    ],
                    "user_after_error": injection["state_after_error"][
                        "user_db_hash"
                    ],
                },
                "tool_execution_evidence": {
                    "executed_in_pinned_environment": True,
                    "tau2_commit": protocol.TAU2_COMMIT,
                    "tool_call_sha256": sha256(
                        {
                            "name": branch_slot["injection_spec"][
                                "error_call"
                            ]["name"],
                            "arguments": branch_slot["injection_spec"][
                                "error_call"
                            ]["arguments"],
                        }
                    ),
                    "tool_result_sha256": sha256(
                        injection["failed_event"][1]
                    ),
                },
                "label_audit": {
                    "future_message_overlap_count": 0,
                    "clean_future_visible": False,
                    "failed_positive_label_count": 0,
                    "error_result_positive_label_count": 0,
                },
                "matched_replay": {
                    "official_task_success": recovery[
                        "official_task_success"
                    ],
                    "official_reward_info": recovery[
                        "official_reward_info"
                    ],
                    "independent_replay_pass": recovery[
                        "independent_replay"
                    ]["pass"],
                    "final_state_valid": (
                        recovery["independent_replay"][
                            "first_final_state"
                        ]
                        == clean["clean_replay"]["first_final_state"]
                    ),
                    "final_agent_db_hash": recovery[
                        "independent_replay"
                    ]["first_final_state"]["agent_db_hash"],
                    "final_user_db_hash": recovery[
                        "independent_replay"
                    ]["first_final_state"]["user_db_hash"],
                    "independent_replay_audit_sha256": sha256(
                        recovery["independent_replay"]
                    ),
                },
                "teacher_unforced_first_action_measurement": (
                    unforced_measurement
                ),
                "official_test_used": False,
                # Preserve producer evidence for later tokenization and
                # qualitative auditing; acceptance is still recomputed.
                "producer_evidence": {
                    "registered_branch": deepcopy(branch_slot),
                    "matched_recovery": recovery,
                    "injection": injection,
                },
                "coverage": coverage,
                "tool_schemas": deepcopy(clean["tool_schemas"]),
                "supervised_target_tokens": None,
                "nonpadding_tokens": None,
                "token_measurement_status": "PENDING_FROZEN_STUDENT_TOKENIZER",
            }
        )
        if (
            observed_branches[-1]["matched_replay"]["final_state_valid"]
            is not True
        ):
            raise V6GenerationError(
                f"{branch_slot['branch_id']}: recovered final state differs "
                "from the successful clean trajectory"
            )

    first_call = corrective_call(registered_pair["branches"][0])
    second_call = corrective_call(registered_pair["branches"][1])
    first_reference_index = (
        registered_pair["branches"][0].get("corrective_action_spec") or {}
    ).get("reference_action_index")
    second_reference_index = (
        registered_pair["branches"][1].get("corrective_action_spec") or {}
    ).get("reference_action_index")
    if call_semantics(first_call) == call_semantics(second_call):
        raise V6GenerationError(
            f"{registered_pair['candidate_pair_id']}: two registered first "
            "actions are semantically identical, so kappa is not identifiable"
        )
    forced_first_q = {
        "q_e1_a1": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[0]["recovery_prompt"],
            forced_call=first_call,
            forced_reference_action_index=first_reference_index,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e1_a1",
            continuation_mode=causal_cell_mode(args),
            reference_plan=reference_plan,
        ),
        "q_e1_a2": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[0]["recovery_prompt"],
            forced_call=second_call,
            forced_reference_action_index=second_reference_index,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e1_a2",
            continuation_mode=causal_cell_mode(args),
            reference_plan=reference_plan,
        ),
        "q_e2_a1": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[1]["recovery_prompt"],
            forced_call=first_call,
            forced_reference_action_index=first_reference_index,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e2_a1",
            continuation_mode=causal_cell_mode(args),
            reference_plan=reference_plan,
        ),
        "q_e2_a2": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[1]["recovery_prompt"],
            forced_call=second_call,
            forced_reference_action_index=second_reference_index,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e2_a2",
            continuation_mode=causal_cell_mode(args),
            reference_plan=reference_plan,
        ),
    }
    clean_labels = successful_assistant_labels(
        clean["clean_messages"],
        include_text=True,
        reject_failed_tools=reference_plan is not None,
    )
    clean_labels = [
        selected and index >= len(clean["prefix"])
        for index, selected in enumerate(clean_labels)
    ]
    if not any(clean_labels):
        raise V6GenerationError("clean trajectory has no successful assistant tool label")
    pair = {
        "protocol": GENERATION_PROTOCOL,
        "registry_sha256": registry_sha256,
        "registered_candidate_pair_sha256": registered_pair[
            "candidate_pair_sha256"
        ],
        "reference_preflight_receipt_sha256": (
            reference_plan["reference_preflight_receipt_sha256"]
            if reference_plan is not None
            else None
        ),
        "reference_preflight_file_sha256": (
            reference_plan["reference_preflight_file_sha256"]
            if reference_plan is not None
            else None
        ),
        "reference_task_preflight_sha256": (
            reference_plan["task_preflight_sha256"]
            if reference_plan is not None
            else None
        ),
        "sanitized_reference_plan_sha256": (
            reference_plan["sanitized_successful_plan_sha256"]
            if reference_plan is not None
            else None
        ),
        "candidate_pair_id": registered_pair["candidate_pair_id"],
        "choice_set_id": registered_pair["choice_set_id"],
        "phase": registered_pair["phase"],
        "partition": "arm_train",
        "task_identity": registered_pair["task_identity"],
        "domain": domain,
        "task_id": registered_pair["task_id"],
        "registered_prefix_spec_sha256": registered_pair["prefix_sha256"],
        "registered_snapshot_spec_sha256": registered_pair[
            "environment_snapshot_sha256"
        ],
        "shared_prefix": deepcopy(clean["prefix"]),
        "shared_prefix_sha256": clean["prefix_sha256"],
        "environment_snapshot": deepcopy(clean["environment_snapshot"]),
        "clean_trace": {
            "messages": deepcopy(clean["clean_messages"]),
            "trace_sha256": sha256(clean["clean_messages"]),
            "official_task_success": 1.0,
            "official_reward_info": deepcopy(
                clean["official_reward_info"]
            ),
            "sanitized_reference_plan_sha256": (
                reference_plan["sanitized_successful_plan_sha256"]
                if reference_plan is not None
                else None
            ),
            "official_test_used": False,
            "final_state_valid": True,
            "final_agent_db_hash": clean["clean_replay"][
                "first_final_state"
            ]["agent_db_hash"],
            "final_user_db_hash": clean["clean_replay"][
                "first_final_state"
            ]["user_db_hash"],
            "independent_replay_audit_sha256": sha256(
                clean["clean_replay"]
            ),
        },
        "prefix_sha256": clean["prefix_sha256"],
        "environment_snapshot_sha256": clean[
            "environment_snapshot_sha256"
        ],
        "branches": observed_branches,
        "tool_schemas": deepcopy(clean["tool_schemas"]),
        "training_system_message": deepcopy(
            clean["training_system_message"]
        ),
        "training_system_message_sha256": clean[
            "training_system_message_sha256"
        ],
        "forced_first_cells": forced_first_q,
        "forced_first_q": forced_first_q,
        "continuation_role_separation": {
            "matched_positive_continuation_mode": matched_positive_mode(args),
            "causal_cell_continuation_mode": causal_cell_mode(args),
            "first_action_measurement_mode": first_action_measurement_mode(args),
            "causal_cells_gold_free": all(
                cell.get("gold_suffix_visible") is False
                and cell.get("fresh_recovery_generated") is True
                for cell in forced_first_q.values()
            ),
        },
        "reference_plan_binding": (
            deepcopy(dict(reference_plan))
            if reference_plan is not None
            else None
        ),
        "first_action_logprobs": None,
        "token_accounting": {
            "supervised_target_tokens": None,
            "nonpadding_tokens": None,
            "status": "PENDING_FROZEN_STUDENT_TOKENIZER",
        },
        "quality": {
            "real_error_executed": True,
            "matched_recovery_replay_success": True,
            "cross_replay_complete": True,
            "no_future_leakage": True,
            "independent_replay_audited": True,
            "official_test_used": False,
            "failed_positive_labels": 0,
            "dynamic_full_official_evaluation_complete": True,
        },
        "clean_view": {
            "clean_id": f"{registered_pair['task_identity']}:clean",
            "messages": deepcopy(clean["clean_messages"]),
            "label_mask": clean_labels,
            "tool_schemas": deepcopy(clean["tool_schemas"]),
            "official_task_success": 1.0,
            "official_reward_info": deepcopy(
                clean["official_reward_info"]
            ),
            "sanitized_reference_plan_sha256": (
                reference_plan["sanitized_successful_plan_sha256"]
                if reference_plan is not None
                else None
            ),
            "official_test_used": False,
            "source_trajectory_sha256": clean["clean_messages_sha256"],
            "c_sup": None,
            "c_nonpad": None,
            "token_measurement_status": "PENDING_FROZEN_STUDENT_TOKENIZER",
        },
        "generation_contract": deepcopy(dict(semantic_contract)),
        "generation_contract_sha256": sha256(dict(semantic_contract)),
        "official_test_used": False,
    }
    pair["generated_candidate_pair_sha256"] = sha256(pair)
    return pair


def materialize_task_pairs(
    args: argparse.Namespace,
    *,
    registered_pairs: Sequence[Mapping[str, Any]],
    clean: Mapping[str, Any],
    task: Any,
    domain: str,
    continuation_seeds: Sequence[int],
    task_log_root: Path,
    registry_sha256: str,
    semantic_contract: Mapping[str, Any],
    reference_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Reject every cheap task-local defect before the first teacher rollout."""

    ordered = sorted(
        registered_pairs,
        key=lambda item: str(item["candidate_pair_id"]),
    )
    if reference_plan is not None:
        for registered_pair in ordered:
            validate_pair_against_reference_plan(
                registered_pair,
                reference_plan,
                task=task,
            )
            for branch_slot in registered_pair["branches"]:
                validate_error_injection(
                    domain=domain,
                    task=task,
                    prefix=clean["prefix"],
                    branch_slot=branch_slot,
                )
    return [
        materialize_pair(
            args,
            registered_pair=row,
            clean=clean,
            task=task,
            domain=domain,
            continuation_seeds=continuation_seeds,
            log_root=task_log_root
            / str(row["candidate_pair_id"]).replace(":", "_"),
            registry_sha256=registry_sha256,
            semantic_contract=semantic_contract,
            reference_plan=reference_plan,
        )
        for row in ordered
    ]


def preflight_selected_phase_tasks(
    *,
    registered_pairs_by_task: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    reference_preflight: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    phase: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Classify the complete selected phase before any model/judge generation."""

    outcomes: dict[str, dict[str, Any]] = {}
    report_rows: list[dict[str, Any]] = []
    for task_identity, task_pairs in sorted(
        registered_pairs_by_task.items()
    ):
        ordered = sorted(
            task_pairs,
            key=lambda item: str(item["candidate_pair_id"]),
        )
        pair_ids = [str(row["candidate_pair_id"]) for row in ordered]
        try:
            reference_plan = verified_reference_plan_for_task(
                reference_preflight,
                task_identity,
                preflight_binding=preflight_binding,
            )
            if len(ordered) < v610_protocol.PAIRS_PER_TASK:
                raise V6GenerationError(
                    f"{task_identity}: fewer than three executable "
                    "candidate pairs"
                )
            if len(ordered) > v610_protocol.PAIRS_PER_TASK:
                raise V6GenerationError(
                    f"{task_identity}: executable registry has more than "
                    "three candidate pairs"
                )
            domain, task = select_task(task_identity)
            static_prefix = reference_plan["evaluation_prefix"]
            branch_evidence: list[dict[str, Any]] = []
            for registered_pair in ordered:
                validate_pair_against_reference_plan(
                    registered_pair,
                    reference_plan,
                    task=task,
                )
                for branch_slot in registered_pair["branches"]:
                    injection = validate_error_injection(
                        domain=domain,
                        task=task,
                        prefix=static_prefix,
                        branch_slot=branch_slot,
                    )
                    branch_evidence.append(
                        {
                            "candidate_pair_id": registered_pair[
                                "candidate_pair_id"
                            ],
                            "branch_id": branch_slot["branch_id"],
                            "error_event_sha256": injection[
                                "error_event_sha256"
                            ],
                            "actual_tool_error": injection[
                                "actual_tool_error"
                            ],
                            "state_unchanged_after_error": injection[
                                "state_unchanged_after_error"
                            ],
                        }
                    )
            outcome = {
                "status": "PASS",
                "domain": domain,
                "task": task,
                "reference_plan": reference_plan,
                "static_prefix": deepcopy(static_prefix),
                "static_prefix_semantic_sha256": reference_plan[
                    "evaluation_prefix_semantic_sha256"
                ],
                "branch_evidence": branch_evidence,
                "candidate_pair_ids": pair_ids,
            }
            report_row = {
                key: deepcopy(value)
                for key, value in outcome.items()
                if key not in {"task", "reference_plan", "static_prefix"}
            }
            report_row["task_identity"] = task_identity
            report_row["reference_task_preflight_sha256"] = (
                reference_plan["task_preflight_sha256"]
            )
            report_row["sanitized_reference_plan_sha256"] = (
                reference_plan["sanitized_successful_plan_sha256"]
            )
        except Exception as error:
            reason_code = classify_task_local_rejection(error)
            if reason_code is None:
                raise
            outcome = {
                "status": "REJECTED",
                "reason_code": reason_code,
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate_pair_ids": pair_ids,
            }
            report_row = {
                "task_identity": task_identity,
                **deepcopy(outcome),
            }
        outcomes[task_identity] = outcome
        report_rows.append(report_row)
    if set(outcomes) != set(registered_pairs_by_task):
        raise V6GenerationError(
            "selected-phase pre-model classification is incomplete"
        )
    report = {
        "protocol": V610_SELECTED_PHASE_PREFLIGHT_PROTOCOL,
        "design_protocol": V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "phase": phase,
        "status": "CLASSIFIED",
        "expected_task_count": len(registered_pairs_by_task),
        "classified_task_count": len(outcomes),
        "passing_task_count": sum(
            outcome["status"] == "PASS" for outcome in outcomes.values()
        ),
        "rejected_task_count": sum(
            outcome["status"] == "REJECTED"
            for outcome in outcomes.values()
        ),
        "tasks": report_rows,
        "model_or_judge_generation_started": False,
        "official_test_used": False,
    }
    report["receipt_sha256"] = sha256(report)
    return outcomes, report


def merge_shard(
    output_dir: Path,
    expected_tasks: Sequence[str],
    *,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
    expected_pair_ids_by_task: Mapping[str, Sequence[str]] | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    if (
        semantic_contract.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        for task_identity in expected_tasks:
            reconcile_v610_task_index(
                output_dir,
                task_identity=str(task_identity),
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic_contract,
                receipt_path=(
                    output_dir
                    / "tasks"
                    / f"{str(task_identity).replace(':', '_')}.json"
                ),
            )
    task_files = sorted((output_dir / "tasks").glob("*.json"))
    receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in task_files
    ]
    if not all(isinstance(row, dict) for row in receipts):
        raise V6GenerationError("task receipt is not an object")
    observed = [str(row.get("task_identity")) for row in receipts]
    if len(observed) != len(set(observed)):
        raise V6GenerationError("duplicate task receipt")
    expected = set(expected_tasks)
    unexpected = sorted(set(observed) - expected)
    if unexpected:
        raise V6GenerationError(
            f"unexpected/stale task receipts in shard: {unexpected}"
        )
    for receipt in receipts:
        identity = str(receipt["task_identity"])
        expected_pair_ids = (
            expected_pair_ids_by_task.get(identity, ())
            if expected_pair_ids_by_task is not None
            else ()
        )
        v610 = (
            semantic_contract.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
        )
        if (
            v610 and receipt.get("scientific_outcome") == "ACCEPTED"
        ) or (
            not v610 and receipt.get("status") == "PASS"
        ):
            validate_task_resume_receipt(
                receipt,
                task_identity=identity,
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic_contract,
                output_dir=output_dir,
                expected_candidate_pair_ids=(
                    expected_pair_ids
                    if expected_pair_ids_by_task is not None
                    else None
                ),
            )
        elif (
            v610 and receipt.get("scientific_outcome") == "REJECTED"
        ) or (
            not v610 and receipt.get("status") == "REJECTED"
        ):
            validate_rejected_task_resume_receipt(
                receipt,
                task_identity=identity,
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic_contract,
                expected_candidate_pair_ids=expected_pair_ids,
                output_dir=output_dir,
            )
        else:
            raise V6GenerationError(
                f"{identity}: task receipt is not terminal PASS/REJECTED"
            )
    missing = sorted(expected - set(observed))
    accepted = [
        pair
        for receipt in receipts
        if (
            receipt.get("scientific_outcome") == "ACCEPTED"
            if semantic_contract.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
            else receipt.get("status") == "PASS"
        )
        for pair in receipt.get("candidate_pairs", [])
    ]
    expected_pair_count = (
        sum(
            len(expected_pair_ids_by_task.get(str(receipt["task_identity"]), ()))
            for receipt in receipts
            if (
                receipt.get("scientific_outcome") == "ACCEPTED"
                if semantic_contract.get("design_protocol")
                == V6_10_PIPELINE_CLOSURE_PROTOCOL
                else receipt.get("status") == "PASS"
            )
        )
        if expected_pair_ids_by_task is not None
        else None
    )
    if (
        not missing
        and expected_pair_count is not None
        and len(accepted) != expected_pair_count
    ):
        raise V6GenerationError(
            "complete shard has incorrect candidate-pair cardinality"
        )
    rejected_receipts = [
        receipt
        for receipt in receipts
        if (
            receipt.get("scientific_outcome") == "REJECTED"
            if semantic_contract.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
            else receipt.get("status") == "REJECTED"
        )
    ]
    failure_rows = [
        {
            "task_identity": receipt["task_identity"],
            "reason_code": receipt["reason_code"],
            "error_type": receipt["error_type"],
            "error": receipt["error"],
            "task_receipt_sha256": receipt["task_receipt_sha256"],
            "attempt_id": receipt["attempt_id"],
            "official_test_used": False,
        }
        for receipt in rejected_receipts
    ]
    passing_tasks = {
        str(receipt["task_identity"])
        for receipt in receipts
        if (
            receipt.get("scientific_outcome") == "ACCEPTED"
            if semantic_contract.get("design_protocol")
            == V6_10_PIPELINE_CLOSURE_PROTOCOL
            else receipt.get("status") == "PASS"
        )
    }
    qualifying_tasks = {
        task
        for task in passing_tasks
        if sum(pair.get("task_identity") == task for pair in accepted)
        >= protocol.MIN_PAIRS_PER_TASK
    }
    domains = {str(pair.get("domain")) for pair in accepted}
    error_families = {
        str(branch.get("coverage", {}).get("error_family"))
        for pair in accepted
        for branch in pair.get("branches", [])
    }
    error_families.discard("None")
    if missing:
        generation_gate = "INCOMPLETE"
    elif phase == "compatibility":
        generation_gate = (
            "COMPATIBILITY_RELEASE_CANDIDATE"
            if len(qualifying_tasks) >= 22
            and len(accepted) >= 66
            else "NO_GO"
        )
    elif phase == "pilot":
        generation_gate = (
            "GO_TO_MEASUREMENT"
            if len(qualifying_tasks)
            >= v610_protocol.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS
            and len(accepted) >= v610_protocol.PROSPECTIVE_MIN_ACCEPTED_PAIRS
            and len(domains) >= 2
            and len(error_families) >= 3
            else "NO_GO"
        )
    elif phase == "formal":
        if len(qualifying_tasks) >= 48 and len(accepted) >= 144:
            generation_gate = "FORMAL_SELECTION_CANDIDATE"
        elif len(qualifying_tasks) >= 40:
            generation_gate = "SCREEN_ONLY"
        else:
            generation_gate = "NO_GO"
    else:
        generation_gate = "NOT_APPLICABLE"
    summary = {
        "protocol": GENERATION_PROTOCOL,
        "status": "PASS" if not missing else "PARTIAL",
        "phase": phase,
        "generation_gate": generation_gate,
        "expected_tasks": len(expected_tasks),
        "completed_tasks": len(set(observed)),
        "missing_tasks": missing,
        "passing_tasks": len(passing_tasks),
        "rejected_tasks": len(rejected_receipts),
        "qualifying_tasks_with_three_pairs": len(qualifying_tasks),
        "observed_domains": sorted(domains),
        "observed_error_families": sorted(error_families),
        "accepted_candidate_pairs": len(accepted),
        "expected_candidate_pairs": expected_pair_count,
        "registry_sha256": registry_sha256,
        "run_contract_sha256": run_contract_sha256,
        "semantic_generation_contract_sha256": sha256(semantic_contract),
        "runtime_provenance": deepcopy(
            semantic_contract.get("runtime_provenance", {})
        ),
        "attempt_id": semantic_contract.get(
            "runtime_provenance", {}
        ).get("attempt_id"),
        "training_started": False,
        "official_test_used": False,
    }
    if (
        semantic_contract.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        task_receipt_hashes = {
            str(receipt["task_identity"]): str(
                receipt["task_receipt_sha256"]
            )
            for receipt in receipts
        }
        summary["execution_status"] = (
            "PASS" if not missing else "INCOMPLETE"
        )
        summary["task_receipt_sha256_by_task"] = task_receipt_hashes
        summary["task_receipts_sha256"] = sha256(task_receipt_hashes)
    generation_receipt_path = output_dir / "generation_receipt.json"
    if (
        semantic_contract.get("design_protocol")
        == V6_10_PIPELINE_CLOSURE_PROTOCOL
    ):
        existing_generation_receipt: dict[str, Any] | None = None
        if generation_receipt_path.exists():
            try:
                loaded_generation_receipt = json.loads(
                    generation_receipt_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise V6GenerationError(
                    "existing generation receipt is not valid JSON"
                ) from error
            if not isinstance(loaded_generation_receipt, dict):
                raise V6GenerationError(
                    "existing generation receipt is not an object"
                )
            unhashed_existing = {
                key: value
                for key, value in loaded_generation_receipt.items()
                if key != "generation_receipt_sha256"
            }
            if (
                loaded_generation_receipt.get("generation_receipt_sha256")
                != sha256(unhashed_existing)
                or loaded_generation_receipt.get("run_contract_sha256")
                != run_contract_sha256
                or loaded_generation_receipt.get(
                    "semantic_generation_contract_sha256"
                )
                != sha256(semantic_contract)
                or loaded_generation_receipt.get("status")
                not in {"PASS", "PARTIAL"}
            ):
                raise V6GenerationError(
                    "existing V6.10 generation receipt is stale/invalid"
                )
            existing_generation_receipt = loaded_generation_receipt

        explicit = v610_explicit_receipt_fields(semantic_contract)
        terminal_end = (
            existing_generation_receipt.get("run_ended_at_utc")
            if existing_generation_receipt is not None
            and existing_generation_receipt.get("status") == "PASS"
            and not missing
            else utc_now()
        )
        run_started = _utc_datetime(explicit["run_started_at_utc"])
        run_ended = _utc_datetime(str(terminal_end))
        task_end_times = [
            _utc_datetime(str(receipt.get("task_ended_at_utc", "")))
            for receipt in receipts
        ]
        if (
            run_ended < run_started
            or any(run_ended < task_end for task_end in task_end_times)
        ):
            raise V6GenerationError(
                "V6.10 generation receipt UTC timing is out of order"
            )
        explicit["run_ended_at_utc"] = str(terminal_end)
        summary.update(explicit)
        summary["generation_receipt_sha256"] = sha256(summary)
        if (
            existing_generation_receipt is not None
            and existing_generation_receipt.get("status") == "PASS"
            and not missing
            and existing_generation_receipt != summary
        ):
            raise V6GenerationError(
                "terminal V6.10 generation receipt resume drift"
            )
    # Validate and serialize the complete closure before publishing any
    # derived pool/ledger bytes.  The generation receipt is the sole final
    # commit point: its absence means the closure is not committed.
    canonical(accepted)
    canonical(failure_rows)
    canonical(summary)
    if not missing:
        write_jsonl(output_dir / "candidate_pairs.unscored.jsonl", accepted)
    write_jsonl(output_dir / "failure_ledger.jsonl", failure_rows)
    write_json(generation_receipt_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.clean_attempts <= 0 or args.recovery_attempts <= 0:
        raise V6GenerationError("clean/recovery attempts must be positive")
    if args.max_tokens <= 0 or args.max_steps <= 0 or args.timeout <= 0:
        raise V6GenerationError("generation limits must be positive")
    continuation_seeds = parse_seed_set(args.continuation_seeds)
    judge_model = args.judge_model or args.user_model
    judge_revision = args.judge_revision or args.user_revision
    judge_api_base = args.judge_api_base or args.user_api_base
    if not args.teacher_revision or not args.user_revision or not judge_revision:
        raise V6GenerationError("all model revisions must be pinned and non-empty")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise V6GenerationError("output path exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "run_contract.json"
    existing_run_contract: dict[str, Any] | None = None
    if contract_path.exists():
        try:
            loaded_contract = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise V6GenerationError(
                "existing run contract is not valid JSON"
            ) from error
        if not isinstance(loaded_contract, dict):
            raise V6GenerationError("existing run contract is not an object")
        existing_run_contract = loaded_contract

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise V6GenerationError("registry root must be an object")
    registry_file_sha256 = sha256_file(args.registry)
    registry_sha256 = verify_registry(registry)
    design_protocol = str(registry["design_protocol"])
    args.v610_contract_active = (
        design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
    )
    validate_v610_modes(
        args,
        design_protocol=design_protocol,
        continuation_seeds=continuation_seeds,
    )
    execution_provenance: dict[str, Any] = {}
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        execution_provenance = build_execution_provenance(
            run_started_at_utc=resolve_v610_run_started_at_utc(
                existing_run_contract
            ),
        )
    runtime_provenance, reference_preflight = v610_runtime_provenance(
        args,
        design_protocol=design_protocol,
        registry=registry,
        registry_file_sha256=registry_file_sha256,
    )
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        verify_v610_registry_preflight_binding(
            registry,
            runtime_provenance["reference_preflight"],
        )
    configure_tau2(args.tau2_root)
    register_agent()
    os.environ.setdefault("OPENAI_API_KEY", "v6-local")
    pairs = phase_pairs(
        registry,
        phase=args.phase,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        smoke_task=args.smoke_task,
    )
    selected_tasks = selected_phase_tasks(
        registry,
        phase=args.phase,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        smoke_task=args.smoke_task,
    )
    by_task: dict[str, list[dict[str, Any]]] = {
        task_identity: [] for task_identity in selected_tasks
    }
    for row in pairs:
        by_task.setdefault(str(row["task_identity"]), []).append(row)
    generation_contract = semantic_generation_contract(
        args,
        continuation_seeds=continuation_seeds,
        judge_model=judge_model,
        judge_revision=judge_revision,
        judge_api_base=judge_api_base,
        design_protocol=design_protocol,
        runtime_provenance=runtime_provenance,
        execution_provenance=execution_provenance,
    )
    run_contract = build_run_contract(
        args,
        registry_file_sha256=registry_file_sha256,
        registry_sha256=registry_sha256,
        task_ids=sorted(by_task),
        semantic_contract=generation_contract,
    )
    run_contract_sha256 = sha256(run_contract)
    process_started_at_utc: str | None = None
    process_execution_id: str | None = None
    run_lock: Any | None = None
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        process_started_at_utc = utc_now()
        process_execution_id = v610_process_execution_id(
            attempt_id=str(runtime_provenance["attempt_id"]),
            process_started_at_utc=process_started_at_utc,
            exact_argv_sha256=str(
                execution_provenance["exact_argv_sha256"]
            ),
        )
        run_lock = acquire_v610_run_lock(
            output_dir,
            process_execution_id=process_execution_id,
            process_started_at_utc=process_started_at_utc,
        )
    if existing_run_contract is not None:
        if existing_run_contract != run_contract:
            raise V6GenerationError("resume run contract drift")
    else:
        write_json(contract_path, run_contract)

    pre_model_outcomes: dict[str, dict[str, Any]] = {}
    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        if reference_preflight is None:
            raise V6GenerationError(
                "V6.10 selected-phase preflight lacks reference receipt"
            )
        pre_model_outcomes, pre_model_report = (
            preflight_selected_phase_tasks(
                registered_pairs_by_task=by_task,
                reference_preflight=reference_preflight,
                preflight_binding=runtime_provenance[
                    "reference_preflight"
                ],
                phase=args.phase,
            )
        )
        pre_model_report_path = (
            output_dir / "selected_phase_pre_model_preflight.json"
        )
        if pre_model_report_path.exists():
            existing_pre_model_report = json.loads(
                pre_model_report_path.read_text(encoding="utf-8")
            )
            if existing_pre_model_report != pre_model_report:
                raise V6GenerationError(
                    "selected-phase pre-model preflight resume drift"
                )
        else:
            write_json(pre_model_report_path, pre_model_report)

    # Validate every prior terminal receipt before the first model/judge
    # generation.  In V6.10 only PASS is resumable: every prior REJECTED
    # receipt is validated as evidence, then quarantined and rerun below.
    resumed_terminal_tasks: set[str] = set()
    current_process_terminal_tasks: set[str] = set()
    for task_identity, task_pairs in sorted(by_task.items()):
        receipt_path = (
            output_dir
            / "tasks"
            / f"{task_identity.replace(':', '_')}.json"
        )
        expected_pair_ids = [
            str(row["candidate_pair_id"]) for row in task_pairs
        ]
        existing_receipt: dict[str, Any] | None = None
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
            reconcile_v610_task_index(
                output_dir,
                task_identity=task_identity,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=generation_contract,
                receipt_path=receipt_path,
            )
        if receipt_path.exists():
            loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise V6GenerationError(
                    f"{task_identity}: existing task receipt is not an object"
                )
            existing_receipt = loaded
            scientific_outcome = loaded.get("scientific_outcome")
            if (
                design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
                and scientific_outcome == "ACCEPTED"
            ) or (
                design_protocol != V6_10_PIPELINE_CLOSURE_PROTOCOL
                and loaded.get("status") == "PASS"
            ):
                validate_task_resume_receipt(
                    loaded,
                    task_identity=task_identity,
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                    output_dir=output_dir,
                )
                resumed_terminal_tasks.add(task_identity)
            elif (
                design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
                and scientific_outcome == "REJECTED"
            ) or (
                design_protocol != V6_10_PIPELINE_CLOSURE_PROTOCOL
                and loaded.get("status") == "REJECTED"
            ):
                validate_rejected_task_resume_receipt(
                    loaded,
                    task_identity=task_identity,
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                    output_dir=output_dir,
                )
                resumed_terminal_tasks.add(task_identity)
            else:
                raise V6GenerationError(
                    f"{task_identity}: existing task receipt is not terminal"
                )
        pre_model = pre_model_outcomes.get(task_identity)
        if (
            design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
            and (
                not isinstance(pre_model, Mapping)
                or pre_model.get("status") not in {"PASS", "REJECTED"}
            )
        ):
            raise V6GenerationError(
                f"{task_identity}: missing terminal pre-model classification"
            )
        if (
            task_identity in resumed_terminal_tasks
            and isinstance(existing_receipt, Mapping)
            and existing_receipt.get("scientific_outcome") == "REJECTED"
            and isinstance(pre_model, Mapping)
            and pre_model.get("status") == "PASS"
        ):
            raise V6GenerationError(
                f"{task_identity}: pre-model PASS/rejection resume drift"
            )
        if (
            isinstance(pre_model, Mapping)
            and pre_model.get("status") == "REJECTED"
        ):
            if task_identity in resumed_terminal_tasks:
                if (
                    existing_receipt is None
                    or existing_receipt.get("scientific_outcome")
                    != "REJECTED"
                    or existing_receipt.get("reason_code")
                    != pre_model.get("reason_code")
                ):
                    raise V6GenerationError(
                        f"{task_identity}: pre-model rejection/resume drift"
                    )
                continue
            if (
                existing_receipt is not None
                and existing_receipt.get("scientific_outcome") == "REJECTED"
                and existing_receipt.get("reason_code")
                != pre_model.get("reason_code")
            ):
                raise V6GenerationError(
                    f"{task_identity}: pre-model rejection/resume drift"
                )
            if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
                assert process_execution_id is not None
                assert process_started_at_utc is not None
                attempt_context = begin_v610_task_attempt(
                    output_dir,
                    task_identity=task_identity,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    process_execution_id=process_execution_id,
                    process_started_at_utc=process_started_at_utc,
                    receipt_path=receipt_path,
                )
                receipt = rejected_task_receipt(
                    task_identity=task_identity,
                    reason_code=str(pre_model["reason_code"]),
                    error=V6GenerationError(str(pre_model["error"])),
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                    task_started_at_utc=str(
                        attempt_context["task_started_at_utc"]
                    ),
                    task_ended_at_utc=utc_now(),
                )
                # Preserve the original classified error type.
                receipt["error_type"] = str(pre_model["error_type"])
                receipt.pop("task_receipt_sha256", None)
                receipt["task_receipt_sha256"] = sha256(receipt)
                finalize_v610_task_attempt(
                    output_dir,
                    task_identity=task_identity,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    attempt_context=attempt_context,
                    receipt_path=receipt_path,
                    receipt=receipt,
                )
                current_process_terminal_tasks.add(task_identity)
            elif existing_receipt is not None:
                if existing_receipt.get("status") != "REJECTED":
                    raise V6GenerationError(
                        f"{task_identity}: pre-model rejection/resume drift"
                    )
            else:
                task_started_at_utc = utc_now()
                task_ended_at_utc = utc_now()
                receipt = rejected_task_receipt(
                    task_identity=task_identity,
                    reason_code=str(pre_model["reason_code"]),
                    error=V6GenerationError(str(pre_model["error"])),
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                    task_started_at_utc=task_started_at_utc,
                    task_ended_at_utc=task_ended_at_utc,
                )
                # Preserve the original classified error type.
                receipt["error_type"] = str(pre_model["error_type"])
                receipt.pop("task_receipt_sha256", None)
                receipt["task_receipt_sha256"] = sha256(receipt)
                write_json(receipt_path, receipt)

    if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
        revalidate_v610_live_model_runtime(runtime_provenance)

    patch_local_nl_judge(
        litellm_openai_model(judge_model),
        endpoint_args(
            judge_api_base,
            "v6-local",
            max_tokens=args.max_tokens,
            seed=continuation_seeds[0],
            temperature=DECODING_TEMPERATURE,
            top_p=DECODING_TOP_P,
        ),
    )

    for task_identity, task_pairs in sorted(by_task.items()):
        receipt_path = (
            output_dir
            / "tasks"
            / f"{task_identity.replace(':', '_')}.json"
        )
        expected_pair_ids = [
            str(row["candidate_pair_id"]) for row in task_pairs
        ]
        attempt_context: dict[str, Any] | None = None
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
            if (
                task_identity in resumed_terminal_tasks
                or task_identity in current_process_terminal_tasks
            ):
                continue
            assert process_execution_id is not None
            assert process_started_at_utc is not None
            attempt_context = begin_v610_task_attempt(
                output_dir,
                task_identity=task_identity,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=generation_contract,
                process_execution_id=process_execution_id,
                process_started_at_utc=process_started_at_utc,
                receipt_path=receipt_path,
            )
            task_started_at_utc = str(
                attempt_context["task_started_at_utc"]
            )
        elif receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise V6GenerationError(
                    f"{task_identity}: existing task receipt is not an object"
                )
            if receipt.get("status") == "PASS":
                validate_task_resume_receipt(
                    receipt,
                    task_identity=task_identity,
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                )
            elif receipt.get("status") == "REJECTED":
                validate_rejected_task_resume_receipt(
                    receipt,
                    task_identity=task_identity,
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=generation_contract,
                    expected_candidate_pair_ids=expected_pair_ids,
                )
            else:
                raise V6GenerationError(
                    f"{task_identity}: existing task receipt is not terminal"
                )
            continue
        else:
            task_started_at_utc = utc_now()
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
            revalidate_v610_live_model_runtime(runtime_provenance)
        try:
            pre_model = pre_model_outcomes.get(task_identity)
            reference_plan = (
                pre_model["reference_plan"]
                if isinstance(pre_model, Mapping)
                and pre_model.get("status") == "PASS"
                else (
                    verified_reference_plan_for_task(
                        reference_preflight,
                        task_identity,
                        preflight_binding=runtime_provenance[
                            "reference_preflight"
                        ],
                    )
                    if reference_preflight is not None
                    else None
                )
            )
            if (
                design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
                and len(task_pairs) < v610_protocol.PAIRS_PER_TASK
            ):
                raise V6GenerationError(
                    f"{task_identity}: fewer than three executable "
                    "candidate pairs"
                )
            if (
                design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
                and len(task_pairs) > v610_protocol.PAIRS_PER_TASK
            ):
                raise V6GenerationError(
                    f"{task_identity}: executable registry has more than "
                    "three candidate pairs"
                )
            if (
                isinstance(pre_model, Mapping)
                and pre_model.get("status") == "PASS"
            ):
                domain = str(pre_model["domain"])
                task = pre_model["task"]
            else:
                domain, task = select_task(task_identity)
            clean_seed = int(task_pairs[0]["choice_seed"])
            task_log_root = (
                Path(attempt_context["evidence_dir"]) / "logs"
                if attempt_context is not None
                else output_dir / "logs" / task_identity.replace(":", "_")
            )
            clean = clean_rollout(
                args,
                task=task,
                domain=domain,
                task_identity=task_identity,
                seed=clean_seed,
                log_root=task_log_root,
                reference_plan=reference_plan,
            )
            observed_pairs = materialize_task_pairs(
                args,
                registered_pairs=task_pairs,
                clean=clean,
                task=task,
                domain=domain,
                continuation_seeds=continuation_seeds,
                task_log_root=task_log_root,
                registry_sha256=registry_sha256,
                semantic_contract=generation_contract,
                reference_plan=reference_plan,
            )
            receipt = {
                "protocol": GENERATION_PROTOCOL,
                "status": "PASS",
                "task_identity": task_identity,
                "domain": domain,
                "registry_sha256": registry_sha256,
                "run_contract_sha256": run_contract_sha256,
                "semantic_generation_contract": deepcopy(generation_contract),
                "semantic_generation_contract_sha256": sha256(
                    generation_contract
                ),
                "runtime_provenance": deepcopy(
                    generation_contract.get("runtime_provenance", {})
                ),
                "attempt_id": generation_contract.get(
                    "runtime_provenance", {}
                ).get("attempt_id"),
                "reference_plan_binding": deepcopy(reference_plan),
                "clean": clean,
                "candidate_pairs": observed_pairs,
                "candidate_pair_count": len(observed_pairs),
                "all_pairs_share_prefix": (
                    len({row["prefix_sha256"] for row in observed_pairs}) == 1
                ),
                "all_pairs_share_environment_snapshot": (
                    len(
                        {
                            row["environment_snapshot_sha256"]
                            for row in observed_pairs
                        }
                    )
                    == 1
                ),
                "official_test_used": False,
            }
            if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
                receipt.update(
                    {
                        "execution_status": "PASS",
                        "scientific_outcome": "ACCEPTED",
                    }
                )
                receipt.update(
                    v610_task_receipt_fields(
                        generation_contract,
                        task_identity=task_identity,
                        task_started_at_utc=task_started_at_utc,
                        task_ended_at_utc=utc_now(),
                    )
                )
            if (
                not receipt["all_pairs_share_prefix"]
                or not receipt["all_pairs_share_environment_snapshot"]
            ):
                raise V6GenerationError(
                    f"{task_identity}: sibling contract drift"
                )
            receipt["task_receipt_sha256"] = sha256(receipt)
        except Exception as error:
            reason_code = (
                classify_task_local_rejection(error)
                if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL
                else None
            )
            if reason_code is None:
                raise
            receipt = rejected_task_receipt(
                task_identity=task_identity,
                reason_code=reason_code,
                error=error,
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=generation_contract,
                expected_candidate_pair_ids=expected_pair_ids,
                task_started_at_utc=task_started_at_utc,
                task_ended_at_utc=utc_now(),
            )
        if design_protocol == V6_10_PIPELINE_CLOSURE_PROTOCOL:
            revalidate_v610_live_model_runtime(runtime_provenance)
            assert attempt_context is not None
            receipt = finalize_v610_task_attempt(
                output_dir,
                task_identity=task_identity,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=generation_contract,
                attempt_context=attempt_context,
                receipt_path=receipt_path,
                receipt=receipt,
            )
            current_process_terminal_tasks.add(task_identity)
        else:
            write_json(receipt_path, receipt)
        print(
            canonical(
                {
                    "task_identity": task_identity,
                    "candidate_pairs": receipt["candidate_pair_count"],
                    "status": receipt["status"],
                    "reason_code": receipt.get("reason_code"),
                }
            ),
            flush=True,
        )
    summary = merge_shard(
        output_dir,
        sorted(by_task),
        registry_sha256=registry_sha256,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=generation_contract,
        expected_pair_ids_by_task={
            task_identity: [
                str(row["candidate_pair_id"]) for row in task_pairs
            ]
            for task_identity, task_pairs in by_task.items()
        },
        phase=args.phase,
    )
    print(canonical(summary))


if __name__ == "__main__":
    main()
