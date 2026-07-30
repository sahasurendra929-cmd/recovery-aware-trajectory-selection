#!/usr/bin/env python3
"""Build fail-closed V6.10 source/container and model-server receipts.

The candidate generator deliberately accepts only strict, hashed runtime
receipts.  This utility creates those receipts from live evidence instead of
asking an operator to hand-write JSON:

* clean source and tau2 commits plus the launched generator byte hash;
* dependency, Python, Torch, CUDA, driver, vLLM, and three-GPU inventory;
* exact role-scoped model/revision/topology declarations;
* live PID/start/cmdline, GPU visibility, listening-socket ownership, and
  content-addressed local-snapshot evidence before and after probing;
* a live ``/models`` identity check; and
* a real >=4096-token chat-completion probe while GPU memory is sampled.

The two receipt files are written canonically and are passed through the
generator's own strict validators before the output directory is published.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

if __package__:
    from scripts import run_v6_candidate_generation as generation
else:  # pragma: no cover - direct script execution
    import run_v6_candidate_generation as generation


SOURCE_RECEIPT_NAME = "source_container_provenance.json"
MODEL_RECEIPT_NAME = "model_server_receipts.json"
HASH_MANIFEST_NAME = "runtime_receipt_hashes.json"
LONG_CONTEXT_REPETITIONS = 6000
PROBE_MAX_NEW_TOKENS = 128
PROBE_SEED = generation.V610_COMPATIBILITY_SEEDS[0]
PROBE_TEMPERATURE = 0.0
ROLE_ORDER = ("teacher", "user", "judge")
SERVICE_SPEC_KEYS = {"roles"}
ROLE_SPEC_KEYS = {
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
}
VLLM_SERVER_MODULE = "vllm.entrypoints.openai.api_server"
FROZEN_MODEL_DTYPE = "float16"
FROZEN_MAX_MODEL_LEN = 8192
FROZEN_GPU_MEMORY_UTILIZATION = 0.90
FROZEN_GPU_MODEL = "NVIDIA RTX PRO 4500 Blackwell"
FROZEN_MIN_GPU_MEMORY_MIB = 32_000
SNAPSHOT_REQUIRED_IDENTITY_FILES = {
    "config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
}
SNAPSHOT_OPTIONAL_IDENTITY_FILES = {
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
}
TOKENIZER_PAYLOAD_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
}


class RuntimeReceiptError(RuntimeError):
    """A live runtime fact or frozen V6.10 identity failed validation."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def run_checked(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeReceiptError(
            f"cannot execute runtime probe: {list(arguments)!r}"
        ) from error
    if completed.returncode != 0:
        raise RuntimeReceiptError(
            "runtime probe failed: "
            f"argv={list(arguments)!r}, stderr={completed.stderr.strip()!r}"
        )
    return completed.stdout.strip()


def git_identity(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    if not (resolved / ".git").exists():
        raise RuntimeReceiptError(f"not a git checkout: {resolved}")
    commit = run_checked(("git", "rev-parse", "HEAD"), cwd=resolved)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeReceiptError(f"git HEAD is not immutable: {resolved}")
    status = run_checked(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=resolved,
    )
    if status:
        raise RuntimeReceiptError(
            f"worktree contains tracked or untracked drift: {resolved}"
        )
    return {
        "commit": commit,
        "tree": run_checked(
            ("git", "rev-parse", "HEAD^{tree}"), cwd=resolved
        ),
        "tracked_worktree_clean": True,
    }


def capture_runtime(runtime_python: Path) -> dict[str, str]:
    executable = Path(os.path.abspath(runtime_python))
    if not executable.is_file():
        raise RuntimeReceiptError(
            f"runtime Python is absent: {executable}"
        )
    probe = (
        "import importlib.metadata,json,platform,torch;"
        "print(json.dumps({"
        "'python_version':platform.python_version(),"
        "'torch_version':importlib.metadata.version('torch'),"
        "'cuda_version':str(torch.version.cuda or ''),"
        "'vllm_version':importlib.metadata.version('vllm')"
        "},sort_keys=True))"
    )
    raw = run_checked((str(executable), "-I", "-c", probe))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeReceiptError(
            "runtime Python returned malformed version JSON"
        ) from error
    keys = {
        "python_version",
        "torch_version",
        "cuda_version",
        "vllm_version",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or any(
            not isinstance(payload.get(key), str) or not payload[key]
            for key in keys
        )
    ):
        raise RuntimeReceiptError("runtime version evidence is incomplete")
    payload["python_executable_path"] = str(executable)
    payload["python_executable_sha256"] = file_sha256(executable)
    return payload


def _parse_nvidia_csv(raw: str) -> list[list[str]]:
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(raw.splitlines())
        if row
    ]
    if not rows:
        raise RuntimeReceiptError("nvidia-smi returned no GPUs")
    return rows


def capture_gpu_inventory() -> tuple[list[dict[str, Any]], str]:
    raw = run_checked(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    rows = _parse_nvidia_csv(raw)
    if len(rows) != 3 or any(len(row) != 5 for row in rows):
        raise RuntimeReceiptError(
            "V6.10 requires exactly three complete GPU inventory rows"
        )
    inventory: list[dict[str, Any]] = []
    drivers: set[str] = set()
    for values in rows:
        try:
            index = int(values[0])
            total_bytes = int(values[3]) * 1024 * 1024
        except ValueError as error:
            raise RuntimeReceiptError(
                f"malformed nvidia-smi inventory row: {values!r}"
            ) from error
        if not values[1] or not values[2] or not values[4]:
            raise RuntimeReceiptError(
                f"incomplete nvidia-smi inventory row: {values!r}"
            )
        inventory.append(
            {
                "index": index,
                "model": values[1],
                "uuid": values[2],
                "total_memory_bytes": total_bytes,
            }
        )
        drivers.add(values[4])
    inventory.sort(key=lambda row: int(row["index"]))
    if (
        [row["index"] for row in inventory] != [0, 1, 2]
        or len({row["uuid"] for row in inventory}) != 3
        or len(drivers) != 1
    ):
        raise RuntimeReceiptError(
            "GPU indices, UUIDs, or driver versions are inconsistent"
        )
    if any(row["model"] != FROZEN_GPU_MODEL for row in inventory):
        raise RuntimeReceiptError(
            f"V6.10 PRO 4500 compatibility freezes exactly three "
            f"{FROZEN_GPU_MODEL} GPUs"
        )
    minimum_bytes = FROZEN_MIN_GPU_MEMORY_MIB * 1024 * 1024
    if any(row["total_memory_bytes"] < minimum_bytes for row in inventory):
        raise RuntimeReceiptError(
            "V6.10 PRO 4500 compatibility requires at least "
            f"{FROZEN_MIN_GPU_MEMORY_MIB} MiB on every GPU"
        )
    return inventory, next(iter(drivers))


def sample_gpu_memory_bytes() -> dict[str, int]:
    raw = run_checked(
        (
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    rows = _parse_nvidia_csv(raw)
    result: dict[str, int] = {}
    for values in rows:
        if len(values) != 2 or not values[0]:
            raise RuntimeReceiptError(
                f"malformed GPU-memory row: {values!r}"
            )
        try:
            used = int(values[1]) * 1024 * 1024
        except ValueError as error:
            raise RuntimeReceiptError(
                f"malformed GPU-memory value: {values!r}"
            ) from error
        if values[0] in result or used < 0:
            raise RuntimeReceiptError(
                f"duplicate or negative GPU-memory row: {values!r}"
            )
        result[values[0]] = used
    return result


def snapshot_identity(
    snapshot: Path,
    *,
    expected_model: str,
    expected_revision: str,
) -> tuple[dict[str, Any], str]:
    """Verify one pinned HF snapshot without rereading all weight bytes."""

    root = snapshot.resolve()
    expected_cache_name = "models--" + expected_model.replace("/", "--")
    if (
        not snapshot.is_absolute()
        or snapshot != root
        or not root.is_dir()
        or root.name != expected_revision
        or root.parent.name != "snapshots"
        or root.parent.parent.name != expected_cache_name
    ):
        raise RuntimeReceiptError(
            "model path is not the exact frozen Hugging Face snapshot"
        )
    present = {path.name for path in root.iterdir() if path.is_file()}
    if not SNAPSHOT_REQUIRED_IDENTITY_FILES.issubset(present):
        raise RuntimeReceiptError(
            "model snapshot lacks required config/tokenizer/weight-index files"
        )
    if not TOKENIZER_PAYLOAD_FILES.intersection(present):
        raise RuntimeReceiptError(
            "model snapshot lacks a tokenizer payload"
        )
    byte_names = sorted(
        (SNAPSHOT_REQUIRED_IDENTITY_FILES | SNAPSHOT_OPTIONAL_IDENTITY_FILES)
        & present
    )
    byte_hashes = {
        name: file_sha256(root / name) for name in byte_names
    }
    weight_paths = sorted(root.glob("*.safetensors"))
    if not weight_paths:
        raise RuntimeReceiptError(
            "model snapshot contains no safetensors weight shards"
        )
    weight_blobs: dict[str, str] = {}
    for weight in weight_paths:
        if not weight.is_symlink():
            raise RuntimeReceiptError(
                "weight shard is not bound to a content-addressed HF blob"
            )
        target = weight.resolve()
        if (
            not target.is_file()
            or target.parent.name != "blobs"
            or re.fullmatch(r"[0-9a-f]{40,64}", target.name) is None
        ):
            raise RuntimeReceiptError(
                "weight shard symlink target is not a valid HF blob"
            )
        weight_blobs[weight.name] = target.name
    try:
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeReceiptError(
            "model weight index is not valid JSON"
        ) from error
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    indexed_names = (
        {
            str(value)
            for value in weight_map.values()
            if isinstance(value, str) and value
        }
        if isinstance(weight_map, Mapping)
        else set()
    )
    if indexed_names != set(weight_blobs):
        raise RuntimeReceiptError(
            "model weight index and snapshot shard set disagree"
        )
    identity = {
        "byte_files": byte_hashes,
        "weight_blob_targets": dict(sorted(weight_blobs.items())),
    }
    return identity, semantic_sha256(identity)


def _read_proc_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeReceiptError(
            f"cannot read live process {label}: {path}"
        ) from error


def _process_start_evidence(
    *,
    pid: int,
    proc_root: Path,
) -> tuple[str, int, str]:
    stat_raw = _read_proc_bytes(
        proc_root / str(pid) / "stat",
        label="stat",
    ).decode("utf-8", errors="strict")
    closing = stat_raw.rfind(")")
    remainder = stat_raw[closing + 2 :].split() if closing >= 0 else []
    try:
        start_ticks = int(remainder[19])
    except (IndexError, ValueError) as error:
        raise RuntimeReceiptError(
            "live process stat has no valid start-time field"
        ) from error
    boot_id = (
        _read_proc_bytes(
            proc_root / "sys" / "kernel" / "random" / "boot_id",
            label="boot id",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id,
    ) is None:
        raise RuntimeReceiptError("kernel boot id is malformed")
    proc_stat = (
        _read_proc_bytes(proc_root / "stat", label="kernel stat")
        .decode("ascii", errors="strict")
        .splitlines()
    )
    try:
        boot_seconds = int(
            next(line for line in proc_stat if line.startswith("btime ")).split()[1]
        )
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (StopIteration, IndexError, ValueError, OSError) as error:
        raise RuntimeReceiptError(
            "cannot derive process start timestamp"
        ) from error
    started = datetime.fromtimestamp(
        boot_seconds + start_ticks / ticks_per_second,
        timezone.utc,
    )
    started_at = (
        started.isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    return boot_id, start_ticks, started_at


def _cuda_visible_devices(
    *,
    pid: int,
    proc_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    environ = _read_proc_bytes(
        proc_root / str(pid) / "environ",
        label="environment",
    ).split(b"\0")
    prefix = b"CUDA_VISIBLE_DEVICES="
    values = [
        row[len(prefix) :].decode("utf-8", errors="strict")
        for row in environ
        if row.startswith(prefix)
    ]
    if len(values) != 1 or not values[0]:
        raise RuntimeReceiptError(
            "live vLLM process lacks one explicit CUDA_VISIBLE_DEVICES"
        )
    raw = values[0]
    tokens = [value.strip() for value in raw.split(",")]
    if not tokens or any(not value for value in tokens):
        raise RuntimeReceiptError("CUDA_VISIBLE_DEVICES is malformed")
    by_index = {str(row["index"]): str(row["uuid"]) for row in inventory}
    available = {str(row["uuid"]) for row in inventory}
    observed: list[str] = []
    for token in tokens:
        if token in by_index:
            observed.append(by_index[token])
        elif token in available:
            observed.append(token)
        else:
            raise RuntimeReceiptError(
                "CUDA_VISIBLE_DEVICES names an unknown GPU"
            )
    if len(observed) != len(set(observed)):
        raise RuntimeReceiptError(
            "CUDA_VISIBLE_DEVICES contains duplicate GPUs"
        )
    return raw, observed


def _selected_process_environment(
    *,
    pid: int,
    proc_root: Path,
) -> dict[str, Any]:
    rows = _read_proc_bytes(
        proc_root / str(pid) / "environ",
        label="environment",
    ).split(b"\0")
    decoded: dict[str, str] = {}
    for row in rows:
        if not row:
            continue
        try:
            key, value = row.decode("utf-8", errors="strict").split("=", 1)
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeReceiptError(
                "live process environment contains a malformed entry"
            ) from error
        if key in decoded:
            raise RuntimeReceiptError(
                "live process environment contains a duplicate key"
            )
        decoded[key] = value
    if decoded.get("PYTHONPATH") not in (None, ""):
        raise RuntimeReceiptError(
            "live vLLM process has an unregistered PYTHONPATH override"
        )
    if decoded.get("PYTHONHOME") not in (None, ""):
        raise RuntimeReceiptError(
            "live vLLM process has an unregistered PYTHONHOME override"
        )
    return {
        "pythonpath_override_absent": True,
        "pythonhome_override_absent": True,
        "virtual_env": decoded.get("VIRTUAL_ENV"),
    }


def _listening_socket_inode(
    *,
    pid: int,
    port: int,
    proc_root: Path,
) -> str:
    fd_root = proc_root / str(pid) / "fd"
    try:
        descriptors = list(fd_root.iterdir())
    except OSError as error:
        raise RuntimeReceiptError(
            "cannot inspect live vLLM socket descriptors"
        ) from error
    owned: set[str] = set()
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match is not None:
            owned.add(match.group(1))
    listening: set[str] = set()
    for table_name in ("tcp", "tcp6"):
        table = proc_root / "net" / table_name
        try:
            rows = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError as error:
            raise RuntimeReceiptError(
                f"cannot inspect live socket table: {table_name}"
            ) from error
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                listening.add(fields[9])
    matches = sorted(owned & listening)
    if len(matches) != 1:
        raise RuntimeReceiptError(
            "declared vLLM PID does not uniquely own the endpoint LISTEN socket"
        )
    return matches[0]


def _canonical_vllm_argv(
    spec: Mapping[str, Any],
    *,
    snapshot_path: str,
    port: int,
    expected_runtime_python: Path,
) -> list[str]:
    command = list(spec["launch_command"])
    executable = Path(command[0]) if command else Path()
    expected = [
        str(executable),
        "-I",
        "-m",
        VLLM_SERVER_MODULE,
        "--model",
        snapshot_path,
        "--revision",
        str(spec["resolved_revision"]),
        "--served-model-name",
        str(spec["model"]),
        "--quantization",
        str(spec["quantization"]),
        "--dtype",
        str(spec["dtype"]),
        "--tensor-parallel-size",
        str(spec["tensor_parallel_size"]),
        "--load-format",
        "safetensors",
        "--max-model-len",
        str(FROZEN_MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(FROZEN_GPU_MEMORY_UTILIZATION),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable != Path(os.path.abspath(expected_runtime_python))
        or command != expected
    ):
        raise RuntimeReceiptError(
            "vLLM launch argv is not the frozen safe command"
        )
    return expected


def inspect_vllm_process(
    spec: Mapping[str, Any],
    *,
    inventory: Sequence[Mapping[str, Any]],
    expected_runtime_python: Path,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Bind one endpoint to its live process, snapshot, GPU set, and socket."""

    pid = spec.get("server_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeReceiptError("service spec server_pid is invalid")
    parsed = urlparse(str(spec["api_base"]))
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeReceiptError(
            "model API must be an explicit loopback /v1 endpoint"
        )
    cmdline_raw = _read_proc_bytes(
        proc_root / str(pid) / "cmdline",
        label="cmdline",
    )
    command = [
        value.decode("utf-8", errors="strict")
        for value in cmdline_raw.split(b"\0")
        if value
    ]
    if command != list(spec["launch_command"]):
        raise RuntimeReceiptError(
            "declared launch command differs from live process argv"
        )
    try:
        model_index = command.index("--model") + 1
        snapshot_path = command[model_index]
    except (ValueError, IndexError) as error:
        raise RuntimeReceiptError("vLLM argv has no model path") from error
    _canonical_vllm_argv(
        spec,
        snapshot_path=snapshot_path,
        port=parsed.port,
        expected_runtime_python=expected_runtime_python,
    )
    executable_link = proc_root / str(pid) / "exe"
    try:
        observed_executable = executable_link.resolve(strict=True)
    except OSError as error:
        raise RuntimeReceiptError(
            "cannot resolve live vLLM process executable"
        ) from error
    frozen_executable = Path(
        os.path.abspath(expected_runtime_python)
    ).resolve()
    if observed_executable != frozen_executable:
        raise RuntimeReceiptError(
            "live vLLM executable differs from runtime Python"
        )
    identity_files, identity_sha256 = snapshot_identity(
        Path(snapshot_path),
        expected_model=str(spec["model"]),
        expected_revision=str(spec["resolved_revision"]),
    )
    if identity_sha256 != spec["tokenizer_or_config_sha256"]:
        raise RuntimeReceiptError(
            "declared snapshot identity hash differs from live files"
        )
    boot_id, start_ticks, started_at = _process_start_evidence(
        pid=pid,
        proc_root=proc_root,
    )
    visible, observed_gpus = _cuda_visible_devices(
        pid=pid,
        proc_root=proc_root,
        inventory=inventory,
    )
    if observed_gpus != list(spec["gpu_uuids"]):
        raise RuntimeReceiptError(
            "live CUDA visibility differs from declared role GPU UUIDs"
        )
    socket_inode = _listening_socket_inode(
        pid=pid,
        port=parsed.port,
        proc_root=proc_root,
    )
    selected_environment = _selected_process_environment(
        pid=pid,
        proc_root=proc_root,
    )
    return {
        "server_pid": pid,
        "process_boot_id": boot_id,
        "process_start_time_ticks": start_ticks,
        "process_started_at_utc": started_at,
        "observed_cmdline_sha256": semantic_sha256(command),
        "observed_executable_path": str(observed_executable),
        "observed_executable_sha256": file_sha256(observed_executable),
        "model_snapshot_path": str(Path(snapshot_path).resolve()),
        "snapshot_identity_files": identity_files,
        "snapshot_identity_files_sha256": identity_sha256,
        "observed_cuda_visible_devices": visible,
        "observed_gpu_uuids": observed_gpus,
        "listening_socket_inode": socket_inode,
        "selected_process_environment": selected_environment,
    }


def http_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = (
        canonical(dict(payload)).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib_request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer v6-local",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib_request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            status = int(getattr(response, "status", 0))
            raw = response.read()
    except (OSError, urllib_error.URLError) as error:
        raise RuntimeReceiptError(
            f"model endpoint request failed: {url}"
        ) from error
    if status != 200:
        raise RuntimeReceiptError(
            f"model endpoint returned HTTP {status}: {url}"
        )
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeReceiptError(
            f"model endpoint returned malformed JSON: {url}"
        ) from error
    if not isinstance(decoded, dict):
        raise RuntimeReceiptError(
            f"model endpoint returned a non-object: {url}"
        )
    return decoded


def generation_request_with_peak(
    *,
    url: str,
    payload: Mapping[str, Any],
    gpu_uuids: Sequence[str],
    timeout_seconds: float,
    request_json: Callable[..., dict[str, Any]] = http_json,
    memory_sampler: Callable[[], dict[str, int]] = sample_gpu_memory_bytes,
) -> tuple[dict[str, Any], int, float]:
    stop = threading.Event()
    samples: list[int] = []
    sampling_errors: list[BaseException] = []
    expected = set(gpu_uuids)

    def sample() -> None:
        while not stop.is_set():
            try:
                observed = memory_sampler()
                if not expected.issubset(observed):
                    raise RuntimeReceiptError(
                        "GPU-memory probe omitted a role GPU UUID"
                    )
                samples.append(sum(observed[uuid] for uuid in expected))
            except BaseException as error:  # saved and raised in caller
                sampling_errors.append(error)
                stop.set()
                return
            stop.wait(0.05)

    worker = threading.Thread(
        target=sample,
        name="v610-gpu-memory-sampler",
        daemon=True,
    )
    worker.start()
    started = time.monotonic()
    try:
        response = request_json(
            url,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
    finally:
        elapsed = time.monotonic() - started
        stop.set()
        worker.join(timeout=5.0)
    if worker.is_alive():
        raise RuntimeReceiptError("GPU-memory sampler did not terminate")
    if sampling_errors:
        raise RuntimeReceiptError(
            f"GPU-memory sampling failed: {sampling_errors[0]}"
        ) from sampling_errors[0]
    if not samples or max(samples) <= 0 or elapsed <= 0.0:
        raise RuntimeReceiptError(
            "long-context probe lacks positive GPU-memory/time evidence"
        )
    return response, max(samples), elapsed


def _model_ids(payload: Mapping[str, Any]) -> list[str]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise RuntimeReceiptError("/models response has no data list")
    values = sorted(
        {
            str(row["id"])
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("id"), str)
            and row["id"]
        }
    )
    if len(values) != len(rows):
        raise RuntimeReceiptError(
            "/models contains malformed or duplicate model identities"
        )
    return values


def _probe_prompt() -> str:
    return (" probe" * LONG_CONTEXT_REPETITIONS).strip()


def probe_service(
    spec: Mapping[str, Any],
    *,
    timeout_seconds: float,
    request_json: Callable[..., dict[str, Any]] = http_json,
    generation_request: Callable[..., tuple[dict[str, Any], int, float]] = (
        generation_request_with_peak
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_base = str(spec["api_base"]).rstrip("/")
    model = str(spec["model"])
    checked_at = utc_now()
    models_payload = request_json(
        f"{api_base}/models",
        payload=None,
        timeout_seconds=timeout_seconds,
    )
    model_ids = _model_ids(models_payload)
    if model_ids != [model]:
        raise RuntimeReceiptError(
            f"exact /models identity mismatch: expected {[model]!r}, "
            f"observed {model_ids!r}"
        )
    prompt = _probe_prompt()
    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": PROBE_MAX_NEW_TOKENS,
        "temperature": PROBE_TEMPERATURE,
        "seed": PROBE_SEED,
        "stream": False,
    }
    response, peak_memory, latency = generation_request(
        url=f"{api_base}/chat/completions",
        payload=request_payload,
        gpu_uuids=list(spec["gpu_uuids"]),
        timeout_seconds=timeout_seconds,
        request_json=request_json,
    )
    usage = response.get("usage")
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = (
        first_choice.get("message")
        if isinstance(first_choice, Mapping)
        else None
    )
    content = message.get("content") if isinstance(message, Mapping) else None
    tool_calls = (
        message.get("tool_calls") if isinstance(message, Mapping) else None
    )
    if (
        not isinstance(usage, Mapping)
        or isinstance(usage.get("prompt_tokens"), bool)
        or not isinstance(usage.get("prompt_tokens"), int)
        or usage["prompt_tokens"] < 4096
        or isinstance(usage.get("completion_tokens"), bool)
        or not isinstance(usage.get("completion_tokens"), int)
        or not 1 <= usage["completion_tokens"] <= PROBE_MAX_NEW_TOKENS
        or not isinstance(choices, list)
        or not choices
        or not isinstance(first_choice, Mapping)
        or not isinstance(message, Mapping)
        or not (
            isinstance(content, str)
            and bool(content.strip())
            or isinstance(tool_calls, list)
            and bool(tool_calls)
        )
    ):
        raise RuntimeReceiptError(
            "long-context generation response lacks valid token evidence"
        )
    probe = {
        "status": "PASS",
        "prompt_sha256": semantic_sha256(prompt),
        "input_token_count": usage["prompt_tokens"],
        "max_new_tokens": PROBE_MAX_NEW_TOKENS,
        "generated_token_count": usage["completion_tokens"],
        "seed": PROBE_SEED,
        "temperature": PROBE_TEMPERATURE,
        "response_sha256": semantic_sha256(response),
        "latency_seconds": float(latency),
        "peak_gpu_memory_bytes": int(peak_memory),
    }
    if (
        not math.isfinite(probe["latency_seconds"])
        or probe["latency_seconds"] <= 0
        or probe["peak_gpu_memory_bytes"] <= 0
    ):
        raise RuntimeReceiptError(
            "long-context generation returned invalid runtime evidence"
        )
    return (
        {
            "status": "PASS",
            "observed_model_ids": model_ids,
            "checked_at_utc": checked_at,
        },
        probe,
    )


def load_service_specs(
    path: Path,
    *,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeReceiptError(
            f"service specs are not valid JSON: {path}"
        ) from error
    roles = payload.get("roles") if isinstance(payload, Mapping) else None
    available = {str(row["uuid"]) for row in inventory}
    expected_models = {
        "teacher": (
            generation.V610_TEACHER_MODEL,
            generation.V610_TEACHER_REVISION,
            generation.V610_ROLE_TENSOR_PARALLEL_SIZE["teacher"],
        ),
        "user": (
            generation.V610_USER_JUDGE_MODEL,
            generation.V610_USER_JUDGE_REVISION,
            generation.V610_ROLE_TENSOR_PARALLEL_SIZE["user"],
        ),
        "judge": (
            generation.V610_USER_JUDGE_MODEL,
            generation.V610_USER_JUDGE_REVISION,
            generation.V610_ROLE_TENSOR_PARALLEL_SIZE["judge"],
        ),
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != SERVICE_SPEC_KEYS
        or not isinstance(roles, dict)
        or set(roles) != set(ROLE_ORDER)
    ):
        raise RuntimeReceiptError("service spec top-level schema drift")
    normalized: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        row = roles.get(role)
        model, revision, tensor_parallel = expected_models[role]
        if (
            not isinstance(row, dict)
            or set(row) != ROLE_SPEC_KEYS
            or row.get("model") != model
            or row.get("resolved_revision") != revision
            or not isinstance(row.get("api_base"), str)
            or not row["api_base"].startswith(("http://", "https://"))
            or row.get("quantization")
            != generation.V610_MODEL_QUANTIZATION
            or row.get("dtype") != FROZEN_MODEL_DTYPE
            or row.get("tensor_parallel_size") != tensor_parallel
            or row.get("max_model_len") != FROZEN_MAX_MODEL_LEN
            or isinstance(row.get("gpu_memory_utilization"), bool)
            or not isinstance(
                row.get("gpu_memory_utilization"), (int, float)
            )
            or not math.isclose(
                float(row["gpu_memory_utilization"]),
                FROZEN_GPU_MEMORY_UTILIZATION,
                abs_tol=1e-12,
            )
            or not isinstance(row.get("gpu_uuids"), list)
            or len(row["gpu_uuids"]) != tensor_parallel
            or len(row["gpu_uuids"]) != len(set(row["gpu_uuids"]))
            or any(uuid not in available for uuid in row["gpu_uuids"])
            or not isinstance(row.get("launch_command"), list)
            or not row["launch_command"]
            or any(
                not isinstance(item, str) or not item
                for item in row["launch_command"]
            )
            or isinstance(row.get("server_pid"), bool)
            or not isinstance(row.get("server_pid"), int)
            or row["server_pid"] <= 0
            or generation.SHA256_RE.fullmatch(
                str(row.get("tokenizer_or_config_sha256"))
            )
            is None
        ):
            raise RuntimeReceiptError(
                f"service spec role path drift: roles.{role}"
            )
        normalized[role] = deepcopy(row)
    teacher_gpus = set(normalized["teacher"]["gpu_uuids"])
    user_gpus = set(normalized["user"]["gpu_uuids"])
    judge_gpus = set(normalized["judge"]["gpu_uuids"])
    if (
        teacher_gpus & user_gpus
        or teacher_gpus | user_gpus != available
        or user_gpus != judge_gpus
        or normalized["user"]["api_base"]
        != normalized["judge"]["api_base"]
        or normalized["user"]["model"] != normalized["judge"]["model"]
        or normalized["user"]["resolved_revision"]
        != normalized["judge"]["resolved_revision"]
        or normalized["user"]["launch_command"]
        != normalized["judge"]["launch_command"]
        or normalized["user"]["server_pid"]
        != normalized["judge"]["server_pid"]
        or normalized["teacher"]["server_pid"]
        == normalized["user"]["server_pid"]
        or normalized["user"]["tokenizer_or_config_sha256"]
        != normalized["judge"]["tokenizer_or_config_sha256"]
    ):
        raise RuntimeReceiptError(
            "service topology must be disjoint TP=2 teacher plus one shared "
            "user/judge GPU endpoint"
        )
    return normalized


def build_source_receipt(
    *,
    source_commit: str,
    generator_sha256: str,
    tau2_commit: str,
    container_image_digest: str,
    dependency_lock_sha256: str,
    runtime: Mapping[str, str],
    inventory: Sequence[Mapping[str, Any]],
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "protocol": generation.V610_SOURCE_CONTAINER_RECEIPT_PROTOCOL,
        "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "status": "PASS",
        "source_commit": source_commit,
        "generation_script_sha256": generator_sha256,
        "tau2_commit": tau2_commit,
        "container_image_digest": container_image_digest,
        "dependency_lock_sha256": dependency_lock_sha256,
        "runtime": deepcopy(dict(runtime)),
        "gpus": [deepcopy(dict(row)) for row in inventory],
        "created_at_utc": created_at_utc,
        "official_test_used": False,
    }


def build_model_receipt(
    *,
    specs: Mapping[str, Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    runtime_python: Path,
    container_image_digest: str,
    timeout_seconds: float,
    probe: Callable[
        ...,
        tuple[dict[str, Any], dict[str, Any]],
    ]
    | None = None,
    process_inspector: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probe = probe or probe_service
    process_inspector = process_inspector or inspect_vllm_process
    roles: dict[str, dict[str, Any]] = {}
    cache: dict[
        tuple[str, str, int, tuple[str, ...]],
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    for role in ROLE_ORDER:
        spec = specs[role]
        key = (
            str(spec["api_base"]).rstrip("/"),
            str(spec["model"]),
            int(spec["server_pid"]),
            tuple(str(value) for value in spec["gpu_uuids"]),
        )
        if key not in cache:
            before = process_inspector(
                spec,
                inventory=inventory,
                expected_runtime_python=runtime_python,
            )
            healthcheck, generation_probe = probe(
                spec,
                timeout_seconds=timeout_seconds,
            )
            after = process_inspector(
                spec,
                inventory=inventory,
                expected_runtime_python=runtime_python,
            )
            if before != after:
                raise RuntimeReceiptError(
                    "vLLM process/snapshot identity changed during live probe"
                )
            cache[key] = (before, healthcheck, generation_probe)
        process_evidence, healthcheck, generation_probe = cache[key]
        roles[role] = {
            "role": role,
            "model": spec["model"],
            "resolved_revision": spec["resolved_revision"],
            "api_base": spec["api_base"],
            "container_image_digest": container_image_digest,
            "quantization": spec["quantization"],
            "dtype": spec["dtype"],
            "tensor_parallel_size": spec["tensor_parallel_size"],
            "max_model_len": spec["max_model_len"],
            "gpu_memory_utilization": spec[
                "gpu_memory_utilization"
            ],
            "gpu_uuids": deepcopy(spec["gpu_uuids"]),
            "launch_command": deepcopy(spec["launch_command"]),
            "launch_command_sha256": semantic_sha256(
                spec["launch_command"]
            ),
            **deepcopy(process_evidence),
            "tokenizer_or_config_sha256": spec[
                "tokenizer_or_config_sha256"
            ],
            "launched_at_utc": process_evidence[
                "process_started_at_utc"
            ],
            "healthcheck": deepcopy(healthcheck),
            "generation_probe": deepcopy(generation_probe),
        }
    return {
        "protocol": generation.V610_MODEL_SERVER_RECEIPTS_PROTOCOL,
        "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "status": "PASS",
        "container_image_digest": container_image_digest,
        "roles": roles,
        "official_test_used": False,
    }


def _write_canonical(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(canonical(dict(value)), encoding="utf-8")


def build_receipts(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    tau2_root = args.tau2_root.resolve()
    dependency_lock = args.dependency_lock.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeReceiptError(
            f"refusing to overwrite runtime receipt directory: {output_dir}"
        )
    if (
        generation.OCI_DIGEST_RE.fullmatch(args.container_image_digest)
        is None
    ):
        raise RuntimeReceiptError("container image digest is invalid")
    if not dependency_lock.is_file():
        raise RuntimeReceiptError(
            f"dependency lock does not exist: {dependency_lock}"
        )
    source = git_identity(source_root)
    tau2 = git_identity(tau2_root)
    if tau2["commit"] != generation.protocol.TAU2_COMMIT:
        raise RuntimeReceiptError(
            "tau2 commit differs from the frozen V6.10 protocol"
        )
    generator_path = (
        source_root / "scripts" / "run_v6_candidate_generation.py"
    )
    if not generator_path.is_file():
        raise RuntimeReceiptError(
            f"candidate generator is absent: {generator_path}"
        )
    inventory, driver_version = capture_gpu_inventory()
    runtime_python = Path(os.path.abspath(args.runtime_python))
    runtime = capture_runtime(runtime_python)
    runtime["driver_version"] = driver_version
    specs = load_service_specs(
        args.service_specs.resolve(),
        inventory=inventory,
    )
    created_at = utc_now()
    source_receipt = build_source_receipt(
        source_commit=source["commit"],
        generator_sha256=file_sha256(generator_path),
        tau2_commit=tau2["commit"],
        container_image_digest=args.container_image_digest,
        dependency_lock_sha256=file_sha256(dependency_lock),
        runtime=runtime,
        inventory=inventory,
        created_at_utc=created_at,
    )
    model_receipt = build_model_receipt(
        specs=specs,
        inventory=inventory,
        runtime_python=runtime_python,
        container_image_digest=args.container_image_digest,
        timeout_seconds=args.request_timeout,
    )
    if any(
        role_row.get("observed_executable_sha256")
        != runtime["python_executable_sha256"]
        or (role_row.get("launch_command") or [None])[0]
        != runtime["python_executable_path"]
        for role_row in model_receipt["roles"].values()
    ):
        raise RuntimeReceiptError(
            "model server executable differs from observed runtime Python"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=str(output_dir.parent),
        )
    )
    try:
        source_path = temporary / SOURCE_RECEIPT_NAME
        model_path = temporary / MODEL_RECEIPT_NAME
        _write_canonical(source_path, source_receipt)
        _write_canonical(model_path, model_receipt)
        generation.load_source_container_provenance(
            source_path,
            expected_sha256=file_sha256(source_path),
            expected_source_commit=source["commit"],
            expected_generation_script_sha256=file_sha256(generator_path),
            expected_tau2_commit=tau2["commit"],
            container_image_digest=args.container_image_digest,
        )
        generation.load_model_server_receipts(
            model_path,
            expected_sha256=file_sha256(model_path),
            expected_roles={
                role: {
                    "model": specs[role]["model"],
                    "revision": specs[role]["resolved_revision"],
                    "api_base": specs[role]["api_base"],
                    "quantization": specs[role]["quantization"],
                    "tensor_parallel_size": specs[role][
                        "tensor_parallel_size"
                    ],
                }
                for role in ROLE_ORDER
            },
            container_image_digest=args.container_image_digest,
            available_gpu_uuids=[
                str(row["uuid"]) for row in inventory
            ],
            expected_python_executable_path=runtime[
                "python_executable_path"
            ],
            expected_python_executable_sha256=runtime[
                "python_executable_sha256"
            ],
        )
        hashes = {
            "protocol": "v6_10_runtime_receipt_hashes_v1",
            "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            "status": "PASS",
            "source_container_provenance": {
                "path": SOURCE_RECEIPT_NAME,
                "file_sha256": file_sha256(source_path),
                "semantic_sha256": semantic_sha256(source_receipt),
            },
            "model_server_receipts": {
                "path": MODEL_RECEIPT_NAME,
                "file_sha256": file_sha256(model_path),
                "semantic_sha256": semantic_sha256(model_receipt),
            },
            "created_at_utc": created_at,
            "official_test_used": False,
        }
        _write_canonical(temporary / HASH_MANIFEST_NAME, hashes)
        os.replace(temporary, output_dir)
    except BaseException:
        for path in sorted(temporary.glob("*")):
            if path.is_file():
                path.unlink()
        temporary.rmdir()
        raise
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "source_container_provenance_sha256": hashes[
            "source_container_provenance"
        ]["file_sha256"],
        "model_server_receipts_sha256": hashes[
            "model_server_receipts"
        ]["file_sha256"],
        "runtime_receipt_hashes_sha256": file_sha256(
            output_dir / HASH_MANIFEST_NAME
        ),
        "official_test_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--service-specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        isinstance(args.request_timeout, bool)
        or not isinstance(args.request_timeout, (int, float))
        or not math.isfinite(float(args.request_timeout))
        or args.request_timeout <= 0
    ):
        raise RuntimeReceiptError("request timeout must be positive")
    print(canonical(build_receipts(args)))


if __name__ == "__main__":
    main()
