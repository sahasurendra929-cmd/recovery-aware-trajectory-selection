#!/usr/bin/env python3
"""Fail-closed unattended controller for the V5.3 low-support diagnostic.

This controller is the only supported operational entrypoint for the
post-yield diagnostic.  It holds a diagnostic-exclusive lock and the source
V5.3-12h controller's shared lock for its whole lifetime, passes the source
lock descriptor to every scientific child, and never reports partial metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

if __package__:
    from . import build_v5_checkpoint_registry as registry_builder
    from . import prepare_v5_3_low_support_diagnostic as preparation
    from . import run_v5_3_single_host as base
    from . import run_v5_sft_causal_eval as evaluator
    from . import summarize_v5_3_low_support_diagnostic as summarizer
    from . import v5_3_12h_protocol as source_protocol
    from . import v5_3_low_support_protocol as protocol
    from .v5_dynamic_audit_contract import load_complete_dynamic_audit
else:
    import build_v5_checkpoint_registry as registry_builder
    import prepare_v5_3_low_support_diagnostic as preparation
    import run_v5_3_single_host as base
    import run_v5_sft_causal_eval as evaluator
    import summarize_v5_3_low_support_diagnostic as summarizer
    import v5_3_12h_protocol as source_protocol
    import v5_3_low_support_protocol as protocol
    from v5_dynamic_audit_contract import load_complete_dynamic_audit


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "artifacts/v5_stage0/manifests/split_manifest.json"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
MIN_GPU_MEMORY_MIB = 30_000
MIN_FREE_DISK_GIB = 20
USER_API_KEY = "diagnostic-user-judge-local"
AGENT_API_KEY = "diagnostic-agent-local"
TERM_GRACE_SECONDS = 30
_DIAGNOSTIC_LOCK: Any | None = None
_SOURCE_LOCK: Any | None = None
_ACTIVE: dict[int, subprocess.Popen[bytes]] = {}
_PROCESS_IDENTITIES: dict[int, dict[str, Any]] = {}
_ACTIVE_LOCK = threading.Lock()
_GLOBAL_ARGS: argparse.Namespace | None = None
_SIGNAL_ACTIVE = False


class DiagnosticControllerError(RuntimeError):
    """A scientific or operational barrier failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticControllerError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise DiagnosticControllerError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise DiagnosticControllerError(
                f"immutable artifact differs on resume: {path}"
            )
        return
    atomic_write(path, value)


def canonical_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["canonical_sha256"] = protocol.canonical_sha256(value)
    return value


def git_output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_processing_checkout(expected_commit: str) -> str:
    if (
        len(expected_commit) != 40
        or expected_commit.lower() != expected_commit
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise DiagnosticControllerError(
            "--expected-processing-commit must be a full lowercase commit"
        )
    observed = git_output("rev-parse", "HEAD^{commit}")
    if observed != expected_commit:
        raise DiagnosticControllerError(
            f"processing commit drift: expected={expected_commit}, observed={observed}"
        )
    if observed == protocol.SOURCE_GENERATION_COMMIT:
        raise DiagnosticControllerError(
            "post-yield processing commit must differ from generation commit"
        )
    for command, label in (
        (("diff", "--quiet", "--"), "unstaged"),
        (("diff", "--cached", "--quiet", "--"), "staged"),
    ):
        if subprocess.run(["git", *command], cwd=ROOT, check=False).returncode:
            raise DiagnosticControllerError(
                f"repository contains {label} tracked source drift"
            )
    return observed


def validate_roots(args: argparse.Namespace) -> None:
    expected = {
        "source_protocol_root": ROOT
        / "data/processed/v5_3_12h_screen_protocol",
        "source_raw_root": ROOT / "data/raw/v5_3_12h_screen_generation",
        "source_processed_root": ROOT / "data/processed/v5_3_12h_screen",
        "source_results_root": ROOT / "results/v5_3_12h_screen",
        "source_runtime_root": ROOT / protocol.SOURCE_RUNTIME_ROOT,
        "processed_root": protocol.artifact_root(ROOT, "processed_root"),
        "results_root": protocol.artifact_root(ROOT, "results_root"),
        "runtime_root": protocol.artifact_root(ROOT, "runtime_root"),
    }
    for field, path in expected.items():
        if getattr(args, field).resolve() != path.resolve():
            raise DiagnosticControllerError(
                f"{field} must equal frozen path {path.resolve()}"
            )
    diagnostic = {
        args.processed_root.resolve(),
        args.results_root.resolve(),
        args.runtime_root.resolve(),
    }
    source = {
        args.source_protocol_root.resolve(),
        args.source_raw_root.resolve(),
        args.source_processed_root.resolve(),
        args.source_results_root.resolve(),
        args.source_runtime_root.resolve(),
    }
    if len(diagnostic) != 3 or diagnostic & source:
        raise DiagnosticControllerError("diagnostic/source artifact roots overlap")


def acquire_whole_run_locks(args: argparse.Namespace) -> None:
    global _DIAGNOSTIC_LOCK, _SOURCE_LOCK
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = args.runtime_root / "controller.lock"
    diagnostic = diagnostic_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(diagnostic.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        diagnostic.close()
        raise DiagnosticControllerError(
            "another low-support diagnostic controller owns the EX lock"
        ) from error
    source_path = args.source_runtime_root / "controller.lock"
    if not source_path.is_file():
        diagnostic.close()
        raise DiagnosticControllerError("source controller lock file is absent")
    source = source_path.open("r", encoding="utf-8")
    try:
        fcntl.flock(source.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as error:
        source.close()
        diagnostic.close()
        raise DiagnosticControllerError(
            "source V5.3-12h controller is still active"
        ) from error
    diagnostic.seek(0)
    diagnostic.truncate()
    diagnostic.write(f"{os.getpid()}\n")
    diagnostic.flush()
    os.fsync(diagnostic.fileno())
    os.set_inheritable(source.fileno(), True)
    os.environ["V5_3_LOW_SUPPORT_LOCK_FD"] = str(source.fileno())
    _DIAGNOSTIC_LOCK = diagnostic
    _SOURCE_LOCK = source


def release_whole_run_locks() -> None:
    global _DIAGNOSTIC_LOCK, _SOURCE_LOCK
    os.environ.pop("V5_3_LOW_SUPPORT_LOCK_FD", None)
    for handle in (_SOURCE_LOCK, _DIAGNOSTIC_LOCK):
        if handle is None:
            continue
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
    _SOURCE_LOCK = None
    _DIAGNOSTIC_LOCK = None


def source_lock_fd() -> int:
    if _SOURCE_LOCK is None:
        raise DiagnosticControllerError("whole-run source SH lock is absent")
    protocol.require_whole_run_source_lock(ROOT)
    return _SOURCE_LOCK.fileno()


def offline_env(
    args: argparse.Namespace, *, gpu: int | None = None
) -> dict[str, str]:
    descriptor = source_lock_fd()
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(args.hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "V5_3_LOW_SUPPORT_LOCK_FD": str(descriptor),
        }
    )
    if gpu is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def linux_process_identity(pid: int) -> dict[str, Any]:
    """Return PID-reuse-resistant identity for one Linux process."""

    if not isinstance(pid, int) or pid <= 1:
        raise DiagnosticControllerError(f"unsafe process PID: {pid!r}")
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        cmdline = (proc / "cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as error:
        raise DiagnosticControllerError(
            f"cannot read live Linux process identity for PID {pid}"
        ) from error
    closing = stat.rfind(")")
    fields = stat[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20 or not boot_id:
        raise DiagnosticControllerError(
            f"malformed live Linux process identity for PID {pid}"
        )
    try:
        process_group_id = int(fields[2])
        start_time_ticks = int(fields[19])
    except ValueError as error:
        raise DiagnosticControllerError(
            f"non-integer Linux process identity for PID {pid}"
        ) from error
    return {
        "protocol": protocol.PROCESS_IDENTITY_PROTOCOL,
        "pid": pid,
        "process_group_id": process_group_id,
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def same_process_identity(
    expected: dict[str, Any], observed: dict[str, Any]
) -> bool:
    stable_fields = (
        "protocol",
        "pid",
        "process_group_id",
        "start_time_ticks",
        "boot_id",
    )
    return all(
        expected.get(field) == observed.get(field)
        for field in stable_fields
    )


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_service_process_identity(
    value: Any,
) -> dict[str, Any]:
    try:
        current_boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise DiagnosticControllerError(
            "cannot read current Linux boot identity"
        ) from error
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
        or value.get("protocol") != protocol.PROCESS_IDENTITY_PROTOCOL
        or not isinstance(value.get("pid"), int)
        or value["pid"] <= 1
        or value.get("process_group_id") != value["pid"]
        or not isinstance(value.get("start_time_ticks"), int)
        or value["start_time_ticks"] <= 0
        or value.get("boot_id") != current_boot_id
        or not valid_sha256(value.get("cmdline_sha256"))
        or value.get("cmdline_sha256")
        == hashlib.sha256(b"").hexdigest()
    ):
        raise DiagnosticControllerError(
            "invalid or stale service process identity receipt"
        )
    return dict(value)


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def register_process(process: subprocess.Popen[bytes]) -> None:
    identity = linux_process_identity(process.pid)
    if identity["process_group_id"] != process.pid:
        raise DiagnosticControllerError(
            "scientific child did not become its own process-group leader"
        )
    with _ACTIVE_LOCK:
        _ACTIVE[process.pid] = process
        _PROCESS_IDENTITIES[process.pid] = identity


def unregister_process(process: subprocess.Popen[bytes]) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(process.pid, None)
        _PROCESS_IDENTITIES.pop(process.pid, None)


def registered_process_identity(
    process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    with _ACTIVE_LOCK:
        identity = _PROCESS_IDENTITIES.get(process.pid)
    if identity is None:
        raise DiagnosticControllerError(
            f"missing registered process identity for PID {process.pid}"
        )
    return dict(identity)


def spawn_logged(
    args: argparse.Namespace,
    command: list[str],
    *,
    log_path: Path,
    gpu: int | None = None,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    handle.write(("\n$ " + " ".join(command) + "\n").encode("utf-8"))
    handle.flush()
    descriptor = source_lock_fd()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=offline_env(args, gpu=gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(descriptor,),
    )
    try:
        register_process(process)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        handle.close()
        raise
    return process, handle


def stop_process(process: subprocess.Popen[bytes]) -> None:
    identity = registered_process_identity(process)
    process_group_id = identity["process_group_id"]
    if not process_group_exists(process_group_id):
        return
    proc_path = Path("/proc") / str(identity["pid"])
    if proc_path.exists():
        observed = linux_process_identity(identity["pid"])
        if not same_process_identity(identity, observed):
            raise DiagnosticControllerError(
                "refusing to signal a PID-reused process group"
            )
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    if process_group_exists(process_group_id):
        if proc_path.exists():
            observed = linux_process_identity(identity["pid"])
            if not same_process_identity(identity, observed):
                raise DiagnosticControllerError(
                    "refusing SIGKILL after process identity changed"
                )
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 10
        while (
            process_group_exists(process_group_id)
            and time.monotonic() < kill_deadline
        ):
            process.poll()
            time.sleep(0.1)
        if process_group_exists(process_group_id):
            raise DiagnosticControllerError(
                f"process group {process_group_id} survived SIGKILL"
            )
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                return
            process.wait(timeout=10)


def cleanup_processes() -> None:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE.values())
    for process in processes:
        try:
            stop_process(process)
        except Exception as error:
            print(
                f"cleanup warning for PID {process.pid}: {error}",
                file=sys.stderr,
            )
        finally:
            unregister_process(process)


def wait_processes(
    rows: Iterable[tuple[subprocess.Popen[bytes], Any, str]]
) -> None:
    values = list(rows)
    failures: list[tuple[str, int]] = []
    try:
        active = list(values)
        while active:
            remaining = []
            for process, handle, label in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, handle, label))
                elif code:
                    failures.append((label, code))
            if failures:
                for process, _handle, _label in remaining:
                    stop_process(process)
                raise DiagnosticControllerError(
                    f"parallel child failure(s): {failures}"
                )
            active = remaining
            if active:
                time.sleep(1)
    finally:
        for process, handle, _ in values:
            try:
                stop_process(process)
            finally:
                unregister_process(process)
                handle.close()


def terminal_path(args: argparse.Namespace) -> Path:
    return args.results_root / "TERMINAL_STATUS.json"


def write_terminal(
    args: argparse.Namespace,
    *,
    status: str,
    reason: str,
    no_claim: bool,
) -> None:
    payload = canonical_receipt(
        {
            "protocol": f"{protocol.CONTROLLER_PROTOCOL}:terminal_v1",
            "status": status,
            "reason": reason,
            "timestamp_utc": utc_now(),
            "processing_source_commit": args.expected_processing_commit,
            "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
            "post_yield_exploratory_diagnostic": True,
            "formal_v5_3_result": False,
            "paper_level_confirmation": False,
            "partial_metrics_reported": False,
            "no_scientific_claim": no_claim,
            "official_test_used": False,
            "official_test_sealed": True,
            "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
        }
    )
    path = terminal_path(args)
    if path.exists():
        existing = read_json(path)
        if existing.get("status") == "DIAGNOSTIC_COMPLETE_DESCRIPTIVE_ONLY":
            return
        # Preserve the first terminal cause rather than rewriting history.
        return
    atomic_write(path, payload)


def signal_handler(signum: int, _frame: Any) -> None:
    global _SIGNAL_ACTIVE
    if _SIGNAL_ACTIVE:
        return
    _SIGNAL_ACTIVE = True
    args = _GLOBAL_ARGS
    if args is not None:
        write_terminal(
            args,
            status="INCOMPLETE_NO_CLAIM",
            reason=f"CONTROLLER_SIGNAL_{signal.Signals(signum).name}",
            no_claim=True,
        )
    cleanup_processes()
    raise DiagnosticControllerError("controller interrupted")


def stage_receipt_path(args: argparse.Namespace, stage: str) -> Path:
    return args.runtime_root / "stages" / f"{stage}.json"


def complete_stage(
    args: argparse.Namespace,
    stage: str,
    *,
    bindings: dict[str, str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for label, digest in bindings.items():
        if (
            not isinstance(label, str)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise DiagnosticControllerError(f"{stage}: invalid binding {label}")
    payload = canonical_receipt(
        {
            "protocol": f"{protocol.CONTROLLER_PROTOCOL}:stage_v1",
            "stage": stage,
            "status": "PASS",
            "completed_at_utc": utc_now(),
            "processing_source_commit": args.expected_processing_commit,
            "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
            "artifact_sha256": dict(sorted(bindings.items())),
            "details": details or {},
            "official_test_used": False,
            "official_test_sealed": True,
        }
    )
    write_immutable(stage_receipt_path(args, stage), payload)
    return payload


def source_protocol_files(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "generation": args.source_protocol_root
        / "source_generation_manifest.json",
        "validation": args.source_protocol_root / "validation_manifest.json",
        "screen": args.source_protocol_root / "screen_manifest.json",
        "generation_audit": args.source_protocol_root
        / "generation_dynamic_audit.json",
        "validation_audit": args.source_protocol_root
        / "validation_dynamic_audit.json",
    }


def validate_source_terminal(terminal: dict[str, Any]) -> None:
    """Admit only the controller-normalized strict abundance-gate terminal."""

    if (
        terminal.get("protocol")
        != f"{source_protocol.PROTOCOL}:terminal_v1"
        or terminal.get("status") != "DATA_GATE_FAIL_NO_TRAIN"
        or terminal.get("reason") != "DO_NOT_TRAIN"
        or terminal.get("no_scientific_claim") is not True
        or terminal.get("official_test_used") is not False
        or terminal.get("official_test_sealed") is not True
        or terminal.get("source_generation_commit")
        != protocol.SOURCE_GENERATION_COMMIT
        or not isinstance(terminal.get("processing_source_commit"), str)
        or len(terminal["processing_source_commit"]) != 40
        or terminal["processing_source_commit"].lower()
        != terminal["processing_source_commit"]
        or any(
            character not in "0123456789abcdef"
            for character in terminal["processing_source_commit"]
        )
        or terminal["processing_source_commit"]
        == protocol.SOURCE_GENERATION_COMMIT
        or not valid_sha256(terminal.get("source_audit_sha256"))
        or not valid_sha256(terminal.get("source_hashes_sha256"))
        or terminal.get("source_snapshot_receipt_sha256")
        != sha256_file(preparation.SOURCE_SNAPSHOT_RECEIPT)
        or terminal.get("source_snapshot_file_ledger_sha256")
        != preparation.source_snapshot.canonical_sha256(
            read_json(preparation.SOURCE_SNAPSHOT_RECEIPT)["file_sha256"]
        )
    ):
        raise DiagnosticControllerError(
            "source terminal must be exact DATA_GATE_FAIL_NO_TRAIN / "
            "DO_NOT_TRAIN with sealed provenance and no claim"
        )


def validate_source_admission(args: argparse.Namespace) -> dict[str, Any]:
    files = source_protocol_files(args)
    if any(not path.is_file() for path in files.values()):
        raise DiagnosticControllerError("source protocol snapshot is incomplete")
    terminal_file = args.source_results_root / "TERMINAL_STATUS.json"
    audit_file = args.source_processed_root / "audit.json"
    hashes_file = args.source_processed_root / "hashes.json"
    if not all(path.is_file() for path in (terminal_file, audit_file, hashes_file)):
        raise DiagnosticControllerError(
            "source TERMINAL_STATUS/audit/hashes evidence is incomplete"
        )
    terminal = read_json(terminal_file)
    audit = read_json(audit_file)
    hashes = read_json(hashes_file)
    validate_source_terminal(terminal)
    try:
        snapshot_identity = (
            preparation.source_snapshot.validate_source_snapshot(
                raw_dir=args.source_raw_root,
                receipt_path=preparation.SOURCE_SNAPSHOT_RECEIPT,
                expected_generation_commit=(
                    protocol.SOURCE_GENERATION_COMMIT
                ),
            )
        )
    except preparation.source_snapshot.SourceSnapshotError as error:
        raise DiagnosticControllerError(str(error)) from error
    pair_counts = {
        task: int((audit.get("task_yield") or {}).get(task, {}).get(
            "selected_pairs", -1
        ))
        for task in source_protocol.PILOT_TASK_IDS
    }
    expected_gate = source_protocol.data_gate(pair_counts)
    frozen_file_ledger = read_json(
        preparation.SOURCE_SNAPSHOT_RECEIPT
    )["file_sha256"]
    frozen_raw_files = {
        str((args.source_raw_root / name).resolve()): digest
        for name, digest in frozen_file_ledger.items()
        if not name.startswith("run_contract.")
    }
    task_yield = audit.get("task_yield")
    if (
        audit.get("protocol") != preparation.v5.DATA_AUDIT_PROTOCOL
        or audit.get("design_protocol") != source_protocol.DATA_PROTOCOL
        or audit.get("design_version") != source_protocol.DESIGN_VERSION
        or audit.get("screen_protocol") != source_protocol.PROTOCOL
        or audit.get("split_manifest_sha256") != base.SPLIT_SHA256
        or audit.get("generation_manifest_sha256")
        != sha256_file(files["generation"])
        or audit.get("screen_manifest_sha256")
        != sha256_file(files["screen"])
        or audit.get("validation_manifest_sha256")
        != sha256_file(files["validation"])
        or audit.get("screen_task_ids")
        != list(source_protocol.PILOT_TASK_IDS)
        or audit.get("derived_validation_used_for_supervision") is not False
        or audit.get("claim_boundary")
        != dict(source_protocol.CLAIM_BOUNDARY)
        or audit.get("raw_files") != frozen_raw_files
        or audit.get("source_snapshot") != snapshot_identity
        or not isinstance(task_yield, dict)
        or set(task_yield) != set(source_protocol.PILOT_TASK_IDS)
        or any(
            not isinstance(value, dict)
            or isinstance(value.get("selected_pairs"), bool)
            or not isinstance(value.get("selected_pairs"), int)
            or value["selected_pairs"] < 0
            or value["selected_pairs"] > source_protocol.MAX_PAIRS_PER_TASK
            for value in task_yield.values()
        )
        or "matching" in audit
        or audit.get("status") != "FAIL_CLOSED"
        or audit.get("decision") != "DO_NOT_TRAIN"
        or audit.get("arm_files_written") is not False
        or audit.get("expected_rollouts") != protocol.EXPECTED_ROLLOUTS
        or audit.get("processing_source_commit")
        != args.expected_processing_commit
        or audit.get("source_generation_commit")
        != protocol.SOURCE_GENERATION_COMMIT
        or (audit.get("generation_contracts") or {}).get(
            "source_generation_commit"
        )
        != protocol.SOURCE_GENERATION_COMMIT
        or audit.get("official_test_used") is not False
        or audit.get("official_test_sealed") is not True
        or audit.get("data_gate") != expected_gate
        or expected_gate.get("status") != "FAIL_CLOSED"
        or expected_gate.get("training_authorized") is not False
    ):
        raise DiagnosticControllerError(
            "source audit is not the registered strict-gate failure"
        )
    if (
        terminal.get("processing_source_commit")
        != args.expected_processing_commit
        or terminal.get("source_audit_sha256") != sha256_file(audit_file)
        or terminal.get("source_hashes_sha256") != sha256_file(hashes_file)
    ):
        raise DiagnosticControllerError(
            "source terminal provenance does not bind source audit/hashes"
        )
    if (
        set(hashes) != {"audit.json", "attempt_audit.jsonl"}
        or hashes.get("audit.json") != sha256_file(audit_file)
        or any(
            Path(name).is_absolute()
            or not (args.source_processed_root / name)
            .resolve()
            .is_relative_to(args.source_processed_root.resolve())
            or not (args.source_processed_root / name).is_file()
            or sha256_file(args.source_processed_root / name) != digest
            for name, digest in hashes.items()
        )
    ):
        raise DiagnosticControllerError("source data-audit hash binding drift")
    dynamic_identity = load_complete_dynamic_audit(
        files["generation_audit"],
        manifest_path=files["generation"],
        split_manifest_path=SPLIT,
        expected_source_split="derived_inner_train",
        expected_task_ids={
            f"{row['domain']}:{row['task_id']}"
            for row in read_json(files["generation"])["rows"]
        },
    )
    contract_audit = preparation.validate_generation_contracts(
        raw_dir=args.source_raw_root,
        screen_manifest_path=files["screen"],
        generation_manifest_path=files["generation"],
        dynamic_identity=dynamic_identity,
        expected_source_commit=protocol.SOURCE_GENERATION_COMMIT,
    )
    declared_names: set[str] = set()
    exact_cases = 0
    contracts: dict[str, str] = {}
    for shard in range(protocol.NUM_GENERATION_SHARDS):
        path = args.source_raw_root / (
            f"run_contract.shard-{shard:03d}-of-"
            f"{protocol.NUM_GENERATION_SHARDS:03d}.json"
        )
        contract = read_json(path)
        declared = contract.get("result_sha256")
        if (
            contract.get("status") != "COMPLETE"
            or contract.get("source_commit")
            != protocol.SOURCE_GENERATION_COMMIT
            or not isinstance(declared, dict)
            or not declared
            or declared_names & set(declared)
        ):
            raise DiagnosticControllerError(
                f"source generation contract {shard} is incomplete/drifted"
            )
        declared_names.update(declared)
        for name, digest in declared.items():
            if Path(name).name != name:
                raise DiagnosticControllerError(
                    f"source result path traversal: {name}"
                )
            result = args.source_raw_root / name
            payload = read_json(result)
            simulations = payload.get("simulations")
            if (
                sha256_file(result) != digest
                or not isinstance(simulations, list)
            ):
                raise DiagnosticControllerError(
                    f"source result binding drift: {name}"
                )
            exact_cases += len(simulations)
        contracts[str(path.resolve())] = sha256_file(path)
    if (
        exact_cases != protocol.EXPECTED_ROLLOUTS
        or contract_audit.get("task_union") != protocol.TASKS
        or contract_audit.get("shards") != protocol.NUM_GENERATION_SHARDS
        or contract_audit.get("source_generation_commit")
        != protocol.SOURCE_GENERATION_COMMIT
    ):
        raise DiagnosticControllerError(
            f"source generation must contain exact 288 cases; found {exact_cases}"
        )
    if any(
        (args.source_results_root / name).exists()
        for name in ("training", "evaluation")
    ):
        raise DiagnosticControllerError(
            "source strict-gate failure unexpectedly contains training/evaluation"
        )
    payload = canonical_receipt(
        {
            "protocol": f"{protocol.CONTROLLER_PROTOCOL}:source_admission_v1",
            "status": "PASS",
            "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
            "source_audit_processing_commit": (
                args.expected_processing_commit
            ),
            "exact_generation_contracts": 3,
            "exact_raw_cases": exact_cases,
            "source_terminal_status": terminal.get("status"),
            "source_strict_gate_status": expected_gate["status"],
            "source_training_started": False,
            "source_terminal_sha256": sha256_file(terminal_file),
            "source_audit_sha256": sha256_file(audit_file),
            "source_hashes_sha256": sha256_file(hashes_file),
            "source_snapshot_receipt_sha256": snapshot_identity[
                "receipt_sha256"
            ],
            "source_snapshot_file_ledger_sha256": snapshot_identity[
                "file_ledger_sha256"
            ],
            "source_contract_sha256": dict(sorted(contracts.items())),
            "strict_judge_evidence_mapping_sha256": contract_audit[
                "strict_judge_evidence_mapping_sha256"
            ],
            "official_test_used": False,
            "official_test_sealed": True,
        }
    )
    write_immutable(
        args.runtime_root / "source_admission_receipt.json", payload
    )
    complete_stage(
        args,
        "00_source_admission",
        bindings={
            "source_admission_receipt.json": sha256_file(
                args.runtime_root / "source_admission_receipt.json"
            )
        },
        details={"exact_raw_cases": exact_cases},
    )
    return payload


def run_capture(
    args: argparse.Namespace,
    command: list[str],
    *,
    gpu: int | None = None,
    cwd: Path = ROOT,
) -> str:
    descriptor = source_lock_fd()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=offline_env(args, gpu=gpu),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=(descriptor,),
    )
    try:
        register_process(process)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        raise
    try:
        stdout, stderr = process.communicate()
    finally:
        try:
            stop_process(process)
        finally:
            unregister_process(process)
    if process.returncode:
        raise DiagnosticControllerError(
            f"preflight command failed ({process.returncode}): "
            f"{' '.join(command)}\n{stderr}"
        )
    return stdout


def cuda_compute_processes(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = run_capture(
        args,
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",", 3)]
        if len(values) != 4:
            raise DiagnosticControllerError(
                f"invalid nvidia-smi compute-app row: {line}"
            )
        try:
            pid = int(values[1])
            used_memory_mib = int(values[3])
        except ValueError as error:
            raise DiagnosticControllerError(
                f"non-integer nvidia-smi compute-app row: {line}"
            ) from error
        rows.append(
            {
                "gpu_uuid": values[0],
                "pid": pid,
                "process_name": values[2],
                "used_memory_mib": used_memory_mib,
            }
        )
    return rows


def require_no_cuda_compute_processes(
    args: argparse.Namespace, *, barrier: str
) -> list[dict[str, Any]]:
    rows = cuda_compute_processes(args)
    if rows:
        raise DiagnosticControllerError(
            f"{barrier}: stale CUDA compute processes detected: {rows}"
        )
    return rows


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not SPLIT.is_file() or sha256_file(SPLIT) != base.SPLIT_SHA256:
        raise DiagnosticControllerError("Stage-0 split hash drift")
    if (
        run_capture(
            args,
            ["git", "rev-parse", "HEAD^{commit}"],
            cwd=args.tau2_root,
        ).strip()
        != TAU2_COMMIT
    ):
        raise DiagnosticControllerError("tau2 commit drift")
    if not args.serve_python.is_file() or not args.train_python.is_file():
        raise DiagnosticControllerError("frozen serve/train Python is absent")
    if not args.vllm.is_file():
        raise DiagnosticControllerError("frozen vLLM executable is absent")
    gpu_text = run_capture(
        args,
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )
    gpus = []
    for line in gpu_text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4:
            raise DiagnosticControllerError(f"invalid nvidia-smi row: {line}")
        gpus.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_mib": int(values[2]),
                "driver": values[3],
            }
        )
    if (
        len(gpus) != 4
        or any(
            row["index"] != index
            or row["name"] != EXPECTED_GPU_MODEL
            or row["memory_mib"] < MIN_GPU_MEMORY_MIB
            for index, row in enumerate(gpus)
        )
    ):
        raise DiagnosticControllerError(f"exact 4×RTX5090 host required: {gpus}")
    occupied = [
        port
        for port in (8001, 8101, 8102, 8103)
        if not base.port_is_free(port)
    ]
    if occupied:
        raise DiagnosticControllerError(
            f"diagnostic evaluation ports already occupied: {occupied}"
        )
    free_gib = shutil.disk_usage(args.workspace_root).free / 1024**3
    if free_gib < MIN_FREE_DISK_GIB:
        raise DiagnosticControllerError(
            f"insufficient free disk: {free_gib:.1f} GiB"
        )
    serve = json.loads(
        run_capture(
            args,
            [
                str(args.serve_python),
                "-c",
                (
                    "import json,sys,transformers,vllm;"
                    "print(json.dumps({'python':sys.version.split()[0],"
                    "'transformers':transformers.__version__,"
                    "'vllm':vllm.__version__}))"
                ),
            ],
        ).strip().splitlines()[-1]
    )
    train = json.loads(
        run_capture(
            args,
            [
                str(args.train_python),
                "-c",
                (
                    "import json,sys,torch,transformers,peft,bitsandbytes,"
                    "datasets,accelerate;"
                    "print(json.dumps({'python':sys.version.split()[0],"
                    "'torch':torch.__version__,'cuda':torch.version.cuda,"
                    "'cuda_available':torch.cuda.is_available(),"
                    "'bf16_supported':torch.cuda.is_bf16_supported(),"
                    "'transformers':transformers.__version__,"
                    "'peft':peft.__version__,"
                    "'bitsandbytes':bitsandbytes.__version__,"
                    "'datasets':datasets.__version__,"
                    "'accelerate':accelerate.__version__}))"
                ),
            ],
            gpu=0,
        ).strip().splitlines()[-1]
    )
    targeted_tests = run_capture(
        args,
        [
            str(args.train_python),
            "-m",
            "unittest",
            "tests.test_v5_3_low_support_diagnostic",
            "tests.test_v5_3_low_support_evaluation",
            "tests.test_v5_3_low_support_controller",
        ],
    )
    if (
        serve.get("transformers") != "4.55.4"
        or serve.get("vllm") != "0.10.2"
        or train.get("cuda") != "12.8"
        or train.get("cuda_available") is not True
        or train.get("bf16_supported") is not True
        or not str(train.get("torch", "")).startswith("2.7.1")
        or train.get("transformers") != "4.52.4"
        or train.get("peft") != "0.15.2"
        or train.get("bitsandbytes") != "0.46.0"
        or train.get("datasets") != "3.6.0"
        or train.get("accelerate") != "1.7.0"
    ):
        raise DiagnosticControllerError(
            f"frozen serve/train environment drift: {serve} / {train}"
        )
    cuda_processes = require_no_cuda_compute_processes(
        args, barrier="preflight"
    )
    payload = canonical_receipt(
        {
            "protocol": f"{protocol.CONTROLLER_PROTOCOL}:preflight_v1",
            "status": "PASS",
            "processing_source_commit": args.expected_processing_commit,
            "tau2_commit": TAU2_COMMIT,
            "split_manifest_sha256": sha256_file(SPLIT),
            "gpus": gpus,
            "free_disk_gib": round(free_gib, 2),
            "serve_environment": serve,
            "train_environment": train,
            "cuda_compute_processes": cuda_processes,
            "targeted_unit_tests": {
                "status": "PASS",
                "modules": 3,
                "stdout_sha256": hashlib.sha256(
                    targeted_tests.encode("utf-8")
                ).hexdigest(),
            },
            "hf_home": str(args.hf_home),
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            "official_test_used": False,
        }
    )
    path = args.results_root / "preflight.json"
    write_immutable(path, payload)
    complete_stage(
        args,
        "01_preflight",
        bindings={"preflight.json": sha256_file(path)},
    )
    return payload


def data_complete(args: argparse.Namespace) -> bool:
    audit_path = args.processed_root / "audit.json"
    hashes_path = args.processed_root / "hashes.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        return False
    try:
        audit = read_json(audit_path)
        hashes = read_json(hashes_path)
        return bool(
            audit.get("status") == "PASS"
            and audit.get("decision")
            == "LOW_SUPPORT_DIAGNOSTIC_TRAINING_AUTHORIZED"
            and audit.get("processing_source_commit")
            == args.expected_processing_commit
            and audit.get("source_generation_commit")
            == protocol.SOURCE_GENERATION_COMMIT
            and audit.get("data_gate", {}).get("status")
            == "PASS_LOW_SUPPORT_DIAGNOSTIC"
            and audit.get("data_gate", {}).get("training_authorized") is True
            and audit.get("data_gate", {})
            .get("source_strict_gate", {})
            .get("status")
            == "FAIL_CLOSED"
            and set(audit.get("arms") or {}) == set(protocol.TRAINED_ARMS)
            and all(
                not Path(name).is_absolute()
                and (args.processed_root / name).resolve().is_relative_to(
                    args.processed_root.resolve()
                )
                and (args.processed_root / name).is_file()
                and sha256_file(args.processed_root / name) == digest
                for name, digest in hashes.items()
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def validate_pair_ledger_equivalence(
    source_audit: dict[str, Any],
    diagnostic_audit: dict[str, Any],
) -> None:
    if (
        diagnostic_audit.get("task_yield")
        != source_audit.get("task_yield")
        or (diagnostic_audit.get("data_gate") or {}).get(
            "source_strict_gate"
        )
        != source_audit.get("data_gate")
        or diagnostic_audit.get("source_snapshot")
        != source_audit.get("source_snapshot")
    ):
        raise DiagnosticControllerError(
            "diagnostic pair ledger differs from the source strict audit"
        )


def prepare(args: argparse.Namespace) -> None:
    if args.processed_root.exists():
        if not data_complete(args):
            raise DiagnosticControllerError(
                "existing diagnostic processed root is partial/invalid"
            )
    else:
        files = source_protocol_files(args)
        command = [
            str(args.train_python),
            "scripts/prepare_v5_3_low_support_diagnostic.py",
            "--split-manifest", str(SPLIT),
            "--generation-manifest", str(files["generation"]),
            "--validation-manifest", str(files["validation"]),
            "--screen-manifest", str(files["screen"]),
            "--generation-dynamic-audit", str(files["generation_audit"]),
            "--validation-dynamic-audit", str(files["validation_audit"]),
            "--source-snapshot-receipt",
            str(preparation.SOURCE_SNAPSHOT_RECEIPT),
            "--raw-dir", str(args.source_raw_root),
            "--tau2-root", str(args.tau2_root),
            "--source-runtime-root", str(args.source_runtime_root),
            "--output-dir", str(args.processed_root),
            "--tokenizer", protocol.STUDENT_MODEL,
            "--tokenizer-revision", protocol.STUDENT_REVISION,
            "--expected-source-commit", args.expected_processing_commit,
            "--expected-generation-source-commit",
            protocol.SOURCE_GENERATION_COMMIT,
            "--local-files-only",
        ]
        process, handle = spawn_logged(
            args,
            command,
            gpu=0,
            log_path=args.results_root / "logs/prepare.log",
        )
        wait_processes([(process, handle, "prepare")])
    if not data_complete(args):
        raise DiagnosticControllerError(
            "low-support 8-task/10-pair gate did not authorize training"
        )
    source_audit = read_json(
        args.source_processed_root / "audit.json"
    )
    diagnostic_audit = read_json(args.processed_root / "audit.json")
    validate_pair_ledger_equivalence(source_audit, diagnostic_audit)
    bindings = {
        str(path.relative_to(args.processed_root)): sha256_file(path)
        for path in (
            args.processed_root / "audit.json",
            args.processed_root / "hashes.json",
            args.processed_root / "validation_loss.jsonl",
            *[
                args.processed_root / "arms" / arm / "train.jsonl"
                for arm in protocol.TRAINED_ARMS
            ],
        )
    }
    complete_stage(args, "02_prepare", bindings=bindings)


def training_complete(
    args: argparse.Namespace, arm: str, mode: str
) -> bool:
    root = args.results_root / "training" / arm / mode
    try:
        manifest = read_json(root / "run_manifest.json")
        checkpoint = root / "checkpoint_final"
        expected_steps = 2 if mode == "smoke" else protocol.OPTIMIZER_STEPS
        expected_rows = 16 if mode == "smoke" else protocol.SCHEDULE_ROWS
        recorded = (manifest.get("checkpoint") or {}).get("file_sha256")
        if (
            manifest.get("protocol") != registry_builder.TRAIN_PROTOCOL
            or manifest.get("source_commit")
            != args.expected_processing_commit
            or manifest.get("model") != protocol.STUDENT_MODEL
            or manifest.get("model_revision") != protocol.STUDENT_REVISION
            or manifest.get("arm") != arm
            or manifest.get("mode") != mode
            or manifest.get("seed") != protocol.BASE_SEED
            or manifest.get("effective_steps") != expected_steps
            or manifest.get("effective_grad_accum") != 8
            or manifest.get("effective_rows") != expected_rows
            or manifest.get("effective_validation_rows") != 0
            or manifest.get("validation_disabled_reason")
            != "fixed_steps_exploratory"
            or (manifest.get("data_provenance") or {}).get("design_version")
            != protocol.DESIGN_VERSION
            or (manifest.get("data_provenance") or {})
            .get("design_provenance", {})
            .get("source_generation_commit")
            != protocol.SOURCE_GENERATION_COMMIT
            or not (checkpoint / "adapter_model.safetensors").is_file()
            or not (checkpoint / "adapter_config.json").is_file()
            or (manifest.get("loss_audit") or {}).get("finite") is not True
            or not isinstance(recorded, dict)
            or recorded.get("adapter_model.safetensors")
            != sha256_file(checkpoint / "adapter_model.safetensors")
            or recorded.get("adapter_config.json")
            != sha256_file(checkpoint / "adapter_config.json")
        ):
            return False
        if mode == "formal":
            registry_builder.validate_run(
                arm=arm,
                run_dir=root,
                source_commit=args.expected_processing_commit,
                base_revision=protocol.STUDENT_REVISION,
                provenance_profile=protocol.REGISTRY_PROFILE,
            )
        return True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def train_mode(args: argparse.Namespace, mode: str) -> None:
    hashes = read_json(args.processed_root / "hashes.json")
    workers = []
    for gpu, arm in enumerate(protocol.TRAINED_ARMS):
        if training_complete(args, arm, mode):
            continue
        output = args.results_root / "training" / arm / mode
        if output.exists() and any(output.iterdir()):
            raise DiagnosticControllerError(
                f"partial training output must be audited: {output}"
            )
        command = [
            str(args.train_python),
            "scripts/train_v5_sft_causal.py",
            "--train-file",
            str(args.processed_root / "arms" / arm / "train.jsonl"),
            "--validation-file",
            str(args.processed_root / "validation_loss.jsonl"),
            "--data-audit", str(args.processed_root / "audit.json"),
            "--data-hashes", str(args.processed_root / "hashes.json"),
            "--output-dir", str(output),
            "--arm", arm,
            "--mode", mode,
            "--model-revision", protocol.STUDENT_REVISION,
            "--expected-source-commit", args.expected_processing_commit,
            "--expected-train-sha256",
            hashes[f"arms/{arm}/train.jsonl"],
            "--expected-validation-sha256",
            hashes["validation_loss.jsonl"],
            "--local-files-only",
        ]
        process, handle = spawn_logged(
            args,
            command,
            gpu=gpu,
            log_path=args.results_root
            / "logs"
            / f"train-{mode}-{arm}-gpu{gpu}.log",
        )
        workers.append((process, handle, f"{mode}-{arm}"))
    wait_processes(workers)
    if not all(training_complete(args, arm, mode) for arm in protocol.TRAINED_ARMS):
        raise DiagnosticControllerError(f"{mode} training barrier failed")
    bindings = {
        f"{arm}/run_manifest.json": sha256_file(
            args.results_root / "training" / arm / mode / "run_manifest.json"
        )
        for arm in protocol.TRAINED_ARMS
    }
    complete_stage(
        args,
        "03_smoke_training" if mode == "smoke" else "04_formal_training",
        bindings=bindings,
        details={"parallel_arms": list(protocol.TRAINED_ARMS)},
    )


def registry_complete(args: argparse.Namespace) -> bool:
    path = args.results_root / "checkpoint_registry.json"
    try:
        value = evaluator.load_checkpoint_registry(
            path, expected_profile=protocol.REGISTRY_PROFILE
        )
        return (
            value.get("source_commit") == args.expected_processing_commit
            and value.get("base_model_revision") == protocol.STUDENT_REVISION
            and set(value.get("entries") or {}) == set(protocol.EVAL_ARMS)
        )
    except (OSError, TypeError, ValueError, RuntimeError):
        return False


def build_registry(args: argparse.Namespace) -> None:
    output = args.results_root / "checkpoint_registry.json"
    if not registry_complete(args):
        if output.exists():
            raise DiagnosticControllerError("invalid checkpoint registry exists")
        command = [
            str(args.train_python),
            "scripts/build_v5_checkpoint_registry.py",
            "--provenance-profile", protocol.REGISTRY_PROFILE,
            "--source-commit", args.expected_processing_commit,
            "--base-revision", protocol.STUDENT_REVISION,
        ]
        for arm in protocol.TRAINED_ARMS:
            command += [
                "--arm",
                f"{arm}={args.results_root / 'training' / arm / 'formal'}",
            ]
        command += ["--output", str(output)]
        process, handle = spawn_logged(
            args,
            command,
            log_path=args.results_root / "logs/build-registry.log",
        )
        wait_processes([(process, handle, "registry")])
    if not registry_complete(args):
        raise DiagnosticControllerError("registry barrier failed")
    complete_stage(
        args,
        "05_registry",
        bindings={"checkpoint_registry.json": sha256_file(output)},
    )


def service_specifications(args: argparse.Namespace) -> list[dict[str, Any]]:
    aliases = {
        arm: protocol.MODEL_IDS[arm].removeprefix("openai/")
        for arm in protocol.EVAL_ARMS
    }
    return [
        {
            "role": "user_and_strict_judge",
            "gpu": 0,
            "port": 8001,
            "api_base": protocol.RUNTIME_ENDPOINTS[0],
            "api_key": USER_API_KEY,
            "model": protocol.USER_JUDGE_MODEL,
            "revision": protocol.USER_JUDGE_REVISION,
            "dtype": "float16",
            "quantization": "awq",
            "expected_models": [
                protocol.USER_JUDGE_MODEL_ID.removeprefix("openai/")
            ],
            "adapters": False,
        },
        *[
            {
                "role": f"agent_shard_{gpu - 1}",
                "gpu": gpu,
                "port": 8100 + gpu,
                "api_base": protocol.RUNTIME_ENDPOINTS[gpu],
                "api_key": AGENT_API_KEY,
                "model": protocol.STUDENT_MODEL,
                "revision": protocol.STUDENT_REVISION,
                "dtype": "bfloat16",
                "quantization": None,
                "expected_models": sorted(aliases.values()),
                "adapters": True,
            }
            for gpu in (1, 2, 3)
        ],
    ]


def service_command(
    args: argparse.Namespace, specification: dict[str, Any]
) -> list[str]:
    command = [
        *base.vllm_base_command(
            args,
            specification["model"],
            specification["revision"],
            dtype=specification["dtype"],
        ),
        "--tokenizer-revision", specification["revision"],
        "--served-model-name", specification["expected_models"][0],
        "--port", str(specification["port"]),
        "--api-key", specification["api_key"],
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1",
    ]
    if specification["quantization"]:
        command += ["--quantization", specification["quantization"]]
    if specification["adapters"]:
        command += [
            "--enable-lora",
            "--max-lora-rank", "16",
            "--max-loras", "1",
            "--max-cpu-loras", "4",
            "--lora-modules",
            *[
                (
                    f"{protocol.MODEL_IDS[arm].removeprefix('openai/')}="
                    f"{args.results_root / 'training' / arm / 'formal' / 'checkpoint_final'}"
                )
                for arm in protocol.TRAINED_ARMS
            ],
        ]
    return command


def exact_models(
    api_base: str, api_key: str
) -> tuple[list[str], dict[str, Any]]:
    payload = base.http_json(f"{api_base}/models", api_key)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise DiagnosticControllerError(f"{api_base}/models lacks data list")
    values = [
        row.get("id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    if len(values) != len(set(values)):
        raise DiagnosticControllerError(f"{api_base}: duplicate model aliases")
    return sorted(values), payload


def alias_smoke(
    *, api_base: str, api_key: str, alias: str
) -> dict[str, Any]:
    payload = base.http_post_json(
        f"{api_base}/chat/completions",
        api_key,
        {
            "model": alias,
            "messages": [
                {"role": "user", "content": "Reply with one short token."}
            ],
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    if (
        payload.get("model") != alias
        or not isinstance(choice, dict)
        or choice.get("finish_reason") not in {"stop", "length"}
    ):
        raise DiagnosticControllerError(
            f"one-token alias smoke failed: {api_base} × {alias}"
        )
    return {
        "status": "PASS",
        "api_base": api_base,
        "model_alias": alias,
        "max_tokens": 1,
        "response_model": payload["model"],
        "finish_reason": choice["finish_reason"],
        "response": payload,
        "response_sha256": protocol.canonical_sha256(payload),
    }


def write_service_pid_receipt(
    path: Path,
    *,
    specification: dict[str, Any],
    command: list[str],
    process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    registered = registered_process_identity(process)
    identity = linux_process_identity(process.pid)
    if (
        not same_process_identity(registered, identity)
        or identity["cmdline_sha256"]
        == hashlib.sha256(b"").hexdigest()
    ):
        raise DiagnosticControllerError(
            "service process identity changed or lacks a live command line"
        )
    payload = canonical_receipt(
        {
            "protocol": protocol.SERVICE_PID_RECEIPT_PROTOCOL,
            "service_role": specification["role"],
            "gpu": specification["gpu"],
            "api_base": specification["api_base"],
            "command_sha256": protocol.canonical_sha256(command),
            "process_identity": identity,
        }
    )
    write_immutable(path, payload)
    return payload


def start_services(
    args: argparse.Namespace,
) -> tuple[list[tuple[subprocess.Popen[bytes], Any, Path]], dict[str, str]]:
    services: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    evidence_rows: list[dict[str, Any]] = []
    try:
        require_no_cuda_compute_processes(
            args, barrier="runtime-service-start"
        )
        for specification in service_specifications(args):
            command = service_command(args, specification)
            process, handle = spawn_logged(
                args,
                command,
                gpu=specification["gpu"],
                log_path=args.results_root
                / "logs"
                / f"eval-server-gpu{specification['gpu']}.log",
            )
            pid_path = (
                args.results_root
                / "pids"
                / f"eval-server-gpu{specification['gpu']}.json"
            )
            services.append((process, handle, pid_path))
            base.wait_for_service(
                port=specification["port"],
                api_key=specification["api_key"],
                expected_models=set(specification["expected_models"]),
                process=process,
                timeout_seconds=args.health_timeout,
            )
            # Capture the stable post-health process command line, not the
            # short-lived launcher state immediately after Popen.
            pid_receipt = write_service_pid_receipt(
                pid_path,
                specification=specification,
                command=command,
                process=process,
            )
            observed, models_response = exact_models(
                specification["api_base"], specification["api_key"]
            )
            if observed != specification["expected_models"]:
                raise DiagnosticControllerError(
                    f"{specification['api_base']}: /models must be exact; "
                    f"expected={specification['expected_models']}, observed={observed}"
                )
            evidence_rows.append(
                {
                    key: specification[key]
                    for key in (
                        "role",
                        "gpu",
                        "api_base",
                        "model",
                        "revision",
                        "dtype",
                        "quantization",
                    )
                }
                | {
                    "tokenizer_revision": specification["revision"],
                    "max_model_len": 32768,
                    "max_num_seqs": 1,
                    "gpu_memory_utilization": 0.90,
                    "expected_models": list(specification["expected_models"]),
                    "observed_models": observed,
                    "models_exact": True,
                    "models_response": models_response,
                    "models_response_sha256": (
                        protocol.canonical_sha256(models_response)
                    ),
                    "command": command,
                    "command_sha256": protocol.canonical_sha256(command),
                    "process_identity": pid_receipt["process_identity"],
                    "pid_receipt_path": str(pid_path.resolve()),
                    "pid_receipt_sha256": sha256_file(pid_path),
                }
            )
        smokes = [
            alias_smoke(
                api_base=specification["api_base"],
                api_key=specification["api_key"],
                alias=alias,
            )
            for specification in service_specifications(args)
            for alias in specification["expected_models"]
        ]
        if len(smokes) != protocol.RUNTIME_ALIAS_SMOKE_COUNT:
            raise DiagnosticControllerError("exact ten alias smokes were not run")
        registry_path = args.results_root / "checkpoint_registry.json"
        registry = evaluator.load_checkpoint_registry(
            registry_path, expected_profile=protocol.REGISTRY_PROFILE
        )
        source_admission = args.runtime_root / "source_admission_receipt.json"
        payload = canonical_receipt(
            {
                "protocol": protocol.RUNTIME_SERVICE_EVIDENCE_PROTOCOL,
                "status": "PASS",
                "created_at_utc": utc_now(),
                "processing_source_commit": args.expected_processing_commit,
                "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
                "source_admission_receipt_sha256": sha256_file(
                    source_admission
                ),
                "checkpoint_registry_path": str(registry_path.resolve()),
                "checkpoint_registry_sha256": sha256_file(registry_path),
                "registry_model_ids": {
                    arm: registry["entries"][arm]["model_id"]
                    for arm in sorted(protocol.EVAL_ARMS)
                },
                "services": evidence_rows,
                "alias_smokes": smokes,
                "alias_smoke_count": len(smokes),
                "all_model_sets_exact": True,
                "official_test_used": False,
                "official_test_sealed": True,
            }
        )
        evidence_path = (
            args.results_root / protocol.RUNTIME_SERVICE_EVIDENCE_NAME
        )
        write_immutable(evidence_path, payload)
        _, binding = evaluator.load_low_support_runtime_service_evidence(
            evidence_path,
            checkpoint_registry_path=registry_path,
            checkpoint_registry=registry,
            expected_source_commit=args.expected_processing_commit,
        )
        complete_stage(
            args,
            "06_runtime_services",
            bindings={
                protocol.RUNTIME_SERVICE_EVIDENCE_NAME: sha256_file(
                    evidence_path
                )
            },
            details={"exact_services": 4, "exact_alias_smokes": 10},
        )
        return services, binding
    except Exception:
        stop_services(services)
        raise


def stop_services(
    services: Iterable[tuple[subprocess.Popen[bytes], Any, Path]]
) -> None:
    failures: list[str] = []
    for process, handle, _pid_path in list(services):
        try:
            stop_process(process)
        except Exception as error:
            failures.append(f"PID {process.pid}: {error}")
        finally:
            unregister_process(process)
            handle.close()
    if failures:
        raise DiagnosticControllerError(
            f"service cleanup failed closed: {failures}"
        )


def revalidate_live_runtime_services(
    args: argparse.Namespace,
    expected_binding: dict[str, str],
) -> None:
    registry_path = args.results_root / "checkpoint_registry.json"
    registry = evaluator.load_checkpoint_registry(
        registry_path, expected_profile=protocol.REGISTRY_PROFILE
    )
    _, binding = evaluator.load_low_support_runtime_service_evidence(
        args.results_root / protocol.RUNTIME_SERVICE_EVIDENCE_NAME,
        checkpoint_registry_path=registry_path,
        checkpoint_registry=registry,
        expected_source_commit=args.expected_processing_commit,
        require_live_services=True,
    )
    if binding != expected_binding:
        raise DiagnosticControllerError(
            "runtime service evidence binding changed during evaluation"
        )
    for specification in service_specifications(args):
        observed, _payload = exact_models(
            specification["api_base"], specification["api_key"]
        )
        if observed != specification["expected_models"]:
            raise DiagnosticControllerError(
                "live runtime service alias set changed during evaluation"
            )


def evaluation_command(
    args: argparse.Namespace, arm: str, shard: int
) -> list[str]:
    files = source_protocol_files(args)
    command = [
        str(args.serve_python),
        "scripts/run_v5_sft_causal_eval.py",
        "--tau2-root", str(args.tau2_root),
        "--split-manifest", str(SPLIT),
        "--manifest", str(files["validation"]),
        "--dynamic-audit", str(files["validation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
        "--runtime-service-evidence",
        str(args.results_root / protocol.RUNTIME_SERVICE_EVIDENCE_NAME),
        "--provenance-profile", protocol.REGISTRY_PROFILE,
    ]
    for trained in protocol.TRAINED_ARMS:
        command += [
            "--adapter-dir",
            (
                f"{trained}={args.results_root / 'training' / trained / 'formal' / 'checkpoint_final'}"
            ),
        ]
    command += [
        "--output-dir", str(args.results_root / "evaluation" / arm),
        "--arm", arm,
        "--agent-model", protocol.MODEL_IDS[arm],
        "--agent-api-base", protocol.RUNTIME_ENDPOINTS[shard + 1],
        "--agent-api-key", AGENT_API_KEY,
        "--user-model", protocol.USER_JUDGE_MODEL_ID,
        "--user-api-base", protocol.RUNTIME_ENDPOINTS[0],
        "--user-api-key", USER_API_KEY,
        "--judge-model", protocol.USER_JUDGE_MODEL_ID,
        "--judge-api-base", protocol.RUNTIME_ENDPOINTS[0],
        "--judge-api-key", USER_API_KEY,
        "--condition", "both",
        "--shard-index", str(shard),
        "--num-shards", str(protocol.EVALUATION_SHARDS),
        "--max-steps", "60",
        "--timeout", "900",
        "--max-tokens", "512",
        "--seed", str(protocol.BASE_SEED),
        "--num-trials", "1",
    ]
    return command


def contract_path(args: argparse.Namespace, arm: str, shard: int) -> Path:
    return (
        args.results_root
        / "evaluation"
        / arm
        / f"run_contract.shard-{shard:03d}-of-003.json"
    )


def contract_complete(
    args: argparse.Namespace,
    arm: str,
    shard: int,
    runtime_binding: dict[str, str],
) -> bool:
    try:
        path = contract_path(args, arm, shard)
        value = read_json(path)
        evaluator.validate_contract_core(value)
        declared = value.get("result_sha256")
        return bool(
            value.get("status") == "COMPLETE"
            and value.get("arm") == arm
            and value.get("shard_index") == shard
            and value.get("num_shards") == protocol.EVALUATION_SHARDS
            and value.get("checkpoint_registry_provenance_profile")
            == protocol.REGISTRY_PROFILE
            and value.get("source_commit")
            == args.expected_processing_commit
            and value.get("runtime_service_evidence") == runtime_binding
            and value.get("agent", {}).get("api_base")
            == protocol.RUNTIME_ENDPOINTS[shard + 1]
            and value.get("user", {}).get("api_base")
            == protocol.RUNTIME_ENDPOINTS[0]
            and isinstance(declared, dict)
            and declared
            and all(
                Path(name).name == name
                and (path.parent / name).is_file()
                and sha256_file(path.parent / name) == digest
                for name, digest in declared.items()
            )
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def evaluate(
    args: argparse.Namespace, runtime_binding: dict[str, str]
) -> None:
    # Across each wave the three shards run concurrently; a shard cannot enter
    # the next arm until its current arm has completed, so every shard's arm
    # order is exactly base -> perfect -> repair.
    for arm in protocol.EVAL_ARMS:
        workers = []
        for shard in range(protocol.EVALUATION_SHARDS):
            if contract_complete(args, arm, shard, runtime_binding):
                continue
            path = contract_path(args, arm, shard)
            if path.exists():
                raise DiagnosticControllerError(
                    f"partial evaluation contract must be audited: {path}"
                )
            process, handle = spawn_logged(
                args,
                evaluation_command(args, arm, shard),
                log_path=args.results_root
                / "logs"
                / f"evaluate-shard-{shard}-{arm}.log",
            )
            workers.append((process, handle, f"shard-{shard}-{arm}"))
        wait_processes(workers)
        revalidate_live_runtime_services(args, runtime_binding)
        if not all(
            contract_complete(args, arm, shard, runtime_binding)
            for shard in range(protocol.EVALUATION_SHARDS)
        ):
            raise DiagnosticControllerError(
                f"evaluation barrier failed for {arm}"
            )
    bindings = {
        f"{arm}/shard-{shard}": sha256_file(contract_path(args, arm, shard))
        for arm in protocol.EVAL_ARMS
        for shard in range(protocol.EVALUATION_SHARDS)
    }
    complete_stage(
        args,
        "07_evaluation",
        bindings=bindings,
        details={
            "shards_concurrent": 3,
            "arms_serial_per_shard": list(protocol.EVAL_ARMS),
            "exact_raw_cases_expected": protocol.EVALUATION_ROLLOUTS,
        },
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.results_root / "diagnostic_summary.json"
    command = [
        str(args.train_python),
        "scripts/summarize_v5_3_low_support_diagnostic.py",
        "--split-manifest", str(SPLIT),
        "--evaluation-manifest",
        str(source_protocol_files(args)["validation"]),
        "--dynamic-audit",
        str(source_protocol_files(args)["validation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
        "--results-root", str(args.results_root),
        "--output", str(output),
    ]
    process, handle = spawn_logged(
        args,
        command,
        log_path=args.results_root / "logs/summarize.log",
    )
    wait_processes([(process, handle, "summary")])
    summary = read_json(output)
    if (
        summary.get("status") != "DIAGNOSTIC_COMPLETE"
        or summary.get("raw_case_count") != protocol.EVALUATION_ROLLOUTS
        or summary.get("evaluated_arms") != list(protocol.EVAL_ARMS)
        or summary.get("formal_v5_3_result") is not False
        or summary.get("paper_level_confirmation") is not False
        or summary.get("official_test") != {"status": "SEALED", "used": False}
        or (summary.get("provenance") or {}).get(
            "runtime_service_evidence"
        )
        is None
    ):
        raise DiagnosticControllerError("raw-bound diagnostic summary failed")
    complete_stage(
        args,
        "08_summary",
        bindings={"diagnostic_summary.json": sha256_file(output)},
        details={"exact_raw_cases": protocol.EVALUATION_ROLLOUTS},
    )
    return summary


def status(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "protocol": protocol.CONTROLLER_PROTOCOL,
        "processing_source_commit": args.expected_processing_commit,
        "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
        "stages": {
            path.stem: read_json(path).get("status")
            for path in sorted(
                (args.runtime_root / "stages").glob("*.json")
            )
        }
        if (args.runtime_root / "stages").is_dir()
        else {},
        "terminal": (
            read_json(terminal_path(args))
            if terminal_path(args).is_file()
            else None
        ),
        "active_child_pids": sorted(_ACTIVE),
        "official_test_used": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def stop_owned_services(args: argparse.Namespace) -> None:
    # Only identity-bound receipts in the isolated namespace may be signalled.
    specifications = service_specifications(args)
    expected_paths = {
        (
            args.results_root
            / "pids"
            / f"eval-server-gpu{row['gpu']}.json"
        ).resolve(): row
        for row in specifications
    }
    observed_paths = {
        path.resolve()
        for path in (args.results_root / "pids").glob(
            "eval-server-gpu*.json"
        )
    }
    unexpected = observed_paths - set(expected_paths)
    if unexpected:
        raise DiagnosticControllerError(
            f"unexpected service PID receipts: {sorted(map(str, unexpected))}"
        )
    runtime_evidence_path = (
        args.results_root / protocol.RUNTIME_SERVICE_EVIDENCE_NAME
    )
    runtime_evidence = (
        read_json(runtime_evidence_path)
        if runtime_evidence_path.is_file()
        else None
    )
    for path in sorted(observed_paths):
        specification = expected_paths[path]
        try:
            receipt = read_json(path)
            canonical = receipt.get("canonical_sha256")
            core = {
                key: value
                for key, value in receipt.items()
                if key != "canonical_sha256"
            }
            identity = validate_service_process_identity(
                receipt.get("process_identity")
            )
            if (
                set(receipt)
                != {
                    "protocol",
                    "service_role",
                    "gpu",
                    "api_base",
                    "command_sha256",
                    "process_identity",
                    "canonical_sha256",
                }
                or receipt.get("protocol")
                != protocol.SERVICE_PID_RECEIPT_PROTOCOL
                or canonical != protocol.canonical_sha256(core)
                or receipt.get("service_role") != specification["role"]
                or receipt.get("gpu") != specification["gpu"]
                or receipt.get("api_base") != specification["api_base"]
                or not valid_sha256(receipt.get("command_sha256"))
            ):
                raise DiagnosticControllerError(
                    f"invalid service PID receipt: {path}"
                )
            if runtime_evidence is not None:
                bound_rows = [
                    row
                    for row in runtime_evidence.get("services", [])
                    if isinstance(row, dict)
                    and row.get("pid_receipt_path") == str(path)
                ]
                if (
                    len(bound_rows) != 1
                    or bound_rows[0].get("pid_receipt_sha256")
                    != sha256_file(path)
                    or bound_rows[0].get("process_identity") != identity
                    or bound_rows[0].get("command_sha256")
                    != receipt.get("command_sha256")
                ):
                    raise DiagnosticControllerError(
                        "runtime evidence does not bind service PID receipt"
                    )
            pid = identity.get("pid")
            pgid = identity.get("process_group_id")
            if not process_group_exists(pgid):
                continue
            if (
                (Path("/proc") / str(pid)).exists()
                and not same_process_identity(
                    identity, linux_process_identity(pid)
                )
            ):
                raise DiagnosticControllerError(
                    "refusing stop-services because PID identity was reused"
                )
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + TERM_GRACE_SECONDS
            while process_group_exists(pgid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if process_group_exists(pgid):
                if (
                    (Path("/proc") / str(pid)).exists()
                    and not same_process_identity(
                        identity, linux_process_identity(pid)
                    )
                ):
                    raise DiagnosticControllerError(
                        "refusing SIGKILL because PID identity changed"
                    )
                os.killpg(pgid, signal.SIGKILL)
                kill_deadline = time.monotonic() + 10
                while (
                    process_group_exists(pgid)
                    and time.monotonic() < kill_deadline
                ):
                    time.sleep(0.1)
                if process_group_exists(pgid):
                    raise DiagnosticControllerError(
                        f"process group {pgid} survived SIGKILL"
                    )
        except (OSError, ProcessLookupError) as error:
            raise DiagnosticControllerError(
                f"failed to stop identity-bound service: {path}"
            ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("all", "status", "stop-services"),
    )
    parser.add_argument("--expected-processing-commit")
    parser.add_argument(
        "--tau2-root", type=Path, default=ROOT / "data/raw/tau2-bench"
    )
    parser.add_argument(
        "--serve-venv", type=Path, default=Path("/workspace/venvs/v5-2-serve")
    )
    parser.add_argument(
        "--train-venv", type=Path, default=Path("/workspace/venvs/v5-2-train")
    )
    parser.add_argument(
        "--hf-home", type=Path, default=Path("/workspace/cache/huggingface")
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--source-protocol-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_12h_screen_protocol",
    )
    parser.add_argument(
        "--source-raw-root",
        type=Path,
        default=ROOT / "data/raw/v5_3_12h_screen_generation",
    )
    parser.add_argument(
        "--source-processed-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_12h_screen",
    )
    parser.add_argument(
        "--source-results-root",
        type=Path,
        default=ROOT / "results/v5_3_12h_screen",
    )
    parser.add_argument(
        "--source-runtime-root",
        type=Path,
        default=ROOT / protocol.SOURCE_RUNTIME_ROOT,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=protocol.artifact_root(ROOT, "processed_root"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=protocol.artifact_root(ROOT, "results_root"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=protocol.artifact_root(ROOT, "runtime_root"),
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    args = parser.parse_args()
    for field in (
        "tau2_root",
        "serve_venv",
        "train_venv",
        "hf_home",
        "workspace_root",
        "source_protocol_root",
        "source_raw_root",
        "source_processed_root",
        "source_results_root",
        "source_runtime_root",
        "processed_root",
        "results_root",
        "runtime_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.serve_python = args.serve_venv / "bin/python"
    args.train_python = args.train_venv / "bin/python"
    args.vllm = args.serve_venv / "bin/vllm"
    return args


def main() -> None:
    global _GLOBAL_ARGS
    args = parse_args()
    validate_roots(args)
    if args.stage == "stop-services":
        stop_owned_services(args)
        return
    if not args.expected_processing_commit:
        raise SystemExit("--expected-processing-commit is required")
    verify_processing_checkout(args.expected_processing_commit)
    if args.stage == "status":
        status(args)
        return
    _GLOBAL_ARGS = args
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)
    try:
        acquire_whole_run_locks(args)
        validate_source_admission(args)
        preflight(args)
        prepare(args)
        train_mode(args, "smoke")
        train_mode(args, "formal")
        build_registry(args)
        services: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
        try:
            services, runtime_binding = start_services(args)
            evaluate(args, runtime_binding)
        finally:
            stop_services(services)
        summarize(args)
        write_terminal(
            args,
            status="DIAGNOSTIC_COMPLETE_DESCRIPTIVE_ONLY",
            reason="EXACT_126_RAW_CASES_RECOMPUTED_AND_BOUND",
            no_claim=False,
        )
        status(args)
    except Exception as error:
        write_terminal(
            args,
            status="INCOMPLETE_NO_CLAIM",
            reason=f"{type(error).__name__}: {error}",
            no_claim=True,
        )
        print(f"low-support diagnostic stopped: {error}", file=sys.stderr)
        raise SystemExit(20) from error
    finally:
        cleanup_processes()
        release_whole_run_locks()


if __name__ == "__main__":
    main()
