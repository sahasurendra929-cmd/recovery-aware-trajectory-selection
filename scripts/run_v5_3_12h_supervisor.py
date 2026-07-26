#!/usr/bin/env python3
"""Detached fail-closed supervisor for the V5.3 12-hour screen controller.

The supervisor owns the authoritative T0, launches the scientific controller
in a dedicated process group, and launches a second detached watchdog session.
The watchdog holds the singleton lock and remains alive if the launcher,
controller, or SSH session disappears.  It may signal only the exact
controller process group recorded for the current Linux boot.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

try:
    import v5_3_12h_protocol as protocol
    import v5_3_source_snapshot_contract as source_snapshot
except ModuleNotFoundError:
    from scripts import v5_3_12h_protocol as protocol
    from scripts import v5_3_source_snapshot_contract as source_snapshot


SUPERVISOR_PROTOCOL = f"{protocol.PROTOCOL}:supervisor_v1"
LEDGER_PROTOCOL = f"{SUPERVISOR_PROTOCOL}:ledger_v1"
RECEIPT_PROTOCOL = f"{SUPERVISOR_PROTOCOL}:receipt_v1"
LOCK_NAME = "supervisor.lock"
LEDGER_NAME = "supervisor_ledger.json"
RECEIPT_NAME = "supervisor_receipt.json"
STOP_REQUEST_NAME = "supervisor_stop_request.json"
WATCHDOG_LOG_NAME = "supervisor_watchdog.log"
POLL_SECONDS = 0.25
DEFAULT_TERM_GRACE_SECONDS = 30.0
FINAL_CONTROLLER_STATUSES = {
    "COMPLETE_DIRECTIONAL_POSITIVE_SCREEN",
    "COMPLETE_EXPLORATORY_INCONCLUSIVE",
}
FINAL_NO_CLAIM_CONTROLLER_STATUSES = {
    "DATA_GATE_FAIL_NO_TRAIN",
    "MATCHING_INCONCLUSIVE_NO_TRAIN",
}
_DETACHED_HANDLES: list[subprocess.Popen[bytes]] = []


def _release_detached_handles_at_exit() -> None:
    """Avoid Popen destructor warnings without waiting for detached sessions."""

    for process in _DETACHED_HANDLES:
        process.poll()
        if process.returncode is None:
            # The child is intentionally reparented when this short launcher
            # exits; marking only this local handle does not signal the child.
            process.returncode = 0


atexit.register(_release_detached_handles_at_exit)


class SupervisorError(RuntimeError):
    """Supervisor state is malformed, stale, conflicting, or unsafe."""


class UnsafeProcessGroup(SupervisorError):
    """A requested signal could reach the watchdog or invoking shell."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupervisorError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise SupervisorError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def current_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    # Non-Linux unit-test/development fallback. It is stable for this boot
    # only within the same host session and is never used on RunPod Linux.
    return f"nonlinux:{os.uname().nodename}:{int(time.monotonic() // 3600)}"


def deadline_path(runtime_root: Path) -> Path:
    return runtime_root / "deadline_receipt.json"


def ledger_path(runtime_root: Path) -> Path:
    return runtime_root / LEDGER_NAME


def receipt_path(runtime_root: Path) -> Path:
    return runtime_root / RECEIPT_NAME


def lock_path(runtime_root: Path) -> Path:
    return runtime_root / LOCK_NAME


def stop_request_path(runtime_root: Path) -> Path:
    return runtime_root / STOP_REQUEST_NAME


def _ledger_core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != "canonical_sha256"
    }


def validate_deadline_and_ledger(
    runtime_root: Path,
    *,
    require_current_boot: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = read_json(deadline_path(runtime_root))
    protocol.validate_deadline_receipt(deadline)
    ledger = read_json(ledger_path(runtime_root))
    if (
        ledger.get("protocol") != LEDGER_PROTOCOL
        or ledger.get("canonical_sha256")
        != canonical_sha256(_ledger_core(ledger))
        or ledger.get("deadline_receipt_path")
        != str(deadline_path(runtime_root).resolve())
        or ledger.get("deadline_receipt_sha256")
        != sha256_file(deadline_path(runtime_root))
        or ledger.get("started_at_utc") != deadline.get("started_at_utc")
        or ledger.get("deadline_utc")
        != deadline.get("core_result_target_deadline_utc")
        or ledger.get("resume_does_not_reset_t0") is not True
        or not isinstance(ledger.get("boot_id"), str)
        or not ledger["boot_id"]
    ):
        raise SupervisorError("supervisor deadline ledger drift")
    if require_current_boot and ledger["boot_id"] != current_boot_id():
        raise SupervisorError("supervisor boot_id mismatch; refusing stale PID action")
    return deadline, ledger


def load_or_create_authoritative_deadline(
    runtime_root: Path,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create T0 once, or validate and return the exact original T0."""

    runtime_root.mkdir(parents=True, exist_ok=True)
    deadline_file = deadline_path(runtime_root)
    ledger_file = ledger_path(runtime_root)
    if deadline_file.exists() or ledger_file.exists():
        if not deadline_file.is_file() or not ledger_file.is_file():
            raise SupervisorError("partial deadline/ledger state must be audited")
        return validate_deadline_and_ledger(
            runtime_root,
            require_current_boot=False,
        )
    started = now or utc_now()
    if started.tzinfo is None:
        raise SupervisorError("authoritative T0 must be timezone-aware")
    deadline = protocol.make_deadline_receipt(started)
    write_exclusive(deadline_file, deadline)
    ledger = {
        "protocol": LEDGER_PROTOCOL,
        "boot_id": current_boot_id(),
        "started_at_utc": deadline["started_at_utc"],
        "deadline_utc": deadline["core_result_target_deadline_utc"],
        "deadline_receipt_path": str(deadline_file.resolve()),
        "deadline_receipt_sha256": sha256_file(deadline_file),
        "resume_does_not_reset_t0": True,
        "launch_generation": 0,
        "controller": None,
        "watchdog": None,
        "launcher": None,
        "last_transition": "T0_CREATED",
    }
    ledger["canonical_sha256"] = canonical_sha256(ledger)
    try:
        write_exclusive(ledger_file, ledger)
    except Exception:
        # A deadline without its ledger is deliberately not auto-repaired:
        # subsequent starts fail closed instead of manufacturing another T0.
        raise
    return deadline, ledger


def _pid_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _group_alive(pgid: int) -> bool:
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def safe_registered_group(
    ledger: dict[str, Any],
    *,
    actor_pid: int | None = None,
    actor_pgid: int | None = None,
) -> int:
    controller = ledger.get("controller")
    if not isinstance(controller, dict):
        raise UnsafeProcessGroup("ledger has no registered controller")
    target = controller.get("pgid")
    pid = controller.get("pid")
    actor_pid = os.getpid() if actor_pid is None else actor_pid
    actor_pgid = os.getpgrp() if actor_pgid is None else actor_pgid
    forbidden_groups = {actor_pgid}
    for field in ("watchdog", "launcher"):
        value = ledger.get(field)
        if isinstance(value, dict) and isinstance(value.get("pgid"), int):
            forbidden_groups.add(value["pgid"])
    try:
        forbidden_groups.add(os.getpgid(os.getppid()))
    except ProcessLookupError:
        pass
    if (
        ledger.get("boot_id") != current_boot_id()
        or isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 1
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or target in forbidden_groups
        or pid == actor_pid
    ):
        raise UnsafeProcessGroup(
            f"refusing unsafe/stale registered process group {target}"
        )
    if _pid_alive(pid):
        try:
            observed = os.getpgid(pid)
        except ProcessLookupError:
            observed = None
        if observed is not None and observed != target:
            raise UnsafeProcessGroup(
                f"controller PID was reused/moved: expected {target}, got {observed}"
            )
    return target


def terminate_registered_group(
    ledger: dict[str, Any],
    *,
    grace_seconds: float,
    actor_pid: int | None = None,
    actor_pgid: int | None = None,
) -> dict[str, Any]:
    target = safe_registered_group(
        ledger,
        actor_pid=actor_pid,
        actor_pgid=actor_pgid,
    )
    evidence = {
        "target_pgid": target,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "group_alive_before": _group_alive(target),
    }
    if not evidence["group_alive_before"]:
        evidence["group_alive_after"] = False
        return evidence
    os.killpg(target, signal.SIGTERM)
    evidence["sigterm_sent"] = True
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while _group_alive(target) and time.monotonic() < deadline:
        time.sleep(min(POLL_SECONDS, max(0.01, deadline - time.monotonic())))
    if _group_alive(target):
        os.killpg(target, signal.SIGKILL)
        evidence["sigkill_sent"] = True
        kill_deadline = time.monotonic() + 5.0
        while _group_alive(target) and time.monotonic() < kill_deadline:
            time.sleep(POLL_SECONDS)
    evidence["group_alive_after"] = _group_alive(target)
    return evidence


def write_action_receipt(
    runtime_root: Path,
    *,
    status: str,
    reason: str,
    action_required: bool,
    ledger: dict[str, Any],
    termination: dict[str, Any] | None,
) -> dict[str, Any]:
    actions = runtime_root / "supervisor_actions"
    actions.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": RECEIPT_PROTOCOL,
        "status": status,
        "reason": reason,
        "action_required": action_required,
        "timestamp_utc": utc_now().isoformat(),
        "boot_id": current_boot_id(),
        "deadline_receipt_path": str(deadline_path(runtime_root).resolve()),
        "deadline_receipt_sha256": (
            sha256_file(deadline_path(runtime_root))
            if deadline_path(runtime_root).is_file()
            else None
        ),
        "authoritative_t0_utc": ledger.get("started_at_utc"),
        "deadline_utc": ledger.get("deadline_utc"),
        "controller": ledger.get("controller"),
        "watchdog": ledger.get("watchdog"),
        "termination": termination,
        "official_test_used": False,
        "no_scientific_claim": status
        not in {
            "CONTROLLER_COMPLETED",
            "CORE_COMPLETE_EXTENSION_INCOMPLETE",
        },
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    historical = actions / (
        f"{time.time_ns()}-{status.lower().replace('_', '-')}.json"
    )
    write_exclusive(historical, payload)
    atomic_write(receipt_path(runtime_root), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def write_emergency_action_receipt(
    runtime_root: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """Emit evidence even when a corrupt ledger cannot be trusted."""

    payload = {
        "protocol": RECEIPT_PROTOCOL,
        "status": "ACTION_REQUIRED",
        "reason": reason,
        "action_required": True,
        "timestamp_utc": utc_now().isoformat(),
        "boot_id": current_boot_id(),
        "deadline_receipt_path": str(deadline_path(runtime_root).resolve()),
        "deadline_receipt_sha256": (
            sha256_file(deadline_path(runtime_root))
            if deadline_path(runtime_root).is_file()
            else None
        ),
        "authoritative_t0_utc": None,
        "deadline_utc": None,
        "controller": None,
        "watchdog": {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
        },
        "termination": None,
        "official_test_used": False,
        "no_scientific_claim": True,
        "unsafe_pid_or_process_group_signal_attempted": False,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    actions = runtime_root / "supervisor_actions"
    actions.mkdir(parents=True, exist_ok=True)
    write_exclusive(
        actions / f"{time.time_ns()}-emergency-action-required.json",
        payload,
    )
    atomic_write(receipt_path(runtime_root), payload)
    return payload


def _deadline_seconds(deadline: dict[str, Any]) -> float:
    value = datetime.fromisoformat(deadline["core_result_target_deadline_utc"])
    return (value - utc_now()).total_seconds()


def _core_preserved_by_terminal(terminal_receipt: Path | None) -> bool:
    if terminal_receipt is None or not terminal_receipt.is_file():
        return False
    try:
        terminal = read_json(terminal_receipt)
    except SupervisorError:
        return False
    return bool(
        terminal.get("protocol") == f"{protocol.PROTOCOL}:terminal_v1"
        and terminal.get("status") == "CORE_COMPLETE_EXTENSION_INCOMPLETE"
        and terminal.get("no_scientific_claim") is False
        and terminal.get("official_test_used") is False
        and terminal.get("official_test_sealed") is True
    )


def _controller_completed_by_terminal(
    terminal_receipt: Path | None,
) -> bool:
    if terminal_receipt is None or not terminal_receipt.is_file():
        return False
    try:
        terminal = read_json(terminal_receipt)
    except SupervisorError:
        return False
    return bool(
        terminal.get("protocol") == f"{protocol.PROTOCOL}:terminal_v1"
        and terminal.get("status") in FINAL_CONTROLLER_STATUSES
        and terminal.get("no_scientific_claim") is False
        and terminal.get("official_test_used") is False
        and terminal.get("official_test_sealed") is True
    )


def _controller_completed_no_claim_by_terminal(
    terminal_receipt: Path | None,
) -> bool:
    if terminal_receipt is None or not terminal_receipt.is_file():
        return False
    try:
        terminal = read_json(terminal_receipt)
    except SupervisorError:
        return False
    try:
        repo_root = terminal_receipt.resolve().parents[2]
    except IndexError:
        return False
    processed_root = repo_root / "data/processed/v5_3_12h_screen"
    audit_path = processed_root / "audit.json"
    hashes_path = processed_root / "hashes.json"
    receipt_path = repo_root / source_snapshot.RECEIPT_NAME
    try:
        audit = read_json(audit_path)
        hashes = read_json(hashes_path)
        snapshot_receipt = read_json(receipt_path)
    except SupervisorError:
        return False
    processing_commit = terminal.get("processing_source_commit")
    return bool(
        terminal.get("protocol") == f"{protocol.PROTOCOL}:terminal_v1"
        and terminal.get("status") in FINAL_NO_CLAIM_CONTROLLER_STATUSES
        and terminal.get("reason") in {
            "DO_NOT_TRAIN",
            "FIXED_JOINT_MATCHER_FOUND_NO_REGISTERED_SCHEDULE",
        }
        and terminal.get("no_scientific_claim") is True
        and terminal.get("official_test_used") is False
        and terminal.get("official_test_sealed") is True
        and isinstance(processing_commit, str)
        and len(processing_commit) == 40
        and processing_commit.lower() == processing_commit
        and all(
            character in "0123456789abcdef"
            for character in processing_commit
        )
        and processing_commit
        != snapshot_receipt.get("source_generation_commit")
        and terminal.get("source_generation_commit")
        == snapshot_receipt.get("source_generation_commit")
        and terminal.get("source_audit_sha256") == sha256_file(audit_path)
        and terminal.get("source_hashes_sha256") == sha256_file(hashes_path)
        and terminal.get("source_snapshot_receipt_sha256")
        == sha256_file(receipt_path)
        and terminal.get("source_snapshot_file_ledger_sha256")
        == source_snapshot.canonical_sha256(
            snapshot_receipt.get("file_sha256")
        )
        and audit.get("processing_source_commit") == processing_commit
        and audit.get("source_generation_commit")
        == terminal.get("source_generation_commit")
        and audit.get("source_snapshot", {}).get("receipt_sha256")
        == sha256_file(receipt_path)
        and set(hashes) == {"audit.json", "attempt_audit.jsonl"}
        and hashes.get("audit.json") == sha256_file(audit_path)
        and (processed_root / "attempt_audit.jsonl").is_file()
        and hashes.get("attempt_audit.jsonl")
        == sha256_file(processed_root / "attempt_audit.jsonl")
    )


def validated_core_snapshot(
    runtime_root: Path,
    terminal_receipt: Path | None,
) -> bool:
    """Validate the immutable core without relying on a live controller."""

    if terminal_receipt is None:
        return False
    results_root = terminal_receipt.parent
    summary_path = results_root / "core_screen_summary.json"
    core_path = runtime_root / "core_complete_receipt.json"
    deadline_file = deadline_path(runtime_root)
    registry_path = results_root / "checkpoint_registry.json"
    if any(
        not path.is_file()
        for path in (summary_path, core_path, deadline_file, registry_path)
    ):
        return False
    try:
        summary = read_json(summary_path)
        core = read_json(core_path)
        deadline = read_json(deadline_file)
        protocol.validate_deadline_receipt(deadline)
        provenance = summary.get("provenance")
        official = summary.get("official_test")
        core_body = {
            key: value
            for key, value in core.items()
            if key != "canonical_sha256"
        }
        completed = datetime.fromisoformat(core["completed_at_utc"])
        hard_deadline = datetime.fromisoformat(
            deadline["core_result_target_deadline_utc"]
        )
        return bool(
            summary.get("protocol")
            == f"{protocol.PROTOCOL}:raw_summary_v1"
            and summary.get("status") == "CORE_COMPLETE"
            and summary.get("core_raw_cases") == protocol.CORE_ROLLOUTS
            and set(summary.get("arms") or {}) == set(
                protocol.CORE_EVAL_ARMS
            )
            and isinstance(official, dict)
            and official.get("status") == "SEALED"
            and official.get("used") is False
            and isinstance(provenance, dict)
            and provenance.get("core_complete_receipt_sha256")
            == sha256_file(core_path)
            and provenance.get("deadline_receipt_sha256")
            == sha256_file(deadline_file)
            and provenance.get("checkpoint_registry_sha256")
            == sha256_file(registry_path)
            and provenance.get("static_protocol_artifact_sha256")
            == core.get("static_protocol_artifact_sha256")
            and core.get("protocol")
            == f"{protocol.PROTOCOL}:core_complete_v1"
            and core.get("exact_raw_case_count") == protocol.CORE_ROLLOUTS
            and core.get("expected_raw_case_count") == protocol.CORE_ROLLOUTS
            and core.get("official_test_used") is False
            and core.get("metric_values_read") is False
            and core.get("deadline_receipt_sha256")
            == sha256_file(deadline_file)
            and core.get("checkpoint_registry_sha256")
            == sha256_file(registry_path)
            and core.get("canonical_sha256")
            == protocol.canonical_sha256(core_body)
            and completed.tzinfo is not None
            and hard_deadline.tzinfo is not None
            and completed <= hard_deadline
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        SupervisorError,
        protocol.ScreenProtocolError,
    ):
        return False


def write_preserved_core_terminal(
    runtime_root: Path,
    terminal_receipt: Path,
    *,
    reason: str,
) -> None:
    """Write the final cutoff status after the controller group is dead."""

    payload = {
        "protocol": f"{protocol.PROTOCOL}:terminal_v1",
        "status": "CORE_COMPLETE_EXTENSION_INCOMPLETE",
        "reason": reason,
        "timestamp_utc": utc_now().isoformat(),
        "deadline_receipt": str(deadline_path(runtime_root).resolve()),
        "deadline_receipt_sha256": sha256_file(
            deadline_path(runtime_root)
        ),
        "no_scientific_claim": False,
        "official_test_used": False,
        "official_test_sealed": True,
        "frozen_v5_3_configs_seeds_data_results_modified": False,
        "shared_code_extension_scope": (
            "explicit v5_3_12h_screen profiles; legacy and V5.3 defaults "
            "must remain regression-verified"
        ),
        "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
        "github_publish_performed_by_controller": False,
        "runpod_management_plane_stop_performed_by_controller": False,
        "outer_agent_actions_required": [
            "publish bounded evidence to the verified result branch",
            "stop the RunPod Pod and verify Stopped status",
        ],
    }
    atomic_write(terminal_receipt, payload)


def watchdog_loop(
    runtime_root: Path,
    *,
    lock_fd: int,
    term_grace_seconds: float,
    terminal_receipt: Path | None,
) -> int:
    # The inherited open-file description retains the launcher's exclusive
    # flock after the launcher exits.
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity_deadline = time.monotonic() + 10.0
    while True:
        deadline, ledger = validate_deadline_and_ledger(
            runtime_root,
            require_current_boot=True,
        )
        watchdog = ledger.get("watchdog")
        if (
            isinstance(watchdog, dict)
            and watchdog.get("pid") == os.getpid()
            and watchdog.get("pgid") == os.getpgrp()
        ):
            break
        if time.monotonic() >= identity_deadline:
            raise SupervisorError("watchdog identity differs from ledger")
        time.sleep(0.05)
    while True:
        if stop_request_path(runtime_root).is_file():
            core_valid = validated_core_snapshot(
                runtime_root, terminal_receipt
            )
            termination = terminate_registered_group(
                ledger,
                grace_seconds=term_grace_seconds,
            )
            if core_valid and terminal_receipt is not None:
                write_preserved_core_terminal(
                    runtime_root,
                    terminal_receipt,
                    reason="VALIDATED_CORE_RETAINED_AFTER_EXPLICIT_LOCAL_STOP",
                )
            core_preserved = core_valid or _core_preserved_by_terminal(
                terminal_receipt
            )
            write_action_receipt(
                runtime_root,
                status=(
                    "CORE_COMPLETE_EXTENSION_INCOMPLETE"
                    if core_preserved
                    else "STOPPED_NO_CLAIM"
                ),
                reason=(
                    "VALIDATED_CORE_RETAINED_AFTER_EXPLICIT_LOCAL_STOP"
                    if core_preserved
                    else "EXPLICIT_LOCAL_STOP_REQUEST"
                ),
                action_required=False,
                ledger=ledger,
                termination=termination,
            )
            return 0
        if _deadline_seconds(deadline) <= 0:
            core_valid = validated_core_snapshot(
                runtime_root, terminal_receipt
            )
            termination = terminate_registered_group(
                ledger,
                grace_seconds=term_grace_seconds,
            )
            if core_valid and terminal_receipt is not None:
                write_preserved_core_terminal(
                    runtime_root,
                    terminal_receipt,
                    reason=(
                        "IMMUTABLE_CORE_RETAINED_AT_12_HOUR_"
                        "EXTENSION_CUTOFF"
                    ),
                )
            core_preserved = core_valid or _core_preserved_by_terminal(
                terminal_receipt
            )
            write_action_receipt(
                runtime_root,
                status=(
                    "CORE_COMPLETE_EXTENSION_INCOMPLETE"
                    if core_preserved
                    else "INCOMPLETE_NO_CLAIM"
                ),
                reason=(
                    "IMMUTABLE_CORE_RETAINED_AT_12_HOUR_EXTENSION_CUTOFF"
                    if core_preserved
                    else "IMMUTABLE_12_HOUR_DEADLINE_EXPIRED"
                ),
                action_required=True,
                ledger=ledger,
                termination=termination,
            )
            return 124
        controller = ledger["controller"]
        controller_alive = _pid_alive(controller["pid"])
        group_alive = _group_alive(controller["pgid"])
        if not controller_alive:
            if group_alive:
                termination = terminate_registered_group(
                    ledger,
                    grace_seconds=term_grace_seconds,
                )
                write_action_receipt(
                    runtime_root,
                    status="ACTION_REQUIRED",
                    reason="CONTROLLER_DIED_WITH_LIVE_GROUP",
                    action_required=True,
                    ledger=ledger,
                    termination=termination,
                )
                return 21
            completed = _controller_completed_by_terminal(
                terminal_receipt
            )
            completed_no_claim = (
                _controller_completed_no_claim_by_terminal(
                    terminal_receipt
                )
            )
            write_action_receipt(
                runtime_root,
                status=(
                    "CONTROLLER_COMPLETED"
                    if completed
                    else (
                        "CONTROLLER_COMPLETED_NO_CLAIM"
                        if completed_no_claim
                        else "ACTION_REQUIRED"
                    )
                ),
                reason=(
                    "AUDITED_TERMINAL_RECEIPT_PRESENT"
                    if completed or completed_no_claim
                    else "CONTROLLER_EXITED_WITHOUT_FINAL_SCREEN_RECEIPT"
                ),
                action_required=not (completed or completed_no_claim),
                ledger=ledger,
                termination=None,
            )
            return 0 if completed or completed_no_claim else 22
        time.sleep(POLL_SECONDS)


def acquire_singleton_lock(runtime_root: Path) -> int:
    runtime_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path(runtime_root), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise SupervisorError("screen supervisor is already active") from error
    return descriptor


def start(
    runtime_root: Path,
    *,
    controller_command: Sequence[str],
    terminal_receipt: Path | None,
    term_grace_seconds: float,
) -> dict[str, Any]:
    if not controller_command:
        raise SupervisorError("start requires a controller command")
    lock_fd = acquire_singleton_lock(runtime_root)
    controller_log = runtime_root / "supervisor_controller.log"
    watchdog_log = runtime_root / WATCHDOG_LOG_NAME
    controller_handle = controller_log.open("ab", buffering=0)
    watchdog_handle = watchdog_log.open("ab", buffering=0)
    controller: subprocess.Popen[bytes] | None = None
    try:
        deadline, ledger = load_or_create_authoritative_deadline(runtime_root)
        if ledger["boot_id"] != current_boot_id():
            raise SupervisorError(
                "boot_id changed; audit stale ledger before any restart"
            )
        if _deadline_seconds(deadline) <= 0:
            write_action_receipt(
                runtime_root,
                status="INCOMPLETE_NO_CLAIM",
                reason="DEADLINE_ALREADY_EXPIRED_ON_RESTART",
                action_required=True,
                ledger=ledger,
                termination=None,
            )
            raise SupervisorError("authoritative screen deadline already expired")
        environment = os.environ.copy()
        environment["V5_3_12H_T0_UTC"] = deadline["started_at_utc"]
        environment["V5_3_12H_DEADLINE_RECEIPT"] = str(
            deadline_path(runtime_root).resolve()
        )
        controller = subprocess.Popen(
            list(controller_command),
            stdin=subprocess.DEVNULL,
            stdout=controller_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
        controller_pgid = os.getpgid(controller.pid)
        ledger.update(
            {
                "launch_generation": int(ledger.get("launch_generation", 0))
                + 1,
                "controller": {
                    "pid": controller.pid,
                    "pgid": controller_pgid,
                    "command": list(controller_command),
                    "command_sha256": canonical_sha256(
                        list(controller_command)
                    ),
                    "launched_at_utc": utc_now().isoformat(),
                },
                "launcher": {
                    "pid": os.getpid(),
                    "pgid": os.getpgrp(),
                },
                "last_transition": "CONTROLLER_LAUNCHED",
            }
        )
        supervisor_script = Path(__file__).resolve()
        watchdog_command = [
            sys.executable,
            str(supervisor_script),
            "watchdog",
            "--runtime-root",
            str(runtime_root),
            "--lock-fd",
            str(lock_fd),
            "--term-grace-seconds",
            str(term_grace_seconds),
        ]
        if terminal_receipt is not None:
            watchdog_command += [
                "--terminal-receipt",
                str(terminal_receipt),
            ]
        # Write a provisional ledger, then atomically replace it after the
        # detached watchdog PID/PGID is known.
        ledger["canonical_sha256"] = canonical_sha256(_ledger_core(ledger))
        atomic_write(ledger_path(runtime_root), ledger)
        watchdog = subprocess.Popen(
            watchdog_command,
            stdin=subprocess.DEVNULL,
            stdout=watchdog_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=(lock_fd,),
        )
        _DETACHED_HANDLES.extend((controller, watchdog))
        ledger["watchdog"] = {
            "pid": watchdog.pid,
            "pgid": os.getpgid(watchdog.pid),
            "launched_at_utc": utc_now().isoformat(),
            "log": str(watchdog_log.resolve()),
        }
        ledger["last_transition"] = "WATCHDOG_ARMED"
        ledger["canonical_sha256"] = canonical_sha256(_ledger_core(ledger))
        atomic_write(ledger_path(runtime_root), ledger)
        return {
            "status": "STARTED",
            "controller_pid": controller.pid,
            "controller_pgid": controller_pgid,
            "watchdog_pid": watchdog.pid,
            "watchdog_pgid": ledger["watchdog"]["pgid"],
            "started_at_utc": deadline["started_at_utc"],
            "deadline_utc": deadline["core_result_target_deadline_utc"],
            "resume_does_not_reset_t0": True,
        }
    except Exception:
        if controller is not None and controller.poll() is None:
            try:
                os.killpg(os.getpgid(controller.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    finally:
        controller_handle.close()
        watchdog_handle.close()
        os.close(lock_fd)


def request_stop(runtime_root: Path) -> dict[str, Any]:
    _, ledger = validate_deadline_and_ledger(
        runtime_root,
        require_current_boot=True,
    )
    request = {
        "protocol": f"{SUPERVISOR_PROTOCOL}:stop_request_v1",
        "requested_at_utc": utc_now().isoformat(),
        "requester_pid": os.getpid(),
        "requester_pgid": os.getpgrp(),
        "target_controller_pgid": (ledger.get("controller") or {}).get("pgid"),
    }
    request["canonical_sha256"] = canonical_sha256(request)
    path = stop_request_path(runtime_root)
    if path.exists():
        existing = read_json(path)
        return existing
    write_exclusive(path, request)
    return request


def audit(runtime_root: Path) -> dict[str, Any]:
    deadline, ledger = validate_deadline_and_ledger(
        runtime_root,
        require_current_boot=False,
    )
    boot_matches = ledger["boot_id"] == current_boot_id()
    controller = ledger.get("controller") or {}
    watchdog = ledger.get("watchdog") or {}
    payload = {
        "protocol": f"{SUPERVISOR_PROTOCOL}:audit_v1",
        "status": "PASS" if boot_matches else "ACTION_REQUIRED",
        "boot_id_matches": boot_matches,
        "authoritative_t0_utc": deadline["started_at_utc"],
        "deadline_utc": deadline["core_result_target_deadline_utc"],
        "remaining_seconds": max(0.0, _deadline_seconds(deadline)),
        "resume_does_not_reset_t0": True,
        "controller_pid": controller.get("pid"),
        "controller_pgid": controller.get("pgid"),
        "controller_alive": _pid_alive(controller.get("pid")),
        "controller_group_alive": _group_alive(controller.get("pgid")),
        "watchdog_pid": watchdog.get("pid"),
        "watchdog_pgid": watchdog.get("pgid"),
        "watchdog_alive": _pid_alive(watchdog.get("pid")),
        "latest_receipt": (
            read_json(receipt_path(runtime_root))
            if receipt_path(runtime_root).is_file()
            else None
        ),
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument(
        "--term-grace-seconds",
        type=float,
        default=DEFAULT_TERM_GRACE_SECONDS,
    )
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("controller_command", nargs=argparse.REMAINDER)
    raw = sys.argv[1:]
    actions = {"start", "status", "audit", "stop", "watchdog"}
    if raw[:1] in (["-h"], ["--help"]):
        parser.print_help()
        print(
            "\naction must be one of: " + ", ".join(sorted(actions)),
            file=sys.stdout,
        )
        raise SystemExit(0)
    if not raw or raw[0] not in actions:
        parser.error(
            "first argument must be one of: "
            + ", ".join(sorted(actions))
        )
    action = raw.pop(0)
    args = parser.parse_args(raw)
    args.action = action
    args.runtime_root = args.runtime_root.expanduser().resolve()
    if args.terminal_receipt is not None:
        args.terminal_receipt = args.terminal_receipt.expanduser().resolve()
    if args.controller_command[:1] == ["--"]:
        args.controller_command = args.controller_command[1:]
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.action == "start":
            result = start(
                args.runtime_root,
                controller_command=args.controller_command,
                terminal_receipt=args.terminal_receipt,
                term_grace_seconds=args.term_grace_seconds,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.action in {"status", "audit"}:
            audit(args.runtime_root)
        elif args.action == "stop":
            print(
                json.dumps(
                    request_stop(args.runtime_root),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.action == "watchdog":
            if args.lock_fd is None:
                raise SupervisorError("watchdog requires inherited --lock-fd")
            raise SystemExit(
                watchdog_loop(
                    args.runtime_root,
                    lock_fd=args.lock_fd,
                    term_grace_seconds=args.term_grace_seconds,
                    terminal_receipt=args.terminal_receipt,
                )
            )
    except (SupervisorError, UnsafeProcessGroup) as error:
        if args.action == "watchdog":
            try:
                _, ledger = validate_deadline_and_ledger(
                    args.runtime_root,
                    require_current_boot=False,
                )
                write_action_receipt(
                    args.runtime_root,
                    status="ACTION_REQUIRED",
                    reason=f"WATCHDOG_FAIL_CLOSED: {type(error).__name__}: {error}",
                    action_required=True,
                    ledger=ledger,
                    termination=None,
                )
            except Exception:
                try:
                    write_emergency_action_receipt(
                        args.runtime_root,
                        reason=(
                            "WATCHDOG_FAIL_CLOSED_WITH_UNTRUSTED_LEDGER: "
                            f"{type(error).__name__}: {error}"
                        ),
                    )
                except Exception:
                    pass
        print(f"supervisor stopped: {error}", file=sys.stderr)
        raise SystemExit(20) from error


if __name__ == "__main__":
    main()
