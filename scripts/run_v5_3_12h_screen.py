#!/usr/bin/env python3
"""Unattended, resumable controller for the isolated V5.3 12-hour screen.

The first ``--stage all`` invocation freezes a UTC deadline receipt.  Every
later invocation reuses that receipt; restarting the controller never restores
time.  Expiry terminates process groups (TERM, then KILL), atomically records
``INCOMPLETE_NO_CLAIM``, and never fabricates a partial metric.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable

if __package__:
    # Package imports must use one canonical module identity.  Falling back to
    # top-level imports here can create a second copy whose mutable hardware
    # constants leak into unrelated formal-controller tests.
    from . import build_v5_checkpoint_registry as registry_builder
    from . import prepare_v5_3_12h_screen as screen_data
    from . import run_v5_3_single_host as base
    from . import run_v5_3_train_only_pilot as pilot
    from . import run_v5_sft_causal_eval as evaluator
    from . import summarize_v5_3_12h_screen as screen_summarizer
    from . import summarize_v5_sft_causal as causal_summary
    from . import v5_3_12h_protocol as protocol
    from . import v5_judge_audit_contract as judge_contract
else:
    import build_v5_checkpoint_registry as registry_builder
    import prepare_v5_3_12h_screen as screen_data
    import run_v5_3_single_host as base
    import run_v5_3_train_only_pilot as pilot
    import run_v5_sft_causal_eval as evaluator
    import summarize_v5_3_12h_screen as screen_summarizer
    import summarize_v5_sft_causal as causal_summary
    import v5_3_12h_protocol as protocol
    import v5_judge_audit_contract as judge_contract


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "artifacts/v5_stage0/manifests/split_manifest.json"
STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
TEACHER_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
USER_MODEL = protocol.USER_JUDGE_MODEL
USER_REVISION = protocol.USER_JUDGE_REVISION
GENERATION_SHARDS = 3
EVALUATION_SHARDS = 3
TERM_GRACE_SECONDS = 30
SCREEN_GPU_MODEL = "NVIDIA GeForce RTX 5090"
SCREEN_MIN_GPU_MEMORY_MIB = 30_000
SCREEN_MIN_FREE_DISK_GIB = protocol.MIN_FREE_DISK_GIB
_ACTIVE_PROCESS_GROUPS: set[int] = set()
_ACTIVE_LOCK = threading.Lock()
_DEADLINE_MONOTONIC: float | None = None
_GLOBAL_ARGS: argparse.Namespace | None = None
_CONTROLLER_LOCK_HANDLE: Any | None = None


class ScreenStageError(RuntimeError):
    """A screen barrier failed closed."""


class DeadlineExpired(ScreenStageError):
    """The immutable 12-hour deadline elapsed."""


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
        raise ScreenStageError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ScreenStageError(f"expected JSON object: {path}")
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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_execution_commit(expected: str) -> str:
    observed = git_commit()
    if len(expected) != 40 or expected.lower() != expected:
        raise ScreenStageError("--expected-source-commit must be a full commit")
    if observed != expected:
        raise ScreenStageError(
            f"execution commit drift: expected {expected}, observed {observed}"
        )
    return observed


def _path_relation(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def validate_isolation(args: argparse.Namespace) -> None:
    screen_roots = {
        "protocol": args.protocol_root,
        "raw": args.raw_root,
        "processed": args.processed_root,
        "results": args.results_root,
        "runtime": args.runtime_root,
    }
    if len({path.resolve() for path in screen_roots.values()}) != len(
        screen_roots
    ):
        raise ScreenStageError("screen roots must be distinct")
    forbidden = {
        "formal_protocol": ROOT / "data/processed/v5_3_protocol",
        "formal_raw": ROOT / "data/raw/v5_3_sft_causal_generation",
        "formal_processed": ROOT / "data/processed/v5_3_sft_causal",
        "formal_results": ROOT / "results/v5_3_sft_causal",
        "pilot": ROOT / "artifacts/v5_3_train_only_pilot",
        "official_stage0": ROOT / "artifacts/v5_stage0",
    }
    for screen_name, screen_path in screen_roots.items():
        for frozen_name, frozen_path in forbidden.items():
            if _path_relation(screen_path, frozen_path):
                raise ScreenStageError(
                    f"{screen_name} root aliases/contains frozen {frozen_name}"
                )


def configure_screen_hardware() -> None:
    """Select isolated screen host limits without changing source defaults."""

    base.EXPECTED_GPU_MODEL = SCREEN_GPU_MODEL
    base.MIN_GPU_MEMORY_MIB = SCREEN_MIN_GPU_MEMORY_MIB
    base.MIN_FREE_DISK_GIB = SCREEN_MIN_FREE_DISK_GIB


def deadline_path(args: argparse.Namespace) -> Path:
    return args.runtime_root / "deadline_receipt.json"


def acquire_controller_lock(args: argparse.Namespace) -> None:
    global _CONTROLLER_LOCK_HANDLE
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    path = args.runtime_root / "controller.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise ScreenStageError("another screen controller owns the lock") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    _CONTROLLER_LOCK_HANDLE = handle


def release_controller_lock() -> None:
    global _CONTROLLER_LOCK_HANDLE
    if _CONTROLLER_LOCK_HANDLE is None:
        return
    fcntl.flock(_CONTROLLER_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    _CONTROLLER_LOCK_HANDLE.close()
    _CONTROLLER_LOCK_HANDLE = None


def load_or_create_deadline(
    args: argparse.Namespace, *, allow_create: bool
) -> dict[str, Any]:
    path = deadline_path(args)
    if not path.exists():
        if not allow_create:
            raise ScreenStageError(
                "deadline receipt is absent; start once with --stage all or preflight"
            )
        receipt = protocol.make_deadline_receipt(datetime.now(timezone.utc))
        write_exclusive(path, receipt)
    receipt = read_json(path)
    protocol.validate_deadline_receipt(receipt)
    return receipt


def remaining_seconds(receipt: dict[str, Any]) -> float:
    deadline = datetime.fromisoformat(
        receipt["core_result_target_deadline_utc"]
    )
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def _register_process(process: subprocess.Popen[Any]) -> None:
    group = os.getpgid(process.pid)
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESS_GROUPS.add(group)
    _persist_process_ledger()


def _unregister_process(process: subprocess.Popen[Any]) -> None:
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        group = process.pid
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESS_GROUPS.discard(group)
        _ACTIVE_PROCESS_GROUPS.discard(process.pid)
    _persist_process_ledger()


def process_ledger_path(args: argparse.Namespace) -> Path:
    return args.runtime_root / "active_process_groups.json"


def _persist_process_ledger() -> None:
    args = _GLOBAL_ARGS
    if args is None:
        return
    with _ACTIVE_LOCK:
        groups = sorted(_ACTIVE_PROCESS_GROUPS)
    atomic_write(
        process_ledger_path(args),
        {
            "protocol": f"{protocol.PROTOCOL}:process_group_ledger_v1",
            "controller_pid": os.getpid(),
            "controller_process_group": os.getpgrp(),
            "active_new_session_process_groups": groups,
            "group_identities": {
                str(group): _process_identity(group) for group in groups
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    if len(stat) < 22:
        return None
    return {
        "pid": pid,
        "start_time_ticks": stat[21],
        "cmdline_sha256": hashlib.sha256(command).hexdigest(),
    }


def _descendant_pids(pid: int) -> set[int]:
    """Return Linux descendants so subprocess.run children are also stopped."""

    observed: set[int] = set()
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [
                int(value)
                for value in children_path.read_text(encoding="utf-8").split()
            ]
        except (OSError, ValueError):
            children = []
        for child in children:
            if child not in observed:
                observed.add(child)
                frontier.append(child)
    return observed


def _termination_targets() -> tuple[set[int], set[int]]:
    with _ACTIVE_LOCK:
        groups = set(_ACTIVE_PROCESS_GROUPS)
    controller_group = os.getpgrp()
    groups.discard(controller_group)
    descendants = _descendant_pids(os.getpid())
    individual = set()
    for pid in descendants:
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group not in groups:
            individual.add(pid)
    return groups, individual


def _terminate_all_process_groups() -> None:
    groups, individual = _termination_targets()
    for group in sorted(groups):
        try:
            os.killpg(group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in sorted(individual, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    end = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < end:
        live_groups = []
        for group in groups:
            try:
                os.killpg(group, 0)
                live_groups.append(group)
            except (ProcessLookupError, PermissionError):
                pass
        live_pids = []
        for pid in individual:
            try:
                os.kill(pid, 0)
                live_pids.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        if not live_groups and not live_pids:
            return
        time.sleep(0.25)
    for group in sorted(groups):
        try:
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in sorted(individual, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def terminal_path(args: argparse.Namespace) -> Path:
    return args.results_root / "TERMINAL_STATUS.json"


def write_terminal(
    args: argparse.Namespace,
    *,
    status: str,
    reason: str,
    no_claim: bool,
) -> None:
    payload = {
        "protocol": f"{protocol.PROTOCOL}:terminal_v1",
        "status": status,
        "reason": reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_receipt": str(deadline_path(args).resolve()),
        "deadline_receipt_sha256": (
            sha256_file(deadline_path(args))
            if deadline_path(args).is_file()
            else None
        ),
        "no_scientific_claim": no_claim,
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
    atomic_write(terminal_path(args), payload)


def validated_core_snapshot(args: argparse.Namespace) -> bool:
    """Verify enough immutable bindings to preserve an already finished core.

    This cutoff check deliberately does not recompute metrics.  The complete
    summary was already built from all raw cases by the strict summarizer; we
    bind that snapshot back to the canonical core receipt, deadline, registry,
    and current static protocol before allowing an interrupted extension to
    leave a claim-bearing core result.
    """

    summary_path = args.results_root / "core_screen_summary.json"
    core_path = args.runtime_root / "core_complete_receipt.json"
    deadline_file = deadline_path(args)
    registry_path = args.results_root / "checkpoint_registry.json"
    required = (summary_path, core_path, deadline_file, registry_path)
    if any(not path.is_file() for path in required):
        return False
    try:
        summary = read_json(summary_path)
        core = read_json(core_path)
        core_body = {
            key: value
            for key, value in core.items()
            if key != "canonical_sha256"
        }
        provenance = summary.get("provenance")
        official_test = summary.get("official_test")
        completed_at = datetime.fromisoformat(core["completed_at_utc"])
        hard_deadline = datetime.fromisoformat(
            read_json(deadline_file)["core_result_target_deadline_utc"]
        )
        return bool(
            summary.get("protocol") == screen_summarizer.SUMMARY_PROTOCOL
            and summary.get("status") == "CORE_COMPLETE"
            and summary.get("core_raw_cases") == protocol.CORE_ROLLOUTS
            and set(summary.get("arms") or {}) == set(protocol.CORE_EVAL_ARMS)
            and isinstance(official_test, dict)
            and official_test.get("used") is False
            and official_test.get("status") == "SEALED"
            and isinstance(provenance, dict)
            and provenance.get("core_complete_receipt_sha256")
            == sha256_file(core_path)
            and provenance.get("deadline_receipt_sha256")
            == sha256_file(deadline_file)
            and provenance.get("checkpoint_registry_sha256")
            == sha256_file(registry_path)
            and provenance.get("static_protocol_artifact_sha256")
            == screen_static_artifact_hashes()
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
            and core.get("static_protocol_artifact_sha256")
            == screen_static_artifact_hashes()
            and core.get("canonical_sha256")
            == protocol.canonical_sha256(core_body)
            and completed_at.tzinfo is not None
            and hard_deadline.tzinfo is not None
            and completed_at <= hard_deadline
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        ScreenStageError,
    ):
        return False


def _deadline_signal(_signum: int, _frame: Any) -> None:
    args = _GLOBAL_ARGS
    if args is not None:
        core_valid = validated_core_snapshot(args)
        write_terminal(
            args,
            status=(
                "CORE_COMPLETE_EXTENSION_INCOMPLETE"
                if core_valid
                else "INCOMPLETE_NO_CLAIM"
            ),
            reason=(
                "IMMUTABLE_CORE_RETAINED_AT_12_HOUR_EXTENSION_CUTOFF"
                if core_valid
                else "IMMUTABLE_12_HOUR_DEADLINE_EXPIRED"
            ),
            no_claim=not core_valid,
        )
    # Persist the scientific terminal state before waiting on stubborn
    # children; the detached supervisor may SIGKILL this group at its grace
    # boundary.
    _terminate_all_process_groups()
    raise DeadlineExpired("immutable 12-hour deadline expired")


def _external_stop_signal(signum: int, _frame: Any) -> None:
    args = _GLOBAL_ARGS
    if args is not None:
        core_valid = validated_core_snapshot(args)
        write_terminal(
            args,
            status=(
                "CORE_COMPLETE_EXTENSION_INCOMPLETE"
                if core_valid
                else "INCOMPLETE_NO_CLAIM"
            ),
            reason=(
                "VALIDATED_CORE_RETAINED_AFTER_CONTROLLER_SIGNAL_"
                f"{signal.Signals(signum).name}"
                if core_valid
                else f"CONTROLLER_SIGNAL_{signal.Signals(signum).name}"
            ),
            no_claim=not core_valid,
        )
    _terminate_all_process_groups()
    raise ScreenStageError("controller interrupted and children stopped")


def arm_watchdog(
    args: argparse.Namespace, receipt: dict[str, Any]
) -> dict[str, Any]:
    global _DEADLINE_MONOTONIC, _GLOBAL_ARGS
    remaining = remaining_seconds(receipt)
    if remaining <= 0:
        write_terminal(
            args,
            status="INCOMPLETE_NO_CLAIM",
            reason="DEADLINE_ALREADY_EXPIRED_ON_RESUME",
            no_claim=True,
        )
        raise DeadlineExpired("deadline already expired")
    _GLOBAL_ARGS = args
    _DEADLINE_MONOTONIC = time.monotonic() + remaining
    signal.signal(signal.SIGALRM, _deadline_signal)
    signal.signal(signal.SIGTERM, _external_stop_signal)
    signal.signal(signal.SIGINT, _external_stop_signal)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    invocation = {
        "protocol": f"{protocol.PROTOCOL}:controller_invocation_v1",
        "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_anchor_ns": time.monotonic_ns(),
        "remaining_seconds_at_start": remaining,
        "deadline_receipt_sha256": sha256_file(deadline_path(args)),
        "resume_does_not_reset_deadline": True,
    }
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    with (args.runtime_root / "controller_invocations.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(invocation, sort_keys=True) + "\n")
    return invocation


def _screen_ports_in_use() -> list[int]:
    active = []
    for port in (8001, 8011, 8101, 8102, 8103):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.settimeout(0.2)
            if handle.connect_ex(("127.0.0.1", port)) == 0:
                active.append(port)
    return active


def emergency_stop(args: argparse.Namespace) -> dict[str, Any]:
    """Stop screen-owned process groups without commit/deadline barriers."""

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    ledger = process_ledger_path(args)
    unsafe_stale_groups = []
    if ledger.is_file():
        value = read_json(ledger)
        identities = value.get("group_identities")
        groups = value.get("active_new_session_process_groups")
        if (
            value.get("protocol")
            != f"{protocol.PROTOCOL}:process_group_ledger_v1"
            or not isinstance(identities, dict)
            or not isinstance(groups, list)
        ):
            raise ScreenStageError("invalid process-group ledger")
        for raw_group in groups:
            group = int(raw_group)
            expected = identities.get(str(group))
            observed = _process_identity(group)
            if observed is None:
                continue
            if observed != expected:
                unsafe_stale_groups.append(group)
                continue
            if group != os.getpgrp():
                with _ACTIVE_LOCK:
                    _ACTIVE_PROCESS_GROUPS.add(group)
    if unsafe_stale_groups:
        raise ScreenStageError(
            "refusing reused/stale process groups: "
            f"{unsafe_stale_groups}"
        )
    try:
        base.stop_stale_services(args)
    except base.StageError:
        # The bound ledger remains authoritative for non-vLLM workers.
        pass
    _terminate_all_process_groups()
    ports = _screen_ports_in_use()
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESS_GROUPS.clear()
    _persist_process_ledger()
    receipt = {
        "protocol": f"{protocol.PROTOCOL}:emergency_stop_v1",
        "status": "PASS" if not ports else "USER_ACTION_REQUIRED_STOP_POD",
        "screen_ports_still_in_use": ports,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "controller_commit_or_deadline_barrier_required": False,
        "runpod_management_plane_stop_performed": False,
        "runpod_stop_is_outer_agent_or_user_responsibility": True,
    }
    atomic_write(args.runtime_root / "EMERGENCY_STOP_RECEIPT.json", receipt)
    if ports:
        raise ScreenStageError(
            f"screen ports remain active after cleanup: {ports}"
        )
    print(json.dumps(receipt, indent=2))
    return receipt


def guard_deadline() -> None:
    if _DEADLINE_MONOTONIC is not None and time.monotonic() >= _DEADLINE_MONOTONIC:
        _deadline_signal(signal.SIGALRM, None)


def spawn_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _register_process(process)
    return process, handle


def wait_processes(
    rows: Iterable[tuple[subprocess.Popen[bytes], Any, str]]
) -> None:
    active = list(rows)
    failures: list[tuple[str, int]] = []
    try:
        while active:
            guard_deadline()
            remaining = []
            for process, handle, label in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, handle, label))
                elif code:
                    failures.append((label, code))
            active = remaining
            if failures:
                for process, _, _ in active:
                    base.stop_process(process)
                raise ScreenStageError(f"worker failures: {failures}")
            if active:
                time.sleep(1)
    finally:
        for process, handle, _ in list(rows):
            if process.poll() is None:
                base.stop_process(process)
            _unregister_process(process)
            handle.close()


def stop_registered_services(
    services: Iterable[tuple[subprocess.Popen[bytes], Any, Path]]
) -> None:
    rows = list(services)
    try:
        base.stop_services(rows)
    finally:
        for process, _, _ in rows:
            _unregister_process(process)


def screen_protocol_files(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "source_generation": args.protocol_root
        / "source_generation_manifest.json",
        "validation": args.protocol_root / "validation_manifest.json",
        "generation_audit": args.protocol_root
        / "generation_dynamic_audit.json",
        "validation_audit": args.protocol_root
        / "validation_dynamic_audit.json",
        "universe": args.protocol_root / "effective_task_universe.json",
        "source_pilot": args.protocol_root / "source_pilot_manifest.json",
        "screen": args.protocol_root / "screen_manifest.json",
    }


def static_screen_audit(args: argparse.Namespace) -> None:
    protocol.validate_constants()
    validate_isolation(args)
    if sha256_file(SPLIT) != base.SPLIT_SHA256:
        raise ScreenStageError("Stage-0 split hash drift")
    config = ROOT / "configs/v5_3_12h_screen.yaml"
    handoff = ROOT / "V5_3_12H_SCREEN_HANDOFF.md"
    prompt = ROOT / "V5_3_12H_RUNPOD_4X5090_AGENT_PROMPT.md"
    supervisor = ROOT / "scripts/run_v5_3_12h_supervisor.py"
    for path in (config, handoff, prompt, supervisor):
        if not path.is_file():
            raise ScreenStageError(f"missing screen protocol artifact: {path}")
    text = config.read_text(encoding="utf-8")
    reject_duplicate_yaml_mapping_keys(text)
    required = {
        "protocol: v5_3_12h_exploratory_screen_v1",
        "seed: 20260731",
        "minimum_free_disk_gib: 80",
        "expected_rollouts: 288",
        "minimum_tasks_with_pair: 14",
        "minimum_capped_pairs: 17",
        "maximum_pairs_per_task: 2",
        "optimizer_steps: 64",
        "mixture_weight_unit: training_row",
        "repair_50: 0.5",
        "recovery_row_ratio_tolerance: 0.0",
        "core_rollouts: 126",
    }
    missing = sorted(fragment for fragment in required if fragment not in text)
    if missing:
        raise ScreenStageError(f"screen config drift: {missing}")
    for path in (handoff, prompt):
        if "TODO_" in path.read_text(encoding="utf-8"):
            raise ScreenStageError(f"unresolved executable TODO in {path.name}")


def screen_static_artifact_hashes() -> dict[str, str]:
    paths = (
        ROOT / "configs/v5_3_12h_screen.yaml",
        ROOT / "V5_3_12H_SCREEN_HANDOFF.md",
        ROOT / "V5_3_12H_RUNPOD_4X5090_AGENT_PROMPT.md",
        ROOT / "scripts/v5_3_12h_protocol.py",
        ROOT / "scripts/prepare_v5_3_12h_screen.py",
        ROOT / "scripts/run_v5_3_12h_screen.py",
        ROOT / "scripts/run_v5_3_12h_supervisor.py",
        ROOT / "scripts/summarize_v5_3_12h_screen.py",
        ROOT / "scripts/train_v5_sft_causal.py",
        ROOT / "scripts/build_v5_checkpoint_registry.py",
    )
    return {
        str(path.relative_to(ROOT)): sha256_file(path) for path in paths
    }


def phase_hardware_receipt(
    args: argparse.Namespace,
    phase: str,
    *,
    bound_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Revalidate exact 4×5090 and bind it to each scientific phase."""

    inventory = base.validate_gpu_inventory()
    preflight_path = args.results_root / "preflight.json"
    if not base.preflight_complete(args) or not preflight_path.is_file():
        raise ScreenStageError(
            f"{phase}: current host no longer matches immutable preflight"
        )
    preflight = read_json(preflight_path)
    if (
        preflight.get("expected_gpu_model") != SCREEN_GPU_MODEL
        or preflight.get("gpus") != inventory
        or (preflight.get("environments") or {})
        .get("train", {})
        .get("cuda")
        != "12.8"
    ):
        raise ScreenStageError(f"{phase}: exact 5090/CUDA 12.8 drift")
    bindings = {}
    for raw_path in bound_paths:
        path = raw_path.resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ScreenStageError(f"{phase}: bound artifact absent: {path}")
        bindings[str(path)] = sha256_file(path)
    payload = {
        "protocol": f"{protocol.PROTOCOL}:phase_hardware_v1",
        "phase": phase,
        "source_commit": args.expected_source_commit,
        "gpu_inventory": inventory,
        "expected_gpu_model": SCREEN_GPU_MODEL,
        "minimum_gpu_memory_mib": SCREEN_MIN_GPU_MEMORY_MIB,
        "cuda_version": "12.8",
        "host_preflight_sha256": sha256_file(preflight_path),
        "static_protocol_artifact_sha256": screen_static_artifact_hashes(),
        "bound_artifact_sha256": dict(sorted(bindings.items())),
        "mid_run_gpu_substitution_allowed": False,
        "official_test_used": False,
    }
    payload["canonical_sha256"] = protocol.canonical_sha256(payload)
    path = args.runtime_root / "phase_hardware" / f"{phase}.json"
    if path.is_file():
        if read_json(path) != payload:
            raise ScreenStageError(f"{phase}: hardware receipt drift")
    else:
        write_exclusive(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "canonical_sha256": payload["canonical_sha256"],
    }


def reject_duplicate_yaml_mapping_keys(text: str) -> None:
    """Reject duplicate keys in the simple mapping-style protocol YAML."""

    stack: list[tuple[int, str]] = []
    observed: dict[tuple[str, ...], set[str]] = {}
    pattern = re.compile(r"^( *)([A-Za-z0-9_.-]+):(?:\s|$)")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        match = pattern.match(line)
        if match is None:
            continue
        indent = len(match.group(1))
        key = match.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = tuple(value for _, value in stack)
        siblings = observed.setdefault(parent, set())
        if key in siblings:
            raise ScreenStageError(
                f"duplicate YAML key at line {number}: "
                f"{'.'.join((*parent, key))}"
            )
        siblings.add(key)
        remainder = line[match.end() :].split("#", 1)[0].strip()
        if not remainder:
            stack.append((indent, key))


def preflight(args: argparse.Namespace) -> None:
    static_screen_audit(args)
    base.preflight(args)


def build_protocol(args: argparse.Namespace) -> None:
    base.build_protocol(args)
    files = screen_protocol_files(args)
    base_generation = args.protocol_root / "generation_manifest.json"
    source_bytes = base_generation.read_bytes()
    if files["source_generation"].exists():
        if files["source_generation"].read_bytes() != source_bytes:
            raise ScreenStageError("source-generation copy drift")
    else:
        files["source_generation"].write_bytes(source_bytes)
    universe, source_pilot = pilot.build_effective_universe_and_pilot(
        tau2_root=args.tau2_root,
        split_manifest_path=SPLIT,
        generation_manifest_path=files["source_generation"],
    )
    generation = read_json(files["source_generation"])
    screen = screen_data.build_screen_manifest(
        effective_universe=universe,
        pilot_manifest=source_pilot,
        generation_manifest=generation,
        generation_manifest_sha256=sha256_file(files["source_generation"]),
    )
    for path, value in (
        (files["universe"], universe),
        (files["source_pilot"], source_pilot),
        (files["screen"], screen),
    ):
        if path.exists():
            if read_json(path) != value:
                raise ScreenStageError(f"immutable protocol artifact drift: {path}")
        else:
            write_exclusive(path, value)
    screen_data.validate_screen_manifest(
        screen,
        generation_manifest_sha256=sha256_file(files["source_generation"]),
        generation_manifest=generation,
        effective_universe=universe,
        pilot_manifest=source_pilot,
    )


def generation_complete(args: argparse.Namespace) -> bool:
    files = screen_protocol_files(args)
    try:
        dynamic = screen_data.load_complete_dynamic_audit(
            files["generation_audit"],
            manifest_path=files["source_generation"],
            split_manifest_path=SPLIT,
            expected_source_split="derived_inner_train",
            expected_task_ids={
                f"{row['domain']}:{row['task_id']}"
                for row in read_json(files["source_generation"])["rows"]
            },
        )
        screen_data.validate_generation_contracts(
            raw_dir=args.raw_root,
            screen_manifest_path=files["screen"],
            generation_manifest_path=files["source_generation"],
            dynamic_identity=dynamic,
            expected_source_commit=args.expected_source_commit,
        )
        pools, _ = screen_data._load_attempts(
            args.raw_root, task_ids=set(protocol.PILOT_TASK_IDS)
        )
        return all(
            len(pools[condition][task]) == 6
            for condition in protocol.CONDITIONS
            for task in protocol.PILOT_TASK_IDS
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def _archive_partial_generation(args: argparse.Namespace, shard: int) -> None:
    contract = args.raw_root / (
        f"run_contract.shard-{shard:03d}-of-{GENERATION_SHARDS:03d}.json"
    )
    if not contract.exists():
        return
    try:
        if read_json(contract).get("status") == "COMPLETE":
            return
    except ScreenStageError:
        pass
    archive = (
        args.runtime_root
        / "failed_attempts"
        / f"generation-shard-{shard}-{time.time_ns()}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in [contract, *args.raw_root.glob(f"*.shard-{shard:03d}-of-003.json")]:
        if path.exists():
            shutil.move(str(path), archive / path.name)


def generate(args: argparse.Namespace) -> None:
    if generation_complete(args):
        print("[skip] exact 288-rollout screen generation is COMPLETE")
        return
    files = screen_protocol_files(args)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    missing = []
    for shard in range(GENERATION_SHARDS):
        _archive_partial_generation(args, shard)
        contract = args.raw_root / (
            f"run_contract.shard-{shard:03d}-of-003.json"
        )
        if not contract.is_file():
            missing.append(shard)
    services = base.start_generation_services(args, phase="screen")
    for process, _, _ in services:
        _register_process(process)
    workers: list[tuple[subprocess.Popen[bytes], Any, str]] = []
    try:
        runtime_evidence = base.generation_runtime_preflight_path(
            args, phase="screen"
        )
        for shard in missing:
            command = [
                str(args.serve_python),
                "scripts/run_v5_sft_causal_generate.py",
                "--tau2-root", str(args.tau2_root),
                "--split-manifest", str(SPLIT),
                "--manifest", str(files["screen"]),
                "--dynamic-audit", str(files["generation_audit"]),
                "--runtime-evidence", str(runtime_evidence),
                "--max-model-len", "32768",
                "--output-dir", str(args.raw_root),
                "--teacher-model", TEACHER_MODEL,
                "--teacher-revision", TEACHER_REVISION,
                "--teacher-api-base", "http://127.0.0.1:8011/v1",
                "--teacher-api-key", "stage1-teacher-local",
                "--user-model", USER_MODEL,
                "--user-revision", USER_REVISION,
                "--user-api-base", "http://127.0.0.1:8001/v1",
                "--user-api-key", "stage1-user-local",
                "--judge-model", USER_MODEL,
                "--judge-revision", USER_REVISION,
                "--judge-api-base", "http://127.0.0.1:8001/v1",
                "--judge-api-key", "stage1-user-local",
                "--teacher-mode", "ground_truth",
                "--condition", "both",
                "--shard-index", str(shard),
                "--num-shards", str(GENERATION_SHARDS),
                "--num-trials", "6",
                "--temperature", "0.2",
                "--top-p", "0.95",
                "--seed", str(protocol.BASE_SEED),
                "--max-steps", "60",
                "--timeout", "900",
                "--max-tokens", "512",
                "--expected-source-commit", args.expected_source_commit,
            ]
            process, handle = spawn_logged(
                command,
                env=base.base_env(offline=True),
                log_path=args.results_root
                / "logs"
                / f"generate-screen-shard-{shard}.log",
            )
            workers.append((process, handle, f"generation-shard-{shard}"))
        wait_processes(workers)
    finally:
        stop_registered_services(services)
    if not generation_complete(args):
        raise ScreenStageError("generation ended without exact COMPLETE contracts")


def data_complete(args: argparse.Namespace) -> bool:
    audit_path = args.processed_root / "audit.json"
    hashes_path = args.processed_root / "hashes.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        return False
    try:
        audit = read_json(audit_path)
        hashes = read_json(hashes_path)
        return (
            audit.get("status") == "PASS"
            and audit.get("decision") == "TRAIN"
            and audit.get("design_version") == protocol.DESIGN_VERSION
            and (audit.get("data_gate") or {}).get("training_authorized")
            is True
            and audit.get("official_test_used") is False
            and all(
                (args.processed_root / name).is_file()
                and sha256_file(args.processed_root / name) == digest
                for name, digest in hashes.items()
            )
            and hashes.get("audit.json") == sha256_file(audit_path)
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def prepare(args: argparse.Namespace) -> None:
    if data_complete(args):
        print("[skip] screen data gate and joint matcher are PASS")
        return
    if args.processed_root.exists() and any(args.processed_root.iterdir()):
        audit = args.processed_root / "audit.json"
        if audit.is_file():
            value = read_json(audit)
            if value.get("status") == "FAIL_CLOSED":
                decision = value.get("decision")
                if "matching" in value:
                    write_terminal(
                        args,
                        status="MATCHING_INCONCLUSIVE_NO_TRAIN",
                        reason="FIXED_JOINT_MATCHER_FOUND_NO_REGISTERED_SCHEDULE",
                        no_claim=True,
                    )
                else:
                    write_terminal(
                        args,
                        status="DATA_GATE_FAIL_NO_TRAIN",
                        reason=str(decision or "DATA_GATE_FAILED"),
                        no_claim=True,
                    )
                raise ScreenStageError("data preparation previously failed closed")
        raise ScreenStageError(
            f"partial processed root must be archived: {args.processed_root}"
        )
    files = screen_protocol_files(args)
    command = [
        str(args.train_python),
        "scripts/prepare_v5_3_12h_screen.py",
        "--split-manifest", str(SPLIT),
        "--generation-manifest", str(files["source_generation"]),
        "--validation-manifest", str(files["validation"]),
        "--screen-manifest", str(files["screen"]),
        "--generation-dynamic-audit", str(files["generation_audit"]),
        "--validation-dynamic-audit", str(files["validation_audit"]),
        "--raw-dir", str(args.raw_root),
        "--tau2-root", str(args.tau2_root),
        "--output-dir", str(args.processed_root),
        "--tokenizer", STUDENT_MODEL,
        "--tokenizer-revision", STUDENT_REVISION,
        "--expected-source-commit", args.expected_source_commit,
        "--local-files-only",
    ]
    process, handle = spawn_logged(
        command,
        env=base.base_env(offline=True, gpu=0),
        log_path=args.results_root / "logs/prepare-screen.log",
    )
    wait_processes([(process, handle, "prepare-screen")])
    if not data_complete(args):
        audit = args.processed_root / "audit.json"
        if audit.is_file() and "matching" in read_json(audit):
            write_terminal(
                args,
                status="MATCHING_INCONCLUSIVE_NO_TRAIN",
                reason="FIXED_JOINT_MATCHER_FOUND_NO_REGISTERED_SCHEDULE",
                no_claim=True,
            )
        else:
            write_terminal(
                args,
                status="DATA_GATE_FAIL_NO_TRAIN",
                reason="REGISTERED_14_TASK_17_PAIR_GATE_FAILED",
                no_claim=True,
            )
        raise ScreenStageError("screen data preparation failed closed")


def training_complete(
    args: argparse.Namespace, arm: str, mode: str
) -> bool:
    path = args.results_root / "training" / arm / mode
    try:
        manifest = read_json(path / "run_manifest.json")
        checkpoint = path / "checkpoint_final"
        complete = (
            manifest.get("protocol") == registry_builder.TRAIN_PROTOCOL
            and manifest.get("source_commit") == args.expected_source_commit
            and manifest.get("model") == STUDENT_MODEL
            and manifest.get("model_revision") == STUDENT_REVISION
            and manifest.get("arm") == arm
            and manifest.get("mode") == mode
            and manifest.get("seed") == protocol.BASE_SEED
            and (manifest.get("data_provenance") or {}).get("design_version")
            == protocol.DESIGN_VERSION
            and manifest.get("effective_validation_rows") == 0
            and manifest.get("validation_disabled_reason")
            == "fixed_steps_exploratory"
            and (manifest.get("loss_audit") or {}).get(
                "validation_loss_values_checked"
            )
            == 0
            and (manifest.get("loss_audit") or {}).get(
                "final_validation_loss"
            )
            is None
            and (checkpoint / "adapter_model.safetensors").is_file()
            and (checkpoint / "adapter_config.json").is_file()
        )
        if not complete:
            return False
        if mode == "formal":
            registry_builder._validate_screen_training_manifest(
                manifest=manifest,
                adapter_config=read_json(
                    checkpoint / "adapter_config.json"
                ),
                arm=arm,
            )
        else:
            loss = manifest.get("loss_audit")
            if (
                manifest.get("effective_steps") != 2
                or manifest.get("effective_grad_accum") != 8
                or manifest.get("effective_rows") != 16
                or not isinstance(loss, dict)
                or loss.get("finite") is not True
                or loss.get("validation_loss_values_checked") != 0
                or loss.get("final_validation_loss") is not None
            ):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def _archive_partial_training(
    args: argparse.Namespace, arm: str, mode: str
) -> None:
    path = args.results_root / "training" / arm / mode
    if not path.exists() or training_complete(args, arm, mode):
        return
    archive = (
        args.runtime_root
        / "failed_attempts"
        / f"training-{arm}-{mode}-{time.time_ns()}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), archive)


def train(args: argparse.Namespace) -> None:
    phase_hardware_receipt(args, "train_entry")
    hashes = read_json(args.processed_root / "hashes.json")
    validation_sha = hashes["validation_loss.jsonl"]
    for mode in ("smoke", "formal"):
        workers: list[tuple[subprocess.Popen[bytes], Any, str]] = []
        for gpu, arm in enumerate(protocol.TRAINED_ARMS):
            if training_complete(args, arm, mode):
                continue
            _archive_partial_training(args, arm, mode)
            output = args.results_root / "training" / arm / mode
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
                "--model-revision", STUDENT_REVISION,
                "--expected-source-commit", args.expected_source_commit,
                "--expected-train-sha256",
                hashes[f"arms/{arm}/train.jsonl"],
                "--expected-validation-sha256", validation_sha,
                "--local-files-only",
            ]
            env = base.base_env(offline=True, gpu=gpu)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process, handle = spawn_logged(
                command,
                env=env,
                log_path=args.results_root
                / "logs"
                / f"train-{mode}-{arm}-gpu{gpu}.log",
            )
            workers.append((process, handle, f"{mode}-{arm}"))
        wait_processes(workers)
        missing = [
            arm
            for arm in protocol.TRAINED_ARMS
            if not training_complete(args, arm, mode)
        ]
        if missing:
            raise ScreenStageError(f"incomplete {mode} training: {missing}")
    phase_hardware_receipt(
        args,
        "train_complete",
        bound_paths=[
            path
            for arm in protocol.TRAINED_ARMS
            for path in (
                args.results_root
                / "training"
                / arm
                / "formal"
                / "run_manifest.json",
                args.results_root
                / "training"
                / arm
                / "formal"
                / "checkpoint_final"
                / "adapter_model.safetensors",
                args.results_root
                / "training"
                / arm
                / "formal"
                / "checkpoint_final"
                / "adapter_config.json",
            )
        ],
    )


def registry_complete(args: argparse.Namespace) -> bool:
    path = args.results_root / "checkpoint_registry.json"
    try:
        evaluator.load_checkpoint_registry(
            path, expected_profile=protocol.REGISTRY_PROFILE
        )
        return True
    except (OSError, TypeError, ValueError, RuntimeError):
        return False


def registry(args: argparse.Namespace) -> None:
    phase_hardware_receipt(
        args,
        "registry_entry",
        bound_paths=[
            args.results_root
            / "training"
            / arm
            / "formal"
            / "run_manifest.json"
            for arm in protocol.TRAINED_ARMS
        ],
    )
    if registry_complete(args):
        phase_hardware_receipt(
            args,
            "registry_complete",
            bound_paths=[args.results_root / "checkpoint_registry.json"],
        )
        return
    output = args.results_root / "checkpoint_registry.json"
    if output.exists():
        raise ScreenStageError(f"invalid registry must be archived: {output}")
    command = [
        str(args.train_python),
        "scripts/build_v5_checkpoint_registry.py",
        "--provenance-profile", protocol.REGISTRY_PROFILE,
        "--source-commit", args.expected_source_commit,
        "--base-revision", STUDENT_REVISION,
    ]
    for arm in protocol.TRAINED_ARMS:
        command += [
            "--arm",
            f"{arm}={args.results_root / 'training' / arm / 'formal'}",
        ]
    command += ["--output", str(output)]
    process, handle = spawn_logged(
        command,
        env=base.base_env(offline=True),
        log_path=args.results_root / "logs/build-screen-registry.log",
    )
    wait_processes([(process, handle, "registry")])
    if not registry_complete(args):
        raise ScreenStageError("screen checkpoint registry failed validation")
    phase_hardware_receipt(
        args,
        "registry_complete",
        bound_paths=[args.results_root / "checkpoint_registry.json"],
    )


def evaluation_service_specifications(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Return the frozen shard-parallel, request-selected LoRA topology."""

    return [
        {
            "gpu": 0,
            "port": 8001,
            "model": USER_MODEL,
            "revision": USER_REVISION,
            "dtype": "float16",
            "quantization": "awq",
            "alias": protocol.USER_JUDGE_MODEL_ID.removeprefix("openai/"),
            "api_key": "screen-user-judge-local",
            "adapters": False,
        },
        *[
            {
                "gpu": gpu,
                "port": 8100 + gpu,
                "model": STUDENT_MODEL,
                "revision": STUDENT_REVISION,
                "dtype": "bfloat16",
                "quantization": None,
                "alias": protocol.MODEL_IDS["base_model"].removeprefix(
                    "openai/"
                ),
                "api_key": "screen-agent-local",
                "adapters": True,
            }
            for gpu in (1, 2, 3)
        ],
    ]


def _evaluation_service_command(
    args: argparse.Namespace, specification: dict[str, Any]
) -> list[str]:
    """Build one role-specific evaluation service command."""

    command = [
        *base.vllm_base_command(
            args,
            specification["model"],
            specification["revision"],
            dtype=specification["dtype"],
        ),
        "--tokenizer-revision", specification["revision"],
        "--served-model-name", specification["alias"],
        "--port", str(specification["port"]),
        "--api-key", specification["api_key"],
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1",
    ]
    if specification["quantization"] is not None:
        command += ["--quantization", specification["quantization"]]
    if specification["adapters"]:
        loras = [
            (
                f"{protocol.MODEL_IDS[arm].removeprefix('openai/')}="
                f"{args.results_root / 'training' / arm / 'formal' / 'checkpoint_final'}"
            )
            for arm in protocol.TRAINED_ARMS
        ]
        command += [
            "--enable-lora",
            "--max-lora-rank", "16",
            "--max-loras", "1",
            "--max-cpu-loras", "4",
            "--lora-modules", *loras,
        ]
    return command


def _smoke_evaluation_alias(
    *, port: int, api_key: str, model_alias: str
) -> dict[str, Any]:
    """Force one token through a base or LoRA alias before formal evaluation."""

    payload = base.http_post_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        api_key,
        {
            "model": model_alias,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with one short token.",
                }
            ],
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("model") != model_alias
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") not in {"stop", "length"}
    ):
        raise ScreenStageError(
            f"evaluation alias smoke failed for {model_alias!r} on port {port}"
        )
    return payload


def _start_evaluation_services(
    args: argparse.Namespace,
) -> list[tuple[subprocess.Popen[bytes], Any, Path]]:
    services: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    specifications = evaluation_service_specifications(args)
    try:
        for row in specifications:
            command = _evaluation_service_command(args, row)
            expected = {row["alias"]}
            if row["adapters"]:
                expected.update(
                    protocol.MODEL_IDS[arm].removeprefix("openai/")
                    for arm in protocol.TRAINED_ARMS
                )
            process, handle = base.spawn_logged(
                command,
                env=base.base_env(offline=True, gpu=row["gpu"]),
                log_path=args.results_root
                / "logs"
                / f"eval-server-gpu{row['gpu']}.log",
            )
            _register_process(process)
            pid_path = (
                args.results_root / "pids" / f"eval-server-gpu{row['gpu']}.pid"
            )
            base.write_pid(pid_path, process)
            services.append((process, handle, pid_path))
            base.wait_for_service(
                port=row["port"],
                api_key=row["api_key"],
                expected_models=expected,
                process=process,
                timeout_seconds=args.health_timeout,
            )
            for alias in sorted(expected):
                _smoke_evaluation_alias(
                    port=row["port"],
                    api_key=row["api_key"],
                    model_alias=alias,
                )
        return services
    except Exception:
        stop_registered_services(services)
        raise


def eval_contract_path(
    args: argparse.Namespace, arm: str, shard: int
) -> Path:
    return (
        args.results_root
        / "evaluation"
        / arm
        / f"run_contract.shard-{shard:03d}-of-{EVALUATION_SHARDS:03d}.json"
    )


def _result_condition(name: str) -> str:
    for condition in protocol.EVALUATION_CONDITIONS:
        if f"_{condition}." in name:
            return condition
    raise ScreenStageError(f"cannot infer result condition: {name}")


def _result_domain(name: str) -> str:
    domain = name.split("_", 1)[0]
    if domain not in {"retail", "airline"}:
        raise ScreenStageError(f"cannot infer result domain: {name}")
    return domain


def evaluation_shard_evidence(
    args: argparse.Namespace, arm: str, shard: int
) -> dict[str, Any]:
    """Raw-recompute one immutable shard before treating it as resumable."""

    if arm not in protocol.ALL_EVAL_ARMS:
        raise ScreenStageError(f"unknown evaluation arm: {arm}")
    if shard not in range(EVALUATION_SHARDS):
        raise ScreenStageError(f"invalid evaluation shard: {shard}")
    registry_path = args.results_root / "checkpoint_registry.json"
    registry = evaluator.load_checkpoint_registry(
        registry_path, expected_profile=protocol.REGISTRY_PROFILE
    )
    validation_path = screen_protocol_files(args)["validation"]
    validation = read_json(validation_path)
    rows = validation.get("rows")
    if not isinstance(rows, list) or len(rows) != 21:
        raise ScreenStageError("screen validation manifest must have 21 rows")
    expected_rows = evaluator.shard_rows(
        rows, shard_index=shard, num_shards=EVALUATION_SHARDS
    )
    expected_row_by_identity = {
        f"{row['domain']}:{row['task_id']}": row for row in expected_rows
    }
    expected_tasks = [
        f"{row['domain']}:{row['task_id']}" for row in expected_rows
    ]
    contract_path = eval_contract_path(args, arm, shard)
    contract = read_json(contract_path)
    evaluator.validate_contract_core(contract)
    user = contract.get("user")
    judge = contract.get("judge")
    if (
        contract.get("status") != "COMPLETE"
        or contract.get("arm") != arm
        or contract.get("shard_index") != shard
        or contract.get("num_shards") != EVALUATION_SHARDS
        or contract.get("conditions") != ["clean", "error"]
        or contract.get("task_ids") != expected_tasks
        or contract.get("checkpoint_registry_provenance_profile")
        != protocol.REGISTRY_PROFILE
        or contract.get("checkpoint_registry_sha256")
        != sha256_file(registry_path)
        or contract.get("checkpoint_entry") != registry["entries"][arm]
        or contract.get("evaluation_manifest_sha256")
        != sha256_file(validation_path)
        or contract.get("split_manifest_sha256") != sha256_file(SPLIT)
        or contract.get("decoding")
        != evaluator.V5_3_12H_FROZEN_DECODING
        or contract.get("official_test_used") is not False
        or contract.get("agent", {}).get("model")
        != protocol.MODEL_IDS[arm]
        or not isinstance(user, dict)
        or user.get("model") != protocol.USER_JUDGE_MODEL_ID
        or user.get("revision") != protocol.USER_JUDGE_REVISION
        or not isinstance(judge, dict)
        or judge.get("model") != protocol.USER_JUDGE_MODEL_ID
        or judge.get("revision") != protocol.USER_JUDGE_REVISION
        or judge.get("api_base") != user.get("api_base")
    ):
        raise ScreenStageError(
            f"{arm}/shard-{shard}: evaluation contract metadata drift"
        )

    expected_files = evaluator._expected_result_task_ids(contract)
    declared = contract.get("result_sha256")
    if not isinstance(declared, dict) or set(declared) != set(expected_files):
        raise ScreenStageError(
            f"{arm}/shard-{shard}: result declaration set drift"
        )
    result_paths: list[Path] = []
    recomputed: dict[str, Any] = {}
    observed_case_keys: set[tuple[str, str, str]] = set()
    for name, task_ids in expected_files.items():
        if Path(name).name != name:
            raise ScreenStageError("evaluation result path traversal")
        path = contract_path.parent / name
        if not path.is_file() or sha256_file(path) != declared[name]:
            raise ScreenStageError(
                f"{arm}/shard-{shard}: raw result hash drift: {name}"
            )
        recomputed[name] = evaluator.audit_result_interface(
            path,
            expected_task_ids=task_ids,
            num_trials=1,
            max_tokens=512,
        )
        condition = _result_condition(name)
        domain = _result_domain(name)
        payload = read_json(path)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise ScreenStageError(f"{name}: simulations missing")
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise ScreenStageError(f"{name}: invalid simulation")
            if (
                simulation.get("trial") != 0
                or simulation.get("seed") != protocol.TRIAL_SEEDS[0]
            ):
                raise ScreenStageError(
                    f"{name}: evaluation trial/derived seed drift"
                )
            if simulation.get("termination_reason") in {
                "infrastructure_error",
                "unexpected_error",
                "context_window_exceeded",
            }:
                raise ScreenStageError(
                    f"{name}: infrastructure termination is not a result"
                )
            identity = f"{domain}:{simulation.get('task_id')}"
            expected_row = expected_row_by_identity.get(identity)
            if expected_row is None:
                raise ScreenStageError(f"{name}: unexpected task {identity}")
            expected_fault = expected_row["error_condition"]
            if condition == "clean":
                causal_summary.verify_clean_has_no_injection(
                    simulation, expected_fault
                )
            else:
                causal_summary.analyze_error_run(
                    simulation, expected_fault
                )
            key = (domain, str(simulation.get("task_id")), condition)
            if key in observed_case_keys:
                raise ScreenStageError(
                    f"{arm}/shard-{shard}: duplicate raw case {key}"
                )
            observed_case_keys.add(key)
        result_paths.append(path.resolve())
    expected_case_keys = {
        (*identity.split(":", 1), condition)
        for identity in expected_tasks
        for condition in protocol.EVALUATION_CONDITIONS
    }
    if observed_case_keys != expected_case_keys:
        raise ScreenStageError(
            f"{arm}/shard-{shard}: exact raw case ledger drift"
        )
    strict = judge_contract.validate_strict_judge_evidence(
        result_paths, maximum_content_attempts=2
    )
    if (
        isinstance(strict.get("expected_calls"), bool)
        or not isinstance(strict.get("expected_calls"), int)
        or strict["expected_calls"] <= 0
        or strict.get("observed_unique_pass_audits")
        != strict["expected_calls"]
        or contract.get("strict_judge_audit_evidence") != strict
    ):
        raise ScreenStageError(
            f"{arm}/shard-{shard}: strict-judge evidence drift"
        )
    completion = contract.get("completion_audit")
    if (
        not isinstance(completion, dict)
        or completion.get("status") != "PASS"
        or completion.get("result_files")
        != dict(sorted(recomputed.items()))
    ):
        raise ScreenStageError(
            f"{arm}/shard-{shard}: completion audit drift"
        )
    return {
        "contract_sha256": sha256_file(contract_path),
        "result_sha256": dict(sorted(declared.items())),
        "strict_judge_mapping_sha256": strict[
            "canonical_mapping_sha256"
        ],
        "task_ids": expected_tasks,
        "case_count": len(observed_case_keys),
    }


def evaluation_shard_complete(
    args: argparse.Namespace, arm: str, shard: int
) -> bool:
    try:
        evaluation_shard_evidence(args, arm, shard)
        return True
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        judge_contract.StrictJudgeEvidenceError,
    ):
        return False


def evaluation_arm_complete(args: argparse.Namespace, arm: str) -> bool:
    try:
        evidence = [
            evaluation_shard_evidence(args, arm, shard)
            for shard in range(EVALUATION_SHARDS)
        ]
        tasks = [
            task_id
            for item in evidence
            for task_id in item["task_ids"]
        ]
        expected = [
            f"{row['domain']}:{row['task_id']}"
            for row in read_json(
                screen_protocol_files(args)["validation"]
            )["rows"]
        ]
        return (
            len(tasks) == len(set(tasks)) == 21
            and set(tasks) == set(expected)
            and sum(item["case_count"] for item in evidence) == 42
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        judge_contract.StrictJudgeEvidenceError,
    ):
        return False


def _archive_partial_eval(
    args: argparse.Namespace, arm: str, shard: int
) -> None:
    contract = eval_contract_path(args, arm, shard)
    if not contract.exists():
        return
    try:
        if read_json(contract).get("status") == "COMPLETE":
            return
    except ScreenStageError:
        pass
    root = contract.parent
    archive = (
        args.runtime_root
        / "failed_attempts"
        / f"eval-{arm}-shard-{shard}-{time.time_ns()}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in [contract, *root.glob(f"*.shard-{shard:03d}-of-003.json")]:
        if path.exists():
            shutil.move(str(path), archive / path.name)


def evaluation_command(
    args: argparse.Namespace, arm: str, shard: int
) -> list[str]:
    files = screen_protocol_files(args)
    command = [
        str(args.serve_python),
        "scripts/run_v5_sft_causal_eval.py",
        "--tau2-root", str(args.tau2_root),
        "--split-manifest", str(SPLIT),
        "--manifest", str(files["validation"]),
        "--dynamic-audit", str(files["validation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
    ]
    for trained in protocol.TRAINED_ARMS:
        command += [
            "--adapter-dir",
            f"{trained}={args.results_root / 'training' / trained / 'formal' / 'checkpoint_final'}",
        ]
    agent_endpoint = f"http://127.0.0.1:{8101 + shard}/v1"
    user_endpoint = "http://127.0.0.1:8001/v1"
    command += [
        "--output-dir", str(args.results_root / "evaluation" / arm),
        "--arm", arm,
        "--agent-model", protocol.MODEL_IDS[arm],
        "--agent-api-base", agent_endpoint,
        "--agent-api-key", "screen-agent-local",
        "--user-model", protocol.USER_JUDGE_MODEL_ID,
        "--user-api-base", user_endpoint,
        "--user-api-key", "screen-user-judge-local",
        "--judge-model", protocol.USER_JUDGE_MODEL_ID,
        "--judge-api-base", user_endpoint,
        "--judge-api-key", "screen-user-judge-local",
        "--condition", "both",
        "--shard-index", str(shard),
        "--num-shards", str(EVALUATION_SHARDS),
        "--max-steps", "60",
        "--timeout", "900",
        "--max-tokens", "512",
        "--seed", str(protocol.BASE_SEED),
        "--num-trials", "1",
    ]
    return command


def pending_evaluation_arms(
    args: argparse.Namespace, arms: tuple[str, ...], shard: int
) -> list[str]:
    return [
        arm
        for arm in arms
        if not evaluation_shard_complete(args, arm, shard)
    ]


def _evaluate_arm_set(
    args: argparse.Namespace, arms: tuple[str, ...]
) -> None:
    if arms not in {
        protocol.CORE_EVAL_ARMS,
        protocol.EXTENSION_EVAL_ARMS,
    }:
        raise ScreenStageError(
            "evaluation arm set must be exactly registered core or extension"
        )
    services = _start_evaluation_services(args)
    workers: list[tuple[subprocess.Popen[bytes], Any, str]] = []
    try:
        for shard in range(EVALUATION_SHARDS):
            pending = pending_evaluation_arms(args, arms, shard)
            for arm in arms:
                if arm not in pending:
                    continue
                _archive_partial_eval(args, arm, shard)
            # One process per GPU loops arms serially to avoid KV-cache races.
            script = (
                "import subprocess,sys\n"
                f"commands={json.dumps([evaluation_command(args, arm, shard) for arm in pending])}\n"
                "for command in commands:\n"
                "    code=subprocess.run(command).returncode\n"
                "    if code:\n"
                "        raise SystemExit(code)\n"
            )
            # Use a small generated Python expression only as orchestration;
            # scientific commands and all values remain explicit above.
            if not pending:
                continue
            process, handle = spawn_logged(
                [str(args.serve_python), "-c", script],
                env=base.base_env(offline=True),
                log_path=args.results_root
                / "logs"
                / f"evaluate-shard-{shard}-{'-'.join(arms)}.log",
            )
            workers.append((process, handle, f"eval-shard-{shard}"))
        wait_processes(workers)
    finally:
        stop_registered_services(services)
    missing = [arm for arm in arms if not evaluation_arm_complete(args, arm)]
    if missing:
        raise ScreenStageError(f"incomplete evaluation arms: {missing}")


def _core_receipt(args: argparse.Namespace) -> dict[str, Any]:
    evidence = {
        arm: {
            str(shard): evaluation_shard_evidence(args, arm, shard)
            for shard in range(EVALUATION_SHARDS)
        }
        for arm in protocol.CORE_EVAL_ARMS
    }
    payload = {
        "protocol": f"{protocol.PROTOCOL}:core_complete_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_evidence": evidence,
        "core_arms": list(protocol.CORE_EVAL_ARMS),
        "evaluation_shards": EVALUATION_SHARDS,
        "exact_raw_case_count": sum(
            item["case_count"]
            for arm in evidence.values()
            for item in arm.values()
        ),
        "expected_raw_case_count": protocol.CORE_ROLLOUTS,
        "checkpoint_registry_sha256": sha256_file(
            args.results_root / "checkpoint_registry.json"
        ),
        "evaluation_manifest_sha256": sha256_file(
            screen_protocol_files(args)["validation"]
        ),
        "deadline_receipt_sha256": sha256_file(deadline_path(args)),
        "static_protocol_artifact_sha256": screen_static_artifact_hashes(),
        "phase_hardware_receipt_sha256": {
            phase: sha256_file(
                args.runtime_root / "phase_hardware" / f"{phase}.json"
            )
            for phase in (
                "train_entry",
                "train_complete",
                "registry_entry",
                "registry_complete",
                "evaluate_entry",
            )
        },
        "metric_values_read": False,
        "official_test_used": False,
    }
    if payload["exact_raw_case_count"] != protocol.CORE_ROLLOUTS:
        raise ScreenStageError("core receipt exact raw case count drift")
    payload["canonical_sha256"] = protocol.canonical_sha256(payload)
    return payload


def evaluate(args: argparse.Namespace) -> None:
    phase_hardware_receipt(
        args,
        "evaluate_entry",
        bound_paths=[args.results_root / "checkpoint_registry.json"],
    )
    _evaluate_arm_set(args, protocol.CORE_EVAL_ARMS)
    core_path = args.runtime_root / "core_complete_receipt.json"
    if not core_path.exists():
        write_exclusive(core_path, _core_receipt(args))
    core = read_json(core_path)
    body = {k: v for k, v in core.items() if k != "canonical_sha256"}
    if core.get("canonical_sha256") != protocol.canonical_sha256(body):
        raise ScreenStageError("core completion receipt hash drift")
    phase_hardware_receipt(
        args,
        "core_evaluation_complete",
        bound_paths=[core_path],
    )
    admission_path = args.runtime_root / "extension_admission.json"
    if not admission_path.exists():
        admission = protocol.extension_admission(
            read_json(deadline_path(args)),
            core_completed_at=datetime.fromisoformat(
                core["completed_at_utc"]
            ),
            core_complete=all(
                evaluation_arm_complete(args, arm)
                for arm in protocol.CORE_EVAL_ARMS
            ),
        )
        admission["core_complete_receipt_sha256"] = sha256_file(core_path)
        admission["deadline_receipt_sha256"] = sha256_file(
            deadline_path(args)
        )
        admission["canonical_sha256"] = protocol.canonical_sha256(admission)
        write_exclusive(admission_path, admission)
    admission = read_json(admission_path)
    admission_core = {
        key: value
        for key, value in admission.items()
        if key != "canonical_sha256"
    }
    expected_admission = protocol.extension_admission(
        read_json(deadline_path(args)),
        core_completed_at=datetime.fromisoformat(core["completed_at_utc"]),
        core_complete=all(
            evaluation_arm_complete(args, arm)
            for arm in protocol.CORE_EVAL_ARMS
        ),
    )
    expected_admission["core_complete_receipt_sha256"] = sha256_file(
        core_path
    )
    expected_admission["deadline_receipt_sha256"] = sha256_file(
        deadline_path(args)
    )
    expected_admission["canonical_sha256"] = protocol.canonical_sha256(
        expected_admission
    )
    if (
        admission != expected_admission
        or admission.get("uses_metric_values") is not False
        or admission.get("core_complete_receipt_sha256")
        != sha256_file(core_path)
        or admission.get("deadline_receipt_sha256")
        != sha256_file(deadline_path(args))
        or admission.get("canonical_sha256")
        != protocol.canonical_sha256(admission_core)
    ):
        raise ScreenStageError("extension admission receipt drift")
    phase_hardware_receipt(
        args,
        "summarize_entry",
        bound_paths=[core_path, admission_path],
    )
    _run_screen_summary(
        args, args.results_root / "core_screen_summary.json"
    )
    if admission.get("extension_authorized") is True:
        _evaluate_arm_set(args, protocol.EXTENSION_EVAL_ARMS)


def _run_screen_summary(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    command = [
        str(args.train_python),
        "scripts/summarize_v5_3_12h_screen.py",
        "--split-manifest", str(SPLIT),
        "--evaluation-manifest",
        str(screen_protocol_files(args)["validation"]),
        "--dynamic-audit",
        str(screen_protocol_files(args)["validation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
        "--deadline-receipt", str(deadline_path(args)),
        "--core-complete-receipt",
        str(args.runtime_root / "core_complete_receipt.json"),
        "--extension-admission",
        str(args.runtime_root / "extension_admission.json"),
        "--results-root", str(args.results_root),
        "--output", str(output),
    ]
    process, handle = spawn_logged(
        command,
        env=base.base_env(offline=True),
        log_path=args.results_root / "logs/summarize-screen.log",
    )
    wait_processes([(process, handle, "summarize")])
    summary = read_json(output)
    if summary.get("status") == "INCOMPLETE_NO_CLAIM":
        write_terminal(
            args,
            status="INCOMPLETE_NO_CLAIM",
            reason="CORE_EVALUATION_OR_RAW_BINDING_INCOMPLETE",
            no_claim=True,
        )
        raise ScreenStageError("screen summary is incomplete")
    return summary


def summarize(args: argparse.Namespace) -> None:
    phase_hardware_receipt(
        args,
        "summarize_entry",
        bound_paths=[
            args.runtime_root / "core_complete_receipt.json",
            args.runtime_root / "extension_admission.json",
        ],
    )
    output = args.results_root / "screen_summary.json"
    summary = _run_screen_summary(args, output)
    directional = (summary.get("directional_gate") or {}).get("status")
    terminal = (
        "COMPLETE_DIRECTIONAL_POSITIVE_SCREEN"
        if directional == "DIRECTIONAL_POSITIVE_SCREEN"
        else "COMPLETE_EXPLORATORY_INCONCLUSIVE"
    )
    write_terminal(
        args,
        status=terminal,
        reason="RAW_BOUND_CORE_SCREEN_COMPLETE",
        no_claim=False,
    )


def status(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "protocol": protocol.PROTOCOL,
        "deadline_receipt_exists": deadline_path(args).is_file(),
        "remaining_seconds": (
            max(0.0, remaining_seconds(read_json(deadline_path(args))))
            if deadline_path(args).is_file()
            else None
        ),
        "preflight_complete": base.preflight_complete(args),
        "protocol_complete": all(
            path.is_file() for path in screen_protocol_files(args).values()
        ),
        "generation_complete": generation_complete(args),
        "data_complete": data_complete(args),
        "training": {
            mode: {
                arm: training_complete(args, arm, mode)
                for arm in protocol.TRAINED_ARMS
            }
            for mode in ("smoke", "formal")
        },
        "registry_complete": registry_complete(args),
        "core_evaluation": {
            arm: evaluation_arm_complete(args, arm)
            for arm in protocol.CORE_EVAL_ARMS
        },
        "extension_evaluation": {
            arm: evaluation_arm_complete(args, arm)
            for arm in protocol.EXTENSION_EVAL_ARMS
        },
        "summary_exists": (
            args.results_root / "screen_summary.json"
        ).is_file(),
        "terminal": (
            read_json(terminal_path(args))
            if terminal_path(args).is_file()
            else None
        ),
        "official_test_used": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "self-test",
            "prefetch",
            "protocol",
            "generate",
            "prepare",
            "train",
            "registry",
            "evaluate",
            "summarize",
            "status",
            "stop-services",
            "all",
        ),
    )
    parser.add_argument("--expected-source-commit")
    parser.add_argument(
        "--tau2-root", type=Path, default=ROOT / "data/raw/tau2-bench"
    )
    parser.add_argument(
        "--serve-venv", type=Path, default=Path("/workspace/venvs/v5-2-serve")
    )
    parser.add_argument(
        "--train-venv", type=Path, default=Path("/workspace/venvs/v5-2-train")
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_12h_screen_protocol",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/v5_3_12h_screen_generation",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_12h_screen",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/v5_3_12h_screen",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "artifacts/v5_3_12h_screen",
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    args = parser.parse_args()
    for field in (
        "tau2_root",
        "serve_venv",
        "train_venv",
        "workspace_root",
        "protocol_root",
        "raw_root",
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
    args = parse_args()
    configure_screen_hardware()
    if args.stage == "stop-services":
        try:
            emergency_stop(args)
        except (ScreenStageError, base.StageError) as error:
            print(f"emergency stop needs attention: {error}", file=sys.stderr)
            raise SystemExit(21) from error
        return
    if not args.expected_source_commit:
        raise SystemExit(
            "--expected-source-commit is required except for stop-services"
        )
    verify_execution_commit(args.expected_source_commit)
    validate_isolation(args)
    if args.stage == "status":
        status(args)
        return
    acquire_controller_lock(args)
    receipt = load_or_create_deadline(
        args, allow_create=args.stage in {"all", "preflight"}
    )
    arm_watchdog(args, receipt)
    stages: dict[str, Callable[[argparse.Namespace], Any]] = {
        "preflight": preflight,
        "self-test": base.self_test,
        "prefetch": base.prefetch_models,
        "protocol": build_protocol,
        "generate": generate,
        "prepare": prepare,
        "train": train,
        "registry": registry,
        "evaluate": evaluate,
        "summarize": summarize,
        "status": status,
        "stop-services": base.stop_stale_services,
    }
    try:
        if args.stage == "all":
            for function in (
                preflight,
                base.self_test,
                base.prefetch_models,
                build_protocol,
                generate,
                prepare,
                train,
                registry,
                evaluate,
                summarize,
            ):
                guard_deadline()
                function(args)
            status(args)
        else:
            stages[args.stage](args)
    except DeadlineExpired:
        raise SystemExit(124)
    except (
        ScreenStageError,
        screen_data.ScreenDataError,
        base.StageError,
        subprocess.CalledProcessError,
    ) as error:
        if not terminal_path(args).is_file():
            write_terminal(
                args,
                status="INCOMPLETE_NO_CLAIM",
                reason=f"{type(error).__name__}: {error}",
                no_claim=True,
            )
        print(f"V5.3-12h screen stopped: {error}", file=sys.stderr)
        raise SystemExit(20) from error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        release_controller_lock()


if __name__ == "__main__":
    main()
