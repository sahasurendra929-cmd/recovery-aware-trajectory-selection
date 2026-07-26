from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_3_12h_supervisor.py"
SPEC = importlib.util.spec_from_file_location(
    "run_v5_3_12h_supervisor_tests",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return None


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class V5312HourSupervisorTests(unittest.TestCase):
    def test_supervisor_accepts_exact_controller_complete_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TERMINAL_STATUS.json"
            for status in sorted(MODULE.FINAL_CONTROLLER_STATUSES):
                MODULE.atomic_write(
                    path,
                    {
                        "protocol": (
                            f"{MODULE.protocol.PROTOCOL}:terminal_v1"
                        ),
                        "status": status,
                        "no_scientific_claim": False,
                        "official_test_used": False,
                        "official_test_sealed": True,
                    },
                )
                self.assertTrue(
                    MODULE._controller_completed_by_terminal(path)
                )
            payload = MODULE.read_json(path)
            payload["status"] = "DIRECTIONAL_POSITIVE_SCREEN"
            MODULE.atomic_write(path, payload)
            self.assertFalse(MODULE._controller_completed_by_terminal(path))

    def test_restart_audit_never_resets_authoritative_t0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_now = datetime(2026, 7, 26, 1, tzinfo=timezone.utc)
            deadline_a, ledger_a = MODULE.load_or_create_authoritative_deadline(
                root,
                now=first_now,
            )
            deadline_bytes = MODULE.deadline_path(root).read_bytes()
            deadline_b, ledger_b = MODULE.load_or_create_authoritative_deadline(
                root,
                now=first_now + timedelta(hours=3),
            )
            self.assertEqual(deadline_b, deadline_a)
            self.assertEqual(
                MODULE.deadline_path(root).read_bytes(),
                deadline_bytes,
            )
            self.assertEqual(ledger_b["started_at_utc"], ledger_a["started_at_utc"])
            self.assertEqual(ledger_b["deadline_utc"], ledger_a["deadline_utc"])
            self.assertTrue(ledger_b["resume_does_not_reset_t0"])

    def test_registered_group_can_never_be_current_shell_group(self):
        ledger = {
            "boot_id": MODULE.current_boot_id(),
            "controller": {
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
            },
            "watchdog": {"pid": 999_998, "pgid": 999_998},
            "launcher": {"pid": os.getpid(), "pgid": os.getpgrp()},
        }
        with mock.patch.object(MODULE.os, "killpg") as killpg:
            with self.assertRaises(MODULE.UnsafeProcessGroup):
                MODULE.terminate_registered_group(
                    ledger,
                    grace_seconds=0,
                    actor_pid=os.getpid(),
                    actor_pgid=os.getpgrp(),
                )
            killpg.assert_not_called()

    def test_boot_id_mismatch_is_action_required_without_pid_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.load_or_create_authoritative_deadline(root)
            ledger = MODULE.read_json(MODULE.ledger_path(root))
            ledger["boot_id"] = "different-boot"
            ledger["canonical_sha256"] = MODULE.canonical_sha256(
                MODULE._ledger_core(ledger)
            )
            MODULE.atomic_write(MODULE.ledger_path(root), ledger)
            with mock.patch.object(MODULE.os, "killpg") as killpg:
                result = MODULE.audit(root)
            self.assertEqual(result["status"], "ACTION_REQUIRED")
            self.assertFalse(result["boot_id_matches"])
            killpg.assert_not_called()

    def test_controller_death_kills_surviving_grandchild_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grandchild_pid_file = root / "grandchild.pid"
            grandchild_code = (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "time.sleep(120)"
            )
            controller_code = (
                "import pathlib,subprocess,sys,time;"
                f"p=subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]);"
                f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(p.pid));"
                "time.sleep(120)"
            )
            result = MODULE.start(
                root,
                controller_command=[sys.executable, "-c", controller_code],
                terminal_receipt=None,
                term_grace_seconds=0.1,
            )
            controller_pid = result["controller_pid"]
            watchdog_pid = result["watchdog_pid"]
            self.addCleanup(self._cleanup_pid, watchdog_pid)
            self.addCleanup(self._cleanup_group, result["controller_pgid"])
            self.assertNotEqual(
                result["controller_pgid"],
                result["watchdog_pgid"],
            )
            grandchild_pid = wait_for(
                lambda: (
                    int(grandchild_pid_file.read_text(encoding="utf-8"))
                    if grandchild_pid_file.is_file()
                    else None
                )
            )
            self.assertIsNotNone(grandchild_pid)
            self.assertTrue(process_exists(grandchild_pid))

            os.kill(controller_pid, signal.SIGKILL)
            os.waitpid(controller_pid, 0)

            receipt = wait_for(
                lambda: (
                    json.loads(
                        MODULE.receipt_path(root).read_text(encoding="utf-8")
                    )
                    if MODULE.receipt_path(root).is_file()
                    else None
                ),
                timeout=15,
            )
            watchdog_log = root / MODULE.WATCHDOG_LOG_NAME
            self.assertIsNotNone(
                receipt,
                watchdog_log.read_text(encoding="utf-8")
                if watchdog_log.is_file()
                else "watchdog log absent",
            )
            self.assertEqual(receipt["status"], "ACTION_REQUIRED")
            self.assertEqual(
                receipt["reason"],
                "CONTROLLER_DIED_WITH_LIVE_GROUP",
            )
            self.assertTrue(receipt["action_required"])
            self.assertTrue(receipt["termination"]["sigterm_sent"])
            self.assertTrue(receipt["termination"]["sigkill_sent"])
            self.assertEqual(
                receipt["termination"]["target_pgid"],
                result["controller_pgid"],
            )
            self.assertNotEqual(
                receipt["termination"]["target_pgid"],
                os.getpgrp(),
            )
            self.assertTrue(
                wait_for(lambda: not process_exists(grandchild_pid), timeout=10)
            )
            self.assertTrue(
                wait_for(
                    lambda: self._reap_child_if_exited(watchdog_pid),
                    timeout=10,
                )
            )

    def test_deadline_cutoff_never_downgrades_valid_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "results" / "TERMINAL_STATUS.json"
            deadline = MODULE.protocol.make_deadline_receipt(
                datetime.now(timezone.utc) - timedelta(hours=13)
            )
            MODULE.atomic_write(MODULE.deadline_path(root), deadline)
            ledger = {
                "watchdog": {
                    "pid": os.getpid(),
                    "pgid": os.getpgrp(),
                },
                "controller": {"pid": 999_991, "pgid": 999_991},
            }
            captured = {}

            def capture_receipt(_root, **kwargs):
                captured.update(kwargs)
                return kwargs

            with (
                mock.patch.object(MODULE.fcntl, "flock"),
                mock.patch.object(
                    MODULE,
                    "validate_deadline_and_ledger",
                    return_value=(deadline, ledger),
                ),
                mock.patch.object(
                    MODULE,
                    "validated_core_snapshot",
                    return_value=True,
                ),
                mock.patch.object(
                    MODULE,
                    "terminate_registered_group",
                    return_value={"sigterm_sent": True},
                ),
                mock.patch.object(
                    MODULE,
                    "write_action_receipt",
                    side_effect=capture_receipt,
                ),
            ):
                code = MODULE.watchdog_loop(
                    root,
                    lock_fd=123,
                    term_grace_seconds=0.0,
                    terminal_receipt=terminal,
                )
            self.assertEqual(code, 124)
            persisted = json.loads(terminal.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["status"],
                "CORE_COMPLETE_EXTENSION_INCOMPLETE",
            )
            self.assertFalse(persisted["no_scientific_claim"])
            self.assertEqual(
                captured["status"],
                "CORE_COMPLETE_EXTENSION_INCOMPLETE",
            )
            self.assertIn("CORE_RETAINED", captured["reason"])

    @staticmethod
    def _cleanup_pid(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _cleanup_group(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _reap_child_if_exited(pid):
        try:
            observed, _ = os.waitpid(pid, os.WNOHANG)
            return observed == pid
        except ChildProcessError:
            return True


if __name__ == "__main__":
    unittest.main()
