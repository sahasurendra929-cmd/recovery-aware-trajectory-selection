from __future__ import annotations

import inspect
import json
import signal
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import run_v5_3_low_support_diagnostic as controller
from scripts import run_v5_sft_causal_eval as evaluator
from scripts import v5_3_low_support_protocol as protocol


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class LowSupportControllerTests(unittest.TestCase):
    def test_controller_freezes_source_commit_and_ten_smokes(self) -> None:
        protocol.validate_constants()
        self.assertEqual(
            protocol.SOURCE_GENERATION_COMMIT,
            "f631dd0d7d795daad037e3548e637045ee1425e7",
        )
        self.assertEqual(protocol.RUNTIME_ALIAS_SMOKE_COUNT, 10)
        specifications = controller.service_specifications(
            SimpleNamespace(results_root=Path("/repo/results"))
        )
        self.assertEqual(len(specifications), 4)
        self.assertEqual(
            sum(len(row["expected_models"]) for row in specifications),
            10,
        )
        self.assertEqual(
            {row["api_base"] for row in specifications},
            set(protocol.RUNTIME_ENDPOINTS),
        )
        prepare_source = inspect.getsource(controller.prepare)
        self.assertIn("--source-snapshot-receipt", prepare_source)
        self.assertIn(
            "preparation.SOURCE_SNAPSHOT_RECEIPT",
            prepare_source,
        )
        self.assertIn(
            "signal.signal(signal.SIGHUP, signal_handler)",
            inspect.getsource(controller.main),
        )

    def test_every_spawn_passes_inherited_source_lock_fd(self) -> None:
        args = SimpleNamespace(
            hf_home=Path("/cache"),
            results_root=Path("/results"),
        )
        fake_process = mock.Mock(pid=123)
        fake_process.poll.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "child.log"
            with (
                mock.patch.object(controller, "source_lock_fd", return_value=17),
                mock.patch.object(
                    controller.subprocess,
                    "Popen",
                    return_value=fake_process,
                ) as popen,
                mock.patch.object(controller, "register_process"),
            ):
                _process, handle = controller.spawn_logged(
                    args,
                    ["python", "-V"],
                    log_path=log,
                )
                handle.close()
        self.assertEqual(popen.call_args.kwargs["pass_fds"], (17,))
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["V5_3_LOW_SUPPORT_LOCK_FD"], "17")
        self.assertEqual(env["HF_HOME"], "/cache")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")

    def test_terminal_is_atomic_and_preserves_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                results_root=Path(directory),
                expected_processing_commit="a" * 40,
            )
            controller.write_terminal(
                args,
                status="INCOMPLETE_NO_CLAIM",
                reason="FIRST",
                no_claim=True,
            )
            controller.write_terminal(
                args,
                status="INCOMPLETE_NO_CLAIM",
                reason="SECOND",
                no_claim=True,
            )
            terminal = controller.read_json(
                controller.terminal_path(args)
            )
            self.assertEqual(terminal["reason"], "FIRST")
            self.assertTrue(terminal["no_scientific_claim"])
            self.assertEqual(
                terminal["canonical_sha256"],
                protocol.canonical_sha256(
                    {
                        key: value
                        for key, value in terminal.items()
                        if key != "canonical_sha256"
                    }
                ),
            )

    def test_source_completed_claim_cannot_enter_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_protocol = root / "protocol"
            for name in (
                "source_generation_manifest.json",
                "validation_manifest.json",
                "screen_manifest.json",
                "generation_dynamic_audit.json",
                "validation_dynamic_audit.json",
            ):
                write_json(source_protocol / name, {})
            source_results = root / "source-results"
            source_processed = root / "source-processed"
            write_json(
                source_results / "TERMINAL_STATUS.json",
                {
                    "status": "COMPLETE_EXPLORATORY_INCONCLUSIVE",
                    "no_scientific_claim": False,
                    "official_test_used": False,
                },
            )
            write_json(source_processed / "audit.json", {})
            write_json(source_processed / "hashes.json", {})
            args = SimpleNamespace(
                source_protocol_root=source_protocol,
                source_results_root=source_results,
                source_processed_root=source_processed,
                source_raw_root=root / "raw",
                runtime_root=root / "runtime",
            )
            with self.assertRaisesRegex(
                controller.DiagnosticControllerError,
                "exact DATA_GATE_FAIL_NO_TRAIN",
            ):
                controller.validate_source_admission(args)

    def test_source_terminal_status_and_reason_are_exact(self) -> None:
        valid = {
            "protocol": (
                f"{controller.source_protocol.PROTOCOL}:terminal_v1"
            ),
            "status": "DATA_GATE_FAIL_NO_TRAIN",
            "reason": "DO_NOT_TRAIN",
            "no_scientific_claim": True,
            "official_test_used": False,
            "official_test_sealed": True,
            "processing_source_commit": "a" * 40,
            "source_generation_commit": protocol.SOURCE_GENERATION_COMMIT,
            "source_audit_sha256": "b" * 64,
            "source_hashes_sha256": "c" * 64,
            "source_snapshot_receipt_sha256": (
                controller.sha256_file(
                    controller.preparation.SOURCE_SNAPSHOT_RECEIPT
                )
            ),
            "source_snapshot_file_ledger_sha256": (
                controller.preparation.source_snapshot.canonical_sha256(
                    controller.read_json(
                        controller.preparation.SOURCE_SNAPSHOT_RECEIPT
                    )["file_sha256"]
                )
            ),
        }
        controller.validate_source_terminal(valid)
        for drift in (
            {
                **valid,
                "status": "INCOMPLETE_NO_CLAIM",
            },
            {
                **valid,
                "reason": "OTHER",
            },
            {
                **valid,
                "no_scientific_claim": False,
            },
            {
                **valid,
                "official_test_used": True,
            },
            {
                **valid,
                "source_generation_commit": "d" * 40,
            },
            {
                **valid,
                "source_audit_sha256": None,
            },
        ):
            with self.assertRaisesRegex(
                controller.DiagnosticControllerError,
                "exact DATA_GATE_FAIL_NO_TRAIN",
            ):
                controller.validate_source_terminal(drift)

    def test_pair_ledger_must_match_source_recomputation(self) -> None:
        source = {
            "task_yield": {
                "retail:10": {
                    "selected_pairs": 1,
                    "selected_pair_ids": ["pair-1"],
                    "selected_paired_attempt_indices": [2],
                    "selected_paired_attempt_seeds": ["725284"],
                }
            },
            "data_gate": {"status": "FAIL_CLOSED"},
            "source_snapshot": {"receipt_sha256": "a" * 64},
        }
        diagnostic = {
            "task_yield": source["task_yield"],
            "data_gate": {"source_strict_gate": source["data_gate"]},
            "source_snapshot": source["source_snapshot"],
        }
        controller.validate_pair_ledger_equivalence(source, diagnostic)
        diagnostic["task_yield"] = {
            "retail:10": {
                **source["task_yield"]["retail:10"],
                "selected_pair_ids": ["different"],
            }
        }
        with self.assertRaisesRegex(
            controller.DiagnosticControllerError,
            "pair ledger differs",
        ):
            controller.validate_pair_ledger_equivalence(source, diagnostic)

    def test_evaluation_command_binds_runtime_evidence_and_shard_port(self) -> None:
        args = SimpleNamespace(
            serve_python=Path("/venv/bin/python"),
            tau2_root=Path("/repo/tau2"),
            source_protocol_root=Path("/repo/source-protocol"),
            results_root=Path("/repo/results/v5_3_low_support_diagnostic"),
        )
        command = controller.evaluation_command(
            args, "repair_50", shard=2
        )
        rendered = " ".join(map(str, command))
        self.assertIn("--runtime-service-evidence", command)
        self.assertIn("--provenance-profile", command)
        self.assertIn(protocol.REGISTRY_PROFILE, command)
        self.assertIn("http://127.0.0.1:8103/v1", command)
        self.assertIn(protocol.USER_JUDGE_MODEL_ID, command)
        self.assertIn("--num-shards 3", rendered)

    def test_runtime_evidence_rejects_missing_or_extra_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = (root / "results").resolve()
            runtime = (root / "runtime").resolve()
            path = results / protocol.RUNTIME_SERVICE_EVIDENCE_NAME
            registry_path = results / "checkpoint_registry.json"
            source_admission_path = runtime / "source_admission_receipt.json"
            write_json(source_admission_path, {"status": "PASS"})
            registry = {
                "source_commit": "a" * 40,
                "entries": {
                    arm: {"model_id": protocol.MODEL_IDS[arm]}
                    for arm in protocol.EVAL_ARMS
                }
            }
            write_json(registry_path, registry)
            service_args = SimpleNamespace(
                results_root=results,
                vllm=Path("/venv/bin/vllm"),
            )
            services = []
            for specification in controller.service_specifications(
                service_args
            ):
                command = controller.service_command(
                    service_args, specification
                )
                models_response = {
                    "object": "list",
                    "data": [
                        {"id": alias, "object": "model"}
                        for alias in specification["expected_models"]
                    ],
                }
                identity = {
                    "protocol": protocol.PROCESS_IDENTITY_PROTOCOL,
                    "pid": 2000 + specification["gpu"],
                    "process_group_id": 2000 + specification["gpu"],
                    "start_time_ticks": 10000 + specification["gpu"],
                    "boot_id": "test-boot-id",
                    "cmdline_sha256": "c" * 64,
                }
                pid_receipt_path = (
                    results
                    / "pids"
                    / f"eval-server-gpu{specification['gpu']}.json"
                )
                pid_receipt = {
                    "protocol": protocol.SERVICE_PID_RECEIPT_PROTOCOL,
                    "service_role": specification["role"],
                    "gpu": specification["gpu"],
                    "api_base": specification["api_base"],
                    "command_sha256": protocol.canonical_sha256(command),
                    "process_identity": identity,
                }
                pid_receipt["canonical_sha256"] = (
                    protocol.canonical_sha256(pid_receipt)
                )
                write_json(pid_receipt_path, pid_receipt)
                services.append(
                    {
                        "role": specification["role"],
                        "gpu": specification["gpu"],
                        "api_base": specification["api_base"],
                        "model": specification["model"],
                        "revision": specification["revision"],
                        "tokenizer_revision": specification["revision"],
                        "dtype": specification["dtype"],
                        "quantization": specification["quantization"],
                        "max_model_len": 32768,
                        "max_num_seqs": 1,
                        "gpu_memory_utilization": 0.90,
                        "expected_models": list(
                            specification["expected_models"]
                        ),
                        "observed_models": list(
                            specification["expected_models"]
                        ),
                        "models_exact": True,
                        "models_response": models_response,
                        "models_response_sha256": (
                            protocol.canonical_sha256(models_response)
                        ),
                        "command": command,
                        "command_sha256": protocol.canonical_sha256(
                            command
                        ),
                        "process_identity": identity,
                        "pid_receipt_path": str(
                            pid_receipt_path.resolve()
                        ),
                        "pid_receipt_sha256": evaluator.sha256_file(
                            pid_receipt_path
                        ),
                    }
                )
            smokes = [
                {
                    "status": "PASS",
                    "api_base": endpoint,
                    "model_alias": alias,
                    "max_tokens": 1,
                    "response_model": alias,
                    "finish_reason": "length",
                    "response": {
                        "model": alias,
                        "choices": [{"finish_reason": "length"}],
                    },
                }
                for endpoint, aliases in (
                    protocol.RUNTIME_ENDPOINT_MODEL_ALIASES.items()
                )
                for alias in aliases
            ]
            for smoke in smokes:
                smoke["response_sha256"] = protocol.canonical_sha256(
                    smoke["response"]
                )
            payload = {
                "protocol": protocol.RUNTIME_SERVICE_EVIDENCE_PROTOCOL,
                "status": "PASS",
                "processing_source_commit": "a" * 40,
                "source_generation_commit": (
                    protocol.SOURCE_GENERATION_COMMIT
                ),
                "source_admission_receipt_sha256": (
                    evaluator.sha256_file(source_admission_path)
                ),
                "checkpoint_registry_path": str(registry_path.resolve()),
                "checkpoint_registry_sha256": (
                    evaluator.sha256_file(registry_path)
                ),
                "registry_model_ids": {
                    arm: protocol.MODEL_IDS[arm]
                    for arm in sorted(protocol.EVAL_ARMS)
                },
                "services": services,
                "alias_smokes": smokes,
                "alias_smoke_count": 10,
                "all_model_sets_exact": True,
                "official_test_used": False,
                "official_test_sealed": True,
            }
            payload["canonical_sha256"] = protocol.canonical_sha256(payload)
            write_json(path, payload)
            with mock.patch.object(
                evaluator.low_support,
                "artifact_root",
                side_effect=lambda _root, key: (
                    results if key == "results_root" else runtime
                ),
            ):
                evaluator.load_low_support_runtime_service_evidence(
                    path,
                    checkpoint_registry_path=registry_path,
                    checkpoint_registry=registry,
                    expected_source_commit="a" * 40,
                    require_live_services=False,
                )
                with mock.patch.object(
                    evaluator,
                    "_linux_process_identity",
                    side_effect=lambda pid: next(
                        row["process_identity"]
                        for row in services
                        if row["process_identity"]["pid"] == pid
                    ),
                ):
                    evaluator.load_low_support_runtime_service_evidence(
                        path,
                        checkpoint_registry_path=registry_path,
                        checkpoint_registry=registry,
                        expected_source_commit="a" * 40,
                    )
                with mock.patch.object(
                    evaluator,
                    "_linux_process_identity",
                    side_effect=lambda pid: (
                        next(
                            row["process_identity"]
                            for row in services
                            if row["process_identity"]["pid"] == pid
                        )
                        | {"start_time_ticks": 999999}
                    ),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "same live service"
                    ):
                        evaluator.load_low_support_runtime_service_evidence(
                            path,
                            checkpoint_registry_path=registry_path,
                            checkpoint_registry=registry,
                            expected_source_commit="a" * 40,
                        )
                original_model_id = payload["services"][0][
                    "models_response"
                ]["data"][0]["id"]
                payload["services"][0]["models_response"]["data"][0][
                    "id"
                ] = "tampered"
                payload["canonical_sha256"] = protocol.canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "canonical_sha256"
                    }
                )
                write_json(path, payload)
                with self.assertRaisesRegex(RuntimeError, "raw /models"):
                    evaluator.load_low_support_runtime_service_evidence(
                        path,
                        checkpoint_registry_path=registry_path,
                        checkpoint_registry=registry,
                        expected_source_commit="a" * 40,
                        require_live_services=False,
                    )
                payload["services"][0]["models_response"]["data"][0][
                    "id"
                ] = original_model_id
                payload["alias_smokes"][0]["response"]["choices"][0][
                    "finish_reason"
                ] = "stop"
                payload["canonical_sha256"] = protocol.canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "canonical_sha256"
                    }
                )
                write_json(path, payload)
                with self.assertRaisesRegex(RuntimeError, "smoke"):
                    evaluator.load_low_support_runtime_service_evidence(
                        path,
                        checkpoint_registry_path=registry_path,
                        checkpoint_registry=registry,
                        expected_source_commit="a" * 40,
                        require_live_services=False,
                    )
                payload["alias_smokes"][0]["response"]["choices"][0][
                    "finish_reason"
                ] = "length"
                payload["services"][1]["observed_models"].append("extra")
                payload["canonical_sha256"] = protocol.canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "canonical_sha256"
                    }
                )
                write_json(path, payload)
                with self.assertRaisesRegex(RuntimeError, "model-set"):
                    evaluator.load_low_support_runtime_service_evidence(
                        path,
                        checkpoint_registry_path=registry_path,
                        checkpoint_registry=registry,
                        expected_source_commit="a" * 40,
                        require_live_services=False,
                    )

    def test_cuda_preflight_rejects_stale_compute_process(self) -> None:
        args = SimpleNamespace()
        with mock.patch.object(
            controller,
            "run_capture",
            return_value="GPU-deadbeef, 4321, python, 1024\n",
        ):
            with self.assertRaisesRegex(
                controller.DiagnosticControllerError,
                "stale CUDA compute processes",
            ):
                controller.require_no_cuda_compute_processes(
                    args, barrier="test"
                )
        with mock.patch.object(controller, "run_capture", return_value=""):
            self.assertEqual(
                controller.require_no_cuda_compute_processes(
                    args, barrier="test"
                ),
                [],
            )

    def test_cleanup_signals_group_even_if_leader_already_exited(self) -> None:
        process = mock.Mock(pid=321)
        process.poll.return_value = 0
        identity = {
            "protocol": protocol.PROCESS_IDENTITY_PROTOCOL,
            "pid": 321,
            "process_group_id": 321,
            "start_time_ticks": 9,
            "boot_id": "boot",
            "cmdline_sha256": "a" * 64,
        }
        with (
            mock.patch.object(
                controller,
                "registered_process_identity",
                return_value=identity,
            ),
            mock.patch.object(
                controller,
                "process_group_exists",
                side_effect=[True, True, False, False],
            ),
            mock.patch.object(controller.Path, "exists", return_value=False),
            mock.patch.object(controller.os, "killpg") as killpg,
            mock.patch.object(
                controller.time,
                "monotonic",
                side_effect=[0.0, 0.1],
            ),
            mock.patch.object(controller.time, "sleep"),
        ):
            controller.stop_process(process)
        killpg.assert_called_once_with(321, signal.SIGTERM)

    def test_cleanup_refuses_pid_reuse(self) -> None:
        process = mock.Mock(pid=654)
        identity = {
            "protocol": protocol.PROCESS_IDENTITY_PROTOCOL,
            "pid": 654,
            "process_group_id": 654,
            "start_time_ticks": 9,
            "boot_id": "boot",
            "cmdline_sha256": "a" * 64,
        }
        reused = dict(identity, start_time_ticks=10)
        with (
            mock.patch.object(
                controller,
                "registered_process_identity",
                return_value=identity,
            ),
            mock.patch.object(
                controller, "process_group_exists", return_value=True
            ),
            mock.patch.object(controller.Path, "exists", return_value=True),
            mock.patch.object(
                controller,
                "linux_process_identity",
                return_value=reused,
            ),
            mock.patch.object(controller.os, "killpg") as killpg,
        ):
            with self.assertRaisesRegex(
                controller.DiagnosticControllerError,
                "PID-reused",
            ):
                controller.stop_process(process)
        killpg.assert_not_called()

    def test_service_identity_receipt_rejects_unsafe_process_group(self) -> None:
        valid = {
            "protocol": protocol.PROCESS_IDENTITY_PROTOCOL,
            "pid": 777,
            "process_group_id": 777,
            "start_time_ticks": 123,
            "boot_id": "boot",
            "cmdline_sha256": "a" * 64,
        }
        with mock.patch.object(
            controller.Path, "read_text", return_value="boot\n"
        ):
            self.assertEqual(
                controller.validate_service_process_identity(valid),
                valid,
            )
            for drift in (
                valid | {"pid": 1, "process_group_id": 1},
                valid | {"process_group_id": 778},
                valid | {"boot_id": "old-boot"},
            ):
                with self.assertRaisesRegex(
                    controller.DiagnosticControllerError,
                    "invalid or stale",
                ):
                    controller.validate_service_process_identity(drift)

    def test_low_support_live_alias_checks_are_exact(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        registry = {
            "provenance_profile": protocol.REGISTRY_PROFILE,
            "entries": {
                arm: {"model_id": protocol.MODEL_IDS[arm]}
                for arm in protocol.EVAL_ARMS
            },
        }
        aliases = [
            protocol.MODEL_IDS[arm].removeprefix("openai/")
            for arm in protocol.EVAL_ARMS
        ]
        with mock.patch.object(
            evaluator,
            "urlopen",
            return_value=FakeResponse(
                {
                    "data": [
                        *[{"id": alias} for alias in aliases],
                        {"id": "unexpected"},
                    ]
                }
            ),
        ):
            evaluator.verify_served_registry_aliases(
                "http://127.0.0.1:8101/v1",
                "key",
                registry,
                require_exact=False,
            )
            with self.assertRaisesRegex(RuntimeError, "exactly"):
                evaluator.verify_served_registry_aliases(
                    "http://127.0.0.1:8101/v1",
                    "key",
                    registry,
                    require_exact=True,
                )
        user_alias = protocol.USER_JUDGE_MODEL_ID.removeprefix("openai/")
        with mock.patch.object(
            evaluator,
            "urlopen",
            return_value=FakeResponse(
                {"data": [{"id": user_alias}, {"id": "unexpected"}]}
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly"):
                evaluator.verify_served_model_id(
                    "http://127.0.0.1:8001/v1",
                    "key",
                    protocol.USER_JUDGE_MODEL_ID,
                    require_exact=True,
                )

    def test_evaluation_barrier_revalidates_live_services(self) -> None:
        args = SimpleNamespace(
            results_root=Path("/repo/results/v5_3_low_support_diagnostic"),
            expected_processing_commit="a" * 40,
        )
        binding = {
            "path": "/repo/results/runtime_service_evidence.json",
            "sha256": "1" * 64,
            "canonical_sha256": "2" * 64,
        }
        specifications = controller.service_specifications(args)
        with (
            mock.patch.object(
                evaluator,
                "load_checkpoint_registry",
                return_value={"source_commit": "a" * 40},
            ),
            mock.patch.object(
                evaluator,
                "load_low_support_runtime_service_evidence",
                return_value=({}, binding),
            ) as load_evidence,
            mock.patch.object(
                controller,
                "exact_models",
                side_effect=[
                    (list(row["expected_models"]), {"data": []})
                    for row in specifications
                ],
            ),
        ):
            controller.revalidate_live_runtime_services(args, binding)
        self.assertTrue(
            load_evidence.call_args.kwargs["require_live_services"]
        )


if __name__ == "__main__":
    unittest.main()
