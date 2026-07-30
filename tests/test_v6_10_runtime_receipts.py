from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_v6_10_runtime_receipts as receipts
from scripts import run_v6_candidate_generation as generation


CONTAINER = "sha256:" + "a" * 64
TIMESTAMP = "2026-07-30T12:00:00Z"
FROZEN_PYTHON = Path(os.path.abspath(sys.executable))
FROZEN_PYTHON_SHA256 = receipts.file_sha256(FROZEN_PYTHON)


def fake_snapshot_files(model: str) -> dict:
    marker = "a" if model == generation.V610_TEACHER_MODEL else "b"
    return {
        "byte_files": {"config.json": marker * 64},
        "weight_blob_targets": {
            "model.safetensors": ("c" if marker == "a" else "d") * 64
        },
    }


def gpu_inventory() -> list[dict]:
    return [
        {
            "index": index,
            "model": receipts.FROZEN_GPU_MODEL,
            "uuid": f"GPU-{index}",
            "total_memory_bytes": 32 * 1024**3,
        }
        for index in range(3)
    ]


def service_specs_payload() -> dict:
    teacher_snapshot = (
        "/models/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/"
        + generation.V610_TEACHER_REVISION
    )
    shared_snapshot = (
        "/models/models--Qwen--Qwen2.5-14B-Instruct-AWQ/snapshots/"
        + generation.V610_USER_JUDGE_REVISION
    )
    teacher_command = [
        str(FROZEN_PYTHON),
        "-I",
        "-m",
        receipts.VLLM_SERVER_MODULE,
        "--model",
        teacher_snapshot,
        "--revision",
        generation.V610_TEACHER_REVISION,
        "--served-model-name",
        generation.V610_TEACHER_MODEL,
        "--quantization",
        "awq",
        "--dtype",
        receipts.FROZEN_MODEL_DTYPE,
        "--tensor-parallel-size",
        "2",
        "--load-format",
        "safetensors",
        "--max-model-len",
        str(receipts.FROZEN_MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(receipts.FROZEN_GPU_MEMORY_UTILIZATION),
        "--host",
        "127.0.0.1",
        "--port",
        "8101",
    ]
    shared_command = [
        str(FROZEN_PYTHON),
        "-I",
        "-m",
        receipts.VLLM_SERVER_MODULE,
        "--model",
        shared_snapshot,
        "--revision",
        generation.V610_USER_JUDGE_REVISION,
        "--served-model-name",
        generation.V610_USER_JUDGE_MODEL,
        "--quantization",
        "awq",
        "--dtype",
        receipts.FROZEN_MODEL_DTYPE,
        "--tensor-parallel-size",
        "1",
        "--load-format",
        "safetensors",
        "--max-model-len",
        str(receipts.FROZEN_MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(receipts.FROZEN_GPU_MEMORY_UTILIZATION),
        "--host",
        "127.0.0.1",
        "--port",
        "8201",
    ]
    return {
        "roles": {
            "teacher": {
                "model": generation.V610_TEACHER_MODEL,
                "resolved_revision": generation.V610_TEACHER_REVISION,
                "api_base": "http://127.0.0.1:8101/v1",
                "quantization": "awq",
                "dtype": receipts.FROZEN_MODEL_DTYPE,
                "tensor_parallel_size": 2,
                "max_model_len": receipts.FROZEN_MAX_MODEL_LEN,
                "gpu_memory_utilization": (
                    receipts.FROZEN_GPU_MEMORY_UTILIZATION
                ),
                "gpu_uuids": ["GPU-0", "GPU-1"],
                "launch_command": teacher_command,
                "server_pid": 101,
                "tokenizer_or_config_sha256": receipts.semantic_sha256(
                    fake_snapshot_files(generation.V610_TEACHER_MODEL)
                ),
            },
            "user": {
                "model": generation.V610_USER_JUDGE_MODEL,
                "resolved_revision": generation.V610_USER_JUDGE_REVISION,
                "api_base": "http://127.0.0.1:8201/v1",
                "quantization": "awq",
                "dtype": receipts.FROZEN_MODEL_DTYPE,
                "tensor_parallel_size": 1,
                "max_model_len": receipts.FROZEN_MAX_MODEL_LEN,
                "gpu_memory_utilization": (
                    receipts.FROZEN_GPU_MEMORY_UTILIZATION
                ),
                "gpu_uuids": ["GPU-2"],
                "launch_command": shared_command,
                "server_pid": 202,
                "tokenizer_or_config_sha256": receipts.semantic_sha256(
                    fake_snapshot_files(generation.V610_USER_JUDGE_MODEL)
                ),
            },
            "judge": {
                "model": generation.V610_USER_JUDGE_MODEL,
                "resolved_revision": generation.V610_USER_JUDGE_REVISION,
                "api_base": "http://127.0.0.1:8201/v1",
                "quantization": "awq",
                "dtype": receipts.FROZEN_MODEL_DTYPE,
                "tensor_parallel_size": 1,
                "max_model_len": receipts.FROZEN_MAX_MODEL_LEN,
                "gpu_memory_utilization": (
                    receipts.FROZEN_GPU_MEMORY_UTILIZATION
                ),
                "gpu_uuids": ["GPU-2"],
                "launch_command": shared_command,
                "server_pid": 202,
                "tokenizer_or_config_sha256": receipts.semantic_sha256(
                    fake_snapshot_files(generation.V610_USER_JUDGE_MODEL)
                ),
            },
        }
    }


def normalized_specs() -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "services.json"
        path.write_text(
            receipts.canonical(service_specs_payload()),
            encoding="utf-8",
        )
        return receipts.load_service_specs(
            path,
            inventory=gpu_inventory(),
        )


def fake_probe(spec, *, timeout_seconds):
    assert timeout_seconds > 0
    return (
        {
            "status": "PASS",
            "observed_model_ids": [spec["model"]],
            "checked_at_utc": TIMESTAMP,
        },
        {
            "status": "PASS",
            "prompt_sha256": "3" * 64,
            "input_token_count": 5000,
            "max_new_tokens": 128,
            "generated_token_count": 64,
            "seed": generation.V610_COMPATIBILITY_SEEDS[0],
            "temperature": 0.0,
            "response_sha256": "4" * 64,
            "latency_seconds": 1.25,
            "peak_gpu_memory_bytes": 1024,
        },
    )


def fake_process_evidence(spec, *, inventory, expected_runtime_python):
    assert len(inventory) == 3
    assert expected_runtime_python == FROZEN_PYTHON
    identity_files = fake_snapshot_files(spec["model"])
    identity_sha = receipts.semantic_sha256(identity_files)
    return {
        "server_pid": spec["server_pid"],
        "process_boot_id": "12345678-1234-1234-1234-123456789abc",
        "process_start_time_ticks": spec["server_pid"] * 100,
        "process_started_at_utc": TIMESTAMP,
        "observed_cmdline_sha256": receipts.semantic_sha256(
            spec["launch_command"]
        ),
        "observed_executable_path": str(expected_runtime_python.resolve()),
        "observed_executable_sha256": FROZEN_PYTHON_SHA256,
        "model_snapshot_path": spec["launch_command"][
            spec["launch_command"].index("--model") + 1
        ],
        "snapshot_identity_files": identity_files,
        "snapshot_identity_files_sha256": identity_sha,
        "observed_cuda_visible_devices": ",".join(
            uuid.rsplit("-", 1)[-1] for uuid in spec["gpu_uuids"]
        ),
        "observed_gpu_uuids": deepcopy(spec["gpu_uuids"]),
        "listening_socket_inode": str(spec["server_pid"] * 10),
        "selected_process_environment": {
            "pythonpath_override_absent": True,
            "pythonhome_override_absent": True,
            "virtual_env": "/frozen",
        },
    }


class RuntimeReceiptTests(unittest.TestCase):
    def test_gpu_inventory_freezes_pro4500_topology(self):
        raw = "\n".join(
            f"{index}, {receipts.FROZEN_GPU_MODEL}, GPU-{index}, "
            "32623, 580.88"
            for index in range(3)
        )
        with patch.object(receipts, "run_checked", return_value=raw):
            inventory, driver = receipts.capture_gpu_inventory()
        self.assertEqual(driver, "580.88")
        self.assertEqual(
            [row["model"] for row in inventory],
            [receipts.FROZEN_GPU_MODEL] * 3,
        )

    def test_gpu_inventory_rejects_unregistered_model(self):
        raw = "\n".join(
            f"{index}, NVIDIA GeForce RTX 5090, GPU-{index}, 32623, 580.88"
            for index in range(3)
        )
        with (
            patch.object(receipts, "run_checked", return_value=raw),
            self.assertRaises(receipts.RuntimeReceiptError),
        ):
            receipts.capture_gpu_inventory()

    def test_live_process_snapshot_gpu_and_socket_identity_is_observed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            pid = 4242
            process = proc / str(pid)
            (process / "fd").mkdir(parents=True)
            (proc / "sys" / "kernel" / "random").mkdir(parents=True)
            (proc / "net").mkdir(parents=True)

            executable = root / "python"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            revision = generation.V610_TEACHER_REVISION
            cache = (
                root
                / "models--Qwen--Qwen2.5-72B-Instruct-AWQ"
            )
            snapshot = cache / "snapshots" / revision
            blobs = cache / "blobs"
            snapshot.mkdir(parents=True)
            snapshot = snapshot.resolve()
            blobs.mkdir()
            for name, value in {
                "config.json": {"model_type": "qwen2"},
                "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
                "tokenizer.json": {"version": "1.0"},
                "model.safetensors.index.json": {
                    "weight_map": {"model.weight": "model-00001-of-00001.safetensors"}
                },
            }.items():
                (snapshot / name).write_text(
                    receipts.canonical(value),
                    encoding="utf-8",
                )
            blob = blobs / ("a" * 64)
            blob.write_bytes(b"weight-fixture")
            (snapshot / "model-00001-of-00001.safetensors").symlink_to(
                blob
            )
            _, snapshot_sha = receipts.snapshot_identity(
                snapshot,
                expected_model=generation.V610_TEACHER_MODEL,
                expected_revision=revision,
            )
            command = [
                str(executable),
                "-I",
                "-m",
                receipts.VLLM_SERVER_MODULE,
                "--model",
                str(snapshot),
                "--revision",
                revision,
                "--served-model-name",
                generation.V610_TEACHER_MODEL,
                "--quantization",
                "awq",
                "--dtype",
                receipts.FROZEN_MODEL_DTYPE,
                "--tensor-parallel-size",
                "2",
                "--load-format",
                "safetensors",
                "--max-model-len",
                str(receipts.FROZEN_MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(receipts.FROZEN_GPU_MEMORY_UTILIZATION),
                "--host",
                "127.0.0.1",
                "--port",
                "8101",
            ]
            start_ticks = 1000
            boot_seconds = 1_700_000_000
            ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
            started_at = datetime.fromtimestamp(
                boot_seconds + start_ticks / ticks_per_second,
                timezone.utc,
            ).isoformat(timespec="seconds").replace("+00:00", "Z")
            spec = deepcopy(service_specs_payload()["roles"]["teacher"])
            spec.update(
                {
                    "server_pid": pid,
                    "launch_command": command,
                    "tokenizer_or_config_sha256": snapshot_sha,
                }
            )
            (process / "cmdline").write_bytes(
                b"\0".join(value.encode("utf-8") for value in command)
                + b"\0"
            )
            stat_fields = ["S", *(["0"] * 18), str(start_ticks)]
            (process / "stat").write_text(
                f"{pid} (python) {' '.join(stat_fields)}\n",
                encoding="utf-8",
            )
            (process / "environ").write_bytes(
                b"CUDA_VISIBLE_DEVICES=0,1\0"
            )
            (process / "exe").symlink_to(executable)
            (
                proc / "sys" / "kernel" / "random" / "boot_id"
            ).write_text(
                "12345678-1234-1234-1234-123456789abc\n",
                encoding="ascii",
            )
            (proc / "stat").write_text(
                f"cpu 1 2 3 4\nbtime {boot_seconds}\n",
                encoding="ascii",
            )
            inode = "12345"
            (process / "fd" / "3").symlink_to(f"socket:[{inode}]")
            header = (
                " sl local_address rem_address st tx_queue rx_queue "
                "tr tm->when retrnsmt uid timeout inode\n"
            )
            (proc / "net" / "tcp").write_text(
                header
                + (
                    "0: 0100007F:1FA5 00000000:0000 0A "
                    "00000000:00000000 00:00000000 00000000 "
                    f"0 0 {inode}\n"
                ),
                encoding="ascii",
            )
            (proc / "net" / "tcp6").write_text(
                header,
                encoding="ascii",
            )

            evidence = receipts.inspect_vllm_process(
                spec,
                inventory=gpu_inventory(),
                expected_runtime_python=executable,
                proc_root=proc,
            )
            self.assertEqual(evidence["server_pid"], pid)
            self.assertEqual(
                evidence["process_started_at_utc"],
                started_at,
            )
            self.assertEqual(
                evidence["snapshot_identity_files_sha256"],
                snapshot_sha,
            )
            self.assertEqual(
                evidence["observed_gpu_uuids"],
                ["GPU-0", "GPU-1"],
            )
            self.assertEqual(evidence["listening_socket_inode"], inode)

            (process / "cmdline").write_bytes(b"/wrong\0")
            with self.assertRaisesRegex(
                receipts.RuntimeReceiptError,
                "differs from live process argv",
            ):
                receipts.inspect_vllm_process(
                    spec,
                    inventory=gpu_inventory(),
                    expected_runtime_python=executable,
                    proc_root=proc,
                )
            (process / "cmdline").write_bytes(
                b"\0".join(value.encode("utf-8") for value in command)
                + b"\0"
            )
            (snapshot / "config.json").write_text(
                receipts.canonical({"model_type": "tampered"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                receipts.RuntimeReceiptError,
                "snapshot identity hash differs",
            ):
                receipts.inspect_vllm_process(
                    spec,
                    inventory=gpu_inventory(),
                    expected_runtime_python=executable,
                    proc_root=proc,
                )

    def test_probe_requires_exact_models_and_real_long_context_tokens(self):
        spec = normalized_specs()["teacher"]
        requests: list[tuple[str, dict | None]] = []

        def request_json(url, *, payload, timeout_seconds):
            requests.append((url, deepcopy(payload)))
            self.assertGreater(timeout_seconds, 0)
            return {"data": [{"id": spec["model"]}]}

        def generation_request(
            *,
            url,
            payload,
            gpu_uuids,
            timeout_seconds,
            request_json,
        ):
            self.assertTrue(url.endswith("/chat/completions"))
            self.assertEqual(payload["model"], spec["model"])
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["seed"], 20260806)
            self.assertGreaterEqual(
                len(payload["messages"][0]["content"].split()),
                4096,
            )
            self.assertEqual(gpu_uuids, ["GPU-0", "GPU-1"])
            return (
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ],
                    "usage": {
                        "prompt_tokens": 5001,
                        "completion_tokens": 8,
                    },
                },
                4096,
                0.5,
            )

        health, probe = receipts.probe_service(
            spec,
            timeout_seconds=10,
            request_json=request_json,
            generation_request=generation_request,
        )
        self.assertEqual(
            requests,
            [("http://127.0.0.1:8101/v1/models", None)],
        )
        self.assertEqual(health["observed_model_ids"], [spec["model"]])
        self.assertEqual(probe["input_token_count"], 5001)
        self.assertEqual(probe["generated_token_count"], 8)
        self.assertEqual(probe["peak_gpu_memory_bytes"], 4096)

        def wrong_models(url, *, payload, timeout_seconds):
            return {"data": [{"id": "wrong"}]}

        with self.assertRaisesRegex(
            receipts.RuntimeReceiptError,
            "exact /models identity mismatch",
        ):
            receipts.probe_service(
                spec,
                timeout_seconds=10,
                request_json=wrong_models,
                generation_request=generation_request,
            )

    def test_probe_rejects_claimed_long_context_without_server_token_evidence(
        self,
    ):
        spec = normalized_specs()["teacher"]

        def request_json(url, *, payload, timeout_seconds):
            return {"data": [{"id": spec["model"]}]}

        def short_generation(**unused):
            return (
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 4095,
                        "completion_tokens": 1,
                    },
                },
                1,
                0.1,
            )

        with self.assertRaisesRegex(
            receipts.RuntimeReceiptError,
            "token evidence",
        ):
            receipts.probe_service(
                spec,
                timeout_seconds=10,
                request_json=request_json,
                generation_request=short_generation,
            )

    def test_model_receipt_caches_shared_endpoint_and_matches_validator(self):
        specs = normalized_specs()
        calls: list[str] = []

        def counted_probe(spec, *, timeout_seconds):
            calls.append(spec["api_base"])
            return fake_probe(spec, timeout_seconds=timeout_seconds)

        payload = receipts.build_model_receipt(
            specs=specs,
            inventory=gpu_inventory(),
            runtime_python=FROZEN_PYTHON,
            container_image_digest=CONTAINER,
            timeout_seconds=10,
            probe=counted_probe,
            process_inspector=fake_process_evidence,
        )
        self.assertEqual(
            calls,
            [
                "http://127.0.0.1:8101/v1",
                "http://127.0.0.1:8201/v1",
            ],
        )
        self.assertEqual(
            payload["roles"]["user"]["generation_probe"],
            payload["roles"]["judge"]["generation_probe"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "models.json"
            path.write_text(receipts.canonical(payload), encoding="utf-8")
            generation.load_model_server_receipts(
                path,
                expected_sha256=receipts.file_sha256(path),
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
                    for role in receipts.ROLE_ORDER
                },
                container_image_digest=CONTAINER,
                available_gpu_uuids=[
                    row["uuid"] for row in gpu_inventory()
                ],
                expected_python_executable_path=str(FROZEN_PYTHON),
                expected_python_executable_sha256=FROZEN_PYTHON_SHA256,
            )

    def test_model_receipt_rejects_process_restart_during_probe(self):
        specs = normalized_specs()
        inspections = 0

        def changing_process(spec, *, inventory, expected_runtime_python):
            nonlocal inspections
            inspections += 1
            evidence = fake_process_evidence(
                spec,
                inventory=inventory,
                expected_runtime_python=expected_runtime_python,
            )
            if inspections == 2:
                evidence["process_start_time_ticks"] += 1
            return evidence

        with self.assertRaisesRegex(
            receipts.RuntimeReceiptError,
            "changed during live probe",
        ):
            receipts.build_model_receipt(
                specs=specs,
                inventory=gpu_inventory(),
                runtime_python=FROZEN_PYTHON,
                container_image_digest=CONTAINER,
                timeout_seconds=10,
                probe=fake_probe,
                process_inspector=changing_process,
            )

    def test_service_specs_fail_closed_on_nonshared_judge_topology(self):
        payload = service_specs_payload()
        payload["roles"]["judge"]["api_base"] = (
            "http://127.0.0.1:8301/v1"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "services.json"
            path.write_text(receipts.canonical(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                receipts.RuntimeReceiptError,
                "shared user/judge",
            ):
                receipts.load_service_specs(
                    path,
                    inventory=gpu_inventory(),
                )

    def test_source_receipt_matches_generator_strict_validator(self):
        runtime = {
            "python_version": "3.11.9",
            "python_executable_path": str(FROZEN_PYTHON),
            "python_executable_sha256": FROZEN_PYTHON_SHA256,
            "torch_version": "2.7.1",
            "cuda_version": "12.8",
            "driver_version": "575.57",
            "vllm_version": "0.10.2",
        }
        payload = receipts.build_source_receipt(
            source_commit="5" * 40,
            generator_sha256="6" * 64,
            tau2_commit=generation.protocol.TAU2_COMMIT,
            container_image_digest=CONTAINER,
            dependency_lock_sha256="7" * 64,
            runtime=runtime,
            inventory=gpu_inventory(),
            created_at_utc=TIMESTAMP,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.json"
            path.write_text(receipts.canonical(payload), encoding="utf-8")
            generation.load_source_container_provenance(
                path,
                expected_sha256=receipts.file_sha256(path),
                expected_source_commit="5" * 40,
                expected_generation_script_sha256="6" * 64,
                expected_tau2_commit=generation.protocol.TAU2_COMMIT,
                container_image_digest=CONTAINER,
            )

    def test_end_to_end_builder_publishes_only_valid_canonical_receipts(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            dependency_lock = root / "uv.lock"
            dependency_lock.write_text("frozen-dependencies\n", encoding="utf-8")
            service_specs = root / "services.json"
            service_specs.write_text(
                receipts.canonical(service_specs_payload()),
                encoding="utf-8",
            )
            output = root / "receipts"
            args = argparse.Namespace(
                source_root=repository,
                tau2_root=tau2,
                dependency_lock=dependency_lock,
                container_image_digest=CONTAINER,
                service_specs=service_specs,
                output_dir=output,
                runtime_python=FROZEN_PYTHON,
                request_timeout=10.0,
            )

            def fake_git(path):
                return {
                    "commit": (
                        generation.protocol.TAU2_COMMIT
                        if path.resolve() == tau2.resolve()
                        else "8" * 40
                    ),
                    "tree": "9" * 40,
                    "tracked_worktree_clean": True,
                }

            with (
                patch.object(receipts, "git_identity", side_effect=fake_git),
                patch.object(
                    receipts,
                    "capture_runtime",
                    return_value={
                        "python_version": "3.11.9",
                        "python_executable_path": str(FROZEN_PYTHON),
                        "python_executable_sha256": FROZEN_PYTHON_SHA256,
                        "torch_version": "2.7.1",
                        "cuda_version": "12.8",
                        "vllm_version": "0.10.2",
                    },
                ),
                patch.object(
                    receipts,
                    "capture_gpu_inventory",
                    return_value=(gpu_inventory(), "575.57"),
                ),
                patch.object(
                    receipts,
                    "probe_service",
                    side_effect=fake_probe,
                ),
                patch.object(
                    receipts,
                    "inspect_vllm_process",
                    side_effect=fake_process_evidence,
                ),
            ):
                summary = receipts.build_receipts(args)

            self.assertEqual(summary["status"], "PASS")
            source_path = output / receipts.SOURCE_RECEIPT_NAME
            model_path = output / receipts.MODEL_RECEIPT_NAME
            hash_path = output / receipts.HASH_MANIFEST_NAME
            self.assertTrue(source_path.is_file())
            self.assertTrue(model_path.is_file())
            self.assertTrue(hash_path.is_file())
            for path in (source_path, model_path, hash_path):
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    receipts.canonical(value),
                )
            hashes = json.loads(hash_path.read_text(encoding="utf-8"))
            self.assertEqual(
                hashes["source_container_provenance"]["file_sha256"],
                receipts.file_sha256(source_path),
            )
            self.assertEqual(
                hashes["model_server_receipts"]["file_sha256"],
                receipts.file_sha256(model_path),
            )

    def test_failed_live_probe_leaves_no_partial_output(self):
        specs = normalized_specs()

        def fail_probe(spec, *, timeout_seconds):
            raise receipts.RuntimeReceiptError("endpoint unavailable")

        with self.assertRaisesRegex(
            receipts.RuntimeReceiptError,
            "endpoint unavailable",
        ):
            receipts.build_model_receipt(
                specs=specs,
                inventory=gpu_inventory(),
                runtime_python=FROZEN_PYTHON,
                container_image_digest=CONTAINER,
                timeout_seconds=10,
                probe=fail_probe,
                process_inspector=fake_process_evidence,
            )


if __name__ == "__main__":
    unittest.main()
