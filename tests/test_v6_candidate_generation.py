from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_v6_candidate_registry as registry_contract
from scripts import run_v6_candidate_generation as generation
from scripts import v6_selection_protocol as protocol


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FakeReferenceAction:
    def __init__(self, call: dict) -> None:
        self.call = deepcopy(call)

    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return deepcopy(self.call)


def test_registered_reference_index_disambiguates_duplicate_calls():
    duplicate = {
        "id": "reference-duplicate",
        "name": "get_order_details",
        "arguments": {"order_id": "#W7449508"},
        "requestor": "assistant",
    }
    actions = [
        FakeReferenceAction(duplicate),
        FakeReferenceAction(
            {
                "id": "other",
                "name": "get_user_details",
                "arguments": {"user_id": "u1"},
                "requestor": "assistant",
            }
        ),
        FakeReferenceAction({**duplicate, "id": "reference-duplicate-again"}),
    ]

    assert (
        generation.resolve_forced_reference_index(
            actions, duplicate, registered_index=2
        )
        == 2
    )


def test_registered_reference_index_still_validates_frozen_call_semantics():
    actions = [
        FakeReferenceAction(
            {
                "id": "registered",
                "name": "get_order_details",
                "arguments": {"order_id": "#W7449508"},
                "requestor": "assistant",
            }
        )
    ]
    mismatched = {
        "id": "forced",
        "name": "get_order_details",
        "arguments": {"order_id": "#DIFFERENT"},
        "requestor": "assistant",
    }

    with unittest.TestCase().assertRaisesRegex(
        generation.V6GenerationError,
        "does not match forced call",
    ):
        generation.resolve_forced_reference_index(
            actions, mismatched, registered_index=0
        )


def branch(pair_id: str, index: int) -> dict:
    return {
        "branch_id": f"{pair_id}:branch:{index + 1}",
        "candidate_pair_id": pair_id,
        "identifier_key": f"id_{index}",
        "tool_name": f"lookup_{index}",
        "corrective_family": f"lookup_{index}::id_{index}",
        "injection_spec": {
            "error_call": {
                "id": f"error-{index}",
                "name": f"lookup_{index}",
                "arguments": {f"id_{index}": "wrong"},
                "requestor": "assistant",
            }
        },
        "corrective_action_spec": {
            "forced_first_action_constructor": {
                "kind": "exact_registered_reference_tool_call",
                "tool_call": {
                    "id": f"correct-{index}",
                    "name": f"lookup_{index}",
                    "arguments": {f"id_{index}": "right"},
                    "requestor": "assistant",
                },
            }
        },
        "recovery_seed": 100 + index,
        "official_test_used": False,
    }


def pair(task: str, index: int, phase: str = "pilot") -> dict:
    pair_id = f"v6:{phase}:{task}:pair:{index}"
    value = {
        "candidate_pair_id": pair_id,
        "choice_set_id": f"v6:{phase}:{task}:choice",
        "phase": phase,
        "partition": "arm_train",
        "task_identity": task,
        "domain": task.split(":", 1)[0],
        "task_id": task.split(":", 1)[1],
        "prefix_sha256": digest(f"{task}:prefix-spec"),
        "environment_snapshot_sha256": digest(f"{task}:snapshot-spec"),
        "branches": [branch(pair_id, 0), branch(pair_id, 1)],
        "official_test_used": False,
    }
    value["candidate_pair_sha256"] = generation.sha256(value)
    return value


def registry() -> dict:
    tasks = ["retail:1", "airline:2"]
    rows = [
        pair(task, index)
        for task in tasks
        for index in range(protocol.MIN_PAIRS_PER_TASK)
    ]
    value = {
        "protocol": registry_contract.REGISTRY_PROTOCOL,
        "design_protocol": protocol.PROTOCOL,
        "selection_unit": "candidate_pair",
        "grouping_unit": "choice_set",
        "official_test_used": False,
        "official_test_sealed": True,
        "official_test_task_content_exported": False,
        "official_test_identity_overlap_count": 0,
        "structural_eligibility_sha256": protocol.STRUCTURAL_ELIGIBILITY_SHA256,
        "phase_registry": {
            "pilot": {"task_ids": tasks},
            "formal": {"task_ids": tasks},
        },
        "candidate_pairs": rows,
    }
    value["registry_sha256"] = generation.sha256(value)
    return value


def test_v6_1_registry_is_explicitly_accepted_and_bound_to_run_contract():
    payload = registry()
    payload["design_protocol"] = registry_contract.V6_1_72B_TEACHER_PROTOCOL
    payload["registry_sha256"] = generation.sha256(
        {key: value for key, value in payload.items() if key != "registry_sha256"}
    )
    generation.verify_registry(payload)

    args = generation_args()
    semantic = generation.semantic_generation_contract(
        args,
        continuation_seeds=[20260806, 20260807, 20260808],
        design_protocol=payload["design_protocol"],
    )
    contract = generation.build_run_contract(
        args,
        registry_file_sha256="a" * 64,
        registry_sha256=payload["registry_sha256"],
        task_ids=["airline:2", "retail:1"],
        semantic_contract=semantic,
    )
    assert contract["design_protocol"] == payload["design_protocol"]
    assert (
        contract["semantic_generation_contract"]["design_protocol"]
        == payload["design_protocol"]
    )


def generation_args(**overrides) -> SimpleNamespace:
    values = {
        "phase": "pilot",
        "shard_index": 0,
        "num_shards": 1,
        "smoke_task": None,
        "teacher_model": "teacher",
        "teacher_revision": "teacher-revision",
        "teacher_api_base": "http://teacher.example/v1",
        "user_model": "user",
        "user_revision": "user-revision",
        "user_api_base": "http://user.example/v1",
        "judge_model": "judge",
        "judge_revision": "judge-revision",
        "judge_api_base": "http://judge.example/v1",
        "max_tokens": 512,
        "max_steps": 60,
        "timeout": 900.0,
        "clean_attempts": 2,
        "recovery_attempts": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def v610_provenance() -> tuple[dict, dict]:
    parent_hashes = {
        key: digest(key)
        for key in generation.V610_PARENT_ARTIFACT_HASH_KEYS
    }
    task_file_hashes = {
        "airline": digest("airline-tasks"),
        "retail": digest("retail-tasks"),
    }
    source_commit = "1" * 40
    source_tree = "2" * 40
    container_image_digest = "sha256:" + "3" * 64
    identities = {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "container_image_digest": container_image_digest,
        "tau2_commit": protocol.TAU2_COMMIT,
        **{
            key: value
            for key, value in parent_hashes.items()
            if key != "release_manifest_sha256"
        },
    }
    relevant_scripts = {
        relative: digest(relative)
        for relative in generation.V610_REQUIRED_RELEASE_SCRIPTS
    }
    release_manifest = {
        "protocol": generation.V610_RELEASE_MANIFEST_PROTOCOL,
        "file_sha256": parent_hashes["release_manifest_sha256"],
        "semantic_sha256": digest("release-semantic"),
        "created_at_utc": "2026-07-30T00:00:00Z",
        "identities": identities,
        "relevant_scripts": relevant_scripts,
        "relevant_scripts_sha256": generation.sha256(relevant_scripts),
        "official_test_used": False,
    }
    runtime = {
        "attempt_id": "v6_10-pilot-a01-20260730T000000Z",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "container_image_digest": container_image_digest,
        "tau2_commit": protocol.TAU2_COMMIT,
        "parent_artifact_hashes": parent_hashes,
        "task_file_sha256": task_file_hashes,
        "release_manifest": release_manifest,
        "release_manifest_sha256": parent_hashes[
            "release_manifest_sha256"
        ],
    }
    execution = generation.build_execution_provenance(
        run_started_at_utc="2026-07-30T00:00:00Z",
        exact_argv=(
            "/usr/bin/python3",
            "scripts/run_v6_candidate_generation.py",
            "--phase",
            "pilot",
        ),
    )
    return runtime, execution


class V6CandidateGenerationTests(unittest.TestCase):
    def test_release_manifest_requires_exact_script_set_and_live_bytes(self):
        identity = {
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "container_image_digest": "sha256:" + "3" * 64,
            "tau2_commit": "4" * 40,
            "config_sha256": digest("config"),
            "preregistration_sha256": digest("preregistration"),
            "split_manifest_sha256": digest("split"),
            "reference_preflight_receipt_sha256": digest("preflight"),
            "registry_file_sha256": digest("registry"),
            "source_container_provenance_sha256": digest("source"),
            "model_server_receipts_sha256": digest("models"),
            "runtime_receipt_hashes_sha256": digest("runtime"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "source"
            config_path = source_root / "configs" / "v6_10_closure.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("protocol: v6-test\n", encoding="utf-8")
            preregistration_path = (
                source_root / "V6_10_CLOSURE_PREREGISTRATION.md"
            )
            preregistration_path.write_text(
                "# Frozen V6 test\n",
                encoding="utf-8",
            )
            identity["config_sha256"] = generation.sha256_file(config_path)
            identity["preregistration_sha256"] = generation.sha256_file(
                preregistration_path
            )
            scripts = {}
            for relative in sorted(
                generation.V610_REQUIRED_RELEASE_SCRIPTS
            ):
                script = source_root / relative
                script.parent.mkdir(parents=True, exist_ok=True)
                script.write_text(f"{relative}\n", encoding="utf-8")
                scripts[relative] = generation.sha256_file(script)
            subprocess.run(
                ["git", "init", "-q", str(source_root)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source_root), "add", "--", "."],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "-c",
                    "user.name=V6 Test",
                    "-c",
                    "user.email=v6@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "frozen scripts",
                ],
                check=True,
            )
            def refresh_identity() -> None:
                identity["source_commit"] = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                identity["source_tree"] = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(source_root),
                        "rev-parse",
                        "HEAD^{tree}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            refresh_identity()
            payload = {
                "protocol": generation.V610_RELEASE_MANIFEST_PROTOCOL,
                "design_protocol": (
                    generation.V6_10_PIPELINE_CLOSURE_PROTOCOL
                ),
                "status": "PASS",
                **identity,
                "relevant_scripts": scripts,
                "created_at_utc": "2026-07-30T00:00:00Z",
                "official_test_used": False,
            }
            manifest = Path(temporary) / "release.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            _, binding = generation.load_release_manifest(
                manifest,
                expected_sha256=generation.sha256_file(manifest),
                source_root=source_root,
                expected_fields=identity,
            )
            self.assertEqual(
                set(binding["relevant_scripts"]),
                generation.V610_REQUIRED_RELEASE_SCRIPTS,
            )

            extra = deepcopy(payload)
            extra["relevant_scripts"]["scripts/unfrozen.py"] = digest(
                "unfrozen"
            )
            manifest.write_text(json.dumps(extra), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "exact schema",
            ):
                generation.load_release_manifest(
                    manifest,
                    expected_sha256=generation.sha256_file(manifest),
                    source_root=source_root,
                    expected_fields=identity,
                )

            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "rm",
                    "--cached",
                    "--",
                    "V6_10_CLOSURE_PREREGISTRATION.md",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "-c",
                    "user.name=V6 Test",
                    "-c",
                    "user.email=v6@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "remove frozen preregistration",
                ],
                check=True,
            )
            refresh_identity()
            payload["source_commit"] = identity["source_commit"]
            payload["source_tree"] = identity["source_tree"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "frozen preregistration is not tracked",
            ):
                generation.load_release_manifest(
                    manifest,
                    expected_sha256=generation.sha256_file(manifest),
                    source_root=source_root,
                    expected_fields=identity,
                )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "add",
                    "--",
                    "V6_10_CLOSURE_PREREGISTRATION.md",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "-c",
                    "user.name=V6 Test",
                    "-c",
                    "user.email=v6@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "restore frozen preregistration",
                ],
                check=True,
            )
            refresh_identity()
            payload["source_commit"] = identity["source_commit"]
            payload["source_tree"] = identity["source_tree"]

            # A live file with the advertised bytes is insufficient when that
            # file is absent from the frozen source commit.
            first_script_relative = sorted(scripts)[0]
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "rm",
                    "--cached",
                    "--",
                    first_script_relative,
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root),
                    "-c",
                    "user.name=V6 Test",
                    "-c",
                    "user.email=v6@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "remove one scientific script",
                ],
                check=True,
            )
            refresh_identity()
            payload["source_commit"] = identity["source_commit"]
            payload["source_tree"] = identity["source_tree"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "not tracked by the frozen source commit",
            ):
                generation.load_release_manifest(
                    manifest,
                    expected_sha256=generation.sha256_file(manifest),
                    source_root=source_root,
                    expected_fields=identity,
                )

            manifest.write_text(json.dumps(payload), encoding="utf-8")
            first_script = source_root / sorted(scripts)[0]
            first_script.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "byte hash drift",
            ):
                generation.load_release_manifest(
                    manifest,
                    expected_sha256=generation.sha256_file(manifest),
                    source_root=source_root,
                    expected_fields=identity,
                )

    def test_attempt_ids_are_phase_specific_and_calendar_valid(self):
        generation.validate_v610_attempt_id(
            "compatibility",
            "v6_10-compat-a01-20260730T123456Z",
        )
        generation.validate_v610_attempt_id(
            "pilot",
            "v6_10-pilot-a02-20260730T123456Z",
        )
        generation.validate_v610_attempt_id(
            "formal",
            "v6_10-formal-a99-20260730T123456Z",
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "phase-specific",
        ):
            generation.validate_v610_attempt_id(
                "formal",
                "v6_10-pilot-a01-20260730T123456Z",
            )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "invalid UTC calendar",
        ):
            generation.validate_v610_attempt_id(
                "pilot",
                "v6_10-pilot-a01-20260230T123456Z",
            )

    def test_v610_resume_reuses_original_run_start_across_process_time(self):
        with patch.object(
            generation,
            "utc_now",
            return_value="2026-07-30T00:00:00Z",
        ):
            started = generation.resolve_v610_run_started_at_utc(None)
        existing = {
            "execution_provenance": {
                "run_started_at_utc": started,
            }
        }
        with patch.object(
            generation,
            "utc_now",
            return_value="2026-07-31T00:00:00Z",
        ):
            resumed = generation.resolve_v610_run_started_at_utc(existing)
        self.assertEqual(resumed, "2026-07-30T00:00:00Z")

    def test_main_passes_loaded_registry_and_byte_hash_to_v610_provenance(self):
        runtime, _ = v610_provenance()
        runtime["reference_preflight"] = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = root / "registry.json"
            registry_payload = {
                "design_protocol": (
                    generation.V6_10_PIPELINE_CLOSURE_PROTOCOL
                )
            }
            registry_path.write_text(
                json.dumps(registry_payload),
                encoding="utf-8",
            )
            registry_file_sha256 = generation.sha256_file(registry_path)
            runtime["parent_artifact_hashes"][
                "registry_file_sha256"
            ] = registry_file_sha256
            runtime["release_manifest"]["identities"][
                "registry_file_sha256"
            ] = registry_file_sha256
            args = generation_args(
                tau2_root=root / "tau2",
                registry=registry_path,
                output_dir=root / "output",
                continuation_seeds="101",
            )
            with (
                patch.object(generation, "parse_args", return_value=args),
                patch.object(
                    generation,
                    "verify_registry",
                    return_value=digest("semantic-registry"),
                ),
                patch.object(generation, "validate_v610_modes"),
                patch.object(
                    generation,
                    "v610_runtime_provenance",
                    return_value=(runtime, {}),
                ) as provenance_call,
                patch.object(
                    generation,
                    "verify_v610_registry_preflight_binding",
                ),
                patch.object(generation, "configure_tau2"),
                patch.object(generation, "register_agent"),
                patch.object(generation, "phase_pairs", return_value=[]),
                patch.object(
                    generation,
                    "selected_phase_tasks",
                    return_value=[],
                ),
                patch.object(
                    generation,
                    "preflight_selected_phase_tasks",
                    return_value=(
                        {},
                        {
                            "status": "CLASSIFIED",
                            "official_test_used": False,
                        },
                    ),
                ),
                patch.object(generation, "patch_local_nl_judge"),
                patch.object(
                    generation,
                    "revalidate_v610_live_model_runtime",
                ),
                patch.object(
                    generation,
                    "merge_shard",
                    return_value={"status": "PASS"},
                ),
                patch.object(
                    generation,
                    "utc_now",
                    return_value="2026-07-30T00:00:00Z",
                ),
                patch("builtins.print"),
            ):
                generation.main()
            self.assertEqual(
                provenance_call.call_args.kwargs["registry"],
                registry_payload,
            )
            self.assertEqual(
                provenance_call.call_args.kwargs["registry_file_sha256"],
                generation.sha256_file(registry_path),
            )

    def test_model_server_receipts_bind_roles_revisions_and_container(self):
        container = "sha256:" + "c" * 64
        runtime_python = str(Path(sys.executable).resolve())
        runtime_python_sha256 = generation.sha256_file(Path(runtime_python))

        def role_receipt(
            role: str,
            model: str,
            revision: str,
            api_base: str,
            tensor_parallel_size: int,
        ) -> dict:
            gpu_uuids = (
                ["GPU-0", "GPU-1"]
                if role == "teacher"
                else ["GPU-2"]
            )
            server_pid = 101 if role == "teacher" else 202
            snapshot_identity = {
                "byte_files": {
                    "config.json": digest(f"{model}-config"),
                    "tokenizer.json": digest(f"{model}-tokenizer"),
                },
                "weight_blob_targets": {
                    "model-00001-of-00001.safetensors": "a" * 64,
                },
            }
            launch_command = [
                runtime_python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                f"/models/{model}/{revision}",
                "--tensor-parallel-size",
                str(tensor_parallel_size),
            ]
            return {
                "role": role,
                "model": model,
                "resolved_revision": revision,
                "api_base": api_base,
                "container_image_digest": container,
                "quantization": "awq",
                "dtype": "float16",
                "tensor_parallel_size": tensor_parallel_size,
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.90,
                "gpu_uuids": gpu_uuids,
                "launch_command": launch_command,
                "launch_command_sha256": generation.sha256(launch_command),
                "server_pid": server_pid,
                "process_boot_id": "12345678-1234-1234-1234-123456789abc",
                "process_start_time_ticks": server_pid * 100,
                "process_started_at_utc": "2026-07-30T01:00:00Z",
                "observed_cmdline_sha256": generation.sha256(launch_command),
                "observed_executable_path": runtime_python,
                "observed_executable_sha256": runtime_python_sha256,
                "model_snapshot_path": f"/models/{model}/{revision}",
                "snapshot_identity_files": snapshot_identity,
                "snapshot_identity_files_sha256": generation.sha256(
                    snapshot_identity
                ),
                "observed_cuda_visible_devices": (
                    "0,1" if role == "teacher" else "2"
                ),
                "observed_gpu_uuids": gpu_uuids,
                "listening_socket_inode": str(server_pid * 10),
                "selected_process_environment": {
                    "pythonpath_override_absent": True,
                    "pythonhome_override_absent": True,
                    "virtual_env": None,
                },
                "tokenizer_or_config_sha256": generation.sha256(
                    snapshot_identity
                ),
                "launched_at_utc": "2026-07-30T01:00:00Z",
                "healthcheck": {
                    "status": "PASS",
                    "observed_model_ids": [model],
                    "checked_at_utc": "2026-07-30T01:01:00Z",
                },
                "generation_probe": {
                    "status": "PASS",
                    "prompt_sha256": digest(f"{role}-probe-prompt"),
                    "input_token_count": 4096,
                    "max_new_tokens": 128,
                    "generated_token_count": 128,
                    "seed": 20260806,
                    "temperature": 0.0,
                    "response_sha256": digest(f"{role}-probe-response"),
                    "latency_seconds": 1.5,
                    "peak_gpu_memory_bytes": 1024,
                },
            }

        payload = {
            "protocol": generation.V610_MODEL_SERVER_RECEIPTS_PROTOCOL,
            "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            "status": "PASS",
            "container_image_digest": container,
            "roles": {
                "teacher": role_receipt(
                    "teacher",
                    "teacher",
                    "teacher-revision",
                    "http://teacher.example/v1",
                    2,
                ),
                "user": role_receipt(
                    "user",
                    "user",
                    "user-revision",
                    "http://user.example/v1",
                    1,
                ),
                "judge": role_receipt(
                    "judge",
                    "user",
                    "user-revision",
                    "http://user.example/v1",
                    1,
                ),
            },
            "official_test_used": False,
        }
        expected_roles = {
            "teacher": {
                "model": "teacher",
                "revision": "teacher-revision",
                "api_base": "http://teacher.example/v1",
                "quantization": "awq",
                "tensor_parallel_size": 2,
            },
            "user": {
                "model": "user",
                "revision": "user-revision",
                "api_base": "http://user.example/v1",
                "quantization": "awq",
                "tensor_parallel_size": 1,
            },
            "judge": {
                "model": "user",
                "revision": "user-revision",
                "api_base": "http://user.example/v1",
                "quantization": "awq",
                "tensor_parallel_size": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model-receipts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            expected = generation.sha256_file(path)
            _, binding = generation.load_model_server_receipts(
                path,
                expected_sha256=expected,
                expected_roles=expected_roles,
                container_image_digest=container,
                available_gpu_uuids=["GPU-0", "GPU-1", "GPU-2"],
                expected_python_executable_path=runtime_python,
                expected_python_executable_sha256=runtime_python_sha256,
            )
            self.assertEqual(binding["file_sha256"], expected)
            self.assertEqual(
                binding["roles"]["teacher"]["api_base"],
                "http://teacher.example/v1",
            )

            # A correct value at an unrelated JSON path must not satisfy the
            # role-scoped contract.
            drifted = deepcopy(payload)
            drifted["roles"]["teacher"]["model"] = "wrong-model"
            drifted["roles"]["teacher"]["healthcheck"][
                "observed_model_ids"
            ].append("teacher")
            path.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "roles.teacher",
            ):
                generation.load_model_server_receipts(
                    path,
                    expected_sha256=generation.sha256_file(path),
                    expected_roles=expected_roles,
                    container_image_digest=container,
                    available_gpu_uuids=["GPU-0", "GPU-1", "GPU-2"],
                    expected_python_executable_path=runtime_python,
                    expected_python_executable_sha256=runtime_python_sha256,
                )

            topology_drift = deepcopy(payload)
            topology_drift["roles"]["judge"]["gpu_uuids"] = ["GPU-1"]
            path.write_text(json.dumps(topology_drift), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "roles.judge|3-GPU topology",
            ):
                generation.load_model_server_receipts(
                    path,
                    expected_sha256=generation.sha256_file(path),
                    expected_roles=expected_roles,
                    container_image_digest=container,
                    available_gpu_uuids=["GPU-0", "GPU-1", "GPU-2"],
                    expected_python_executable_path=runtime_python,
                    expected_python_executable_sha256=runtime_python_sha256,
                )

            impossible_date = deepcopy(payload)
            impossible_date["roles"]["teacher"][
                "launched_at_utc"
            ] = "2026-02-30T01:00:00Z"
            path.write_text(json.dumps(impossible_date), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "roles.teacher",
            ):
                generation.load_model_server_receipts(
                    path,
                    expected_sha256=generation.sha256_file(path),
                    expected_roles=expected_roles,
                    container_image_digest=container,
                    available_gpu_uuids=["GPU-0", "GPU-1", "GPU-2"],
                    expected_python_executable_path=runtime_python,
                    expected_python_executable_sha256=runtime_python_sha256,
                )

    def test_source_container_receipt_binds_release_identities(self):
        container = "sha256:" + "d" * 64
        payload = {
            "protocol": generation.V610_SOURCE_CONTAINER_RECEIPT_PROTOCOL,
            "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            "status": "PASS",
            "source_commit": "1" * 40,
            "generation_script_sha256": "2" * 64,
            "tau2_commit": "3" * 40,
            "container_image_digest": container,
            "dependency_lock_sha256": digest("dependency-lock"),
            "runtime": {
                "python_version": "3.12.3",
                "python_executable_path": str(Path(sys.executable).resolve()),
                "python_executable_sha256": generation.sha256_file(
                    Path(sys.executable).resolve()
                ),
                "torch_version": "2.7.1",
                "cuda_version": "12.8",
                "driver_version": "575.57",
                "vllm_version": "0.9.2",
            },
            "gpus": [
                {
                    "index": index,
                    "model": "NVIDIA GeForce RTX 5090",
                    "uuid": f"GPU-{index}",
                    "total_memory_bytes": 34_000_000_000,
                }
                for index in range(3)
            ],
            "created_at_utc": "2026-07-30T01:00:00Z",
            "official_test_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source-container.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            expected = generation.sha256_file(path)
            _, binding = generation.load_source_container_provenance(
                path,
                expected_sha256=expected,
                expected_source_commit="1" * 40,
                expected_generation_script_sha256="2" * 64,
                expected_tau2_commit="3" * 40,
                container_image_digest=container,
            )
            self.assertEqual(binding["file_sha256"], expected)

            drifted = deepcopy(payload)
            drifted["source_commit"] = "9" * 40
            drifted["note"] = "1" * 40
            path.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "exact schema",
            ):
                generation.load_source_container_provenance(
                    path,
                    expected_sha256=generation.sha256_file(path),
                    expected_source_commit="1" * 40,
                    expected_generation_script_sha256="2" * 64,
                    expected_tau2_commit="3" * 40,
                    container_image_digest=container,
                )

            impossible_date = deepcopy(payload)
            impossible_date["created_at_utc"] = "2026-02-30T01:00:00Z"
            path.write_text(json.dumps(impossible_date), encoding="utf-8")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "exact schema",
            ):
                generation.load_source_container_provenance(
                    path,
                    expected_sha256=generation.sha256_file(path),
                    expected_source_commit="1" * 40,
                    expected_generation_script_sha256="2" * 64,
                    expected_tau2_commit="3" * 40,
                    container_image_digest=container,
                )

    def test_live_model_endpoint_requires_exact_served_model_identity(self):
        class Response:
            status = 200

            def __init__(
                self,
                model_ids: list[str],
                *,
                created: int = 1,
                permission_id: str = "permission-1",
            ) -> None:
                self.body = json.dumps(
                    {
                        "data": [
                            {
                                "id": model_id,
                                "created": created,
                                "permission": [
                                    {
                                        "id": permission_id,
                                        "created": created,
                                    }
                                ],
                            }
                            for model_id in model_ids
                        ]
                    }
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self) -> bytes:
                return self.body

        with patch.object(
            generation.urllib_request,
            "urlopen",
            return_value=Response(["teacher"]),
        ) as urlopen:
            receipt = generation.verify_live_model_endpoint(
                role="teacher",
                api_base="http://teacher.example/v1",
                expected_model="teacher",
            )
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["observed_model_ids"], ["teacher"])
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "http://teacher.example/v1/models",
        )
        with patch.object(
            generation.urllib_request,
            "urlopen",
            return_value=Response(
                ["teacher"],
                created=999,
                permission_id="random-vllm-uuid",
            ),
        ):
            resumed_receipt = generation.verify_live_model_endpoint(
                role="teacher",
                api_base="http://teacher.example/v1",
                expected_model="teacher",
            )
        self.assertEqual(resumed_receipt, receipt)
        self.assertTrue(
            receipt["raw_response_excluded_from_semantic_contract"]
        )
        self.assertEqual(
            receipt["stable_identity_sha256"],
            generation.sha256(
                {
                    "status_code": 200,
                    "observed_model_ids": ["teacher"],
                }
            ),
        )

        with (
            patch.object(
                generation.urllib_request,
                "urlopen",
                return_value=Response(["unrelated-model"]),
            ),
            self.assertRaisesRegex(
                generation.V6GenerationError,
                "identity mismatch",
            ),
        ):
            generation.verify_live_model_endpoint(
                role="teacher",
                api_base="http://teacher.example/v1",
                expected_model="teacher",
            )

    def test_reference_replay_retains_expected_failed_lookup_before_correction(self):
        actions = [
            SimpleNamespace(
                requestor="assistant",
                name="find_user_id_by_email",
                arguments={"email": "missing@example.com"},
            ),
            SimpleNamespace(
                requestor="assistant",
                name="find_user_id_by_email",
                arguments={"email": "present@example.com"},
            ),
        ]
        results = [
            (
                {"role": "assistant", "tool_calls": [{"name": actions[0].name}]},
                {"role": "tool", "error": True, "content": "User not found"},
            ),
            (
                {"role": "assistant", "tool_calls": [{"name": actions[1].name}]},
                {"role": "tool", "error": False, "content": "user-1"},
            ),
        ]

        with patch.object(generation, "execute_call", side_effect=results) as execute:
            observed = generation.execute_reference_actions(
                object(),
                task_id=35,
                actions=actions,
            )

        self.assertEqual(execute.call_count, 2)
        self.assertTrue(observed[1]["error"])
        self.assertFalse(observed[3]["error"])
        self.assertEqual(observed[1]["content"], "User not found")

    def test_all_reference_replay_paths_use_expected_error_aware_helper(self):
        for function in (
            generation.deterministic_reference_clean_simulation,
            generation.deterministic_reference_tail_simulation,
            generation.deterministic_reference_completion_simulation,
        ):
            self.assertIn(
                "execute_reference_actions",
                __import__("inspect").getsource(function),
            )

    def test_reference_replay_rejects_action_index_cardinality_drift(self):
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "cardinality mismatch",
        ):
            generation.execute_reference_actions(
                object(),
                task_id=35,
                actions=[SimpleNamespace()],
                action_indices=[],
            )

    def test_training_system_message_binds_domain_policy(self):
        message = generation.training_system_message("Always verify the user.")
        self.assertEqual(message["role"], "system")
        self.assertIn("<policy>\nAlways verify the user.\n</policy>", message["content"])
        with self.assertRaises(generation.V6GenerationError):
            generation.training_system_message("")

    def test_registry_hash_and_test_seal_are_verified(self):
        value = registry()
        self.assertEqual(
            generation.verify_registry(value), value["registry_sha256"]
        )
        changed = deepcopy(value)
        changed["official_test_used"] = True
        with self.assertRaisesRegex(generation.V6GenerationError, "test seal"):
            generation.verify_registry(changed)

    def test_generator_accepts_the_canonical_executable_v610_registry(self):
        from tests.test_v6_10_registry import _build

        payload = _build()
        self.assertEqual(
            generation.verify_registry(payload),
            payload["registry_sha256"],
        )

    def test_phase_sharding_keeps_all_pairs_for_a_task_together(self):
        value = registry()
        first = generation.phase_pairs(
            value,
            phase="pilot",
            shard_index=0,
            num_shards=2,
            smoke_task=None,
        )
        second = generation.phase_pairs(
            value,
            phase="pilot",
            shard_index=1,
            num_shards=2,
            smoke_task=None,
        )
        first_tasks = {row["task_identity"] for row in first}
        second_tasks = {row["task_identity"] for row in second}
        self.assertTrue(first_tasks)
        self.assertTrue(second_tasks)
        self.assertTrue(first_tasks.isdisjoint(second_tasks))
        self.assertEqual(
            {row["task_identity"] for row in first + second},
            {"retail:1", "airline:2"},
        )
        self.assertTrue(
            all(
                sum(
                    row["task_identity"] == task for row in first + second
                )
                == protocol.MIN_PAIRS_PER_TASK
                for task in {"retail:1", "airline:2"}
            )
        )

    def test_semantic_hash_ignores_only_transport_metadata(self):
        left = {
            "role": "assistant",
            "content": "done",
            "timestamp": "one",
            "usage": {"tokens": 1},
        }
        right = {
            "role": "assistant",
            "content": "done",
            "timestamp": "two",
            "usage": {"tokens": 999},
        }
        self.assertEqual(
            generation.semantic_sha256(left),
            generation.semantic_sha256(right),
        )
        right["content"] = "changed"
        self.assertNotEqual(
            generation.semantic_sha256(left),
            generation.semantic_sha256(right),
        )

    def test_fresh_suffix_rejects_future_or_history_drift(self):
        prompt = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "x"}],
            },
            {"role": "tool", "content": "error", "error": True},
        ]
        suffix = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "y"}],
            },
            {"role": "tool", "content": "ok", "error": False},
        ]
        self.assertEqual(
            generation.extract_fresh_suffix([*prompt, *suffix], prompt),
            suffix,
        )
        changed = deepcopy(prompt)
        changed[0]["content"] = "different"
        with self.assertRaisesRegex(
            generation.V6GenerationError, "history differs"
        ):
            generation.extract_fresh_suffix([*changed, *suffix], prompt)
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "suffix is empty",
        ):
            generation.extract_fresh_suffix(prompt, prompt)
        self.assertEqual(
            generation.extract_fresh_suffix(
                prompt,
                prompt,
                allow_empty=True,
            ),
            [],
        )

    def test_fresh_suffix_ignores_only_default_tau2_transport_expansion(self):
        compact = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"id": "wrong"},
                        "requestor": "assistant",
                    }
                ],
            }
        ]
        expanded = deepcopy(compact)
        expanded[0].update(
            deepcopy(generation.DEFAULT_TAU2_MESSAGE_TRANSPORT_FIELDS)
        )
        suffix = [{"role": "tool", "content": "Error", "error": True}]
        self.assertEqual(
            generation.extract_fresh_suffix(
                [*expanded, *suffix],
                compact,
            ),
            suffix,
        )
        for field, non_default in (
            ("is_audio", True),
            ("is_final_chunk", False),
            ("contains_speech", False),
            ("source", "microphone"),
        ):
            changed = deepcopy(expanded)
            changed[0][field] = non_default
            with self.subTest(field=field), self.assertRaisesRegex(
                generation.V6GenerationError,
                "history differs",
            ):
                generation.extract_fresh_suffix(
                    [*changed, *suffix],
                    compact,
                )

    def test_fresh_teacher_causal_cell_preserves_empty_continuation(self):
        prompt = [{"role": "tool", "content": "injected error", "error": True}]
        forced_call = {
            "id": "corrective",
            "requestor": "assistant",
            "name": "lookup",
            "arguments": {"id": "right"},
        }
        call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [deepcopy(forced_call)],
        }
        tool_result = {
            "role": "tool",
            "content": "ok",
            "error": False,
        }
        forced_history = [*prompt, call_message, tool_result]
        reference_plan = {
            "reference_slot_ids_by_index": {"0": "slot-0"},
            "task_preflight_sha256": digest("task-preflight"),
            "sanitized_successful_plan_sha256": digest("plan"),
            "reference_preflight_receipt_sha256": digest("receipt"),
            "reference_preflight_file_sha256": digest("receipt-file"),
        }
        replay = {"pass": True}
        with (
            patch.object(generation, "initial_environment", return_value=object()),
            patch.object(
                generation,
                "execute_call",
                return_value=(deepcopy(call_message), deepcopy(tool_result)),
            ),
            patch.object(generation, "make_recovery_task", return_value=object()),
            patch.object(
                generation,
                "run_one",
                return_value=(object(), 0.5),
            ),
            patch.object(
                generation,
                "dynamic_official_reward",
                return_value=(0.0, {"environment": 0.0}),
            ),
            patch.object(
                generation,
                "messages",
                return_value=deepcopy(forced_history),
            ),
            patch.object(
                generation,
                "independent_replay",
                return_value=replay,
            ),
        ):
            cell = generation.forced_first_cell(
                generation_args(
                    causal_cell_continuation_mode="fresh_teacher"
                ),
                task=object(),
                domain="retail",
                error_prompt=prompt,
                forced_call=forced_call,
                forced_reference_action_index=0,
                continuation_seeds=(123,),
                log_root=Path("/tmp/empty-causal-cell"),
                cell_name="matched",
                reference_plan=reference_plan,
            )
        self.assertEqual(cell["task_success"], 0.0)
        self.assertEqual(cell["trials"][0]["continuation_messages"], [])
        self.assertEqual(
            cell["trials"][0]["official_reward_info"],
            {"environment": 0.0},
        )
        self.assertEqual(cell["trials"][0]["independent_replay"], replay)

    def test_failed_calls_never_enter_success_label_mask(self):
        values = [
            {"role": "assistant", "tool_calls": [{"name": "bad"}]},
            {"role": "tool", "error": True},
            {"role": "assistant", "tool_calls": [{"name": "good"}]},
            {"role": "tool", "error": False},
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(
            generation.successful_assistant_labels(values),
            [False, False, True, False, False],
        )

    def test_absent_user_database_is_explicit_null_not_a_fake_hash(self):
        base = {
            "agent_db_hash_before_error": "agent-state",
            "agent_db_hash_after_error": "agent-state",
            "user_db_hash_available": False,
            "user_db_hash_before_error": None,
            "user_db_hash_after_error": None,
        }
        self.assertTrue(
            protocol.failed_injection_databases_unchanged(base)
        )
        for field in (
            "user_db_hash_available",
            "user_db_hash_before_error",
            "user_db_hash_after_error",
        ):
            missing = deepcopy(base)
            missing.pop(field)
            with self.subTest(missing=field):
                self.assertFalse(
                    protocol.failed_injection_databases_unchanged(missing)
                )
        fabricated = deepcopy(base)
        fabricated["user_db_hash_before_error"] = "unavailable"
        fabricated["user_db_hash_after_error"] = "unavailable"
        self.assertFalse(
            protocol.failed_injection_databases_unchanged(fabricated)
        )
        available = deepcopy(base)
        available.update(
            {
                "user_db_hash_available": True,
                "user_db_hash_before_error": "user-state",
                "user_db_hash_after_error": "user-state",
            }
        )
        self.assertTrue(
            protocol.failed_injection_databases_unchanged(available)
        )

    def test_seed_set_is_fixed_nonempty_and_unique(self):
        self.assertEqual(generation.parse_seed_set("1,2,3"), (1, 2, 3))
        with self.assertRaisesRegex(generation.V6GenerationError, "distinct"):
            generation.parse_seed_set("1,1")
        with self.assertRaisesRegex(generation.V6GenerationError, "non-empty"):
            generation.parse_seed_set("")

    def test_run_contract_binds_endpoints_tau2_limits_and_full_decoding(self):
        args = generation_args()
        seeds = (101, 202)
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=seeds,
        )
        contract = generation.build_run_contract(
            args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=digest("registry"),
            task_ids=("retail:1",),
            semantic_contract=semantic,
        )

        self.assertEqual(contract["tau2_commit"], protocol.TAU2_COMMIT)
        self.assertEqual(
            (
                contract["teacher_api_base"],
                contract["user_api_base"],
                contract["judge_api_base"],
            ),
            (
                args.teacher_api_base,
                args.user_api_base,
                args.judge_api_base,
            ),
        )
        self.assertEqual(contract["max_tokens"], args.max_tokens)
        self.assertEqual(contract["max_steps"], args.max_steps)
        self.assertEqual(
            contract["decoding"],
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "parallel_tool_calls": False,
                "applies_to": ["teacher", "user", "judge"],
                "assistant_tool_only_normalization": (
                    "execute_first_tool_call_then_replan"
                ),
            },
        )
        self.assertEqual(contract["runner"]["max_steps"], 60)
        self.assertEqual(contract["runner"]["timeout_seconds"], 900.0)
        self.assertEqual(contract["clean_attempts"], 2)
        self.assertEqual(contract["recovery_attempts"], 3)
        self.assertEqual(contract["continuation_seeds"], list(seeds))
        self.assertEqual(
            contract["semantic_generation_contract_sha256"],
            generation.sha256(semantic),
        )

        baseline = generation.sha256(semantic)
        for field, changed_value in (
            ("teacher_api_base", "http://other-teacher.example/v1"),
            ("user_api_base", "http://other-user.example/v1"),
            ("judge_api_base", "http://other-judge.example/v1"),
            ("max_tokens", 1024),
            ("max_steps", 61),
            ("timeout", 901.0),
            ("clean_attempts", 4),
            ("recovery_attempts", 5),
        ):
            changed_args = generation_args(**{field: changed_value})
            changed = generation.semantic_generation_contract(
                changed_args,
                continuation_seeds=seeds,
            )
            self.assertNotEqual(
                generation.sha256(changed),
                baseline,
                msg=f"{field} must be bound by the semantic contract",
            )
        changed_seeds = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101, 303),
        )
        self.assertNotEqual(generation.sha256(changed_seeds), baseline)

    def test_task_resume_receipt_rejects_changed_generation_contract(self):
        args = generation_args()
        registry_sha256 = digest("registry")
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101, 202),
        )
        run_contract = generation.build_run_contract(
            args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=registry_sha256,
            task_ids=("retail:1",),
            semantic_contract=semantic,
        )
        run_contract_sha256 = generation.sha256(run_contract)
        semantic_sha256 = generation.sha256(semantic)
        receipt = {
            "protocol": generation.GENERATION_PROTOCOL,
            "status": "PASS",
            "task_identity": "retail:1",
            "registry_sha256": registry_sha256,
            "run_contract_sha256": run_contract_sha256,
            "semantic_generation_contract": deepcopy(semantic),
            "semantic_generation_contract_sha256": semantic_sha256,
            "candidate_pair_count": 1,
            "candidate_pairs": [
                {
                    "task_identity": "retail:1",
                    "registry_sha256": registry_sha256,
                    "generation_contract": deepcopy(semantic),
                    "generation_contract_sha256": semantic_sha256,
                }
            ],
        }
        generation.validate_task_resume_receipt(
            receipt,
            task_identity="retail:1",
            registry_sha256=registry_sha256,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=semantic,
        )

        changed_args = generation_args(max_tokens=1024)
        changed_semantic = generation.semantic_generation_contract(
            changed_args,
            continuation_seeds=(101, 202),
        )
        changed_run_contract = generation.build_run_contract(
            changed_args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=registry_sha256,
            task_ids=("retail:1",),
            semantic_contract=changed_semantic,
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "run/semantic generation contract drift",
        ):
            generation.validate_task_resume_receipt(
                receipt,
                task_identity="retail:1",
                registry_sha256=registry_sha256,
                run_contract_sha256=generation.sha256(changed_run_contract),
                semantic_contract=changed_semantic,
            )

        tampered = deepcopy(receipt)
        tampered["candidate_pairs"][0]["generation_contract"][
            "teacher_api_base"
        ] = "http://stale-teacher.example/v1"
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "run/semantic generation contract drift",
        ):
            generation.validate_task_resume_receipt(
                tampered,
                task_identity="retail:1",
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
            )

    def test_v610_mode_separation_requires_gold_free_causal_teacher(self):
        args = generation_args(
            teacher_model=generation.V610_TEACHER_MODEL,
            teacher_revision=generation.V610_TEACHER_REVISION,
            user_model=generation.V610_USER_JUDGE_MODEL,
            user_revision=generation.V610_USER_JUDGE_REVISION,
            judge_model=generation.V610_USER_JUDGE_MODEL,
            judge_revision=generation.V610_USER_JUDGE_REVISION,
            judge_api_base="http://user.example/v1",
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        generation.validate_v610_modes(
            args,
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            continuation_seeds=generation.V610_SCIENTIFIC_SEEDS,
        )

        args.causal_cell_continuation_mode = (
            "deterministic_reference_completion"
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "mode separation",
        ):
            generation.validate_v610_modes(
                args,
                design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                continuation_seeds=generation.V610_SCIENTIFIC_SEEDS,
            )

    def test_v610_freezes_single_shard_until_cross_shard_merger_exists(self):
        args = generation_args(
            teacher_model=generation.V610_TEACHER_MODEL,
            teacher_revision=generation.V610_TEACHER_REVISION,
            user_model=generation.V610_USER_JUDGE_MODEL,
            user_revision=generation.V610_USER_JUDGE_REVISION,
            judge_model=generation.V610_USER_JUDGE_MODEL,
            judge_revision=generation.V610_USER_JUDGE_REVISION,
            judge_api_base="http://user.example/v1",
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
            num_shards=2,
            shard_index=0,
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "single_shard_execution",
        ):
            generation.validate_v610_modes(
                args,
                design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                continuation_seeds=generation.V610_SCIENTIFIC_SEEDS,
            )

        # Historical protocols keep their already-defined sharding behavior.
        generation.validate_v610_modes(
            args,
            design_protocol=protocol.PROTOCOL,
            continuation_seeds=(1,),
        )

    def test_v610_semantic_contract_binds_provenance_and_separate_roles(self):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            recovery_continuation_mode="fresh_teacher",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101, 202),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        self.assertEqual(
            semantic["matched_positive_continuation_mode"],
            "deterministic_reference_completion",
        )
        self.assertEqual(
            semantic["causal_cell_continuation_mode"], "fresh_teacher"
        )
        self.assertFalse(semantic["fresh_recovery_generated"])
        self.assertTrue(semantic["matched_positive_gold_suffix_used"])
        self.assertTrue(semantic["causal_cell_fresh_recovery_generated"])
        self.assertFalse(semantic["causal_cell_gold_suffix_visible"])
        self.assertEqual(semantic["runtime_provenance"], provenance)
        self.assertEqual(semantic["execution_provenance"], execution)

        run_contract = generation.build_run_contract(
            args,
            registry_file_sha256=provenance["parent_artifact_hashes"][
                "registry_file_sha256"
            ],
            registry_sha256="5" * 64,
            task_ids=("retail:1",),
            semantic_contract=semantic,
        )
        self.assertEqual(
            run_contract["attempt_id"], provenance["attempt_id"]
        )
        self.assertEqual(run_contract["runtime_provenance"], provenance)
        self.assertEqual(run_contract["exact_argv"], execution["exact_argv"])
        self.assertEqual(
            run_contract["parent_artifact_hashes"],
            provenance["parent_artifact_hashes"],
        )
        self.assertEqual(
            run_contract["task_file_sha256"],
            provenance["task_file_sha256"],
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "registry file hash differs",
        ):
            generation.build_run_contract(
                args,
                registry_file_sha256="4" * 64,
                registry_sha256="5" * 64,
                task_ids=("retail:1",),
                semantic_contract=semantic,
            )
        drifted_provenance = deepcopy(provenance)
        drifted_provenance["release_manifest"]["identities"][
            "registry_file_sha256"
        ] = digest("unrelated-registry")
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "receipt provenance drift",
        ):
            generation.semantic_generation_contract(
                args,
                continuation_seeds=(101, 202),
                design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                runtime_provenance=drifted_provenance,
                execution_provenance=execution,
            )

    def test_v610_task_and_generation_receipts_close_timing_and_resume(self):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        registry_sha256 = digest("registry")
        run_contract_sha256 = digest("run-contract")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            tasks = output_dir / "tasks"
            tasks.mkdir()
            receipt_path = tasks / "retail_35.json"
            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-30T00:01:00Z",
            ):
                attempt = generation.begin_v610_task_attempt(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=semantic,
                    process_execution_id=digest("process-1"),
                    process_started_at_utc="2026-07-30T00:00:30Z",
                    receipt_path=receipt_path,
                )
            rejected = generation.rejected_task_receipt(
                task_identity="retail:35",
                reason_code="REFERENCE_PREFLIGHT_TASK_REJECTED",
                error=generation.V6GenerationError(
                    "no verified pass reference plan"
                ),
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
                expected_candidate_pair_ids=(),
                task_started_at_utc=attempt["task_started_at_utc"],
                task_ended_at_utc="2026-07-30T00:02:00Z",
            )
            rejected = generation.finalize_v610_task_attempt(
                output_dir,
                task_identity="retail:35",
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
                attempt_context=attempt,
                receipt_path=receipt_path,
                receipt=rejected,
            )
            self.assertEqual(
                rejected["task_file"],
                {
                    "domain": "retail",
                    "relative_path": "data/tau2/domains/retail/tasks.json",
                    "sha256": provenance["task_file_sha256"]["retail"],
                },
            )
            generation.validate_rejected_task_resume_receipt(
                rejected,
                task_identity="retail:35",
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
                expected_candidate_pair_ids=(),
                output_dir=output_dir,
            )
            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-30T00:10:00Z",
            ):
                first = generation.merge_shard(
                    output_dir,
                    ("retail:35",),
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=semantic,
                    expected_pair_ids_by_task={"retail:35": ()},
                    phase="pilot",
                )
            self.assertEqual(
                first["run_started_at_utc"],
                "2026-07-30T00:00:00Z",
            )
            self.assertEqual(
                first["run_ended_at_utc"],
                "2026-07-30T00:10:00Z",
            )
            self.assertEqual(
                first["generation_receipt_sha256"],
                generation.sha256(
                    {
                        key: value
                        for key, value in first.items()
                        if key != "generation_receipt_sha256"
                    }
                ),
            )

            # A later process must reuse the immutable run start and terminal
            # merge end instead of creating contract/receipt drift.
            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-31T00:00:00Z",
            ):
                resumed = generation.merge_shard(
                    output_dir,
                    ("retail:35",),
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=semantic,
                    expected_pair_ids_by_task={"retail:35": ()},
                    phase="pilot",
                )
            self.assertEqual(resumed, first)

            tampered = deepcopy(rejected)
            tampered["task_ended_at_utc"] = "2026-07-29T23:59:59Z"
            tampered["task_receipt_sha256"] = generation.sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "task_receipt_sha256"
                }
            )
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "stale/invalid",
            ):
                generation.validate_rejected_task_resume_receipt(
                    tampered,
                    task_identity="retail:35",
                    registry_sha256=registry_sha256,
                    run_contract_sha256=run_contract_sha256,
                    semantic_contract=semantic,
                    expected_candidate_pair_ids=(),
                    output_dir=output_dir,
                )

    def test_v610_crash_restart_quarantines_evidence_and_refuses_third_restart(
        self,
    ):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        run_hash = digest("run-contract")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            receipt_path = output_dir / "tasks" / "retail_35.json"
            attempts = []
            for number in range(1, 4):
                with patch.object(
                    generation,
                    "utc_now",
                    return_value=f"2026-07-30T00:0{number}:00Z",
                ):
                    attempt = generation.begin_v610_task_attempt(
                        output_dir,
                        task_identity="retail:35",
                        run_contract_sha256=run_hash,
                        semantic_contract=semantic,
                        process_execution_id=digest(f"process-{number}"),
                        process_started_at_utc=(
                            f"2026-07-30T00:0{number - 1}:30Z"
                        ),
                        receipt_path=receipt_path,
                    )
                (attempt["evidence_dir"] / f"attempt-{number}.log").write_text(
                    f"evidence-{number}\n",
                    encoding="utf-8",
                )
                attempts.append(attempt)

            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-30T00:04:00Z",
            ):
                with self.assertRaisesRegex(
                    generation.V6GenerationError,
                    "refusing a third process-level task restart",
                ):
                    generation.begin_v610_task_attempt(
                        output_dir,
                        task_identity="retail:35",
                        run_contract_sha256=run_hash,
                        semantic_contract=semantic,
                        process_execution_id=digest("process-4"),
                        process_started_at_utc="2026-07-30T00:03:30Z",
                        receipt_path=receipt_path,
                    )
            quarantined = list(
                (output_dir / "quarantine" / "retail_35").glob(
                    "attempt-*"
                )
            )
            self.assertEqual(len(quarantined), 3)
            preserved = sorted(
                path.read_text(encoding="utf-8").strip()
                for root in quarantined
                for path in (root / "evidence").glob("*.log")
            )
            self.assertEqual(
                preserved,
                ["evidence-1", "evidence-2", "evidence-3"],
            )

    def test_v610_output_lock_rejects_concurrent_writer(self):
        if generation.fcntl is None:
            self.skipTest("fcntl is unavailable")
        generation._release_v610_run_locks()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            generation.acquire_v610_run_lock(
                output_dir,
                process_execution_id=digest("writer-one"),
                process_started_at_utc="2026-07-30T00:00:00Z",
            )
            try:
                with self.assertRaisesRegex(
                    generation.V6GenerationError,
                    "active writer",
                ):
                    generation.acquire_v610_run_lock(
                        output_dir,
                        process_execution_id=digest("writer-two"),
                        process_started_at_utc="2026-07-30T00:01:00Z",
                    )
            finally:
                generation._release_v610_run_locks()

    def test_v610_rejects_symlinked_task_staging_root(self):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            staging_parent = output_dir / "task_staging"
            staging_parent.mkdir()
            external = output_dir / "external-staging"
            external.mkdir()
            (staging_parent / "retail_35").symlink_to(
                external,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "symlink",
            ):
                generation.reconcile_v610_task_index(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=digest("run"),
                    semantic_contract=semantic,
                    receipt_path=output_dir / "tasks" / "retail_35.json",
                )

    def test_v610_scientific_rejection_is_completed_and_resumable(self):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        registry_hash = digest("registry")
        run_hash = digest("run-contract")
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            receipt_path = output_dir / "tasks" / "retail_35.json"
            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-30T00:01:00Z",
            ):
                first = generation.begin_v610_task_attempt(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    process_execution_id=digest("process-1"),
                    process_started_at_utc="2026-07-30T00:00:30Z",
                    receipt_path=receipt_path,
                )
            rejected = generation.rejected_task_receipt(
                task_identity="retail:35",
                reason_code="REFERENCE_PREFLIGHT_TASK_REJECTED",
                error=generation.V6GenerationError(
                    "no verified pass reference plan"
                ),
                registry_sha256=registry_hash,
                run_contract_sha256=run_hash,
                semantic_contract=semantic,
                expected_candidate_pair_ids=(),
                task_started_at_utc=first["task_started_at_utc"],
                task_ended_at_utc="2026-07-30T00:02:00Z",
            )
            rejected = generation.finalize_v610_task_attempt(
                output_dir,
                task_identity="retail:35",
                run_contract_sha256=run_hash,
                semantic_contract=semantic,
                attempt_context=first,
                receipt_path=receipt_path,
                receipt=rejected,
            )
            generation.validate_rejected_task_resume_receipt(
                rejected,
                task_identity="retail:35",
                registry_sha256=registry_hash,
                run_contract_sha256=run_hash,
                semantic_contract=semantic,
                expected_candidate_pair_ids=(),
                output_dir=output_dir,
            )
            self.assertEqual(rejected["execution_status"], "PASS")
            self.assertEqual(rejected["scientific_outcome"], "REJECTED")
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "must be resumed",
            ):
                generation.begin_v610_task_attempt(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    process_execution_id=digest("process-2"),
                    process_started_at_utc="2026-07-30T00:02:30Z",
                    receipt_path=receipt_path,
                )
            self.assertEqual(
                len(
                    list(
                    (output_dir / "task_bundles" / "retail_35").glob(
                        "attempt-*"
                    )
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    list(
                        (output_dir / "quarantine" / "retail_35").glob(
                            "attempt-01-*"
                        )
                    )
                ),
                0,
            )

    def test_v610_atomic_pass_bundle_is_resumable_and_tamper_evident(self):
        args = generation_args(
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        provenance, execution = v610_provenance()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            runtime_provenance=provenance,
            execution_provenance=execution,
        )
        registry_hash = digest("registry")
        run_hash = digest("run-contract")
        pair_id = "v6:pilot:retail:35:pair:1"
        pair_row = {
            "protocol": generation.GENERATION_PROTOCOL,
            "candidate_pair_id": pair_id,
            "task_identity": "retail:35",
            "domain": "retail",
            "registry_sha256": registry_hash,
            "generation_contract": deepcopy(semantic),
            "generation_contract_sha256": generation.sha256(semantic),
            "branches": [],
            "official_test_used": False,
        }
        pair_row["generated_candidate_pair_sha256"] = generation.sha256(
            pair_row
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            receipt_path = output_dir / "tasks" / "retail_35.json"
            with patch.object(
                generation,
                "utc_now",
                return_value="2026-07-30T00:01:00Z",
            ):
                attempt = generation.begin_v610_task_attempt(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    process_execution_id=digest("process-pass"),
                    process_started_at_utc="2026-07-30T00:00:30Z",
                    receipt_path=receipt_path,
                )
            evidence_file = attempt["evidence_dir"] / "generation.log"
            evidence_file.write_text("complete\n", encoding="utf-8")
            receipt = {
                "protocol": generation.GENERATION_PROTOCOL,
                "status": "PASS",
                "task_identity": "retail:35",
                "domain": "retail",
                "registry_sha256": registry_hash,
                "run_contract_sha256": run_hash,
                "semantic_generation_contract": deepcopy(semantic),
                "semantic_generation_contract_sha256": generation.sha256(
                    semantic
                ),
                "runtime_provenance": deepcopy(
                    semantic["runtime_provenance"]
                ),
                "attempt_id": semantic["runtime_provenance"]["attempt_id"],
                "candidate_pair_count": 1,
                "candidate_pairs": [pair_row],
                "all_pairs_share_prefix": True,
                "all_pairs_share_environment_snapshot": True,
                "execution_status": "PASS",
                "scientific_outcome": "ACCEPTED",
                "official_test_used": False,
            }
            receipt.update(
                generation.v610_task_receipt_fields(
                    semantic,
                    task_identity="retail:35",
                    task_started_at_utc=attempt["task_started_at_utc"],
                    task_ended_at_utc="2026-07-30T00:02:00Z",
                )
            )
            receipt["task_receipt_sha256"] = generation.sha256(receipt)
            receipt = generation.finalize_v610_task_attempt(
                output_dir,
                task_identity="retail:35",
                run_contract_sha256=run_hash,
                semantic_contract=semantic,
                attempt_context=attempt,
                receipt_path=receipt_path,
                receipt=receipt,
            )
            generation.validate_task_resume_receipt(
                receipt,
                task_identity="retail:35",
                registry_sha256=registry_hash,
                run_contract_sha256=run_hash,
                semantic_contract=semantic,
                expected_candidate_pair_ids=(pair_id,),
                output_dir=output_dir,
            )
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "must be resumed",
            ):
                generation.begin_v610_task_attempt(
                    output_dir,
                    task_identity="retail:35",
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    process_execution_id=digest("process-other"),
                    process_started_at_utc="2026-07-30T00:02:30Z",
                    receipt_path=receipt_path,
                )
            bundle = (
                output_dir
                / receipt["task_execution"]["bundle_relative_path"]
            )
            external = output_dir / "external-identical.log"
            external.write_text("complete\n", encoding="utf-8")
            extra_link = bundle / "unregistered-link"
            extra_link.symlink_to(external)
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "valid PASS resume",
            ):
                generation.validate_task_resume_receipt(
                    receipt,
                    task_identity="retail:35",
                    registry_sha256=registry_hash,
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    expected_candidate_pair_ids=(pair_id,),
                    output_dir=output_dir,
                )
            extra_link.unlink()
            evidence_file = (
                output_dir
                / receipt["task_execution"]["evidence_relative_path"]
                / "generation.log"
            )
            evidence_file.unlink()
            evidence_file.symlink_to(external)
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "valid PASS resume",
            ):
                generation.validate_task_resume_receipt(
                    receipt,
                    task_identity="retail:35",
                    registry_sha256=registry_hash,
                    run_contract_sha256=run_hash,
                    semantic_contract=semantic,
                    expected_candidate_pair_ids=(pair_id,),
                    output_dir=output_dir,
                )

    def test_v610_freezes_models_limits_and_phase_seed_sets(self):
        args = generation_args(
            teacher_model=generation.V610_TEACHER_MODEL,
            teacher_revision=generation.V610_TEACHER_REVISION,
            user_model=generation.V610_USER_JUDGE_MODEL,
            user_revision=generation.V610_USER_JUDGE_REVISION,
            judge_model=generation.V610_USER_JUDGE_MODEL,
            judge_revision=generation.V610_USER_JUDGE_REVISION,
            judge_api_base="http://user.example/v1",
            clean_agent_mode="single_turn_user_reference_replay",
            matched_positive_continuation_mode=(
                "deterministic_reference_completion"
            ),
            causal_cell_continuation_mode="fresh_teacher",
            first_action_measurement_mode="teacher_unforced",
            completion_renderer="explicit_user_direct_v3",
        )
        generation.validate_v610_modes(
            args,
            design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
            continuation_seeds=generation.V610_SCIENTIFIC_SEEDS,
        )
        args.teacher_model = "wrong"
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "mode separation",
        ):
            generation.validate_v610_modes(
                args,
                design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                continuation_seeds=generation.V610_SCIENTIFIC_SEEDS,
            )
        args.teacher_model = generation.V610_TEACHER_MODEL
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "mode separation",
        ):
            generation.validate_v610_modes(
                args,
                design_protocol=generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                continuation_seeds=(20260806,),
            )

    def test_v610_resume_requires_exact_pair_set_and_untampered_hash(self):
        args = generation_args()
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101,),
            runtime_provenance={"attempt_id": "attempt-1"},
        )
        registry_sha256 = digest("registry")
        run_contract_sha256 = digest("run")
        pair_row = {
            "protocol": generation.GENERATION_PROTOCOL,
            "candidate_pair_id": "pair-1",
            "task_identity": "retail:1",
            "registry_sha256": registry_sha256,
            "generation_contract": deepcopy(semantic),
            "generation_contract_sha256": generation.sha256(semantic),
            "official_test_used": False,
        }
        pair_row["generated_candidate_pair_sha256"] = generation.sha256(
            pair_row
        )
        receipt = {
            "protocol": generation.GENERATION_PROTOCOL,
            "status": "PASS",
            "task_identity": "retail:1",
            "registry_sha256": registry_sha256,
            "run_contract_sha256": run_contract_sha256,
            "semantic_generation_contract": deepcopy(semantic),
            "semantic_generation_contract_sha256": generation.sha256(
                semantic
            ),
            "runtime_provenance": {"attempt_id": "attempt-1"},
            "attempt_id": "attempt-1",
            "candidate_pair_count": 1,
            "candidate_pairs": [pair_row],
        }
        generation.validate_task_resume_receipt(
            receipt,
            task_identity="retail:1",
            registry_sha256=registry_sha256,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=semantic,
            expected_candidate_pair_ids=("pair-1",),
        )

        stale = deepcopy(receipt)
        stale["candidate_pairs"][0]["candidate_pair_id"] = "other"
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "valid PASS resume",
        ):
            generation.validate_task_resume_receipt(
                stale,
                task_identity="retail:1",
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
                expected_candidate_pair_ids=("pair-1",),
            )

    def test_tail_contract_requires_exact_registered_reference_index(self):
        import inspect

        signature = inspect.signature(
            generation.deterministic_reference_tail_simulation
        )
        self.assertIn("forced_reference_action_index", signature.parameters)
        source = inspect.getsource(
            generation.deterministic_reference_tail_simulation
        )
        self.assertIn("resolve_forced_reference_index", source)

    def test_sanitized_reference_plan_preserves_exact_source_indices(self):
        actions = [object(), object(), object(), object()]
        selected = generation.select_reference_action_plan(actions, [1, 3])
        self.assertEqual([index for index, _ in selected], [1, 3])
        self.assertIs(selected[0][1], actions[1])
        self.assertIs(selected[1][1], actions[3])

        for invalid in ([3, 1], [1, 1], [4], [-1], [True], []):
            with self.subTest(invalid=invalid):
                with self.assertRaises(generation.V6GenerationError):
                    generation.select_reference_action_plan(actions, invalid)

    def test_pair_rejects_corrective_slot_not_authorized_by_preflight(self):
        registered = pair("retail:1", 0)
        task_preflight_sha256 = "a" * 64
        sanitized_plan_sha256 = generation.sha256([0, 1, 2])
        receipt_sha256 = "b" * 64
        file_sha256 = "c" * 64
        registered[
            "reference_task_preflight_sha256"
        ] = task_preflight_sha256
        registered["reference_preflight_receipt_sha256"] = receipt_sha256
        registered["reference_preflight_file_sha256"] = file_sha256
        registered[
            "sanitized_reference_plan_sha256"
        ] = sanitized_plan_sha256
        slots_by_index = {
            str(index): {
                "reference_action_index": index,
                "reference_slot_id": f"retail:1:reference:{index:03d}",
                "reference_slot_sha256": f"{index + 1:x}" * 64,
                "call_semantics_sha256": f"{index + 4:x}" * 64,
            }
            for index in range(3)
        }
        for index, branch_slot in enumerate(registered["branches"]):
            corrective = branch_slot["corrective_action_spec"]
            corrective["reference_action_index"] = index
            branch_slot["reference_preflight_binding"] = {
                "reference_preflight_receipt_sha256": receipt_sha256,
                "reference_preflight_file_sha256": file_sha256,
                "reference_task_preflight_sha256": task_preflight_sha256,
                "sanitized_reference_plan_sha256": sanitized_plan_sha256,
                **slots_by_index[str(index)],
            }
        task = SimpleNamespace(
            evaluation_criteria=SimpleNamespace(
                actions=[object(), object(), object()]
            )
        )
        reference_plan = {
            "task_identity": "retail:1",
            "reference_preflight_receipt_sha256": receipt_sha256,
            "reference_preflight_file_sha256": file_sha256,
            "task_preflight_sha256": task_preflight_sha256,
            "sanitized_successful_plan_sha256": sanitized_plan_sha256,
            "reference_slots_by_index": slots_by_index,
            "reference_slot_ids_by_index": {
                str(index): f"retail:1:reference:{index:03d}"
                for index in range(3)
            },
            "sanitized_successful_reference_indices": [0, 1, 2],
            "eligible_forced_first_reference_indices": [0],
            "expected_error_indices": [],
        }
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "not authorized",
        ):
            generation.validate_pair_against_reference_plan(
                registered,
                reference_plan,
                task=task,
            )

    def test_v610_positive_paths_receive_sanitized_indices(self):
        import inspect

        clean_source = inspect.getsource(generation.clean_rollout)
        matched_source = inspect.getsource(generation.matched_recovery)
        materialize_source = inspect.getsource(generation.materialize_pair)
        self.assertIn(
            'reference_plan["sanitized_successful_reference_indices"]',
            clean_source,
        )
        self.assertIn(
            'reference_plan["sanitized_successful_reference_indices"]',
            matched_source,
        )
        self.assertIn("reject_failed_tools=reference_plan is not None", matched_source)
        self.assertIn(
            "reject_failed_tools=reference_plan is not None",
            materialize_source,
        )
        self.assertIn("causal_cell_mode(args) != \"fresh_teacher\"", materialize_source)

    def test_all_pairs_are_preflighted_before_any_teacher_materialization(self):
        rows = [
            {
                "candidate_pair_id": f"pair-{index}",
                "branches": [{"branch_id": f"pair-{index}:branch"}],
            }
            for index in range(2)
        ]
        with (
            patch.object(
                generation,
                "validate_pair_against_reference_plan",
                side_effect=[
                    None,
                    generation.V6GenerationError("later pair is invalid"),
                ],
            ) as validate,
            patch.object(
                generation,
                "validate_error_injection",
                return_value={},
            ),
            patch.object(generation, "materialize_pair") as materialize,
        ):
            with self.assertRaisesRegex(
                generation.V6GenerationError,
                "later pair",
            ):
                generation.materialize_task_pairs(
                    generation_args(),
                    registered_pairs=rows,
                    clean={"prefix": []},
                    task=object(),
                    domain="retail",
                    continuation_seeds=(1,),
                    task_log_root=Path("/tmp/preflight-before-teacher"),
                    registry_sha256="a" * 64,
                    semantic_contract={},
                    reference_plan={"task_identity": "retail:1"},
                )
        self.assertEqual(validate.call_count, 2)
        materialize.assert_not_called()

    def test_selected_phase_preflight_rejects_late_task_before_clean_rollout(self):
        def task_pairs(task_identity: str) -> list[dict]:
            return [
                {
                    "candidate_pair_id": f"{task_identity}:pair:{pair_index}",
                    "branches": [
                        {
                            "branch_id": (
                                f"{task_identity}:pair:{pair_index}:branch:"
                                f"{branch_index}"
                            )
                        }
                        for branch_index in range(2)
                    ],
                }
                for pair_index in range(3)
            ]

        by_task = {
            "retail:1": task_pairs("retail:1"),
            "retail:2": task_pairs("retail:2"),
        }
        plan = {
            "evaluation_prefix": [{"role": "user", "content": "hello"}],
            "evaluation_prefix_semantic_sha256": digest("static-prefix"),
            "task_preflight_sha256": digest("task-preflight"),
            "sanitized_successful_plan_sha256": digest("sanitized-plan"),
        }

        def injection(*, branch_slot, **unused):
            if branch_slot["branch_id"] == "retail:2:pair:2:branch:1":
                raise generation.V6GenerationError(
                    "late branch: injected call was not a real tool error"
                )
            return {
                "error_event_sha256": digest(branch_slot["branch_id"]),
                "actual_tool_error": True,
                "state_unchanged_after_error": True,
            }

        with (
            patch.object(
                generation,
                "verified_reference_plan_for_task",
                return_value=plan,
            ),
            patch.object(
                generation,
                "select_task",
                side_effect=lambda identity: ("retail", object()),
            ),
            patch.object(
                generation,
                "validate_pair_against_reference_plan",
            ) as validate_pair,
            patch.object(
                generation,
                "validate_error_injection",
                side_effect=injection,
            ) as validate_injection,
            patch.object(generation, "clean_rollout") as clean_rollout,
        ):
            outcomes, report = generation.preflight_selected_phase_tasks(
                registered_pairs_by_task=by_task,
                reference_preflight={},
                preflight_binding={},
                phase="pilot",
            )
        self.assertEqual(outcomes["retail:1"]["status"], "PASS")
        self.assertEqual(outcomes["retail:2"]["status"], "REJECTED")
        self.assertEqual(
            outcomes["retail:2"]["reason_code"],
            "INJECTED_ERROR_INVALID",
        )
        self.assertEqual(report["classified_task_count"], 2)
        self.assertEqual(report["rejected_task_count"], 1)
        self.assertEqual(validate_pair.call_count, 6)
        self.assertEqual(validate_injection.call_count, 12)
        clean_rollout.assert_not_called()

    def test_unforced_teacher_probe_uses_one_registered_seed_without_gold(self):
        prompt = [{"role": "tool", "content": "error", "error": True}]
        first_call = {
            "id": "correct",
            "requestor": "assistant",
            "name": "lookup",
            "arguments": {"id": "right"},
        }
        with (
            patch.object(
                generation,
                "raw_teacher_first_response",
                return_value=(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [deepcopy(first_call)],
                    },
                    0.25,
                ),
            ) as raw,
            patch.object(generation, "write_json"),
        ):
            result = generation.teacher_unforced_first_action_measurement(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=prompt,
                expected_corrective_call=first_call,
                measurement_seed=777,
                log_root=__import__("pathlib").Path("/tmp/probe"),
            )
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(raw.call_args.kwargs["seed"], 777)
        self.assertEqual(result["trial_count"], 1)
        self.assertEqual(result["generation_count"], 1)
        self.assertEqual(
            result["matched_registered_corrective_accuracy"], 1.0
        )
        self.assertFalse(result["gold_suffix_visible"])
        self.assertFalse(result["fresh_recovery_generated"])
        self.assertFalse(result["normalization_applied"])
        self.assertEqual(
            result["task_success_status"],
            "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE",
        )

    def test_raw_probe_path_has_one_generation_and_no_runner_or_normalizer(self):
        import inspect

        source = inspect.getsource(generation.raw_teacher_first_response)
        self.assertEqual(source.count("agent.generate_next_message("), 1)
        self.assertIn('llm_args["num_retries"] = 0', source)
        self.assertNotIn("run_one(", source)
        self.assertNotIn("normalize_tool_only_message", source)
        measurement = inspect.getsource(
            generation.teacher_unforced_first_action_measurement
        )
        self.assertNotIn("dynamic_official_reward", measurement)
        self.assertNotIn("independent_replay", measurement)

    def test_raw_probe_passes_zero_provider_retries_to_tau2(self):
        captured: dict = {"generate_calls": 0}

        class FakeAgent:
            def __init__(self, *, tools, domain_policy, llm, llm_args):
                captured["llm_args"] = deepcopy(llm_args)

            def get_init_state(self, *, message_history):
                return object()

            def generate_next_message(self, message, state):
                captured["generate_calls"] += 1
                response = SimpleNamespace(
                    model_dump=lambda *, mode: {
                        "role": "assistant",
                        "content": "stop",
                        "tool_calls": None,
                    }
                )
                return response, state

        environment = SimpleNamespace(
            get_tools=lambda: [],
            get_policy=lambda: "policy",
        )
        with (
            patch(
                "tau2.agent.llm_agent.LLMAgent",
                FakeAgent,
            ),
            patch.object(
                generation,
                "initial_environment",
                return_value=environment,
            ),
            patch.object(
                generation,
                "parse_messages",
                return_value=[object()],
            ),
        ):
            raw, seconds = generation.raw_teacher_first_response(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=[{"role": "tool", "content": "error"}],
                seed=777,
            )
        self.assertEqual(captured["generate_calls"], 1)
        self.assertEqual(captured["llm_args"]["num_retries"], 0)
        self.assertEqual(raw["content"], "stop")
        self.assertGreaterEqual(seconds, 0.0)

    def test_raw_probe_records_malformed_json_as_incorrect_evidence(self):
        class MalformedAgent:
            def __init__(self, **unused):
                pass

            def get_init_state(self, *, message_history):
                return object()

            def generate_next_message(self, message, state):
                raise json.JSONDecodeError(
                    "invalid tool arguments",
                    '{"id":',
                    6,
                )

        environment = SimpleNamespace(
            get_tools=lambda: [],
            get_policy=lambda: "policy",
        )
        with (
            patch(
                "tau2.agent.llm_agent.LLMAgent",
                MalformedAgent,
            ),
            patch.object(
                generation,
                "initial_environment",
                return_value=environment,
            ),
            patch.object(
                generation,
                "parse_messages",
                return_value=[object()],
            ),
            patch.object(generation, "write_json"),
        ):
            result = generation.teacher_unforced_first_action_measurement(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=[{"role": "tool", "content": "error"}],
                expected_corrective_call={
                    "id": "correct",
                    "requestor": "assistant",
                    "name": "lookup",
                    "arguments": {"id": "right"},
                },
                measurement_seed=777,
                log_root=Path("/tmp/probe-malformed-json"),
            )
        trial = result["trials"][0]
        evidence = trial["raw_response"]["malformed_response_evidence"]
        self.assertEqual(result["matched_registered_corrective_accuracy"], 0.0)
        self.assertEqual(trial["first_response_status"], "INCORRECT_OR_MALFORMED")
        self.assertEqual(evidence["kind"], "MALFORMED_MODEL_RESPONSE")
        self.assertEqual(evidence["exception_type"], "JSONDecodeError")
        self.assertEqual(evidence["raw_json"], '{"id":')

    def test_raw_probe_transport_failure_remains_typed_global_failure(self):
        class FailingAgent:
            def __init__(self, **unused):
                pass

            def get_init_state(self, *, message_history):
                return object()

            def generate_next_message(self, message, state):
                raise RuntimeError("endpoint disconnected")

        environment = SimpleNamespace(
            get_tools=lambda: [],
            get_policy=lambda: "policy",
        )
        with (
            patch(
                "tau2.agent.llm_agent.LLMAgent",
                FailingAgent,
            ),
            patch.object(
                generation,
                "initial_environment",
                return_value=environment,
            ),
            patch.object(
                generation,
                "parse_messages",
                return_value=[object()],
            ),
            self.assertRaises(generation.V6RawProbeInfrastructureError),
        ):
            generation.raw_teacher_first_response(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=[{"role": "tool", "content": "error"}],
                seed=777,
            )

    def test_unforced_probe_rejects_raw_text_plus_tool_without_normalizing(self):
        prompt = [{"role": "tool", "content": "error", "error": True}]
        expected = {
            "id": "correct",
            "requestor": "assistant",
            "name": "lookup",
            "arguments": {"id": "right"},
        }
        with (
            patch.object(
                generation,
                "raw_teacher_first_response",
                return_value=(
                    {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [deepcopy(expected)],
                    },
                    0.1,
                ),
            ),
            patch.object(generation, "write_json"),
        ):
            result = generation.teacher_unforced_first_action_measurement(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=prompt,
                expected_corrective_call=expected,
                measurement_seed=777,
                log_root=Path("/tmp/probe-text-first"),
            )
        self.assertEqual(
            result["matched_registered_corrective_accuracy"],
            0.0,
        )
        self.assertEqual(
            result["trials"][0]["malformed_reason"],
            "RAW_FIRST_RESPONSE_NOT_ONE_TOOL_ONLY_CALL",
        )
        self.assertFalse(result["normalization_applied"])
        self.assertFalse(result["trials"][0]["judge_invoked"])
        self.assertFalse(result["trials"][0]["full_rollout_generated"])

    def test_unforced_probe_counts_raw_multi_call_as_incorrect(self):
        prompt = [{"role": "tool", "content": "error", "error": True}]
        expected = {
            "id": "correct",
            "requestor": "assistant",
            "name": "lookup",
            "arguments": {"id": "right"},
        }
        with (
            patch.object(
                generation,
                "raw_teacher_first_response",
                return_value=(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            deepcopy(expected),
                            {**deepcopy(expected), "id": "second"},
                        ],
                    },
                    0.1,
                ),
            ),
            patch.object(generation, "write_json"),
        ):
            result = generation.teacher_unforced_first_action_measurement(
                generation_args(),
                task=object(),
                domain="retail",
                error_prompt=prompt,
                expected_corrective_call=expected,
                measurement_seed=777,
                log_root=Path("/tmp/probe-multi"),
            )
        self.assertEqual(
            result["matched_registered_corrective_accuracy"],
            0.0,
        )
        self.assertFalse(result["trials"][0]["first_action_observed"])


class V610TaskLocalRuntimeClassificationTests(unittest.TestCase):
    def test_context_window_exhaustion_is_task_local_but_other_400_is_not(self):
        context_error_type = type(
            "ContextWindowExceededError",
            (RuntimeError,),
            {"__module__": "litellm.exceptions"},
        )
        context_error = context_error_type(
            "'max_tokens' is too large: 512. This model's maximum context "
            "length is 8192 tokens and the request has 7999 input tokens."
        )
        self.assertEqual(
            generation.classify_task_local_rejection(context_error),
            "CONTEXT_WINDOW_EXCEEDED",
        )

        other_bad_request_type = type(
            "BadRequestError",
            (RuntimeError,),
            {"__module__": "litellm.exceptions"},
        )
        self.assertIsNone(
            generation.classify_task_local_rejection(
                other_bad_request_type("tool parser is disabled")
            )
        )

    def test_context_window_classifier_is_fail_closed_on_message_and_origin(self):
        same_name_wrong_origin = type(
            "ContextWindowExceededError",
            (RuntimeError,),
            {"__module__": "untrusted"},
        )
        self.assertIsNone(
            generation.classify_task_local_rejection(
                same_name_wrong_origin(
                    "maximum context length; max_tokens is too large"
                )
            )
        )

        correct_origin_wrong_message = type(
            "ContextWindowExceededError",
            (RuntimeError,),
            {"__module__": "litellm.exceptions"},
        )
        self.assertIsNone(
            generation.classify_task_local_rejection(
                correct_origin_wrong_message("upstream request failed")
            )
        )


if __name__ == "__main__":
    unittest.main()
