from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_v6_10_release_manifest as release
from scripts import build_v6_10_runtime_receipts as runtime
from scripts import run_v6_candidate_generation as generation


CONTAINER = "sha256:" + "a" * 64
SOURCE_COMMIT = "1" * 40
SOURCE_TREE = "2" * 40
TIMESTAMP = "2026-07-30T12:00:00Z"
PREFLIGHT_SEMANTIC_SHA = "3" * 64


def write_canonical(path: Path, value: dict) -> None:
    path.write_text(runtime.canonical(value), encoding="utf-8")


def frozen_hashes(repository: Path) -> dict[str, str]:
    return {
        "config_sha256": runtime.file_sha256(
            repository / release.CONFIG_PATH
        ),
        "preregistration_sha256": runtime.file_sha256(
            repository / release.PREREGISTRATION_PATH
        ),
        "split_manifest_sha256": runtime.file_sha256(
            repository / release.SPLIT_MANIFEST_PATH
        ),
    }


def model_role(
    role: str,
    *,
    api_base: str,
    gpu_uuids: list[str],
) -> dict:
    teacher = role == "teacher"
    return {
        "role": role,
        "model": (
            generation.V610_TEACHER_MODEL
            if teacher
            else generation.V610_USER_JUDGE_MODEL
        ),
        "resolved_revision": (
            generation.V610_TEACHER_REVISION
            if teacher
            else generation.V610_USER_JUDGE_REVISION
        ),
        "api_base": api_base,
        "quantization": generation.V610_MODEL_QUANTIZATION,
        "tensor_parallel_size": 2 if teacher else 1,
        "gpu_uuids": gpu_uuids,
    }


def create_artifacts(root: Path, repository: Path) -> dict[str, Path]:
    hashes = frozen_hashes(repository)
    preflight = {
        "source_commit": SOURCE_COMMIT,
        "tau2_commit": generation.protocol.TAU2_COMMIT,
        "config_sha256": hashes["config_sha256"],
        "split_manifest_sha256": hashes["split_manifest_sha256"],
        "receipt_sha256": PREFLIGHT_SEMANTIC_SHA,
        "official_test_used": False,
    }
    preflight_path = root / "reference_preflight_receipt.json"
    write_canonical(preflight_path, preflight)
    preflight_file_sha = runtime.file_sha256(preflight_path)

    registry = {
        "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "source": {
            "source_commit": SOURCE_COMMIT,
            "tau2_commit": generation.protocol.TAU2_COMMIT,
            "config_file_sha256": hashes["config_sha256"],
            "split_file_sha256": hashes["split_manifest_sha256"],
            "reference_preflight": {
                "receipt_sha256": PREFLIGHT_SEMANTIC_SHA,
                "file_sha256": preflight_file_sha,
            },
        },
        "reference_preflight_receipt_sha256": PREFLIGHT_SEMANTIC_SHA,
        "reference_preflight_file_sha256": preflight_file_sha,
        "registry_sha256": "4" * 64,
    }
    registry_path = root / "registry.runtime.json"
    write_canonical(registry_path, registry)

    source_payload = {
        "source_commit": SOURCE_COMMIT,
        "tau2_commit": generation.protocol.TAU2_COMMIT,
        "container_image_digest": CONTAINER,
        "gpus": [
            {"index": index, "uuid": f"GPU-{index}"}
            for index in range(3)
        ],
        "official_test_used": False,
    }
    source_path = root / runtime.SOURCE_RECEIPT_NAME
    write_canonical(source_path, source_payload)
    model_payload = {
        "container_image_digest": CONTAINER,
        "roles": {
            "teacher": model_role(
                "teacher",
                api_base="http://127.0.0.1:8101/v1",
                gpu_uuids=["GPU-0", "GPU-1"],
            ),
            "user": model_role(
                "user",
                api_base="http://127.0.0.1:8201/v1",
                gpu_uuids=["GPU-2"],
            ),
            "judge": model_role(
                "judge",
                api_base="http://127.0.0.1:8201/v1",
                gpu_uuids=["GPU-2"],
            ),
        },
        "official_test_used": False,
    }
    model_path = root / runtime.MODEL_RECEIPT_NAME
    write_canonical(model_path, model_payload)
    runtime_hashes = {
        "protocol": release.RUNTIME_HASH_PROTOCOL,
        "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "status": "PASS",
        "source_container_provenance": {
            "path": runtime.SOURCE_RECEIPT_NAME,
            "file_sha256": runtime.file_sha256(source_path),
            "semantic_sha256": generation.sha256(source_payload),
        },
        "model_server_receipts": {
            "path": runtime.MODEL_RECEIPT_NAME,
            "file_sha256": runtime.file_sha256(model_path),
            "semantic_sha256": generation.sha256(model_payload),
        },
        "created_at_utc": TIMESTAMP,
        "official_test_used": False,
    }
    runtime_hashes_path = root / runtime.HASH_MANIFEST_NAME
    write_canonical(runtime_hashes_path, runtime_hashes)
    return {
        "preflight": preflight_path,
        "registry": registry_path,
        "source": source_path,
        "model": model_path,
        "runtime_hashes": runtime_hashes_path,
    }


def arguments(
    *,
    repository: Path,
    tau2: Path,
    artifacts: dict[str, Path],
    output: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=repository,
        tau2_root=tau2,
        container_image_digest=CONTAINER,
        reference_preflight=artifacts["preflight"],
        registry=artifacts["registry"],
        source_container_provenance=artifacts["source"],
        model_server_receipts=artifacts["model"],
        runtime_receipt_hashes=artifacts["runtime_hashes"],
        output=output,
    )


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[1]
        self.release_loader_calls = 0
        self.committed_source_paths: set[str] = set()

    def patched_dependencies(self, tau2: Path):
        def git_identity(path):
            if path.resolve() == tau2.resolve():
                return {
                    "commit": generation.protocol.TAU2_COMMIT,
                    "tree": "5" * 40,
                    "tracked_worktree_clean": True,
                }
            return {
                "commit": SOURCE_COMMIT,
                "tree": SOURCE_TREE,
                "tracked_worktree_clean": True,
            }

        def reference_loader(
            path,
            *,
            expected_sha256,
            expected_source_commit,
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                expected_sha256,
                runtime.file_sha256(path),
            )
            self.assertEqual(expected_source_commit, SOURCE_COMMIT)
            return payload, {"receipt_sha256": PREFLIGHT_SEMANTIC_SHA}

        def source_loader(
            path,
            *,
            expected_sha256,
            expected_source_commit,
            expected_generation_script_sha256,
            expected_tau2_commit,
            container_image_digest,
        ):
            self.assertEqual(expected_sha256, runtime.file_sha256(path))
            self.assertEqual(expected_source_commit, SOURCE_COMMIT)
            self.assertEqual(
                expected_tau2_commit,
                generation.protocol.TAU2_COMMIT,
            )
            self.assertEqual(container_image_digest, CONTAINER)
            self.assertRegex(expected_generation_script_sha256, r"^[0-9a-f]{64}$")
            return {}, {
                "gpus": [
                    {"uuid": f"GPU-{index}"} for index in range(3)
                ],
                "runtime": {
                    "python_executable_path": "/frozen/python",
                    "python_executable_sha256": "f" * 64,
                },
            }

        def committed_source_loader(
            *,
            source_root,
            source_commit,
            relative,
            expected_sha256,
            label,
        ):
            self.assertEqual(Path(source_root).resolve(), self.repository)
            self.assertEqual(source_commit, SOURCE_COMMIT)
            self.assertEqual(
                expected_sha256,
                runtime.file_sha256(self.repository / relative),
            )
            self.assertEqual(label, "release-builder frozen source")
            self.committed_source_paths.add(relative)

        def release_loader(
            path,
            *,
            expected_sha256,
            source_root,
            expected_fields,
        ):
            self.release_loader_calls += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(expected_sha256, runtime.file_sha256(path))
            self.assertEqual(Path(source_root).resolve(), self.repository)
            self.assertEqual(
                expected_fields,
                release.release_validator_expected(payload),
            )
            return payload, {
                "identities": dict(expected_fields),
                "relevant_scripts_sha256": generation.sha256(
                    payload["relevant_scripts"]
                ),
            }

        return (
            patch.object(
                release.runtime,
                "git_identity",
                side_effect=git_identity,
            ),
            patch.object(
                generation,
                "load_reference_preflight",
                side_effect=reference_loader,
            ),
            patch.object(
                generation,
                "verify_registry",
                return_value="4" * 64,
            ),
            patch.object(
                generation,
                "load_source_container_provenance",
                side_effect=source_loader,
            ),
            patch.object(
                generation,
                "load_model_server_receipts",
                return_value=({}, {}),
            ),
            patch.object(
                release.runtime,
                "utc_now",
                return_value=TIMESTAMP,
            ),
            patch.object(
                generation,
                "verify_committed_source_file",
                side_effect=committed_source_loader,
            ),
            patch.object(
                generation,
                "load_release_manifest",
                side_effect=release_loader,
            ),
        )

    def test_builder_binds_exact_release_graph_and_self_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            artifacts = create_artifacts(root, self.repository)
            output = root / "release_manifest.json"
            args = arguments(
                repository=self.repository,
                tau2=tau2,
                artifacts=artifacts,
                output=output,
            )
            patches = self.patched_dependencies(tau2)
            with ExitStack() as stack:
                for dependency_patch in patches:
                    stack.enter_context(dependency_patch)
                summary = release.build_manifest(args)
            self.assertEqual(summary["status"], "PASS")
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                runtime.canonical(manifest),
            )
            self.assertEqual(manifest["source_commit"], SOURCE_COMMIT)
            self.assertEqual(manifest["source_tree"], SOURCE_TREE)
            self.assertEqual(
                set(manifest["relevant_scripts"]),
                set(release.RELEVANT_SCRIPTS),
            )
            for relative, digest in manifest["relevant_scripts"].items():
                self.assertEqual(
                    digest,
                    runtime.file_sha256(self.repository / relative),
                )
            self.assertEqual(
                manifest["reference_preflight_receipt_sha256"],
                runtime.file_sha256(artifacts["preflight"]),
            )
            self.assertEqual(
                manifest["runtime_receipt_hashes_sha256"],
                runtime.file_sha256(artifacts["runtime_hashes"]),
            )
            self.assertEqual(self.release_loader_calls, 1)
            self.assertEqual(
                self.committed_source_paths,
                {
                    release.CONFIG_PATH.as_posix(),
                    release.PREREGISTRATION_PATH.as_posix(),
                    *release.RELEVANT_SCRIPTS,
                },
            )
            self.assertEqual(
                release.release_validator_expected(manifest)[
                    "source_commit"
                ],
                SOURCE_COMMIT,
            )
            self.assertRegex(
                generation.sha256(manifest["relevant_scripts"]),
                r"^[0-9a-f]{64}$",
            )

    def test_restarted_service_uses_new_session_and_reuses_static_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            sessions = root / "runtime-sessions"
            session_a = sessions / "v6_10-runtime-a01-20260730T120000Z"
            session_b = sessions / "v6_10-runtime-a02-20260730T130000Z"
            session_a.mkdir(parents=True)
            session_b.mkdir(parents=True)
            artifacts_a = create_artifacts(session_a, self.repository)
            artifacts_b = create_artifacts(session_b, self.repository)

            # Static reference evidence is source-bound, so session B consumes
            # the exact same immutable files instead of copying or rebuilding
            # them. Runtime evidence remains session-local.
            artifacts_b["preflight"] = artifacts_a["preflight"]
            artifacts_b["registry"] = artifacts_a["registry"]
            model_b = json.loads(
                artifacts_b["model"].read_text(encoding="utf-8")
            )
            model_b["roles"]["teacher"]["process_identity"] = {
                "server_pid": 22002,
                "process_start_ticks": 88002,
            }
            write_canonical(artifacts_b["model"], model_b)
            runtime_hashes_b = json.loads(
                artifacts_b["runtime_hashes"].read_text(encoding="utf-8")
            )
            runtime_hashes_b["model_server_receipts"][
                "file_sha256"
            ] = runtime.file_sha256(artifacts_b["model"])
            runtime_hashes_b["model_server_receipts"][
                "semantic_sha256"
            ] = generation.sha256(model_b)
            write_canonical(
                artifacts_b["runtime_hashes"],
                runtime_hashes_b,
            )

            output_a = session_a / "release_manifest.json"
            output_b = session_b / "release_manifest.json"
            patches = self.patched_dependencies(tau2)
            with ExitStack() as stack:
                for dependency_patch in patches:
                    stack.enter_context(dependency_patch)
                release.build_manifest(
                    arguments(
                        repository=self.repository,
                        tau2=tau2,
                        artifacts=artifacts_a,
                        output=output_a,
                    )
                )
                session_a_bytes = {
                    path.relative_to(session_a): path.read_bytes()
                    for path in session_a.rglob("*")
                    if path.is_file()
                }
                release.build_manifest(
                    arguments(
                        repository=self.repository,
                        tau2=tau2,
                        artifacts=artifacts_b,
                        output=output_b,
                    )
                )

            manifest_a = json.loads(output_a.read_text(encoding="utf-8"))
            manifest_b = json.loads(output_b.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_a["reference_preflight_receipt_sha256"],
                manifest_b["reference_preflight_receipt_sha256"],
            )
            self.assertEqual(
                manifest_a["registry_file_sha256"],
                manifest_b["registry_file_sha256"],
            )
            self.assertNotEqual(
                manifest_a["model_server_receipts_sha256"],
                manifest_b["model_server_receipts_sha256"],
            )
            self.assertNotEqual(
                manifest_a["runtime_receipt_hashes_sha256"],
                manifest_b["runtime_receipt_hashes_sha256"],
            )
            self.assertNotEqual(
                runtime.file_sha256(output_a),
                runtime.file_sha256(output_b),
            )
            self.assertEqual(
                session_a_bytes,
                {
                    path.relative_to(session_a): path.read_bytes()
                    for path in session_a.rglob("*")
                    if path.is_file()
                },
            )

    def test_registry_config_hash_drift_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            artifacts = create_artifacts(root, self.repository)
            registry = json.loads(
                artifacts["registry"].read_text(encoding="utf-8")
            )
            registry["source"]["config_file_sha256"] = "f" * 64
            write_canonical(artifacts["registry"], registry)
            output = root / "release_manifest.json"
            patches = self.patched_dependencies(tau2)
            with ExitStack() as stack:
                for dependency_patch in patches:
                    stack.enter_context(dependency_patch)
                with self.assertRaisesRegex(
                    release.ReleaseManifestError,
                    "identity drift",
                ):
                    release.build_manifest(
                        arguments(
                            repository=self.repository,
                            tau2=tau2,
                            artifacts=artifacts,
                            output=output,
                        )
                    )
            self.assertFalse(output.exists())

    def test_runtime_hash_manifest_tampering_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            artifacts = create_artifacts(root, self.repository)
            hashes = json.loads(
                artifacts["runtime_hashes"].read_text(encoding="utf-8")
            )
            hashes["model_server_receipts"]["file_sha256"] = "0" * 64
            write_canonical(artifacts["runtime_hashes"], hashes)
            output = root / "release_manifest.json"
            patches = self.patched_dependencies(tau2)
            with ExitStack() as stack:
                for dependency_patch in patches:
                    stack.enter_context(dependency_patch)
                with self.assertRaisesRegex(
                    release.ReleaseManifestError,
                    "exact receipt bytes",
                ):
                    release.build_manifest(
                        arguments(
                            repository=self.repository,
                            tau2=tau2,
                            artifacts=artifacts,
                            output=output,
                        )
                    )
            self.assertFalse(output.exists())

    def test_uncommitted_release_source_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tau2 = root / "tau2"
            tau2.mkdir()
            artifacts = create_artifacts(root, self.repository)
            output = root / "release_manifest.json"
            patches = self.patched_dependencies(tau2)
            with ExitStack() as stack:
                for dependency_patch in patches:
                    stack.enter_context(dependency_patch)
                stack.enter_context(
                    patch.object(
                        generation,
                        "verify_committed_source_file",
                        side_effect=generation.V6GenerationError(
                            "not tracked"
                        ),
                    )
                )
                with self.assertRaisesRegex(
                    release.ReleaseManifestError,
                    "exact frozen commit",
                ):
                    release.build_manifest(
                        arguments(
                            repository=self.repository,
                            tau2=tau2,
                            artifacts=artifacts,
                            output=output,
                        )
                    )
            self.assertFalse(output.exists())

    def test_atomic_publish_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release_manifest.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(
                release.ReleaseManifestError,
                "refusing to overwrite",
            ):
                release.atomic_publish(
                    output,
                    {
                        "protocol": release.PROTOCOL,
                        "official_test_used": False,
                    },
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_generator_self_verification_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release_manifest.json"
            with patch.object(
                generation,
                "load_release_manifest",
                side_effect=RuntimeError("schema mismatch"),
                create=True,
            ):
                with self.assertRaisesRegex(
                    release.ReleaseManifestError,
                    "generator rejected",
                ):
                    release.atomic_publish(
                        output,
                        {
                            "protocol": release.PROTOCOL,
                            "official_test_used": False,
                        },
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
