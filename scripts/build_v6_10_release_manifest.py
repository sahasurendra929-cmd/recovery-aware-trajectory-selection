#!/usr/bin/env python3
"""Build the immutable, cross-artifact V6.10 release manifest.

This command runs only after runtime receipts, the static reference preflight,
and the executable registry exist.  It fails closed unless their hashes and
embedded source/config identities all agree, then publishes one canonical
manifest using an atomic no-overwrite hard link.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

if __package__:
    from scripts import build_v6_10_runtime_receipts as runtime
    from scripts import prepare_v6_candidate_registry as registry_source
    from scripts import run_v6_candidate_generation as generation
else:  # pragma: no cover - direct script execution
    import build_v6_10_runtime_receipts as runtime
    import prepare_v6_candidate_registry as registry_source
    import run_v6_candidate_generation as generation


PROTOCOL = "v6_10_release_manifest_v1"
RELEVANT_SCRIPTS = (
    "scripts/run_v6_candidate_generation.py",
    "scripts/audit_v6_candidates.py",
    "scripts/measure_v6_candidate_tokens.py",
    "scripts/score_v6_candidates.py",
    "scripts/build_v6_selector_manifests.py",
    "scripts/build_v6_checkpoint_registry.py",
    "scripts/materialize_v6_sft.py",
    "scripts/train_v6_directional_sft.py",
    "scripts/preflight_v6_reference_traces.py",
    "scripts/prepare_v6_10_registry.py",
    "scripts/build_v6_10_runtime_receipts.py",
    "scripts/build_v6_10_release_manifest.py",
    "scripts/v6_10_selection_protocol.py",
    "scripts/v6_reference_contract.py",
)
CONFIG_PATH = Path("configs/v6_10_closure.yaml")
PREREGISTRATION_PATH = Path("V6_10_CLOSURE_PREREGISTRATION.md")
SPLIT_MANIFEST_PATH = Path(
    "artifacts/v5_stage0/manifests/split_manifest.json"
)
RUNTIME_HASH_PROTOCOL = "v6_10_runtime_receipt_hashes_v1"
RUNTIME_HASH_KEYS = {
    "protocol",
    "design_protocol",
    "status",
    "source_container_provenance",
    "model_server_receipts",
    "created_at_utc",
    "official_test_used",
}
RUNTIME_FILE_KEYS = {"path", "file_sha256", "semantic_sha256"}


class ReleaseManifestError(RuntimeError):
    """The proposed V6.10 release is incomplete or internally inconsistent."""


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseManifestError(f"{label} is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseManifestError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ReleaseManifestError(f"{label} root is not an object")
    return value


def _sha256(value: Any) -> str:
    return generation.sha256(value)


def _file_sha256(path: Path) -> str:
    return runtime.file_sha256(path)


def frozen_source_paths(source_root: Path) -> dict[str, Path]:
    root = source_root.resolve()
    paths = {
        "config": root / CONFIG_PATH,
        "preregistration": root / PREREGISTRATION_PATH,
        "split_manifest": root / SPLIT_MANIFEST_PATH,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ReleaseManifestError(
            f"frozen source artifacts are absent: {sorted(missing)}"
        )
    return paths


def verify_frozen_config(paths: Mapping[str, Path]) -> dict[str, str]:
    try:
        config = registry_source.load_config(paths["config"])
    except Exception as error:
        raise ReleaseManifestError(
            f"cannot parse frozen V6.10 config: {error}"
        ) from error
    benchmark = config.get("benchmark")
    official = (
        benchmark.get("official_test")
        if isinstance(benchmark, Mapping)
        else None
    )
    if (
        config.get("protocol")
        != generation.V6_10_PIPELINE_CLOSURE_PROTOCOL
        or str(config.get("design_version")) != "6.10"
        or not isinstance(benchmark, Mapping)
        or benchmark.get("commit") != generation.protocol.TAU2_COMMIT
        or benchmark.get("split_manifest")
        != SPLIT_MANIFEST_PATH.as_posix()
        or not isinstance(official, Mapping)
        or official.get("sealed") is not True
        or official.get(
            "used_during_compatibility_generation_scoring_selection_or_model_selection"
        )
        is not False
    ):
        raise ReleaseManifestError(
            "frozen V6.10 config protocol/source/test seal drift"
        )
    split = load_json(paths["split_manifest"], label="split manifest")
    if split.get("official_test_used") not in (None, False):
        raise ReleaseManifestError(
            "split manifest reports official-test use"
        )
    return {
        "config_sha256": _file_sha256(paths["config"]),
        "preregistration_sha256": _file_sha256(
            paths["preregistration"]
        ),
        "split_manifest_sha256": _file_sha256(
            paths["split_manifest"]
        ),
    }


def relevant_script_hashes(source_root: Path) -> dict[str, str]:
    root = source_root.resolve()
    result: dict[str, str] = {}
    for relative in RELEVANT_SCRIPTS:
        path = root / relative
        if not path.is_file():
            raise ReleaseManifestError(
                f"required release script is absent: {relative}"
            )
        result[relative] = _file_sha256(path)
    if set(result) != set(RELEVANT_SCRIPTS):
        raise ReleaseManifestError("relevant script set drift")
    return dict(sorted(result.items()))


def verify_committed_release_sources(
    *,
    source_root: Path,
    source_commit: str,
    frozen_hashes: Mapping[str, str],
    script_hashes: Mapping[str, str],
) -> None:
    """Bind every live code/protocol input to an exact blob in the commit."""

    required = {
        CONFIG_PATH.as_posix(): frozen_hashes["config_sha256"],
        PREREGISTRATION_PATH.as_posix(): frozen_hashes[
            "preregistration_sha256"
        ],
        **dict(script_hashes),
    }
    if set(script_hashes) != set(RELEVANT_SCRIPTS):
        raise ReleaseManifestError("relevant script set drift")
    try:
        for relative, expected_sha256 in sorted(required.items()):
            generation.verify_committed_source_file(
                source_root=source_root.resolve(),
                source_commit=source_commit,
                relative=relative,
                expected_sha256=expected_sha256,
                label="release-builder frozen source",
            )
    except Exception as error:
        raise ReleaseManifestError(
            f"release source is not the exact frozen commit: {error}"
        ) from error


def verify_reference_and_registry(
    *,
    reference_preflight_path: Path,
    registry_path: Path,
    source_commit: str,
    frozen_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    preflight_file_sha256 = _file_sha256(reference_preflight_path)
    try:
        preflight, binding = generation.load_reference_preflight(
            reference_preflight_path,
            expected_sha256=preflight_file_sha256,
            expected_source_commit=source_commit,
        )
    except Exception as error:
        raise ReleaseManifestError(
            f"reference preflight failed canonical verification: {error}"
        ) from error
    registry = load_json(registry_path, label="V6.10 registry")
    try:
        generation.verify_registry(registry)
    except Exception as error:
        raise ReleaseManifestError(
            f"V6.10 registry failed canonical verification: {error}"
        ) from error
    registry_file_sha256 = _file_sha256(registry_path)
    registry_source_row = registry.get("source")
    registry_preflight = (
        registry_source_row.get("reference_preflight")
        if isinstance(registry_source_row, Mapping)
        else None
    )
    semantic_receipt_hash = (
        binding.get("receipt_sha256")
        if isinstance(binding, Mapping)
        else None
    )
    if (
        registry.get("design_protocol")
        != generation.V6_10_PIPELINE_CLOSURE_PROTOCOL
        or not isinstance(registry_source_row, Mapping)
        or registry_source_row.get("source_commit") != source_commit
        or registry_source_row.get("tau2_commit")
        != generation.protocol.TAU2_COMMIT
        or registry_source_row.get("config_file_sha256")
        != frozen_hashes["config_sha256"]
        or registry_source_row.get("split_file_sha256")
        != frozen_hashes["split_manifest_sha256"]
        or registry.get("reference_preflight_file_sha256")
        != preflight_file_sha256
        or registry.get("reference_preflight_receipt_sha256")
        != semantic_receipt_hash
        or not isinstance(registry_preflight, Mapping)
        or registry_preflight.get("file_sha256")
        != preflight_file_sha256
        or registry_preflight.get("receipt_sha256")
        != semantic_receipt_hash
        or preflight.get("source_commit") != source_commit
        or preflight.get("tau2_commit")
        != generation.protocol.TAU2_COMMIT
        or preflight.get("config_sha256")
        != frozen_hashes["config_sha256"]
        or preflight.get("split_manifest_sha256")
        != frozen_hashes["split_manifest_sha256"]
        or preflight.get("official_test_used") is not False
    ):
        raise ReleaseManifestError(
            "preflight, registry, source, config, or split identity drift"
        )
    return (
        preflight,
        registry,
        preflight_file_sha256,
        registry_file_sha256,
    )


def verify_runtime_receipts(
    *,
    source_container_path: Path,
    model_server_path: Path,
    runtime_hashes_path: Path,
    source_commit: str,
    container_image_digest: str,
    generator_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_payload = load_json(
        source_container_path,
        label="source/container provenance",
    )
    model_payload = load_json(
        model_server_path,
        label="model-server receipts",
    )
    hashes = load_json(
        runtime_hashes_path,
        label="runtime receipt hash manifest",
    )
    source_file_sha256 = _file_sha256(source_container_path)
    model_file_sha256 = _file_sha256(model_server_path)
    source_hash_row = hashes.get("source_container_provenance")
    model_hash_row = hashes.get("model_server_receipts")
    if (
        set(hashes) != RUNTIME_HASH_KEYS
        or hashes.get("protocol") != RUNTIME_HASH_PROTOCOL
        or hashes.get("design_protocol")
        != generation.V6_10_PIPELINE_CLOSURE_PROTOCOL
        or hashes.get("status") != "PASS"
        or generation.UTC_TIMESTAMP_RE.fullmatch(
            str(hashes.get("created_at_utc"))
        )
        is None
        or hashes.get("official_test_used") is not False
        or not isinstance(source_hash_row, dict)
        or set(source_hash_row) != RUNTIME_FILE_KEYS
        or source_hash_row.get("path")
        != runtime.SOURCE_RECEIPT_NAME
        or source_hash_row.get("file_sha256") != source_file_sha256
        or source_hash_row.get("semantic_sha256")
        != _sha256(source_payload)
        or not isinstance(model_hash_row, dict)
        or set(model_hash_row) != RUNTIME_FILE_KEYS
        or model_hash_row.get("path") != runtime.MODEL_RECEIPT_NAME
        or model_hash_row.get("file_sha256") != model_file_sha256
        or model_hash_row.get("semantic_sha256")
        != _sha256(model_payload)
    ):
        raise ReleaseManifestError(
            "runtime receipt hash manifest does not bind exact receipt bytes"
        )
    try:
        _, source_binding = (
            generation.load_source_container_provenance(
                source_container_path,
                expected_sha256=source_file_sha256,
                expected_source_commit=source_commit,
                expected_generation_script_sha256=generator_sha256,
                expected_tau2_commit=generation.protocol.TAU2_COMMIT,
                container_image_digest=container_image_digest,
            )
        )
    except Exception as error:
        raise ReleaseManifestError(
            f"source/container provenance failed verification: {error}"
        ) from error
    roles = model_payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(
        runtime.ROLE_ORDER
    ):
        raise ReleaseManifestError(
            "model-server receipt role population drift"
        )
    expected_roles: dict[str, dict[str, Any]] = {}
    frozen_models = {
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
    for role in runtime.ROLE_ORDER:
        row = roles[role]
        model, revision, tensor_parallel = frozen_models[role]
        if (
            not isinstance(row, Mapping)
            or row.get("model") != model
            or row.get("resolved_revision") != revision
            or row.get("quantization")
            != generation.V610_MODEL_QUANTIZATION
            or row.get("tensor_parallel_size") != tensor_parallel
            or not isinstance(row.get("api_base"), str)
            or not row["api_base"]
        ):
            raise ReleaseManifestError(
                f"frozen model identity drift: roles.{role}"
            )
        expected_roles[role] = {
            "model": model,
            "revision": revision,
            "api_base": row["api_base"],
            "quantization": generation.V610_MODEL_QUANTIZATION,
            "tensor_parallel_size": tensor_parallel,
        }
    try:
        generation.load_model_server_receipts(
            model_server_path,
            expected_sha256=model_file_sha256,
            expected_roles=expected_roles,
            container_image_digest=container_image_digest,
            available_gpu_uuids=[
                str(row["uuid"]) for row in source_binding["gpus"]
            ],
            expected_python_executable_path=source_binding["runtime"][
                "python_executable_path"
            ],
            expected_python_executable_sha256=source_binding["runtime"][
                "python_executable_sha256"
            ],
        )
    except Exception as error:
        raise ReleaseManifestError(
            f"model-server receipts failed verification: {error}"
        ) from error
    if (
        source_payload.get("source_commit") != source_commit
        or source_payload.get("tau2_commit")
        != generation.protocol.TAU2_COMMIT
        or source_payload.get("container_image_digest")
        != container_image_digest
        or model_payload.get("container_image_digest")
        != container_image_digest
        or source_payload.get("official_test_used") is not False
        or model_payload.get("official_test_used") is not False
    ):
        raise ReleaseManifestError(
            "runtime receipts disagree on release/container identity"
        )
    return source_payload, model_payload, hashes


def release_validator_expected(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact immutable identity map consumed by the generator."""

    fields = (
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
    )
    return {field: deepcopy(manifest[field]) for field in fields}


def self_verify_if_available(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    source_root: Path,
) -> None:
    loader = getattr(generation, "load_release_manifest", None)
    if loader is None:
        return
    try:
        loader(
            path,
            expected_sha256=_file_sha256(path),
            source_root=source_root.resolve(),
            expected_fields=release_validator_expected(manifest),
        )
    except Exception as error:
        raise ReleaseManifestError(
            f"generator rejected the release manifest: {error}"
        ) from error


def atomic_publish(
    path: Path,
    value: Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> None:
    output = path.resolve()
    if output.exists():
        raise ReleaseManifestError(
            f"refusing to overwrite release manifest: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=str(output.parent),
    )
    temporary = Path(temporary_name)
    try:
        encoded = runtime.canonical(dict(value)).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self_verify_if_available(
            temporary,
            manifest=value,
            source_root=(
                Path(__file__).resolve().parents[1]
                if source_root is None
                else source_root
            ),
        )
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ReleaseManifestError(
                f"release manifest appeared concurrently: {output}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    tau2_root = args.tau2_root.resolve()
    source = runtime.git_identity(source_root)
    tau2 = runtime.git_identity(tau2_root)
    if tau2.get("commit") != generation.protocol.TAU2_COMMIT:
        raise ReleaseManifestError(
            "tau2 checkout is not the frozen V6.10 commit"
        )
    if generation.OCI_DIGEST_RE.fullmatch(
        args.container_image_digest
    ) is None:
        raise ReleaseManifestError("container image digest is invalid")
    paths = frozen_source_paths(source_root)
    frozen_hashes = verify_frozen_config(paths)
    script_hashes = relevant_script_hashes(source_root)
    verify_committed_release_sources(
        source_root=source_root,
        source_commit=str(source["commit"]),
        frozen_hashes=frozen_hashes,
        script_hashes=script_hashes,
    )
    generator_sha256 = script_hashes[
        "scripts/run_v6_candidate_generation.py"
    ]
    (
        _,
        _,
        preflight_file_sha256,
        registry_file_sha256,
    ) = verify_reference_and_registry(
        reference_preflight_path=args.reference_preflight.resolve(),
        registry_path=args.registry.resolve(),
        source_commit=str(source["commit"]),
        frozen_hashes=frozen_hashes,
    )
    verify_runtime_receipts(
        source_container_path=args.source_container_provenance.resolve(),
        model_server_path=args.model_server_receipts.resolve(),
        runtime_hashes_path=args.runtime_receipt_hashes.resolve(),
        source_commit=str(source["commit"]),
        container_image_digest=args.container_image_digest,
        generator_sha256=generator_sha256,
    )
    manifest = {
        "protocol": PROTOCOL,
        "design_protocol": generation.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "status": "PASS",
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "container_image_digest": args.container_image_digest,
        "tau2_commit": tau2["commit"],
        "config_sha256": frozen_hashes["config_sha256"],
        "preregistration_sha256": frozen_hashes[
            "preregistration_sha256"
        ],
        "split_manifest_sha256": frozen_hashes[
            "split_manifest_sha256"
        ],
        "relevant_scripts": script_hashes,
        "reference_preflight_receipt_sha256": preflight_file_sha256,
        "registry_file_sha256": registry_file_sha256,
        "source_container_provenance_sha256": _file_sha256(
            args.source_container_provenance.resolve()
        ),
        "model_server_receipts_sha256": _file_sha256(
            args.model_server_receipts.resolve()
        ),
        "runtime_receipt_hashes_sha256": _file_sha256(
            args.runtime_receipt_hashes.resolve()
        ),
        "created_at_utc": runtime.utc_now(),
        "official_test_used": False,
    }
    if set(manifest) != {
        "protocol",
        "design_protocol",
        "status",
        "source_commit",
        "source_tree",
        "container_image_digest",
        "tau2_commit",
        "config_sha256",
        "preregistration_sha256",
        "split_manifest_sha256",
        "relevant_scripts",
        "reference_preflight_receipt_sha256",
        "registry_file_sha256",
        "source_container_provenance_sha256",
        "model_server_receipts_sha256",
        "runtime_receipt_hashes_sha256",
        "created_at_utc",
        "official_test_used",
    }:
        raise ReleaseManifestError("release manifest schema drift")
    atomic_publish(args.output, manifest, source_root=source_root)
    return {
        "status": "PASS",
        "output": str(args.output.resolve()),
        "release_manifest_sha256": _file_sha256(args.output.resolve()),
        "source_commit": source["commit"],
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
    parser.add_argument(
        "--container-image-digest",
        required=True,
    )
    parser.add_argument(
        "--reference-preflight",
        type=Path,
        required=True,
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--source-container-provenance",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-server-receipts",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--runtime-receipt-hashes",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(runtime.canonical(build_manifest(parse_args())))


if __name__ == "__main__":
    main()
