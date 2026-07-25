#!/usr/bin/env python3
"""Build the immutable V5 Stage-1 checkpoint registry.

Only complete formal runs from the four canonical SFT arms are accepted.  The
registry binds each served model alias to both the adapter bytes and the
training run manifest that produced them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "v5_stage1_checkpoint_registry"
TRAIN_PROTOCOL = "v5_stage1_message_masked_sft_7b"
DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ARMS = ("perfect_success", "failure_raw", "repair_50", "repair_100")
MODEL_IDS = {
    "base_model": "openai/v5-base",
    "perfect_success": "openai/v5-perfect-success",
    "failure_raw": "openai/v5-failure-raw",
    "repair_50": "openai/v5-repair-50",
    "repair_100": "openai/v5-repair-100",
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def parse_arm_bindings(values: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    resolved_paths: set[Path] = set()
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"--arm must be ARM=RUN_DIR, got {value!r}")
        arm, raw_path = value.split("=", 1)
        if arm not in ARMS:
            raise RuntimeError(f"unsupported Stage-1 arm {arm!r}")
        if arm in bindings:
            raise RuntimeError(f"duplicate arm binding: {arm}")
        if not raw_path:
            raise RuntimeError(f"{arm}: empty run directory")
        path = Path(raw_path).expanduser().resolve()
        if path in resolved_paths:
            raise RuntimeError(f"duplicate run directory: {path}")
        resolved_paths.add(path)
        bindings[arm] = path
    if set(bindings) != set(ARMS):
        missing = sorted(set(ARMS) - set(bindings))
        extra = sorted(set(bindings) - set(ARMS))
        raise RuntimeError(
            f"arm bindings must be exactly {ARMS}; missing={missing}, extra={extra}"
        )
    return bindings


def validate_run(
    *,
    arm: str,
    run_dir: Path,
    source_commit: str,
    base_revision: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not run_dir.is_dir():
        raise RuntimeError(f"{arm}: formal run directory missing: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_dir = run_dir / "checkpoint_final"
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"{arm}: checkpoint_final directory missing")
    if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
        raise RuntimeError(f"{arm}: adapter_model.safetensors missing or empty")
    if (
        not adapter_config_path.is_file()
        or adapter_config_path.stat().st_size <= 0
    ):
        raise RuntimeError(f"{arm}: adapter_config.json missing or empty")
    # Parsing here prevents a byte-bound but unusable config from entering the
    # serving registry.
    adapter_config = load_json(adapter_config_path)
    if adapter_config.get("peft_type") != "LORA":
        raise RuntimeError(f"{arm}: adapter_config.json is not a LoRA adapter")
    manifest = load_json(manifest_path)
    expected = {
        "protocol": TRAIN_PROTOCOL,
        "mode": "formal",
        "arm": arm,
        "source_commit": source_commit,
        "model": BASE_MODEL,
        "model_revision": base_revision,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"{arm}: run_manifest {field} drift: "
                f"expected {value!r}, got {manifest.get(field)!r}"
            )
    if manifest.get("held_out_test_accessed") is not False:
        raise RuntimeError(f"{arm}: run manifest does not certify held-out seal")
    provenance = manifest.get("data_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{arm}: run manifest lacks data provenance")
    if (
        provenance.get("official_test_used") is not False
        or provenance.get("official_test_sealed") is not True
    ):
        raise RuntimeError(f"{arm}: data provenance lacks held-out seal")
    common_provenance: dict[str, Any] = {
        "data_audit_sha256": provenance.get("data_audit_sha256"),
        "data_hashes_sha256": provenance.get("data_hashes_sha256"),
        "dynamic_audits": provenance.get("dynamic_audits"),
        "official_test_used": False,
        "official_test_sealed": True,
    }
    if any(
        not isinstance(common_provenance[field], str)
        or SHA256_RE.fullmatch(common_provenance[field]) is None
        for field in ("data_audit_sha256", "data_hashes_sha256")
    ):
        raise RuntimeError(f"{arm}: data provenance SHA drift")
    dynamic = common_provenance["dynamic_audits"]
    if not isinstance(dynamic, dict) or set(dynamic) != {
        "generation",
        "validation",
    }:
        raise RuntimeError(f"{arm}: data provenance dynamic audits drift")
    for name, expected_count in (("generation", 83), ("validation", 21)):
        identity = dynamic.get(name)
        if (
            not isinstance(identity, dict)
            or identity.get("protocol") != DYNAMIC_AUDIT_PROTOCOL
            or identity.get("verified_injections") != expected_count
            or identity.get("official_test_used") is not False
            or identity.get("official_test_sealed") is not True
        ):
            raise RuntimeError(f"{arm}: {name} dynamic audit identity drift")
        for field in ("sha256", "manifest_sha256", "split_manifest_sha256"):
            value = identity.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RuntimeError(
                    f"{arm}: {name} dynamic audit {field} drift"
                )
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{arm}: run manifest lacks checkpoint metadata")
    recorded_files = checkpoint.get("file_sha256")
    if not isinstance(recorded_files, dict):
        raise RuntimeError(f"{arm}: run manifest lacks checkpoint file hashes")
    adapter_sha = sha256_file(adapter_path)
    adapter_config_sha = sha256_file(adapter_config_path)
    if recorded_files.get("adapter_model.safetensors") != adapter_sha:
        raise RuntimeError(f"{arm}: adapter SHA differs from run manifest")
    if recorded_files.get("adapter_config.json") != adapter_config_sha:
        raise RuntimeError(
            f"{arm}: adapter config SHA differs from run manifest"
        )
    manifest_sha = sha256_file(manifest_path)
    if any(
        SHA256_RE.fullmatch(value) is None
        for value in (adapter_sha, adapter_config_sha, manifest_sha)
    ):
        raise RuntimeError(f"{arm}: invalid computed SHA-256")
    return (
        {
            "model_id": MODEL_IDS[arm],
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha,
            "training_run_manifest_sha256": manifest_sha,
        },
        common_provenance,
    )


def build_registry(
    *,
    source_commit: str,
    base_revision: str,
    arm_dirs: dict[str, Path],
) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("source commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(base_revision) is None:
        raise RuntimeError("base revision must be a full lowercase revision")
    if set(arm_dirs) != set(ARMS):
        raise RuntimeError(f"arm directories must be exactly {ARMS}")
    entries: dict[str, Any] = {
        "base_model": {
            "model_id": MODEL_IDS["base_model"],
            "adapter_sha256": None,
            "adapter_config_sha256": None,
            "training_run_manifest_sha256": None,
        }
    }
    resolved = [Path(arm_dirs[arm]).expanduser().resolve() for arm in ARMS]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("trained arms contain duplicate run directories")
    provenance_by_arm: dict[str, dict[str, Any]] = {}
    for arm, run_dir in zip(ARMS, resolved):
        entries[arm], provenance_by_arm[arm] = validate_run(
            arm=arm,
            run_dir=run_dir,
            source_commit=source_commit,
            base_revision=base_revision,
        )
    serialized_provenance = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in provenance_by_arm.values()
    }
    if len(serialized_provenance) != 1:
        raise RuntimeError(
            "trained arms do not share one data/dynamic-audit provenance"
        )
    common_provenance = provenance_by_arm[ARMS[0]]
    model_ids = [entry["model_id"] for entry in entries.values()]
    if len(set(model_ids)) != 5:
        raise RuntimeError("all five model_id aliases must be unique")
    adapter_hashes = [entries[arm]["adapter_sha256"] for arm in ARMS]
    manifest_hashes = [
        entries[arm]["training_run_manifest_sha256"] for arm in ARMS
    ]
    if len(set(adapter_hashes)) != len(adapter_hashes):
        raise RuntimeError("trained arms contain duplicate adapter bytes")
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise RuntimeError("trained arms contain duplicate run manifests")
    return {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "base_model_revision": base_revision,
        "training_data_provenance": common_provenance,
        "entries": entries,
    }


def atomic_write_registry(path: Path, registry: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Canonical ARM=FORMAL_RUN_DIR binding; provide exactly four.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bindings = parse_arm_bindings(args.arm)
    registry = build_registry(
        source_commit=args.source_commit,
        base_revision=args.base_revision,
        arm_dirs=bindings,
    )
    output = args.output.expanduser().resolve()
    atomic_write_registry(output, registry)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256_file(output),
                "model_ids": {
                    arm: entry["model_id"]
                    for arm, entry in registry["entries"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
