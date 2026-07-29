#!/usr/bin/env python3
"""Build the immutable V6 directional-screen checkpoint registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from train_v6_directional_sft import (
        ARMS,
        MODEL_ID,
        MODEL_REVISION,
        PROTOCOL as TRAINING_PROTOCOL,
        TOKENIZER_REVISION,
        TRAIN_SEED,
    )
except ModuleNotFoundError:
    from scripts.train_v6_directional_sft import (
        ARMS,
        MODEL_ID,
        MODEL_REVISION,
        PROTOCOL as TRAINING_PROTOCOL,
        TOKENIZER_REVISION,
        TRAIN_SEED,
    )


PROTOCOL = "v6_directional_checkpoint_registry_v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_DIRS = {
    "flawless_only": "flawless_only",
    "random_stratified": "random_stratified_seed_20260806",
    "full_proposed": "full_proposed",
}


class RegistryError(RuntimeError):
    """A frozen V6 checkpoint contract was violated."""


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
        raise RegistryError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RegistryError(f"expected JSON object: {path}")
    return value


def current_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checked_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RegistryError(f"{label} must be a lowercase SHA-256")
    return value


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RegistryError(f"incomplete checkpoint file: {path}")
        digest = sha256_file(path)
        file_hashes[name] = digest
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "path": str(checkpoint.resolve()),
        "fingerprint": aggregate.hexdigest(),
        "file_sha256": file_hashes,
    }


def validate_run(
    run_root: Path,
    *,
    arm: str,
    source_commit: str,
) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    audit_path = run_root / "audit.json"
    metrics_path = run_root / "training_metrics.json"
    files_path = run_root / "files_sha256.json"
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    metrics = read_json(metrics_path)
    recorded_files = read_json(files_path)
    identities = manifest.get("identities")
    hyperparameters = manifest.get("hyperparameters")
    if (
        manifest.get("protocol") != TRAINING_PROTOCOL
        or manifest.get("status") != "PASS"
        or manifest.get("source_commit") != source_commit
        or manifest.get("arm") != arm
        or manifest.get("mode") != "formal"
        or manifest.get("official_test_used") is not False
        or not isinstance(identities, dict)
        or identities.get("model") != MODEL_ID
        or identities.get("model_revision") != MODEL_REVISION
        or identities.get("tokenizer_revision") != TOKENIZER_REVISION
        or not isinstance(hyperparameters, dict)
        or hyperparameters.get("train_seed") != TRAIN_SEED
    ):
        raise RegistryError(f"{arm}: training manifest identity drift")
    if (
        audit.get("training_finite") is not True
        or audit.get("official_test_used") is not False
        or metrics.get("protocol") != TRAINING_PROTOCOL
        or metrics.get("arm") != arm
        or metrics.get("mode") != "formal"
        or not isinstance(metrics.get("finite_audit"), dict)
        or metrics["finite_audit"].get("status") != "PASS"
    ):
        raise RegistryError(f"{arm}: training audit did not pass")
    checkpoint = checkpoint_identity(run_root / "checkpoint_final")
    if manifest.get("adapter_files_sha256") != checkpoint["file_sha256"]:
        raise RegistryError(f"{arm}: checkpoint bytes differ from manifest")
    required_hashes = {
        "audit.json": sha256_file(audit_path),
        "training_metrics.json": sha256_file(metrics_path),
        "run_manifest.json": sha256_file(manifest_path),
        "checkpoint_final/adapter_config.json": checkpoint["file_sha256"][
            "adapter_config.json"
        ],
        "checkpoint_final/adapter_model.safetensors": checkpoint["file_sha256"][
            "adapter_model.safetensors"
        ],
    }
    for name, digest in required_hashes.items():
        if recorded_files.get(name) != digest:
            raise RegistryError(f"{arm}: files_sha256 drift for {name}")
    return {
        "paper_arm": arm,
        "training_seed": TRAIN_SEED,
        "model_id": f"openai/v6-{arm}-seed-{TRAIN_SEED}",
        "checkpoint": checkpoint,
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256_file(manifest_path),
        "selector_manifest_sha256": checked_hash(
            manifest.get("selector_manifest_sha256"),
            f"{arm}.selector_manifest_sha256",
        ),
        "train_file_sha256": checked_hash(
            manifest.get("train_file_sha256"),
            f"{arm}.train_file_sha256",
        ),
        "candidate_pool_sha256": checked_hash(
            manifest.get("candidate_pool_sha256"),
            f"{arm}.candidate_pool_sha256",
        ),
        "hyperparameters_sha256": checked_hash(
            manifest.get("hyperparameters_sha256"),
            f"{arm}.hyperparameters_sha256",
        ),
        "audit_sha256": sha256_file(audit_path),
        "training_metrics_sha256": sha256_file(metrics_path),
    }


def build(
    *,
    training_root: Path,
    output: Path,
    source_commit: str,
    user_judge_model: str,
    user_judge_revision: str,
    user_judge_alias: str,
) -> dict[str, Any]:
    if output.exists():
        raise RegistryError(f"refusing to overwrite {output}")
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RegistryError("source commit must be a full lowercase commit")
    if current_commit() != source_commit:
        raise RegistryError("local source commit differs from requested source commit")
    if COMMIT_RE.fullmatch(user_judge_revision) is None:
        raise RegistryError("user/judge revision must be a full lowercase commit")
    if not user_judge_alias.startswith("openai/"):
        raise RegistryError("user/judge alias must use the openai/ prefix")
    entries = {
        arm: validate_run(
            training_root / RUN_DIRS[arm] / f"seed_{TRAIN_SEED}",
            arm=arm,
            source_commit=source_commit,
        )
        for arm in ARMS
    }
    hyperparameter_hashes = {
        entry["hyperparameters_sha256"] for entry in entries.values()
    }
    if len(hyperparameter_hashes) != 1:
        raise RegistryError("training hyperparameters differ across V6 arms")
    payload = {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "base_model_alias": "openai/v6-base",
        "user_judge": {
            "model": user_judge_model,
            "revision": user_judge_revision,
            "model_id": user_judge_alias,
            "roles": ["user_simulator", "strict_nl_judge"],
        },
        "registered_arms": list(ARMS),
        "registered_training_seeds": [TRAIN_SEED],
        "entries": entries,
        "cross_arm_hyperparameters_sha256": next(iter(hyperparameter_hashes)),
        "official_test_used": False,
        "official_test_sealed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--user-judge-model",
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
    )
    parser.add_argument("--user-judge-revision", required=True)
    parser.add_argument("--user-judge-alias", default="openai/v6-user-judge")
    args = parser.parse_args()
    payload = build(
        training_root=args.training_root.resolve(),
        output=args.output.resolve(),
        source_commit=args.source_commit,
        user_judge_model=args.user_judge_model,
        user_judge_revision=args.user_judge_revision,
        user_judge_alias=args.user_judge_alias,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "registry": str(args.output.resolve()),
                "arms": payload["registered_arms"],
                "training_seeds": payload["registered_training_seeds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
