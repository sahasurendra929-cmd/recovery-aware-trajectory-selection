#!/usr/bin/env python3
"""Build an immutable V5.5 checkpoint registry from completed QLoRA runs."""
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
    import v5_5_full_protocol as full
except ModuleNotFoundError:
    from scripts import v5_5_full_protocol as full


PROTOCOL = "v5_5_checkpoint_registry_v1"
TRAINER_ARMS = {
    "perfect_success": "r0_perfect",
    "repair_25": "r25",
    "repair_50": "r50",
    "repair_75": "r75",
    "repair_100": "r100_recovery",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_csv(value: str, *, allowed: set[str], label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or not set(values) <= allowed:
        raise RuntimeError(f"invalid {label}: {values}")
    return values


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    required = ("adapter_config.json", "adapter_model.safetensors")
    hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"incomplete checkpoint file: {path}")
        digest = sha256_file(path)
        hashes[name] = digest
        aggregate.update(name.encode())
        aggregate.update(digest.encode())
    return {
        "path": str(checkpoint.resolve()),
        "fingerprint": aggregate.hexdigest(),
        "file_sha256": hashes,
    }


def validate_run(
    path: Path,
    *,
    arm: str,
    seed: int,
    source_commit: str,
    data_audit_sha256: str,
    model_revision: str,
) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("protocol") != "v5_stage1_message_masked_sft_7b"
        or manifest.get("source_commit") != source_commit
        or manifest.get("arm") != arm
        or manifest.get("mode") != "formal"
        or manifest.get("seed") != seed
        or manifest.get("model") != full.MODEL
        or manifest.get("model_revision") != model_revision
        or manifest.get("held_out_test_accessed") is not False
    ):
        raise RuntimeError(f"{arm}/{seed}: training manifest identity drift")
    provenance = manifest.get("data_provenance")
    design = (
        provenance.get("design_provenance")
        if isinstance(provenance, dict)
        else None
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("design_version") != full.DESIGN_VERSION
        or provenance.get("data_audit_sha256") != data_audit_sha256
        or provenance.get("official_test_used") is not False
        or provenance.get("official_test_sealed") is not True
        or not isinstance(design, dict)
        or design.get("training_mixture_basis") != "supervised_token_mass"
        or design.get("failed_action_positive_labels") != 0
    ):
        raise RuntimeError(f"{arm}/{seed}: V5.5 data provenance drift")
    checkpoint = path / "checkpoint_final"
    identity = checkpoint_identity(checkpoint)
    recorded = manifest.get("checkpoint")
    if (
        not isinstance(recorded, dict)
        or recorded.get("fingerprint") != identity["fingerprint"]
        or recorded.get("file_sha256") != identity["file_sha256"]
    ):
        raise RuntimeError(f"{arm}/{seed}: checkpoint bytes differ from manifest")
    return {
        "trainer_arm": arm,
        "paper_arm": TRAINER_ARMS[arm],
        "training_seed": seed,
        "model_id": f"openai/v55-{TRAINER_ARMS[arm]}-seed-{seed}",
        "checkpoint": identity,
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256_file(manifest_path),
        "train_file_sha256": provenance["train_file_sha256"],
        "data_audit_sha256": provenance["data_audit_sha256"],
        "claim_level": design["scientific_claim_level"],
    }


def build(
    *,
    training_root: Path,
    data_root: Path,
    output: Path,
    source_commit: str,
    model_revision: str,
    arms: list[str],
    seeds: list[int],
    user_judge_model: str,
    user_judge_revision: str,
    user_judge_alias: str,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("source commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(model_revision) is None:
        raise RuntimeError("model revision must be a full lowercase commit")
    if COMMIT_RE.fullmatch(user_judge_revision) is None:
        raise RuntimeError("user/judge revision must be a full lowercase commit")
    if not user_judge_alias.startswith("openai/"):
        raise RuntimeError("user/judge alias must use the openai/ LiteLLM prefix")
    if git_commit() != source_commit:
        raise RuntimeError("local source commit differs from registry source commit")
    data_audit_path = data_root / "audit.json"
    data_hashes_path = data_root / "hashes.json"
    data_audit = read_json(data_audit_path)
    data_hashes = read_json(data_hashes_path)
    if (
        data_audit.get("protocol") != "v5_5_full_sft_data_v1"
        or data_audit.get("design_version") != full.DESIGN_VERSION
        or data_audit.get("status") != "PASS"
        or data_audit.get("official_test_used") is not False
        or data_hashes.get("audit.json") != sha256_file(data_audit_path)
    ):
        raise RuntimeError("V5.5 data bundle is not an authorized immutable input")
    data_audit_sha = sha256_file(data_audit_path)
    entries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        entries[arm] = {}
        for seed in seeds:
            run_root = training_root / arm / str(seed)
            entries[arm][str(seed)] = validate_run(
                run_root,
                arm=arm,
                seed=seed,
                source_commit=source_commit,
                data_audit_sha256=data_audit_sha,
                model_revision=model_revision,
            )
    payload = {
        "protocol": PROTOCOL,
        "design_protocol": full.PROTOCOL,
        "design_version": full.DESIGN_VERSION,
        "source_commit": source_commit,
        "base_model": full.MODEL,
        "base_model_revision": model_revision,
        "base_model_alias": "openai/v55-base",
        "user_judge": {
            "model": user_judge_model,
            "revision": user_judge_revision,
            "model_id": user_judge_alias,
            "roles": ["user_simulator", "strict_nl_judge"],
        },
        "data": {
            "root": str(data_root.resolve()),
            "audit_path": str(data_audit_path.resolve()),
            "audit_sha256": data_audit_sha,
            "hashes_path": str(data_hashes_path.resolve()),
            "hashes_sha256": sha256_file(data_hashes_path),
            "pair_mode": data_audit["pair_mode"],
            "claim_level": data_audit["input_authorization"][
                "scientific_claim_level"
            ],
            "training_fault_tools": data_audit["training_fault_tools"],
        },
        "registered_arms": arms,
        "registered_training_seeds": seeds,
        "entries": entries,
        "official_test_used": False,
        "official_test_sealed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--arms",
        default=",".join(TRAINER_ARMS),
        help="Comma-separated trainer arm names; screen may use a strict subset.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in full.TRAINING_SEEDS),
    )
    parser.add_argument(
        "--user-judge-model",
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
    )
    parser.add_argument("--user-judge-revision", required=True)
    parser.add_argument(
        "--user-judge-alias",
        default="openai/v55-user-judge",
    )
    args = parser.parse_args()
    arms = parse_csv(
        args.arms,
        allowed=set(TRAINER_ARMS),
        label="arms",
    )
    seed_strings = parse_csv(
        args.seeds,
        allowed={str(seed) for seed in full.TRAINING_SEEDS},
        label="seeds",
    )
    result = build(
        training_root=args.training_root.resolve(),
        data_root=args.data_root.resolve(),
        output=args.output.resolve(),
        source_commit=args.source_commit,
        model_revision=args.model_revision,
        arms=arms,
        seeds=[int(seed) for seed in seed_strings],
        user_judge_model=args.user_judge_model,
        user_judge_revision=args.user_judge_revision,
        user_judge_alias=args.user_judge_alias,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "registry": str(args.output.resolve()),
                "arms": result["registered_arms"],
                "seeds": result["registered_training_seeds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
