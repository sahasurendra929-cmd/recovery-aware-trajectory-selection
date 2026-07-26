#!/usr/bin/env python3
"""Immutable post-generation snapshot contract for the V5.3 12-hour run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "v5_3_12h_source_snapshot_c3_v1"
RECEIPT_NAME = "V5_3_12H_SOURCE_SNAPSHOT_C3.json"
EXPECTED_FILES = 15
EXPECTED_CONTRACTS = 3
EXPECTED_RESULTS = 12
EXPECTED_CASES = 288


class SourceSnapshotError(RuntimeError):
    """The frozen source snapshot is absent, malformed, or has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceSnapshotError(
            f"invalid source snapshot JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SourceSnapshotError("source snapshot receipt must be an object")
    return value


def validate_source_snapshot(
    *,
    raw_dir: Path,
    receipt_path: Path,
    expected_generation_commit: str,
) -> dict[str, Any]:
    """Recompute the exact pre-repair raw snapshot and return its identity."""

    receipt = _load_object(receipt_path)
    file_sha256 = receipt.get("file_sha256")
    strict_judge = receipt.get("strict_judge_mapping_sha256_by_contract")
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("source_generation_commit")
        != expected_generation_commit
        or receipt.get("exact_raw_cases") != EXPECTED_CASES
        or receipt.get("generation_contracts") != EXPECTED_CONTRACTS
        or receipt.get("result_files") != EXPECTED_RESULTS
        or receipt.get("official_test_used") is not False
        or receipt.get("official_test_sealed") is not True
        or not isinstance(file_sha256, dict)
        or len(file_sha256) != EXPECTED_FILES
        or not isinstance(strict_judge, dict)
        or len(strict_judge) != EXPECTED_CONTRACTS
    ):
        raise SourceSnapshotError("source snapshot receipt identity drift")
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(digest, str)
        or len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in file_sha256.items()
    ):
        raise SourceSnapshotError("source snapshot file ledger is malformed")
    observed_names = {
        path.name for path in raw_dir.glob("*.json") if path.is_file()
    }
    if observed_names != set(file_sha256):
        raise SourceSnapshotError("source snapshot filename set drift")
    for name, expected_digest in file_sha256.items():
        if sha256_file(raw_dir / name) != expected_digest:
            raise SourceSnapshotError(f"source snapshot hash drift: {name}")
    sha256sum_stream = "".join(
        f"{file_sha256[name]}  {name}\n" for name in sorted(file_sha256)
    )
    if (
        hashlib.sha256(sha256sum_stream.encode("utf-8")).hexdigest()
        != receipt.get("sha256sum_stream_sha256")
    ):
        raise SourceSnapshotError("source snapshot aggregate hash drift")

    contracts = sorted(
        name for name in file_sha256 if name.startswith("run_contract.")
    )
    results = sorted(set(file_sha256) - set(contracts))
    if (
        len(contracts) != EXPECTED_CONTRACTS
        or len(results) != EXPECTED_RESULTS
        or set(strict_judge) != set(contracts)
    ):
        raise SourceSnapshotError("source snapshot contract/result set drift")
    declared_results: dict[str, str] = {}
    for name in contracts:
        contract = _load_object(raw_dir / name)
        declared = contract.get("result_sha256")
        evidence = contract.get("strict_judge_audit_evidence")
        if (
            contract.get("status") != "COMPLETE"
            or contract.get("source_commit") != expected_generation_commit
            or contract.get("official_test_used") is not False
            or not isinstance(declared, dict)
            or not isinstance(evidence, dict)
            or evidence.get("status") != "PASS"
            or evidence.get("canonical_mapping_sha256")
            != strict_judge[name]
            or set(declared_results) & set(declared)
        ):
            raise SourceSnapshotError(
                f"source generation contract drift: {name}"
            )
        declared_results.update(declared)
    expected_result_hashes = {
        name: file_sha256[name] for name in results
    }
    if declared_results != expected_result_hashes:
        raise SourceSnapshotError(
            "source contracts do not bind the frozen result ledger"
        )

    exact_cases = 0
    for name in results:
        payload = _load_object(raw_dir / name)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise SourceSnapshotError(
                f"source result simulations are malformed: {name}"
            )
        exact_cases += len(simulations)
    if exact_cases != EXPECTED_CASES:
        raise SourceSnapshotError(
            f"source snapshot case count drift: {exact_cases}"
        )

    return {
        "protocol": PROTOCOL,
        "status": "PASS",
        "source_generation_commit": expected_generation_commit,
        "receipt_sha256": sha256_file(receipt_path),
        "file_ledger_sha256": canonical_sha256(file_sha256),
        "sha256sum_stream_sha256": receipt[
            "sha256sum_stream_sha256"
        ],
        "strict_judge_mapping_sha256_by_contract": dict(
            sorted(strict_judge.items())
        ),
        "exact_files": len(file_sha256),
        "exact_generation_contracts": len(contracts),
        "exact_result_files": len(results),
        "exact_raw_cases": exact_cases,
        "official_test_used": False,
        "official_test_sealed": True,
    }
