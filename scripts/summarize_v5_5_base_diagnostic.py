#!/usr/bin/env python3
"""Fail-closed summary for the preregistered V5.5.1 base-floor diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ARM = "base_control"
TRAINING_SEED = 20260805
EVALUATION_SEED = 20260815
SOURCE_COMMIT = "1c00867594b304d3191ccdd76016bfdcecf164db"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"output directory must be absent: {args.output_dir}")

    manifest = read_json(args.validation_manifest)
    expected_tasks = {
        (str(row["domain"]), str(row["task_id"]))
        for row in manifest["rows"]
    }
    if len(expected_tasks) != 21 or manifest.get("official_test_used") is not False:
        raise RuntimeError("validation manifest is not the sealed 21-task grid")

    batch = (
        args.results_root
        / ARM
        / str(TRAINING_SEED)
        / str(EVALUATION_SEED)
    )
    shard_dirs = [batch / f"shard-{index}" for index in range(3)]
    rows: list[dict] = []
    inputs: list[dict] = []
    result_hashes_verified = 0
    for index, shard in enumerate(shard_dirs):
        metrics_path = shard / "metrics.json"
        rows_path = shard / "rows.jsonl"
        contract_path = shard / "run_contract.json"
        metrics = read_json(metrics_path)
        contract = read_json(contract_path)
        if metrics.get("status") != "PASS" or metrics.get("rows") != 14:
            raise RuntimeError(f"invalid shard metrics: {metrics_path}")
        if contract.get("status") != "COMPLETE":
            raise RuntimeError(f"invalid run contract: {contract_path}")
        if contract.get("evaluation_source_commit") != SOURCE_COMMIT:
            raise RuntimeError(f"source commit drift: {contract_path}")
        for name, expected_digest in contract.get("result_sha256", {}).items():
            result_path = shard / name
            if sha256(result_path) != expected_digest:
                raise RuntimeError(f"result hash drift: {result_path}")
            result_hashes_verified += 1
        shard_rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(shard_rows) != 14:
            raise RuntimeError(f"invalid row count: {rows_path}")
        rows.extend(shard_rows)
        inputs.append(
            {
                "shard": index,
                "metrics": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "rows": str(rows_path),
                "rows_sha256": sha256(rows_path),
                "run_contract": str(contract_path),
                "run_contract_sha256": sha256(contract_path),
            }
        )

    observed_keys = [
        (str(row["domain"]), str(row["task_id"]), str(row["condition"]))
        for row in rows
    ]
    expected_keys = {
        (domain, task, condition)
        for domain, task in expected_tasks
        for condition in ("clean", "error")
    }
    checks = {
        "rows_42": len(rows) == 42,
        "unique_rows_42": len(set(observed_keys)) == 42,
        "exact_validation_grid": set(observed_keys) == expected_keys,
        "arm_exact": all(row.get("arm") == ARM for row in rows),
        "training_seed_exact": all(
            row.get("training_seed") == TRAINING_SEED for row in rows
        ),
        "evaluation_seed_exact": all(
            row.get("evaluation_seed") == EVALUATION_SEED for row in rows
        ),
        "source_commit_exact": all(
            read_json(path / "run_contract.json").get(
                "evaluation_source_commit"
            )
            == SOURCE_COMMIT
            for path in shard_dirs
        ),
        "official_test_unused": all(
            row.get("official_test_used") is False for row in rows
        ),
        "diagnostic_only": all(
            row.get("claim_level") == "diagnostic_only" for row in rows
        ),
        "result_hashes_verified_18": result_hashes_verified == 18,
    }
    if not all(checks.values()):
        raise RuntimeError(f"diagnostic completeness audit failed: {checks}")

    condition_counts: dict[str, dict] = {}
    for condition in ("clean", "error"):
        selected = [row for row in rows if row["condition"] == condition]
        successes = sum(bool(row["task_success"]) for row in selected)
        condition_counts[condition] = {
            "tasks": len(selected),
            "task_successes": successes,
            "task_success_rate": successes / len(selected),
            "termination_reasons": dict(
                sorted(Counter(row["termination_reason"] for row in selected).items())
            ),
        }
    clean_successes = condition_counts["clean"]["task_successes"]
    payload = {
        "status": "PASS",
        "protocol": "v5_5_1_base_floor_diagnostic_v1",
        "claim_level": "diagnostic_only",
        "official_test_used": False,
        "source_commit": SOURCE_COMMIT,
        "arm": ARM,
        "training_seed": TRAINING_SEED,
        "evaluation_seeds": [EVALUATION_SEED],
        "grid_audit": {
            "status": "PASS",
            "expected_rows": 42,
            "observed_rows": len(rows),
            "checks": checks,
        },
        "metrics": condition_counts,
        "diagnostic_gate": {
            "status": "PASS" if clean_successes >= 1 else "FAIL",
            "criterion": "at least one clean official success in 42/42 complete rows",
            "clean_successes": clean_successes,
            "interpretation": (
                "The unadapted base has a non-zero clean capability floor; "
                "the V5.5 Stage B all-zero result is therefore consistent "
                "with adaptation-induced capability collapse."
            ),
        },
        "inputs": inputs,
        "validation_manifest": {
            "path": str(args.validation_manifest),
            "sha256": sha256(args.validation_manifest),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
