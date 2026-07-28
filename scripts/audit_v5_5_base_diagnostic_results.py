#!/usr/bin/env python3
"""Audit preregistered and accidental V5.5.1 base diagnostic artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REGISTERED_SEED = 20260815
ACCIDENTAL_SEEDS = (20260816, 20260817)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    audited = []
    all_rows = []
    hashes_verified = 0
    for evaluation_seed in (REGISTERED_SEED, *ACCIDENTAL_SEEDS):
        for shard_index in range(3):
            shard = (
                args.results_root
                / "evaluation"
                / "base_control"
                / "20260805"
                / str(evaluation_seed)
                / f"shard-{shard_index}"
            )
            contract_path = shard / "run_contract.json"
            metrics_path = shard / "metrics.json"
            rows_path = shard / "rows.jsonl"
            contract = read_json(contract_path)
            metrics = read_json(metrics_path)
            rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            checks = {
                "contract_complete": contract.get("status") == "COMPLETE",
                "metrics_pass": metrics.get("status") == "PASS",
                "rows_14": len(rows) == 14 and metrics.get("rows") == 14,
                "source_commit_exact": (
                    contract.get("evaluation_source_commit") == SOURCE_COMMIT
                ),
                "seed_exact": (
                    contract.get("evaluation_seed") == evaluation_seed
                    and all(
                        row.get("evaluation_seed") == evaluation_seed
                        for row in rows
                    )
                ),
                "official_test_unused": (
                    contract.get("official_test_used") is False
                    and metrics.get("official_test_used") is False
                    and all(
                        row.get("official_test_used") is False for row in rows
                    )
                ),
            }
            for name, expected in contract.get("result_sha256", {}).items():
                if sha256(shard / name) != expected:
                    raise RuntimeError(f"result hash drift: {shard / name}")
                hashes_verified += 1
            if not all(checks.values()):
                raise RuntimeError(f"shard audit failed: {shard}: {checks}")
            all_rows.extend(rows)
            audited.append(
                {
                    "evaluation_seed": evaluation_seed,
                    "scope": (
                        "preregistered"
                        if evaluation_seed == REGISTERED_SEED
                        else "accidental_exploratory_excluded"
                    ),
                    "shard": shard_index,
                    "checks": checks,
                    "metrics_sha256": sha256(metrics_path),
                    "rows_sha256": sha256(rows_path),
                    "run_contract_sha256": sha256(contract_path),
                }
            )

    keys = [
        (
            row["evaluation_seed"],
            row["domain"],
            row["task_id"],
            row["condition"],
        )
        for row in all_rows
    ]
    seed_counts = Counter(row["evaluation_seed"] for row in all_rows)
    checks = {
        "shards_9": len(audited) == 9,
        "rows_126": len(all_rows) == 126,
        "unique_rows_126": len(set(keys)) == 126,
        "rows_42_per_seed": seed_counts == {
            REGISTERED_SEED: 42,
            ACCIDENTAL_SEEDS[0]: 42,
            ACCIDENTAL_SEEDS[1]: 42,
        },
        "result_hashes_verified_54": hashes_verified == 54,
        "official_test_unused": all(
            row.get("official_test_used") is False for row in all_rows
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
        "claim_level": "diagnostic_only",
        "checks": checks,
        "preregistered_evaluation_seed": REGISTERED_SEED,
        "accidental_evaluation_seeds_excluded": list(ACCIDENTAL_SEEDS),
        "shards": audited,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "PASS":
        raise SystemExit("integrity audit failed")


if __name__ == "__main__":
    main()
