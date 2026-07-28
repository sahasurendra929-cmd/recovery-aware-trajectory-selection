#!/usr/bin/env python3
"""Generate frozen V5.5 natural clean trajectories from inner-train only.

This is the missing Stage-C entry point.  It deliberately delegates trajectory
execution to the already audited Stage-1 generator, but fixes the condition to
``clean`` and records a V5.5 receipt.  Validation and official-test manifests
are rejected by the delegated generator and by the split/manifest contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "v5_5_natural_raw_generation_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-api-base", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-revision", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--judge-api-base")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite natural raw output: {output}")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_v5_sft_causal_generate.py"),
        "--tau2-root", str(args.tau2_root.resolve()),
        "--split-manifest", str(args.split_manifest.resolve()),
        "--manifest", str(args.generation_manifest.resolve()),
        "--dynamic-audit", str(args.dynamic_audit.resolve()),
        "--output-dir", str(output),
        "--teacher-model", args.teacher_model,
        "--teacher-revision", args.teacher_revision,
        "--teacher-api-base", args.teacher_api_base,
        "--user-model", args.user_model,
        "--user-revision", args.user_revision,
        "--user-api-base", args.user_api_base,
        "--judge-model", args.user_model,
        "--judge-revision", args.user_revision,
        "--judge-api-base", args.judge_api_base or args.user_api_base,
        "--teacher-mode", "ground_truth",
        "--condition", "clean",
        "--num-trials", str(args.num_trials),
        "--seed", str(args.seed),
        "--shard-index", str(args.shard_index),
        "--num-shards", str(args.num_shards),
        "--max-steps", "60",
        "--timeout", "900",
        "--max-tokens", "512",
        "--expected-source-commit", args.source_commit,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "condition": "clean",
        "teacher_mode": "ground_truth",
        "source_commit": args.source_commit,
        "split_manifest_sha256": sha256_file(args.split_manifest.resolve()),
        "generation_manifest_sha256": sha256_file(args.generation_manifest.resolve()),
        "dynamic_audit_sha256": sha256_file(args.dynamic_audit.resolve()),
        "seed": args.seed,
        "num_trials": args.num_trials,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "official_test_used": False,
    }
    (output / "v5_5_natural_generation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
