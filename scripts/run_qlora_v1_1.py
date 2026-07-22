#!/usr/bin/env python3
"""Single-command, reproducible runner for the clean QLoRA baseline v1.1 rerun.

It deliberately creates v1.1 directories rather than overwriting the first
attempt.  Every GPU runs the same script and changes only --arm.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMS = ("random_success", "shortest_success", "recovery_balanced")
SEED = 20260722


def command(*parts):
    print("+", " ".join(map(str, parts)), flush=True)
    subprocess.run(list(map(str, parts)), check=True)


def require_build(output_dir: Path):
    summary = json.loads((output_dir / "build_summary.json").read_text())
    splits = summary.get("splits", {})
    if summary.get("seed") != SEED or splits.get("validation_examples") != 361 or splits.get("test_examples") != 900:
        raise RuntimeError(f"unexpected frozen data summary: {summary}")


def cuda_preflight(output_root: Path):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is missing; install a CUDA-enabled PyTorch wheel before training.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not run this QLoRA baseline on CPU or MPS.")
    output_root.mkdir(parents=True, exist_ok=True)
    environment = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    (output_root / "environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps({"cuda_preflight": environment}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "evaluate", "all"), default="all")
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v1_1"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v1_1"))
    parser.add_argument("--output-root", type=Path, default=Path("results/qlora_v1_1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    if args.stage in ("prepare", "all"):
        command(py, root / "run_data_baseline.py", "--data-dir", args.data_dir, "--output-dir", args.selection_dir)
        command(py, root / "build_sft_data.py", "--data-dir", args.data_dir, "--manifest-dir", args.selection_dir, "--output-dir", args.processed_dir)
    require_build(args.processed_dir)
    train_file = args.processed_dir / args.arm / "train.jsonl"
    validation_file = args.processed_dir / "shared" / "validation.jsonl"
    test_file = args.processed_dir / "shared" / "test.jsonl"
    formal_dir = args.output_root / args.arm

    if args.stage != "prepare":
        cuda_preflight(args.output_root)

    if args.stage in ("smoke", "all"):
        command(py, root / "train_qlora.py", "--train-file", train_file, "--validation-file", validation_file, "--output-dir", args.output_root / f"{args.arm}_smoke", "--smoke-test")
    if args.stage in ("train", "all"):
        command(py, root / "train_qlora.py", "--train-file", train_file, "--validation-file", validation_file, "--output-dir", formal_dir)
    if args.stage in ("evaluate", "all"):
        checkpoint = formal_dir / "checkpoint_final"
        if not checkpoint.exists():
            raise RuntimeError(f"missing formal checkpoint: {checkpoint}")
        command(py, root / "evaluate_tool_actions.py", "--test-file", test_file, "--adapter", checkpoint, "--output", formal_dir / "metrics.json", "--load-in-4bit", "--max-prompt-tokens", "512", "--progress-every", "25")
    print(json.dumps({"experiment": "qlora_v1_1", "arm": args.arm, "stage": args.stage, "output_dir": str(formal_dir)}, indent=2))


if __name__ == "__main__":
    main()
