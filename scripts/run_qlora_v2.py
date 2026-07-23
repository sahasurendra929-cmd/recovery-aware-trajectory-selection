#!/usr/bin/env python3
"""Isolated single-GPU runner for the QLoRA v2 experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMS = ("random_success", "shortest_success", "recovery_coverage")


def command(*parts: object) -> None:
    print("+", " ".join(map(str, parts)), flush=True)
    subprocess.run(list(map(str, parts)), check=True)


def load_contract(processed_dir: Path) -> dict:
    path = processed_dir / "build_summary.json"
    if not path.exists():
        raise RuntimeError(f"missing v2 data contract: {path}; run --stage prepare first")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("protocol") != "qlora_v2":
        raise RuntimeError(f"wrong data protocol in {path}")
    training = contract["training"]
    expected = {"gradient_accumulation": 16, "pad_to_max_sequence_tokens": 2048}
    for key, value in expected.items():
        if training.get(key) != value:
            raise RuntimeError(f"v2 contract drift for training.{key}: {training.get(key)!r} != {value!r}")
    if training.get("microbatches") != training.get("gradient_accumulation") * training.get("optimizer_steps"):
        raise RuntimeError("v2 microbatch/optimizer-step contract is inconsistent")
    totals = {arm["selected_sft_tokens"] for arm in contract["arms"]}
    if len(totals) != 1:
        raise RuntimeError(f"selected SFT token budgets differ: {totals}")
    return contract


def cuda_preflight(output_root: Path) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is missing. Install the CUDA build before starting v2 training.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Do not run formal v2 training/evaluation on CPU or MPS.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen v2 contract requires CUDA bfloat16 support.")
    output_root.mkdir(parents=True, exist_ok=True)
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    (output_root / "preflight_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps({"cuda_preflight": environment}, indent=2), flush=True)


def run_arm(py: str, scripts: Path, arm: str, stage: str, processed: Path, output_root: Path, model: str, revision: str, training: dict) -> None:
    train_file = processed / arm / "train_schedule.jsonl"
    validation_file = processed / "shared" / "validation.jsonl"
    test_file = processed / "shared" / "test.jsonl"
    formal_dir = output_root / arm
    common_train = (
        py, scripts / "train_qlora_v2.py",
        "--train-file", train_file,
        "--validation-file", validation_file,
        "--model", model,
        "--model-revision", revision,
        "--max-seq-len", training["pad_to_max_sequence_tokens"],
        "--max-steps", training["optimizer_steps"],
        "--grad-accum", training["gradient_accumulation"],
    )
    if stage in ("smoke", "all"):
        command(*common_train, "--output-dir", output_root / "smoke" / arm, "--smoke-test")
    if stage in ("train", "all"):
        command(*common_train, "--output-dir", formal_dir)
    if stage in ("evaluate", "all"):
        checkpoint = formal_dir / "checkpoint_final"
        if not checkpoint.exists():
            raise RuntimeError(f"missing v2 checkpoint for {arm}: {checkpoint}")
        command(
            py, scripts / "evaluate_tool_actions_v2.py",
            "--test-file", test_file,
            "--adapter", checkpoint,
            "--output", formal_dir / "metrics.json",
            "--model", model,
            "--model-revision", revision,
            "--resume",
        )


def evaluate_base(py: str, scripts: Path, processed: Path, output_root: Path, model: str, revision: str) -> None:
    command(
        py, scripts / "evaluate_tool_actions_v2.py",
        "--test-file", processed / "shared" / "test.jsonl",
        "--output", output_root / "base_model" / "metrics.json",
        "--model", model,
        "--model-revision", revision,
        "--resume",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=(*ARMS, "base_model", "all"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "evaluate", "all"), default="all")
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v2"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/qlora_v2"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    py = sys.executable

    if args.stage in ("prepare", "all"):
        command(
            py, scripts / "prepare_qlora_v2.py",
            "--data-dir", args.data_dir,
            "--selection-dir", args.selection_dir,
            "--processed-dir", args.processed_dir,
            "--model", args.model,
            "--model-revision", args.model_revision,
        )
    contract = load_contract(args.processed_dir)
    model = contract["model"]["name"]
    revision = contract["model"]["resolved_revision"]
    training = contract["training"]
    if model != args.model:
        raise RuntimeError(f"prepared model {model!r} differs from requested model {args.model!r}")

    if args.stage == "prepare":
        print(json.dumps({"experiment": "qlora_v2", "stage": "prepare", "contract": str(args.processed_dir / "build_summary.json")}, indent=2))
        return
    cuda_preflight(args.output_root)

    if args.arm == "base_model":
        if args.stage not in ("evaluate", "all"):
            raise RuntimeError("base_model supports only --stage evaluate or all")
        evaluate_base(py, scripts, args.processed_dir, args.output_root, model, revision)
    elif args.arm == "all":
        for arm in ARMS:
            run_arm(py, scripts, arm, args.stage, args.processed_dir, args.output_root, model, revision, training)
        if args.stage in ("evaluate", "all"):
            evaluate_base(py, scripts, args.processed_dir, args.output_root, model, revision)
    else:
        run_arm(py, scripts, args.arm, args.stage, args.processed_dir, args.output_root, model, revision, training)
    print(json.dumps({"experiment": "qlora_v2", "arm": args.arm, "stage": args.stage, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
