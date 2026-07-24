#!/usr/bin/env python3
"""Train the V4 matched clean-SFT checkpoint under the V3 compute contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20260722
MAX_SEQUENCE_TOKENS = 2048
FORMAL_BATCH_SIZE = 1
FORMAL_GRAD_ACCUM = 16
FORMAL_LEARNING_RATE = 1.0e-4
FROZEN_TAG = "v4-frozen-20260724-p1"
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRAIN_SHA256 = (
    "4b28da48082ef5bd3396e7df4b5b723c4efffe4b2e5438f47c8c2ca9d709f386"
)
EXPECTED_VALIDATION_SHA256 = (
    "e1942db9c2766c44f3f59c7baa47a7412d4b8a8992a185d306f346a8c5c9fc53"
)
EXPECTED_ROWS = 1088
EXPECTED_STEPS = 68
SMOKE_STEPS = 2
SMOKE_GRAD_ACCUM = 8
SMOKE_MAX_RESERVED_BYTES = 8_053_063_680  # 7.5 GiB


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def frozen_source_commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tagged = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{FROZEN_TAG}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != tagged:
        raise RuntimeError(f"HEAD {head} differs from {FROZEN_TAG}={tagged}")
    return head


def finite_training_audit(log_history: list[dict], metrics: dict) -> dict:
    losses: list[float] = []
    grad_norms: list[float] = []
    eval_losses: list[float] = []
    checked = 0
    for record in [*log_history, metrics]:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(
                        f"non-finite clean-SFT metric {key}: {value!r}"
                    )
                checked += 1
                if "loss" in key.lower():
                    losses.append(numeric)
                if key == "grad_norm":
                    grad_norms.append(numeric)
                if key == "eval_loss":
                    eval_losses.append(numeric)
    if not losses:
        raise RuntimeError("clean-SFT produced no auditable loss")
    if not grad_norms:
        raise RuntimeError("clean-SFT produced no auditable grad_norm")
    if not eval_losses:
        raise RuntimeError("clean-SFT produced no auditable validation loss")
    return {
        "finite": True,
        "numeric_values_checked": checked,
        "loss_values_checked": len(losses),
        "grad_norm_values_checked": len(grad_norms),
        "validation_loss_values_checked": len(eval_losses),
        "final_train_loss": float(metrics["train_loss"]),
        "final_validation_loss": eval_losses[-1],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        target = path / filename
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"incomplete clean-SFT checkpoint: {target}")
        file_hash = sha256_file(target)
        file_hashes[filename] = file_hash
        digest.update(filename.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), file_hashes


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--max-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=FORMAL_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=FORMAL_LEARNING_RATE)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or equal ':4096:8' "
            "before importing PyTorch"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import transformers
    from datasets import Dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    source_commit = frozen_source_commit()
    frozen_args = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_seq_len": MAX_SEQUENCE_TOKENS,
        "max_steps": EXPECTED_STEPS,
        "batch_size": FORMAL_BATCH_SIZE,
        "grad_accum": FORMAL_GRAD_ACCUM,
        "learning_rate": FORMAL_LEARNING_RATE,
    }
    for field, expected in frozen_args.items():
        if getattr(args, field) != expected:
            raise RuntimeError(
                f"V4 clean-SFT requires {field}={expected!r}; "
                f"found {getattr(args, field)!r}"
            )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"V4 requires Python 3.11; found {sys.version.split()[0]}"
        )
    frozen_packages = {
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "bitsandbytes": "0.46.0",
        "datasets": "3.6.0",
        "accelerate": "1.7.0",
    }
    for package, expected in frozen_packages.items():
        observed = package_version(package)
        if observed != expected:
            raise RuntimeError(
                f"V4 requires {package}=={expected}; found {observed}"
            )
    if not str(torch.__version__).startswith("2.7.1"):
        raise RuntimeError(
            f"V4 requires torch 2.7.1+cu128; found {torch.__version__}"
        )
    if torch.version.cuda != "12.8":
        raise RuntimeError(
            f"V4 requires CUDA runtime 12.8; found {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "V4 QLoRA requires CUDA; use the Mac only for preparation/audit."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V4 freezes bfloat16 compute, but BF16 is unavailable.")
    train_sha = sha256_file(args.train_file)
    validation_sha = sha256_file(args.validation_file)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f"clean train schedule hash drift: {train_sha}")
    if validation_sha != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError(f"clean validation hash drift: {validation_sha}")

    rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"clean schedule must contain {EXPECTED_ROWS} rows; found {len(rows)}"
        )
    if any(row.get("linked_tool_outcome") != "success" for row in rows):
        raise RuntimeError("clean schedule contains a failed-action label")
    if any(
        row.get("linked_tool_outcome") != "success"
        for row in validation_rows
    ):
        raise RuntimeError("clean validation contains a failed-action label")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise RuntimeError(
            f"clean-SFT output must be absent or empty: {args.output_dir}"
        )
    full_train_rows = rows
    if args.smoke_test:
        rows = sorted(
            rows,
            key=lambda row: (
                row["sequence_tokens"],
                row["example_id"],
            ),
            reverse=True,
        )[: SMOKE_STEPS * SMOKE_GRAD_ACCUM]
        validation_rows = validation_rows[:16]
        args.max_steps = SMOKE_STEPS
        args.grad_accum = SMOKE_GRAD_ACCUM

    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def encode(row: dict) -> dict:
        prompt_ids = tokenizer(
            row["prompt"],
            add_special_tokens=True,
        )["input_ids"]
        completion_ids = tokenizer(
            row["completion"] + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        if (
            len(prompt_ids) != row["prompt_tokens"]
            or len(completion_ids) != row["completion_tokens"]
        ):
            raise RuntimeError(
                f"tokenizer drift for {row['example_id']}: "
                f"{len(prompt_ids)}/{len(completion_ids)}"
            )
        ids = prompt_ids + completion_ids
        if len(ids) != row["sequence_tokens"] or len(ids) > args.max_seq_len:
            raise RuntimeError(
                f"sequence contract violation for {row['example_id']}"
            )
        pad = args.max_seq_len - len(ids)
        return {
            "input_ids": ids + [tokenizer.pad_token_id] * pad,
            "attention_mask": [1] * len(ids) + [0] * pad,
            "labels": (
                [-100] * len(prompt_ids)
                + completion_ids
                + [-100] * pad
            ),
        }

    def dataset(source_rows: list[dict]) -> Dataset:
        raw = Dataset.from_list(source_rows)
        return raw.map(
            encode,
            remove_columns=raw.column_names,
            desc="v4-clean-token-contract-check",
        )

    train_dataset = dataset(rows)
    validation_dataset = dataset(validation_rows)

    class FixedCollator:
        def __call__(self, features):
            return {
                key: torch.tensor(
                    [feature[key] for feature in features],
                    dtype=torch.long,
                )
                for key in ("input_ids", "attention_mask", "labels")
            }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer_state"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=1 if args.smoke_test else 4,
        eval_strategy="steps",
        eval_steps=args.max_steps,
        save_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=FixedCollator(),
    )
    result = trainer.train()
    training_audit = finite_training_audit(
        trainer.state.log_history,
        result.metrics,
    )
    checkpoint = output / "checkpoint_final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    output_fingerprint, output_file_hashes = checkpoint_fingerprint(checkpoint)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if args.smoke_test and peak_reserved > SMOKE_MAX_RESERVED_BYTES:
        raise RuntimeError(
            "clean-SFT smoke peak reserved memory exceeds the frozen "
            f"7.5 GiB unattended-run gate: {peak_reserved} bytes"
        )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16": True,
        "deterministic_algorithms_enabled":
            torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    manifest = {
        "protocol": "qlora_v4_clean_sft",
        "source_tag": FROZEN_TAG,
        "source_commit": source_commit,
        "arm": "clean_sft",
        "objective": "matched_clean_completion_only_cross_entropy",
        "failed_action_labels": 0,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_dropout": 0.05,
        "padded_training_tokens": len(rows) * args.max_seq_len,
        "scheduled_nonpad_tokens": sum(
            row["sequence_tokens"] for row in rows
        ),
        "scheduled_loss_tokens": sum(
            row["completion_tokens"] for row in rows
        ),
        "formal_schedule_rows": len(full_train_rows),
        "train_file": str(args.train_file),
        "train_file_sha256": train_sha,
        "validation_file": str(args.validation_file),
        "validation_file_sha256": validation_sha,
        "smoke_test": args.smoke_test,
        "held_out_test_accessed": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "loss_audit": training_audit,
        "output_checkpoint": {
            "path": str(checkpoint),
            "checkpoint_fingerprint": output_fingerprint,
            "file_sha256": output_file_hashes,
        },
        "environment": environment,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "training_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "metrics": result.metrics,
                "environment": environment,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
