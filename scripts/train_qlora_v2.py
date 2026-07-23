#!/usr/bin/env python3
"""Train one v2 QLoRA arm under a fixed-compute contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True, help="The fixed v2 train_schedule.jsonl, not train.jsonl.")
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA v2 requires CUDA. The Mac is the data/audit coordinator, not a CUDA trainer.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("QLoRA v2 freezes bfloat16 compute; this GPU/PyTorch build does not support it.")
    expected_rows = args.max_steps * args.batch_size * args.grad_accum
    rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file)
    if args.smoke_test:
        rows = rows[:16]
        validation_rows = validation_rows[:16]
        args.max_steps = 1
        args.grad_accum = min(args.grad_accum, len(rows))
    elif len(rows) != expected_rows:
        raise RuntimeError(f"fixed schedule must contain exactly {expected_rows} rows; found {len(rows)}")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def encode(row: dict) -> dict:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(row["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) != row["prompt_tokens"] or len(completion_ids) != row["completion_tokens"]:
            raise RuntimeError(
                f"tokenizer drift for {row['example_id']}: observed prompt/completion "
                f"{len(prompt_ids)}/{len(completion_ids)}, expected {row['prompt_tokens']}/{row['completion_tokens']}"
            )
        ids = prompt_ids + completion_ids
        if len(ids) != row["sequence_tokens"] or len(ids) > args.max_seq_len:
            raise RuntimeError(f"sequence contract violation for {row['example_id']}: {len(ids)} tokens")
        pad = args.max_seq_len - len(ids)
        return {
            "input_ids": ids + [tokenizer.pad_token_id] * pad,
            "attention_mask": [1] * len(ids) + [0] * pad,
            # Standard causal-language-model cross entropy is computed only on
            # the assistant tool-call completion; prompt and padding are masked.
            "labels": [-100] * len(prompt_ids) + completion_ids + [-100] * pad,
        }

    def dataset(source_rows: list[dict]) -> Dataset:
        raw = Dataset.from_list(source_rows)
        return raw.map(encode, remove_columns=raw.column_names, desc="token-contract-check")

    train_dataset = dataset(rows)
    validation_dataset = dataset(validation_rows)

    class FixedCollator:
        def __call__(self, features):
            return {key: torch.tensor([feature[key] for feature in features], dtype=torch.long) for key in ("input_ids", "attention_mask", "labels")}

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
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
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
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=FixedCollator(),
    )
    result = trainer.train()
    checkpoint = output / "checkpoint_final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16": True,
    }
    manifest = {
        "protocol": "qlora_v2",
        "objective": "completion_only_causal_language_model_cross_entropy",
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "padded_training_tokens": len(rows) * args.max_seq_len,
        "scheduled_nonpad_tokens": sum(row["sequence_tokens"] for row in rows),
        "scheduled_loss_tokens": sum(row["completion_tokens"] for row in rows),
        "train_file": str(args.train_file),
        "train_file_sha256": sha256_file(args.train_file),
        "validation_file": str(args.validation_file),
        "validation_file_sha256": sha256_file(args.validation_file),
        "smoke_test": args.smoke_test,
        "environment": environment,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "training_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    (output / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "metrics": result.metrics, "environment": environment}, indent=2))


if __name__ == "__main__":
    main()
