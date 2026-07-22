#!/usr/bin/env python3
"""CUDA QLoRA trainer for the frozen baseline v1 text protocol."""
from __future__ import annotations

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--validation-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available(): raise RuntimeError("QLoRA v1 requires a CUDA GPU; use the Mac only for data/evaluation coordination.")
    set_seed(args.seed)
    rows = [json.loads(x) for x in args.train_file.read_text().splitlines() if x.strip()]
    validation_rows = [json.loads(x) for x in args.validation_file.read_text().splitlines() if x.strip()]
    if args.smoke_test:
        rows = rows[:10]
        validation_rows = validation_rows[:10]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def encode(row):
        completion_ids = tokenizer(row["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"][-args.max_seq_len:]
        # Keep the target labels intact.  If context is long, discard its oldest
        # tokens rather than silently producing an all--100 label sequence.
        prompt_budget = max(0, args.max_seq_len - len(completion_ids))
        previous_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=True, truncation=True, max_length=prompt_budget)["input_ids"] if prompt_budget else []
        tokenizer.truncation_side = previous_side
        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels}
    def make_dataset(source_rows):
        source = Dataset.from_list(source_rows)
        return source.map(encode, remove_columns=source.column_names)
    dataset = make_dataset(rows)
    validation_dataset = make_dataset(validation_rows)
    class Collator:
        def __call__(self, features):
            max_len=max(len(x["input_ids"]) for x in features); out={"input_ids":[],"attention_mask":[],"labels":[]}
            for x in features:
                pad=max_len-len(x["input_ids"]); out["input_ids"].append(x["input_ids"]+[tokenizer.pad_token_id]*pad); out["attention_mask"].append(x["attention_mask"]+[0]*pad); out["labels"].append(x["labels"]+[-100]*pad)
            return {k: torch.tensor(v) for k,v in out.items()}
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    train_args = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate, logging_steps=5, save_strategy="epoch", eval_strategy="epoch", report_to=[], fp16=True, seed=args.seed, remove_unused_columns=False)
    Trainer(model=model, args=train_args, train_dataset=dataset, eval_dataset=validation_dataset, data_collator=Collator()).train()
    args.output_dir.mkdir(parents=True, exist_ok=True); model.save_pretrained(args.output_dir / "checkpoint_final"); tokenizer.save_pretrained(args.output_dir / "checkpoint_final")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(vars(args), default=str, indent=2))

if __name__ == "__main__": main()
