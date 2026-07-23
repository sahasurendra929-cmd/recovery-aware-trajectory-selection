#!/usr/bin/env python3
"""Resumable v2 next-tool-call evaluation with task-cluster uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC_KEYS = ("json_valid", "tool_name_correct", "arguments_exact", "full_call_exact")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return " ".join(value.split())
    if isinstance(value, dict):
        return {key: canonical_arguments(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_arguments(item) for item in value]
    return value


def normalize_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or "arguments" not in value:
        return None
    return {"name": value["name"], "arguments": canonical_arguments(value["arguments"])}


def parse_call(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        normalized = normalize_call(value)
        if normalized is not None:
            return normalized
    return None


def mean(items: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(item[key]) for item in items) / len(items) if items else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def task_cluster_bootstrap(items: list[dict[str, Any]], seed: int, samples: int) -> dict[str, list[float | None]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_key"]].append(item)
    tasks = sorted(by_task)
    if not tasks:
        return {key: [None, None] for key in METRIC_KEYS}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for _ in range(samples):
        sampled = [task for _ in tasks for task in [rng.choice(tasks)]]
        values = [item for task in sampled for item in by_task[task]]
        for key in METRIC_KEYS:
            draws[key].append(mean(values, key))
    return {key: [percentile(values, 0.025), percentile(values, 0.975)] for key, values in draws.items()}


def group_metrics(items: list[dict[str, Any]], seed: int, bootstrap_samples: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_key"]].append(item)
    task_macro = {
        key: sum(mean(task_items, key) for task_items in by_task.values()) / len(by_task) if by_task else None
        for key in METRIC_KEYS
    }
    return {
        "examples": len(items),
        "tasks": len(by_task),
        "micro": {key: mean(items, key) for key in METRIC_KEYS},
        "task_macro": task_macro,
        "task_cluster_bootstrap_95": task_cluster_bootstrap(items, seed, bootstrap_samples),
    }


def checkpoint_fingerprint(adapter: Path | None) -> str:
    if adapter is None:
        return "base_model"
    paths = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete adapter checkpoint: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Omit for the zero-shot base-model control.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=1664)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke testing only; formal aggregation rejects limited evaluations.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.test_file)
    formal_examples = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    predictions_path = args.output.with_name(args.output.stem + ".predictions.jsonl")
    contract_path = args.output.with_name(args.output.stem + ".contract.json")
    fingerprint = checkpoint_fingerprint(args.adapter)
    contract = {
        "protocol": "qlora_v2",
        "test_file_sha256": sha256_file(args.test_file),
        "formal_test_examples": formal_examples,
        "evaluated_examples": len(rows),
        "model": args.model,
        "model_revision": args.model_revision,
        "checkpoint_fingerprint": fingerprint,
        "base_model_loading": "nf4_4bit",
        "max_prompt_tokens": args.max_prompt_tokens,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "batch_size": 1},
        "limited": bool(args.limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError("existing evaluation contract differs; use a new output path instead of mixing predictions")
    elif predictions_path.exists():
        raise RuntimeError("predictions exist without a contract; use a new output path")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    if args.dry_run:
        dry = {**contract, "mode": "dry_run_no_model_loaded"}
        args.output.write_text(json.dumps(dry, indent=2), encoding="utf-8")
        print(json.dumps(dry, indent=2))
        return

    completed: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        if not args.resume:
            raise RuntimeError(f"{predictions_path} already exists; pass --resume only if its contract matches")
        for item in read_jsonl(predictions_path):
            completed[item["example_id"]] = item
    elif args.resume:
        predictions_path.touch()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("formal v2 generation evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter) if args.adapter else base
    model.eval()

    with predictions_path.open("a", encoding="utf-8", buffering=1) as handle:
        for index, row in enumerate(rows, start=1):
            if row["example_id"] in completed:
                continue
            inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
            observed_tokens = inputs["input_ids"].shape[1]
            if observed_tokens != row["prompt_tokens"] or observed_tokens > args.max_prompt_tokens:
                raise RuntimeError(
                    f"prompt token contract violation for {row['example_id']}: "
                    f"observed={observed_tokens}, recorded={row['prompt_tokens']}, cap={args.max_prompt_tokens}"
                )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output_ids[0][observed_tokens:], skip_special_tokens=True)
            prediction = parse_call(generated)
            target = normalize_call(json.loads(row["completion"]))
            scored = {
                "example_id": row["example_id"],
                "trace_id": row["trace_id"],
                "task_key": row["task_key"],
                "source": row["source"],
                "recovery_mode": row["recovery_mode"],
                "prior_error_type": row.get("prior_error_type"),
                "generated_text": generated,
                "prediction": prediction,
                "target": target,
                "json_valid": prediction is not None,
                "tool_name_correct": bool(prediction and prediction["name"] == target["name"]),
                "arguments_exact": bool(prediction and prediction["arguments"] == target["arguments"]),
                "full_call_exact": prediction == target,
            }
            handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row["example_id"]] = scored
            if args.progress_every and index % args.progress_every == 0:
                print(f"evaluated {len(completed)}/{len(rows)}", flush=True)

    ordered = [completed[row["example_id"]] for row in rows if row["example_id"] in completed]
    if len(ordered) != len(rows):
        raise RuntimeError(f"evaluation incomplete: {len(ordered)}/{len(rows)}")
    groups = {
        "overall": ordered,
        "recovery": [item for item in ordered if item["recovery_mode"] != "none"],
        "agent_initiated": [item for item in ordered if item["recovery_mode"] == "agent_initiated"],
        "user_assisted": [item for item in ordered if item["recovery_mode"] == "user_assisted"],
    }
    result = {
        **contract,
        "claim_boundary": "offline held-out next-tool-call imitation; not executable agent success",
        "groups": {
            name: group_metrics(items, args.seed + offset, args.bootstrap_samples)
            for offset, (name, items) in enumerate(groups.items())
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
