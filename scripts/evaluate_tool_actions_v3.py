#!/usr/bin/env python3
"""Frozen V2 next-tool-call evaluator with strict, recoverable V3 resume.

The model loading, greedy generation, parsing, scoring, grouping, and
task-cluster bootstrap semantics intentionally match
``evaluate_tool_actions_v2.py``.  V3 only adds resume-integrity checks:

* a malformed final line is recoverable only when it lacks a terminating
  newline (the signature of an interrupted append);
* malformed middle lines, duplicate example IDs, and IDs outside the frozen
  test set are fatal;
* every tail repair is appended to a separate audit log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL = "qlora_v3"
FROZEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FROZEN_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FROZEN_FORMAL_EXAMPLES = 959
FROZEN_MAX_PROMPT_TOKENS = 1664
FROZEN_MAX_NEW_TOKENS = 128
FROZEN_SEED = 20260722
FROZEN_TEST_SHA256 = "0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96"
METRIC_KEYS = ("json_valid", "tool_name_correct", "arguments_exact", "full_call_exact")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _append_recovery_audit(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_resumable_predictions(path: Path, audit_path: Path) -> list[dict[str, Any]]:
    """Read predictions and repair only an unterminated malformed final line."""
    payload = path.read_bytes()
    if not payload:
        return []
    physical_lines = payload.splitlines(keepends=True)
    nonblank = [
        index
        for index, line in enumerate(physical_lines)
        if line.rstrip(b"\r\n").strip()
    ]
    if not nonblank:
        raise RuntimeError(f"{path} contains only blank data; refusing ambiguous resume")
    last_nonblank = nonblank[-1]
    rows: list[dict[str, Any]] = []
    offset = 0
    repaired_partial = False
    for index, raw_line in enumerate(physical_lines):
        line_start = offset
        offset += len(raw_line)
        encoded = raw_line.rstrip(b"\r\n")
        if not encoded.strip():
            raise RuntimeError(f"blank line in predictions at physical line {index + 1}")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Formal runs are on Windows and use CRLF.  A lone trailing CR can
            # itself be the first half of an interrupted newline write.
            unterminated = not raw_line.endswith(b"\n")
            if index != last_nonblank or not unterminated:
                raise RuntimeError(
                    f"corrupt predictions at {path}:{index + 1}; only an "
                    "unterminated malformed final line is recoverable"
                ) from exc
            removed = payload[line_start:]
            event = {
                "event": "truncated_incomplete_final_prediction_line",
                "utc": datetime.now(timezone.utc).isoformat(),
                "predictions_file": str(path),
                "old_size_bytes": len(payload),
                "new_size_bytes": line_start,
                "removed_bytes": len(removed),
                "removed_sha256": hashlib.sha256(removed).hexdigest(),
                "completed_rows_retained": len(rows),
            }
            with path.open("r+b") as handle:
                handle.truncate(line_start)
                handle.flush()
                os.fsync(handle.fileno())
            _append_recovery_audit(audit_path, event)
            print(json.dumps({"resume_recovery": event}, ensure_ascii=False), flush=True)
            repaired_partial = True
            break
        if not isinstance(value, dict):
            raise RuntimeError(f"prediction at {path}:{index + 1} is not a JSON object")
        rows.append(value)
    if not repaired_partial and not payload.endswith(b"\n"):
        # A complete JSON object without its delimiter is valid data, but
        # appending the next object directly would corrupt it.  Complete only
        # the delimiter and record the repair.
        event = {
            "event": "completed_missing_final_newline",
            "utc": datetime.now(timezone.utc).isoformat(),
            "predictions_file": str(path),
            "old_size_bytes": len(payload),
            "new_size_bytes": len(payload) + 1,
            "removed_bytes": 0,
            "completed_rows_retained": len(rows),
        }
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _append_recovery_audit(audit_path, event)
        print(json.dumps({"resume_recovery": event}, ensure_ascii=False), flush=True)
    return rows


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


def score_generated(row: dict[str, Any], generated: str) -> dict[str, Any]:
    prediction = parse_call(generated)
    target = normalize_call(json.loads(row["completion"]))
    if target is None:
        raise RuntimeError(f"invalid frozen target call for {row['example_id']}")
    return {
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


def validate_completed_prediction(item: dict[str, Any], row: dict[str, Any]) -> None:
    generated = item.get("generated_text")
    if not isinstance(generated, str):
        raise RuntimeError(
            f"resume prediction generated_text is not a string for {row['example_id']}"
        )
    expected = score_generated(row, generated)
    for key in (
        "example_id",
        "trace_id",
        "task_key",
        "source",
        "recovery_mode",
        "prior_error_type",
        "prediction",
        "target",
        *METRIC_KEYS,
    ):
        if item.get(key) != expected.get(key):
            raise RuntimeError(
                f"resume prediction integrity failure for {row['example_id']}: field {key!r} differs"
            )


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


def task_cluster_bootstrap(
    items: list[dict[str, Any]],
    seed: int,
    samples: int,
) -> dict[str, list[float | None]]:
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
    return {
        key: [percentile(values, 0.025), percentile(values, 0.975)]
        for key, values in draws.items()
    }


def group_metrics(
    items: list[dict[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[item["task_key"]].append(item)
    task_macro = {
        key: sum(mean(task_items, key) for task_items in by_task.values()) / len(by_task)
        if by_task
        else None
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
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"incomplete adapter checkpoint: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def validate_frozen_cli(args: argparse.Namespace, formal_examples: int) -> None:
    expected = {
        "model": FROZEN_MODEL,
        "model_revision": FROZEN_MODEL_REVISION,
        "max_prompt_tokens": FROZEN_MAX_PROMPT_TOKENS,
        "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
        "seed": FROZEN_SEED,
    }
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise RuntimeError(f"frozen V3 evaluator requires {key}={value!r}")
    if formal_examples != FROZEN_FORMAL_EXAMPLES:
        raise RuntimeError(
            f"frozen V3 test requires {FROZEN_FORMAL_EXAMPLES} rows; found {formal_examples}"
        )


def select_evaluation_rows(
    all_rows: list[dict[str, Any]],
    limit: int | None,
    smoke_longest_prompt: bool,
) -> list[dict[str, Any]]:
    """Select a frozen evaluation subset without changing formal semantics."""
    if limit is not None and not 1 <= limit <= len(all_rows):
        raise RuntimeError(
            f"--limit must be in [1, {len(all_rows)}]; received {limit}"
        )
    if smoke_longest_prompt:
        if limit != 1:
            raise RuntimeError("--smoke-longest-prompt requires exactly --limit 1")
        return [
            max(
                all_rows,
                key=lambda row: (
                    row.get("prompt_tokens", -1),
                    str(row.get("example_id", "")),
                ),
            )
        ]
    return all_rows[:limit] if limit is not None else all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Omit only for the zero-shot base-model control.")
    parser.add_argument("--model", default=FROZEN_MODEL)
    parser.add_argument("--model-revision", default=FROZEN_MODEL_REVISION)
    parser.add_argument("--max-prompt-tokens", type=int, default=FROZEN_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=FROZEN_MAX_NEW_TOKENS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke testing only; formal aggregation rejects it.")
    parser.add_argument(
        "--smoke-longest-prompt",
        action="store_true",
        help="With --limit 1, evaluate the frozen test row with the largest prompt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_rows = read_jsonl(args.test_file)
    formal_examples = len(all_rows)
    validate_frozen_cli(args, formal_examples)
    observed_test_hash = sha256_file(args.test_file)
    if observed_test_hash != FROZEN_TEST_SHA256:
        raise RuntimeError(
            "frozen V3 test-file hash drift: "
            f"{observed_test_hash} != {FROZEN_TEST_SHA256}"
        )
    expected_all: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            raise RuntimeError("every frozen test row must have a string example_id")
        if example_id in expected_all:
            raise RuntimeError(f"duplicate example_id in frozen test: {example_id}")
        expected_all[example_id] = row
        if row.get("prompt_tokens", FROZEN_MAX_PROMPT_TOKENS + 1) > FROZEN_MAX_PROMPT_TOKENS:
            raise RuntimeError(f"recorded prompt exceeds frozen cap: {example_id}")
    rows = select_evaluation_rows(
        all_rows,
        args.limit,
        args.smoke_longest_prompt,
    )
    evaluated_ids = {row["example_id"] for row in rows}

    predictions_path = args.output.with_name(args.output.stem + ".predictions.jsonl")
    contract_path = args.output.with_name(args.output.stem + ".contract.json")
    recovery_audit_path = args.output.with_name(args.output.stem + ".resume_recovery.jsonl")
    fingerprint = checkpoint_fingerprint(args.adapter)
    contract = {
        "protocol": PROTOCOL,
        "training_and_evaluation_protocol": "qlora_v2_frozen",
        "test_file_sha256": observed_test_hash,
        "formal_test_examples": formal_examples,
        "evaluated_examples": len(rows),
        "model": args.model,
        "model_revision": args.model_revision,
        "checkpoint_fingerprint": fingerprint,
        "base_model_loading": "nf4_4bit",
        "max_prompt_tokens": args.max_prompt_tokens,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": 1,
        },
        "limited": args.limit is not None,
    }
    if args.smoke_longest_prompt:
        contract["smoke_selection"] = "longest_prompt"
        contract["smoke_forced_new_tokens"] = args.max_new_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError(
                "existing evaluation contract differs; use a new output path instead of mixing predictions"
            )
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
        for item in read_resumable_predictions(predictions_path, recovery_audit_path):
            example_id = item.get("example_id")
            if not isinstance(example_id, str):
                raise RuntimeError("resume prediction is missing a string example_id")
            if example_id not in evaluated_ids:
                scope = "formal test" if example_id not in expected_all else "limited evaluation subset"
                raise RuntimeError(f"foreign example_id in predictions ({scope}): {example_id}")
            if example_id in completed:
                raise RuntimeError(f"duplicate example_id in predictions: {example_id}")
            validate_completed_prediction(item, expected_all[example_id])
            completed[example_id] = item
    elif args.resume:
        predictions_path.touch()

    if len(completed) < len(rows):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("formal V3 generation evaluation requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the frozen V3 NF4 evaluator requires CUDA bfloat16 support")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.model_revision,
            trust_remote_code=True,
        )
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
                        f"observed={observed_tokens}, recorded={row['prompt_tokens']}, "
                        f"cap={args.max_prompt_tokens}"
                    )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    generation = {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if args.smoke_longest_prompt:
                        # Force the full KV-cache growth only in the one-row
                        # memory smoke. Formal generation remains V2-identical.
                        generation["min_new_tokens"] = args.max_new_tokens
                    output_ids = model.generate(**inputs, **generation)
                generated_token_count = int(
                    output_ids.shape[1] - observed_tokens
                )
                generated = tokenizer.decode(
                    output_ids[0][observed_tokens:],
                    skip_special_tokens=True,
                )
                scored = score_generated(row, generated)
                if args.smoke_longest_prompt:
                    scored["generated_token_count"] = generated_token_count
                handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
                handle.flush()
                completed[row["example_id"]] = scored
                if args.progress_every and index % args.progress_every == 0:
                    print(f"evaluated {len(completed)}/{len(rows)}", flush=True)
    else:
        print(
            f"resume audit found all {len(rows)} predictions complete; "
            "recomputing metrics without reloading the model",
            flush=True,
        )

    ordered = [completed[row["example_id"]] for row in rows if row["example_id"] in completed]
    if len(ordered) != len(rows):
        raise RuntimeError(f"evaluation incomplete: {len(ordered)}/{len(rows)}")
    groups = {
        "overall": ordered,
        "non_recovery": [
            item for item in ordered if item["recovery_mode"] == "none"
        ],
        "recovery": [item for item in ordered if item["recovery_mode"] != "none"],
        "agent_initiated": [
            item for item in ordered if item["recovery_mode"] == "agent_initiated"
        ],
        "user_assisted": [
            item for item in ordered if item["recovery_mode"] == "user_assisted"
        ],
    }
    recovery_audit = read_jsonl(recovery_audit_path) if recovery_audit_path.exists() else []
    result = {
        **contract,
        "claim_boundary": (
            "diagnostic offline held-out next-tool-call imitation on an already "
            "inspected V2 test set; not paper-final or executable agent success"
        ),
        "predictions_sha256": sha256_file(predictions_path),
        "resume_recovery_events": len(recovery_audit),
        "resume_recovery_audit": recovery_audit,
        "groups": {
            name: group_metrics(items, args.seed + offset, args.bootstrap_samples)
            for offset, (name, items) in enumerate(groups.items())
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
