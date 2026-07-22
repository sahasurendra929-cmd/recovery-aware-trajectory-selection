#!/usr/bin/env python3
"""Build leakage-free next-tool-call SFT records for QLoRA baseline v1.

This generator deliberately serializes a simple textual protocol instead of
claiming a production function-calling format. Every target is an assistant
tool call; messages after that target are never included in its prompt.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 20260722
ERROR_RULES = {
    "not_found": re.compile(r"\b(not found|does not exist|unknown)\b", re.I),
    "invalid_argument": re.compile(r"\b(invalid|malformed|incorrect|must provide|missing)\b", re.I),
    "unauthorized": re.compile(r"\b(unauthorized|forbidden|not permitted)\b", re.I),
    "conflict": re.compile(r"\b(conflict|already|not available)\b", re.I),
    "generic_error": re.compile(r"^\s*error\s*[:\-]", re.I),
}


def canonical_args(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return re.sub(r"\s+", " ", value.strip())
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tool_calls(message):
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name"):
            calls.append({"name": fn["name"], "arguments": canonical_args(fn.get("arguments", {}))})
    return calls


def trace_id(record, source):
    return f"{source}:task{record['task_id']}:trial{record.get('trial', 0)}"


def signature(record):
    calls = []
    for message in record["traj"]:
        calls.extend((x["name"], x["arguments"]) for x in tool_calls(message))
    return (f"retail:{record['task_id']}", tuple(calls))


def load_records(data_dir):
    raw = []
    for path in sorted(data_dir.glob("*retail.json")):
        for record in json.loads(path.read_text()):
            if record.get("reward") == 1:
                raw.append({"record": record, "source": path.stem, "trace_id": trace_id(record, path.stem), "task_key": f"retail:{record['task_id']}"})
    seen, deduped = set(), []
    for item in sorted(raw, key=lambda x: (x["task_key"], x["source"], x["trace_id"])):
        key = signature(item["record"])
        if key not in seen:
            seen.add(key); deduped.append(item)
    return deduped


def task_split(records):
    tasks = sorted({x["task_key"] for x in records})
    random.Random(SEED).shuffle(tasks)
    n_train, n_val = round(.70 * len(tasks)), round(.10 * len(tasks))
    return set(tasks[:n_train]), set(tasks[n_train:n_train+n_val]), set(tasks[n_train+n_val:])


def compact_history(messages, limit):
    """Keep system policy plus the most recent history within a character cap."""
    system = [m for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]
    system_text = "\n".join(render_message(m) for m in system)
    tail = "\n".join(render_message(m) for m in other)
    remaining = max(0, limit - len(system_text) - 32)
    if len(tail) > remaining:
        tail = "[TRUNCATED_OLDER_HISTORY]\n" + tail[-remaining:]
    return (system_text + "\n" + tail).strip()


def render_message(message):
    role = (message.get("role") or "unknown").upper()
    if message.get("tool_calls"):
        payload = [{"name": x["name"], "arguments": x["arguments"]} for x in tool_calls(message)]
        return f"{role}_TOOL_CALL: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    name = f" name={message['name']}" if message.get("name") else ""
    content = message.get("content") or ""
    return f"{role}{name}: {content}"


def error_type(text):
    return next((name for name, rule in ERROR_RULES.items() if rule.search(text or "")), None)


def examples_for_trace(item, max_context_chars):
    traj = item["record"]["traj"]
    examples = []
    for idx, message in enumerate(traj):
        calls = tool_calls(message)
        if message.get("role") != "assistant" or not calls:
            continue
        target = calls[0]  # τ-bench policy expects at most one call per assistant turn.
        prior_tool_error = None
        for prev in reversed(traj[:idx]):
            if prev.get("role") == "tool":
                prior_tool_error = error_type(prev.get("content") or "")
                break
        prompt = compact_history(traj[:idx], max_context_chars) + "\nASSISTANT_TOOL_CALL:\n"
        completion = json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        examples.append({
            "example_id": f"{item['trace_id']}:action{idx}",
            "trace_id": item["trace_id"],
            "task_key": item["task_key"],
            "prompt": prompt,
            "completion": completion,
            "target_tool": target["name"],
            "is_error_resolution_target": prior_tool_error is not None,
            "prior_error_type": prior_tool_error,
        })
    return examples


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    args = parser.parse_args()
    records = load_records(args.data_dir)
    train_tasks, val_tasks, test_tasks = task_split(records)
    by_id = {x["trace_id"]: x for x in records}
    shared = {}
    for name, tasks in [("validation", val_tasks), ("test", test_tasks)]:
        rows = [e for item in records if item["task_key"] in tasks for e in examples_for_trace(item, args.max_context_chars)]
        shared[name] = rows
        write_jsonl(args.output_dir / "shared" / f"{name}.jsonl", rows)
    summary = {"seed": SEED, "deduplicated_success_trajectories": len(records), "task_counts": {"train": len(train_tasks), "validation": len(val_tasks), "test": len(test_tasks)}, "splits": {"validation_examples": len(shared["validation"]), "test_examples": len(shared["test"])} }
    for arm in ["random_success", "shortest_success", "recovery_balanced"]:
        manifest = json.loads((args.manifest_dir / f"{arm}_manifest.json").read_text())
        chosen = [by_id[x] for x in manifest["trace_ids"]]
        if any(x["task_key"] not in train_tasks for x in chosen):
            raise ValueError(f"{arm} manifest leaks a non-train task")
        rows = [e for item in chosen for e in examples_for_trace(item, args.max_context_chars)]
        write_jsonl(args.output_dir / arm / "train.jsonl", rows)
        write_jsonl(args.output_dir / arm / "manifest.jsonl", [{"trace_id": x["trace_id"], "task_key": x["task_key"]} for x in chosen])
        summary[arm] = {"trajectories": len(chosen), "examples": len(rows), "recovery_targets": sum(x["is_error_resolution_target"] for x in rows), "target_tool_counts": Counter(x["target_tool"] for x in rows)}
    (args.output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, default=dict))
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
