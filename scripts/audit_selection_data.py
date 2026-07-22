#!/usr/bin/env python3
"""Audit the frozen v1 selection arms before model training.

The report is descriptive only: it checks split integrity and documents which
error-resolution signal each arm retained. It does not evaluate an agent.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from run_data_baseline import SEED, deduplicate, parse_record, stats

ARMS = ("random_success", "shortest_success", "recovery_balanced")


def load_records(data_dir: Path):
    raw = []
    for path in sorted(data_dir.glob("*retail.json")):
        for record in json.loads(path.read_text()):
            if record.get("reward") == 1:
                raw.append(parse_record(record, path.stem))
    return deduplicate(raw)


def group_split(records):
    tasks = sorted({record["task_key"] for record in records})
    random.Random(SEED).shuffle(tasks)
    train_count, val_count = round(.70 * len(tasks)), round(.10 * len(tasks))
    return set(tasks[:train_count]), set(tasks[train_count:train_count + val_count]), set(tasks[train_count + val_count:])


def row_for_arm(name, selected, train_tasks):
    base = stats(name, selected)
    events = [event for record in selected for event in record["events"]]
    base.update({
        "split_leakage": sum(record["task_key"] not in train_tasks for record in selected),
        "agent_initiated_events": sum(not event["user_assisted"] for event in events),
        "user_assisted_events": sum(event["user_assisted"] for event in events),
        "error_type_counts": dict(sorted(Counter(event["error_type"] for event in events).items())),
        "failed_tool_counts": dict(sorted(Counter(event["failed_tool"] for event in events).items())),
        "repair_tool_counts": dict(sorted(Counter(event["repair_tool"] for event in events).items())),
    })
    return base


def markdown(rows, split_counts):
    lines = [
        "# QLoRA v1 selection-data audit",
        "",
        "This is a descriptive audit of the fixed offline SFT inputs; it is not an end-to-end Agent result.",
        "",
        f"- seed: `{SEED}`",
        f"- task-group split: train/validation/test = `{split_counts['train']}/{split_counts['validation']}/{split_counts['test']}`",
        "",
        "| arm | est. tokens | traces | tasks | recovery events | agent-initiated | user-assisted | split leakage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {method} | {estimated_tokens} | {traces} | {unique_tasks} | {recovery_events} | {agent_initiated_events} | {user_assisted_events} | {split_leakage} |".format(**row))
    for row in rows:
        lines.extend(["", f"## {row['method']}", "", f"- error types: `{json.dumps(row['error_type_counts'], sort_keys=True)}`", f"- failed tools: `{json.dumps(row['failed_tool_counts'], sort_keys=True)}`", f"- repair tools: `{json.dumps(row['repair_tool_counts'], sort_keys=True)}`"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.data_dir)
    by_id = {record["trace_id"]: record for record in records}
    train_tasks, validation_tasks, test_tasks = group_split(records)
    rows = []
    for arm in ARMS:
        manifest = json.loads((args.manifest_dir / f"{arm}_manifest.json").read_text())
        selected = [by_id[trace_id] for trace_id in manifest["trace_ids"]]
        rows.append(row_for_arm(arm, selected, train_tasks))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_audit.json").write_text(json.dumps({"seed": SEED, "task_split": {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)}, "arms": rows}, indent=2))
    (args.output_dir / "selection_audit.md").write_text(markdown(rows, {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)}))
    fields = ["method", "estimated_tokens", "traces", "unique_tasks", "recovery_traces", "recovery_events", "agent_initiated_events", "user_assisted_events", "error_types", "tools", "tool_sequences", "split_leakage"]
    with (args.output_dir / "selection_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{field: row[field] for field in fields} for row in rows])
    if any(row["split_leakage"] for row in rows):
        raise SystemExit("split leakage detected")
    print((args.output_dir / "selection_audit.md").read_text())


if __name__ == "__main__":
    main()
