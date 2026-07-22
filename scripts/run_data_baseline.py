#!/usr/bin/env python3
"""Create equal-budget trajectory-selection manifests for pilot v1.

Standard-library only. This script intentionally measures a data-selection
pilot; it does not train an LLM or report end-to-end agent success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
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


def calls(message):
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name"):
            out.append((fn["name"], canonical_args(fn.get("arguments", {}))))
    return out


def classify_error(text):
    for name, rule in ERROR_RULES.items():
        if rule.search(text or ""):
            return name
    return None


def parse_record(record, source):
    trajectory = record["traj"]
    tool_at = {}
    all_calls = []
    for index, message in enumerate(trajectory):
        msg_calls = calls(message)
        if msg_calls:
            tool_at[index] = msg_calls[0]
            all_calls.extend(msg_calls)

    events = []
    for index, message in enumerate(trajectory):
        if message.get("role") != "tool":
            continue
        error = classify_error(message.get("content", ""))
        if not error:
            continue
        failed = next((tool_at[j] for j in range(index - 1, -1, -1) if j in tool_at), None)
        if failed is None:
            continue
        repair, user_assisted = None, False
        for j in range(index + 1, len(trajectory)):
            user_assisted = user_assisted or trajectory[j].get("role") == "user"
            if j in tool_at:
                repair = tool_at[j]
                break
        if repair and repair != failed:
            events.append({"error_type": error, "failed_tool": failed[0], "repair_tool": repair[0], "user_assisted": user_assisted})

    trace_id = f"{source}:task{record['task_id']}:trial{record.get('trial', 0)}"
    return {
        "trace_id": trace_id,
        "task_id": int(record["task_id"]),
        # Keep the domain prefix: it is significant when datasets from more
        # than one environment are later combined, and preserves the original
        # pilot's deterministic task ordering.
        "task_key": f"retail:{record['task_id']}",
        "source": source,
        "calls": all_calls,
        "estimated_tokens": max(1, math.ceil(len(json.dumps(trajectory, ensure_ascii=False, sort_keys=True)) / 4)),
        "events": events,
    }


def deduplicate(records):
    seen, out = set(), []
    for record in sorted(records, key=lambda x: (x["task_key"], x["source"], x["trace_id"])):
        signature = (record["task_key"], tuple(record["calls"]))
        if signature not in seen:
            seen.add(signature)
            out.append(record)
    return out


def choose_to_budget(records, budget):
    selected, used = [], 0
    for record in records:
        if used + record["estimated_tokens"] <= budget:
            selected.append(record)
            used += record["estimated_tokens"]
    return selected


def balanced_choose(records, budget):
    rng = random.Random(SEED)
    positive = [x for x in records if x["events"]]
    negative = [x for x in records if not x["events"]]

    # Stratify the fill order by trajectory length. Without this, a recovery
    # quota can accidentally select only long traces and confound recovery
    # coverage with token length.
    lengths = sorted(x["estimated_tokens"] for x in records)
    cuts = [lengths[int((len(lengths) - 1) * q / 5)] for q in range(1, 5)]
    def bucket(record):
        return sum(record["estimated_tokens"] > cut for cut in cuts)
    pos_by, neg_by = defaultdict(list), defaultdict(list)
    for record in positive:
        pos_by[bucket(record)].append(record)
    for record in negative:
        neg_by[bucket(record)].append(record)
    for group in list(pos_by.values()) + list(neg_by.values()):
        rng.shuffle(group)

    pos_order = [record for bucket_id in range(5) for record in pos_by[bucket_id]]
    rng.shuffle(pos_order)
    selected = choose_to_budget(pos_order, int(.30 * budget))
    selected_ids = {x["trace_id"] for x in selected}
    remainder = [x for x in records if x["trace_id"] not in selected_ids]
    remainder_by = defaultdict(list)
    for record in remainder:
        remainder_by[bucket(record)].append(record)
    for group in remainder_by.values():
        rng.shuffle(group)
    ordered = []
    while any(remainder_by.values()):
        for bucket_id in range(5):
            if remainder_by[bucket_id]:
                ordered.append(remainder_by[bucket_id].pop())
    return selected + choose_to_budget(ordered, budget - sum(x["estimated_tokens"] for x in selected))


def stats(name, records):
    events = [event for record in records for event in record["events"]]
    return {
        "method": name,
        "estimated_tokens": sum(x["estimated_tokens"] for x in records),
        "traces": len(records),
        "unique_tasks": len({x["task_key"] for x in records}),
        "mean_trace_tokens": round(sum(x["estimated_tokens"] for x in records) / len(records), 1),
        "recovery_traces": sum(bool(x["events"]) for x in records),
        "recovery_events": len(events),
        "agent_initiated_events": sum(not x["user_assisted"] for x in events),
        "user_assisted_events": sum(x["user_assisted"] for x in events),
        "error_types": len({x["error_type"] for x in events}),
        "tools": len({call[0] for record in records for call in record["calls"]}),
        "tool_sequences": len({tuple(call[0] for call in record["calls"]) for record in records}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = []
    for path in sorted(args.data_dir.glob("*retail.json")):
        for record in json.loads(path.read_text()):
            if record.get("reward") == 1:
                raw.append(parse_record(record, path.stem))
    records = deduplicate(raw)
    tasks = sorted({x["task_key"] for x in records}); random.Random(SEED).shuffle(tasks)
    train_tasks = set(tasks[:round(.70 * len(tasks))])
    train = [x for x in records if x["task_key"] in train_tasks]
    budget = int(.30 * sum(x["estimated_tokens"] for x in train))
    rng = random.Random(SEED); shuffled = train[:]; rng.shuffle(shuffled)
    arms = {
        "random_success": choose_to_budget(shuffled, budget),
        "shortest_success": choose_to_budget(sorted(train, key=lambda x: (x["estimated_tokens"], x["trace_id"])), budget),
        "recovery_balanced": balanced_choose(train, budget),
    }

    rows = []
    for name, selected in arms.items():
        rows.append(stats(name, selected))
        (args.output_dir / f"{name}_manifest.json").write_text(json.dumps({
            "seed": SEED, "estimated_token_budget": budget,
            "trace_ids": [x["trace_id"] for x in selected],
        }, indent=2))
    with (args.output_dir / "selection_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
