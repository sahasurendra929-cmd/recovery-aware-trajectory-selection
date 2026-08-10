#!/usr/bin/env python3
"""Materialize the frozen A08/A12 exploratory screen without formal-arm reuse.

This deliberately produces a separate protocol: it must not be consumed as a
formal V6 selector/training artifact.  Each selected candidate-pair keeps both
sibling branches and labels only the recorded recovery suffix assistant turns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

CLASS = "EXPLORATORY_NOT_FOR_FORMAL_GATE"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temp_name = handle.name
    os.replace(temp_name, path)
    sha = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha}  {path.name}\n")
    return sha


def compact(message):
    out = {"role": message["role"], "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    if out["role"] == "tool":
        out["tool_call_id"] = message.get("tool_call_id", message.get("id"))
        out["error"] = bool(message.get("error"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-index", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pool = json.loads(args.pool_index.read_text())
    selection = json.loads(args.selection.read_text())
    if selection.get("classification") != CLASS:
        raise SystemExit("selection classification drift")
    wanted = {(x["task_identity"], x["candidate_pair_id"], x["_session"]) for x in selection["selected_pairs"]}
    if len(wanted) != 24:
        raise SystemExit("expected exactly 24 selected candidate pairs")
    source = {}
    for entry in pool["files"]:
        if entry["session"] not in {"A08", "A12"}:
            continue
        for line in Path(entry["path"]).read_text().splitlines():
            row = json.loads(line)
            key = (row["task_identity"], row["candidate_pair_id"], entry["session"])
            if key in wanted:
                source[key] = row
    if set(source) != wanted:
        raise SystemExit("selection references missing or non-A08/A12 candidates")
    rows = []
    for task, pair_id, session in sorted(wanted):
        pair = source[(task, pair_id, session)]
        system = pair.get("training_system_message")
        if isinstance(system, dict):
            system = system.get("content")
        if not isinstance(system, str) or not system:
            raise SystemExit(f"{pair_id}: missing frozen training system message")
        branches = pair.get("branches")
        if not isinstance(branches, list) or len(branches) != 2:
            raise SystemExit(f"{pair_id}: expected two sibling branches")
        for branch in branches:
            prompt, suffix, mask = (branch.get("recovery_prompt"), branch.get("recovery_suffix"), branch.get("label_mask"))
            if not isinstance(prompt, list) or not isinstance(suffix, list) or not isinstance(mask, list):
                raise SystemExit(f"{pair_id}: recovery fields missing")
            messages = [{"role": "system", "content": system}] + [compact(m) for m in prompt + suffix]
            labels = [False] + list(mask)
            if len(messages) != len(labels) or not any(labels):
                raise SystemExit(f"{pair_id}: label mask drift")
            for index, message in enumerate(messages):
                if message["role"] == "tool" or (message["role"] == "assistant" and message.get("tool_calls") and index + 1 < len(messages) and messages[index + 1].get("error") is True):
                    if labels[index]:
                        raise SystemExit(f"{pair_id}: failed action or tool result was labeled")
            rows.append({
                "id": f"exploratory:{selection['arm']}:{pair_id}:{branch['branch_id']}",
                "messages": messages,
                "label_mask": labels,
                "metadata": {"classification": CLASS, "arm": selection["arm"], "task_identity": task, "candidate_pair_id": pair_id, "branch_id": branch["branch_id"], "session": session, "pair_sha256": pair.get("generated_candidate_pair_sha256"), "official_test_used": False},
            })
    if len(rows) != 48 or len({r["id"] for r in rows}) != 48:
        raise SystemExit("pair/branch cardinality drift")
    rows.sort(key=lambda row: row["id"])
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    file_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{file_sha}  {args.output.name}\n")
    write_json(args.output.with_suffix(".audit.json"), {"protocol": "v6_10_exploratory_training_rows_v1", "status": "PASS", "classification": CLASS, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(), "rows": len(rows), "candidate_pairs": 24, "sibling_branches_per_pair": 2, "sessions": ["A08", "A12"], "failed_action_positive_labels": 0, "row_content_sha256": digest(rows), "official_test_used": False})


if __name__ == "__main__":
    main()
