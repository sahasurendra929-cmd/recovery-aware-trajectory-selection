#!/usr/bin/env python3
"""Aggregate frozen v1 GPU outputs into a comparison table.

Expected input layout:
  results/qlora_v1/<arm>/metrics.json
  results/qlora_v1/<arm>/run_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = ("random_success", "shortest_success", "recovery_balanced")
EXPECTED = {"seed": 20260722, "model": "Qwen/Qwen2.5-0.5B-Instruct", "max_seq_len": 512, "batch_size": 1, "grad_accum": 16, "learning_rate": 1e-4, "epochs": 1}
METRICS = ("examples", "tool_name_accuracy", "full_tool_call_exact_match", "recovery_examples", "recovery_tool_name_accuracy", "recovery_full_tool_call_exact_match")


def compatibility_errors(manifest):
    return [f"{key}: expected {value!r}, got {manifest.get(key)!r}" for key, value in EXPECTED.items() if manifest.get(key) != value]


def render(rows):
    lines = [
        "# QLoRA baseline v1 comparison",
        "",
        "This report evaluates held-out next-tool-call prediction only; it does not report end-to-end Agent success.",
        "",
        "| arm | status | tool-name acc. | full call EM | recovery tool acc. | recovery full EM | compatibility |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        def pct(key):
            value = row.get(key)
            return "—" if value is None else f"{100 * value:.2f}%"
        lines.append(f"| {row['arm']} | {row['status']} | {pct('tool_name_accuracy')} | {pct('full_tool_call_exact_match')} | {pct('recovery_tool_name_accuracy')} | {pct('recovery_full_tool_call_exact_match')} | {row['compatibility']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); rows = []
    for arm in ARMS:
        arm_dir = args.results_root / arm
        metrics_path, manifest_path = arm_dir / "metrics.json", arm_dir / "run_manifest.json"
        row = {"arm": arm, "status": "missing", **{metric: None for metric in METRICS}, "compatibility": "missing outputs"}
        if metrics_path.exists() and manifest_path.exists():
            metrics, manifest = json.loads(metrics_path.read_text()), json.loads(manifest_path.read_text())
            errors = compatibility_errors(manifest)
            row.update({metric: metrics.get(metric) for metric in METRICS})
            row.update({"status": "complete" if not errors else "incompatible", "compatibility": "OK" if not errors else "; ".join(errors)})
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "status", *METRICS, "compatibility"]
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(json.dumps({"expected_contract": EXPECTED, "arms": rows}, indent=2))
    (args.output_dir / "comparison.md").write_text(render(rows))
    print((args.output_dir / "comparison.md").read_text())
    if any(row["status"] != "complete" for row in rows):
        raise SystemExit("waiting for complete, compatible outputs from all three arms")


if __name__ == "__main__":
    main()
