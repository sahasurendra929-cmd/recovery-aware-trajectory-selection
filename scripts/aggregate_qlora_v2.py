#!/usr/bin/env python3
"""Validate and aggregate the base control plus all three v2 arms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = ("base_model", "random_success", "shortest_success", "recovery_coverage")
TRAINED_ARMS = ARMS[1:]


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def interval(value):
    return "—" if not value or value[0] is None else f"[{100 * value[0]:.2f}, {100 * value[1]:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/qlora_v2"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/selection_v2"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/qlora_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis_v2"))
    args = parser.parse_args()

    build = json.loads((args.processed_dir / "build_summary.json").read_text(encoding="utf-8"))
    formal_examples = build["shared_splits"]["test_examples"]
    expected_revision = build["model"]["resolved_revision"]
    training_contract = build["training"]
    selected_tokens = {}
    build_arms = {item["arm"]: item for item in build["arms"]}
    rows, errors = [], []
    environment_signatures = set()
    for arm in ARMS:
        metrics_path = args.results_root / arm / "metrics.json"
        row = {"arm": arm, "status": "missing"}
        if not metrics_path.exists():
            errors.append(f"{arm}: missing {metrics_path}")
            rows.append(row)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        arm_errors = []
        if metrics.get("protocol") != "qlora_v2":
            arm_errors.append("wrong protocol")
        if metrics.get("limited") or metrics.get("evaluated_examples") != formal_examples:
            arm_errors.append(f"incomplete formal test: {metrics.get('evaluated_examples')}/{formal_examples}")
        if metrics.get("model_revision") != expected_revision:
            arm_errors.append("model revision drift")
        if metrics.get("test_file_sha256") != build.get("hashes", {}).get("test_jsonl"):
            arm_errors.append("test file hash drift")
        if metrics.get("max_prompt_tokens") != 1664 or metrics.get("generation") != {"do_sample": False, "max_new_tokens": 128, "batch_size": 1}:
            arm_errors.append("evaluation context/generation drift")
        if metrics.get("base_model_loading") != "nf4_4bit":
            arm_errors.append("base loading drift")
        if arm != "base_model":
            manifest_path = args.results_root / arm / "run_manifest.json"
            selection_path = args.selection_dir / f"{arm}_manifest.json"
            if not manifest_path.exists() or not selection_path.exists():
                arm_errors.append("missing training or selection manifest")
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selected_tokens[arm] = selection["selected_sft_tokens"]
                expected_train = {"protocol": "qlora_v2", "model_revision": expected_revision, "max_seq_len": training_contract["pad_to_max_sequence_tokens"], "max_steps": training_contract["optimizer_steps"], "batch_size": 1, "grad_accum": training_contract["gradient_accumulation"], "smoke_test": False}
                for key, value in expected_train.items():
                    if manifest.get(key) != value:
                        arm_errors.append(f"training {key} drift")
                if manifest.get("train_file_sha256") != build_arms[arm].get("train_schedule_sha256"):
                    arm_errors.append("training schedule hash drift")
                expected_validation_hash = build.get("hashes", {}).get("validation_jsonl")
                if expected_validation_hash and manifest.get("validation_file_sha256") != expected_validation_hash:
                    arm_errors.append("validation file hash drift")
                env = manifest.get("environment", {})
                environment_signatures.add((env.get("torch"), env.get("transformers"), env.get("cuda_runtime"), env.get("gpu")))
        overall = metrics.get("groups", {}).get("overall", {})
        recovery = metrics.get("groups", {}).get("recovery", {})
        agent = metrics.get("groups", {}).get("agent_initiated", {})
        row.update({
            "status": "complete" if not arm_errors else "incompatible",
            "json_valid": overall.get("micro", {}).get("json_valid"),
            "tool_accuracy": overall.get("micro", {}).get("tool_name_correct"),
            "full_call_exact": overall.get("micro", {}).get("full_call_exact"),
            "task_macro_full_call_exact": overall.get("task_macro", {}).get("full_call_exact"),
            "full_call_ci_low": (overall.get("task_cluster_bootstrap_95", {}).get("full_call_exact") or [None, None])[0],
            "full_call_ci_high": (overall.get("task_cluster_bootstrap_95", {}).get("full_call_exact") or [None, None])[1],
            "recovery_examples": recovery.get("examples"),
            "recovery_full_call_exact": recovery.get("micro", {}).get("full_call_exact"),
            "agent_examples": agent.get("examples"),
            "agent_full_call_exact": agent.get("micro", {}).get("full_call_exact"),
            "errors": "; ".join(arm_errors) if arm_errors else "OK",
        })
        errors.extend(f"{arm}: {message}" for message in arm_errors)
        rows.append(row)

    if selected_tokens and len(set(selected_tokens.values())) != 1:
        errors.append(f"selected token budgets differ: {selected_tokens}")
    if len(environment_signatures) > 1:
        errors.append(f"trained arms used different environments: {sorted(environment_signatures)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "status", "json_valid", "tool_accuracy", "full_call_exact", "task_macro_full_call_exact", "full_call_ci_low", "full_call_ci_high", "recovery_examples", "recovery_full_call_exact", "agent_examples", "agent_full_call_exact", "errors"]
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"protocol": "qlora_v2", "valid": not errors, "errors": errors, "arms": rows}
    (args.output_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# QLoRA v2 comparison",
        "",
        "Offline held-out next-tool-call imitation only; this is not executable Agent success.",
        "",
        "| arm | status | JSON valid | tool acc. | full call EM | task-macro EM | task-cluster 95% CI | recovery EM | agent-initiated EM |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        ci = [row.get("full_call_ci_low"), row.get("full_call_ci_high")]
        lines.append(f"| {row['arm']} | {row['status']} | {pct(row.get('json_valid'))} | {pct(row.get('tool_accuracy'))} | {pct(row.get('full_call_exact'))} | {pct(row.get('task_macro_full_call_exact'))} | {interval(ci)} | {pct(row.get('recovery_full_call_exact'))} | {pct(row.get('agent_full_call_exact'))} |")
    if errors:
        lines.extend(["", "## Contract errors", "", *[f"- {message}" for message in errors]])
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.output_dir / "comparison.md").read_text(encoding="utf-8"))
    if errors:
        raise SystemExit("v2 aggregation blocked by missing or incompatible outputs")


if __name__ == "__main__":
    main()
