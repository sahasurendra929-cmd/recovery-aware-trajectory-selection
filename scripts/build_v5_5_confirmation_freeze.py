#!/usr/bin/env python3
"""Freeze the only admissible R0-vs-selected-arm official-test contract.

This command does not open the sealed split.  It creates the immutable human
approval artifact that a separately reviewed official-test executor must
require.  Keeping data access out of this step prevents accidental test use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import v5_5_full_protocol as full
except ModuleNotFoundError:
    from scripts import v5_5_full_protocol as full


PROTOCOL = "v5_5_confirmation_freeze_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def build(summary_path: Path, registry_path: Path, pair_audit_path: Path) -> dict:
    summary = load(summary_path)
    registry = load(registry_path)
    audit = load(pair_audit_path)
    if (
        summary.get("protocol") != "v5_5_task_cluster_summary_v1"
        or summary.get("status") != "PASS"
        or summary.get("official_test_used") is not False
        or registry.get("protocol") != "v5_5_checkpoint_registry_v1"
        or registry.get("official_test_used") is not False
        or registry.get("official_test_sealed") is not True
    ):
        raise RuntimeError("summary/checkpoint registry is not a frozen sealed PASS")
    selection = (summary.get("statistics") or {}).get("selection") or {}
    selected_trainer_arm = selection.get("selected_arm")
    selected = selection.get("selected_paper_arm")
    if selected not in full.RECOVERY_RATIOS or selected == "r0_perfect":
        raise RuntimeError("summary lacks a frozen non-zero selected arm")
    if selection.get("positive_screen") is not True:
        raise RuntimeError("validation screen is not positive")
    observed = audit.get("observed") or {}
    if (
        audit.get("protocol") != "v5_5_natural_counterfactual_pairs_v1"
        or audit.get("official_test_used") is not False
        or audit.get("official_test_sealed") is not True
        or audit.get("status") != "PASS_TRAINING_AUTHORIZED"
        or observed.get("paper_target_tasks_reached") is not True
        or observed.get("paper_target_pairs_reached") is not True
    ):
        raise RuntimeError("natural 30-task/60-pair confirmation gate is closed")
    trainer_arm = selected_trainer_arm
    if trainer_arm not in {"repair_25", "repair_50", "repair_75", "repair_100"}:
        raise RuntimeError("summary selected trainer arm is invalid")
    entries = registry.get("entries") or {}
    for arm in ("perfect_success", trainer_arm):
        if arm not in entries:
            raise RuntimeError(f"checkpoint registry lacks {arm}")
    return {
        "protocol": PROTOCOL,
        "status": "AWAITING_HUMAN_OFFICIAL_TEST_APPROVAL",
        "design_protocol": full.PROTOCOL,
        "comparison": ["r0_perfect", selected],
        "trainer_arms": ["perfect_success", trainer_arm],
        "training_seeds": list(full.TRAINING_SEEDS),
        "evaluation_seeds": list(full.EVALUATION_SEEDS),
        "conditions": ["clean", "controlled_error"],
        "controlled_error_strata": ["in_family", "heldout_tool_family"],
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_registry_sha256": sha256_file(registry_path),
        "natural_pair_audit_sha256": sha256_file(pair_audit_path),
        "clean_noninferiority_rule": "lower_95pct_task_bootstrap_ci_at_least_minus_0_05",
        "official_test_access_count_before_approval": 0,
        "official_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--natural-pair-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite confirmation freeze: {args.output}")
    payload = build(args.summary, args.checkpoint_registry, args.natural_pair_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
