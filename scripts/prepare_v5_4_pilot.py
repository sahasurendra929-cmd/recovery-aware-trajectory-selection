#!/usr/bin/env python3
"""Materialize the frozen outcome-free V5.4 pilot registries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import v5_4_pilot_protocol as protocol
except ModuleNotFoundError:
    from scripts import v5_4_pilot_protocol as protocol


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build_protocol() -> dict[str, Any]:
    payload = {
        "protocol": protocol.PROTOCOL,
        "design_version": protocol.DESIGN_VERSION,
        "stage": "counterfactual_branch_data_feasibility_pilot",
        "source_split": "derived_inner_train",
        "official_test_used": False,
        "official_test_sealed": True,
        "training": False,
        "base_seed": protocol.BASE_SEED,
        "trial_seeds": list(protocol.TRIAL_SEEDS),
        "tasks": list(protocol.PILOT_TASK_IDS),
        "slot_budget": {
            "clean_per_task": protocol.CLEAN_SLOTS_PER_TASK,
            "recovery_per_task": protocol.RECOVERY_SLOTS_PER_TASK,
            "maximum_total": protocol.MAX_EXECUTED_SLOTS,
            "unused_slots_reallocated": False,
        },
        "stopping": {
            "first_eligible_clean_stops_remaining_clean_slots": True,
            "maximum_pairs_per_task": protocol.MAX_PAIRS_PER_TASK,
            "pair_cap_stops_remaining_recovery_slots": True,
        },
        "branch": {
            "eligible_tool_calls": "read_only_only",
            "location_rule": "sha256(protocol,task_identity,clean_seed)",
            "single_controlled_error": True,
            "clean_future_deleted_before_recovery": True,
            "recovery_replans_after_actual_error_result": True,
            "failed_call_and_error_result_masked_from_positive_labels": True,
        },
        "primary_error_taxonomy": sorted(protocol.PRIMARY_ERROR_TAXONOMY),
        "gate": {
            "minimum_tasks_with_pair": protocol.MIN_TASKS_WITH_PAIR,
            "minimum_capped_pairs": protocol.MIN_CAPPED_PAIRS,
            "must_beat_v5_3_observed_tasks": (
                protocol.V5_3_OBSERVED_TASKS_WITH_PAIR
            ),
            "must_beat_v5_3_observed_pairs": (
                protocol.V5_3_OBSERVED_CAPPED_PAIRS
            ),
            "must_improve_quality_adjusted_pair_yield": True,
            "both_domains_required": True,
        },
        "claim_boundary": {
            "data_feasibility_only": True,
            "training_authorized_by_pilot": False,
            "official_test_used": False,
            "positive_result_guaranteed": False,
        },
    }
    payload["canonical_sha256"] = protocol.canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/v5_4_counterfactual_pilot/protocol"),
    )
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty {args.output_root}")
    task_rows = [
        {
            "task_identity": identity,
            "domain": identity.split(":", 1)[0],
            "task_id": identity.split(":", 1)[1],
            "source_split": "derived_inner_train",
            "official_test_used": False,
        }
        for identity in protocol.PILOT_TASK_IDS
    ]
    write_json(args.output_root / "v5_4_pilot_protocol.json", build_protocol())
    write_json(args.output_root / "task_registry.json", {"rows": task_rows})
    write_jsonl(args.output_root / "slot_registry.jsonl", protocol.slot_registry())
    print(
        json.dumps(
            {
                "status": "PASS",
                "tasks": len(task_rows),
                "registered_slots": protocol.MAX_EXECUTED_SLOTS,
                "training_started": False,
                "official_test_used": False,
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
