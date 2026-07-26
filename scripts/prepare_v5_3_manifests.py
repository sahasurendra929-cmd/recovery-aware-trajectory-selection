#!/usr/bin/env python3
"""Build the frozen V5.3 manifests and exclude GT-incompatible train tasks.

The historical split remains immutable.  V5.3 derives its generation universe
solely from a structural property of the pinned tau2 tasks: a ground-truth
teacher must have at least one expected assistant tool action.  This filter is
computed before any rollout, reward, validation result, or official-test
content is observed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import prepare_v5_stage1_manifests as stage1
except ModuleNotFoundError:
    from scripts import prepare_v5_stage1_manifests as stage1


PROTOCOL = "v5_3_multifault_data_construction"
FILTER_PROTOCOL = "v5_3_gt_compatibility_filter_v1"
FROZEN_EXCLUDED_TASK_IDS = (
    "airline:0",
    "airline:10",
    "airline:28",
    "airline:34",
    "retail:24",
)
EXPECTED_SOURCE_TASK_COUNT = 83
EXPECTED_INCLUDED_TASK_COUNT = 78


def _ground_truth_incompatible_tasks(
    tau2_root: Path, split_manifest: dict[str, Any]
) -> list[str]:
    rows: list[str] = []
    domains_root = tau2_root / "data" / "tau2" / "domains"
    for domain in stage1.DOMAINS:
        allowed = {
            str(task_id)
            for task_id in split_manifest["domains"][domain]["inner_train_ids"]
        }
        tasks = stage1.load_json(domains_root / domain / "tasks.json")
        if not isinstance(tasks, list):
            raise RuntimeError(f"{domain}: tasks.json must contain a list")
        for task in tasks:
            if not isinstance(task, dict):
                raise RuntimeError(f"{domain}: malformed task")
            if (
                str(task.get("id")) in allowed
                and not stage1.task_reference_tools(task)
            ):
                rows.append(f"{domain}:{task['id']}")
    return sorted(rows)


def _filter_contract() -> dict[str, Any]:
    return {
        "protocol": FILTER_PROTOCOL,
        "policy": "exclude_before_sharding",
        "teacher_mode": "ground_truth",
        "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
        "included_task_count": EXPECTED_INCLUDED_TASK_COUNT,
        "excluded_task_ids": list(FROZEN_EXCLUDED_TASK_IDS),
        "exclusion_reason": "no_expected_tool_actions",
        "selection_uses_rollouts_rewards_validation_or_test": False,
        "official_test_used": False,
    }


def prepare(
    *,
    tau2_root: Path,
    split_manifest_path: Path,
    output_dir: Path,
    seed: int = stage1.SEED,
) -> dict[str, Any]:
    """Create Stage-1 manifests, then freeze the V5.3 train-only universe."""

    base_audit = stage1.prepare(
        tau2_root=tau2_root,
        split_manifest_path=split_manifest_path,
        output_dir=output_dir,
        seed=seed,
    )
    generation_path = output_dir / "generation_manifest.json"
    payload = stage1.load_json(generation_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_TASK_COUNT:
        raise RuntimeError("V5.3 source generation manifest is not the frozen 83 tasks")

    split_manifest = stage1.load_json(split_manifest_path)
    observed_incompatible = _ground_truth_incompatible_tasks(
        tau2_root, split_manifest
    )
    if observed_incompatible != list(FROZEN_EXCLUDED_TASK_IDS):
        raise RuntimeError(
            "pinned tau2 GT-incompatible task set drift: "
            f"observed={observed_incompatible}, "
            f"expected={list(FROZEN_EXCLUDED_TASK_IDS)}"
        )
    excluded = set(FROZEN_EXCLUDED_TASK_IDS)
    included_rows = [
        row
        for row in rows
        if f"{row.get('domain')}:{row.get('task_id')}" not in excluded
    ]
    included_ids = {
        f"{row.get('domain')}:{row.get('task_id')}" for row in included_rows
    }
    source_ids = {f"{row.get('domain')}:{row.get('task_id')}" for row in rows}
    if (
        len(source_ids) != EXPECTED_SOURCE_TASK_COUNT
        or len(included_ids) != EXPECTED_INCLUDED_TASK_COUNT
        or included_ids & excluded
        or included_ids | excluded != source_ids
    ):
        raise RuntimeError("V5.3 GT compatibility filter is not a disjoint cover")

    payload["protocol"] = PROTOCOL
    payload["paired_task_count"] = len(included_rows)
    payload["domain_counts"] = {
        domain: sum(row["domain"] == domain for row in included_rows)
        for domain in stage1.DOMAINS
    }
    payload["rows"] = included_rows
    payload["gt_compatibility_filter"] = _filter_contract()
    generation_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit_path = output_dir / "audit.json"
    audit = stage1.load_json(audit_path)
    audit["status"] = "PASS"
    audit["v5_3_generation_universe"] = {
        "protocol": PROTOCOL,
        "source_tasks": EXPECTED_SOURCE_TASK_COUNT,
        "included_tasks": EXPECTED_INCLUDED_TASK_COUNT,
        "excluded_tasks": len(excluded),
        "gt_compatibility_filter": _filter_contract(),
    }
    audit["generation_manifest_sha256"] = stage1.sha256_file(generation_path)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    hashes = {
        "generation_manifest.json": stage1.sha256_file(generation_path),
        "validation_manifest.json": stage1.sha256_file(
            output_dir / "validation_manifest.json"
        ),
        "audit.json": stage1.sha256_file(audit_path),
    }
    (output_dir / "hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=stage1.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != stage1.SEED:
        raise RuntimeError(f"V5.3 seed is frozen at {stage1.SEED}")
    result = prepare(
        tau2_root=args.tau2_root.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
