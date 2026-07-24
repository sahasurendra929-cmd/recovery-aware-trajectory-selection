#!/usr/bin/env python3
"""Prepare the deterministic V5 Stage-0 split and paired smoke manifests.

This script intentionally uses only the Python standard library. It audits the
official tau2-bench split, derives validation data from official train only,
and never exports official-test task content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SEED = 20260722
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
DOMAINS = ("retail", "airline")
EXPECTED_COUNTS = {
    "retail": {"train": 74, "test": 40, "validation": 15, "inner_train": 59},
    "airline": {"train": 30, "test": 20, "validation": 6, "inner_train": 24},
}
SMOKE_QUOTAS = {"retail": 7, "airline": 3}
SAFE_INJECTIONS = {
    "find_user_id_by_email": {"email": "v5-stage0-missing@example.invalid"},
    "find_user_id_by_name_zip": {
        "first_name": "V5Missing",
        "last_name": "Identity",
        "zip": "00000",
    },
    "get_user_details": {"user_id": "v5_stage0_missing_user"},
    "get_reservation_details": {"reservation_id": "ZZZZZZ"},
}
DOMAIN_DEFAULT_INJECTION = {
    "retail": ("get_user_details", SAFE_INJECTIONS["get_user_details"]),
    "airline": (
        "get_reservation_details",
        SAFE_INJECTIONS["get_reservation_details"],
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_order(ids: list[str], domain: str, seed: int) -> list[str]:
    return sorted(
        ids,
        key=lambda task_id: (
            sha256_bytes(f"{seed}:{domain}:{task_id}".encode("utf-8")),
            task_id,
        ),
    )


def numeric_order(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda value: (int(value) if value.isdigit() else value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def first_safe_action(task: dict[str, Any]) -> dict[str, Any] | None:
    criteria = task.get("evaluation_criteria") or {}
    for action in criteria.get("actions") or []:
        if action.get("requestor", "assistant") != "assistant":
            continue
        if action.get("name") in SAFE_INJECTIONS:
            return action
    return None


def prepare(tau2_root: Path, output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    revision = resolve_git_commit(tau2_root)
    if revision != TAU2_COMMIT:
        raise RuntimeError(
            f"tau2-bench revision mismatch: expected {TAU2_COMMIT}, got {revision}"
        )

    domains_root = tau2_root / "data" / "tau2" / "domains"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest: dict[str, Any] = {
        "protocol": "v5_stage0_tau2_end_to_end",
        "seed": seed,
        "benchmark": {
            "repository": "https://github.com/sierra-research/tau2-bench.git",
            "release": "v1.0.1",
            "commit": revision,
        },
        "domains": {},
        "guarantees": {
            "validation_derived_from_official_train_only": True,
            "official_test_task_content_exported": False,
            "official_test_used_for_model_or_smoke_selection": False,
        },
    }
    smoke_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        domain_root = domains_root / domain
        split_path = domain_root / "split_tasks.json"
        tasks_path = domain_root / "tasks.json"
        split = load_json(split_path)
        tasks_list = load_json(tasks_path)
        tasks = {str(task["id"]): task for task in tasks_list}
        train_ids = [str(task_id) for task_id in split["train"]]
        test_ids = [str(task_id) for task_id in split["test"]]
        expected = EXPECTED_COUNTS[domain]

        if len(train_ids) != expected["train"] or len(test_ids) != expected["test"]:
            raise RuntimeError(f"Unexpected official split counts for {domain}")
        if set(train_ids) & set(test_ids):
            raise RuntimeError(f"Official train/test overlap for {domain}")
        if set(train_ids) | set(test_ids) != set(tasks):
            raise RuntimeError(f"Official split does not cover every {domain} task")

        ordered_train = deterministic_order(train_ids, domain=domain, seed=seed)
        validation_ids = numeric_order(ordered_train[: expected["validation"]])
        inner_train_ids = numeric_order(ordered_train[expected["validation"] :])

        if set(validation_ids) & set(inner_train_ids):
            raise RuntimeError(f"Derived train/validation overlap for {domain}")
        if set(validation_ids) & set(test_ids):
            raise RuntimeError(f"Derived validation/test overlap for {domain}")

        eligible: list[tuple[str, dict[str, Any]]] = []
        for task_id in validation_ids:
            action = first_safe_action(tasks[task_id])
            if action is not None:
                eligible.append((task_id, action))
        quota = SMOKE_QUOTAS[domain]
        if len(eligible) < quota:
            raise RuntimeError(
                f"{domain} has {len(eligible)} safe smoke tasks, needs {quota}"
            )

        selected = eligible[:quota]
        for task_id, action in selected:
            smoke_rows.append(
                {
                    "pair_id": f"{domain}:{task_id}",
                    "domain": domain,
                    "task_id": task_id,
                    "source_split": "derived_validation",
                    "clean_condition": {"inject_error": False},
                    "error_condition": {
                        "inject_error": True,
                        "tool_name": action["name"],
                        "tool_call_id": f"v5-stage0-injected-{domain}-{task_id}",
                        "arguments": SAFE_INJECTIONS[action["name"]],
                        "expected_tool_error": True,
                        "expected_state_mutation": False,
                    },
                }
            )

        split_manifest["domains"][domain] = {
            "source_files": {
                "split_tasks_json_sha256": sha256_file(split_path),
                "tasks_json_sha256": sha256_file(tasks_path),
            },
            "counts": {
                "official_train": len(train_ids),
                "derived_inner_train": len(inner_train_ids),
                "derived_validation": len(validation_ids),
                "sealed_official_test": len(test_ids),
            },
            "inner_train_ids": inner_train_ids,
            "validation_ids": validation_ids,
            "sealed_test_ids": numeric_order(test_ids),
            "smoke_ids": [task_id for task_id, _ in selected],
        }

    pair_ids = [row["pair_id"] for row in smoke_rows]
    if len(smoke_rows) != 10 or len(set(pair_ids)) != 10:
        raise RuntimeError("Stage-0 must contain exactly ten unique paired tasks")

    smoke_manifest = {
        "protocol": "v5_stage0_paired_clean_error_smoke",
        "seed": seed,
        "training": False,
        "scientific_claim_allowed": False,
        "paired_task_count": len(smoke_rows),
        "total_end_to_end_runs": 2 * len(smoke_rows),
        "rows": smoke_rows,
    }
    formal_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        tool_name, arguments = DOMAIN_DEFAULT_INJECTION[domain]
        for task_id in split_manifest["domains"][domain]["inner_train_ids"]:
            formal_rows.append(
                {
                    "pair_id": f"{domain}:{task_id}",
                    "domain": domain,
                    "task_id": task_id,
                    "source_split": "derived_inner_train",
                    "clean_condition": {"inject_error": False},
                    "error_condition": {
                        "inject_error": True,
                        "tool_name": tool_name,
                        "tool_call_id": f"v5-stage0-formal-{domain}-{task_id}",
                        "arguments": arguments,
                        "expected_tool_error": True,
                        "expected_state_mutation": False,
                    },
                }
            )
    formal_manifest = {
        "protocol": "v5_stage0_formal_data_construction",
        "seed": seed,
        "training": False,
        "scientific_claim_allowed": False,
        "paired_task_count": len(formal_rows),
        "rows": formal_rows,
    }

    split_path_out = output_dir / "split_manifest.json"
    smoke_path_out = output_dir / "smoke_manifest.json"
    formal_path_out = output_dir / "formal_manifest.json"
    split_path_out.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    smoke_path_out.write_text(
        json.dumps(smoke_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    formal_path_out.write_text(
        json.dumps(formal_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit = {
        "status": "PASS",
        "benchmark_commit": revision,
        "official_train_total": sum(
            domain["counts"]["official_train"]
            for domain in split_manifest["domains"].values()
        ),
        "inner_train_total": sum(
            domain["counts"]["derived_inner_train"]
            for domain in split_manifest["domains"].values()
        ),
        "validation_total": sum(
            domain["counts"]["derived_validation"]
            for domain in split_manifest["domains"].values()
        ),
        "sealed_test_total": sum(
            domain["counts"]["sealed_official_test"]
            for domain in split_manifest["domains"].values()
        ),
        "paired_smoke_tasks": len(smoke_rows),
        "total_smoke_runs": 2 * len(smoke_rows),
        "formal_inner_train_tasks": len(formal_rows),
        "split_manifest_sha256": sha256_file(split_path_out),
        "smoke_manifest_sha256": sha256_file(smoke_path_out),
        "formal_manifest_sha256": sha256_file(formal_path_out),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau2-root",
        type=Path,
        default=Path("data/raw/tau2-bench"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v5_stage0"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = prepare(
        tau2_root=args.tau2_root.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
