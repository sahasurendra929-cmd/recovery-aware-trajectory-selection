#!/usr/bin/env python3
"""Audit and summarize the frozen V6.10 official evaluation by task cluster."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTOCOL = "v6_10_task_cluster_summary_v1"
EVALUATION_PROTOCOL = "v6_10_official_end_to_end_evaluation_v1"
ARMS = ("flawless_only", "random_stratified", "full_proposed")
CONDITIONS = ("clean", "controlled_error")
TRAINING_SEED = 20260722
EVALUATION_SEEDS = (20260722, 20260723, 20260724)
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 10_000
CLEAN_NONINFERIORITY_MARGIN = -0.05
PRIMARY_CONTROL = "random_stratified"
PRIMARY_METHOD = "full_proposed"


class SummaryError(RuntimeError):
    """The frozen V6.10 evaluation/statistical contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot read evaluation rows: {path}") from error
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SummaryError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise SummaryError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def read_evaluation_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load either one merged file or the frozen 18-shard directory."""

    if path.is_file():
        return read_jsonl(path), {str(path.resolve()): sha256_file(path)}
    if not path.is_dir():
        raise SummaryError(f"evaluation rows path is absent: {path}")
    row_paths = sorted(path.rglob("rows.jsonl"))
    expected_shards = len(ARMS) * len(EVALUATION_SEEDS) * 2
    if len(row_paths) != expected_shards:
        raise SummaryError(
            "V6.10 evaluation directory must contain exactly "
            f"{expected_shards} rows.jsonl shards, observed={len(row_paths)}"
        )
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    observed_jobs: set[tuple[str, int, int]] = set()
    for row_path in row_paths:
        receipt_path = row_path.with_name("evaluation_receipt.json")
        receipt = read_json(receipt_path)
        job = (
            str(receipt.get("arm")),
            receipt.get("evaluation_seed"),
            receipt.get("shard_index"),
        )
        if (
            receipt.get("protocol")
            != "v6_10_official_evaluation_shard_receipt_v1"
            or receipt.get("status") != "PASS"
            or job
            not in {
                (arm, seed, shard)
                for arm in ARMS
                for seed in EVALUATION_SEEDS
                for shard in range(2)
            }
            or job in observed_jobs
            or receipt.get("num_shards") != 2
            or receipt.get("row_count") != 60
            or receipt.get("rows_sha256") != sha256_file(row_path)
            or receipt.get("official_test_used") is not True
            or receipt.get("changes_after_unseal") is not False
        ):
            raise SummaryError(
                f"evaluation shard receipt drift: {receipt_path}"
            )
        observed_jobs.add(job)
        rows.extend(read_jsonl(row_path))
        hashes[str(row_path.resolve())] = sha256_file(row_path)
    if len(observed_jobs) != expected_shards:
        raise SummaryError("evaluation shard job grid is incomplete")
    return rows, hashes


def _numeric_reward(value: Any, *, key: tuple[Any, ...]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{key}: official_reward is not numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SummaryError(f"{key}: official_reward is outside [0, 1]")
    return result


def expected_task_ids(split_manifest: Mapping[str, Any]) -> set[str]:
    domains = split_manifest.get("domains")
    if (
        split_manifest.get("protocol") != "v5_stage0_tau2_end_to_end"
        or not isinstance(domains, Mapping)
    ):
        raise SummaryError("split manifest protocol drift")
    expected_counts = {"retail": 40, "airline": 20}
    result: set[str] = set()
    for domain, expected_count in expected_counts.items():
        payload = domains.get(domain)
        ids = (
            payload.get("sealed_test_ids")
            if isinstance(payload, Mapping)
            else None
        )
        if (
            not isinstance(ids, list)
            or len(ids) != expected_count
            or len({str(value) for value in ids}) != expected_count
        ):
            raise SummaryError(f"{domain}: sealed test population drift")
        result.update(f"{domain}:{value}" for value in ids)
    if len(result) != 60:
        raise SummaryError("official test population must contain 60 tasks")
    return result


def audit_grid(
    rows: list[dict[str, Any]],
    *,
    task_ids: set[str],
    unseal_receipt_sha256: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    expected = {
        (arm, TRAINING_SEED, evaluation_seed, task_id, condition)
        for arm in ARMS
        for evaluation_seed in EVALUATION_SEEDS
        for task_id in task_ids
        for condition in CONDITIONS
    }
    observed: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    for row in rows:
        task_id = f"{row.get('domain')}:{row.get('task_id')}"
        key = (
            row.get("arm"),
            row.get("training_seed"),
            row.get("evaluation_seed"),
            task_id,
            row.get("condition"),
        )
        if key in observed:
            duplicates.append(key)
        observed.add(key)
        if (
            row.get("protocol") != EVALUATION_PROTOCOL
            or row.get("official_test_used") is not True
            or row.get("unseal_receipt_sha256") != unseal_receipt_sha256
            or row.get("evaluator_sha256") != evaluator_sha256
        ):
            raise SummaryError(f"{key}: evaluation provenance drift")
        terminal_status = row.get("terminal_status")
        if terminal_status != "PASS":
            raise SummaryError(f"{key}: nonterminal evaluation row")
        reward = _numeric_reward(row.get("official_reward"), key=key)
        if (
            not isinstance(row.get("task_success"), bool)
            or row["task_success"] != (reward == 1.0)
        ):
            raise SummaryError(f"{key}: task_success/reward disagreement")
        if row.get("condition") == "clean":
            if (
                row.get("injection_count") != 0
                or row.get("injection_observed_as_error") is not None
            ):
                raise SummaryError(f"{key}: clean row contains an injection")
        elif (
            row.get("condition") == "controlled_error"
            and (
                row.get("injection_count") != 1
                or row.get("injection_observed_as_error") is not True
                or row.get("injection_repeated_by_harness") is not False
            )
        ):
            raise SummaryError(f"{key}: controlled-error evidence failed")
    missing = expected - observed
    extra = observed - expected
    if duplicates or missing or extra or len(rows) != len(expected):
        raise SummaryError(
            "V6.10 evaluation grid is incomplete: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, "
            f"extra={len(extra)}, rows={len(rows)}/{len(expected)}"
        )
    return {
        "status": "PASS",
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "task_count": len(task_ids),
        "ineligible_controlled_error_tasks": [],
        "ineligible_controlled_error_task_count": 0,
        "independent_unit": "task_id",
        "evaluation_seeds_are_repeated_measurements": True,
    }


def task_means(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    condition: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("arm") != arm or row.get("condition") != condition:
            continue
        if row.get("terminal_status") != "PASS":
            continue
        grouped[f"{row['domain']}:{row['task_id']}"].append(
            float(bool(row["task_success"]))
        )
    result: dict[str, float] = {}
    for task_id, values in grouped.items():
        if len(values) != len(EVALUATION_SEEDS):
            raise SummaryError(
                f"{arm}/{condition}/{task_id}: evaluation-seed coverage drift"
            )
        result[task_id] = sum(values) / len(values)
    return result


def paired_task_deltas(
    rows: list[dict[str, Any]],
    *,
    condition: str,
) -> tuple[list[str], list[float]]:
    method = task_means(rows, arm=PRIMARY_METHOD, condition=condition)
    control = task_means(rows, arm=PRIMARY_CONTROL, condition=condition)
    tasks = sorted(set(method) & set(control))
    if not tasks or set(method) != set(control):
        raise SummaryError(f"{condition}: paired task coverage drift")
    return tasks, [method[task] - control[task] for task in tasks]


def mean(values: list[float]) -> float:
    if not values:
        raise SummaryError("cannot average an empty task population")
    return sum(values) / len(values)


def cluster_bootstrap_ci(
    values: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    if replicates != BOOTSTRAP_REPLICATES:
        raise SummaryError("V6.10 requires exactly 10,000 bootstrap replicates")
    if not values or any(not math.isfinite(value) for value in values):
        raise SummaryError("bootstrap values are empty or non-finite")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(replicates)
    )
    lower = estimates[int(0.025 * replicates)]
    upper = estimates[min(replicates - 1, int(0.975 * replicates))]
    return [lower, upper]


def summarize(
    rows: list[dict[str, Any]],
    *,
    grid_audit: Mapping[str, Any],
) -> dict[str, Any]:
    ineligible = list(
        grid_audit.get("ineligible_controlled_error_tasks") or []
    )
    clean_tasks, clean_values = paired_task_deltas(rows, condition="clean")
    clean_ci = cluster_bootstrap_ci(clean_values)
    clean = {
        "task_count": len(clean_tasks),
        "paired_task_mean_delta": mean(clean_values),
        "task_cluster_bootstrap_ci95": clean_ci,
        "noninferiority_margin": CLEAN_NONINFERIORITY_MARGIN,
        "noninferior": clean_ci[0] >= CLEAN_NONINFERIORITY_MARGIN,
    }
    primary_tasks, primary_values = paired_task_deltas(
        rows, condition="controlled_error"
    )
    primary_ci = cluster_bootstrap_ci(primary_values)
    primary = {
        "status": "PASS",
        "task_count": len(primary_tasks),
        "ineligible_task_count": 0,
        "paired_task_mean_delta": mean(primary_values),
        "task_cluster_bootstrap_ci95": primary_ci,
    }
    positive_claim = (
        primary["status"] == "PASS"
        and primary["paired_task_mean_delta"] > 0.0
        and primary["task_cluster_bootstrap_ci95"][0] > 0.0
        and clean["noninferior"]
    )
    return {
        "primary_contrast": (
            f"{PRIMARY_METHOD}_minus_{PRIMARY_CONTROL}"
        ),
        "primary_estimand": (
            "paired_task_mean_controlled_error_success_delta"
        ),
        "primary": primary,
        "clean_noninferiority": clean,
        "decision": (
            "POSITIVE_DIRECTIONAL_SCREEN"
            if positive_claim
            else "NO_POSITIVE_DIRECTIONAL_SCREEN_CLAIM"
        ),
        "positive_claim": positive_claim,
    }


def build_summary(
    *,
    rows_path: Path,
    split_manifest_path: Path,
    unseal_receipt_path: Path,
    output: Path,
    expected_evaluator_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        raise SummaryError(f"refusing to overwrite {output}")
    rows, row_hashes = read_evaluation_rows(rows_path)
    split = read_json(split_manifest_path)
    unseal = read_json(unseal_receipt_path)
    if (
        unseal.get("protocol") != "v6_10_official_test_unseal_v1"
        or unseal.get("status") != "UNSEALED_ONCE"
        or unseal.get("official_test_access_count") != 1
        or unseal.get("evaluator_sha256") != expected_evaluator_sha256
    ):
        raise SummaryError("official-test unseal receipt drift")
    unseal_sha = sha256_file(unseal_receipt_path)
    grid = audit_grid(
        rows,
        task_ids=expected_task_ids(split),
        unseal_receipt_sha256=unseal_sha,
        evaluator_sha256=expected_evaluator_sha256,
    )
    statistics = summarize(rows, grid_audit=grid)
    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "rows_path": str(rows_path.resolve()),
        "row_file_sha256": row_hashes,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "unseal_receipt_sha256": unseal_sha,
        "evaluator_sha256": expected_evaluator_sha256,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "training_seed": TRAINING_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "grid_audit": grid,
        "statistics": statistics,
        "official_test_used": True,
        "changes_after_unseal": False,
    }
    payload["summary_sha256"] = hashlib.sha256(
        canonical(payload).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--unseal-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    args = parser.parse_args()
    payload = build_summary(
        rows_path=args.rows.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        unseal_receipt_path=args.unseal_receipt.resolve(),
        output=args.output.resolve(),
        expected_evaluator_sha256=args.expected_evaluator_sha256,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["statistics"]["decision"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
