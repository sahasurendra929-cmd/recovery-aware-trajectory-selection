#!/usr/bin/env python3
"""Audit and summarize the complete V5.5 validation grid by task cluster."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import v5_5_full_protocol as full
    from build_v5_5_checkpoint_registry import (
        PROTOCOL as REGISTRY_PROTOCOL,
        TRAINER_ARMS,
    )
    from run_v5_5_end_to_end_eval import PROTOCOL as EVAL_PROTOCOL
except ModuleNotFoundError:
    from scripts import v5_5_full_protocol as full
    from scripts.build_v5_5_checkpoint_registry import (
        PROTOCOL as REGISTRY_PROTOCOL,
        TRAINER_ARMS,
    )
    from scripts.run_v5_5_end_to_end_eval import PROTOCOL as EVAL_PROTOCOL


PROTOCOL = "v5_5_task_cluster_summary_v1"
CONTROL_ARM = "perfect_success"
ARM_DOSE = {
    "perfect_success": 0.0,
    "repair_25": 0.25,
    "repair_50": 0.50,
    "repair_75": 0.75,
    "repair_100": 1.0,
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def parse_csv(value: str, *, allowed: set[str], label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or not set(values) <= allowed:
        raise RuntimeError(f"invalid {label}: {values}")
    return values


def discover_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = sorted(root.rglob("rows.jsonl"))
    if not paths:
        raise RuntimeError(f"no V5.5 rows.jsonl files under {root}")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        relative = str(path.relative_to(root))
        hashes[relative] = sha256_file(path)
        for row in read_jsonl(path):
            if (
                row.get("protocol") != EVAL_PROTOCOL
                or row.get("official_test_used") is not False
            ):
                raise RuntimeError(f"{path}: non-V5.5 or test-opened row")
            rows.append(row)
    return rows, hashes


def audit_grid(
    rows: list[dict[str, Any]],
    *,
    task_ids: set[tuple[str, str]],
    arms: list[str],
    training_seeds: list[int],
    evaluation_seeds: list[int],
) -> dict[str, Any]:
    expected = {
        (arm, train_seed, eval_seed, domain, task_id, condition)
        for arm in arms
        for train_seed in training_seeds
        for eval_seed in evaluation_seeds
        for domain, task_id in task_ids
        for condition in ("clean", "error")
    }
    observed: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    for row in rows:
        key = (
            row.get("arm"),
            row.get("training_seed"),
            row.get("evaluation_seed"),
            str(row.get("domain")),
            str(row.get("task_id")),
            row.get("condition"),
        )
        if key in observed:
            duplicates.append(key)
        observed.add(key)
        if not isinstance(row.get("task_success"), bool):
            raise RuntimeError(f"non-boolean task_success at {key}")
        reward = row.get("official_reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise RuntimeError(f"invalid official reward at {key}")
        if row["condition"] == "error" and (
            row.get("injected_fault_count") != 1
            or row.get("injected_fault_observed_as_error") is not True
            or row.get("fault_scope") not in {"in_family", "heldout_tool_family"}
        ):
            raise RuntimeError(f"incomplete controlled-error evidence at {key}")
    missing = expected - observed
    extra = observed - expected
    if duplicates or missing or extra:
        raise RuntimeError(
            "V5.5 evaluation grid is incomplete: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    return {
        "status": "PASS",
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "duplicate_rows": 0,
        "missing_rows": 0,
        "extra_rows": 0,
        "independent_unit": "task_id",
    }


def task_means(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    condition: str,
    fault_scope: str | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm or row["condition"] != condition:
            continue
        if fault_scope is not None and row.get("fault_scope") != fault_scope:
            continue
        grouped[f"{row['domain']}:{row['task_id']}"].append(
            float(row["task_success"])
        )
    return {
        task: sum(values) / len(values)
        for task, values in grouped.items()
    }


def paired_values(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[list[str], list[float]]:
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("paired comparison has no common task IDs")
    return common, [left[task] - right[task] for task in common]


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average empty values")
    return sum(values) / len(values)


def cluster_bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    replicates: int,
    alpha: float = full.ALPHA,
) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(replicates)
    )
    lower = estimates[int((alpha / 2) * replicates)]
    upper = estimates[min(replicates - 1, int((1 - alpha / 2) * replicates))]
    return [lower, upper]


def sign_flip_pvalue(
    values: list[float],
    *,
    seed: int,
    replicates: int,
) -> float:
    observed = abs(mean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        statistic = abs(mean([value * rng.choice((-1, 1)) for value in values]))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (replicates + 1)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda arm: (raw[arm], arm))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, arm in enumerate(ordered):
        candidate = min(1.0, raw[arm] * (count - index))
        running = max(running, candidate)
        adjusted[arm] = running
    return adjusted


def summarize(
    rows: list[dict[str, Any]],
    *,
    arms: list[str],
    replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control_clean = task_means(rows, arm=CONTROL_ARM, condition="clean")
    control_error = task_means(
        rows,
        arm=CONTROL_ARM,
        condition="error",
        fault_scope="in_family",
    )
    raw_p: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for arm in arms:
        clean = task_means(rows, arm=arm, condition="clean")
        clean_rows = [
            row for row in rows
            if row["arm"] == arm and row["condition"] == "clean"
        ]
        natural_error_tasks: dict[str, bool] = {}
        for row in clean_rows:
            identity = f"{row['domain']}:{row['task_id']}"
            natural_error_tasks[identity] = (
                natural_error_tasks.get(identity, False)
                or int(row.get("tool_errors") or 0) > 0
            )
        error = task_means(
            rows,
            arm=arm,
            condition="error",
            fault_scope="in_family",
        )
        out_error = task_means(
            rows,
            arm=arm,
            condition="error",
            fault_scope="heldout_tool_family",
        )
        _, clean_delta = paired_values(clean, control_clean)
        _, error_delta = paired_values(error, control_error)
        result = {
            "paper_arm": TRAINER_ARMS[arm],
            "dose": ARM_DOSE[arm],
            "clean_tasks": len(clean),
            "clean_success": mean(list(clean.values())),
            "natural_agent_error_run_rate_diagnostic": (
                sum(int(row.get("tool_errors") or 0) > 0 for row in clean_rows)
                / len(clean_rows)
            ),
            "natural_agent_error_task_rate_diagnostic": (
                sum(natural_error_tasks.values()) / len(natural_error_tasks)
            ),
            "controlled_error_in_family_tasks": len(error),
            "controlled_error_in_family_success": mean(list(error.values())),
            "controlled_error_heldout_tool_family_tasks": len(out_error),
            "controlled_error_heldout_tool_family_success": (
                mean(list(out_error.values())) if out_error else None
            ),
            "clean_delta_vs_r0": mean(clean_delta),
            "clean_delta_ci95": cluster_bootstrap_ci(
                clean_delta,
                seed=full.TRAINING_SEEDS[0] + int(ARM_DOSE[arm] * 100),
                replicates=replicates,
            ),
            "error_delta_vs_r0": mean(error_delta),
            "error_delta_ci95": cluster_bootstrap_ci(
                error_delta,
                seed=full.EVALUATION_SEEDS[0] + int(ARM_DOSE[arm] * 100),
                replicates=replicates,
            ),
        }
        result["clean_noninferior_point_estimate"] = (
            result["clean_delta_vs_r0"] >= -full.CLEAN_NONINFERIORITY_MARGIN
        )
        result["clean_noninferior_ci95"] = (
            result["clean_delta_ci95"][0]
            >= -full.CLEAN_NONINFERIORITY_MARGIN
        )
        if arm != CONTROL_ARM:
            raw_p[arm] = sign_flip_pvalue(
                error_delta,
                seed=full.EVALUATION_SEEDS[1] + int(ARM_DOSE[arm] * 100),
                replicates=replicates,
            )
        comparisons[arm] = result
        table.append({"arm": arm, **result})
    adjusted = holm_adjust(raw_p)
    for arm in raw_p:
        comparisons[arm]["paired_randomization_p"] = raw_p[arm]
        comparisons[arm]["holm_adjusted_p"] = adjusted[arm]
        for row in table:
            if row["arm"] == arm:
                row["paired_randomization_p"] = raw_p[arm]
                row["holm_adjusted_p"] = adjusted[arm]

    candidates = [
        arm
        for arm in arms
        if arm != CONTROL_ARM
        and comparisons[arm]["clean_noninferior_ci95"]
    ]
    selected = (
        max(
            candidates,
            key=lambda arm: (
                full.validation_utility(
                    clean_success=comparisons[arm]["clean_success"],
                    error_success=comparisons[arm][
                        "controlled_error_in_family_success"
                    ],
                    baseline_clean_success=comparisons[CONTROL_ARM][
                        "clean_success"
                    ],
                ),
                -ARM_DOSE[arm],
            ),
        )
        if candidates
        else None
    )
    selection = {
        "selected_arm": selected,
        "selected_paper_arm": TRAINER_ARMS[selected] if selected else None,
        "rule": (
            "maximize in-family error success after the frozen 5-point clean "
            "non-inferiority 95% task-bootstrap-CI filter; ties choose the "
            "lower recovery dose"
        ),
        "positive_screen": (
            selected is not None
            and comparisons[selected]["error_delta_vs_r0"] > 0
        ),
    }
    return {
        "comparisons": comparisons,
        "holm_adjusted_p": adjusted,
        "selection": selection,
    }, table


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm",
        "paper_arm",
        "dose",
        "clean_tasks",
        "clean_success",
        "controlled_error_in_family_tasks",
        "controlled_error_in_family_success",
        "controlled_error_heldout_tool_family_tasks",
        "controlled_error_heldout_tool_family_success",
        "clean_delta_vs_r0",
        "clean_delta_ci95",
        "error_delta_vs_r0",
        "error_delta_ci95",
        "clean_noninferior_point_estimate",
        "paired_randomization_p",
        "holm_adjusted_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        canonical(row.get(field))
                        if isinstance(row.get(field), (dict, list))
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    claim_level: str,
) -> None:
    lines = [
        "# V5.5 Validation Summary",
        "",
        f"Claim level: `{claim_level}`",
        "",
        "| Arm | Recovery dose | Clean success | In-family error success | Δ error vs R0 | 95% task-bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {paper_arm} | {dose:.0%} | {clean_success:.3f} | "
            "{controlled_error_in_family_success:.3f} | "
            "{error_delta_vs_r0:+.3f} | [{lo:+.3f}, {hi:+.3f}] |".format(
                lo=row["error_delta_ci95"][0],
                hi=row["error_delta_ci95"][1],
                **row,
            )
        )
    lines.extend(
        [
            "",
            f"Frozen validation selection: `{selection['selected_paper_arm']}`.",
            f"Positive screen: `{selection['positive_screen']}`.",
            "",
            "The independent unit is task ID. Training/evaluation seeds are "
            "repeated measurements, not additional independent tasks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default=",".join(TRAINER_ARMS),
    )
    parser.add_argument(
        "--training-seeds",
        default=",".join(str(seed) for seed in full.TRAINING_SEEDS),
    )
    parser.add_argument(
        "--evaluation-seeds",
        default=",".join(str(seed) for seed in full.EVALUATION_SEEDS),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=full.BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"output directory must be absent: {args.output_dir}")
    if args.bootstrap_replicates < 1_000:
        raise RuntimeError("formal summary requires at least 1,000 resamples")
    arms = parse_csv(args.arms, allowed=set(TRAINER_ARMS), label="arms")
    if CONTROL_ARM not in arms:
        raise RuntimeError("V5.5 summary requires the R0 control arm")
    training_seeds = [
        int(value)
        for value in parse_csv(
            args.training_seeds,
            allowed={str(seed) for seed in full.TRAINING_SEEDS},
            label="training seeds",
        )
    ]
    evaluation_seeds = [
        int(value)
        for value in parse_csv(
            args.evaluation_seeds,
            allowed={str(seed) for seed in full.EVALUATION_SEEDS},
            label="evaluation seeds",
        )
    ]
    registry = read_json(args.checkpoint_registry)
    if registry.get("protocol") != REGISTRY_PROTOCOL:
        raise RuntimeError("checkpoint registry protocol drift")
    if (
        registry.get("registered_arms") != arms
        or registry.get("registered_training_seeds") != training_seeds
    ):
        raise RuntimeError("summary grid differs from checkpoint registry")
    manifest = read_json(args.validation_manifest)
    task_ids = {
        (str(row["domain"]), str(row["task_id"]))
        for row in manifest.get("rows", [])
    }
    if len(task_ids) != full.DERIVED_VALIDATION_TASKS:
        raise RuntimeError("validation manifest is not the frozen 21-task split")
    rows, result_hashes = discover_rows(args.results_root)
    grid = audit_grid(
        rows,
        task_ids=task_ids,
        arms=arms,
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    statistical, table = summarize(
        rows,
        arms=arms,
        replicates=args.bootstrap_replicates,
    )
    claim_levels = {str(row["claim_level"]) for row in rows}
    if len(claim_levels) != 1:
        raise RuntimeError("evaluation rows mix scientific claim levels")
    claim_level = next(iter(claim_levels))
    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "design_protocol": full.PROTOCOL,
        "grid_audit": grid,
        "arms": arms,
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "independent_unit": "task_id",
        "bootstrap_replicates": args.bootstrap_replicates,
        "statistics": statistical,
        "claim_level": claim_level,
        "result_file_sha256": result_hashes,
        "checkpoint_registry_sha256": sha256_file(
            args.checkpoint_registry
        ),
        "validation_manifest_sha256": sha256_file(
            args.validation_manifest
        ),
        "official_test_used": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "table.csv", table)
    write_markdown(
        args.output_dir / "report.md",
        table,
        selection=statistical["selection"],
        claim_level=claim_level,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": grid["observed_rows"],
                "selected_arm": statistical["selection"]["selected_paper_arm"],
                "positive_screen": statistical["selection"]["positive_screen"],
                "claim_level": claim_level,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
