#!/usr/bin/env python3
"""Recompute the isolated V5.3 low-support diagnostic from raw evidence.

This summarizer accepts exactly the registered base_model, perfect_success,
and repair_50 evaluations on the existing 21-task derived-validation split.
It is deliberately incapable of producing a formal V5.3 or paper-level
confirmation claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    import run_v5_sft_causal_eval as evaluation
    import summarize_v5_3_12h_screen as screen_summary
    import v5_3_low_support_protocol as protocol
    from v5_dynamic_audit_contract import load_complete_dynamic_audit
except ModuleNotFoundError:
    from scripts import run_v5_sft_causal_eval as evaluation
    from scripts import summarize_v5_3_12h_screen as screen_summary
    from scripts import v5_3_low_support_protocol as protocol
    from scripts.v5_dynamic_audit_contract import load_complete_dynamic_audit


SUMMARY_PROTOCOL = f"{protocol.PROTOCOL}:raw_summary_v1"
EXACT_ARMS = ("base_model", "perfect_success", "repair_50")
EXACT_TRAINED_ARMS = ("perfect_success", "repair_50")
ROOT = Path(__file__).resolve().parents[1]


class SummaryError(RuntimeError):
    """The low-support raw evidence is absent, inconsistent, or out of scope."""


def _require_frozen_identity() -> None:
    protocol.validate_constants()
    if (
        protocol.REGISTRY_PROFILE
        != evaluation.V5_3_LOW_SUPPORT_PROFILE
        or protocol.DESIGN_VERSION
        != evaluation.V5_3_LOW_SUPPORT_DESIGN_VERSION
        or protocol.DATA_PROTOCOL
        != evaluation.V5_3_LOW_SUPPORT_DESIGN_PROTOCOL
        or protocol.PROTOCOL
        != evaluation.V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL
        or tuple(protocol.TRAINED_ARMS) != EXACT_TRAINED_ARMS
        or tuple(protocol.EVAL_ARMS) != EXACT_ARMS
        or protocol.USER_JUDGE_MODEL_ID
        != evaluation.V5_3_LOW_SUPPORT_USER_JUDGE["model_id"]
        or protocol.USER_JUDGE_REVISION
        != evaluation.V5_3_LOW_SUPPORT_USER_JUDGE["revision"]
    ):
        raise SummaryError("low-support protocol/evaluator identity drift")


def _require_exact_result_arm_set(results_root: Path) -> None:
    evaluation_root = results_root / "evaluation"
    observed = {
        path.name
        for path in evaluation_root.iterdir()
        if path.is_dir()
        and any(path.glob("run_contract.shard-*-of-003.json"))
    } if evaluation_root.is_dir() else set()
    if observed != set(EXACT_ARMS):
        raise SummaryError(
            "low-support evaluation arm set must be exact; "
            f"expected={list(EXACT_ARMS)}, observed={sorted(observed)}"
        )


def _descriptive_direction(
    arm_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repair = arm_summaries["repair_50"]["conditions"]
    perfect = arm_summaries["perfect_success"]["conditions"]
    error_delta = (
        repair["error"]["successes"] - perfect["error"]["successes"]
    )
    clean_delta = (
        repair["clean"]["successes"] - perfect["clean"]["successes"]
    )
    return {
        "contrast": "repair_50_minus_perfect_success",
        "error_success_count_delta": error_delta,
        "clean_success_count_delta": clean_delta,
        "error_success_rate_delta": error_delta / 21,
        "clean_success_rate_delta": clean_delta / 21,
        "interpretation": (
            "post-yield descriptive direction only; no formal gate, "
            "confirmatory inference, or paper-level confirmation"
        ),
    }


def _training_support(registry: dict[str, Any]) -> dict[str, Any]:
    design_provenance = registry["training_data_provenance"].get(
        "design_provenance"
    )
    if not isinstance(design_provenance, dict):
        raise SummaryError("registry lacks low-support training support")
    support = {
        "eligible_distinct_tasks": design_provenance.get(
            "eligible_distinct_tasks"
        ),
        "eligible_capped_pairs": design_provenance.get(
            "eligible_capped_pairs"
        ),
        "maximum_pairs_per_task": design_provenance.get(
            "maximum_pairs_per_task"
        ),
        "schedule_rows_per_arm": design_provenance.get(
            "schedule_rows_per_arm"
        ),
        "repeated_schedule_rows_are_independent_examples": (
            design_provenance.get(
                "repeated_schedule_rows_are_independent_examples"
            )
        ),
    }
    if (
        isinstance(support["eligible_distinct_tasks"], bool)
        or not isinstance(support["eligible_distinct_tasks"], int)
        or support["eligible_distinct_tasks"] < protocol.MIN_DISTINCT_TASKS
        or isinstance(support["eligible_capped_pairs"], bool)
        or not isinstance(support["eligible_capped_pairs"], int)
        or support["eligible_capped_pairs"] < protocol.MIN_CAPPED_PAIRS
        or support["maximum_pairs_per_task"]
        != protocol.MAX_PAIRS_PER_TASK
        or support["schedule_rows_per_arm"] != protocol.SCHEDULE_ROWS
        or support["repeated_schedule_rows_are_independent_examples"]
        is not False
    ):
        raise SummaryError("registry low-support counts/limits drift")
    support["minimum_distinct_tasks"] = protocol.MIN_DISTINCT_TASKS
    support["minimum_capped_pairs"] = protocol.MIN_CAPPED_PAIRS
    return support


def _summary_source_provenance(
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Bind canonical recomputation to the clean registered source commit."""

    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SummaryError(
                f"repository contains {label} tracked summarizer drift"
            )
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    expected = registry.get("source_commit")
    if observed != expected:
        raise SummaryError(
            "summarizer source commit drift: "
            f"HEAD={observed}, registry={expected}"
        )
    return {
        "summarizer_source_commit": observed,
        "summarizer_tracked_source_clean": True,
    }


def write_incomplete_receipt(output: Path, error: Exception) -> Path:
    """Record failure beside, never in, the immutable canonical summary."""

    incomplete = {
        "protocol": SUMMARY_PROTOCOL,
        "status": "INCOMPLETE_NO_CLAIM",
        "error_type": type(error).__name__,
        "error": str(error),
        "partial_metrics_reported": False,
        "post_yield_exploratory_diagnostic": True,
        "formal_v5_3_result": False,
        "paper_level_confirmation": False,
        "official_test_used": False,
        "official_test_sealed": True,
    }
    path = output.with_name(f"{output.stem}.incomplete.json")
    screen_summary.atomic_write(path, incomplete)
    return path


def summarize(
    *,
    split_manifest_path: Path,
    evaluation_manifest_path: Path,
    dynamic_audit_path: Path,
    checkpoint_registry_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    _require_frozen_identity()
    _require_exact_result_arm_set(results_root)

    split = evaluation.load_split_manifest(split_manifest_path)
    manifest = evaluation.load_manifest(
        evaluation_manifest_path,
        split_manifest=split,
        split_manifest_sha256=screen_summary.sha256_file(
            split_manifest_path
        ),
    )
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 21:
        raise SummaryError(
            "diagnostic evaluation manifest must contain exactly 21 tasks"
        )
    expected_rows = {
        f"{row['domain']}:{row['task_id']}": row for row in rows
    }
    if len(expected_rows) != 21:
        raise SummaryError("diagnostic validation task identities are duplicated")

    dynamic_identity = load_complete_dynamic_audit(
        dynamic_audit_path,
        manifest_path=evaluation_manifest_path,
        split_manifest_path=split_manifest_path,
        expected_source_split="derived_validation",
        expected_task_ids=set(expected_rows),
    )
    registry = evaluation.load_checkpoint_registry(
        checkpoint_registry_path,
        expected_profile=protocol.REGISTRY_PROFILE,
    )
    if (
        set(registry["entries"]) != set(EXACT_ARMS)
        or registry.get("diagnostic_evaluator")
        != evaluation.V5_3_LOW_SUPPORT_USER_JUDGE
        or registry.get("diagnostic_claim_boundary")
        != evaluation.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
    ):
        raise SummaryError("checkpoint registry diagnostic identity drift")
    if (
        registry["training_data_provenance"]["dynamic_audits"]["validation"]
        != dynamic_identity
    ):
        raise SummaryError("registry/dynamic-validation audit drift")
    summary_source_provenance = _summary_source_provenance(registry)
    training_support = _training_support(registry)

    registry_sha = screen_summary.sha256_file(checkpoint_registry_path)
    manifest_sha = screen_summary.sha256_file(evaluation_manifest_path)
    split_sha = screen_summary.sha256_file(split_manifest_path)
    required_contract_metadata = {
        "diagnostic_protocol": protocol.PROTOCOL,
        "diagnostic_claim_boundary": (
            evaluation.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
        ),
    }
    loaded = {
        arm: screen_summary.load_arm_raw(
            arm=arm,
            arm_dir=results_root / "evaluation" / arm,
            expected_rows=expected_rows,
            registry=registry,
            registry_sha256=registry_sha,
            evaluation_manifest_sha256=manifest_sha,
            split_manifest_sha256=split_sha,
            dynamic_identity=dynamic_identity,
            expected_registry_profile=protocol.REGISTRY_PROFILE,
            expected_user_judge_model_id=protocol.USER_JUDGE_MODEL_ID,
            expected_user_judge_revision=protocol.USER_JUDGE_REVISION,
            expected_decoding=evaluation.V5_3_LOW_SUPPORT_FROZEN_DECODING,
            expected_trial_seed=protocol.TRIAL_SEEDS[0],
            required_contract_metadata=required_contract_metadata,
        )
        for arm in EXACT_ARMS
    }
    schedule = screen_summary.canonical(
        loaded["base_model"]["tasks_by_shard"]
    )
    if any(
        screen_summary.canonical(loaded[arm]["tasks_by_shard"]) != schedule
        for arm in EXACT_ARMS
    ):
        raise SummaryError("diagnostic arm shard schedules differ")

    arm_summaries = {
        arm: screen_summary.arm_metrics(loaded[arm], expected_rows)
        for arm in EXACT_ARMS
    }
    comparisons = {
        "repair_50_minus_perfect_success": screen_summary.compare(
            loaded["repair_50"], loaded["perfect_success"]
        ),
        "repair_50_minus_base_model": screen_summary.compare(
            loaded["repair_50"], loaded["base_model"]
        ),
        "perfect_success_minus_base_model": screen_summary.compare(
            loaded["perfect_success"], loaded["base_model"]
        ),
    }
    provenance = {
        **summary_source_provenance,
        "split_manifest_sha256": split_sha,
        "evaluation_manifest_sha256": manifest_sha,
        "dynamic_audit_sha256": dynamic_identity["sha256"],
        "checkpoint_registry_sha256": registry_sha,
        "contract_sha256": {
            arm: loaded[arm]["contract_sha256"] for arm in EXACT_ARMS
        },
        "result_sha256": {
            arm: loaded[arm]["result_sha256"] for arm in EXACT_ARMS
        },
        "strict_judge_mapping_sha256": {
            arm: loaded[arm]["strict_judge_mapping_sha256"]
            for arm in EXACT_ARMS
        },
        "raw_recomputation": True,
        "hand_entered_counts_accepted": False,
    }
    provenance["canonical_input_binding_sha256"] = (
        protocol.canonical_sha256(provenance)
    )
    return {
        "protocol": SUMMARY_PROTOCOL,
        "status": "DIAGNOSTIC_COMPLETE",
        "design_version": protocol.DESIGN_VERSION,
        "design_protocol": protocol.DATA_PROTOCOL,
        "experiment_protocol": protocol.PROTOCOL,
        "post_yield_exploratory_diagnostic": True,
        "formal_v5_3_result": False,
        "paper_level_confirmation": False,
        "primary_metric": (
            "tau2_official_composite_end_to_end_task_success"
        ),
        "independent_unit": "task_id",
        "evaluated_arms": list(EXACT_ARMS),
        "trained_adapters": list(EXACT_TRAINED_ARMS),
        "validation_tasks": 21,
        "conditions": ["clean", "error"],
        "trials": 1,
        "trial_seed": protocol.TRIAL_SEEDS[0],
        "raw_case_count": 126,
        "training_support": training_support,
        "official_test": {"status": "SEALED", "used": False},
        "extensions": {"allowed": False, "evaluated": False},
        "arms": arm_summaries,
        "paired_comparisons": comparisons,
        "descriptive_direction": _descriptive_direction(arm_summaries),
        "claim_classification": {
            "formal_v5_3_claim_allowed": False,
            "paper_level_confirmation_allowed": False,
            "causal_or_confirmatory_claim_allowed": False,
            "allowed_language": (
                "post-yield, one-seed, low-support directional diagnostic "
                "on the frozen 21-task derived-validation split"
            ),
        },
        "claim_boundary": dict(
            evaluation.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
        ),
        "scientific_scope": dict(protocol.CLAIM_BOUNDARY),
        "provenance": provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    expected_results_root = protocol.artifact_root(ROOT, "results_root")
    if args.results_root.resolve() != expected_results_root:
        raise SummaryError(
            "low-support summary input is outside its isolated results root"
        )
    if output != expected_results_root / "diagnostic_summary.json":
        raise SummaryError(
            "low-support summary output is outside its isolated results root"
        )
    protocol.require_whole_run_source_lock(ROOT)
    try:
        summary = summarize(
            split_manifest_path=args.split_manifest.resolve(),
            evaluation_manifest_path=args.evaluation_manifest.resolve(),
            dynamic_audit_path=args.dynamic_audit.resolve(),
            checkpoint_registry_path=args.checkpoint_registry.resolve(),
            results_root=args.results_root.resolve(),
        )
    except Exception as error:
        incomplete_output = write_incomplete_receipt(output, error)
        incomplete = screen_summary.load_json(incomplete_output)
        print(json.dumps(incomplete, indent=2), file=sys.stderr)
        raise SystemExit(20) from error
    if output.exists():
        existing = screen_summary.load_json(output)
        if existing != summary:
            raise SummaryError(
                "existing immutable diagnostic summary differs from "
                "raw recomputation"
            )
    else:
        screen_summary.atomic_write(output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
