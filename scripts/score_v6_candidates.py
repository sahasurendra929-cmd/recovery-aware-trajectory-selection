#!/usr/bin/env python3
"""Audit and score frozen V6 causal-recovery candidate pairs.

The selector atom is one ``candidate_pair`` containing exactly two sibling
error branches.  This program never filters a quality-passing pair based on
its score: in particular, low-kappa pairs remain in the scored pool so they
can serve as controls and so that selector comparisons use one frozen pool.

The input contract intentionally contains only train-pool evidence.  No
validation loss, end-to-end model outcome, or official-test result is accepted
as a scoring feature.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import v6_selection_protocol as protocol
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package
    from scripts import v6_selection_protocol as protocol


SCORING_PROTOCOL = "v6_causal_recovery_candidate_scoring_v1"
Q_KEYS = ("q_e1_a1", "q_e1_a2", "q_e2_a1", "q_e2_a2")
MATCHED_Q_KEYS = ("q_e1_a1", "q_e2_a2")
CROSS_Q_KEYS = ("q_e1_a2", "q_e2_a1")
DEFAULT_QUALITY_TRUE_FIELDS = (
    "real_error_executed",
    "matched_recovery_replay_success",
    "cross_replay_complete",
    "no_future_leakage",
    "independent_replay_audited",
)
DEFAULT_QUALITY_FALSE_FIELDS = ("official_test_used",)
DEFAULT_QUALITY_ZERO_FIELDS = ("failed_positive_labels",)
DEFAULT_COVERAGE_FIELDS = (
    "failed_tool",
    "corrective_action",
    "recovery_length_bin",
    "recovery_mode",
)
DEFAULT_LOW_KAPPA_THRESHOLD = 0.25
DEFAULT_WINSOR_LIMITS = (0.05, 0.95)


class CandidateScoringError(RuntimeError):
    """The frozen V6 candidate-pair contract was violated."""


def _constant(name: str, default: Any) -> Any:
    return getattr(protocol, name, default)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateScoringError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CandidateScoringError(
                    f"{path}:{line_number}: every row must be an object"
                )
            rows.append(row)
    if not rows:
        raise CandidateScoringError(f"{path}: candidate pool is empty")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(temporary, path)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateScoringError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    text = _nonempty_string(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CandidateScoringError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateScoringError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateScoringError(f"{label} must be a finite number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateScoringError(f"{label} must be a positive integer")
    return value


def validate_quality(quality: Any, label: str) -> None:
    if not isinstance(quality, dict):
        raise CandidateScoringError(f"{label} must be an object")
    true_fields = tuple(
        _constant(
            "PAIR_TRUE_QUALITY_FIELDS",
            _constant("QUALITY_TRUE_FIELDS", DEFAULT_QUALITY_TRUE_FIELDS),
        )
    )
    false_fields = tuple(
        _constant(
            "PAIR_FALSE_QUALITY_FIELDS",
            _constant("QUALITY_FALSE_FIELDS", DEFAULT_QUALITY_FALSE_FIELDS),
        )
    )
    zero_fields = tuple(
        _constant(
            "PAIR_ZERO_QUALITY_FIELDS",
            _constant("QUALITY_ZERO_FIELDS", DEFAULT_QUALITY_ZERO_FIELDS),
        )
    )
    for field in true_fields:
        if quality.get(field) is not True:
            raise CandidateScoringError(f"{label}.{field} must be true")
    for field in false_fields:
        if quality.get(field) is not False:
            raise CandidateScoringError(f"{label}.{field} must be false")
    for field in zero_fields:
        value = quality.get(field)
        if isinstance(value, bool) or value != 0:
            raise CandidateScoringError(f"{label}.{field} must equal 0")


def _validate_q_cell(cell: Any, label: str) -> dict[str, Any]:
    if not isinstance(cell, dict):
        raise CandidateScoringError(f"{label} must be an object")
    raw_success = cell.get("task_success", cell.get("official_task_success"))
    if isinstance(raw_success, bool):
        task_success = float(raw_success)
    else:
        task_success = _finite_number(raw_success, f"{label}.task_success")
    if not 0.0 <= task_success <= 1.0:
        raise CandidateScoringError(f"{label}.task_success must be in [0, 1]")
    if cell.get("forced_first_only") is not True:
        raise CandidateScoringError(f"{label}.forced_first_only must be true")
    if cell.get("gold_suffix_visible") is not False:
        raise CandidateScoringError(f"{label}.gold_suffix_visible must be false")
    policy = _sha256(
        cell.get("continuation_policy_sha256"),
        f"{label}.continuation_policy_sha256",
    )
    seeds = _sha256(
        cell.get("continuation_seed_set_sha256"),
        f"{label}.continuation_seed_set_sha256",
    )
    decoding = _sha256(cell.get("decoding_sha256"), f"{label}.decoding_sha256")
    budget = _positive_int(cell.get("rollout_budget"), f"{label}.rollout_budget")
    return {
        "task_success": task_success,
        "continuation_policy_sha256": policy,
        "continuation_seed_set_sha256": seeds,
        "decoding_sha256": decoding,
        "rollout_budget": budget,
    }


def forced_first_kappa(row: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    evidence = row.get("forced_first_cells", row.get("forced_first_q"))
    if not isinstance(evidence, dict) or set(evidence) != set(Q_KEYS):
        raise CandidateScoringError(
            f"{row.get('candidate_pair_id', '<unknown>')}.forced_first_q "
            f"must contain exactly {list(Q_KEYS)}"
        )
    checked = {
        key: _validate_q_cell(
            evidence[key], f"{row.get('candidate_pair_id', '<unknown>')}.forced_first_q.{key}"
        )
        for key in Q_KEYS
    }
    for invariant in (
        "continuation_policy_sha256",
        "continuation_seed_set_sha256",
        "decoding_sha256",
        "rollout_budget",
    ):
        values = {cell[invariant] for cell in checked.values()}
        if len(values) != 1:
            raise CandidateScoringError(
                f"{row.get('candidate_pair_id', '<unknown>')}: "
                f"forced-first {invariant} differs across the four cells"
            )
    q = {key: float(checked[key]["task_success"]) for key in Q_KEYS}
    kappa = 0.5 * (
        q["q_e1_a1"] + q["q_e2_a2"] - q["q_e1_a2"] - q["q_e2_a1"]
    )
    if not -1.0 <= kappa <= 1.0:
        raise CandidateScoringError(
            f"{row.get('candidate_pair_id', '<unknown>')}: kappa outside [-1, 1]"
        )
    return kappa, q


def _mean_logprob(cell: Any, label: str) -> float:
    if not isinstance(cell, dict):
        raise CandidateScoringError(f"{label} must be an object")
    total = _finite_number(cell.get("sum_logprob"), f"{label}.sum_logprob")
    count = _positive_int(cell.get("token_count"), f"{label}.token_count")
    if total > 1e-9:
        raise CandidateScoringError(f"{label}.sum_logprob cannot be positive")
    return total / count


def length_normalized_hardness(
    row: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    evidence = row.get("first_action_logprobs")
    if not isinstance(evidence, dict) or set(evidence) != set(Q_KEYS):
        raise CandidateScoringError(
            f"{row.get('candidate_pair_id', '<unknown>')}.first_action_logprobs "
            f"must contain exactly {list(Q_KEYS)}"
        )
    means = {
        key: _mean_logprob(
            evidence[key],
            f"{row.get('candidate_pair_id', '<unknown>')}.first_action_logprobs.{key}",
        )
        for key in Q_KEYS
    }
    # Higher is harder: the frozen base model prefers the sibling-wrong first
    # action over the matched first action by a larger per-token log-prob gap.
    error_1_gap = means["q_e1_a2"] - means["q_e1_a1"]
    error_2_gap = means["q_e2_a1"] - means["q_e2_a2"]
    return max(error_1_gap, error_2_gap), {
        **means,
        "error_1_wrong_minus_matched": error_1_gap,
        "error_2_wrong_minus_matched": error_2_gap,
    }


def _branch_coverage(
    branch: Mapping[str, Any], label: str, *, domain: str
) -> dict[str, str]:
    coverage = branch.get("coverage")
    if not isinstance(coverage, dict):
        raise CandidateScoringError(f"{label}.coverage must be an object")
    fields = tuple(_constant("COVERAGE_FIELDS", DEFAULT_COVERAGE_FIELDS))
    result = {}
    for field in fields:
        if field == "domain" and coverage.get(field) is None:
            value = domain
        elif field == "error_family" and coverage.get(field) is None:
            value = branch.get("error_family")
        else:
            value = coverage.get(field)
        result[field] = _nonempty_string(value, f"{label}.coverage.{field}")
    if result.get("domain", domain) != domain:
        raise CandidateScoringError(f"{label}.coverage.domain differs from pair domain")
    error_family = _nonempty_string(branch.get("error_family"), f"{label}.error_family")
    if "error_family" in coverage and coverage["error_family"] != error_family:
        raise CandidateScoringError(
            f"{label}: direct error_family and coverage.error_family differ"
        )
    result["error_family"] = error_family
    return result


def validate_candidate_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = _nonempty_string(row.get("candidate_pair_id"), "candidate_pair_id")
    task_identity = _nonempty_string(
        row.get("task_identity"), f"{pair_id}.task_identity"
    )
    domain = _nonempty_string(row.get("domain"), f"{pair_id}.domain")
    if not task_identity.startswith(f"{domain}:"):
        raise CandidateScoringError(
            f"{pair_id}: task_identity must be namespaced by domain"
        )
    _sha256(row.get("prefix_sha256"), f"{pair_id}.prefix_sha256")
    _sha256(
        row.get("environment_snapshot_sha256"),
        f"{pair_id}.environment_snapshot_sha256",
    )
    if "audit_status" in row:
        if row.get("audit_status") != "ACCEPTED":
            raise CandidateScoringError(f"{pair_id}: audit_status must be ACCEPTED")
        audited_hash = row.get("audited_candidate_pair_sha256")
        _sha256(audited_hash, f"{pair_id}.audited_candidate_pair_sha256")
        unhashed = dict(row)
        unhashed.pop("audited_candidate_pair_sha256", None)
        # A previously scored row may be revalidated by the manifest builder.
        # Remove only deterministic fields owned by this scorer before checking
        # the upstream auditor's immutable payload hash.
        for scorer_field in (
            "scoring_protocol",
            "scores",
            "coverage_categories",
            "branch_ids",
        ):
            unhashed.pop(scorer_field, None)
        if isinstance(unhashed.get("cost"), dict):
            upstream_cost = dict(unhashed["cost"])
            upstream_cost.pop("selector_atom", None)
            unhashed["cost"] = upstream_cost
        if canonical_sha256(unhashed) != audited_hash:
            raise CandidateScoringError(
                f"{pair_id}: audited_candidate_pair_sha256 does not verify"
            )
    validate_quality(row.get("quality"), f"{pair_id}.quality")
    branches = row.get("branches")
    if not isinstance(branches, list) or len(branches) != 2:
        raise CandidateScoringError(f"{pair_id}.branches must contain exactly two rows")
    branch_ids: set[str] = set()
    action_keys: set[str] = set()
    error_events: set[str] = set()
    full_first_actions: set[str] = set()
    c_sup = 0
    c_nonpad = 0
    coverages: list[dict[str, str]] = []
    for index, branch in enumerate(branches, start=1):
        label = f"{pair_id}.branches[{index - 1}]"
        if not isinstance(branch, dict):
            raise CandidateScoringError(f"{label} must be an object")
        branch_ids.add(_nonempty_string(branch.get("branch_id"), f"{label}.branch_id"))
        action_keys.add(
            _nonempty_string(
                branch.get("first_recovery_action_key"),
                f"{label}.first_recovery_action_key",
            )
        )
        error_events.add(
            _sha256(branch.get("error_event_sha256"), f"{label}.error_event_sha256")
        )
        first_action = branch.get("first_recovery_action")
        if not isinstance(first_action, dict):
            raise CandidateScoringError(
                f"{label}.first_recovery_action must contain the full canonical call"
            )
        full_first_actions.add(canonical(first_action))
        if row.get("c_sup") is None and not isinstance(row.get("cost"), dict):
            branch_sup = _positive_int(
                branch.get("supervised_target_tokens"),
                f"{label}.supervised_target_tokens",
            )
            branch_nonpad = _positive_int(
                branch.get("nonpadding_tokens"), f"{label}.nonpadding_tokens"
            )
            if branch_sup > branch_nonpad:
                raise CandidateScoringError(
                    f"{label}: supervised_target_tokens exceeds nonpadding_tokens"
                )
            c_sup += branch_sup
            c_nonpad += branch_nonpad
        coverages.append(_branch_coverage(branch, label, domain=domain))
    if len(branch_ids) != 2 or len(action_keys) != 2 or len(error_events) != 2:
        raise CandidateScoringError(
            f"{pair_id}: sibling branches must have distinct IDs, errors, and "
            "first recovery actions"
        )
    if len(full_first_actions) != 2:
        raise CandidateScoringError(
            f"{pair_id}: sibling branches must have two distinct complete "
            "first_recovery_action calls; different identifier keys alone are insufficient"
        )
    top_cost = row.get("cost") if isinstance(row.get("cost"), dict) else row
    if row.get("c_sup") is not None or isinstance(row.get("cost"), dict):
        c_sup = _positive_int(top_cost.get("c_sup"), f"{pair_id}.c_sup")
        c_nonpad = _positive_int(top_cost.get("c_nonpad"), f"{pair_id}.c_nonpad")
        if c_sup > c_nonpad:
            raise CandidateScoringError(f"{pair_id}: c_sup exceeds c_nonpad")
    kappa, q = forced_first_kappa(row)
    if row.get("kappa") is not None and not math.isclose(
        _finite_number(row.get("kappa"), f"{pair_id}.kappa"),
        kappa,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CandidateScoringError(
            f"{pair_id}: producer kappa differs from forced-first recomputation"
        )
    hardness, hardness_evidence = length_normalized_hardness(row)
    categories = {f"domain={domain}"}
    for coverage in coverages:
        categories.update(
            f"{field}={value}" for field, value in coverage.items()
        )
    action_families = sorted(
        {coverage["corrective_action"] for coverage in coverages}
    )
    return {
        "candidate_pair_id": pair_id,
        "task_identity": task_identity,
        "domain": domain,
        "branch_ids": sorted(branch_ids),
        "c_sup": c_sup,
        "c_nonpad": c_nonpad,
        "kappa": kappa,
        "q": q,
        "hardness_raw": hardness,
        "hardness_evidence": hardness_evidence,
        "coverage_categories": sorted(categories),
        "action_families": action_families,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CandidateScoringError("cannot take a quantile of an empty group")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def winsorized_midrank_percentiles(
    values: Sequence[float], lower: float, upper: float
) -> tuple[list[float], tuple[float, float]]:
    if not 0.0 <= lower < upper <= 1.0:
        raise CandidateScoringError("winsor limits must satisfy 0 <= low < high <= 1")
    low_value = _quantile(values, lower)
    high_value = _quantile(values, upper)
    clipped = [min(max(value, low_value), high_value) for value in values]
    percentiles: list[float] = []
    for value in clipped:
        below = sum(other < value for other in clipped)
        equal = sum(other == value for other in clipped)
        percentiles.append((below + 0.5 * equal) / len(clipped))
    return percentiles, (low_value, high_value)


def score_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    low_kappa_threshold: float | None = None,
    winsor_limits: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise CandidateScoringError("candidate pool is empty")
    threshold = (
        float(_constant("LOW_KAPPA_THRESHOLD", DEFAULT_LOW_KAPPA_THRESHOLD))
        if low_kappa_threshold is None
        else low_kappa_threshold
    )
    limits = (
        tuple(_constant("HARDNESS_WINSOR_LIMITS", DEFAULT_WINSOR_LIMITS))
        if winsor_limits is None
        else winsor_limits
    )
    if len(limits) != 2:
        raise CandidateScoringError("winsor limits must contain two probabilities")
    checked = [validate_candidate_pair(row) for row in rows]
    ids = [item["candidate_pair_id"] for item in checked]
    if len(ids) != len(set(ids)):
        raise CandidateScoringError("candidate_pair_id values must be globally unique")

    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    # A pair with two action families participates once in each calibration
    # stratum; its final percentile is the mean of those calibrated views.
    for index, item in enumerate(checked):
        for action_family in item["action_families"]:
            grouped_indices[(item["domain"], action_family)].append(index)
    calibrated: dict[int, list[float]] = defaultdict(list)
    group_audit: dict[str, Any] = {}
    for key in sorted(grouped_indices):
        indices = grouped_indices[key]
        values = [checked[index]["hardness_raw"] for index in indices]
        percentiles, bounds = winsorized_midrank_percentiles(
            values, float(limits[0]), float(limits[1])
        )
        for index, percentile in zip(indices, percentiles):
            calibrated[index].append(percentile)
        group_audit[f"{key[0]}::{key[1]}"] = {
            "pairs": len(indices),
            "winsor_low": bounds[0],
            "winsor_high": bounds[1],
        }

    result: list[dict[str, Any]] = []
    for index, (source, item) in enumerate(zip(rows, checked)):
        if not calibrated[index]:
            raise CandidateScoringError(
                f"{item['candidate_pair_id']}: no hardness calibration stratum"
            )
        percentile = sum(calibrated[index]) / len(calibrated[index])
        scored = deepcopy(dict(source))
        scored["scoring_protocol"] = SCORING_PROTOCOL
        scored["scores"] = {
            "forced_first_kappa": item["kappa"],
            "forced_first_q": item["q"],
            "low_kappa": item["kappa"] < threshold,
            "low_kappa_threshold": threshold,
            "hardness_length_normalized": item["hardness_raw"],
            "hardness_components": item["hardness_evidence"],
            "hardness_winsorized_percentile": percentile,
            "hardness_calibration_strata": [
                f"{item['domain']}::{family}"
                for family in item["action_families"]
            ],
        }
        scored["cost"] = {
            "c_sup": item["c_sup"],
            "c_nonpad": item["c_nonpad"],
            "selector_atom": "complete_candidate_pair_two_sibling_branches",
        }
        scored["coverage_categories"] = item["coverage_categories"]
        scored["branch_ids"] = item["branch_ids"]
        result.append(scored)

    result.sort(key=lambda row: (str(row["task_identity"]), str(row["candidate_pair_id"])))
    task_counts = Counter(str(row["task_identity"]) for row in result)
    low_count = sum(bool(row["scores"]["low_kappa"]) for row in result)
    audit = {
        "protocol": SCORING_PROTOCOL,
        "design_protocol": _constant(
            "PROTOCOL", "v6_causal_recovery_selection_v1"
        ),
        "status": "PASS",
        "candidate_pairs_in": len(rows),
        "candidate_pairs_out": len(result),
        "tasks": len(task_counts),
        "pairs_per_task": dict(sorted(task_counts.items())),
        "quality_rejections": 0,
        "low_kappa_pairs_retained": low_count,
        "high_kappa_pairs_retained": len(result) - low_count,
        "low_kappa_threshold": threshold,
        "hardness": {
            "definition": (
                "max sibling-wrong minus matched first-action mean log-prob"
            ),
            "length_normalized": True,
            "winsor_limits": list(limits),
            "calibration_group": "domain_x_corrective_action",
            "groups": group_audit,
        },
        "cost": {
            "c_sup": "sum of both branches' supervised_target_tokens",
            "c_nonpad": "sum of both branches' nonpadding_tokens",
        },
        "official_test_used": False,
        "outcomes_used_for_scoring": [
            "train_pool_forced_first_environment_task_success_only"
        ],
        "validation_or_official_test_outcomes_used": False,
    }
    return result, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--low-kappa-threshold", type=float)
    parser.add_argument("--winsor-low", type=float)
    parser.add_argument("--winsor-high", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.audit_output.exists():
        raise CandidateScoringError("output and audit paths must be absent")
    limits = None
    if args.winsor_low is not None or args.winsor_high is not None:
        if args.winsor_low is None or args.winsor_high is None:
            raise CandidateScoringError(
                "--winsor-low and --winsor-high must be supplied together"
            )
        limits = (args.winsor_low, args.winsor_high)
    rows = read_jsonl(args.input)
    scored, audit = score_candidates(
        rows,
        low_kappa_threshold=args.low_kappa_threshold,
        winsor_limits=limits,
    )
    write_jsonl(args.output, scored)
    audit["input_sha256"] = sha256_file(args.input)
    audit["output_sha256"] = sha256_file(args.output)
    audit["scored_pool_sha256"] = canonical_sha256(scored)
    write_json(args.audit_output, audit)


if __name__ == "__main__":
    main()
