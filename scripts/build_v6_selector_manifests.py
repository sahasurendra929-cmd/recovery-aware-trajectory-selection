#!/usr/bin/env python3
"""Build fail-closed, equal-budget V6 matched-task selector manifests.

Every recovery selector chooses exactly one complete ``candidate_pair`` per
task.  A candidate pair contains two sibling recovery branches and is never
split.  All recovery arms use the same task IDs and the exact same supervised
target-token budget.  Selection is solved as an exact multiple-choice
knapsack; if the registered budget is infeasible, the program stops rather
than silently relaxing the budget or dropping a task.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import score_v6_candidates as scoring
    import v6_selection_protocol as protocol
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import score_v6_candidates as scoring
    from scripts import v6_selection_protocol as protocol


MANIFEST_PROTOCOL = "v6_matched_task_selector_manifests_v1"
DEFAULT_RANDOM_SEEDS = (20260806, 20260807, 20260808)
DEFAULT_SELECTORS = (
    "random_stratified",
    "shortest",
    "coverage_only",
    "hardness_only",
    "causal_only",
    "causal_hardness",
    "full_proposed",
)
DEFAULT_MIN_PAIRS_PER_TASK = 3
DEFAULT_MAX_BUDGET_SPREAD = 0.005
DEFAULT_MAX_DP_STATES = 300_000
DEFAULT_FULL_WEIGHTS = {
    "causal": 0.50,
    "hardness": 0.25,
    "coverage": 0.25,
}


class SelectorManifestError(RuntimeError):
    """The V6 matched-task or equal-budget contract was violated."""


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectorManifestError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise SelectorManifestError(f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelectorManifestError(f"{label} must be a positive integer")
    return value


def verify_scored_pool(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise SelectorManifestError("scored candidate pool is empty")
    try:
        rescored, _ = scoring.score_candidates(rows)
    except scoring.CandidateScoringError as exc:
        raise SelectorManifestError(f"scored-pool source audit failed: {exc}") from exc
    recomputed = {
        str(row["candidate_pair_id"]): row
        for row in rescored
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pair_id = str(row.get("candidate_pair_id", ""))
        if not pair_id or pair_id in seen:
            raise SelectorManifestError(
                "candidate_pair_id must be present and globally unique"
            )
        seen.add(pair_id)
        if row.get("scoring_protocol") != scoring.SCORING_PROTOCOL:
            raise SelectorManifestError(f"{pair_id}: scoring_protocol drift")
        expected = recomputed[pair_id]
        for field in ("scores", "cost", "coverage_categories", "branch_ids"):
            if canonical(row.get(field)) != canonical(expected.get(field)):
                raise SelectorManifestError(
                    f"{pair_id}: scored field {field} does not match recomputation"
                )
        result.append(dict(row))
    result.sort(key=lambda row: (str(row["task_identity"]), str(row["candidate_pair_id"])))
    return result


def group_matched_tasks(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row["task_identity"])
        groups[task].append(row)
    minimum = int(
        _constant(
            "MIN_PAIRS_PER_TASK",
            _constant(
                "MIN_CANDIDATE_PAIRS_PER_TASK", DEFAULT_MIN_PAIRS_PER_TASK
            ),
        )
    )
    if minimum < 3:
        raise SelectorManifestError(
            "matched-task design requires at least three candidate pairs per task"
        )
    for task, candidates in groups.items():
        if len(candidates) < minimum:
            raise SelectorManifestError(
                f"{task}: only {len(candidates)} candidate pairs; need >= {minimum}"
            )
        domains = {str(row["domain"]) for row in candidates}
        choice_sets = {str(row.get("choice_set_id", "")) for row in candidates}
        prefixes = {str(row["prefix_sha256"]) for row in candidates}
        snapshots = {
            str(row["environment_snapshot_sha256"]) for row in candidates
        }
        if len(domains) != 1:
            raise SelectorManifestError(f"{task}: domain drift within task")
        if "" in choice_sets or len(choice_sets) != 1:
            raise SelectorManifestError(
                f"{task}: primary matched-task candidates must share one choice_set_id"
            )
        if len(prefixes) != 1 or len(snapshots) != 1:
            raise SelectorManifestError(
                f"{task}: candidates must share a frozen prefix and environment snapshot"
            )
        for row in candidates:
            if len(row.get("branches", [])) != 2:
                raise SelectorManifestError(
                    f"{row['candidate_pair_id']}: selector atom was split or malformed"
                )
    return {task: sorted(values, key=lambda row: str(row["candidate_pair_id"]))
            for task, values in sorted(groups.items())}


def coverage_rarity_scores(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    frequencies = Counter(
        category
        for row in rows
        for category in row["coverage_categories"]
        if not str(category).startswith("domain=")
    )
    result: dict[str, float] = {}
    for row in rows:
        categories = [
            category
            for category in row["coverage_categories"]
            if not str(category).startswith("domain=")
        ]
        if not categories:
            raise SelectorManifestError(
                f"{row['candidate_pair_id']}: no non-domain coverage categories"
            )
        result[str(row["candidate_pair_id"])] = sum(
            1.0 / math.sqrt(frequencies[category]) for category in categories
        ) / len(categories)
    return result


def _hash_utility(seed: int, task: str, pair_id: str) -> float:
    digest = hashlib.sha256(
        f"{MANIFEST_PROTOCOL}|{seed}|{task}|{pair_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def selector_utilities(
    rows: Sequence[Mapping[str, Any]],
    selector: str,
    *,
    random_seed: int | None,
    coverage_scores: Mapping[str, float],
) -> dict[str, float]:
    nonpad_values = [int(row["cost"]["c_nonpad"]) for row in rows]
    median_nonpad = float(statistics.median(nonpad_values))
    weights = dict(_constant("FULL_OBJECTIVE_WEIGHTS", DEFAULT_FULL_WEIGHTS))
    if set(weights) != set(DEFAULT_FULL_WEIGHTS):
        raise SelectorManifestError(
            f"full selector weights must contain {sorted(DEFAULT_FULL_WEIGHTS)}"
        )
    result: dict[str, float] = {}
    for row in rows:
        pair_id = str(row["candidate_pair_id"])
        scores = row["scores"]
        kappa = _finite(scores["forced_first_kappa"], f"{pair_id}.kappa")
        causal_01 = (kappa + 1.0) / 2.0
        hardness = _finite(
            scores["hardness_winsorized_percentile"],
            f"{pair_id}.hardness_percentile",
        )
        coverage = float(coverage_scores[pair_id])
        c_nonpad = float(row["cost"]["c_nonpad"])
        if selector == "random_stratified":
            if random_seed is None:
                raise SelectorManifestError("random selector requires a seed")
            utility = _hash_utility(
                random_seed, str(row["task_identity"]), pair_id
            )
        elif selector == "shortest":
            utility = -c_nonpad
        elif selector == "coverage_only":
            raise SelectorManifestError(
                "coverage_only must use the dynamic log1p marginal selector"
            )
        elif selector == "hardness_only":
            utility = hardness
        elif selector == "causal_only":
            utility = kappa
        elif selector == "causal_hardness":
            utility = 0.5 * causal_01 + 0.5 * hardness
        elif selector == "full_proposed":
            raise SelectorManifestError(
                "full_proposed must use the dynamic log1p marginal selector"
            )
        else:
            raise SelectorManifestError(f"unknown recovery selector: {selector}")
        result[pair_id] = utility
    return result


def automatic_exact_budget(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> int:
    # The sum of deterministic per-task medians is guaranteed feasible because
    # the corresponding concrete candidate combination exists.
    total = 0
    for task in sorted(groups):
        candidates = sorted(
            groups[task],
            key=lambda row: (
                int(row["cost"]["c_sup"]),
                str(row["candidate_pair_id"]),
            ),
        )
        total += int(candidates[(len(candidates) - 1) // 2]["cost"]["c_sup"])
    return total


def _coverage_features(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for category in row["coverage_categories"]:
        text = str(category)
        if "=" not in text:
            raise SelectorManifestError(
                f"{row['candidate_pair_id']}: malformed coverage category {text}"
            )
        field, value = text.split("=", 1)
        if field not in protocol.COVERAGE_WEIGHTS or not value:
            raise SelectorManifestError(
                f"{row['candidate_pair_id']}: unregistered coverage category {text}"
            )
        result.add((field, value))
    return result


def _coverage_marginal(
    counts: Counter[tuple[str, str]], row: Mapping[str, Any]
) -> float:
    return sum(
        float(protocol.COVERAGE_WEIGHTS[field])
        * (math.log1p(counts[(field, value)] + 1) - math.log1p(counts[(field, value)]))
        for field, value in _coverage_features(row)
    )


def _reachable_exact_cost(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    tasks: Sequence[str],
    target: int,
) -> bool:
    if target < 0:
        return False
    reachable = {0}
    for task in tasks:
        next_reachable = {
            used + int(row["cost"]["c_sup"])
            for used in reachable
            for row in groups[task]
            if used + int(row["cost"]["c_sup"]) <= target
        }
        if not next_reachable:
            return False
        reachable = next_reachable
    return target in reachable


def dynamic_marginal_selection(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    *,
    selector: str,
    target_c_sup: int,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Frozen task-order greedy with exact-budget feasibility lookahead."""
    if selector not in {"coverage_only", "full_proposed"}:
        raise SelectorManifestError("dynamic selector name is invalid")
    weights = dict(_constant("FULL_OBJECTIVE_WEIGHTS", DEFAULT_FULL_WEIGHTS))
    if weights != DEFAULT_FULL_WEIGHTS:
        raise SelectorManifestError(
            f"FULL_OBJECTIVE_WEIGHTS drift: {weights} != {DEFAULT_FULL_WEIGHTS}"
        )
    tasks = sorted(groups)
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    selected_cost = 0
    realized_utilities: dict[str, float] = {}
    trace: list[dict[str, Any]] = []
    for position, task in enumerate(tasks):
        remaining_tasks = tasks[position + 1 :]
        feasible: list[tuple[float, str, dict[str, Any], float]] = []
        for row in groups[task]:
            cost = int(row["cost"]["c_sup"])
            if not _reachable_exact_cost(
                groups,
                remaining_tasks,
                target_c_sup - selected_cost - cost,
            ):
                continue
            coverage_gain = _coverage_marginal(counts, row)
            if selector == "coverage_only":
                numerator = coverage_gain
            else:
                causal = (
                    float(row["scores"]["forced_first_kappa"]) + 1.0
                ) / 2.0
                hardness = float(
                    row["scores"]["hardness_winsorized_percentile"]
                )
                numerator = (
                    weights["causal"] * causal
                    + weights["hardness"] * hardness
                    + weights["coverage"] * coverage_gain
                )
            marginal_per_cost = numerator / cost
            feasible.append(
                (
                    marginal_per_cost,
                    str(row["candidate_pair_id"]),
                    row,
                    coverage_gain,
                )
            )
        if not feasible:
            raise SelectorManifestError(
                f"{selector}: no feasible exact-budget candidate for {task}"
            )
        feasible.sort(key=lambda item: (-item[0], item[1]))
        utility, _, chosen, coverage_gain = feasible[0]
        selected.append(chosen)
        selected_cost += int(chosen["cost"]["c_sup"])
        for feature in _coverage_features(chosen):
            counts[feature] += 1
        realized_utilities[str(chosen["candidate_pair_id"])] = utility
        trace.append(
            {
                "task_identity": task,
                "candidate_pair_id": chosen["candidate_pair_id"],
                "marginal_utility_per_c_sup": utility,
                "coverage_marginal_gain": coverage_gain,
                "feasible_candidates": len(feasible),
                "greedy_best_feasible_selected": True,
                "cumulative_c_sup": selected_cost,
            }
        )
    if selected_cost != target_c_sup:
        raise SelectorManifestError(
            f"{selector}: achieved c_sup={selected_cost}, expected {target_c_sup}"
        )
    return selected, realized_utilities, {
        "solver": "dynamic_weighted_log1p_marginal_greedy_with_exact_budget_lookahead",
        "objective_additive": False,
        "target_c_sup": target_c_sup,
        "achieved_c_sup": selected_cost,
        "approximated": False,
        "task_order": tasks,
        "trace": trace,
        "coverage_value": sum(
            protocol.COVERAGE_WEIGHTS[field] * math.log1p(count)
            for (field, _), count in counts.items()
        ),
        "marginal_audit": {
            "all_steps_selected_best_feasible_marginal": True,
            "count_each_feature_value_at_most_once_per_candidate_pair": True,
            "coverage_objective": "weighted_sum_log_one_plus_count",
        },
    }


@dataclass(frozen=True)
class _State:
    utility: float
    pair_ids: tuple[str, ...]
    c_nonpad: int


def _better(candidate: _State, incumbent: _State | None) -> bool:
    if incumbent is None:
        return True
    if candidate.utility > incumbent.utility + 1e-12:
        return True
    if abs(candidate.utility - incumbent.utility) <= 1e-12:
        if candidate.c_nonpad < incumbent.c_nonpad:
            return True
        if (
            candidate.c_nonpad == incumbent.c_nonpad
            and candidate.pair_ids < incumbent.pair_ids
        ):
            return True
    return False


def exact_matched_selection(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    utilities: Mapping[str, float],
    *,
    target_c_sup: int,
    max_states: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = sorted(groups)
    if target_c_sup <= 0:
        raise SelectorManifestError("target_c_sup must be positive")
    future_min = [0] * (len(tasks) + 1)
    future_max = [0] * (len(tasks) + 1)
    for index in range(len(tasks) - 1, -1, -1):
        costs = [int(row["cost"]["c_sup"]) for row in groups[tasks[index]]]
        future_min[index] = future_min[index + 1] + min(costs)
        future_max[index] = future_max[index + 1] + max(costs)
    if not future_min[0] <= target_c_sup <= future_max[0]:
        raise SelectorManifestError(
            f"target_c_sup={target_c_sup} outside matched-task feasible "
            f"range [{future_min[0]}, {future_max[0]}]"
        )

    states: dict[int, _State] = {0: _State(0.0, (), 0)}
    peak_states = 1
    for index, task in enumerate(tasks):
        next_states: dict[int, _State] = {}
        for used, state in states.items():
            for row in groups[task]:
                pair_id = str(row["candidate_pair_id"])
                new_used = used + int(row["cost"]["c_sup"])
                if (
                    new_used + future_min[index + 1] > target_c_sup
                    or new_used + future_max[index + 1] < target_c_sup
                ):
                    continue
                candidate = _State(
                    state.utility + float(utilities[pair_id]),
                    state.pair_ids + (pair_id,),
                    state.c_nonpad + int(row["cost"]["c_nonpad"]),
                )
                if _better(candidate, next_states.get(new_used)):
                    next_states[new_used] = candidate
        if not next_states:
            raise SelectorManifestError(
                "no exactly budget-matched selection remains feasible "
                f"after task {task}"
            )
        if len(next_states) > max_states:
            raise SelectorManifestError(
                f"exact DP requires {len(next_states)} states after {task}, "
                f"above fail-closed limit {max_states}; do not approximate"
            )
        states = next_states
        peak_states = max(peak_states, len(states))
    final = states.get(target_c_sup)
    if final is None:
        raise SelectorManifestError(
            f"no exactly budget-matched selection at c_sup={target_c_sup}"
        )
    by_id = {
        str(row["candidate_pair_id"]): row
        for candidates in groups.values()
        for row in candidates
    }
    selected = [by_id[pair_id] for pair_id in final.pair_ids]
    return selected, {
        "solver": "exact_multiple_choice_knapsack",
        "objective_additive": True,
        "target_c_sup": target_c_sup,
        "achieved_c_sup": target_c_sup,
        "objective_utility": final.utility,
        "peak_dp_states": peak_states,
        "max_dp_states": max_states,
        "approximated": False,
    }


def marginal_audit(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Sequence[Mapping[str, Any]],
    utilities: Mapping[str, float],
) -> dict[str, Any]:
    chosen = {
        str(row["task_identity"]): row
        for row in selected
    }
    local_regrets: dict[str, float] = {}
    same_cost_violations: list[dict[str, Any]] = []
    for task, options in groups.items():
        current = chosen[task]
        selected_utility = float(utilities[str(current["candidate_pair_id"])])
        best_utility = max(float(utilities[str(row["candidate_pair_id"])]) for row in options)
        local_regrets[task] = best_utility - selected_utility
        current_cost = int(current["cost"]["c_sup"])
        for alternative in options:
            if int(alternative["cost"]["c_sup"]) != current_cost:
                continue
            gain = (
                float(utilities[str(alternative["candidate_pair_id"])])
                - selected_utility
            )
            if gain > 1e-10:
                same_cost_violations.append(
                    {
                        "task_identity": task,
                        "selected": current["candidate_pair_id"],
                        "alternative": alternative["candidate_pair_id"],
                        "utility_gain": gain,
                    }
                )

    pairwise_violations: list[dict[str, Any]] = []
    tasks = sorted(groups)
    for left_index, left_task in enumerate(tasks):
        left_selected = chosen[left_task]
        for right_task in tasks[left_index + 1 :]:
            right_selected = chosen[right_task]
            current_cost = int(left_selected["cost"]["c_sup"]) + int(
                right_selected["cost"]["c_sup"]
            )
            current_utility = float(
                utilities[str(left_selected["candidate_pair_id"])]
            ) + float(utilities[str(right_selected["candidate_pair_id"])])
            for left_alt in groups[left_task]:
                for right_alt in groups[right_task]:
                    if (
                        int(left_alt["cost"]["c_sup"])
                        + int(right_alt["cost"]["c_sup"])
                        != current_cost
                    ):
                        continue
                    gain = (
                        float(utilities[str(left_alt["candidate_pair_id"])])
                        + float(utilities[str(right_alt["candidate_pair_id"])])
                        - current_utility
                    )
                    if gain > 1e-10:
                        pairwise_violations.append(
                            {
                                "tasks": [left_task, right_task],
                                "selected": [
                                    left_selected["candidate_pair_id"],
                                    right_selected["candidate_pair_id"],
                                ],
                                "alternatives": [
                                    left_alt["candidate_pair_id"],
                                    right_alt["candidate_pair_id"],
                                ],
                                "utility_gain": gain,
                            }
                        )
                        break
                if pairwise_violations and pairwise_violations[-1]["tasks"] == [
                    left_task,
                    right_task,
                ]:
                    break
    if same_cost_violations or pairwise_violations:
        raise SelectorManifestError(
            "selected manifest failed exact-budget marginal optimality audit"
        )
    return {
        "same_task_same_cost_improving_swaps": 0,
        "two_task_budget_neutral_improving_swaps": 0,
        "mean_unconstrained_local_regret": (
            sum(local_regrets.values()) / len(local_regrets)
        ),
        "max_unconstrained_local_regret": max(local_regrets.values()),
        "local_regret_is_expected_when_exact_global_budget_binds": True,
    }


def _selected_record(
    row: Mapping[str, Any], utility: float
) -> dict[str, Any]:
    return {
        "candidate_pair_id": row["candidate_pair_id"],
        "choice_set_id": row["choice_set_id"],
        "task_identity": row["task_identity"],
        "domain": row["domain"],
        "branch_ids": row["branch_ids"],
        "c_sup": row["cost"]["c_sup"],
        "c_nonpad": row["cost"]["c_nonpad"],
        "forced_first_kappa": row["scores"]["forced_first_kappa"],
        "hardness_winsorized_percentile": row["scores"][
            "hardness_winsorized_percentile"
        ],
        "low_kappa": row["scores"]["low_kappa"],
        "coverage_categories": row["coverage_categories"],
        "selector_utility": utility,
        # Preserve the complete immutable selector atom.  A downstream
        # materializer can therefore build assistant-only-masked training rows
        # without re-querying a mutable candidate database or splitting the
        # sibling pair.
        "candidate_pair": dict(row),
    }


def _flawless_manifest(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_c_sup: int,
    pool_sha256: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for task, rows in groups.items():
        views = [
            (row.get("clean_view"), str(row["candidate_pair_id"]))
            for row in rows
        ]
        if any(not isinstance(view, dict) for view, _ in views):
            raise SelectorManifestError(
                f"{task}: flawless_only requested but clean_view is missing"
            )
        # The clean counterpart may be pair-bound.  Freeze one without reading
        # selector scores or downstream outcomes: canonical clean-view hash,
        # then candidate_pair_id as the deterministic tie break.
        views.sort(
            key=lambda item: (
                canonical_sha256(item[0]),
                item[1],
            )
        )
        view = dict(views[0][0])
        clean_id = str(view.get("clean_id", ""))
        if not clean_id:
            raise SelectorManifestError(f"{task}: clean_view.clean_id is missing")
        c_sup = _positive_int(view.get("c_sup"), f"{task}.clean_view.c_sup")
        c_nonpad = _positive_int(
            view.get("c_nonpad"), f"{task}.clean_view.c_nonpad"
        )
        if c_sup > c_nonpad:
            raise SelectorManifestError(f"{task}: clean c_sup exceeds c_nonpad")
        selected.append(
            {
                "clean_id": clean_id,
                "task_identity": task,
                "domain": rows[0]["domain"],
                "c_sup": c_sup,
                "c_nonpad": c_nonpad,
                "training_view": "flawless_only",
                "selection_rule": (
                    "minimum_canonical_clean_view_sha256_then_candidate_pair_id"
                ),
                "clean_view": view,
            }
        )
    selected.sort(key=lambda row: str(row["task_identity"]))
    actual_sup = sum(int(row["c_sup"]) for row in selected)
    return {
        "protocol": MANIFEST_PROTOCOL,
        "selector": "flawless_only",
        "selection_seed": None,
        "training_view": "flawless_only",
        "candidate_pool_sha256": pool_sha256,
        "matched_task_ids": [row["task_identity"] for row in selected],
        "selected": selected,
        "budget": {
            "target_c_sup": target_c_sup,
            "actual_c_sup": actual_sup,
            "remaining_c_sup": target_c_sup - actual_sup,
            "actual_c_nonpad": sum(int(row["c_nonpad"]) for row in selected),
            "recovery_primary_target_c_sup": target_c_sup,
            "relative_difference_from_recovery_target": (
                (actual_sup - target_c_sup) / target_c_sup
            ),
            "budget_comparable_primary": False,
            "reason": "natural_clean_token_mass_not_posthoc_manipulated",
        },
        "official_test_used": False,
    }


def build_manifests(
    rows: Sequence[Mapping[str, Any]],
    *,
    selectors: Sequence[str] | None = None,
    target_c_sup: int | None = None,
    max_dp_states: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    verified = verify_scored_pool(rows)
    groups = group_matched_tasks(verified)
    registered_tasks = sorted(groups)
    budget = (
        automatic_exact_budget(groups)
        if target_c_sup is None
        else _positive_int(target_c_sup, "target_c_sup")
    )
    state_limit = (
        int(_constant("MAX_SELECTION_DP_STATES", DEFAULT_MAX_DP_STATES))
        if max_dp_states is None
        else max_dp_states
    )
    requested = list(selectors or _constant("SELECTORS", DEFAULT_SELECTORS))
    aliases = {
        "random": "random_stratified",
        "coverage": "coverage_only",
        "hardness": "hardness_only",
        "causal": "causal_only",
        "full": "full_proposed",
        "proposed": "full_proposed",
    }
    requested = [aliases.get(item, item) for item in requested]
    allowed = {*DEFAULT_SELECTORS, "flawless_only"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise SelectorManifestError(f"unknown selectors: {unknown}")
    pool_sha256 = canonical_sha256(verified)
    coverage_scores = coverage_rarity_scores(verified)
    random_seeds = tuple(
        int(seed)
        for seed in _constant("RANDOM_STRATIFIED_SEEDS", DEFAULT_RANDOM_SEEDS)
    )
    if len(random_seeds) != 3 or len(set(random_seeds)) != 3:
        raise SelectorManifestError(
            "random_stratified requires exactly three distinct frozen seeds"
        )

    manifests: dict[str, dict[str, Any]] = {}
    if "flawless_only" in requested:
        manifests["flawless_only"] = _flawless_manifest(
            groups, target_c_sup=budget, pool_sha256=pool_sha256
        )
    for selector in requested:
        if selector in {"flawless_only", "random_stratified"}:
            continue
        if selector in {"coverage_only", "full_proposed"}:
            selected, utilities, solver_audit = dynamic_marginal_selection(
                groups, selector=selector, target_c_sup=budget
            )
            marginal = solver_audit["marginal_audit"]
        else:
            utilities = selector_utilities(
                verified,
                selector,
                random_seed=None,
                coverage_scores=coverage_scores,
            )
            selected, solver_audit = exact_matched_selection(
                groups,
                utilities,
                target_c_sup=budget,
                max_states=state_limit,
            )
            marginal = marginal_audit(groups, selected, utilities)
        records = [
            _selected_record(row, utilities[str(row["candidate_pair_id"])])
            for row in selected
        ]
        records.sort(key=lambda row: str(row["task_identity"]))
        manifests[selector] = {
            "protocol": MANIFEST_PROTOCOL,
            "selector": selector,
            "selection_seed": None,
            "training_view": "complete_candidate_pair_two_sibling_branches",
            "candidate_pool_sha256": pool_sha256,
            "matched_task_ids": registered_tasks,
            "selected": records,
            "budget": {
                "target_c_sup": budget,
                "actual_c_sup": sum(int(row["c_sup"]) for row in records),
                "remaining_c_sup": budget
                - sum(int(row["c_sup"]) for row in records),
                "actual_c_nonpad": sum(int(row["c_nonpad"]) for row in records),
            },
            "solver_audit": solver_audit,
            "marginal_audit": marginal,
            "official_test_used": False,
        }
    if "random_stratified" in requested:
        for seed in random_seeds:
            name = f"random_stratified_seed_{seed}"
            utilities = selector_utilities(
                verified,
                "random_stratified",
                random_seed=seed,
                coverage_scores=coverage_scores,
            )
            selected, solver_audit = exact_matched_selection(
                groups,
                utilities,
                target_c_sup=budget,
                max_states=state_limit,
            )
            marginal = marginal_audit(groups, selected, utilities)
            records = [
                _selected_record(row, utilities[str(row["candidate_pair_id"])])
                for row in selected
            ]
            records.sort(key=lambda row: str(row["task_identity"]))
            manifests[name] = {
                "protocol": MANIFEST_PROTOCOL,
                "selector": "random_stratified",
                "manifest_name": name,
                "selection_seed": seed,
                "stratification": "exactly_one_pair_per_task_and_fixed_domain_task_support",
                "training_view": "complete_candidate_pair_two_sibling_branches",
                "candidate_pool_sha256": pool_sha256,
                "matched_task_ids": registered_tasks,
                "selected": records,
                "budget": {
                    "target_c_sup": budget,
                    "actual_c_sup": sum(int(row["c_sup"]) for row in records),
                    "remaining_c_sup": budget
                    - sum(int(row["c_sup"]) for row in records),
                    "actual_c_nonpad": sum(
                        int(row["c_nonpad"]) for row in records
                    ),
                },
                "solver_audit": solver_audit,
                "marginal_audit": marginal,
                "official_test_used": False,
            }

    recovery_manifests = {
        name: manifest
        for name, manifest in manifests.items()
        if manifest["selector"] != "flawless_only"
    }
    if not recovery_manifests and "flawless_only" not in manifests:
        raise SelectorManifestError("no manifests requested")
    for name, manifest in recovery_manifests.items():
        task_ids = [str(row["task_identity"]) for row in manifest["selected"]]
        if task_ids != registered_tasks or len(set(task_ids)) != len(registered_tasks):
            raise SelectorManifestError(
                f"{name}: must select exactly one pair for every matched task"
            )
        pair_ids = [str(row["candidate_pair_id"]) for row in manifest["selected"]]
        if len(pair_ids) != len(set(pair_ids)):
            raise SelectorManifestError(f"{name}: duplicate candidate pair")
        if manifest["budget"]["actual_c_sup"] != budget:
            raise SelectorManifestError(f"{name}: exact c_sup budget drift")
    sup_totals = [
        int(manifest["budget"]["actual_c_sup"])
        for manifest in recovery_manifests.values()
    ]
    spread = (
        (max(sup_totals) - min(sup_totals)) / max(sup_totals)
        if sup_totals
        else 0.0
    )
    max_spread = float(
        _constant(
            "MAX_ARM_TOKEN_SPREAD_FRACTION",
            _constant(
                "MAX_SUPERVISED_BUDGET_SPREAD", DEFAULT_MAX_BUDGET_SPREAD
            ),
        )
    )
    if spread > max_spread + 1e-12:
        raise SelectorManifestError(
            f"recovery-arm c_sup spread {spread:.6f} exceeds {max_spread:.6f}"
        )
    nonpad_totals = {
        name: int(manifest["budget"]["actual_c_nonpad"])
        for name, manifest in recovery_manifests.items()
    }
    nonpad_spread = (
        (max(nonpad_totals.values()) - min(nonpad_totals.values()))
        / max(nonpad_totals.values())
        if nonpad_totals
        else 0.0
    )
    audit = {
        "protocol": MANIFEST_PROTOCOL,
        "design_protocol": _constant(
            "PROTOCOL", "v6_causal_recovery_selection_v1"
        ),
        "status": "PASS",
        "candidate_pool_sha256": pool_sha256,
        "candidate_pairs": len(verified),
        "tasks": len(registered_tasks),
        "candidate_pairs_per_task": {
            task: len(groups[task]) for task in registered_tasks
        },
        "matched_task_ids": registered_tasks,
        "exactly_one_candidate_pair_per_task": True,
        "candidate_pair_atom_split": False,
        "target_c_sup": budget,
        "automatic_budget": target_c_sup is None,
        "recovery_arm_c_sup": {
            name: manifest["budget"]["actual_c_sup"]
            for name, manifest in recovery_manifests.items()
        },
        "recovery_arm_c_sup_relative_spread": spread,
        "max_registered_c_sup_relative_spread": max_spread,
        "recovery_arm_c_nonpad": nonpad_totals,
        "recovery_arm_c_nonpad_relative_spread": nonpad_spread,
        "c_nonpad_is_reported_not_posthoc_equalized": True,
        "selectors": sorted(manifests),
        "random_stratified_seeds": list(random_seeds),
        "low_kappa_candidates_were_eligible": True,
        "validation_or_official_test_outcomes_used_for_selection": False,
        "official_test_used": False,
        "official_test_sealed": True,
    }
    return manifests, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selectors",
        default=",".join(DEFAULT_SELECTORS),
        help=(
            "Comma-separated selectors. random_stratified expands to the three "
            "frozen random seeds. flawless_only is supported when clean_view is present."
        ),
    )
    parser.add_argument("--target-c-sup", type=int)
    parser.add_argument("--max-dp-states", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SelectorManifestError(f"output directory must be absent: {args.output_dir}")
    selectors = [item.strip() for item in args.selectors.split(",") if item.strip()]
    rows = scoring.read_jsonl(args.scored_pool)
    manifests, audit = build_manifests(
        rows,
        selectors=selectors,
        target_c_sup=args.target_c_sup,
        max_dp_states=args.max_dp_states,
    )
    args.output_dir.mkdir(parents=True)
    for name, manifest in sorted(manifests.items()):
        write_json(args.output_dir / f"{name}.json", manifest)
        manifest["manifest_sha256"] = sha256_file(
            args.output_dir / f"{name}.json"
        )
    audit["input_scored_pool_sha256"] = sha256_file(args.scored_pool)
    audit["manifests"] = {
        name: {
            "path": f"{name}.json",
            "sha256": sha256_file(args.output_dir / f"{name}.json"),
        }
        for name in sorted(manifests)
    }
    write_json(args.output_dir / "selector_audit.json", audit)


if __name__ == "__main__":
    main()
