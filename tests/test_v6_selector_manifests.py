from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import build_v6_selector_manifests as build
from scripts import score_v6_candidates as score
from scripts import v6_selection_protocol as protocol
from tests.test_v6_candidate_scoring import candidate_pair


def scored_pool(tasks: int = 3) -> list[dict]:
    raw = []
    for task in range(1, tasks + 1):
        raw.extend(
            [
                candidate_pair(
                    task,
                    0,
                    c_sup=10,
                    kappa_cells=(0.95, 0.05, 0.05, 0.95),
                    hardness_offset=0.0,
                ),
                candidate_pair(
                    task,
                    1,
                    c_sup=20,
                    kappa_cells=(0.60, 0.45, 0.45, 0.60),
                    hardness_offset=2.0,
                ),
                candidate_pair(
                    task,
                    2,
                    c_sup=30,
                    kappa_cells=(0.80, 0.20, 0.20, 0.80),
                    hardness_offset=4.0,
                ),
            ]
        )
    scored, _ = score.score_candidates(raw)
    return scored


def test_all_selectors_are_exact_budget_and_matched_task():
    rows = scored_pool()
    manifests, audit = build.build_manifests(
        rows,
        selectors=build.DEFAULT_SELECTORS,
    )
    # Seven deterministic selector families plus three random-seed manifests,
    # with random_stratified represented only by its three seeded instances.
    assert len(manifests) == 9
    assert len([name for name in manifests if name.startswith("random_")]) == 3
    assert audit["target_c_sup"] == 60
    assert audit["recovery_arm_c_sup_relative_spread"] == 0.0
    expected_tasks = ["retail:1", "retail:2", "retail:3"]
    for manifest in manifests.values():
        assert manifest["matched_task_ids"] == expected_tasks
        assert manifest["budget"]["actual_c_sup"] == 60
        assert manifest["budget"]["remaining_c_sup"] == 0
        assert len(manifest["selected"]) == 3
        assert {row["task_identity"] for row in manifest["selected"]} == set(
            expected_tasks
        )
        assert all(len(row["branch_ids"]) == 2 for row in manifest["selected"])
        assert all(
            len(row["candidate_pair"]["branches"]) == 2
            for row in manifest["selected"]
        )
        if manifest["selector"] in {"coverage_only", "full_proposed"}:
            assert manifest["marginal_audit"][
                "all_steps_selected_best_feasible_marginal"
            ] is True
        else:
            assert manifest["marginal_audit"][
                "two_task_budget_neutral_improving_swaps"
            ] == 0


def test_random_stratified_is_reproducible_for_all_three_frozen_seeds():
    rows = scored_pool()
    first, _ = build.build_manifests(
        rows, selectors=["random_stratified"]
    )
    second, _ = build.build_manifests(
        rows, selectors=["random_stratified"]
    )
    assert first == second
    assert len(first) == 3
    assert {manifest["selection_seed"] for manifest in first.values()} == set(
        build.DEFAULT_RANDOM_SEEDS
    )


def test_flawless_only_can_be_emitted_with_directional_arms():
    rows = scored_pool()
    for row in rows:
        row["clean_view"]["c_sup"] = 5
        row["clean_view"]["c_nonpad"] = 40
    manifests, audit = build.build_manifests(
        rows,
        selectors=["flawless_only", "random_stratified", "full_proposed"],
    )
    assert "flawless_only" in manifests
    assert "full_proposed" in manifests
    assert len(manifests) == 5
    clean = manifests["flawless_only"]
    assert clean["budget"]["actual_c_sup"] == 15
    assert audit["target_c_sup"] == 60
    assert clean["budget"]["budget_comparable_primary"] is False
    assert clean["budget"]["reason"] == "natural_clean_token_mass_not_posthoc_manipulated"
    assert all(row["training_view"] == "flawless_only" for row in clean["selected"])
    assert all("messages" in row["clean_view"] for row in clean["selected"])


def test_full_weights_bind_exactly_to_frozen_protocol_and_have_no_cost_term():
    assert build.DEFAULT_FULL_WEIGHTS == protocol.FULL_OBJECTIVE_WEIGHTS
    assert build.DEFAULT_FULL_WEIGHTS == {
        "causal": 0.50,
        "hardness": 0.25,
        "coverage": 0.25,
    }


def test_coverage_marginal_has_registered_log1p_diminishing_returns():
    row = scored_pool(tasks=1)[0]
    counts = build.Counter()
    first = build._coverage_marginal(counts, row)
    for feature in build._coverage_features(row):
        counts[feature] += 1
    second = build._coverage_marginal(counts, row)
    assert second < first


def test_primary_matched_task_fails_with_fewer_than_three_pairs():
    rows = scored_pool(tasks=2)
    rows = [
        row
        for row in rows
        if not (
            row["task_identity"] == "retail:2"
            and row["candidate_pair_id"].endswith("pair-2")
        )
    ]
    with pytest.raises(build.SelectorManifestError, match="need >= 3"):
        build.build_manifests(rows, selectors=["full_proposed"])


def test_infeasible_exact_budget_fails_without_relaxation():
    rows = scored_pool()
    with pytest.raises(build.SelectorManifestError, match="no exactly budget-matched"):
        build.build_manifests(
            rows,
            selectors=["causal_only"],
            target_c_sup=61,
        )


def test_score_tampering_is_detected_before_selection():
    rows = scored_pool()
    rows[0]["scores"]["forced_first_kappa"] = -999.0
    with pytest.raises(build.SelectorManifestError, match="does not match recomputation"):
        build.build_manifests(rows, selectors=["full_proposed"])


def test_choice_set_or_snapshot_drift_is_rejected():
    rows = scored_pool()
    changed = deepcopy(rows)
    changed[0]["choice_set_id"] = "retail:1:other-prefix"
    with pytest.raises(build.SelectorManifestError, match="one choice_set_id"):
        build.build_manifests(changed, selectors=["full_proposed"])
