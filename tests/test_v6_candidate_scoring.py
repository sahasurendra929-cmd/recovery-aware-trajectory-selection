from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from scripts import score_v6_candidates as score


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def q_cell(success: float, *, salt: str = "common") -> dict:
    return {
        "task_success": success,
        "forced_first_only": True,
        "continuation_policy_sha256": digest(f"policy:{salt}"),
        "continuation_seed_set_sha256": digest(f"seeds:{salt}"),
        "decoding_sha256": digest(f"decoding:{salt}"),
        "rollout_budget": 4,
        "gold_suffix_visible": False,
    }


def logprob_cell(total: float, tokens: int) -> dict:
    return {"sum_logprob": total, "token_count": tokens}


def candidate_pair(
    task_index: int,
    option_index: int,
    *,
    c_sup: int = 20,
    kappa_cells: tuple[float, float, float, float] = (0.8, 0.2, 0.1, 0.6),
    hardness_offset: float = 0.0,
) -> dict:
    task = f"retail:{task_index}"
    pair_id = f"{task}:pair-{option_index}"
    # Split odd budgets deterministically while preserving the exact pair sum.
    first_sup = c_sup // 2
    second_sup = c_sup - first_sup
    q11, q12, q21, q22 = kappa_cells
    return {
        "candidate_pair_id": pair_id,
        "choice_set_id": f"{task}:prefix-0",
        "task_identity": task,
        "domain": "retail",
        "prefix_sha256": digest(f"{task}:prefix"),
        "environment_snapshot_sha256": digest(f"{task}:snapshot"),
        "quality": {
            "real_error_executed": True,
            "matched_recovery_replay_success": True,
            "cross_replay_complete": True,
            "no_future_leakage": True,
            "independent_replay_audited": True,
            "official_test_used": False,
            "failed_positive_labels": 0,
        },
        "branches": [
            {
                "branch_id": f"{pair_id}:e1",
                "error_family": f"error-{option_index}-a",
                "error_event_sha256": digest(f"{pair_id}:error-a"),
                "first_recovery_action_key": f"action-{option_index}-a",
                "first_recovery_action": {
                    "name": "search_order",
                    "arguments": {"order_id": f"{task_index}-{option_index}-a"},
                    "requestor": "assistant",
                },
                "supervised_target_tokens": first_sup,
                "nonpadding_tokens": first_sup + 20 + option_index,
                "coverage": {
                    "failed_tool": "lookup_order",
                    "corrective_action": f"repair-{option_index}-a",
                    "recovery_length_bin": "short",
                    "recovery_mode": "agent_initiated",
                },
                "prompt": [{"role": "tool", "error": True}],
                "fresh_recovery_suffix": [{"role": "assistant", "tool_calls": []}],
                "loss_mask": [False, True],
            },
            {
                "branch_id": f"{pair_id}:e2",
                "error_family": f"error-{option_index}-b",
                "error_event_sha256": digest(f"{pair_id}:error-b"),
                "first_recovery_action_key": f"action-{option_index}-b",
                "first_recovery_action": {
                    "name": "lookup_user",
                    "arguments": {"user_id": f"{task_index}-{option_index}-b"},
                    "requestor": "assistant",
                },
                "supervised_target_tokens": second_sup,
                "nonpadding_tokens": second_sup + 25 + option_index,
                "coverage": {
                    "failed_tool": "lookup_order",
                    "corrective_action": f"repair-{option_index}-b",
                    "recovery_length_bin": "medium",
                    "recovery_mode": "user_assisted",
                },
                "prompt": [{"role": "tool", "error": True}],
                "fresh_recovery_suffix": [{"role": "assistant", "tool_calls": []}],
                "loss_mask": [False, True],
            },
        ],
        "forced_first_q": {
            "q_e1_a1": q_cell(q11),
            "q_e1_a2": q_cell(q12),
            "q_e2_a1": q_cell(q21),
            "q_e2_a2": q_cell(q22),
        },
        "first_action_logprobs": {
            "q_e1_a1": logprob_cell(-4.0 - hardness_offset, 2),
            "q_e1_a2": logprob_cell(-1.0, 1),
            "q_e2_a1": logprob_cell(-6.0, 2),
            "q_e2_a2": logprob_cell(-6.0 - hardness_offset, 3),
        },
        "clean_view": {
            "clean_id": f"{task}:clean",
            "c_sup": 20,
            "c_nonpad": 100,
            "messages": [{"role": "assistant", "content": "complete"}],
            "label_mask": [True],
        },
    }


def test_forced_first_kappa_and_length_normalized_hardness():
    row = candidate_pair(1, 0)
    kappa, cells = score.forced_first_kappa(row)
    hardness, components = score.length_normalized_hardness(row)
    assert kappa == pytest.approx(0.55)
    assert cells == {
        "q_e1_a1": 0.8,
        "q_e1_a2": 0.2,
        "q_e2_a1": 0.1,
        "q_e2_a2": 0.6,
    }
    assert hardness == pytest.approx(1.0)
    assert components["error_1_wrong_minus_matched"] == pytest.approx(1.0)
    assert components["error_2_wrong_minus_matched"] == pytest.approx(-1.0)


def test_scoring_retains_low_kappa_and_computes_both_costs():
    rows = [
        candidate_pair(1, 0, c_sup=19),
        candidate_pair(
            1,
            1,
            c_sup=20,
            kappa_cells=(0.5, 0.45, 0.45, 0.5),
            hardness_offset=2.0,
        ),
        candidate_pair(1, 2, c_sup=21, hardness_offset=4.0),
    ]
    scored, audit = score.score_candidates(rows)
    assert len(scored) == 3
    assert audit["candidate_pairs_in"] == audit["candidate_pairs_out"] == 3
    assert audit["low_kappa_pairs_retained"] == 1
    by_id = {row["candidate_pair_id"]: row for row in scored}
    low = by_id["retail:1:pair-1"]
    assert low["scores"]["low_kappa"] is True
    assert low["cost"]["c_sup"] == 20
    assert low["cost"]["c_nonpad"] > low["cost"]["c_sup"]
    assert 0.0 <= low["scores"]["hardness_winsorized_percentile"] <= 1.0
    assert "error_family=error-1-a" in low["coverage_categories"]
    assert low["branches"][0]["prompt"] == [{"role": "tool", "error": True}]


def test_quality_gate_is_fail_closed():
    row = candidate_pair(1, 0)
    row["quality"]["no_future_leakage"] = False
    with pytest.raises(score.CandidateScoringError, match="no_future_leakage"):
        score.score_candidates([row])


def test_q_cells_require_identical_continuation_contract():
    row = candidate_pair(1, 0)
    row["forced_first_q"]["q_e1_a2"]["rollout_budget"] = 5
    with pytest.raises(score.CandidateScoringError, match="rollout_budget differs"):
        score.score_candidates([row])


def test_pair_requires_distinct_sibling_actions():
    row = candidate_pair(1, 0)
    row["branches"][1]["first_recovery_action_key"] = row["branches"][0][
        "first_recovery_action_key"
    ]
    with pytest.raises(score.CandidateScoringError, match="distinct IDs, errors"):
        score.score_candidates([row])


def test_same_complete_call_with_different_declared_keys_is_rejected():
    row = candidate_pair(1, 0)
    row["branches"][1]["first_recovery_action"] = deepcopy(
        row["branches"][0]["first_recovery_action"]
    )
    # The producer-declared action keys remain different; complete call
    # semantics, not those labels, control identifiability.
    assert (
        row["branches"][0]["first_recovery_action_key"]
        != row["branches"][1]["first_recovery_action_key"]
    )
    with pytest.raises(score.CandidateScoringError, match="complete"):
        score.score_candidates([row])


def test_tampered_positive_log_probability_is_rejected():
    row = candidate_pair(1, 0)
    row["first_action_logprobs"]["q_e1_a1"]["sum_logprob"] = 0.1
    with pytest.raises(score.CandidateScoringError, match="cannot be positive"):
        score.score_candidates([row])


def test_accepts_and_verifies_normalized_auditor_schema():
    row = candidate_pair(1, 0)
    row["forced_first_cells"] = row.pop("forced_first_q")
    for cell in row["forced_first_cells"].values():
        cell["official_task_success"] = cell.pop("task_success")
    row["kappa"] = 0.55
    row["c_sup"] = 20
    row["c_nonpad"] = sum(
        branch["nonpadding_tokens"] for branch in row["branches"]
    )
    row["cost"] = {"c_sup": row["c_sup"], "c_nonpad": row["c_nonpad"]}
    row["audit_status"] = "ACCEPTED"
    row["audited_candidate_pair_sha256"] = score.canonical_sha256(row)
    scored, _ = score.score_candidates([row])
    assert scored[0]["cost"]["c_sup"] == 20
    assert scored[0]["scores"]["forced_first_kappa"] == pytest.approx(0.55)

    row["c_sup"] = 21
    with pytest.raises(score.CandidateScoringError, match="does not verify"):
        score.score_candidates([row])
