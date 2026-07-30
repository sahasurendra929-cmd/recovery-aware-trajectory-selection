from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from scripts import v6_selection_protocol as protocol


def forced_first_cells(kappa: float = 1.0):
    common = {
        "forced_first_only": True,
        "gold_suffix_visible": False,
        "continuation_policy_sha256": "policy",
        "continuation_seed_set_sha256": "seeds",
        "decoding_sha256": "decoding",
        "rollout_budget": 60,
    }
    cross_reward = 1.0 - kappa
    return {
        "q_e1_a1": {**common, "task_success": 1.0},
        "q_e1_a2": {**common, "task_success": cross_reward},
        "q_e2_a1": {**common, "task_success": cross_reward},
        "q_e2_a2": {**common, "task_success": 1.0},
    }


def _branch_contract(domain: str, branch_index: int):
    if domain == "airline":
        values = (
            ("get_reservation_details", "reservation_id", "READ"),
            ("cancel_reservation", "reservation_id", "WRITE"),
        )
    else:
        values = (
            ("get_order_details", "order_id", "READ"),
            ("cancel_pending_order", "order_id", "WRITE"),
        )
    tool, key, action_kind = values[branch_index]
    return tool, key, action_kind, f"{tool}::{key}"


def valid_candidate_pair(task: str, index: int, *, kappa: float | None = None):
    domain = task.split(":", 1)[0]
    choice_set_id = f"{task}:site:0"
    kappa = (1.0, 0.0, 0.5, 0.5)[index % 4] if kappa is None else kappa
    error_families = (
        "not_found",
        "invalid_state",
        "policy_rejection",
        "ambiguous_lookup",
    )
    branches = []
    for branch_index in range(2):
        tool, key, kind, family = _branch_contract(domain, branch_index)
        reference_arguments = {
            key: f"valid-{index}-{branch_index}",
            **({"reason": "requested"} if kind == "WRITE" else {}),
        }
        reference = protocol.canonical_reference_action_key(
            tool, reference_arguments
        )
        branches.append(
            {
                "branch_id": f"{choice_set_id}:candidate:{index}:branch:{branch_index}",
                "domain": domain,
                "tool_name": tool,
                "identifier_key": key,
                "reference_action_kind": kind,
                "reference_arguments": reference_arguments,
                "corrective_family": family,
                "original_identifier": f"valid-{index}-{branch_index}",
                "mutated_identifier": f"invalid-{index}-{branch_index}",
                "identifier_field_allowlisted": True,
                "identifier_mutation_type_preserving": True,
                "error_event_sha256": f"error-{task}-{index}-{branch_index}",
                "error_family": error_families[
                    (index + branch_index) % len(error_families)
                ],
                "failed_tool": tool,
                "first_recovery_action_key": reference,
                "reference_action_key": reference,
                "corrective_action": family,
                "recovery_length_bin": "short" if index % 2 == 0 else "long",
                "recovery_mode": "agent_initiated",
                "actual_tool_error": True,
                "state_unchanged_after_error": True,
                "corrective_first_action_is_reference_call": True,
                "matched_task_success": True,
                "agent_db_hash_before_error": f"agent-{task}",
                "agent_db_hash_after_error": f"agent-{task}",
                "user_db_hash_available": True,
                "user_db_hash_before_error": f"user-{task}",
                "user_db_hash_after_error": f"user-{task}",
            }
        )
    return {
        "candidate_pair_id": f"{choice_set_id}:candidate:{index}",
        "choice_set_id": choice_set_id,
        "task_identity": task,
        "domain": domain,
        "partition": "arm_train",
        "prefix_sha256": f"prefix-{task}",
        "environment_snapshot_sha256": f"snapshot-{task}",
        "branches": branches,
        "forced_first_cells": forced_first_cells(kappa),
        "kappa": kappa,
        "c_sup": 200 + index,
        "c_nonpad": 1000 + index,
        "quality": {
            "real_error_executed": True,
            "matched_recovery_replay_success": True,
            "cross_replay_complete": True,
            "no_future_leakage": True,
            "failed_positive_labels": 0,
            "independent_replay_audited": True,
            "official_test_used": False,
        },
    }


def valid_choice_set(task: str, pair_count: int = 4):
    return {
        "choice_set_id": f"{task}:site:0",
        "task_identity": task,
        "domain": task.split(":", 1)[0],
        "prefix_sha256": f"prefix-{task}",
        "environment_snapshot_sha256": f"snapshot-{task}",
        "candidate_pairs": [
            valid_candidate_pair(task, index) for index in range(pair_count)
        ],
    }


class V6SelectionProtocolTests(unittest.TestCase):
    def test_frozen_partition_structural_registry_pilot_and_test_seal(self):
        protocol.validate_constants()
        summary = protocol.frozen_summary()
        self.assertEqual(summary["partition"]["source_inner_train_tasks"], 83)
        self.assertEqual(summary["partition"]["arm_train_tasks"], 70)
        self.assertEqual(len(summary["partition"]["loss_validation_task_ids"]), 8)
        self.assertEqual(len(summary["partition"]["excluded_task_ids"]), 5)
        self.assertEqual(
            summary["structural_eligibility"]["eligible_counts"],
            {"retail": 48, "airline": 12},
        )
        self.assertEqual(
            summary["structural_eligibility"]["canonical_sha256"],
            "002b5a2c5d83a4dab6dc8ce398d81bb8541bf7bfc6b25c04e62ce9ed179587f7",
        )
        self.assertFalse(summary["structural_eligibility"]["legacy_hash_match"])
        self.assertTrue(
            summary["structural_eligibility"]["legacy_mismatch_acknowledged"]
        )
        action = summary["structural_eligibility"]["action_identifiability"]
        self.assertEqual(action["eligible_counts"], {"retail": 39, "airline": 11})
        self.assertEqual(
            action["canonical_sha256"],
            "fd47afd115fbe341ec2432533231545c24e0cf583b745e17f4862979263922aa",
        )
        self.assertEqual(len(action["excluded_task_ids"]), 10)
        self.assertEqual(summary["pilot"]["tasks"], 24)
        self.assertEqual(summary["pilot"]["retail"], 18)
        self.assertEqual(summary["pilot"]["airline"], 6)
        self.assertEqual(summary["formal"]["minimum_tasks_with_three_pairs"], 48)
        self.assertEqual(summary["formal"]["minimum_accepted_pairs"], 144)
        self.assertEqual(summary["selection"]["selection_unit"], "candidate_pair")
        self.assertEqual(
            summary["selection"]["directional_screen_arms"],
            [
                "flawless_only",
                "random_stratified",
                "full_proposed",
            ],
        )
        self.assertEqual(
            summary["selection"]["directional_screen_training_seeds"],
            [20260722],
        )
        self.assertEqual(
            summary["selection"]["directional_screen_evaluation_seeds"],
            [20260722, 20260723, 20260724],
        )
        self.assertFalse(summary["official_test_used"])
        self.assertTrue(summary["official_test_sealed"])

    def test_pilot_registry_is_structural_domain_hash_selection(self):
        expected_retail = [
            "retail:16", "retail:98", "retail:47", "retail:31",
            "retail:92", "retail:99", "retail:11", "retail:104",
            "retail:107", "retail:37", "retail:21", "retail:19",
            "retail:35", "retail:30", "retail:1", "retail:87",
            "retail:72", "retail:7",
        ]
        expected_airline = [
            "airline:21", "airline:40", "airline:33",
            "airline:12", "airline:4", "airline:14",
        ]
        self.assertEqual(
            [x for x in protocol.PILOT_TASK_IDS if x.startswith("retail:")],
            expected_retail,
        )
        self.assertEqual(
            [x for x in protocol.PILOT_TASK_IDS if x.startswith("airline:")],
            expected_airline,
        )
        self.assertEqual(
            protocol.sha256(list(protocol.PILOT_TASK_IDS)),
            protocol.PILOT_REGISTRY_SHA256,
        )
        self.assertFalse(
            set(protocol.PILOT_TASK_IDS)
            & set(protocol.ACTION_IDENTIFIABILITY_EXCLUSIONS)
        )

    def test_forced_first_kappa_requires_one_common_continuation(self):
        cells = forced_first_cells(0.75)
        self.assertEqual(
            protocol.forced_first_common_continuation_kappa(cells), 0.75
        )
        mismatch = deepcopy(cells)
        mismatch["q_e2_a1"]["continuation_policy_sha256"] = "other"
        with self.assertRaisesRegex(protocol.V6ProtocolError, "one continuation"):
            protocol.forced_first_common_continuation_kappa(mismatch)
        leaked = deepcopy(cells)
        leaked["q_e1_a1"]["gold_suffix_visible"] = True
        with self.assertRaisesRegex(
            protocol.V6ProtocolError, "gold recovery suffix"
        ):
            protocol.forced_first_common_continuation_kappa(leaked)

    def test_structural_audit_allows_read_write_but_not_fake_family(self):
        candidate = valid_candidate_pair("retail:1", 0)
        structural = protocol.structural_eligibility(candidate)
        self.assertTrue(structural["eligible"])
        self.assertEqual(structural["reason_codes"], ["ELIGIBLE"])
        self.assertTrue(protocol.candidate_pair_is_accepted(candidate))

        fake = deepcopy(candidate)
        fake["branches"][1]["identifier_key"] = "zip"
        result = protocol.structural_eligibility(fake)
        self.assertFalse(result["eligible"])
        self.assertIn(
            "FEWER_THAN_TWO_SAFE_ON_ERROR_ACTION_FAMILIES",
            result["reason_codes"],
        )

        family_only = valid_candidate_pair("retail:1", 0)
        family_only["branches"][1]["reference_action_key"] = (
            family_only["branches"][0]["reference_action_key"]
        )
        result = protocol.structural_eligibility(family_only)
        self.assertFalse(result["eligible"])
        self.assertIn(
            "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS",
            result["reason_codes"],
        )

        same_family_distinct_calls = valid_candidate_pair("retail:1", 0)
        first = same_family_distinct_calls["branches"][0]
        second = same_family_distinct_calls["branches"][1]
        second["tool_name"] = first["tool_name"]
        second["identifier_key"] = first["identifier_key"]
        second["reference_action_kind"] = first["reference_action_kind"]
        second["corrective_family"] = first["corrective_family"]
        second["reference_arguments"] = {"order_id": "another-valid-order"}
        second["reference_action_key"] = protocol.canonical_reference_action_key(
            second["tool_name"], second["reference_arguments"]
        )
        second["first_recovery_action_key"] = second["reference_action_key"]
        result = protocol.structural_eligibility(same_family_distinct_calls)
        self.assertTrue(result["eligible"])
        self.assertFalse(result["checks"]["two_distinct_corrective_families"])
        self.assertTrue(result["checks"]["two_distinct_full_reference_actions"])

    def test_runtime_audit_recomputes_state_and_failed_label_count(self):
        candidate = valid_candidate_pair("retail:1", 0)
        changed = deepcopy(candidate)
        changed["branches"][0]["agent_db_hash_after_error"] = "changed"
        self.assertFalse(
            protocol.audit_candidate_pair(changed)[
                "failed_injection_agent_and_user_state_unchanged"
            ]
        )
        contaminated = deepcopy(candidate)
        contaminated["quality"]["failed_positive_labels"] = 1
        self.assertFalse(
            protocol.audit_candidate_pair(contaminated)[
                "failed_positive_label_count_zero"
            ]
        )

    def test_hardness_cost_and_coverage_are_frozen(self):
        self.assertEqual(
            protocol.hardness_margin(
                "matched",
                {"matched": -2.0, "wrong_a": -1.0, "wrong_b": -3.0},
            ),
            1.0,
        )
        pair = valid_candidate_pair("retail:1", 0)
        self.assertEqual(protocol.supervised_token_cost([pair]), 200)
        self.assertEqual(protocol.nonpadding_token_cost([pair]), 1000)
        self.assertGreater(
            protocol.coverage_marginal_gain([], pair),
            protocol.coverage_marginal_gain([pair], pair),
        )

    def test_identifiability_requires_high_low_and_teacher_switching(self):
        result = protocol.identifiability_gate(
            [1.0, 0.8, 0.5, 0.2, 0.0],
            teacher_action_switch_accuracy=0.80,
            error_blind_action_switch_accuracy=0.60,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["kappa_bucket_fraction"]["high"], 0.4)
        self.assertAlmostEqual(result["kappa_bucket_fraction"]["low"], 0.4)
        result = protocol.identifiability_gate(
            [1.0, 0.8, 0.5, 0.2, 0.0],
            teacher_action_switch_accuracy=0.79,
            error_blind_action_switch_accuracy=0.50,
        )
        self.assertEqual(result["status"], "FAIL_CLOSED")

    def test_pilot_go_requires_12_tasks_and_48_candidate_pairs(self):
        tasks = list(protocol.PILOT_TASK_IDS[:12])
        choice_sets = [valid_choice_set(task, 4) for task in tasks]
        result = protocol.pilot_gate(
            choice_sets,
            registered_task_ids=protocol.PILOT_TASK_IDS,
            teacher_action_switch_accuracy=0.80,
            error_blind_action_switch_accuracy=0.60,
        )
        self.assertEqual(result["status"], "GO_FORMAL_POOL")
        self.assertEqual(result["pool"]["tasks_with_at_least_three_pairs"], 12)
        self.assertEqual(result["pool"]["accepted_candidate_pairs"], 48)

    @staticmethod
    def pool(task_count: int):
        rows = []
        for index in range(task_count):
            domain = "airline" if index < 6 else "retail"
            rows.append(valid_choice_set(f"{domain}:{1000 + index}", 3))
        return rows

    def test_formal_48_144_screen_40_to_47_and_stop_below_40(self):
        common = {
            "teacher_action_switch_accuracy": 0.80,
            "error_blind_action_switch_accuracy": 0.60,
        }
        result = protocol.formal_pool_gate(self.pool(48), **common)
        self.assertEqual(result["status"], "FORMAL_SELECTION_AUTHORIZED")
        self.assertEqual(
            result["pool"]["tasks_with_at_least_three_pairs"], 48
        )
        self.assertEqual(result["pool"]["pairs_from_qualifying_tasks"], 144)
        self.assertEqual(
            protocol.formal_pool_gate(self.pool(43), **common)["status"],
            "SCREEN_ONLY",
        )
        self.assertEqual(
            protocol.formal_pool_gate(self.pool(39), **common)["status"],
            "STOP_INSUFFICIENT_POOL",
        )

    def test_matched_selector_and_token_budget_stop_are_fail_closed(self):
        tasks = ["airline:1000", "retail:1001"]
        random_rows = [valid_candidate_pair(task, 0) for task in tasks]
        causal_rows = [valid_candidate_pair(task, 1) for task in tasks]
        result = protocol.audit_matched_task_selection(
            {"random": random_rows, "full_proposed": causal_rows},
            expected_task_ids=tasks,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(
            protocol.selection_should_stop(
                selected_cost=700,
                token_budget=1000,
                remaining_candidate_pair_costs=[301, 250],
            )
        )
        self.assertTrue(
            protocol.selection_should_stop(
                selected_cost=751,
                token_budget=1000,
                remaining_candidate_pair_costs=[301, 250],
            )
        )
        self.assertEqual(
            protocol.audit_arm_token_budgets(
                {"random": 9950, "full_proposed": 9990},
                token_budget=10_000,
            )["status"],
            "PASS",
        )
        self.assertEqual(
            protocol.audit_arm_token_budgets(
                {"random": 9900, "full_proposed": 9990},
                token_budget=10_000,
            )["status"],
            "FAIL_CLOSED",
        )

    def test_yaml_contains_same_frozen_boundary_values(self):
        text = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v6_causal_recovery_selection.yaml"
        ).read_text(encoding="utf-8")
        for value in (
            protocol.PROTOCOL,
            protocol.STRUCTURAL_PROTOCOL,
            protocol.STRUCTURAL_ELIGIBILITY_SHA256,
            protocol.ACTION_IDENTIFIABLE_SHA256,
            protocol.PILOT_REGISTRY_SHA256,
            "selection_unit: candidate_pair",
            "minimum_pairs_from_qualifying_tasks: 144",
            "sealed: true",
        ):
            self.assertIn(str(value), text)


if __name__ == "__main__":
    unittest.main()
