from __future__ import annotations

import unittest

from scripts import build_v5_4_branch_manifest as branch
from scripts import v5_4_pilot_protocol as protocol


def clean_record() -> dict:
    return {
        "task_identity": "retail:2",
        "seed": protocol.TRIAL_SEEDS[0],
        "eligible_clean_success": True,
        "messages": [
            {"role": "user", "content": "Please check order A."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "clean-1", "name": "get_order_details", "arguments": {"order_id": "A"}}
                ],
            },
            {"role": "tool", "tool_call_id": "clean-1", "content": {"status": "pending"}},
            {"role": "assistant", "content": "Your order is pending."},
        ],
    }


def perturbation() -> dict:
    return {
        "taxonomy": "wrong_identifier_same_type",
        "clean_tool_name": "get_order_details",
        "schema_valid": True,
        "clean_action_correct": True,
        "injected_action_task_incorrect": True,
        "consequential": True,
        "injected_error_count": 1,
        "injected_call": {
            "id": "error-1",
            "name": "get_order_details",
            "arguments": {"order_id": "B"},
        },
        "actual_error_result": {
            "role": "tool",
            "tool_call_id": "error-1",
            "content": {"error": "order not found"},
        },
    }


class BranchManifestTests(unittest.TestCase):
    def test_clean_future_is_absent_and_hash_bound(self) -> None:
        result = branch.build_branch(
            clean_record(),
            perturbation(),
            recovery_seed=protocol.TRIAL_SEEDS[8],
        )
        self.assertEqual(len(result["messages"]), 3)
        self.assertNotIn("Your order is pending.", str(result["messages"]))
        evidence = result["evidence"]
        self.assertTrue(all(protocol.leakage_checks(evidence).values()))
        self.assertFalse(evidence["failed_call_supervised"])
        self.assertEqual(
            result["supervision_starts_after_message_index"], 2
        )

    def test_nonconsequential_error_is_rejected(self) -> None:
        bad = perturbation()
        bad["consequential"] = False
        with self.assertRaises(protocol.PilotProtocolError):
            branch.build_branch(
                clean_record(),
                bad,
                recovery_seed=protocol.TRIAL_SEEDS[8],
            )


if __name__ == "__main__":
    unittest.main()
