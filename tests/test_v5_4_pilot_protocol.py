from __future__ import annotations

from copy import deepcopy
import unittest

from scripts import v5_4_pilot_protocol as protocol


def terminal_slots(pair_tasks: int = 0) -> list[dict]:
    rows = protocol.slot_registry()
    paired = set(protocol.PILOT_TASK_IDS[:pair_tasks])
    double = set(protocol.PILOT_TASK_IDS[: min(pair_tasks, 3)])
    for row in rows:
        if row["slot_kind"] == "clean":
            row["status"] = (
                "CLEAN_ELIGIBLE"
                if row["slot_index"] == 1
                else "CLEAN_SUCCESS_EARLY_STOP"
            )
        else:
            if row["task_identity"] in paired and row["slot_index"] == 9:
                row["status"] = "PAIR_ELIGIBLE"
            elif row["task_identity"] in double and row["slot_index"] == 10:
                row["status"] = "PAIR_ELIGIBLE"
            elif row["task_identity"] in double:
                row["status"] = "PAIR_CAP_REACHED"
            else:
                row["status"] = "RECOVERY_INELIGIBLE"
    return rows


def pair(task_identity: str, suffix: str, branch: int = 9) -> dict:
    digest = (suffix * 64)[:64]
    return {
        "task_identity": task_identity,
        "slot_id": f"{task_identity}:recovery:{branch:02d}",
        "eligible": True,
        "schema_valid": True,
        "clean_action_correct": True,
        "injected_action_task_incorrect": True,
        "consequential": True,
        "taxonomy": "wrong_identifier_same_type",
        "clean_tool_name": "get_order_details",
        "injected_error_count": 1,
        "shared_prefix_sha256": digest,
        "recovery_prefix_sha256": digest,
        "clean_future_sha256": "b" * 64,
        "recovery_prompt_sha256": "c" * 64,
        "error_event_sha256": "d" * 64,
        "clean_future_present_in_prompt": False,
        "failed_call_supervised": False,
        "error_result_supervised": False,
    }


class PilotProtocolTests(unittest.TestCase):
    def test_budget_seed_and_registry(self) -> None:
        protocol.validate_constants()
        rows = protocol.slot_registry()
        self.assertEqual(len(rows), 288)
        self.assertEqual(
            sum(row["slot_kind"] == "clean" for row in rows), 192
        )
        self.assertEqual(
            sum(row["slot_kind"] == "recovery" for row in rows), 96
        )

    def test_injection_selection_is_order_independent(self) -> None:
        left = protocol.select_injection_index(
            [9, 2, 5],
            task_identity="retail:2",
            clean_seed=protocol.TRIAL_SEEDS[0],
        )
        right = protocol.select_injection_index(
            [5, 9, 2],
            task_identity="retail:2",
            clean_seed=protocol.TRIAL_SEEDS[0],
        )
        self.assertEqual(left, right)
        self.assertIn(left, {2, 5, 9})

    def test_real_error_is_strict(self) -> None:
        value = pair("retail:2", "a")
        self.assertTrue(all(protocol.real_error_checks(value).values()))
        value["consequential"] = False
        self.assertFalse(protocol.real_error_checks(value)["consequential"])
        value["consequential"] = True
        value["clean_tool_name"] = "cancel_order"
        self.assertFalse(protocol.real_error_checks(value)["read_only_injection"])

    def test_gate_passes_only_quality_adjusted_pairs(self) -> None:
        pairs = []
        for index, task in enumerate(protocol.PILOT_TASK_IDS[:14]):
            pairs.append(pair(task, chr(97 + index % 20), 9))
        for index, task in enumerate(protocol.PILOT_TASK_IDS[:3]):
            pairs.append(pair(task, chr(70 + index), 10))
        result = protocol.pilot_decision(terminal_slots(14), pairs)
        self.assertEqual(result["status"], "GO_FULL_V5_4")
        self.assertEqual(result["observed"]["tasks_with_pair"], 14)
        self.assertEqual(result["observed"]["capped_pairs"], 17)
        self.assertFalse(result["training_authorized"])

        broken = deepcopy(pairs)
        broken[0]["failed_call_supervised"] = True
        result = protocol.pilot_decision(terminal_slots(14), broken)
        self.assertEqual(result["status"], "NO_GO_STOP")
        self.assertFalse(result["checks"]["all_claimed_pairs_audit_valid"])

    def test_missing_slot_fails_closed(self) -> None:
        with self.assertRaises(protocol.PilotProtocolError):
            protocol.pilot_decision(terminal_slots()[:-1], [])


if __name__ == "__main__":
    unittest.main()
