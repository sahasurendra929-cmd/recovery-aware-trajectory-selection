from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from scripts import v5_3_12h_protocol as protocol


class ScreenProtocolTests(unittest.TestCase):
    def test_frozen_counts_and_seed_isolation(self) -> None:
        protocol.validate_constants()
        self.assertEqual(protocol.EXPECTED_ROLLOUTS, 288)
        self.assertEqual(protocol.CORE_EVALUATION_ROLLOUTS, 126)
        self.assertEqual(len(protocol.PILOT_TASK_IDS), 24)
        self.assertFalse(
            set(protocol.TRIAL_SEEDS) & set(protocol.PILOT_TRIAL_SEEDS)
        )
        self.assertFalse(
            set(protocol.TRIAL_SEEDS) & set(protocol.FORMAL_TRIAL_SEEDS)
        )

    def test_data_gate_is_exact_and_capped(self) -> None:
        counts = {task: 0 for task in protocol.PILOT_TASK_IDS}
        for task in protocol.PILOT_TASK_IDS[:14]:
            counts[task] = 1
        for task in protocol.PILOT_TASK_IDS[:3]:
            counts[task] = 99
        result = protocol.data_gate(counts)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["observed"]["tasks_with_pair"], 14)
        self.assertEqual(result["observed"]["capped_pairs"], 17)

        counts[protocol.PILOT_TASK_IDS[13]] = 0
        result = protocol.data_gate(counts)
        self.assertEqual(result["status"], "FAIL_CLOSED")
        self.assertFalse(result["training_authorized"])

    def test_hour_nine_gate_never_uses_metrics(self) -> None:
        start = datetime(2026, 7, 26, tzinfo=timezone.utc)
        receipt = protocol.make_deadline_receipt(start)
        before = protocol.extension_admission(
            receipt,
            core_completed_at=start + timedelta(hours=8, minutes=59),
            core_complete=True,
        )
        after = protocol.extension_admission(
            receipt,
            core_completed_at=start + timedelta(hours=9, seconds=1),
            core_complete=True,
        )
        self.assertTrue(before["extension_authorized"])
        self.assertFalse(after["extension_authorized"])
        self.assertFalse(before["uses_metric_values"])
        self.assertFalse(after["uses_metric_values"])
        boundary = protocol.extension_admission(
            receipt,
            core_completed_at=start + timedelta(hours=9),
            core_complete=True,
        )
        after_boundary = protocol.extension_admission(
            receipt,
            core_completed_at=start
            + timedelta(hours=9, microseconds=1),
            core_complete=True,
        )
        self.assertTrue(boundary["extension_authorized"])
        self.assertFalse(after_boundary["extension_authorized"])
        with self.assertRaises(protocol.ScreenProtocolError):
            protocol.extension_admission(
                receipt,
                core_completed_at=datetime(2026, 7, 26),
                core_complete=True,
            )

    def test_deadline_receipt_schema_and_derivation_are_exact(self) -> None:
        start = datetime(2026, 7, 26, tzinfo=timezone.utc)
        receipt = protocol.make_deadline_receipt(start)
        self.assertEqual(protocol.validate_deadline_receipt(receipt), receipt)
        tampered = dict(receipt)
        tampered["extension_admission_deadline_utc"] = (
            start + timedelta(hours=10)
        ).isoformat()
        body = {
            key: value
            for key, value in tampered.items()
            if key != "canonical_sha256"
        }
        tampered["canonical_sha256"] = protocol.canonical_sha256(body)
        with self.assertRaises(protocol.ScreenProtocolError):
            protocol.validate_deadline_receipt(tampered)

    def test_directional_gate_and_claim_boundary(self) -> None:
        positive = protocol.directional_gate(
            perfect_error_successes=5,
            repair_error_successes=7,
            perfect_clean_successes=10,
            repair_clean_successes=9,
        )
        self.assertEqual(
            positive["status"], "DIRECTIONAL_POSITIVE_SCREEN"
        )
        self.assertTrue(
            positive["claim_boundary"]["exploratory_screen_only"]
        )
        self.assertFalse(positive["claim_boundary"]["formal_v5_3_result"])
        self.assertFalse(positive["claim_boundary"]["official_test_used"])

        inconclusive = protocol.directional_gate(
            perfect_error_successes=5,
            repair_error_successes=6,
            perfect_clean_successes=10,
            repair_clean_successes=10,
        )
        self.assertEqual(
            inconclusive["status"], "EXPLORATORY_INCONCLUSIVE"
        )


if __name__ == "__main__":
    unittest.main()
