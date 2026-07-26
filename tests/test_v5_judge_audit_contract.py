import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import v5_judge_audit_contract as module


class StrictJudgeEvidenceContractTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        with_assertion: bool = True,
        audit_outcome: str = "expected outcome",
        audit_count: int = 1,
    ) -> tuple[Path, list[Path]]:
        reward_info = {"reward": 1.0}
        if with_assertion:
            reward_info["nl_assertions"] = [
                {
                    "nl_assertion": "expected outcome",
                    "met": True,
                    "justification": "fixture",
                }
            ]
        result = root / "retail_clean.json"
        result.write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "task_id": "7",
                            "id": "fixture-simulation",
                            "trial": 0,
                            "seed": 123,
                            "reward_info": reward_info,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        audit_dir = (
            root
            / "logs"
            / result.stem
            / "artifacts"
            / "task_7"
            / "sim_fixture-simulation"
            / "llm_debug"
        )
        audits: list[Path] = []
        for index in range(audit_count):
            audit_dir.mkdir(parents=True, exist_ok=True)
            raw_content = json.dumps(
                [
                    {
                        "nl_assertion": audit_outcome,
                        "met": True,
                        "justification": "fixture",
                    }
                ],
                separators=(",", ":"),
            )
            audit = (
                audit_dir
                / f"strict_nl_judge_audit_fixture_{index}.json"
            )
            audit.write_text(
                json.dumps(
                    {
                        "protocol": module.STRICT_JUDGE_PROTOCOL,
                        "status": "PASS",
                        "expected_outcomes": [audit_outcome],
                        "attempt_count": 1,
                        "attempts": [
                            {
                                "attempt": 1,
                                "call_name": (
                                    "nl_assertions_eval_strict_attempt_1"
                                ),
                                "raw_content": raw_content,
                                "raw_content_utf8_bytes": len(
                                    raw_content.encode("utf-8")
                                ),
                                "raw_content_sha256": hashlib.sha256(
                                    raw_content.encode("utf-8")
                                ).hexdigest(),
                                "raw_provider_response": {
                                    "fixture": True
                                },
                                "schema_status": "PASS",
                                "schema_error": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audits.append(audit)
        return result, audits

    def test_exact_call_mapping_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            result, audits = self._fixture(Path(directory))
            evidence = module.validate_strict_judge_evidence(
                [result],
                audit_paths=audits,
            )
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["expected_calls"], 1)
        self.assertEqual(evidence["observed_unique_pass_audits"], 1)
        self.assertRegex(
            evidence["canonical_mapping_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            evidence["calls"][0]["expected_outcomes"],
            ["expected outcome"],
        )

    def test_missing_expected_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._fixture(Path(directory), audit_count=0)
            with self.assertRaisesRegex(
                module.StrictJudgeEvidenceError,
                "expected exactly one",
            ):
                module.validate_strict_judge_evidence([result])

    def test_duplicate_audits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._fixture(Path(directory), audit_count=2)
            with self.assertRaisesRegex(
                module.StrictJudgeEvidenceError,
                "expected exactly one",
            ):
                module.validate_strict_judge_evidence([result])

    def test_extra_audit_without_expected_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, audits = self._fixture(
                Path(directory),
                with_assertion=False,
            )
            with self.assertRaisesRegex(
                module.StrictJudgeEvidenceError,
                "without a matching",
            ):
                module.validate_strict_judge_evidence(
                    [result],
                    audit_paths=audits,
                )

    def test_assertion_outcome_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._fixture(
                Path(directory),
                audit_outcome="wrong outcome",
            )
            with self.assertRaisesRegex(
                module.StrictJudgeEvidenceError,
                "outcomes drift",
            ):
                module.validate_strict_judge_evidence([result])

    def test_duplicate_supplied_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, audits = self._fixture(Path(directory))
            with self.assertRaisesRegex(
                module.StrictJudgeEvidenceError,
                "duplicate strict-judge audit path",
            ):
                module.validate_strict_judge_evidence(
                    [result],
                    audit_paths=[audits[0], audits[0]],
                )


if __name__ == "__main__":
    unittest.main()
