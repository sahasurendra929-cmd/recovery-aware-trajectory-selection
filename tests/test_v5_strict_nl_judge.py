import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v5_strict_nl_judge.py"
SPEC = importlib.util.spec_from_file_location("v5_strict_nl_judge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


ASSERTIONS = ["The refund was issued.", "The customer was informed."]


def payload(*, first=True, second=False):
    return {
        "results": [
            {
                "expectedOutcome": ASSERTIONS[0],
                "reasoning": "The tool state confirms the refund.",
                "metExpectation": first,
            },
            {
                "expectedOutcome": ASSERTIONS[1],
                "reasoning": "No final confirmation was sent.",
                "metExpectation": second,
            },
        ]
    }


def response(content, *, request_id):
    return SimpleNamespace(
        content=content,
        raw_data={"id": request_id, "provider": "test"},
    )


class StrictNLJudgeTests(unittest.TestCase):
    def test_accepts_bare_and_fenced_json_and_restores_registered_order(self):
        bare = json.dumps(payload())
        rows = MODULE.parse_and_validate_response(bare, ASSERTIONS)
        self.assertEqual(
            [row.expected_outcome for row in rows],
            ASSERTIONS,
        )
        self.assertEqual(
            [row.met_expectation for row in rows],
            [True, False],
        )

        reversed_payload = payload()
        reversed_payload["results"].reverse()
        fenced = f"```JSON\n{json.dumps(reversed_payload)}\n```"
        rows = MODULE.parse_and_validate_response(fenced, ASSERTIONS)
        self.assertEqual(
            [row.expected_outcome for row in rows],
            ASSERTIONS,
        )

    def test_rejects_prose_around_fence_instead_of_salvaging_json(self):
        fenced = f"Here is the result:\n```json\n{json.dumps(payload())}\n```"
        with self.assertRaisesRegex(
            MODULE.StrictNLJudgeResponseError,
            "not valid JSON",
        ):
            MODULE.parse_and_validate_response(fenced, ASSERTIONS)

    def test_empty_partial_duplicate_unknown_and_non_boolean_results_fail(self):
        invalid_payloads = {
            "empty": {"results": []},
            "missing_results": {},
            "partial": {"results": payload()["results"][:1]},
            "duplicate": {
                "results": [
                    payload()["results"][0],
                    {
                        **payload()["results"][0],
                        "reasoning": "A duplicate row.",
                    },
                ]
            },
            "unknown": {
                "results": [
                    payload()["results"][0],
                    {
                        **payload()["results"][1],
                        "expectedOutcome": "An unregistered outcome.",
                    },
                ]
            },
            "string_boolean": {
                "results": [
                    {
                        **payload()["results"][0],
                        "metExpectation": "true",
                    },
                    payload()["results"][1],
                ]
            },
            "integer_boolean": {
                "results": [
                    {
                        **payload()["results"][0],
                        "metExpectation": 1,
                    },
                    payload()["results"][1],
                ]
            },
        }
        for name, candidate in invalid_payloads.items():
            with self.subTest(name=name):
                with self.assertRaises(MODULE.StrictNLJudgeResponseError):
                    MODULE.parse_and_validate_response(
                        json.dumps(candidate),
                        ASSERTIONS,
                    )

    def test_content_schema_failure_retries_once_and_preserves_raw_audit(self):
        calls = []
        audits = []

        def request(attempt, previous_error):
            calls.append((attempt, previous_error))
            if attempt == 1:
                return response('{"results":[]}', request_id="bad")
            return response(
                f"```json\n{json.dumps(payload())}\n```",
                request_id="good",
            )

        rows = MODULE.run_strict_content_retry(
            nl_assertions=ASSERTIONS,
            request=request,
            max_content_attempts=2,
            audit_writer=audits.append,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([call[0] for call in calls], [1, 2])
        self.assertIsNone(calls[0][1])
        self.assertIn("nonempty JSON array", calls[1][1])
        self.assertEqual(len(audits), 1)
        audit = audits[0]
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["attempt_count"], 2)
        self.assertEqual(
            [item["schema_status"] for item in audit["attempts"]],
            ["REJECTED", "PASS"],
        )
        self.assertEqual(
            audit["attempts"][0]["raw_provider_response"]["id"],
            "bad",
        )
        self.assertEqual(audit["attempts"][0]["raw_content"], '{"results":[]}')
        self.assertRegex(
            audit["attempts"][0]["raw_content_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_exhausted_content_retries_raise_and_publish_fail_closed_audit(self):
        audits = []

        with self.assertRaisesRegex(
            MODULE.StrictNLJudgeResponseError,
            "exhausted content retries",
        ) as caught:
            MODULE.run_strict_content_retry(
                nl_assertions=ASSERTIONS,
                request=lambda attempt, previous_error: response(
                    "not-json",
                    request_id=f"bad-{attempt}",
                ),
                max_content_attempts=2,
                audit_writer=audits.append,
            )

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["status"], "FAIL_CLOSED")
        self.assertEqual(audits[0]["attempt_count"], 2)
        self.assertEqual(caught.exception.audit, audits[0])

    def test_provider_error_is_not_mislabeled_as_content_retry(self):
        calls = []

        def request(attempt, previous_error):
            calls.append(attempt)
            raise ConnectionError("provider unavailable")

        with self.assertRaisesRegex(ConnectionError, "provider unavailable"):
            MODULE.run_strict_content_retry(
                nl_assertions=ASSERTIONS,
                request=request,
                max_content_attempts=2,
                audit_writer=lambda value: None,
            )
        self.assertEqual(calls, [1])

    def test_retry_bound_is_small_and_explicit(self):
        for invalid in (0, 4, 1.5, True):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    MODULE.run_strict_content_retry(
                        nl_assertions=ASSERTIONS,
                        request=lambda attempt, previous_error: None,
                        max_content_attempts=invalid,
                    )


if __name__ == "__main__":
    unittest.main()
