import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v5_stage0.py"
SPEC = importlib.util.spec_from_file_location("prepare_v5_stage0", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_v5_stage0.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_v5_stage0", SUMMARY_SCRIPT
)
SUMMARY_MODULE = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)
RUN_SCRIPT = ROOT / "scripts" / "run_v5_stage0.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_v5_stage0", RUN_SCRIPT)
RUN_MODULE = importlib.util.module_from_spec(RUN_SPEC)
assert RUN_SPEC.loader is not None
RUN_SPEC.loader.exec_module(RUN_MODULE)


class V5Stage0ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tau2_root = Path(
            os.environ.get("TAU2_ROOT", ROOT / "data" / "raw" / "tau2-bench")
        )
        if not cls.tau2_root.exists():
            raise unittest.SkipTest("Frozen tau2-bench checkout is unavailable")
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tempdir.name)
        cls.audit = MODULE.prepare(cls.tau2_root, cls.output)
        cls.split = json.loads(
            (cls.output / "split_manifest.json").read_text(encoding="utf-8")
        )
        cls.smoke = json.loads(
            (cls.output / "smoke_manifest.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_official_and_derived_counts(self):
        self.assertEqual(self.audit["official_train_total"], 104)
        self.assertEqual(self.audit["inner_train_total"], 83)
        self.assertEqual(self.audit["validation_total"], 21)
        self.assertEqual(self.audit["sealed_test_total"], 60)

    def test_all_splits_are_disjoint(self):
        for domain in ("retail", "airline"):
            payload = self.split["domains"][domain]
            inner = set(payload["inner_train_ids"])
            validation = set(payload["validation_ids"])
            test = set(payload["sealed_test_ids"])
            self.assertFalse(inner & validation)
            self.assertFalse(inner & test)
            self.assertFalse(validation & test)

    def test_smoke_is_paired_and_validation_only(self):
        self.assertEqual(self.smoke["paired_task_count"], 10)
        self.assertEqual(self.smoke["total_end_to_end_runs"], 20)
        pair_ids = [row["pair_id"] for row in self.smoke["rows"]]
        self.assertEqual(len(pair_ids), len(set(pair_ids)))
        for row in self.smoke["rows"]:
            validation = set(
                self.split["domains"][row["domain"]]["validation_ids"]
            )
            self.assertIn(row["task_id"], validation)
            self.assertFalse(row["clean_condition"]["inject_error"])
            self.assertTrue(row["error_condition"]["inject_error"])

    def test_injections_are_read_only_nonexistent_identifiers(self):
        allowed = set(MODULE.SAFE_INJECTIONS)
        for row in self.smoke["rows"]:
            injection = row["error_condition"]
            self.assertIn(injection["tool_name"], allowed)
            self.assertEqual(
                injection["arguments"],
                MODULE.SAFE_INJECTIONS[injection["tool_name"]],
            )
            self.assertTrue(injection["expected_tool_error"])
            self.assertFalse(injection["expected_state_mutation"])

    def test_single_pair_filter_is_exact(self):
        pair_id = self.smoke["rows"][0]["pair_id"]
        rows = RUN_MODULE.filter_manifest_rows(self.smoke, pair_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair_id"], pair_id)
        with self.assertRaises(RuntimeError):
            RUN_MODULE.filter_manifest_rows(self.smoke, "missing-pair")

    def test_preparation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as second_dir:
            second = Path(second_dir)
            second_audit = MODULE.prepare(self.tau2_root, second)
            self.assertEqual(
                self.audit["split_manifest_sha256"],
                second_audit["split_manifest_sha256"],
            )
            self.assertEqual(
                self.audit["smoke_manifest_sha256"],
                second_audit["smoke_manifest_sha256"],
            )

    def test_summary_requires_real_paired_injected_errors(self):
        with tempfile.TemporaryDirectory() as results_temp:
            results_dir = Path(results_temp)
            rows_by_domain = {"retail": [], "airline": []}
            for row in self.smoke["rows"]:
                rows_by_domain[row["domain"]].append(row)
            for domain, rows in rows_by_domain.items():
                clean_simulations = []
                error_simulations = []
                for row in rows:
                    task_id = row["task_id"]
                    injection = row["error_condition"]
                    common = {
                        "task_id": task_id,
                        "termination_reason": "agent_stop",
                        "duration": 1.0,
                        "reward_info": {"reward": 1.0},
                    }
                    clean_simulations.append(
                        {**common, "messages": [{"role": "user", "content": "test"}]}
                    )
                    error_simulations.append(
                        {
                            **common,
                            "messages": [
                                {"role": "user", "content": "test"},
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": injection["tool_call_id"],
                                            "name": injection["tool_name"],
                                            "arguments": injection["arguments"],
                                        }
                                    ],
                                },
                                {
                                    "role": "tool",
                                    "id": injection["tool_call_id"],
                                    "error": True,
                                    "content": "not found",
                                },
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": f"recovery-{domain}-{task_id}",
                                            "name": injection["tool_name"],
                                            "arguments": {"corrected": "identifier"},
                                        }
                                    ],
                                },
                                {
                                    "role": "tool",
                                    "id": f"recovery-{domain}-{task_id}",
                                    "error": False,
                                    "content": "ok",
                                },
                            ],
                        }
                    )
                (results_dir / f"{domain}_clean.json").write_text(
                    json.dumps({"simulations": clean_simulations}),
                    encoding="utf-8",
                )
                (results_dir / f"{domain}_error.json").write_text(
                    json.dumps({"simulations": error_simulations}),
                    encoding="utf-8",
                )
            summary = SUMMARY_MODULE.summarize(
                self.output / "smoke_manifest.json",
                results_dir,
                results_dir / "summary.json",
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["total_end_to_end_runs"], 20)
            self.assertEqual(summary["injected_error_observed_rate"], 1.0)
            self.assertEqual(summary["repeated_identical_error_rate"], 0.0)
            self.assertEqual(summary["valid_post_error_tool_result_rate"], 1.0)
            self.assertEqual(summary["token_usage"]["clean_prompt_tokens"], 0)
            self.assertEqual(
                summary["termination_reasons"]["clean"]["agent_stop"], 10
            )


if __name__ == "__main__":
    unittest.main()
