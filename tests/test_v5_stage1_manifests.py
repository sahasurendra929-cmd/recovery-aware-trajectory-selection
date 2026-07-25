import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v5_stage1_manifests.py"
SPEC = importlib.util.spec_from_file_location("prepare_v5_stage1_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
GENERATION_SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_generate.py"
GENERATION_SPEC = importlib.util.spec_from_file_location(
    "run_v5_sft_causal_generate_for_manifest_test", GENERATION_SCRIPT
)
GENERATION_MODULE = importlib.util.module_from_spec(GENERATION_SPEC)
assert GENERATION_SPEC.loader is not None
GENERATION_SPEC.loader.exec_module(GENERATION_MODULE)
EVALUATION_SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_eval.py"
EVALUATION_SPEC = importlib.util.spec_from_file_location(
    "run_v5_sft_causal_eval_for_manifest_test", EVALUATION_SCRIPT
)
EVALUATION_MODULE = importlib.util.module_from_spec(EVALUATION_SPEC)
assert EVALUATION_SPEC.loader is not None
EVALUATION_SPEC.loader.exec_module(EVALUATION_MODULE)


class V5Stage1ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tau2_root = ROOT / "data" / "raw" / "tau2-bench"
        cls.split_path = (
            ROOT / "data" / "processed" / "v5_stage0" / "split_manifest.json"
        )
        if not cls.tau2_root.is_dir() or not cls.split_path.is_file():
            raise unittest.SkipTest(
                "pinned tau2 checkout or historical Stage-0 split is unavailable"
            )

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def build(self, name):
        output = self.root / name
        before = self.split_path.read_bytes()
        audit = MODULE.prepare(
            tau2_root=self.tau2_root,
            split_manifest_path=self.split_path,
            output_dir=output,
            seed=MODULE.SEED,
        )
        self.assertEqual(self.split_path.read_bytes(), before)
        return output, audit

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_builds_complete_safe_train_and_validation_protocol(self):
        output, audit = self.build("first")
        generation = self.read(output / "generation_manifest.json")
        validation = self.read(output / "validation_manifest.json")
        split = self.read(self.split_path)

        self.assertEqual(generation["paired_task_count"], 83)
        self.assertEqual(validation["paired_task_count"], 21)
        self.assertEqual(generation["domain_counts"], {"retail": 59, "airline": 24})
        self.assertEqual(validation["domain_counts"], {"retail": 15, "airline": 6})
        self.assertEqual(generation["protocol"], MODULE.GENERATION_PROTOCOL)
        self.assertEqual(validation["protocol"], MODULE.VALIDATION_PROTOCOL)
        self.assertEqual(
            generation["fault_protocol"]["protocol"],
            MODULE.FAULT_PROTOCOL,
        )
        self.assertEqual(
            generation["fault_protocol"]["claim_scope"],
            "multi_fault_family_post_fault_robustness_screen",
        )
        self.assertEqual(
            generation["fault_protocol"]["family_level_inference"],
            "descriptive_only",
        )
        self.assertFalse(generation["official_test_used"])
        self.assertFalse(validation["official_test_used"])
        self.assertTrue(generation["official_test_sealed"])
        self.assertTrue(validation["official_test_sealed"])
        self.assertEqual(
            generation["split_manifest_sha256"],
            MODULE.sha256_file(self.split_path),
        )
        self.assertEqual(generation["tau2_commit"], MODULE.TAU2_COMMIT)
        self.assertEqual(generation["source_files"], validation["source_files"])

        all_rows = generation["rows"] + validation["rows"]
        invalid_parameters = set()
        call_ids = set()
        for manifest, split_name in (
            (generation, "inner_train_ids"),
            (validation, "validation_ids"),
        ):
            observed = {
                domain: {
                    row["task_id"]
                    for row in manifest["rows"]
                    if row["domain"] == domain
                }
                for domain in MODULE.DOMAINS
            }
            for domain in MODULE.DOMAINS:
                self.assertEqual(
                    observed[domain],
                    set(split["domains"][domain][split_name]),
                )
                self.assertFalse(
                    observed[domain]
                    & set(split["domains"][domain]["sealed_test_ids"])
                )
                domain_rows = [
                    row for row in manifest["rows"] if row["domain"] == domain
                ]
                self.assertGreaterEqual(
                    len(
                        {
                            row["error_condition"]["fault_family"]
                            for row in domain_rows
                        }
                    ),
                    2,
                )
                self.assertGreaterEqual(
                    len(
                        {
                            row["error_condition"]["tool_name"]
                            for row in domain_rows
                        }
                    ),
                    2,
                )

        for row in all_rows:
            error = row["error_condition"]
            self.assertEqual(error["tool_type"], "READ")
            self.assertTrue(error["expected_tool_error"])
            self.assertFalse(error["expected_state_mutation"])
            self.assertIs(type(error["on_reference_path"]), bool)
            self.assertIn(
                error["fault_relevance"],
                {
                    "reference_path_or_operation_aligned",
                    "domain_plausible_fallback",
                },
            )
            identity = MODULE.canonical(
                {
                    error["invalid_argument_key"]: error["arguments"][
                        error["invalid_argument_key"]
                    ]
                }
            )
            self.assertNotIn(identity, invalid_parameters)
            invalid_parameters.add(identity)
            self.assertNotIn(error["tool_call_id"], call_ids)
            call_ids.add(error["tool_call_id"])

        self.assertEqual(len(invalid_parameters), 104)
        self.assertEqual(len(call_ids), 104)
        self.assertTrue(audit["invalid_parameters_unique"])
        self.assertEqual(audit["invalid_parameter_count"], 104)

    def test_source_hashes_match_local_pinned_tau2_and_fail_closed(self):
        output, _ = self.build("sources")
        generation = self.read(output / "generation_manifest.json")
        validation = self.read(output / "validation_manifest.json")
        GENERATION_MODULE.validate_local_manifest_sources(
            generation,
            self.tau2_root,
        )
        EVALUATION_MODULE.validate_local_manifest_sources(
            validation,
            self.tau2_root,
        )
        generation["source_files"]["retail"]["db_json_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "db_json_sha256"):
            GENERATION_MODULE.validate_local_manifest_sources(
                generation,
                self.tau2_root,
            )
        validation["source_files"]["airline"]["db_json_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "db_json_sha256"):
            EVALUATION_MODULE.validate_local_manifest_sources(
                validation,
                self.tau2_root,
            )

    def test_outputs_are_byte_deterministic(self):
        first, _ = self.build("deterministic-a")
        second, _ = self.build("deterministic-b")
        for filename in (
            "generation_manifest.json",
            "validation_manifest.json",
            "audit.json",
            "hashes.json",
        ):
            self.assertEqual(
                (first / filename).read_bytes(),
                (second / filename).read_bytes(),
                filename,
            )


if __name__ == "__main__":
    unittest.main()
