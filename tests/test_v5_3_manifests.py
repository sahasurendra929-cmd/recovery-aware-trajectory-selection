import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v5_3_manifests.py"
SPEC = importlib.util.spec_from_file_location("prepare_v5_3_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class V53ManifestTests(unittest.TestCase):
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
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def build(self, name):
        output = self.root / name
        split_before = self.split_path.read_bytes()
        audit = MODULE.prepare(
            tau2_root=self.tau2_root,
            split_manifest_path=self.split_path,
            output_dir=output,
            seed=MODULE.stage1.SEED,
        )
        self.assertEqual(self.split_path.read_bytes(), split_before)
        return output, audit

    def test_builds_frozen_78_task_generation_universe_and_keeps_validation(self):
        output, audit = self.build("protocol")
        generation = self.read(output / "generation_manifest.json")
        validation = self.read(output / "validation_manifest.json")
        split = self.read(self.split_path)
        expected_filter = MODULE._filter_contract()

        self.assertEqual(generation["protocol"], MODULE.PROTOCOL)
        self.assertEqual(generation["paired_task_count"], 78)
        self.assertEqual(
            generation["domain_counts"],
            {"retail": 58, "airline": 20},
        )
        self.assertEqual(
            generation["gt_compatibility_filter"],
            expected_filter,
        )
        self.assertFalse(
            expected_filter[
                "selection_uses_rollouts_rewards_validation_or_test"
            ]
        )
        self.assertFalse(expected_filter["official_test_used"])
        self.assertEqual(
            validation["protocol"],
            MODULE.stage1.VALIDATION_PROTOCOL,
        )
        self.assertEqual(validation["paired_task_count"], 21)

        included = {
            f"{row['domain']}:{row['task_id']}"
            for row in generation["rows"]
        }
        excluded = set(MODULE.FROZEN_EXCLUDED_TASK_IDS)
        historical_inner = {
            f"{domain}:{task_id}"
            for domain in MODULE.stage1.DOMAINS
            for task_id in split["domains"][domain]["inner_train_ids"]
        }
        self.assertFalse(included & excluded)
        self.assertEqual(included | excluded, historical_inner)
        self.assertEqual(len(included), 78)
        self.assertEqual(len(excluded), 5)

        universe = audit["v5_3_generation_universe"]
        self.assertEqual(universe["protocol"], MODULE.PROTOCOL)
        self.assertEqual(universe["source_tasks"], 83)
        self.assertEqual(universe["included_tasks"], 78)
        self.assertEqual(universe["excluded_tasks"], 5)
        self.assertEqual(
            universe["gt_compatibility_filter"],
            expected_filter,
        )

    def test_hashes_bind_rewritten_generation_validation_and_audit(self):
        output, audit = self.build("hashes")
        hashes = self.read(output / "hashes.json")
        for filename in (
            "generation_manifest.json",
            "validation_manifest.json",
            "audit.json",
        ):
            self.assertEqual(
                hashes[filename],
                MODULE.stage1.sha256_file(output / filename),
            )
        self.assertEqual(
            audit["generation_manifest_sha256"],
            hashes["generation_manifest.json"],
        )

    def test_pinned_tau2_structural_filter_matches_frozen_five(self):
        self.assertEqual(
            MODULE._ground_truth_incompatible_tasks(
                self.tau2_root,
                self.read(self.split_path),
            ),
            list(MODULE.FROZEN_EXCLUDED_TASK_IDS),
        )

    def test_gt_incompatible_set_drift_fails_closed(self):
        with mock.patch.object(
            MODULE,
            "_ground_truth_incompatible_tasks",
            return_value=["retail:24"],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "GT-incompatible task set drift",
            ):
                MODULE.prepare(
                    tau2_root=self.tau2_root,
                    split_manifest_path=self.split_path,
                    output_dir=self.root / "drift",
                    seed=MODULE.stage1.SEED,
                )

    def test_outputs_are_byte_deterministic(self):
        first, _ = self.build("first")
        second, _ = self.build("second")
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
