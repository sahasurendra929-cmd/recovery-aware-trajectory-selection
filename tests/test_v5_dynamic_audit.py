import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.v5_multifault_fixtures import (
    TAU2_COMMIT,
    multifault_manifest,
    multifault_rows,
    write_dynamic_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = load_script("v5_dynamic_audit_contract.py")
VERIFY = load_script("verify_v5_stage0_injections.py")
GENERATE = load_script("run_v5_sft_causal_generate.py")
PREPARE = load_script("prepare_v5_sft_causal.py")
EVALUATE = load_script("run_v5_sft_causal_eval.py")


def split_manifest():
    return {
        "protocol": "v5_stage0_tau2_end_to_end",
        "guarantees": {
            "validation_derived_from_official_train_only": True,
            "official_test_task_content_exported": False,
            "official_test_used_for_model_or_smoke_selection": False,
        },
        "domains": {
            "retail": {
                "inner_train_ids": [str(index) for index in range(59)],
                "validation_ids": [str(index) for index in range(100, 115)],
                "sealed_test_ids": [str(index) for index in range(200, 240)],
            },
            "airline": {
                "inner_train_ids": [str(index) for index in range(24)],
                "validation_ids": [str(index) for index in range(100, 106)],
                "sealed_test_ids": [str(index) for index in range(200, 220)],
            },
        },
    }


class V5DynamicAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.split_path = self.root / "split.json"
        self.split_path.write_text(json.dumps(split_manifest()), encoding="utf-8")
        rows = multifault_rows(
            {
                "retail": ["0", "1"],
                "airline": ["0", "1"],
            },
            source_split="derived_inner_train",
        )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                multifault_manifest(
                    rows,
                    protocol="v5_stage1_multifault_data_construction",
                    source_split="derived_inner_train",
                    split_sha256=AUDIT.sha256_file(self.split_path),
                )
            ),
            encoding="utf-8",
        )
        self.audit_path = self.root / "dynamic_audit.json"
        write_dynamic_audit(
            self.audit_path,
            manifest_path=self.manifest_path,
            split_manifest_path=self.split_path,
        )
        self.expected_ids = {row["pair_id"] for row in rows}

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return AUDIT.load_complete_dynamic_audit(
            self.audit_path,
            manifest_path=self.manifest_path,
            split_manifest_path=self.split_path,
            expected_source_split="derived_inner_train",
            expected_task_ids=self.expected_ids,
        )

    def test_complete_dynamic_audit_is_bound_to_bytes_tasks_and_db_state(self):
        identity = self.validate()
        self.assertEqual(identity["verified_injections"], 4)
        self.assertEqual(identity["sha256"], AUDIT.sha256_file(self.audit_path))

        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        payload["rows"][0]["agent_database_after"] = "mutated"
        self.audit_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "safety evidence"):
            self.validate()

    def test_absent_user_database_is_explicitly_recorded_and_accepted(self):
        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            row["user_database_before"] = None
            row["user_database_after"] = None
        self.audit_path.write_text(json.dumps(payload), encoding="utf-8")

        identity = self.validate()
        self.assertEqual(identity["verified_injections"], 4)

        payload["rows"][0].pop("user_database_before")
        self.audit_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "safety evidence"):
            self.validate()

    def test_incomplete_or_manifest_unbound_dynamic_audit_is_rejected(self):
        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        payload["status"] = "PASS"
        self.audit_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
            self.validate()

        write_dynamic_audit(
            self.audit_path,
            manifest_path=self.manifest_path,
            split_manifest_path=self.split_path,
        )
        payload = json.loads(self.audit_path.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = "0" * 64
        self.audit_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "manifest SHA drift"):
            self.validate()

    def test_all_tau2_entrypoints_reject_tracked_dirty_checkout(self):
        failing = SimpleNamespace(returncode=1)
        modules = (VERIFY, GENERATE, PREPARE, EVALUATE)
        for module in modules:
            with self.subTest(module=module.__name__):
                with mock.patch.object(
                    module.subprocess, "run", return_value=failing
                ):
                    with self.assertRaisesRegex(RuntimeError, "tracked"):
                        if module is VERIFY:
                            module.require_clean_tracked_checkout(self.root)
                        else:
                            module.require_clean_tracked_checkout(
                                self.root, label="tau2 checkout"
                            )


class V5VerifierManifestBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tau2 = self.root / "tau2"
        for domain in ("retail", "airline"):
            data = self.tau2 / "data" / "tau2" / "domains" / domain
            source = self.tau2 / "src" / "tau2" / "domains" / domain
            data.mkdir(parents=True)
            source.mkdir(parents=True)
            (data / "tasks.json").write_text("[]\n", encoding="utf-8")
            (data / "db.json").write_text("{}\n", encoding="utf-8")
            (source / "tools.py").write_text("# fixture\n", encoding="utf-8")
        self.split_path = self.root / "split.json"
        self.split_path.write_text(json.dumps(split_manifest()), encoding="utf-8")
        rows = multifault_rows(
            {
                "retail": split_manifest()["domains"]["retail"][
                    "inner_train_ids"
                ],
                "airline": split_manifest()["domains"]["airline"][
                    "inner_train_ids"
                ],
            },
            source_split="derived_inner_train",
        )
        self.manifest = multifault_manifest(
            rows,
            protocol="v5_stage1_multifault_data_construction",
            source_split="derived_inner_train",
            split_sha256=VERIFY.sha256_file(self.split_path),
        )
        self.manifest["tau2_commit"] = TAU2_COMMIT
        self.manifest["source_files"] = {
            domain: {
                "tasks_json_sha256": VERIFY.sha256_file(
                    self.tau2
                    / "data"
                    / "tau2"
                    / "domains"
                    / domain
                    / "tasks.json"
                ),
                "db_json_sha256": VERIFY.sha256_file(
                    self.tau2
                    / "data"
                    / "tau2"
                    / "domains"
                    / domain
                    / "db.json"
                ),
                "tools_py_sha256": VERIFY.sha256_file(
                    self.tau2
                    / "src"
                    / "tau2"
                    / "domains"
                    / domain
                    / "tools.py"
                ),
            }
            for domain in ("retail", "airline")
        }

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, manifest=None):
        with (
            mock.patch.object(VERIFY, "git_commit", return_value=TAU2_COMMIT),
            mock.patch.object(
                VERIFY,
                "PUBLISHED_STAGE0_SPLIT_SHA256",
                VERIFY.sha256_file(self.split_path),
            ),
        ):
            VERIFY.validate_multifault_manifest(
                tau2_root=self.tau2,
                manifest=self.manifest if manifest is None else manifest,
                split_manifest=self.split_path,
            )

    def test_generation_manifest_requires_exact_inner_train_binding(self):
        self.validate()
        broken = json.loads(json.dumps(self.manifest))
        broken["source_split"] = "derived_validation"
        with self.assertRaisesRegex(RuntimeError, "source_split"):
            self.validate(broken)

        broken = json.loads(json.dumps(self.manifest))
        broken["official_test_used"] = True
        with self.assertRaisesRegex(RuntimeError, "opened official test"):
            self.validate(broken)

        broken = json.loads(json.dumps(self.manifest))
        broken["rows"][0]["task_id"] = "200"
        broken["rows"][0]["pair_id"] = "retail:200"
        with self.assertRaisesRegex(RuntimeError, "official-test task leaked"):
            self.validate(broken)


if __name__ == "__main__":
    unittest.main()
