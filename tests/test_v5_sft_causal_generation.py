import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    TAU2_COMMIT,
    fault_protocol,
    multifault_manifest,
    multifault_row,
    write_dynamic_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_generate.py"
SPEC = importlib.util.spec_from_file_location("run_v5_sft_causal_generate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(domain, task_id):
    return multifault_row(
        domain,
        task_id,
        source_split="derived_inner_train",
    )


def split_manifest():
    return {
        "protocol": "v5_stage0_tau2_end_to_end",
        "guarantees": {
            "validation_derived_from_official_train_only": True,
            "official_test_task_content_exported": False,
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


def all_inner_rows():
    return [
        *[row("retail", index) for index in range(59)],
        *[row("airline", index) for index in range(24)],
    ]


class V5SFTCausalGenerationTests(unittest.TestCase):
    def write(self, payload, split=None):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.json"
        split_path = Path(directory.name) / "split_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        split_path.write_text(
            json.dumps(split if split is not None else split_manifest()),
            encoding="utf-8",
        )
        return directory, path, split_path

    def test_inner_train_manifest_is_accepted(self):
        rows = all_inner_rows()
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        split = MODULE.load_split_manifest(split_path)
        loaded = MODULE.load_inner_train_manifest(
            path,
            split_manifest=split,
            split_manifest_sha256=MODULE.sha256_file(split_path),
        )
        self.assertEqual(len(loaded["rows"]), 83)

    def test_validation_or_unsafe_manifest_is_rejected(self):
        rows = all_inner_rows()
        rows[0] = row("retail", "100")
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "validation/test"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        rows = all_inner_rows()
        payload["rows"] = rows
        rows[0]["error_condition"]["expected_state_mutation"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_split_protocol_counts_and_declared_hash_fail_closed(self):
        rows = all_inner_rows()
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="0" * 64,
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "split_manifest_sha256"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        broken = split_manifest()
        broken["domains"]["retail"]["sealed_test_ids"].pop()
        split_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "count drift"):
            MODULE.load_split_manifest(split_path)

    @mock.patch.object(MODULE, "require_clean_tracked_source")
    def test_generation_provenance_requires_exact_commits(self, _clean_source):
        source = "a" * 40
        with mock.patch.object(MODULE, "git_commit", return_value=source):
            self.assertEqual(
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="b" * 40,
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                ),
                source,
            )
        with mock.patch.object(MODULE, "git_commit", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "teacher-revision"):
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="main",
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                )
        with mock.patch.object(MODULE, "git_commit", return_value="e" * 40):
            with self.assertRaisesRegex(RuntimeError, "source commit drift"):
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="b" * 40,
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                )

    def test_four_shards_are_balanced_and_disjoint(self):
        rows = [row("retail", index) for index in range(83)]
        shards = [MODULE.shard_rows(rows, index, 4) for index in range(4)]
        keys = [item["pair_id"] for shard in shards for item in shard]
        self.assertEqual(len(keys), 83)
        self.assertEqual(len(set(keys)), 83)
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_output_filename_records_shard(self):
        self.assertEqual(
            MODULE.output_filename("retail", "error", 1, 4),
            "retail_error.shard-001-of-004.json",
        )

    def test_local_vllm_models_use_explicit_litellm_provider(self):
        self.assertEqual(
            MODULE.litellm_openai_model("Qwen/Qwen2.5-14B-Instruct-AWQ"),
            "openai/Qwen/Qwen2.5-14B-Instruct-AWQ",
        )
        with self.assertRaisesRegex(RuntimeError, "raw served model name"):
            MODULE.litellm_openai_model(
                "openai/Qwen/Qwen2.5-14B-Instruct-AWQ"
            )

    def test_generation_contract_stays_incomplete_until_atomic_finalize(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        manifest = root / "manifest.json"
        split = root / "split.json"
        split.write_text("{}\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                multifault_manifest(
                    [row("retail", "7")],
                    protocol=MODULE.GENERATION_PROTOCOL,
                    source_split="derived_inner_train",
                    split_sha256=MODULE.sha256_file(split),
                )
            ),
            encoding="utf-8",
        )
        dynamic_audit_path = root / "dynamic_audit.json"
        write_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest,
            split_manifest_path=split,
        )
        dynamic_audit_identity = MODULE.load_complete_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest,
            split_manifest_path=split,
            expected_source_split="derived_inner_train",
            expected_task_ids={"retail:7"},
        )
        output = root / "raw"
        args = SimpleNamespace(
            output_dir=output,
            num_shards=4,
            shard_index=2,
            expected_source_commit="a" * 40,
            num_trials=3,
            teacher_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
            teacher_revision="b" * 40,
            teacher_api_base="http://teacher/v1",
            teacher_mode="ground_truth",
            user_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            user_revision="c" * 40,
            user_api_base="http://user/v1",
            judge_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            judge_revision="c" * 40,
            judge_api_base="http://user/v1",
            max_tokens=512,
            max_steps=60,
            timeout=900,
            seed=20260722,
        )
        contract = MODULE.write_contract(
            args,
            manifest.resolve(),
            split.resolve(),
            [row("retail", "7")],
            dynamic_audit_path=dynamic_audit_path,
            dynamic_audit_identity=dynamic_audit_identity,
        )
        initial = json.loads(contract.read_text(encoding="utf-8"))
        self.assertEqual(initial["status"], "INCOMPLETE")
        self.assertEqual(initial["result_sha256"], {})
        self.assertFalse(initial["decoding"]["parallel_tool_calls"])
        self.assertEqual(
            initial["decoding"]["mixed_tool_call_content_normalization"],
            "drop_text_preserve_sha256",
        )
        self.assertEqual(initial["fault_protocol"], fault_protocol())
        self.assertEqual(initial["tau2_commit"], TAU2_COMMIT)
        self.assertEqual(initial["source_files"], SOURCE_FILES)
        self.assertEqual(
            initial["dynamic_audit_identity"], dynamic_audit_identity
        )

        # A failed generation never calls finalize and therefore cannot look
        # like a complete artifact set.
        self.assertEqual(
            json.loads(contract.read_text(encoding="utf-8"))["status"],
            "INCOMPLETE",
        )

        result = output / "retail_clean.shard-002-of-004.json"
        result.write_text('{"simulations": []}\n', encoding="utf-8")
        hashes = MODULE.finalize_contract(contract, [result])
        completed = json.loads(contract.read_text(encoding="utf-8"))
        self.assertEqual(completed["status"], "COMPLETE")
        self.assertEqual(completed["result_sha256"], hashes)
        self.assertEqual(hashes[result.name], MODULE.sha256_file(result))
        with self.assertRaisesRegex(RuntimeError, "not INCOMPLETE"):
            MODULE.finalize_contract(contract, [result])

    def test_endpoint_disables_parallel_tool_calls(self):
        args = MODULE.endpoint_args(
            "http://localhost/v1",
            "key",
            max_tokens=512,
            seed=20260722,
        )
        self.assertIs(args["parallel_tool_calls"], False)


if __name__ == "__main__":
    unittest.main()
