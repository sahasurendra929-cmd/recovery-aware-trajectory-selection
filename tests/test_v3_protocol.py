from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_tool_actions_v2 as eval_v2
import evaluate_tool_actions_v3 as eval_v3
import aggregate_qlora_v3 as aggregate_v3
import package_qlora_v3_results as package_v3
import prepare_qlora_v3 as prepare_v3
import run_qlora_v3 as run_v3


class V3ProtocolTests(unittest.TestCase):
    def test_v3_frozen_selection_constants_match_release_contract(self):
        self.assertEqual(
            prepare_v3.EXPECTED_V3_TRACE_SHA256,
            "a65bba64baf7c9a6e816e721b382511211aa9df6f5204e7c4cce74f78b992cc5",
        )
        self.assertEqual(prepare_v3.EXPECTED_V3_STATS["examples"], 1069)
        self.assertEqual(prepare_v3.EXPECTED_V3_STATS["recovery_targets"], 102)
        self.assertEqual(prepare_v3.EXPECTED_V3_STATS["agent_initiated_targets"], 20)
        self.assertEqual(prepare_v3.EXPECTED_V3_STATS["scheduled_loss_tokens"], 36599)
        self.assertEqual(
            run_v3.EXPECTED_V3_STATS[
                "scheduled_target_tool_tvd_from_v2_random"
            ],
            0.07444852941176469,
        )
        self.assertEqual(aggregate_v3.NON_RECOVERY_EXAMPLES, 906)
        self.assertEqual(aggregate_v3.V2_RANDOM_NON_RECOVERY_CORRECT, 286)
        self.assertEqual(
            aggregate_v3.NON_RECOVERY_RETENTION_FLOOR,
            0.29567328918322294,
        )
        self.assertEqual(
            run_v3.EXPECTED_V3_SCHEDULE_SHA256,
            prepare_v3.EXPECTED_V3_SCHEDULE_SHA256,
        )
        config = (ROOT / "configs" / "qlora_v3.yaml").read_text(encoding="utf-8")
        for value in (
            prepare_v3.EXPECTED_V3_TRACE_SHA256,
            prepare_v3.EXPECTED_V3_TRAIN_SHA256,
            prepare_v3.EXPECTED_V3_SCHEDULE_SHA256,
        ):
            self.assertIn(value, config)
        self.assertIn("threshold: 0.29567328918322294", config)

    def test_jsonl_writer_reproduces_windows_v2_crlf_bytes(self):
        rows = [{"x": 1}, {"text": "中文"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            prepare_v3.write_jsonl(path, rows)
            payload = path.read_bytes()
        expected = (
            json.dumps(rows[0], ensure_ascii=False).encode("utf-8")
            + b"\r\n"
            + json.dumps(rows[1], ensure_ascii=False).encode("utf-8")
            + b"\r\n"
        )
        self.assertEqual(payload, expected)
        self.assertNotIn(b"\n", payload.replace(b"\r\n", b""))

    def test_v3_parser_is_semantically_identical_to_v2(self):
        samples = (
            '{"name":"lookup","arguments":{"id":1}}',
            'prefix {"name":"lookup","arguments":"{\\"id\\": 1}"} suffix',
            "not JSON",
        )
        for sample in samples:
            self.assertEqual(eval_v3.parse_call(sample), eval_v2.parse_call(sample))

    def test_longest_prompt_smoke_selects_true_maximum_and_requires_limit_one(self):
        rows = [
            {"example_id": "short", "prompt_tokens": 10},
            {"example_id": "long-a", "prompt_tokens": 20},
            {"example_id": "long-b", "prompt_tokens": 20},
        ]
        selected = eval_v3.select_evaluation_rows(rows, 1, True)
        self.assertEqual(selected[0]["example_id"], "long-b")
        with self.assertRaisesRegex(RuntimeError, "requires exactly --limit 1"):
            eval_v3.select_evaluation_rows(rows, 2, True)
        with self.assertRaisesRegex(RuntimeError, "must be in"):
            eval_v3.select_evaluation_rows(rows, 0, False)

    def test_longest_prompt_smoke_artifact_validator_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "smoke"
            shared = root / "processed" / "shared"
            smoke.mkdir()
            shared.mkdir(parents=True)
            checkpoint = smoke / "checkpoint_final"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            rows = [
                {
                    "example_id": f"example-{index:03d}",
                    "prompt_tokens": 1664 if index == 958 else 100,
                }
                for index in range(run_v3.TEST_EXAMPLES)
            ]
            (shared / "test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            contract = {
                "protocol": run_v3.EVALUATION_PROTOCOL,
                "training_and_evaluation_protocol": "qlora_v2_frozen",
                "test_file_sha256": run_v3.EXPECTED_TEST_SHA256,
                "formal_test_examples": run_v3.TEST_EXAMPLES,
                "evaluated_examples": 1,
                "model": run_v3.MODEL,
                "model_revision": run_v3.MODEL_REVISION,
                "checkpoint_fingerprint": run_v3.checkpoint_fingerprint(checkpoint),
                "base_model_loading": "nf4_4bit",
                "max_prompt_tokens": run_v3.MAX_PROMPT_TOKENS,
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": 128,
                    "batch_size": 1,
                },
                "limited": True,
                "smoke_selection": "longest_prompt",
                "smoke_forced_new_tokens": 128,
            }
            metrics = {
                **contract,
                "groups": {"overall": {"examples": 1}},
            }
            (smoke / "longest_prompt_metrics.contract.json").write_text(
                json.dumps(contract),
                encoding="utf-8",
            )
            (smoke / "longest_prompt_metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
            (smoke / "longest_prompt_metrics.predictions.jsonl").write_text(
                json.dumps({
                    "example_id": "example-958",
                    "generated_token_count": 128,
                }) + "\n",
                encoding="utf-8",
            )
            metrics["predictions_sha256"] = run_v3.sha256_file(
                smoke / "longest_prompt_metrics.predictions.jsonl"
            )
            (smoke / "longest_prompt_metrics.json").write_text(
                json.dumps(metrics),
                encoding="utf-8",
            )
            (smoke / "smoke_artifact_audit.json").write_text(
                json.dumps({"status": "PASS"}),
                encoding="utf-8",
            )
            audit = run_v3.validate_longest_prompt_smoke(
                smoke,
                root / "processed",
            )
            self.assertEqual(audit["longest_prompt_evaluation"], "PASS")
            self.assertEqual(audit["longest_prompt_tokens"], 1664)
            predictions = json.loads(
                (smoke / "longest_prompt_metrics.predictions.jsonl").read_text()
            )
            predictions["example_id"] = "example-000"
            (smoke / "longest_prompt_metrics.predictions.jsonl").write_text(
                json.dumps(predictions) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "prediction example"):
                run_v3.validate_longest_prompt_smoke(
                    smoke,
                    root / "processed",
                )

    def test_tail_only_resume_repairs_partial_final_line_and_audits(self):
        complete = {
            "example_id": "example-1",
            "generated_text": "{}",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            predictions = directory_path / "metrics.predictions.jsonl"
            audit = directory_path / "metrics.resume_recovery.jsonl"
            prefix = (json.dumps(complete) + "\n").encode("utf-8")
            predictions.write_bytes(prefix + b'{"example_id":"partial"')
            rows = eval_v3.read_resumable_predictions(predictions, audit)
            self.assertEqual(rows, [complete])
            self.assertEqual(predictions.read_bytes(), prefix)
            events = eval_v3.read_jsonl(audit)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "truncated_incomplete_final_prediction_line")
        self.assertEqual(events[0]["completed_rows_retained"], 1)

    def test_resume_rejects_corrupt_middle_line(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            predictions = directory_path / "metrics.predictions.jsonl"
            audit = directory_path / "audit.jsonl"
            predictions.write_bytes(b'{"example_id":"ok"}\nnot-json\n{"example_id":"later"}\n')
            with self.assertRaisesRegex(RuntimeError, "only an unterminated malformed final line"):
                eval_v3.read_resumable_predictions(predictions, audit)
            self.assertFalse(audit.exists())

    def test_resume_completes_delimiter_for_valid_final_object(self):
        row = {"example_id": "complete"}
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            predictions = directory_path / "metrics.predictions.jsonl"
            audit = directory_path / "audit.jsonl"
            predictions.write_bytes(json.dumps(row).encode("utf-8"))
            rows = eval_v3.read_resumable_predictions(predictions, audit)
            self.assertEqual(rows, [row])
            self.assertTrue(predictions.read_bytes().endswith(b"\n"))
            event = eval_v3.read_jsonl(audit)[0]
        self.assertEqual(event["event"], "completed_missing_final_newline")

    def test_strict_tier_rejects_each_primary_contract_violation(self):
        tier = dict(prepare_v3.CONSTRAINT_TIERS[0])
        anchor = {"scheduled_loss_tokens": 36706, "trajectories": 141}
        stats = {
            "examples": 1069,
            "unique_tasks": 76,
            "anchor_task_overlap": 63,
            "non_recovery_targets": 967,
            "recovery_targets": 102,
            "agent_initiated_targets": 20,
            "target_tool_tvd_from_v2_random": 0.069,
            "scheduled_target_tool_tvd_from_v2_random": 0.069,
            "anchor_token_overlap_ratio": 0.672,
            "scheduled_loss_tokens": 36599,
            "unpaired_failed_action_excess": 2,
        }
        self.assertTrue(run_v3._tier_satisfied(stats, anchor, tier))
        violations = {
            "examples": 1089,
            "unique_tasks": 59,
            "anchor_task_overlap": 56,
            "non_recovery_targets": 938,
            "recovery_targets": 140,
            "agent_initiated_targets": 19,
            "target_tool_tvd_from_v2_random": 0.076,
            "scheduled_target_tool_tvd_from_v2_random": 0.076,
            "anchor_token_overlap_ratio": 0.649,
            "scheduled_loss_tokens": 40000,
            "unpaired_failed_action_excess": 4,
        }
        for key, bad_value in violations.items():
            changed = dict(stats)
            changed[key] = bad_value
            with self.subTest(key=key):
                self.assertFalse(run_v3._tier_satisfied(changed, anchor, tier))

    def test_trace_fingerprint_is_order_independent_and_newline_delimited(self):
        observed = run_v3.trace_set_fingerprint(["b", "a"])
        expected = hashlib.sha256(b"a\nb\n").hexdigest()
        self.assertEqual(observed, expected)
        self.assertEqual(observed, run_v3.trace_set_fingerprint(["a", "b"]))

    def test_v3_paths_and_protocol_do_not_alias_v2_outputs(self):
        source = (ROOT / "scripts" / "run_qlora_v3.py").read_text(encoding="utf-8")
        self.assertIn('Path("results/selection_v3")', source)
        self.assertIn('Path("data/processed/qlora_v3")', source)
        self.assertIn('Path("results/qlora_v3")', source)
        self.assertEqual(eval_v3.PROTOCOL, "qlora_v3")
        self.assertEqual(run_v3.BUILD_PROTOCOL, "qlora_v3_constrained_recovery")

    def test_packager_accepts_only_the_frozen_source_tag_commit(self):
        commit = "a" * 40
        with patch.object(
            package_v3.subprocess,
            "run",
            side_effect=[
                CompletedProcess([], 0, commit + "\n", ""),
                CompletedProcess([], 0, commit + "\n", ""),
                CompletedProcess([], 0, "", ""),
            ],
        ):
            self.assertEqual(package_v3.source_commit(), commit)
        with patch.object(
            package_v3.subprocess,
            "run",
            side_effect=[
                CompletedProcess([], 0, commit + "\n", ""),
                CompletedProcess([], 0, "b" * 40 + "\n", ""),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from frozen tag"):
                package_v3.source_commit()

    def test_packager_rejects_tracked_source_drift(self):
        commit = "a" * 40
        with patch.object(
            package_v3.subprocess,
            "run",
            side_effect=[
                CompletedProcess([], 0, commit + "\n", ""),
                CompletedProcess([], 0, commit + "\n", ""),
                CompletedProcess([], 0, " M scripts/evaluator.py\n", ""),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked source/config"):
                package_v3.source_commit()

    def test_packager_allows_nonfinite_diagnostic_but_rejects_nonfinite_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_log.json"
            path.write_text(
                json.dumps([{"loss": 0.5, "grad_norm": float("nan")}]),
                encoding="utf-8",
            )
            rows = package_v3.load_training_log(path)
            self.assertEqual(rows[0]["loss"], 0.5)
            path.write_text(
                json.dumps([{"eval_loss": float("nan")}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "objective loss"):
                package_v3.load_training_log(path)


if __name__ == "__main__":
    unittest.main()
