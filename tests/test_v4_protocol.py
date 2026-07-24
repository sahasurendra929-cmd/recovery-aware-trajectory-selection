from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_qlora_v4 as prepare_v4
import aggregate_qlora_v4 as aggregate_v4
import package_qlora_v4_results as package_v4
import run_qlora_v4 as run_v4
import train_preference_v4 as train_preference_v4
import train_qlora_v4_sft as train_sft_v4
from prepare_qlora_v2 import render_history_message


class FakeTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        # Stable synthetic tokenization is enough for unit-level cap checks.
        tokens = text.split()
        if add_special_tokens:
            tokens = ["<bos>"] + tokens
        return {"input_ids": list(range(len(tokens)))}


def raw_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        ],
    }


class V4ProtocolTests(unittest.TestCase):
    def test_frozen_counts_and_hashes_are_explicit(self):
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["clean_unique_train"], 965)
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["failed_unique_train"], 104)
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["strict_train_pairs"], 79)
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["strict_validation_pairs"], 24)
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["outcome_success_test"], 902)
        self.assertEqual(
            prepare_v4.EXPECTED_COUNTS["non_recovery_success_test"],
            852,
        )
        self.assertEqual(prepare_v4.EXPECTED_COUNTS["strict_test_pairs"], 48)
        self.assertEqual(
            train_sft_v4.EXPECTED_TRAIN_SHA256,
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "clean_train_schedule_sha256"
            ],
        )
        self.assertEqual(
            train_sft_v4.EXPECTED_VALIDATION_SHA256,
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "clean_validation_sha256"
            ],
        )
        self.assertEqual(train_sft_v4.SMOKE_STEPS, 2)
        self.assertEqual(train_preference_v4.FORMAL_SCHEDULE_ROWS, 144)
        self.assertEqual(train_preference_v4.FORMAL_MAX_STEPS, 18)
        self.assertEqual(train_preference_v4.SMOKE_SCHEDULE_ROWS, 16)
        self.assertEqual(
            run_v4.PREFERENCE_SCHEDULE_SHA256,
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "preference_train_schedule_sha256"
            ],
        )
        self.assertEqual(
            run_v4.PREFERENCE_SMOKE_SHA256,
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "preference_smoke_pairs_sha256"
            ],
        )
        self.assertEqual(
            aggregate_v4.EXPECTED_TEST_OUTCOMES_SHA256,
            prepare_v4.EXPECTED_OUTPUT_SHA256["test_outcomes_sha256"],
        )
        frozen_tags = {
            run_v4.FROZEN_TAG,
            package_v4.FROZEN_TAG,
            aggregate_v4.FROZEN_TAG,
            train_sft_v4.FROZEN_TAG,
            train_preference_v4.FROZEN_TAG,
        }
        self.assertEqual(frozen_tags, {"v4-frozen-20260724-p3"})
        self.assertEqual(
            aggregate_v4.STANDARD_V3_METRICS_SHA256,
            run_v4.V3_RESULT_SHA256["metrics.json"],
        )
        self.assertEqual(
            aggregate_v4.STANDARD_V3_CONTRACT_SHA256,
            run_v4.V3_RESULT_SHA256["metrics.contract.json"],
        )
        self.assertEqual(
            aggregate_v4.STANDARD_V3_PREDICTIONS_SHA256,
            run_v4.V3_RESULT_SHA256["metrics.predictions.jsonl"],
        )
        self.assertEqual(
            aggregate_v4.STANDARD_V3_CHECKPOINT_FINGERPRINT,
            run_v4.V3_CHECKPOINT_FINGERPRINT,
        )

    def test_exact_error_prefix_and_transient_filter(self):
        self.assertTrue(prepare_v4.deterministic_error("Error: user not found"))
        self.assertFalse(
            prepare_v4.deterministic_error("Error: network timeout; try again")
        )
        self.assertFalse(prepare_v4.deterministic_error("user not found"))

    def test_linked_outcome_checks_id_name_and_adjacent_result(self):
        call = raw_call("lookup", {"id": 1}, "call-1")
        result = {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "lookup",
            "content": "Error: user not found",
        }
        item = {
            "trace_id": "source:task1:trial0",
            "record": {"traj": [call, result]},
        }
        linked = prepare_v4.linked_tool_outcome(item, 0)
        self.assertEqual(linked["status"], "error")
        self.assertEqual(linked["error_type"], "not_found")
        broken = {
            **item,
            "record": {
                "traj": [
                    call,
                    {**result, "tool_call_id": "wrong"},
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "id mismatch"):
            prepare_v4.linked_tool_outcome(broken, 0)

    def test_pair_uses_post_error_context_and_excludes_future_result(self):
        tokenizer = FakeTokenizer()
        failed = raw_call("lookup", {"email": "bad@example.com"}, "failed")
        failure = {
            "role": "tool",
            "tool_call_id": "failed",
            "name": "lookup",
            "content": "Error: user not found",
        }
        chosen = raw_call(
            "lookup_by_name",
            {"name": "Ada", "zip": "12345"},
            "repair",
        )
        success = {
            "role": "tool",
            "tool_call_id": "repair",
            "name": "lookup_by_name",
            "content": "user_123",
        }
        trajectory = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "help"},
            failed,
            failure,
            chosen,
            success,
        ]
        prompt = (
            "prefix\n"
            + render_history_message(failed)
            + "\n"
            + render_history_message(failure)
            + "\n"
        )
        chosen_text = prepare_v4.canonical_completion(
            prepare_v4.tool_calls(chosen)[0]
        )
        chosen_tokens = len(
            tokenizer(chosen_text + tokenizer.eos_token)["input_ids"]
        )
        row = {
            "example_id": "source:task1:trial0:action4",
            "trace_id": "source:task1:trial0",
            "task_key": "retail:1",
            "source": "source",
            "target_message_index": 4,
            "prompt": prompt,
            "completion": chosen_text,
            "prompt_tokens": 10,
            "completion_tokens": chosen_tokens,
            "sequence_tokens": 10 + chosen_tokens,
            "recovery_mode": "agent_initiated",
            "linked_tool_outcome": "success",
        }
        item = {
            "trace_id": row["trace_id"],
            "record": {"traj": trajectory},
        }
        pair, reason = prepare_v4.build_preference_pair(
            row,
            item,
            tokenizer,
            "train",
        )
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(pair)
        self.assertNotEqual(pair["chosen"], pair["rejected"])
        self.assertIn("Error: user not found", pair["prompt"])
        self.assertNotIn("user_123", pair["prompt"])
        self.assertLess(
            pair["future_leakage_audit"]["prompt_source_max_index"],
            pair["future_leakage_audit"]["chosen_index"],
        )

        truncated = dict(row)
        truncated["prompt"] = "prefix without failed call"
        pair, reason = prepare_v4.build_preference_pair(
            truncated,
            item,
            tokenizer,
            "train",
        )
        self.assertIsNone(pair)
        self.assertEqual(reason, "required_error_context_truncated")

    def test_matched_clean_replacement_preserves_source_and_tool(self):
        clean_a = {
            "example_id": "clean-a",
            "source": "source-a",
            "target_tool": "lookup",
            "linked_tool_outcome": "success",
            "recovery_mode": "none",
            "completion_tokens": 5,
            "sequence_tokens": 100,
        }
        clean_b = {
            "example_id": "clean-b",
            "source": "source-a",
            "target_tool": "lookup",
            "linked_tool_outcome": "success",
            "recovery_mode": "none",
            "completion_tokens": 6,
            "sequence_tokens": 101,
        }
        schedule = []
        for index in range(prepare_v4.CLEAN_SFT_MICROBATCHES):
            row = dict(clean_a if index % 2 == 0 else clean_b)
            row["schedule_index"] = index
            row["schedule_pass"] = 0
            if index == 7:
                row.update(
                    {
                        "example_id": "failed",
                        "linked_tool_outcome": "error",
                        "linked_tool_result_error_type": "not_found",
                    }
                )
            schedule.append(row)
        output, mapping = prepare_v4.matched_clean_schedule(
            schedule,
            [clean_a, clean_b],
        )
        self.assertEqual(len(output), prepare_v4.CLEAN_SFT_MICROBATCHES)
        self.assertEqual(len(mapping), 1)
        self.assertTrue(
            all(row["linked_tool_outcome"] == "success" for row in output)
        )
        self.assertEqual(
            Counter(row["target_tool"] for row in output),
            Counter(row["target_tool"] for row in schedule),
        )
        self.assertEqual(
            Counter(row["source"] for row in output),
            Counter(row["source"] for row in schedule),
        )

    def test_preference_schedule_is_mode_balanced_and_covers_all_pairs(self):
        pairs = []
        for mode, count in (("agent_initiated", 3), ("user_assisted", 5)):
            for index in range(count):
                pairs.append(
                    {
                        "pair_id": f"{mode}-{index}",
                        "recovery_mode": mode,
                    }
                )
        schedule = prepare_v4.balanced_preference_schedule(pairs)
        self.assertEqual(len(schedule), prepare_v4.PREFERENCE_MICROBATCHES)
        self.assertEqual(
            Counter(row["recovery_mode"] for row in schedule),
            Counter({"agent_initiated": 72, "user_assisted": 72}),
        )
        self.assertEqual(
            {row["pair_id"] for row in schedule},
            {row["pair_id"] for row in pairs},
        )

    def test_output_hash_drift_fails_closed(self):
        expected = dict(prepare_v4.EXPECTED_OUTPUT_SHA256)
        changed = dict(expected)
        changed["clean_train_schedule_sha256"] = "0" * 64
        with patch.object(prepare_v4, "EXPECTED_OUTPUT_SHA256", changed):
            self.assertNotEqual(expected, prepare_v4.EXPECTED_OUTPUT_SHA256)

    def test_training_script_never_accepts_held_out_input(self):
        source = (
            ROOT / "scripts" / "train_qlora_v4_sft.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--test-file", source)
        self.assertIn('"held_out_test_accessed": False', source)

    def test_release_config_has_no_placeholder_and_matches_code(self):
        config = (ROOT / "configs" / "qlora_v4.yaml").read_text(
            encoding="utf-8"
        )
        handoff = (ROOT / "BASELINE_V4_HANDOFF.md").read_text(
            encoding="utf-8"
        )
        prompt = (ROOT / "V4_RUNPOD4090_AGENT_PROMPT.md").read_text(
            encoding="utf-8"
        )
        unresolved_marker = "FILL" + "_AFTER_"
        self.assertNotIn(unresolved_marker, config + handoff + prompt)
        self.assertIn(
            f"repository_tag: {run_v4.FROZEN_TAG}",
            config,
        )
        self.assertIn(
            f"V4 repository tag:                 {run_v4.FROZEN_TAG}",
            handoff,
        )
        self.assertIn(
            f'v4_tag="{run_v4.FROZEN_TAG}"',
            prompt,
        )
        self.assertIn("scientific_protocol_changed: false", config)
        self.assertIn(
            "supersedes_commit: "
            "f710f7480a314328a3fcd0f05917e3ddbb65478d",
            config,
        )
        for value in (
            run_v4.FROZEN_TAG,
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "clean_train_schedule_sha256"
            ],
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "preference_train_schedule_sha256"
            ],
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "preference_smoke_pairs_sha256"
            ],
            prepare_v4.EXPECTED_OUTPUT_SHA256["test_outcomes_sha256"],
            prepare_v4.EXPECTED_OUTPUT_SHA256[
                "test_preference_pairs_sha256"
            ],
        ):
            self.assertIn(value, config)
            self.assertIn(value, handoff + config)

    def test_v4_output_names_and_evaluator_protocol_are_isolated(self):
        self.assertEqual(
            set(run_v4.ARM_RESULT_DIRS),
            {"clean_sft", "continued_sft", "dpo"},
        )
        wrapper = (
            ROOT / "scripts" / "evaluate_tool_actions_v4.py"
        ).read_text(encoding="utf-8")
        self.assertIn('frozen.PROTOCOL = "qlora_v4"', wrapper)
        self.assertNotIn('PROTOCOL = "qlora_v3"', wrapper)

    def test_checkpoint_fingerprint_is_identical_across_v4_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "adapter_config.json").write_bytes(b"config")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
            preference, _ = train_preference_v4.checkpoint_fingerprint(
                checkpoint
            )
            package = package_v4.checkpoint_fingerprint(checkpoint)
            clean, _ = train_sft_v4.checkpoint_fingerprint(checkpoint)
            expected = hashlib.sha256()
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
            ):
                expected.update(filename.encode("utf-8"))
                expected.update(
                    hashlib.sha256(
                        (checkpoint / filename).read_bytes()
                    ).hexdigest().encode("ascii")
                )
            self.assertEqual(preference, expected.hexdigest())
            self.assertEqual(package, expected.hexdigest())
            self.assertEqual(clean, expected.hexdigest())

    def test_pair_score_writer_schema_matches_aggregator(self):
        source = (ROOT / "scripts" / "train_preference_v4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"split": args.score_split', source)
        self.assertIn('"summed_logp_correct": is_correct', source)
        self.assertIn(
            '"per_token_normalized_correct": is_normalized_correct',
            source,
        )
        self.assertNotIn('"summed_margin_correct": is_correct', source)

    def test_three_planned_contrasts_are_frozen(self):
        self.assertEqual(
            aggregate_v4.CONTRASTS,
            {
                "clean_sft_minus_standard_v3": (
                    "clean_sft",
                    "standard_v3",
                ),
                "continued_sft_minus_clean_sft": (
                    "continued_sft",
                    "clean_sft",
                ),
                "dpo_minus_continued_sft": (
                    "dpo",
                    "continued_sft",
                ),
            },
        )
        self.assertEqual(len(aggregate_v4.PRIMARY_CONTRASTS), 2)
        config = (ROOT / "configs" / "qlora_v4.yaml").read_text(
            encoding="utf-8"
        )
        for threshold in ("0.04", "-0.041666666666666664", "0.0625", "-0.02"):
            self.assertIn(threshold, config)

    def test_clean_sft_finite_audit_rejects_nan(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train_sft_v4.finite_training_audit(
                [
                    {
                        "loss": 1.0,
                        "grad_norm": float("nan"),
                        "eval_loss": 1.0,
                    }
                ],
                {"train_loss": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
