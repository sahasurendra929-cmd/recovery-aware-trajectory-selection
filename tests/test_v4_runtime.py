from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_tool_actions_v4 as evaluate_v4
import run_qlora_v4 as run_v4
import train_preference_v4 as preference_v4


SOURCE_COMMIT = "a" * 40


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def make_checkpoint(path: Path, suffix: bytes = b"") -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_bytes(b"config" + suffix)
    (path / "adapter_model.safetensors").write_bytes(b"weights" + suffix)
    return run_v4.checkpoint_fingerprint(path)


class V4RuntimeTests(unittest.TestCase):
    def test_evaluator_warms_cuda_before_reset_and_measures_generation_peak(self):
        """The Windows evaluator must initialize its allocator before reset.

        ``torch.cuda.is_available()`` can use a driver/NVML check without
        creating the CUDA allocator in the current process.  On the RTX 5060
        Windows runner, resetting peak statistics in that state reproducibly
        raises ``RuntimeError: Invalid device argument``.  ``torch.cuda.init``
        is the explicit boundary that initializes the caching allocator; an
        optional allocation can further exercise it without changing the
        measured interval because the reset remains immediately before the
        frozen evaluator.
        """
        events: list[str] = []

        class FakeCuda:
            def __init__(self) -> None:
                self.initialized = False

            def is_available(self) -> bool:
                events.append("is_available")
                return True

            def init(self) -> None:
                events.append("cuda_init")
                self.initialized = True

            def set_device(self, device: int) -> None:
                if not self.initialized:
                    raise RuntimeError("set_device called before CUDA init")
                events.append(f"set_device:{device}")

            def synchronize(self, device: int) -> None:
                events.append(f"synchronize:{device}")

            def empty_cache(self) -> None:
                events.append("empty_cache")

            def reset_peak_memory_stats(self, device: int) -> None:
                events.append(f"reset_peak:{device}")
                if not self.initialized:
                    raise RuntimeError(
                        "Invalid device argument; did you call init?"
                    )

            def max_memory_allocated(self, device: int) -> int:
                events.append(f"max_allocated:{device}")
                return 5_000_000_000

            def max_memory_reserved(self, device: int) -> int:
                events.append(f"max_reserved:{device}")
                return 6_000_000_000

        def empty(*shape: int, **kwargs: object) -> object:
            events.append(
                "allocator_warmup:"
                f"shape={shape}:device={kwargs.get('device')}"
            )
            return object()

        fake_torch = types.SimpleNamespace(cuda=FakeCuda(), empty=empty)

        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "metrics.json"

            def frozen_main() -> None:
                events.append("frozen_generation")
                write_json(metrics_path, {"protocol": "qlora_v4"})

            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_tool_actions_v4.py",
                        "--output",
                        str(metrics_path),
                    ],
                ),
                patch.object(evaluate_v4.frozen, "main", frozen_main),
            ):
                evaluate_v4.main()

            required = [
                "is_available",
                "cuda_init",
                "set_device:0",
                "reset_peak:0",
                "frozen_generation",
                "max_allocated:0",
                "max_reserved:0",
            ]
            for event in required:
                self.assertIn(event, events)
            self.assertEqual(events.count("reset_peak:0"), 1)
            self.assertEqual(
                [events.index(event) for event in required],
                sorted(events.index(event) for event in required),
            )
            observed = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                observed["runtime_memory"],
                {
                    "peak_cuda_memory_allocated_bytes": 5_000_000_000,
                    "peak_cuda_memory_reserved_bytes": 6_000_000_000,
                },
            )

    def test_evaluator_accepts_both_output_argument_forms(self):
        self.assertEqual(
            evaluate_v4.output_path_from_argv(
                ["--output", "metrics.json"]
            ),
            Path("metrics.json"),
        )
        self.assertEqual(
            evaluate_v4.output_path_from_argv(
                ["--output=other.json"]
            ),
            Path("other.json"),
        )
        with self.assertRaisesRegex(RuntimeError, "unable to locate"):
            evaluate_v4.output_path_from_argv(["--output"])

    def test_command_receipts_are_atomic_and_reject_a_torn_existing_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            with patch.object(run_v4, "V4_RESULTS", results):
                run_v4.append_command({"stage": "one"})
                run_v4.append_command({"stage": "two"})
                self.assertEqual(
                    [row["stage"] for row in run_v4.load_jsonl(
                        results / "commands.jsonl"
                    )],
                    ["one", "two"],
                )
                self.assertFalse(
                    (results / "commands.jsonl.atomic.tmp").exists()
                )
                commands = results / "commands.jsonl"
                commands.write_bytes(commands.read_bytes() + b"{")
                before = commands.read_bytes()
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unterminated tail",
                ):
                    run_v4.append_command({"stage": "three"})
                self.assertEqual(commands.read_bytes(), before)

    def test_validated_completion_receipt_is_recoverable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            with (
                patch.object(run_v4, "V4_RESULTS", results),
                patch.object(
                    run_v4,
                    "source_commit",
                    return_value=SOURCE_COMMIT,
                ),
            ):
                run_v4.record_validated_completion("evaluate-clean_sft")
                run_v4.record_validated_completion("evaluate-clean_sft")
            rows = run_v4.load_jsonl(results / "commands.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["returncode"], 0)
            self.assertTrue(rows[0]["recovered_from_validated_artifacts"])
            self.assertEqual(rows[0]["source_commit"], SOURCE_COMMIT)

    def test_archive_retry_preserves_partial_output_and_handles_file_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            partial = results / "continued_sft"
            partial.mkdir(parents=True)
            (partial / "console.log").write_text("failure", encoding="utf-8")
            with patch.object(run_v4, "V4_RESULTS", results):
                with self.assertRaisesRegex(RuntimeError, "non-empty"):
                    run_v4.prepare_output_directory(
                        partial,
                        "continued-SFT",
                        archive_partial=False,
                    )
                run_v4.prepare_output_directory(
                    partial,
                    "continued-SFT",
                    archive_partial=True,
                )
            self.assertTrue(partial.is_dir())
            self.assertEqual(list(partial.iterdir()), [])
            archived = list((results / "failed_attempts").glob("*continued_sft"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(
                (archived[0] / "console.log").read_text(encoding="utf-8"),
                "failure",
            )

            file_target = results / "dpo"
            file_target.write_text("partial", encoding="utf-8")
            with patch.object(run_v4, "V4_RESULTS", results):
                run_v4.prepare_output_directory(
                    file_target,
                    "DPO",
                    archive_partial=True,
                )
            self.assertTrue(file_target.is_dir())
            self.assertEqual(list(file_target.iterdir()), [])

    def test_clean_smoke_reuses_an_already_valid_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            smoke = results / "smoke_clean"
            smoke.mkdir(parents=True)
            (smoke / "run_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            with (
                patch.object(run_v4, "V4_RESULTS", results),
                patch.object(run_v4, "validate_prepare_audit"),
                patch.object(run_v4, "validate_clean_training_result"),
                patch.object(run_v4, "run_longest_prompt_smoke") as longest,
                patch.object(
                    run_v4,
                    "prepare_output_directory",
                ) as prepare_output,
            ):
                run_v4.smoke_clean(archive_partial=False)
            longest.assert_called_once()
            prepare_output.assert_not_called()

    def test_preference_finite_audit_ignores_boolean_loss_flags(self):
        audit = preference_v4.finite_loss_audit(
            [
                {
                    "loss": 0.7,
                    "grad_norm": 0.4,
                    "loss_enabled": True,
                }
            ],
            {"train_loss": 0.6},
        )
        self.assertTrue(audit["finite"])
        self.assertEqual(audit["value_count"], 2)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            preference_v4.finite_loss_audit(
                [{"loss": math.nan, "grad_norm": 0.4}],
                {"train_loss": 0.6},
            )

    def test_runtime_memory_injection_is_validated_and_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.json"
            write_json(metrics, {"protocol": "qlora_v4"})
            evaluate_v4.inject_runtime_memory(
                metrics,
                peak_allocated=123,
                peak_reserved=456,
            )
            observed = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertEqual(
                observed["runtime_memory"],
                {
                    "peak_cuda_memory_allocated_bytes": 123,
                    "peak_cuda_memory_reserved_bytes": 456,
                },
            )
            self.assertFalse(
                metrics.with_name(
                    metrics.name + ".runtime-memory.tmp"
                ).exists()
            )
            before = metrics.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "invalid CUDA"):
                evaluate_v4.inject_runtime_memory(
                    metrics,
                    peak_allocated=500,
                    peak_reserved=499,
                )
            self.assertEqual(metrics.read_bytes(), before)

    def test_clean_and_preference_stage_gates_match_writer_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / "clean_sft"
            clean_fingerprint = make_checkpoint(
                clean / "checkpoint_final",
                b"-clean",
            )
            write_json(
                clean / "run_manifest.json",
                {
                    "protocol": "qlora_v4_clean_sft",
                    "source_tag": run_v4.FROZEN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "arm": "clean_sft",
                    "smoke_test": False,
                    "failed_action_labels": 0,
                    "max_steps": 68,
                    "grad_accum": 16,
                    "train_file_sha256": run_v4.EXPECTED_OUTPUT_SHA256[
                        "clean_train_schedule_sha256"
                    ],
                    "validation_file_sha256": run_v4.EXPECTED_OUTPUT_SHA256[
                        "clean_validation_sha256"
                    ],
                    "held_out_test_accessed": False,
                    "loss_audit": {"finite": True},
                    "output_checkpoint": {
                        "checkpoint_fingerprint": clean_fingerprint
                    },
                    "environment": {
                        "peak_cuda_reserved_bytes": 6_000_000_000
                    },
                },
            )

            continued = root / "continued_sft"
            continued_fingerprint = make_checkpoint(
                continued / "checkpoint_final",
                b"-continued",
            )
            write_json(
                continued / "run_manifest.json",
                {
                    "protocol": "qlora_v4_preference_continuation",
                    "source_tag": run_v4.FROZEN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "formal_result": True,
                    "arm": "continued_sft",
                    "data": {
                        "train_schedule_sha256":
                            run_v4.PREFERENCE_SCHEDULE_SHA256,
                        "held_out_test_accessed": False,
                    },
                    "compute_contract": {
                        "optimizer_steps": 18,
                        "scheduled_microbatches": 144,
                        "gradient_accumulation_steps": 8,
                    },
                    "clean_sft_initialization": {
                        "checkpoint_fingerprint": clean_fingerprint
                    },
                    "output_checkpoint": {
                        "checkpoint_fingerprint": continued_fingerprint
                    },
                    "loss_audit": {"finite": True},
                    "runtime": {
                        "peak_cuda_memory_reserved_bytes": 6_500_000_000
                    },
                    "comparison_contract_id": "comparison",
                },
            )
            arm_dirs = {
                "clean_sft": clean,
                "continued_sft": continued,
                "dpo": root / "dpo",
            }
            with (
                patch.object(run_v4, "ARM_RESULT_DIRS", arm_dirs),
                patch.object(
                    run_v4,
                    "source_commit",
                    return_value=SOURCE_COMMIT,
                ),
            ):
                self.assertEqual(
                    run_v4.validate_clean_training_result(
                        clean,
                        smoke=False,
                    ),
                    clean_fingerprint,
                )
                self.assertEqual(
                    run_v4.validate_preference_training_result(
                        continued,
                        "continued_sft",
                        smoke=False,
                    ),
                    (continued_fingerprint, "comparison"),
                )

    def test_formal_evaluation_gate_checks_frozen_order_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "clean_sft"
            fingerprint = make_checkpoint(result / "checkpoint_final")
            test_file = root / "test.jsonl"
            test_rows = [
                {"example_id": f"example-{index}"}
                for index in range(959)
            ]
            write_jsonl(test_file, test_rows)
            predictions = [
                {
                    "example_id": row["example_id"],
                    "generated_text": "{}",
                    "target": {"name": "tool", "arguments": {}},
                    "json_valid": True,
                    "tool_name_correct": True,
                    "arguments_exact": True,
                    "full_call_exact": True,
                }
                for row in test_rows
            ]
            predictions_path = result / "metrics.predictions.jsonl"
            write_jsonl(predictions_path, predictions)
            contract = {
                "protocol": "qlora_v4",
                "test_file_sha256": run_v4.EXPECTED_TEST_SHA256,
                "formal_test_examples": 959,
                "evaluated_examples": 959,
                "model": run_v4.MODEL,
                "model_revision": run_v4.MODEL_REVISION,
                "base_model_loading": "nf4_4bit",
                "max_prompt_tokens": 1664,
                "limited": False,
                "checkpoint_fingerprint": fingerprint,
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": 128,
                    "batch_size": 1,
                },
            }
            metrics = {
                **contract,
                "predictions_sha256": run_v4.sha256_file(predictions_path),
                "groups": {
                    "overall": {
                        "examples": 959,
                        "micro": {
                            "json_valid": 1.0,
                            "tool_name_correct": 1.0,
                            "arguments_exact": 1.0,
                            "full_call_exact": 1.0,
                        },
                    }
                },
                "runtime_memory": {
                    "peak_cuda_memory_allocated_bytes": 5_000_000_000,
                    "peak_cuda_memory_reserved_bytes": 6_000_000_000,
                },
            }
            write_json(result / "metrics.contract.json", contract)
            write_json(result / "metrics.json", metrics)
            with patch.object(run_v4, "TEST_FILE", test_file):
                run_v4.validate_formal_metrics(result)
                predictions[0]["example_id"] = "foreign"
                write_jsonl(predictions_path, predictions)
                metrics["predictions_sha256"] = run_v4.sha256_file(
                    predictions_path
                )
                write_json(result / "metrics.json", metrics)
                with self.assertRaisesRegex(RuntimeError, "IDs/order"):
                    run_v4.validate_formal_metrics(result)

    def test_pair_score_gate_matches_writer_and_aggregator_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            pair_file = (
                processed / "evaluation" / "test_preference_pairs.jsonl"
            )
            pairs = [
                {
                    "pair_id": f"pair-{index}",
                    "task_key": f"task-{index // 2}",
                    "prompt": f"prompt-{index}",
                    "chosen": f"chosen-{index}",
                    "rejected": f"rejected-{index}",
                }
                for index in range(48)
            ]
            write_jsonl(pair_file, pairs)
            pair_file_sha = run_v4.sha256_file(pair_file)

            arm_dir = root / "clean_sft"
            fingerprint = make_checkpoint(
                arm_dir / "checkpoint_final",
            )
            score_dir = root / "pair_scores" / "clean_sft"
            scores = []
            for index, pair in enumerate(pairs):
                content = hashlib.sha256(
                    json.dumps(
                        {
                            "prompt": pair["prompt"],
                            "chosen": pair["chosen"],
                            "rejected": pair["rejected"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                scores.append(
                    {
                        "pair_index": index,
                        "pair_id": pair["pair_id"],
                        "pair_content_sha256": content,
                        "split": "test",
                        "task_key": pair["task_key"],
                        "chosen_tokens_including_eos": 2,
                        "rejected_tokens_including_eos": 2,
                        "chosen_summed_logp_including_eos": -1.0,
                        "rejected_summed_logp_including_eos": -2.0,
                        "summed_logp_margin_chosen_minus_rejected": 1.0,
                        "chosen_per_token_logp_including_eos": -0.5,
                        "rejected_per_token_logp_including_eos": -1.0,
                        "per_token_normalized_margin_chosen_minus_rejected":
                            0.5,
                        "summed_logp_correct": True,
                        "per_token_normalized_correct": True,
                    }
                )
            scores_path = score_dir / "pair_scores.jsonl"
            write_jsonl(scores_path, scores)
            metrics_path = score_dir / "metrics.json"
            write_json(
                metrics_path,
                {
                    "protocol":
                        "qlora_v4_preference_continuation_pair_scoring",
                    "split": "test",
                    "pair_count": 48,
                    "completion_eos_included": True,
                    "pair_accuracy_summed_logp": 1.0,
                    "summed_logp_correct_count": 48,
                    "summed_logp_tie_count": 0,
                    "mean_summed_logp_margin": 1.0,
                    "median_summed_logp_margin": 1.0,
                    "per_token_normalized_pair_accuracy": 1.0,
                    "per_token_normalized_correct_count": 48,
                    "mean_per_token_normalized_margin": 0.5,
                    "median_per_token_normalized_margin": 0.5,
                    "peak_cuda_memory_allocated_bytes": 5_000_000_000,
                    "peak_cuda_memory_reserved_bytes": 6_000_000_000,
                },
            )
            metrics_sha = run_v4.sha256_file(metrics_path)
            manifest = {
                "protocol":
                    "qlora_v4_preference_continuation_pair_scoring",
                "source_tag": run_v4.FROZEN_TAG,
                "source_commit": SOURCE_COMMIT,
                "split": "test",
                "training_performed": False,
                "limited": False,
                "complete": True,
                "expected_pairs": 48,
                "completed_pairs": 48,
                "pair_count": 48,
                "preference_pairs_sha256": pair_file_sha,
                "pair_scores_sha256": run_v4.sha256_file(scores_path),
                "metrics_sha256": metrics_sha,
                "checkpoint_fingerprint": fingerprint,
                "outputs": {
                    "metrics_sha256": metrics_sha,
                    "pair_scores_rows": 48,
                },
            }
            write_json(score_dir / "score_manifest.json", manifest)
            arm_dirs = {
                "clean_sft": arm_dir,
                "continued_sft": root / "continued_sft",
                "dpo": root / "dpo",
            }
            with (
                patch.object(run_v4, "V4_PROCESSED", processed),
                patch.object(run_v4, "ARM_RESULT_DIRS", arm_dirs),
                patch.object(
                    run_v4,
                    "TEST_PREFERENCE_PAIRS_SHA256",
                    pair_file_sha,
                ),
                patch.object(
                    run_v4,
                    "source_commit",
                    return_value=SOURCE_COMMIT,
                ),
            ):
                run_v4.validate_pair_score_result(
                    score_dir,
                    "clean_sft",
                )
                manifest.pop("metrics_sha256")
                write_json(score_dir / "score_manifest.json", manifest)
                with self.assertRaisesRegex(RuntimeError, "output hash"):
                    run_v4.validate_pair_score_result(
                        score_dir,
                        "clean_sft",
                    )


if __name__ == "__main__":
    unittest.main()
