from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import package_qlora_v4_results as package_v4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class V4PackageTests(unittest.TestCase):
    def test_checkpoint_fingerprint_uses_frozen_unseparated_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "adapter_config.json").write_text(
                '{"peft_type":"LORA"}\n',
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"x" * 32)
            observed, hashes = package_v4.checkpoint_identity(checkpoint)
            expected = hashlib.sha256()
            for filename in package_v4.CHECKPOINT_FILES:
                expected.update(filename.encode("utf-8"))
                expected.update(hashes[filename].encode("ascii"))
            self.assertEqual(observed, expected.hexdigest())

    def test_formal_jsonl_rejects_blank_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "blank line"):
                package_v4.load_jsonl(path)

    def test_required_pair_score_is_strictly_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            score_dir = Path(temporary)
            rows = [
                {"pair_id": f"pair-{index}"}
                for index in range(package_v4.STRICT_PAIR_EXAMPLES)
            ]
            scores_path = score_dir / "pair_scores.jsonl"
            scores_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = {
                "protocol": package_v4.SCORE_PROTOCOL,
                "split": "test",
                "pair_count": package_v4.STRICT_PAIR_EXAMPLES,
                "completion_eos_included": True,
            }
            metrics_path = score_dir / "metrics.json"
            write_json(metrics_path, metrics)
            checkpoint_hashes = {
                "adapter_config.json": "1" * 64,
                "adapter_model.safetensors": "2" * 64,
            }
            fingerprint = "3" * 64
            manifest = {
                "protocol": package_v4.SCORE_PROTOCOL,
                "source_tag": package_v4.FROZEN_TAG,
                "source_commit": "4" * 40,
                "mode": "score",
                "split": "test",
                "training_performed": False,
                "limited": False,
                "complete": True,
                "expected_pairs": package_v4.STRICT_PAIR_EXAMPLES,
                "completed_pairs": package_v4.STRICT_PAIR_EXAMPLES,
                "pair_count": package_v4.STRICT_PAIR_EXAMPLES,
                "preference_pairs_sha256": package_v4.PAIR_SHA256,
                "pair_scores_sha256": package_v4.sha256_file(scores_path),
                "checkpoint_fingerprint": fingerprint,
                "model": package_v4.preference_v4.MODEL,
                "model_revision": package_v4.preference_v4.MODEL_REVISION,
                "adapter": {
                    "checkpoint_fingerprint": fingerprint,
                    "file_sha256": checkpoint_hashes,
                },
                "data": {
                    "strict_pair_file_sha256": package_v4.PAIR_SHA256,
                    "expected_strict_pair_file_sha256": package_v4.PAIR_SHA256,
                    "strict_pair_file_hash_matches_expected": True,
                    "split_audit": {"all_rows_match_expected_split": True},
                },
                "scoring_contract": {
                    "completion_eos_included": True,
                    "runtime_truncation_count": 0,
                    "dropout": 0.0,
                    "quantization": "NF4 double-quantized base",
                    "bf16": True,
                },
                "environment": {
                    "python": "3.11.9",
                    "torch": "2.7.1+cu128",
                    "transformers": "4.52.4",
                    "peft": "0.15.2",
                    "bitsandbytes": "0.46.0",
                    "datasets": "3.6.0",
                    "accelerate": "1.7.0",
                    "trl": "0.18.2",
                    "cuda_runtime": "12.8",
                    "bf16_supported": True,
                    "deterministic_algorithms_enabled": True,
                    "cublas_workspace_config": ":4096:8",
                },
                "outputs": {
                    "metrics_sha256": package_v4.sha256_file(metrics_path),
                    "pair_scores_sha256": package_v4.sha256_file(scores_path),
                    "pair_scores_rows": package_v4.STRICT_PAIR_EXAMPLES,
                },
            }
            manifest_path = score_dir / "score_manifest.json"
            write_json(manifest_path, manifest)
            result = package_v4.validate_score(
                score_dir,
                fingerprint,
                checkpoint_hashes,
                "4" * 40,
            )
            self.assertEqual(
                result["score_manifest_sha256"],
                package_v4.sha256_file(manifest_path),
            )

            manifest["limited"] = True
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "limited"):
                package_v4.validate_score(
                    score_dir,
                    fingerprint,
                    checkpoint_hashes,
                    "4" * 40,
                )

    def test_comparison_binding_rejects_a_stale_arm_hash(self):
        evaluations = {}
        training = {}
        scores = {}
        arms = {}
        input_arms = {}
        for index, arm in enumerate(package_v4.ARM_ORDER):
            prefix = str(index + 1)
            evaluations[arm] = {
                "metrics_sha256": prefix * 64,
                "contract_sha256": str(index + 2) * 64,
                "predictions_sha256": str(index + 3) * 64,
                "checkpoint_fingerprint": str(index + 4) * 64,
            }
            arms[arm] = {
                "checkpoint_fingerprint": evaluations[arm][
                    "checkpoint_fingerprint"
                ]
            }
            input_arms[arm] = dict(evaluations[arm])
            if arm != "standard_v3":
                training[arm] = {
                    "run_manifest_sha256": str(index + 5) * 64,
                }
                input_arms[arm]["run_manifest_sha256"] = training[arm][
                    "run_manifest_sha256"
                ]
                scores[arm] = {
                    "metrics_sha256": str(index + 6) * 64,
                    "pair_scores_sha256": str(index + 7) * 64,
                    "score_manifest_sha256": str(index + 8) * 64,
                }
        standard = {
            "file_sha256": {
                "metrics.json": evaluations["standard_v3"]["metrics_sha256"],
                "metrics.contract.json": evaluations["standard_v3"][
                    "contract_sha256"
                ],
                "metrics.predictions.jsonl": evaluations["standard_v3"][
                    "predictions_sha256"
                ],
            }
        }
        score_arms = {
            "standard_v3": {"status": "not_available"},
            **{
                arm: {"status": "complete", **scores[arm]}
                for arm in package_v4.TRAINED_ARMS
            },
        }
        comparison = {
            "valid": True,
            "protocol": package_v4.ANALYSIS_PROTOCOL,
            "examples": package_v4.FORMAL_EXAMPLES,
            "frozen_test_sha256": package_v4.TEST_SHA256,
            "bootstrap": {
                "unit": "task_key",
                "paired": True,
                "samples": package_v4.aggregate_v4.BOOTSTRAP_SAMPLES,
            },
            "cluster_sign_flip": {
                "samples": package_v4.aggregate_v4.SIGN_FLIP_SAMPLES,
            },
            "arms": arms,
            "paired_comparisons": {
                name: {} for name in package_v4.CONTRASTS
            },
            "input_artifacts": {
                "test_outcomes": {
                    "sha256": package_v4.prepare_v4.EXPECTED_OUTPUT_SHA256[
                        "test_outcomes_sha256"
                    ]
                },
                "test_preference_pairs": {
                    "sha256": package_v4.prepare_v4.EXPECTED_OUTPUT_SHA256[
                        "test_preference_pairs_sha256"
                    ]
                },
                "arms": input_arms,
            },
            "pair_scoring": {
                "strict_pair_examples": package_v4.STRICT_PAIR_EXAMPLES,
                "arms": score_arms,
                "dpo_minus_continued_sft": {"status": "complete"},
            },
        }
        package_v4.require_report_binding(
            comparison,
            training,
            evaluations,
            scores,
            standard,
            {"build_summary_sha256": "9" * 64},
        )
        comparison["input_artifacts"]["arms"]["dpo"]["metrics_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "comparison dpo metrics_sha256"):
            package_v4.require_report_binding(
                comparison,
                training,
                evaluations,
                scores,
                standard,
                {"build_summary_sha256": "9" * 64},
            )

    def test_release_source_discovery_includes_package_and_protocol_test(self):
        relative = {
            path.relative_to(package_v4.ROOT).as_posix()
            for path in package_v4.discovered_v4_source_paths()
        }
        self.assertIn("scripts/package_qlora_v4_results.py", relative)
        self.assertIn("tests/test_v4_protocol.py", relative)
        self.assertIn("tests/test_v4_package.py", relative)
        self.assertIn("configs/qlora_v4.yaml", relative)

    def test_source_gate_rejects_an_untracked_v4_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            required = (
                "README.md",
                "BASELINE_V4_HANDOFF.md",
                "V4_RTX5060_AGENT_PROMPT.md",
                "requirements-gpu-v4.txt",
                "configs/qlora_v4.yaml",
                "scripts/package_qlora_v4_results.py",
                "tests/test_v4_protocol.py",
            )
            for relative in required:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=V4 Test",
                    "-c",
                    "user.email=v4@example.invalid",
                    "commit",
                    "-qm",
                    "frozen",
                ],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "tag", package_v4.FROZEN_TAG],
                cwd=repo,
                check=True,
            )
            with patch.object(package_v4, "ROOT", repo):
                commit, hashes = package_v4.source_commit()
                self.assertEqual(len(commit), 40)
                self.assertIn(
                    "scripts/package_qlora_v4_results.py",
                    hashes,
                )
                (repo / "scripts" / "rogue_v4.py").write_text(
                    "print('untracked')\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "not tracked"):
                    package_v4.source_commit()


if __name__ == "__main__":
    unittest.main()
