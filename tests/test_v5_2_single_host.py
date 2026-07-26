import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_2_single_host.py"
SPEC = importlib.util.spec_from_file_location("run_v5_2_single_host", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PREP_SCRIPT = ROOT / "scripts" / "prepare_v5_2_sft_causal.py"
PREP_SPEC = importlib.util.spec_from_file_location(
    "prepare_v5_2_sft_causal_for_controller_test", PREP_SCRIPT
)
PREP_MODULE = importlib.util.module_from_spec(PREP_SPEC)
assert PREP_SPEC.loader is not None
PREP_SPEC.loader.exec_module(PREP_MODULE)


def args_for(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vllm=Path("/workspace/venvs/v5-2-serve/bin/vllm"),
        serve_python=Path("/workspace/venvs/v5-2-serve/bin/python"),
        train_python=Path("/workspace/venvs/v5-2-train/bin/python"),
        tau2_root=root / "tau2",
        protocol_root=root / "protocol",
        raw_root=root / "raw",
        processed_root=root / "processed",
        results_root=root / "results",
        health_timeout=30,
    )


class V52SingleHostTests(unittest.TestCase):
    def test_vllm_command_freezes_32768_and_one_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            command = MODULE.vllm_base_command(
                args,
                MODULE.STUDENT_MODEL,
                MODULE.STUDENT_REVISION,
                dtype="bfloat16",
            )
        self.assertEqual(command.count("--max-model-len"), 1)
        index = command.index("--max-model-len")
        self.assertEqual(command[index + 1], "32768")
        self.assertEqual(command.count("--dtype"), 1)
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")

    def test_generation_completion_requires_three_shards_and_twelve_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            args.raw_root.mkdir()
            for shard in range(3):
                result_name = (
                    f"retail_clean.shard-{shard:03d}-of-003.json"
                )
                result_path = args.raw_root / result_name
                result_path.write_text(f"shard-{shard}", encoding="utf-8")
                payload = {
                    "status": "COMPLETE",
                    "shard_index": shard,
                    "num_shards": 3,
                    "num_trials": 12,
                    "official_test_used": False,
                    "result_sha256": {
                        result_name: MODULE.sha256_file(result_path)
                    },
                }
                path = (
                    args.raw_root
                    / f"run_contract.shard-{shard:03d}-of-003.json"
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(MODULE.generation_complete(args))
            path.write_text(
                json.dumps({**payload, "num_trials": 3}), encoding="utf-8"
            )
            self.assertFalse(MODULE.generation_complete(args))

    def test_training_completion_uses_held_out_test_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint_final"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            manifest = {
                "protocol": "v5_stage1_message_masked_sft_7b",
                "mode": "formal",
                "held_out_test_accessed": False,
            }
            (output / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(
                MODULE.training_run_complete(output, mode="formal")
            )
            manifest["held_out_test_accessed"] = True
            (output / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(
                MODULE.training_run_complete(output, mode="formal")
            )

    def test_evaluation_command_uses_one_host_shard_and_all_adapter_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            command = MODULE.evaluation_command(
                args, arm="repair_50", shard=2
            )
        self.assertEqual(command[command.index("--shard-index") + 1], "2")
        self.assertEqual(command[command.index("--num-shards") + 1], "4")
        self.assertEqual(
            command[command.index("--agent-api-base") + 1],
            "http://127.0.0.1:8102/v1",
        )
        self.assertEqual(
            command[command.index("--agent-model") + 1],
            "openai/v5-repair-50",
        )
        self.assertEqual(command.count("--adapter-dir"), 4)
        self.assertNotIn("official_test", " ".join(command))

    def test_frozen_topology_uses_gpu_zero_for_user_and_three_teacher_shards(self):
        self.assertEqual(MODULE.GENERATION_SHARDS, 3)
        self.assertEqual(MODULE.GENERATION_ATTEMPTS_PER_CONDITION, 12)
        self.assertEqual(MODULE.EVALUATION_SHARDS, 4)
        self.assertEqual(len(MODULE.ARMS), 4)
        self.assertEqual(
            PREP_MODULE.EXPECTED_TEACHER_API_BASE_BY_SHARD,
            {
                0: "http://127.0.0.1:8011/v1",
                1: "http://127.0.0.1:8012/v1",
                2: "http://127.0.0.1:8013/v1",
            },
        )


if __name__ == "__main__":
    unittest.main()
