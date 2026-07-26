import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v5_checkpoint_registry.py"
SPEC = importlib.util.spec_from_file_location("build_v5_checkpoint_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
EVAL_SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_eval.py"
EVAL_SPEC = importlib.util.spec_from_file_location(
    "run_v5_sft_causal_eval_for_registry_test", EVAL_SCRIPT
)
EVAL_MODULE = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVAL_MODULE)


class V5CheckpointRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_commit = "1" * 40
        self.base_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
        self.arm_dirs = {}
        for index, arm in enumerate(MODULE.ARMS):
            self.arm_dirs[arm] = self.make_run(arm, index)

    def tearDown(self):
        self.temp.cleanup()

    def make_run(
        self,
        directory_arm,
        index,
        *,
        provenance_profile="legacy",
        **overrides,
    ):
        run_dir = self.root / f"{directory_arm}-{index}"
        checkpoint = run_dir / "checkpoint_final"
        checkpoint.mkdir(parents=True)
        adapter = checkpoint / "adapter_model.safetensors"
        adapter.write_bytes(f"adapter:{directory_arm}:{index}".encode())
        adapter_config = checkpoint / "adapter_config.json"
        adapter_config.write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "base_model_name_or_path": MODULE.BASE_MODEL,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                    "target_modules": sorted(
                        MODULE.SCREEN_LORA_TARGET_MODULES
                    ),
                }
            ),
            encoding="utf-8",
        )
        data_provenance = {
            "data_audit_sha256": "a" * 64,
            "data_hashes_sha256": "b" * 64,
            "official_test_used": False,
            "official_test_sealed": True,
            "dynamic_audits": {
                "generation": {
                    "protocol": MODULE.DYNAMIC_AUDIT_PROTOCOL,
                    "sha256": "c" * 64,
                    "manifest_sha256": "d" * 64,
                    "split_manifest_sha256": "e" * 64,
                    "source_split": "derived_inner_train",
                    "verified_injections": (
                        78
                        if provenance_profile
                        in {"v5_3", "v5_3_12h_screen"}
                        else 83
                    ),
                    "official_test_used": False,
                    "official_test_sealed": True,
                },
                "validation": {
                    "protocol": MODULE.DYNAMIC_AUDIT_PROTOCOL,
                    "sha256": "f" * 64,
                    "manifest_sha256": "1" * 64,
                    "split_manifest_sha256": "e" * 64,
                    "source_split": "derived_validation",
                    "verified_injections": 21,
                    "official_test_used": False,
                    "official_test_sealed": True,
                },
            },
        }
        if provenance_profile == "v5_3":
            data_provenance.update(
                {
                    "design_version": "5.3",
                    "design_provenance": {
                        "design_version": "5.3",
                        "design_protocol": (
                            "v5_3_task_level_cross_seed_sft_screen"
                        ),
                        "effective_generation_tasks": 78,
                        "validation_tasks": 21,
                    },
                }
            )
        elif provenance_profile == "v5_3_12h_screen":
            data_provenance.update(
                {
                    "design_version": MODULE.V5_3_12H_DESIGN_VERSION,
                    "design_provenance": {
                        "design_version": MODULE.V5_3_12H_DESIGN_VERSION,
                        "design_protocol": MODULE.V5_3_12H_DESIGN_PROTOCOL,
                        "screen_protocol": MODULE.V5_3_12H_SCREEN_PROTOCOL,
                        "screen_tasks": 24,
                        "screen_rollouts": 288,
                        "validation_tasks": 21,
                        "shared_outcome_free_protocol_inputs_read_only": True,
                        "prior_v5_3_rollout_checkpoint_metric_decision_bytes_reused": False,
                        "screen_outputs_may_enter_formal_v5_3": False,
                    },
                }
            )
        target_ratio = MODULE.SCREEN_TARGET_RECOVERY_ROW_RATIOS[
            directory_arm
        ]
        recovery_rows = int(512 * target_ratio)
        failed_labels = 8 if directory_arm == "failure_raw" else 0
        manifest = {
            "protocol": MODULE.TRAIN_PROTOCOL,
            "source_commit": self.source_commit,
            "arm": directory_arm,
            "mode": "formal",
            "model": MODULE.BASE_MODEL,
            "model_revision": self.base_revision,
            "objective": "message_masked_causal_language_model_cross_entropy",
            "quantization": {
                "bits": 4,
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": sorted(
                    MODULE.SCREEN_LORA_TARGET_MODULES
                ),
            },
            "seed": MODULE.V5_3_12H_TRAINING_SEED,
            "max_sequence_tokens": 8192,
            "truncation": False,
            "formal_steps": 64,
            "formal_batch_size": 1,
            "formal_grad_accum": 8,
            "effective_steps": 64,
            "effective_grad_accum": 8,
            "learning_rate": 1.0e-4,
            "effective_rows": 512,
            "effective_validation_rows": 0,
            "validation_disabled_reason": "fixed_steps_exploratory",
            "formal_schedule": {
                "rows": 512,
                "nonpad_tokens": 10000,
                "supervised_tokens": 2000,
                "recovery_rows": recovery_rows,
                "realized_recovery_row_ratio": target_ratio,
                "expected_recovery_row_ratio": target_ratio,
                "recovery_mixture_basis": (
                    "row_mean_microbatch_equal_weight"
                ),
                "realized_recovery_supervised_token_ratio": (
                    0.37 if directory_arm == "repair_50" else target_ratio
                ),
                "expected_recovery_supervised_token_ratio": None,
                "failed_action_label_messages": failed_labels,
                "max_sequence_tokens": 1024,
            },
            "fit_partition": {
                "validation_loss_source_examples": 0,
                "overlap": 0,
                "validation_disabled_reason": "fixed_steps_exploratory",
            },
            "loss_audit": {
                "finite": True,
                "numeric_values_checked": 10,
                "loss_values_checked": 2,
                "grad_norm_values_checked": 2,
                "validation_loss_values_checked": 0,
                "final_train_loss": 0.5,
                "final_validation_loss": None,
            },
            "held_out_test_accessed": False,
            "data_provenance": data_provenance,
            "checkpoint": {
                "path": str(checkpoint),
                "file_sha256": {
                    "adapter_model.safetensors": MODULE.sha256_file(adapter),
                    "adapter_config.json": MODULE.sha256_file(adapter_config),
                },
            },
        }
        manifest.update(overrides)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return run_dir

    def build(self, provenance_profile=None):
        return MODULE.build_registry(
            source_commit=self.source_commit,
            base_revision=self.base_revision,
            arm_dirs=self.arm_dirs,
            provenance_profile=provenance_profile,
        )

    def test_builds_exact_five_unique_entries_and_eval_accepts_it(self):
        registry = self.build()
        self.assertEqual(registry["protocol"], MODULE.PROTOCOL)
        self.assertEqual(registry["provenance_profile"], "legacy")
        self.assertEqual(
            set(registry["entries"]), {"base_model", *MODULE.ARMS}
        )
        model_ids = [
            entry["model_id"] for entry in registry["entries"].values()
        ]
        self.assertEqual(len(model_ids), len(set(model_ids)))
        self.assertEqual(
            {
                arm: entry["model_id"]
                for arm, entry in registry["entries"].items()
            },
            MODULE.MODEL_IDS,
        )
        self.assertTrue(all(alias.startswith("openai/") for alias in model_ids))
        self.assertIsNone(
            registry["entries"]["base_model"]["adapter_sha256"]
        )
        self.assertIsNone(
            registry["entries"]["base_model"]["adapter_config_sha256"]
        )
        self.assertEqual(
            registry["training_data_provenance"]["data_audit_sha256"],
            "a" * 64,
        )
        output = self.root / "checkpoint_registry.json"
        MODULE.atomic_write_registry(output, registry)
        loaded = EVAL_MODULE.load_checkpoint_registry(output)
        self.assertEqual(loaded, registry)
        local_identity = EVAL_MODULE.validate_local_adapter_identity(
            loaded,
            {
                arm: run_dir / "checkpoint_final"
                for arm, run_dir in self.arm_dirs.items()
            },
        )
        self.assertEqual(set(local_identity), set(MODULE.ARMS))
        for arm in MODULE.ARMS:
            self.assertEqual(
                local_identity[arm]["adapter_sha256"],
                registry["entries"][arm]["adapter_sha256"],
            )
            self.assertEqual(
                local_identity[arm]["adapter_config_sha256"],
                registry["entries"][arm]["adapter_config_sha256"],
            )
        self.assertRegex(MODULE.sha256_file(output), r"^[0-9a-f]{64}$")
        self.assertFalse(list(self.root.glob(".checkpoint_registry.json.*.tmp")))

    def test_v5_3_profile_is_safely_inferred_and_uses_frozen_aliases(self):
        self.arm_dirs = {
            arm: self.make_run(
                arm,
                f"v5-3-{index}",
                provenance_profile="v5_3",
            )
            for index, arm in enumerate(MODULE.ARMS)
        }
        registry = self.build()
        self.assertEqual(registry["provenance_profile"], "v5_3")
        self.assertEqual(
            {
                arm: entry["model_id"]
                for arm, entry in registry["entries"].items()
            },
            MODULE.V5_3_MODEL_IDS,
        )
        self.assertEqual(
            registry["training_data_provenance"]["dynamic_audits"][
                "generation"
            ]["verified_injections"],
            78,
        )
        output = self.root / "v5_3_checkpoint_registry.json"
        MODULE.atomic_write_registry(output, registry)
        loaded = EVAL_MODULE.load_checkpoint_registry(
            output,
            expected_profile="v5_3",
        )
        self.assertEqual(loaded, registry)

    def test_12h_screen_rejects_fixed_training_contract_tampering(self):
        serial = [0]

        def make_screen_runs():
            serial[0] += 1
            return {
                arm: self.make_run(
                    arm,
                    f"screen-{serial[0]}-{index}",
                    provenance_profile="v5_3_12h_screen",
                )
                for index, arm in enumerate(MODULE.ARMS)
            }

        self.arm_dirs = make_screen_runs()
        registry = self.build()
        self.assertEqual(
            registry["provenance_profile"], "v5_3_12h_screen"
        )
        cases = {
            "effective_steps": 63,
            "formal_grad_accum": 4,
            "effective_rows": 511,
            "seed": 1,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                self.arm_dirs = make_screen_runs()
                path = (
                    self.arm_dirs["repair_50"] / "run_manifest.json"
                )
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest[field] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fixed-64"):
                    self.build()

        self.arm_dirs = make_screen_runs()
        path = self.arm_dirs["repair_50"] / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["loss_audit"]["final_train_loss"] = float("nan")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "finite loss"):
            self.build()

        self.arm_dirs = make_screen_runs()
        path = self.arm_dirs["repair_50"] / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["formal_schedule"]["recovery_rows"] = 255
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "row-weighted"):
            self.build()

    def test_profiles_reject_cross_version_counts_aliases_and_arbitrary_names(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported provenance"):
            self.build(provenance_profile="v5_4")

        self.arm_dirs = {
            arm: self.make_run(
                arm,
                f"v5-3-mismatch-{index}",
                provenance_profile="v5_3",
            )
            for index, arm in enumerate(MODULE.ARMS)
        }
        with self.assertRaisesRegex(RuntimeError, "legacy provenance"):
            self.build(provenance_profile="legacy")
        registry = self.build(provenance_profile="v5_3")
        registry["entries"]["repair_50"]["model_id"] = "openai/arbitrary"
        output = self.root / "alias_drift_registry.json"
        output.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "model_id drift"):
            EVAL_MODULE.load_checkpoint_registry(output)

    def test_cli_profile_is_optional_but_only_registered_values_are_accepted(self):
        required = [
            str(SCRIPT),
            "--source-commit",
            self.source_commit,
            "--base-revision",
            self.base_revision,
            *[
                value
                for arm, path in self.arm_dirs.items()
                for value in ("--arm", f"{arm}={path}")
            ],
            "--output",
            str(self.root / "registry.json"),
        ]
        with mock.patch("sys.argv", required):
            self.assertIsNone(MODULE.parse_args().provenance_profile)
        with mock.patch(
            "sys.argv",
            [*required, "--provenance-profile", "v5_3"],
        ):
            self.assertEqual(
                MODULE.parse_args().provenance_profile,
                "v5_3",
            )

    def test_rejects_smoke_wrong_arm_commit_revision_and_protocol(self):
        cases = {
            "smoke": {"mode": "smoke"},
            "wrong_arm": {"arm": "repair_100"},
            "wrong_commit": {"source_commit": "2" * 40},
            "wrong_revision": {"model_revision": "3" * 40},
            "wrong_protocol": {"protocol": "not-stage1"},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label):
                arm = "perfect_success"
                self.arm_dirs[arm] = self.make_run(
                    arm, f"invalid-{label}", **overrides
                )
                with self.assertRaises(RuntimeError):
                    self.build()
                self.arm_dirs[arm] = self.make_run(arm, f"valid-{label}")

    def test_rejects_missing_empty_or_unbound_adapter(self):
        arm = "failure_raw"
        adapter = self.arm_dirs[arm] / "checkpoint_final" / "adapter_model.safetensors"
        adapter.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
            self.build()

        self.arm_dirs[arm] = self.make_run(arm, "empty")
        adapter = self.arm_dirs[arm] / "checkpoint_final" / "adapter_model.safetensors"
        adapter.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
            self.build()

        self.arm_dirs[arm] = self.make_run(arm, "hash-drift")
        manifest_path = self.arm_dirs[arm] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint"]["file_sha256"]["adapter_model.safetensors"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "adapter SHA"):
            self.build()

        self.arm_dirs[arm] = self.make_run(arm, "missing-config")
        (
            self.arm_dirs[arm]
            / "checkpoint_final"
            / "adapter_config.json"
        ).unlink()
        with self.assertRaisesRegex(RuntimeError, "adapter_config.json"):
            self.build()

        self.arm_dirs[arm] = self.make_run(arm, "config-hash-drift")
        manifest_path = self.arm_dirs[arm] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint"]["file_sha256"]["adapter_config.json"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "adapter config SHA"):
            self.build()

    def test_rejects_duplicate_bindings_run_dirs_and_adapter_bytes(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate arm"):
            MODULE.parse_arm_bindings(
                [
                    f"perfect_success={self.arm_dirs['perfect_success']}",
                    f"perfect_success={self.arm_dirs['failure_raw']}",
                ]
            )
        duplicate_dirs = dict(self.arm_dirs)
        duplicate_dirs["repair_100"] = duplicate_dirs["repair_50"]
        with self.assertRaisesRegex(RuntimeError, "duplicate run directories"):
            MODULE.build_registry(
                source_commit=self.source_commit,
                base_revision=self.base_revision,
                arm_dirs=duplicate_dirs,
            )
        source = (
            self.arm_dirs["perfect_success"]
            / "checkpoint_final"
            / "adapter_model.safetensors"
        )
        target = (
            self.arm_dirs["failure_raw"]
            / "checkpoint_final"
            / "adapter_model.safetensors"
        )
        target.write_bytes(source.read_bytes())
        manifest_path = self.arm_dirs["failure_raw"] / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint"]["file_sha256"][
            "adapter_model.safetensors"
        ] = MODULE.sha256_file(target)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "duplicate adapter"):
            self.build()

    def test_rejects_cross_arm_data_provenance_drift(self):
        arm = "repair_50"
        run_dir = self.arm_dirs[arm]
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["data_provenance"]["data_audit_sha256"] = "9" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "do not share one data"):
            self.build()

    def test_atomic_writer_refuses_overwrite(self):
        output = self.root / "checkpoint_registry.json"
        original = b"immutable\n"
        output.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            MODULE.atomic_write_registry(output, self.build())
        self.assertEqual(output.read_bytes(), original)

    def test_invalid_or_incomplete_cli_bindings_fail(self):
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            MODULE.parse_arm_bindings(
                [f"perfect_success={self.arm_dirs['perfect_success']}"]
            )
        with self.assertRaisesRegex(RuntimeError, "ARM=RUN_DIR"):
            MODULE.parse_arm_bindings(["perfect_success"])
        with self.assertRaisesRegex(RuntimeError, "source commit"):
            MODULE.build_registry(
                source_commit="short",
                base_revision=self.base_revision,
                arm_dirs=self.arm_dirs,
            )


if __name__ == "__main__":
    unittest.main()
