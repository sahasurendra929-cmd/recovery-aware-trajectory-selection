import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    fault_protocol,
    multifault_row,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_generate.py"
SPEC = importlib.util.spec_from_file_location(
    "run_v5_sft_causal_generate_screen_tests",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def screen_task_ids():
    return sorted(MODULE.SCREEN_TASK_IDS)


def screen_rows():
    return [
        multifault_row(
            *identity.split(":", 1),
            source_split="derived_inner_train",
        )
        for identity in screen_task_ids()
    ]


def screen_split():
    selected = {
        domain: [
            identity.split(":", 1)[1]
            for identity in screen_task_ids()
            if identity.startswith(f"{domain}:")
        ]
        for domain in ("retail", "airline")
    }
    retail_fill = [
        f"retail-fill-{index}"
        for index in range(59 - len(selected["retail"]))
    ]
    airline_fill = [
        f"airline-fill-{index}"
        for index in range(24 - len(selected["airline"]))
    ]
    return {
        "protocol": MODULE.SPLIT_PROTOCOL,
        "guarantees": {
            "validation_derived_from_official_train_only": True,
            "official_test_task_content_exported": False,
        },
        "domains": {
            "retail": {
                "inner_train_ids": selected["retail"] + retail_fill,
                "validation_ids": [
                    f"retail-validation-{index}" for index in range(15)
                ],
                "sealed_test_ids": [
                    f"retail-test-{index}" for index in range(40)
                ],
            },
            "airline": {
                "inner_train_ids": selected["airline"] + airline_fill,
                "validation_ids": [
                    f"airline-validation-{index}" for index in range(6)
                ],
                "sealed_test_ids": [
                    f"airline-test-{index}" for index in range(20)
                ],
            },
        },
    }


def make_screen_manifest():
    rows = screen_rows()
    excluded = [
        "retail:retail-fill-0",
        "retail:retail-fill-1",
        "retail:retail-fill-2",
        "airline:airline-fill-0",
        "airline:airline-fill-1",
    ]
    source_sha = "a" * 64
    payload = {
        "protocol": MODULE.SCREEN_MANIFEST_PROTOCOL,
        "screen_protocol": MODULE.SCREEN_PROTOCOL,
        "design_version": "5.3-12h-screen",
        "formal_v5_3_data": False,
        "formal_data": False,
        "screen_outputs_may_enter_formal": False,
        "screen_outputs_may_enter_formal_v5_3": False,
        "official_test_used": False,
        "official_test_sealed": True,
        "source_split": "derived_inner_train",
        "base_seed": MODULE.SCREEN_SEED,
        "trial_seeds": list(MODULE.SCREEN_TRIAL_SEEDS),
        "attempts_per_task_per_condition": MODULE.SCREEN_NUM_TRIALS,
        "conditions": ["clean", "error"],
        "expected_rollouts": 288,
        "formal_v5_3_seed_or_bytes_reused": False,
        "replacement_or_rescue_attempts": False,
        "planned_task_ids": screen_task_ids(),
        "paired_task_count": 24,
        "source_generation_manifest_sha256": source_sha,
        "generation_manifest_sha256": source_sha,
        "tau2_commit": MODULE.SCREEN_TAU2_COMMIT,
        "split_manifest_sha256": MODULE.SCREEN_SPLIT_SHA256,
        "source_files": deepcopy(SOURCE_FILES),
        "fault_protocol": fault_protocol(),
        "gt_compatibility_filter": {
            "protocol": MODULE.GT_FILTER_PROTOCOL,
            "policy": "exclude_before_sharding",
            "teacher_mode": "ground_truth",
            "source_task_count": 83,
            "included_task_count": 78,
            "excluded_task_ids": excluded,
            "exclusion_reason": "no_expected_tool_actions",
            "selection_uses_rollouts_rewards_validation_or_test": False,
            "official_test_used": False,
        },
        "generation": {
            "teacher": {
                "model": MODULE.SCREEN_TEACHER_MODEL,
                "revision": MODULE.SCREEN_TEACHER_REVISION,
            },
            "user": {
                "model": MODULE.SCREEN_USER_MODEL,
                "revision": MODULE.SCREEN_USER_REVISION,
            },
            "judge": {
                "model": MODULE.SCREEN_USER_MODEL,
                "revision": MODULE.SCREEN_USER_REVISION,
            },
            "teacher_mode": "ground_truth",
            "temperature": MODULE.SCREEN_TEMPERATURE,
            "top_p": MODULE.SCREEN_TOP_P,
            "max_model_len": MODULE.SCREEN_MAX_MODEL_LEN,
            "max_tokens": MODULE.SCREEN_MAX_TOKENS,
            "num_shards": MODULE.SCREEN_NUM_SHARDS,
            "num_trials": MODULE.SCREEN_NUM_TRIALS,
            "base_seed": MODULE.SCREEN_SEED,
            "trial_seeds": list(MODULE.SCREEN_TRIAL_SEEDS),
        },
        "rows": rows,
    }
    payload["canonical_sha256"] = MODULE.canonical_sha256(payload)
    return payload


def write_host_preflight(path):
    payload = {
        "status": "PASS",
        "protocol": "v5_3_single_host_preflight",
        "environments": {
            "train": {
                "cuda": MODULE.SCREEN_CUDA_VERSION,
                "cuda_available": True,
            }
        },
        "official_test_used": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def runtime_capacity_evidence(request_id):
    return {
        "status": "PASS",
        "probe_kind": MODULE.SCREEN_RUNTIME_CAPACITY_PROBE_KIND,
        "request_id": request_id,
        "endpoint": "chat/completions",
        "finish_reason": "stop",
        "max_tokens": MODULE.SCREEN_MAX_TOKENS,
        "prompt_tokens": 27_500,
        "completion_tokens": 5,
        "total_tokens": 27_505,
        "tool_call_count": 0,
        "content_utf8_bytes": 2,
        "content_sha256": "a" * 64,
        "response_sha256": "b" * 64,
    }


def runtime_tool_evidence(request_id):
    return {
        "status": "PASS",
        "probe_kind": MODULE.SCREEN_RUNTIME_TOOL_PROBE_KIND,
        "request_id": request_id,
        "endpoint": "chat/completions",
        "tool_choice_mode": MODULE.SCREEN_RUNTIME_TOOL_CHOICE_MODE,
        "finish_reason": MODULE.SCREEN_RUNTIME_FINISH_REASON,
        "parallel_tool_calls": False,
        "max_tokens": MODULE.SCREEN_RUNTIME_TOOL_MAX_TOKENS,
        "prompt_tokens": 83,
        "completion_tokens": 23,
        "total_tokens": 106,
        "tool_call_count": 1,
        "tool_name": MODULE.SCREEN_RUNTIME_TOOL_NAME,
        "request_schema_sha256": MODULE.canonical_sha256(
            MODULE.screen_runtime_probe_tool(request_id)
        ),
        "tool_arguments_sha256": MODULE.canonical_sha256(
            {
                "marker": MODULE.SCREEN_RUNTIME_MARKER,
                "request_id": request_id,
            }
        ),
        "response_sha256": "c" * 64,
    }


def make_runtime_receipt(source_commit, host_path):
    inventory = [
        {
            "index": index,
            "name": MODULE.SCREEN_GPU_MODEL,
            "uuid": f"GPU-{index}",
            "driver": "575.57.08",
        }
        for index in range(4)
    ]
    specifications = {
        "teacher": (
            MODULE.SCREEN_TEACHER_MODEL,
            MODULE.SCREEN_TEACHER_REVISION,
            [1, 2],
            2,
        ),
        "user_and_judge": (
            MODULE.SCREEN_USER_MODEL,
            MODULE.SCREEN_USER_REVISION,
            [0],
            1,
        ),
    }
    roles = {}
    services = {}
    for role, (model, revision, indices, tensor_parallel_size) in (
        specifications.items()
    ):
        long_request_id = f"screen-{role}-long"
        roles[role] = {
            "model": model,
            "revision": revision,
            "api_base": (
                "http://127.0.0.1:8011/v1"
                if role == "teacher"
                else "http://127.0.0.1:8001/v1"
            ),
            "tensor_parallel_size": tensor_parallel_size,
            "long_context_probe": runtime_capacity_evidence(
                long_request_id
            ),
            "tool_interface_probe": runtime_tool_evidence(
                f"screen-{role}-tool-interface"
            ),
            "concurrency_probe": {
                "status": "PASS",
                "probe_kind": MODULE.SCREEN_RUNTIME_CAPACITY_PROBE_KIND,
                "requests": 3,
                "client_barrier_size": 3,
                "max_tokens": MODULE.SCREEN_MAX_TOKENS,
                "results": [
                    runtime_capacity_evidence(
                        f"screen-{role}-concurrent-{index}"
                    )
                    for index in range(3)
                ],
            },
        }
        services[role] = {
            "visible_gpu_indices": indices,
            "gpu_identity": [inventory[index] for index in indices],
            "model_snapshot": {
                "model": model,
                "requested_revision": revision,
                "resolved_revision": revision,
            },
        }
    payload = {
        "protocol": MODULE.SCREEN_RUNTIME_EVIDENCE_PROTOCOL,
        "status": "PASS",
        "phase": "screen",
        "source_commit": source_commit,
        "max_model_len": MODULE.SCREEN_MAX_MODEL_LEN,
        "expected_gpu_model": MODULE.SCREEN_GPU_MODEL,
        "official_test_used": False,
        "host_preflight": {
            "path": str(host_path.resolve()),
            "sha256": MODULE.sha256_file(host_path),
        },
        "gpu_inventory": inventory,
        "roles": roles,
        "invocation": {"services": services},
    }
    payload["canonical_receipt_sha256"] = MODULE.canonical_sha256(payload)
    return payload


def screen_args(runtime_evidence):
    return SimpleNamespace(
        output_dir=runtime_evidence.parent / "raw",
        num_shards=3,
        shard_index=1,
        expected_source_commit="b" * 40,
        num_trials=MODULE.SCREEN_NUM_TRIALS,
        teacher_model=MODULE.SCREEN_TEACHER_MODEL,
        teacher_revision=MODULE.SCREEN_TEACHER_REVISION,
        teacher_api_base="http://127.0.0.1:8011/v1",
        teacher_mode="ground_truth",
        user_model=MODULE.SCREEN_USER_MODEL,
        user_revision=MODULE.SCREEN_USER_REVISION,
        user_api_base="http://127.0.0.1:8001/v1",
        judge_model=MODULE.SCREEN_USER_MODEL,
        judge_revision=MODULE.SCREEN_USER_REVISION,
        judge_api_base="http://127.0.0.1:8001/v1",
        max_tokens=MODULE.SCREEN_MAX_TOKENS,
        max_model_len=MODULE.SCREEN_MAX_MODEL_LEN,
        max_steps=60,
        timeout=900.0,
        seed=MODULE.SCREEN_SEED,
        temperature=MODULE.SCREEN_TEMPERATURE,
        top_p=MODULE.SCREEN_TOP_P,
        runtime_evidence=runtime_evidence,
    )


class V5312HourGenerationTests(unittest.TestCase):
    def test_valid_screen_manifest_sampling_runtime_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "screen_manifest.json"
            split_path = root / "split_manifest.json"
            runtime_path = root / "runtime.json"
            host_path = root / "preflight.json"
            manifest = make_screen_manifest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            split_path.write_text(json.dumps(screen_split()), encoding="utf-8")
            write_host_preflight(host_path)
            runtime_path.write_text(
                json.dumps(make_runtime_receipt("b" * 40, host_path)),
                encoding="utf-8",
            )

            loaded = MODULE.load_inner_train_manifest(
                manifest_path,
                split_manifest=MODULE.load_split_manifest(split_path),
                split_manifest_sha256=MODULE.SCREEN_SPLIT_SHA256,
            )
            sampling = MODULE.validate_sampling_contract(
                manifest_protocol=MODULE.SCREEN_MANIFEST_PROTOCOL,
                temperature=MODULE.SCREEN_TEMPERATURE,
                top_p=MODULE.SCREEN_TOP_P,
                num_trials=MODULE.SCREEN_NUM_TRIALS,
                seed=MODULE.SCREEN_SEED,
            )
            self.assertEqual(
                sampling["derived_trial_seeds"],
                MODULE.SCREEN_TRIAL_SEEDS,
            )
            args = screen_args(runtime_path)
            runtime = MODULE.validate_screen_execution_contract(args, loaded)
            self.assertEqual(runtime["max_model_len"], 32768)

            contract_path = MODULE.write_contract(
                args,
                manifest_path.resolve(),
                split_path.resolve(),
                MODULE.shard_rows(loaded["rows"], 1, 3),
                dynamic_audit_path=root / "dynamic.json",
                dynamic_audit_identity={"sha256": "c" * 64},
                sampling_contract=sampling,
                runtime_evidence=runtime,
            )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(
                contract["subprotocol"],
                MODULE.SCREEN_GENERATION_SUBPROTOCOL,
            )
            self.assertFalse(contract["formal_data"])
            self.assertFalse(contract["screen_outputs_may_enter_formal"])
            self.assertFalse(
                contract["screen_outputs_may_enter_formal_v5_3"]
            )
            self.assertEqual(
                contract["screen_manifest_sha256"],
                MODULE.sha256_file(manifest_path),
            )
            self.assertEqual(
                contract["generation_manifest_sha256"],
                manifest["source_generation_manifest_sha256"],
            )
            self.assertEqual(contract["trial_seeds"], MODULE.SCREEN_TRIAL_SEEDS)
            self.assertEqual(contract["max_model_len"], 32768)

    def test_screen_manifest_semantic_tamper_fails_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_screen_manifest()
            manifest["generation"]["teacher"]["revision"] = "d" * 40
            manifest["canonical_sha256"] = MODULE.canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "canonical_sha256"
                }
            )
            manifest_path = root / "screen_manifest.json"
            split_path = root / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            split_path.write_text(json.dumps(screen_split()), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest contract drift"):
                MODULE.load_inner_train_manifest(
                    manifest_path,
                    split_manifest=MODULE.load_split_manifest(split_path),
                    split_manifest_sha256=MODULE.SCREEN_SPLIT_SHA256,
                )

    def test_screen_seed_and_trial_count_are_frozen(self):
        invalid = (
            (MODULE.SCREEN_SEED + 1, MODULE.SCREEN_NUM_TRIALS),
            (MODULE.SCREEN_SEED, MODULE.SCREEN_NUM_TRIALS - 1),
        )
        for seed, trials in invalid:
            with self.subTest(seed=seed, trials=trials):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "screen sampling contract drift",
                ):
                    MODULE.validate_sampling_contract(
                        manifest_protocol=MODULE.SCREEN_MANIFEST_PROTOCOL,
                        temperature=MODULE.SCREEN_TEMPERATURE,
                        top_p=MODULE.SCREEN_TOP_P,
                        num_trials=trials,
                        seed=seed,
                    )

    def test_runtime_context_or_model_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.json"
            host_path = root / "preflight.json"
            write_host_preflight(host_path)
            for mutation in (
                "context",
                "model",
                "old_protocol",
                "capacity_kind",
                "tool_finish",
                "tool_schema",
                "missing_concurrency",
            ):
                with self.subTest(mutation=mutation):
                    payload = make_runtime_receipt("b" * 40, host_path)
                    if mutation == "context":
                        payload["max_model_len"] = 16384
                    elif mutation == "model":
                        payload["roles"]["teacher"]["model"] = "wrong/model"
                    elif mutation == "old_protocol":
                        payload["protocol"] = (
                            "v5_3_generation_runtime_preflight_v2"
                        )
                    elif mutation == "capacity_kind":
                        payload["roles"]["user_and_judge"][
                            "long_context_probe"
                        ]["probe_kind"] = "named_tool_interface"
                    elif mutation == "tool_finish":
                        payload["roles"]["teacher"]["tool_interface_probe"][
                            "finish_reason"
                        ] = "length"
                    elif mutation == "tool_schema":
                        payload["roles"]["teacher"]["tool_interface_probe"][
                            "request_schema_sha256"
                        ] = "0" * 64
                    else:
                        payload["roles"]["teacher"].pop(
                            "concurrency_probe"
                        )
                    payload["canonical_receipt_sha256"] = MODULE.canonical_sha256(
                        {
                            key: value
                            for key, value in payload.items()
                            if key
                            not in {
                                "canonical_receipt_sha256",
                                "receipt_pointer",
                            }
                        }
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "runtime"):
                        MODULE.validate_screen_runtime_evidence(
                            path,
                            expected_source_commit="b" * 40,
                            max_model_len=32768,
                        )

    def test_screen_finalize_rejects_zero_strict_judge_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "run_contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "status": "INCOMPLETE",
                        "generation_manifest_protocol": (
                            MODULE.SCREEN_MANIFEST_PROTOCOL
                        ),
                        "result_sha256": {},
                        "strict_judge_audit_evidence": {},
                    }
                ),
                encoding="utf-8",
            )
            result = root / "retail_clean.json"
            result.write_text(
                json.dumps(
                    {
                        "simulations": [
                            {
                                "task_id": "2",
                                "id": "zero-judge-call",
                                "reward_info": {"nl_assertions": []},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "zero or incomplete"):
                MODULE.finalize_contract(contract, [result])
            self.assertEqual(
                json.loads(contract.read_text(encoding="utf-8"))["status"],
                "INCOMPLETE",
            )


if __name__ == "__main__":
    unittest.main()
