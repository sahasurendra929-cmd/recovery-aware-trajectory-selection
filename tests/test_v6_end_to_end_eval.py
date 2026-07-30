from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_v6_end_to_end_eval as evaluator


def manifest_rows() -> list[dict]:
    rows = []
    for domain, count in (("retail", 40), ("airline", 20)):
        for task_id in range(count):
            rows.append(
                {
                    "domain": domain,
                    "task_id": str(task_id),
                    "error_condition": {
                        "fault_family": f"{domain}_missing_id",
                        "tool_name": "get_order_details",
                        "tool_call_id": f"fault-{domain}-{task_id}",
                        "arguments": {"order_id": f"#BAD{task_id}"},
                    },
                }
            )
    return rows


def message_call(call_id: str, order_id: str, *, injected: bool) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "name": "get_order_details",
                "arguments": {"order_id": order_id},
            }
        ],
        "raw_data": (
            {"v5_stage1_injected_fault": True} if injected else {}
        ),
    }


def result_file(
    path: Path,
    *,
    task_id: str,
    condition: str,
    reward: float,
) -> None:
    messages = [{"role": "user", "content": "help"}]
    if condition == "error":
        messages.extend(
            [
                message_call("fault", f"#BAD{task_id}", injected=True),
                {
                    "role": "tool",
                    "id": "fault",
                    "content": "Error: Order not found",
                    "error": True,
                },
            ]
        )
    messages.extend(
        [
            message_call("recovery", "#GOOD", injected=False),
            {
                "role": "tool",
                "id": "recovery",
                "content": "{}",
                "error": False,
            },
        ]
    )
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": task_id,
                        "reward_info": {"reward": reward},
                        "duration": 1.5,
                        "termination_reason": "user_stop",
                        "messages": messages,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_exact_two_shard_partition() -> None:
    rows = manifest_rows()
    left = evaluator.shard_rows(rows, shard_index=0)
    right = evaluator.shard_rows(rows, shard_index=1)
    left_ids = {(row["domain"], row["task_id"]) for row in left}
    right_ids = {(row["domain"], row["task_id"]) for row in right}
    assert len(left) == len(right) == 30
    assert not left_ids & right_ids
    assert left_ids | right_ids == {
        (row["domain"], row["task_id"]) for row in rows
    }
    with pytest.raises(evaluator.EvaluationError, match="two shards"):
        evaluator.shard_rows(rows, shard_index=0, num_shards=3)


def test_extract_rows_records_v610_official_provenance(tmp_path: Path) -> None:
    clean = tmp_path / "retail_clean.shard-000-of-002.json"
    error = tmp_path / "retail_error.shard-000-of-002.json"
    result_file(clean, task_id="1", condition="clean", reward=1.0)
    result_file(error, task_id="1", condition="error", reward=0.0)
    manifest = [
        {
            "domain": "retail",
            "task_id": "1",
            "error_condition": {
                "fault_family": "retail_missing_order_id",
                "tool_name": "get_order_details",
                "arguments": {"order_id": "#BAD1"},
            },
        }
    ]
    rows = evaluator.extract_rows(
        [clean, error],
        manifest_rows=manifest,
        arm="full_proposed",
        evaluation_seed=20260722,
        shard_index=0,
        unseal_receipt_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        checkpoint_fingerprint="c" * 64,
        agent_service_receipt_sha256="d" * 64,
        user_judge_service_receipt_sha256="e" * 64,
    )
    assert len(rows) == 2
    assert {row["condition"] for row in rows} == {
        "clean",
        "controlled_error",
    }
    assert all(row["protocol"] == evaluator.PROTOCOL for row in rows)
    assert all(row["official_test_used"] is True for row in rows)
    controlled = next(
        row for row in rows if row["condition"] == "controlled_error"
    )
    assert controlled["injection_count"] == 1
    assert controlled["injection_observed_as_error"] is True
    assert controlled["injection_repeated_by_harness"] is False


def test_extract_rejects_changed_or_unobserved_injection(
    tmp_path: Path,
) -> None:
    error = tmp_path / "retail_error.shard-000-of-002.json"
    result_file(error, task_id="1", condition="error", reward=0.0)
    payload = json.loads(error.read_text(encoding="utf-8"))
    payload["simulations"][0]["messages"][1]["tool_calls"][0]["arguments"] = {
        "order_id": "#CHANGED"
    }
    error.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ"):
        evaluator.extract_rows(
            [error],
            manifest_rows=[
                {
                    "domain": "retail",
                    "task_id": "1",
                    "error_condition": {
                        "fault_family": "retail_missing_order_id",
                        "tool_name": "get_order_details",
                        "arguments": {"order_id": "#BAD1"},
                    },
                }
            ],
            arm="full_proposed",
            evaluation_seed=20260722,
            shard_index=0,
            unseal_receipt_sha256="a" * 64,
            evaluator_sha256="b" * 64,
            checkpoint_fingerprint="c" * 64,
            agent_service_receipt_sha256="d" * 64,
            user_judge_service_receipt_sha256="e" * 64,
        )


def test_validate_unseal_recomputes_receipt_and_executable_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(evaluator.__file__).resolve().parents[1]
    tau2 = tmp_path / "tau2"
    tau2.mkdir()
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    bound_artifacts = {}
    for label in evaluator.REQUIRED_ARTIFACT_LABELS:
        path = tmp_path / f"{label}.artifact"
        path.write_text(label, encoding="utf-8")
        bound_artifacts[label] = {
            "path": str(path.resolve()),
            "sha256": evaluator.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    summarizer = source / "scripts" / "summarize_v6_task_clusters.py"
    payload = {
        "protocol": evaluator.UNSEAL_PROTOCOL,
        "status": "UNSEALED_ONCE",
        "official_test_access_count_before_unseal": 0,
        "official_test_access_count": 1,
        "changes_after_unseal": "FORBIDDEN",
        "source_commit": "a" * 40,
        "tau2_commit": evaluator.TAU2_COMMIT,
        "evaluator_sha256": evaluator.sha256_file(
            Path(evaluator.__file__).resolve()
        ),
        "evaluator_path": str(Path(evaluator.__file__).resolve()),
        "summarizer_path": str(summarizer.resolve()),
        "summarizer_sha256": evaluator.sha256_file(summarizer),
        "bound_artifacts": bound_artifacts,
        "split_manifest": {
            "path": str(split.resolve()),
            "sha256": evaluator.sha256_file(split),
        },
    }
    payload["receipt_sha256"] = evaluator._receipt_payload_sha256(payload)
    receipt = tmp_path / "unseal.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    def identity(path: Path) -> tuple[str, bool]:
        return (
            ("a" * 40, True)
            if path == source
            else (evaluator.TAU2_COMMIT, True)
        )

    monkeypatch.setattr(evaluator, "git_identity", identity)
    assert evaluator.validate_unseal(
        receipt,
        source_root=source,
        tau2_root=tau2,
        split_manifest=split,
    )["status"] == "UNSEALED_ONCE"
    changed = Path(
        bound_artifacts["checkpoint_registry"]["path"]
    )
    changed.write_text("changed", encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="artifact drift"):
        evaluator.validate_unseal(
            receipt,
            source_root=source,
            tau2_root=tau2,
            split_manifest=split,
        )
    changed.write_text("checkpoint_registry", encoding="utf-8")
    payload["receipt_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="binding drift"):
        evaluator.validate_unseal(
            receipt,
            source_root=source,
            tau2_root=tau2,
            split_manifest=split,
        )
