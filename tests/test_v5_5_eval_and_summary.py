from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_v5_5_end_to_end_eval as evaluator
from scripts import summarize_v5_5_results as summary


def _call(call_id: str, order_id: str, *, injected: bool = False) -> dict:
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


def _result(call_id: str, *, error: bool) -> dict:
    return {
        "role": "tool",
        "id": call_id,
        "content": "Error" if error else "{}",
        "error": error,
    }


def _write_result(
    path: Path,
    *,
    task_id: str,
    error_condition: bool,
    reward: float,
) -> None:
    messages = [{"role": "user", "content": "help"}]
    if error_condition:
        messages.extend(
            [
                _call("bad", "#BAD", injected=True),
                _result("bad", error=True),
            ]
        )
    messages.extend(
        [
            _call("good", "#GOOD"),
            _result("good", error=False),
        ]
    )
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": task_id,
                        "reward_info": {"reward": reward},
                        "duration": 2.0,
                        "messages": messages,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_extract_rows_requires_observed_injection(tmp_path: Path):
    clean = tmp_path / "retail_clean.json"
    error = tmp_path / "retail_error.json"
    _write_result(clean, task_id="1", error_condition=False, reward=0.0)
    _write_result(error, task_id="1", error_condition=True, reward=1.0)
    manifest = [
        {
            "domain": "retail",
            "task_id": "1",
            "error_condition": {
                "fault_family": "missing_order",
                "tool_name": "get_order_details",
                "arguments": {"order_id": "#BAD"},
            },
        }
    ]
    rows = evaluator.extract_result_rows(
        [clean, error],
        manifest_rows=manifest,
        arm="repair_50",
        training_seed=20260805,
        evaluation_seed=20260815,
        claim_level="diagnostic_only",
        training_fault_tools={"retail:get_order_details"},
    )
    assert len(rows) == 2
    error_row = next(row for row in rows if row["condition"] == "error")
    assert error_row["task_success"] is True
    assert error_row["fault_scope"] == "in_family"
    assert error_row["injected_fault_observed_as_error"] is True
    assert error_row["repeated_same_injected_call"] is False

    payload = json.loads(error.read_text())
    payload["simulations"][0]["messages"][2]["error"] = False
    error.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not observed"):
        evaluator.extract_result_rows(
            [error],
            manifest_rows=manifest,
            arm="repair_50",
            training_seed=20260805,
            evaluation_seed=20260815,
            claim_level="diagnostic_only",
            training_fault_tools={"retail:get_order_details"},
        )


def grid_rows() -> list[dict]:
    rows = []
    tasks = (("retail", "1"), ("airline", "2"))
    for arm in ("perfect_success", "repair_50"):
        for evaluation_seed in (20260815, 20260816, 20260817):
            for domain, task_id in tasks:
                for condition in ("clean", "error"):
                    success = True
                    if condition == "error" and arm == "perfect_success":
                        success = task_id == "2"
                    rows.append(
                        {
                            "protocol": evaluator.PROTOCOL,
                            "arm": arm,
                            "paper_arm": (
                                "r0_perfect"
                                if arm == "perfect_success"
                                else "r50"
                            ),
                            "training_seed": 20260805,
                            "evaluation_seed": evaluation_seed,
                            "domain": domain,
                            "task_id": task_id,
                            "condition": condition,
                            "fault_scope": (
                                "in_family" if condition == "error" else None
                            ),
                            "official_reward": float(success),
                            "task_success": success,
                            "injected_fault_count": (
                                1 if condition == "error" else 0
                            ),
                            "injected_fault_observed_as_error": (
                                True if condition == "error" else None
                            ),
                            "repeated_same_injected_call": False,
                            "tool_calls": 2,
                            "tool_results": 2,
                            "tool_errors": (
                                1 if condition == "error" else 0
                            ),
                            "claim_level": "diagnostic_only",
                            "official_test_used": False,
                        }
                    )
    return rows


def test_grid_audit_and_task_cluster_summary():
    rows = grid_rows()
    audit = summary.audit_grid(
        rows,
        task_ids={("retail", "1"), ("airline", "2")},
        arms=["perfect_success", "repair_50"],
        training_seeds=[20260805],
        evaluation_seeds=[20260815, 20260816, 20260817],
    )
    assert audit["status"] == "PASS"
    result, table = summary.summarize(
        rows,
        arms=["perfect_success", "repair_50"],
        replicates=1000,
    )
    assert result["comparisons"]["repair_50"]["error_delta_vs_r0"] == 0.5
    assert result["selection"]["selected_arm"] == "repair_50"
    assert result["selection"]["positive_screen"] is True
    assert len(table) == 2

    rows.pop()
    with pytest.raises(RuntimeError, match="incomplete"):
        summary.audit_grid(
            rows,
            task_ids={("retail", "1"), ("airline", "2")},
            arms=["perfect_success", "repair_50"],
            training_seeds=[20260805],
            evaluation_seeds=[20260815, 20260816, 20260817],
        )
