from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_v6_task_clusters as summary


def rows() -> list[dict]:
    result = []
    tasks = ("retail:1", "airline:2")
    for arm in summary.ARMS:
        for seed in summary.EVALUATION_SEEDS:
            for identity in tasks:
                domain, task_id = identity.split(":")
                for condition in summary.CONDITIONS:
                    row = {
                        "protocol": summary.EVALUATION_PROTOCOL,
                        "arm": arm,
                        "training_seed": summary.TRAINING_SEED,
                        "evaluation_seed": seed,
                        "domain": domain,
                        "task_id": task_id,
                        "condition": condition,
                        "unseal_receipt_sha256": "a" * 64,
                        "evaluator_sha256": "b" * 64,
                        "official_test_used": True,
                    }
                    success = arm == "full_proposed"
                    if condition == "clean":
                        success = True
                    row.update(
                        {
                            "terminal_status": "PASS",
                            "official_reward": float(success),
                            "task_success": success,
                            "injection_count": (
                                0 if condition == "clean" else 1
                            ),
                            "injection_observed_as_error": (
                                None if condition == "clean" else True
                            ),
                            "injection_repeated_by_harness": (
                                None if condition == "clean" else False
                            ),
                        }
                    )
                    result.append(row)
    return result


def test_grid_is_exact_and_provenance_bound() -> None:
    values = rows()
    audit = summary.audit_grid(
        values,
        task_ids={"retail:1", "airline:2"},
        unseal_receipt_sha256="a" * 64,
        evaluator_sha256="b" * 64,
    )
    assert audit["status"] == "PASS"
    assert audit["observed_rows"] == 36
    values.pop()
    with pytest.raises(summary.SummaryError, match="incomplete"):
        summary.audit_grid(
            values,
            task_ids={"retail:1", "airline:2"},
            unseal_receipt_sha256="a" * 64,
            evaluator_sha256="b" * 64,
        )


def test_seeds_are_averaged_within_task_before_cluster_bootstrap() -> None:
    values = rows()
    audit = summary.audit_grid(
        values,
        task_ids={"retail:1", "airline:2"},
        unseal_receipt_sha256="a" * 64,
        evaluator_sha256="b" * 64,
    )
    result = summary.summarize(values, grid_audit=audit)
    assert result["primary"]["task_count"] == 2
    assert result["primary"]["paired_task_mean_delta"] == 1.0
    assert result["primary"]["task_cluster_bootstrap_ci95"] == [1.0, 1.0]
    assert result["clean_noninferiority"]["paired_task_mean_delta"] == 0.0
    assert result["positive_claim"] is True


def test_clean_injection_and_changed_provenance_fail_closed() -> None:
    values = rows()
    values[0]["injection_count"] = 1
    with pytest.raises(summary.SummaryError, match="clean row"):
        summary.audit_grid(
            values,
            task_ids={"retail:1", "airline:2"},
            unseal_receipt_sha256="a" * 64,
            evaluator_sha256="b" * 64,
        )
    values = rows()
    values[0]["evaluator_sha256"] = "c" * 64
    with pytest.raises(summary.SummaryError, match="provenance"):
        summary.audit_grid(
            values,
            task_ids={"retail:1", "airline:2"},
            unseal_receipt_sha256="a" * 64,
            evaluator_sha256="b" * 64,
        )


def test_bootstrap_contract_is_frozen() -> None:
    with pytest.raises(summary.SummaryError, match="10,000"):
        summary.cluster_bootstrap_ci([0.0, 1.0], replicates=9999)


def test_evaluation_directory_requires_exact_receipted_job_grid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    all_rows = rows()
    # The loader contract is independent of the full 60-task grid audit, so
    # repeat these compact fixture rows to exercise all 18 job receipts.
    for arm in summary.ARMS:
        for seed in summary.EVALUATION_SEEDS:
            matching = [
                row
                for row in all_rows
                if row["arm"] == arm and row["evaluation_seed"] == seed
            ]
            for shard in range(2):
                job = root / arm / str(seed) / f"shard-{shard}"
                job.mkdir(parents=True)
                shard_rows = matching * 15
                rows_path = job / "rows.jsonl"
                rows_path.write_text(
                    "".join(
                        summary.canonical(row) + "\n" for row in shard_rows
                    ),
                    encoding="utf-8",
                )
                (job / "evaluation_receipt.json").write_text(
                    json.dumps(
                        {
                            "protocol": (
                                "v6_10_official_evaluation_shard_receipt_v1"
                            ),
                            "status": "PASS",
                            "arm": arm,
                            "evaluation_seed": seed,
                            "shard_index": shard,
                            "num_shards": 2,
                            "row_count": 60,
                            "rows_sha256": summary.sha256_file(rows_path),
                            "official_test_used": True,
                            "changes_after_unseal": False,
                        }
                    ),
                    encoding="utf-8",
                )
    loaded, hashes = summary.read_evaluation_rows(root)
    assert len(loaded) == 1080
    assert len(hashes) == 18
    next(root.rglob("evaluation_receipt.json")).unlink()
    with pytest.raises(summary.SummaryError, match="invalid JSON object"):
        summary.read_evaluation_rows(root)


def test_expected_official_population_is_40_plus_20() -> None:
    split = {
        "protocol": "v5_stage0_tau2_end_to_end",
        "domains": {
            "retail": {"sealed_test_ids": list(range(40))},
            "airline": {"sealed_test_ids": list(range(20))},
        },
    }
    assert len(summary.expected_task_ids(split)) == 60
    split["domains"]["airline"]["sealed_test_ids"].pop()
    with pytest.raises(summary.SummaryError, match="population"):
        summary.expected_task_ids(split)


def test_summary_output_is_atomic_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "protocol": "v5_stage0_tau2_end_to_end",
                "domains": {
                    "retail": {"sealed_test_ids": list(range(40))},
                    "airline": {"sealed_test_ids": list(range(20))},
                },
            }
        ),
        encoding="utf-8",
    )
    unseal = tmp_path / "unseal.json"
    unseal.write_text(
        json.dumps(
            {
                "protocol": "v6_10_official_test_unseal_v1",
                "status": "UNSEALED_ONCE",
                "official_test_access_count": 1,
                "evaluator_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    rows_path = tmp_path / "rows.jsonl"
    task_ids = {
        *(f"retail:{value}" for value in range(40)),
        *(f"airline:{value}" for value in range(20)),
    }
    complete = []
    for arm in summary.ARMS:
        for seed in summary.EVALUATION_SEEDS:
            for identity in task_ids:
                domain, task_id = identity.split(":")
                for condition in summary.CONDITIONS:
                    complete.append(
                        {
                            "protocol": summary.EVALUATION_PROTOCOL,
                            "arm": arm,
                            "training_seed": summary.TRAINING_SEED,
                            "evaluation_seed": seed,
                            "domain": domain,
                            "task_id": task_id,
                            "condition": condition,
                            "terminal_status": "PASS",
                            "official_reward": 1.0,
                            "task_success": True,
                            "injection_count": (
                                0 if condition == "clean" else 1
                            ),
                            "injection_observed_as_error": (
                                None if condition == "clean" else True
                            ),
                            "injection_repeated_by_harness": (
                                None if condition == "clean" else False
                            ),
                            "unseal_receipt_sha256": summary.sha256_file(unseal),
                            "evaluator_sha256": "b" * 64,
                            "official_test_used": True,
                        }
                    )
    rows_path.write_text(
        "".join(summary.canonical(row) + "\n" for row in complete),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    result = summary.build_summary(
        rows_path=rows_path,
        split_manifest_path=split,
        unseal_receipt_path=unseal,
        output=output,
        expected_evaluator_sha256="b" * 64,
    )
    assert result["grid_audit"]["observed_rows"] == 1080
    assert output.is_file()
    with pytest.raises(summary.SummaryError, match="overwrite"):
        summary.build_summary(
            rows_path=rows_path,
            split_manifest_path=split,
            unseal_receipt_path=unseal,
            output=output,
            expected_evaluator_sha256="b" * 64,
        )
