from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import v5_3_source_snapshot_contract as snapshot


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class SourceSnapshotContractTests(unittest.TestCase):
    def build_snapshot(self, root: Path) -> tuple[Path, Path, str]:
        raw = root / "raw"
        raw.mkdir()
        commit = "a" * 40
        result_names = [
            f"{domain}_{condition}.shard-{shard:03d}-of-003.json"
            for domain in ("airline", "retail")
            for condition in ("clean", "error")
            for shard in range(3)
        ]
        for name in result_names:
            write_json(
                raw / name,
                {
                    "simulations": [
                        {"task_id": str(index)} for index in range(24)
                    ]
                },
            )
        strict = {}
        contract_names = []
        for shard in range(3):
            name = f"run_contract.shard-{shard:03d}-of-003.json"
            contract_names.append(name)
            shard_results = [
                result
                for result in result_names
                if f"shard-{shard:03d}-of-003" in result
            ]
            mapping = f"{shard + 1}" * 64
            strict[name] = mapping
            write_json(
                raw / name,
                {
                    "status": "COMPLETE",
                    "source_commit": commit,
                    "official_test_used": False,
                    "result_sha256": {
                        result: snapshot.sha256_file(raw / result)
                        for result in shard_results
                    },
                    "strict_judge_audit_evidence": {
                        "status": "PASS",
                        "canonical_mapping_sha256": mapping,
                    },
                },
            )
        file_sha256 = {
            path.name: snapshot.sha256_file(path)
            for path in sorted(raw.glob("*.json"))
        }
        stream = "".join(
            f"{file_sha256[name]}  {name}\n"
            for name in sorted(file_sha256)
        )
        receipt = root / snapshot.RECEIPT_NAME
        write_json(
            receipt,
            {
                "protocol": snapshot.PROTOCOL,
                "source_generation_commit": commit,
                "exact_raw_cases": 288,
                "generation_contracts": 3,
                "result_files": 12,
                "official_test_used": False,
                "official_test_sealed": True,
                "sha256sum_stream_sha256": hashlib.sha256(
                    stream.encode("utf-8")
                ).hexdigest(),
                "strict_judge_mapping_sha256_by_contract": strict,
                "file_sha256": file_sha256,
            },
        )
        return raw, receipt, commit

    def test_exact_snapshot_passes_and_result_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw, receipt, commit = self.build_snapshot(Path(temporary))
            identity = snapshot.validate_source_snapshot(
                raw_dir=raw,
                receipt_path=receipt,
                expected_generation_commit=commit,
            )
            self.assertEqual(identity["status"], "PASS")
            self.assertEqual(identity["exact_raw_cases"], 288)
            target = raw / "airline_clean.shard-000-of-003.json"
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                snapshot.SourceSnapshotError,
                "hash drift",
            ):
                snapshot.validate_source_snapshot(
                    raw_dir=raw,
                    receipt_path=receipt,
                    expected_generation_commit=commit,
                )

    def test_contract_or_generation_commit_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw, receipt, commit = self.build_snapshot(Path(temporary))
            with self.assertRaisesRegex(
                snapshot.SourceSnapshotError,
                "identity drift",
            ):
                snapshot.validate_source_snapshot(
                    raw_dir=raw,
                    receipt_path=receipt,
                    expected_generation_commit="b" * 40,
                )
            contract = raw / "run_contract.shard-000-of-003.json"
            value = json.loads(contract.read_text(encoding="utf-8"))
            value["status"] = "INCOMPLETE"
            write_json(contract, value)
            with self.assertRaisesRegex(
                snapshot.SourceSnapshotError,
                "hash drift",
            ):
                snapshot.validate_source_snapshot(
                    raw_dir=raw,
                    receipt_path=receipt,
                    expected_generation_commit=commit,
                )


if __name__ == "__main__":
    unittest.main()
