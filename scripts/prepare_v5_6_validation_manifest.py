#!/usr/bin/env python3
"""Register structurally eligible derived-validation reference pairs for V5.6.

This selector never reads inner-train outcomes or official-test tasks.  It
retains every derived-validation task with an eligible, executable read-only
identifier lookup so the subsequent log-probability score remains task held out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import prepare_v5_5_manifest as v55
    import v5_5_protocol as pairs
    import v5_6_context_protocol as protocol
except ModuleNotFoundError:
    from scripts import prepare_v5_5_manifest as v55
    from scripts import v5_5_protocol as pairs
    from scripts import v5_6_context_protocol as protocol


def prepare(tau2_root: Path, split_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    split = v55.load(split_manifest)
    candidates: list[dict[str, Any]] = []
    for domain in ("airline", "retail"):
        # V5 Stage-0 records the derived-validation partition as
        # ``validation_ids``.  It is derived solely from official-train tasks;
        # the sealed official-test IDs remain explicitly excluded.
        allowed = {str(value) for value in split["domains"][domain]["validation_ids"]}
        task_rows = {str(task["id"]): task for task in v55.load(tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json")}
        for task_id in sorted(allowed):
            actions = ((task_rows[task_id].get("evaluation_criteria") or {}).get("actions") or [])
            eligible = pairs.eligible_reference_actions(domain, actions)
            if eligible:
                candidates.append({"task_identity": f"{domain}:{task_id}", "domain": domain, "task_id": task_id, "reference_action_count": len(actions), "eligible_site_count": len(eligible), **pairs.select_site(f"{domain}:{task_id}", eligible)})
    executable, rejections = v55.executable_candidates(tau2_root, candidates)
    domains = {row["domain"] for row in executable}
    if domains != {"airline", "retail"} or any(sum(row["domain"] == domain for row in executable) < 2 for domain in domains):
        raise RuntimeError("derived validation lacks two executable tasks per domain for shuffled context")
    rows = []
    for selected in sorted(executable, key=lambda item: item["task_identity"]):
        for variant in range(pairs.PAIRS_PER_TASK):
            rows.append({**selected, "pair_id": f"{selected['task_identity']}:v5_6_validation:{variant + 1}", "mutation_variant": variant, "mutated_identifier": pairs.mutate_identifier(selected["correct_identifier"], variant), "protocol": pairs.PROTOCOL, "source_split": "derived_validation", "status": "REGISTERED", "official_test_used": False})
    result = {"protocol": protocol.PROTOCOL, "purpose": "derived_validation_context_scoring_only", "source_split": "derived_validation", "source_split_sha256": pairs.sha256(split), "registered_tasks": len(executable), "registered_pairs": len(rows), "task_ids": [row["task_identity"] for row in executable], "rows": rows, "tool_execution_preflight_performed": True, "tool_execution_rejections": rejections, "official_test_used": False, "official_test_sealed": True}
    v55.dump(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.tau2_root.resolve(), args.split_manifest.resolve(), args.output.resolve())
    print(json.dumps({"status": "PASS", "registered_tasks": result["registered_tasks"], "registered_pairs": result["registered_pairs"], "official_test_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
