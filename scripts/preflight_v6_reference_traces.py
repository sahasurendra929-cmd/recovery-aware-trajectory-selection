#!/usr/bin/env python3
"""Build the registry-independent V6.10 reference-execution receipt.

This preflight consumes only the frozen V6.10 config, identity-only split,
pinned tau2 task files, and tau2's deterministic environment/evaluator.  It
does not consume a candidate registry and makes no model, teacher, simulator,
judge, CUDA, or network call.

Candidate injection × corrective-action cells are intentionally absent here.
They are a later registry-v2 gate; making them a prerequisite for the
reference receipt would create a circular dependency.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__:
    from scripts import v6_reference_contract as contract
else:  # pragma: no cover - direct script execution
    import v6_reference_contract as contract


EXPECTED_FORMAL_TASKS = 50
FROZEN_RENDERER = "explicit_user_direct_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(
    root: Path,
    *arguments: str,
    allow_failure: bool = False,
) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 and not allow_failure:
        raise contract.ReferenceContractError(
            f"git {' '.join(arguments)} failed in {root}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def repository_provenance(root: Path) -> dict[str, Any]:
    tree_listing = _git(root, "ls-tree", "-r", "--full-tree", "HEAD")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": contract.sha256({"git_ls_tree_r": tree_listing}),
        "git_tree_object": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": (
            _git(
                root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
            == ""
        ),
        "worktree_scope": "tracked_and_untracked_files",
    }


def tau2_provenance(root: Path, expected_commit: str) -> dict[str, Any]:
    observed = _git(root, "rev-parse", "HEAD")
    if observed != expected_commit:
        raise contract.ReferenceContractError(
            f"tau2 commit drift: {observed} != {expected_commit}"
        )
    tree_listing = _git(root, "ls-tree", "-r", "--full-tree", "HEAD")
    return {
        "commit": observed,
        "tree": contract.sha256({"git_ls_tree_r": tree_listing}),
        "git_tree_object": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": (
            _git(
                root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
            == ""
        ),
        "worktree_scope": "tracked_and_untracked_files",
    }


def _config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != contract.DESIGN_PROTOCOL:
        raise contract.ReferenceContractError("V6.10 config protocol drift")
    benchmark = config.get("benchmark")
    populations = config.get("task_populations")
    reference = config.get("reference_plan_audit")
    construction = config.get("positive_data_construction")
    if not all(
        isinstance(value, Mapping)
        for value in (benchmark, populations, reference, construction)
    ):
        raise contract.ReferenceContractError(
            "V6.10 config lacks benchmark/reference sections"
        )
    formal = populations.get("formal_pool")
    if not isinstance(formal, Mapping):
        raise contract.ReferenceContractError("config lacks formal_pool")
    task_ids = formal.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != EXPECTED_FORMAL_TASKS
        or len(task_ids) != len(set(task_ids))
        or formal.get("exact_tasks") != EXPECTED_FORMAL_TASKS
    ):
        raise contract.ReferenceContractError("formal task list is not exact 50")
    if formal.get("ordered_task_ids_sha256") != contract.sha256(task_ids):
        raise contract.ReferenceContractError("formal task-list hash drift")
    if reference.get("receipt_protocol") != contract.RECEIPT_PROTOCOL:
        raise contract.ReferenceContractError(
            "config reference receipt protocol drift"
        )
    renderer = (
        construction.get("clean_view", {})
        if isinstance(construction.get("clean_view"), Mapping)
        else {}
    ).get("completion_renderer")
    if renderer != FROZEN_RENDERER:
        raise contract.ReferenceContractError("completion renderer drift")
    if reference.get("scope") != "all_50_formal_pool_tasks_before_any_gpu_generation":
        raise contract.ReferenceContractError("reference preflight scope drift")
    if benchmark.get("official_test", {}).get("sealed") is not True:
        raise contract.ReferenceContractError("official test is not sealed")
    return {
        "task_ids": [str(value) for value in task_ids],
        "tau2_commit": str(benchmark.get("commit")),
        "completion_renderer": renderer,
        "split_manifest": str(benchmark.get("split_manifest")),
    }


def _environment_evaluator(
    *,
    domain: str,
    task: Any,
    generation: Any,
):
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    def evaluate(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        timestamp = "1970-01-01T00:00:00Z"
        simulation = SimulationRun(
            id=f"v610-reference-preflight-{domain}-{task.id}",
            task_id=str(task.id),
            timestamp=timestamp,
            start_time=timestamp,
            end_time=timestamp,
            duration=0.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=generation.parse_messages(messages),
            seed=None,
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ENV,
            solo_mode=False,
            domain=domain,
            strict_replay=True,
        )
        return {
            "reward": float(reward_info.reward),
            "reward_info": reward_info.model_dump(mode="json"),
        }

    return evaluate


def _task_source_hash(task_payload: Mapping[str, Any]) -> str:
    criteria = task_payload.get("evaluation_criteria")
    return contract.sha256(
        {
            "id": task_payload.get("id"),
            "initial_state": task_payload.get("initial_state"),
            "evaluation_criteria": criteria,
        }
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    tau2_root = args.tau2_root.resolve()
    config_path = args.config.resolve()
    split_path = args.split_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise contract.ReferenceContractError(
            f"refusing to overwrite preflight receipt: {output}"
        )
    if not all(path.is_file() for path in (config_path, split_path)):
        raise contract.ReferenceContractError("config or split manifest is missing")

    # Imports are intentionally lazy so --help and pure unit tests need no tau2.
    try:
        from scripts import prepare_v6_candidate_registry as source_loader
        from scripts import run_v6_candidate_generation as generation
        from scripts import v6_selection_protocol as selection_protocol
    except ModuleNotFoundError:  # pragma: no cover - direct script path
        import prepare_v6_candidate_registry as source_loader
        import run_v6_candidate_generation as generation
        import v6_selection_protocol as selection_protocol

    config = source_loader.load_config(config_path)
    frozen = _config_contract(config)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise contract.ReferenceContractError("split manifest root is not object")
    arm_ids, _, forbidden = source_loader.frozen_arm_train(split, strict=True)
    task_ids = frozen["task_ids"]
    if not set(task_ids).issubset(set(arm_ids)):
        raise contract.ReferenceContractError(
            "formal reference tasks are not a subset of frozen arm-train"
        )
    if set(task_ids) & (forbidden["sealed"] | forbidden["validation"]):
        raise contract.ReferenceContractError(
            "formal reference tasks overlap validation/official test"
        )
    configured_split = (source_root / frozen["split_manifest"]).resolve()
    if configured_split != split_path:
        raise contract.ReferenceContractError(
            f"split path differs from frozen config: {split_path} != "
            f"{configured_split}"
        )

    catalog, catalog_hashes = source_loader.load_task_catalog(tau2_root, split)
    if any(task not in catalog for task in task_ids):
        raise contract.ReferenceContractError("pinned task catalog lacks formal task")

    tau2 = tau2_provenance(tau2_root, frozen["tau2_commit"])
    source = repository_provenance(source_root)
    generation.configure_tau2(tau2_root)
    try:
        from loguru import logger
    except ModuleNotFoundError:
        pass
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    task_rows: list[dict[str, Any]] = []
    task_source_hashes: dict[str, str] = {}
    for task_identity in task_ids:
        payload = catalog[task_identity]
        task_source_hashes[task_identity] = _task_source_hash(payload)
        try:
            domain, task = generation.select_task(task_identity)
            criteria = task.evaluation_criteria
            if criteria is None or not criteria.actions:
                raise contract.ReferenceContractError(
                    f"{task_identity}: no reference actions"
                )
            renderer_inputs = contract.build_renderer_lint_inputs(
                task_identity=task_identity,
                communicate_info=criteria.communicate_info,
                nl_assertions=criteria.nl_assertions,
                fallback="The requested changes have been completed.",
            )
            rendered = generation.deterministic_completion_message(
                criteria,
                renderer=frozen["completion_renderer"],
                fallback="The requested changes have been completed.",
            )

            def environment_factory(
                _domain: str = domain,
                _task: Any = task,
            ) -> Any:
                return generation.initial_environment(_domain, _task, [])

            row = contract.preflight_task(
                task_identity=task_identity,
                actions=list(criteria.actions),
                allowed_identifier_keys=(
                    selection_protocol.SAFE_ON_ERROR_IDENTIFIER_KEYS[domain]
                ),
                prefix=[],
                environment_factory=environment_factory,
                execute_call=generation.execute_call,
                state_hashes=generation.database_hashes,
                environment_evaluator=_environment_evaluator(
                    domain=domain,
                    task=task,
                    generation=generation,
                ),
                renderer_lint_inputs=renderer_inputs,
                rendered_message=rendered,
            )
        except Exception as error:
            row = contract.task_error_row(task_identity, error)
        task_rows.append(row)

    files = {
        "preflight_script_sha256": contract.sha256_file(Path(__file__).resolve()),
        "contract_module_sha256": contract.sha256_file(
            Path(contract.__file__).resolve()
        ),
        "generation_adapter_sha256": contract.sha256_file(
            Path(generation.__file__).resolve()
        ),
        "selection_protocol_sha256": contract.sha256_file(
            Path(selection_protocol.__file__).resolve()
        ),
        "config_sha256": contract.sha256_file(config_path),
        "split_manifest_sha256": contract.sha256_file(split_path),
        "task_catalog_file_sha256": catalog_hashes,
    }
    provenance = {
        "source": source,
        "tau2": tau2,
        "files": files,
        "completion_renderer": frozen["completion_renderer"],
        "official_test_sealed": True,
        "official_test_identity_overlap_count": 0,
    }
    receipt = contract.build_preflight_receipt(
        expected_task_ids=task_ids,
        task_results=task_rows,
        provenance=provenance,
        task_source_hashes=task_source_hashes,
    )
    contract.atomic_write_receipt(output, receipt)
    return receipt


def main() -> None:
    args = parse_args()
    try:
        receipt = run_preflight(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "protocol": contract.RECEIPT_PROTOCOL,
                    "status": "NO_GO",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                "protocol": receipt["protocol"],
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                **receipt["summary"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if receipt["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
