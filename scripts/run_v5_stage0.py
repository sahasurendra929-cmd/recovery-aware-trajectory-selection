#!/usr/bin/env python3
"""Run paired clean/error-injected end-to-end τ²-bench Stage-0 tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


FAULT_INJECTIONS: dict[str, dict[str, Any]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="openai/Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--api-key", default="stage0-local")
    parser.add_argument("--condition", choices=("clean", "error", "both"), default="both")
    parser.add_argument(
        "--pair-id",
        help="Run exactly one manifest pair (clean/error) for a minimal preflight.",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def configure_tau2_path(tau2_root: Path) -> None:
    source = tau2_root.resolve() / "src"
    if not source.exists():
        raise FileNotFoundError(f"tau2 source directory not found: {source}")
    sys.path.insert(0, str(source))


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "v5_stage0_paired_clean_error_smoke":
        raise RuntimeError("Unexpected Stage-0 manifest protocol")
    if payload.get("paired_task_count") != 10:
        raise RuntimeError("Stage-0 manifest must contain ten paired tasks")
    return payload


def filter_manifest_rows(
    manifest: dict[str, Any], pair_id: str | None
) -> list[dict[str, Any]]:
    rows = list(manifest["rows"])
    if pair_id is None:
        return rows
    selected = [row for row in rows if row["pair_id"] == pair_id]
    if len(selected) != 1:
        raise RuntimeError(
            f"--pair-id must identify exactly one manifest row; got {len(selected)}"
        )
    return selected


def register_fault_agent() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    class DeterministicFaultAgent(LLMAgent):
        """Inject exactly one manifest-defined read-only failure, then act normally."""

        def __init__(self, tools, domain_policy, task, llm, llm_args):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=llm_args,
            )
            injection = FAULT_INJECTIONS.get(str(task.id))
            if not injection:
                raise RuntimeError(f"Task {task.id} is missing Stage-0 injection data")
            self._stage0_injection = injection
            self._stage0_injected = False

        def _generate_next_message(self, message, state):
            if not self._stage0_injected and isinstance(message, UserMessage):
                state.messages.append(message)
                injection = self._stage0_injection
                self._stage0_injected = True
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=injection["tool_call_id"],
                            name=injection["tool_name"],
                            arguments=injection["arguments"],
                            requestor="assistant",
                        )
                    ],
                    cost=0.0,
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    raw_data={
                        "v5_stage0_injected_fault": True,
                        "expected_tool_error": True,
                    },
                    generation_time_seconds=0.0,
                )
            return super()._generate_next_message(message, state)

    def create_fault_agent(tools, domain_policy, **kwargs):
        return DeterministicFaultAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs.get("task"),
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory("v5_stage0_fault_agent") is None:
        registry.register_agent_factory(
            create_fault_agent,
            "v5_stage0_fault_agent",
        )


def patch_local_nl_judge(model: str, llm_args: dict[str, Any]) -> None:
    import tau2.evaluator.evaluator_nl_assertions as nl_module

    nl_module.DEFAULT_LLM_NL_ASSERTIONS = model
    nl_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(llm_args)


def select_tasks(domain: str, rows: list[dict[str, Any]]):
    from tau2.registry import registry

    all_tasks = {
        str(task.id): task for task in registry.get_tasks_loader(domain)(None)
    }
    selected = []
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in all_tasks:
            raise RuntimeError(f"Missing {domain} task {task_id}")
        selected.append(all_tasks[task_id])
    return selected


def run_condition(
    *,
    domain: str,
    rows: list[dict[str, Any]],
    condition: str,
    args: argparse.Namespace,
) -> Path:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_tasks

    llm_args = {
        "api_base": args.api_base,
        "api_key": args.api_key,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    patch_local_nl_judge(args.model, llm_args)
    agent_name = "llm_agent" if condition == "clean" else "v5_stage0_fault_agent"
    if condition == "error":
        FAULT_INJECTIONS.clear()
        FAULT_INJECTIONS.update(
            {
                str(row["task_id"]): dict(row["error_condition"])
                for row in rows
            }
        )
    tasks = select_tasks(domain, rows)
    output_path = args.output_dir / f"{domain}_{condition}.json"
    save_dir = args.output_dir / "logs" / f"{domain}_{condition}"
    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=args.model,
        llm_args_agent=llm_args,
        llm_user=args.model,
        llm_args_user=llm_args,
        num_trials=1,
        max_steps=args.max_steps,
        max_errors=10,
        timeout=args.timeout,
        max_concurrency=1,
        seed=args.seed,
        log_level="INFO",
        max_retries=1,
        retry_delay=1.0,
        auto_resume=False,
        hallucination_retries=0,
        enforce_communication_protocol=True,
        verbose_logs=True,
    )
    run_tasks(
        config,
        tasks,
        save_path=output_path,
        save_dir=save_dir,
        evaluation_type=EvaluationType.ALL,
        console_display=True,
        results_format="json",
    )
    if not output_path.exists():
        raise RuntimeError(f"τ²-bench did not write {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    configure_tau2_path(args.tau2_root)
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    manifest = load_manifest(args.manifest.resolve())
    selected_rows = filter_manifest_rows(manifest, args.pair_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    register_fault_agent()

    conditions = ("clean", "error") if args.condition == "both" else (args.condition,)
    output_files = []
    for domain in ("retail", "airline"):
        domain_rows = [
            row for row in selected_rows if row["domain"] == domain
        ]
        if not domain_rows:
            continue
        for condition in conditions:
            output_files.append(
                run_condition(
                    domain=domain,
                    rows=domain_rows,
                    condition=condition,
                    args=args,
                )
            )
    print(
        json.dumps(
            {"status": "PASS", "result_files": [str(path) for path in output_files]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
