#!/usr/bin/env python3
"""V4 entry point for the byte-identical frozen V3 generation evaluator.

V4 changes training supervision, not inference or next-tool-call scoring.
Keeping the implementation in one place prevents an evaluator fork while the
protocol name in each new result contract remains explicit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import evaluate_tool_actions_v3 as frozen


def inject_runtime_memory(
    output_path: Path,
    *,
    peak_allocated: int,
    peak_reserved: int,
) -> None:
    """Add measured CUDA peaks without risking a torn metrics.json rewrite."""
    if (
        isinstance(peak_allocated, bool)
        or not isinstance(peak_allocated, int)
        or peak_allocated < 0
        or isinstance(peak_reserved, bool)
        or not isinstance(peak_reserved, int)
        or peak_reserved < peak_allocated
    ):
        raise RuntimeError(
            "invalid CUDA peak-memory counters: "
            f"allocated={peak_allocated!r}, reserved={peak_reserved!r}"
        )
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise RuntimeError(f"frozen evaluator output is not an object: {output_path}")
    metrics["runtime_memory"] = {
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
    }
    temporary = output_path.with_name(output_path.name + ".runtime-memory.tmp")
    try:
        temporary.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_path_from_argv(arguments: list[str]) -> Path:
    for index, argument in enumerate(arguments):
        if argument == "--output":
            if index + 1 >= len(arguments):
                break
            return Path(arguments[index + 1])
        if argument.startswith("--output="):
            value = argument.partition("=")[2]
            if value:
                return Path(value)
            break
    raise RuntimeError("unable to locate frozen --output argument")


def main() -> None:
    # Configure only when this entry point runs.  Importing the helper module
    # must not mutate the frozen V3 evaluator used by other protocol checks.
    frozen.PROTOCOL = "qlora_v4"
    if "--dry-run" in sys.argv:
        frozen.main()
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V4 generation evaluation requires CUDA")
    # Resolve the wrapper-only postprocessing target before loading a model so
    # unsupported abbreviated spellings fail without wasting a full run.
    output_path = output_path_from_argv(sys.argv[1:])
    torch.cuda.reset_peak_memory_stats(0)
    frozen.main()
    inject_runtime_memory(
        output_path,
        peak_allocated=torch.cuda.max_memory_allocated(0),
        peak_reserved=torch.cuda.max_memory_reserved(0),
    )


if __name__ == "__main__":
    main()
