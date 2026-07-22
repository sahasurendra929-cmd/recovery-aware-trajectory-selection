# Contributing

## One source of truth

Read `PROJECT_SPEC.md` and the relevant file in `configs/` before changing code. Do not silently change the task split, token budget, labels, or metrics.

## Module boundaries

- `data/`: schemas and dataset adapters
- `scripts/`: selection and experiment runners
- `configs/`: frozen configurations only
- `results/`: observed outputs, manifests, and report-ready tables

## Pull-request checklist

- [ ] State the input data version and source.
- [ ] State the configuration and random seed.
- [ ] Save selected trajectory IDs when selection is involved.
- [ ] Do not mix rollouts from one task across train and test.
- [ ] State whether a result is offline teacher-forced, offline retrieval, or executable end-to-end.
- [ ] Record limitations and negative results.

## AI-agent contract

An AI coding agent may modify only the requested module. It must report changed files, input version, output artifacts, reproducible command, and assumptions. It must stop rather than redefine the research question or evaluation protocol.
