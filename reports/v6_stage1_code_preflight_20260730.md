# V6 Stage 1 — Code and Unit-Test Preflight

Date: 2026-07-30 (Asia/Shanghai)

Status: **PASS**

Source under test:
`9f620d438884270f3231101924ff9ef2f6cc5d09`

## Results

1. Python byte-compilation passed for the V6 protocol, registry, generation,
   audit, measurement, scoring, selector, materialization, and directional
   trainer entry points.
2. The handoff's `unittest` group passed: **38 tests passed**.
3. The remaining V6 `pytest` group passed: **45 tests passed**.
4. Total observed V6 tests passed: **83**.

No source file was modified to obtain these results.

## Dependency incident

The base image initially lacked `pytest`. The historical V5 virtual
environment on the migrated network volume did not start within a 90-second
fail-closed probe, so it was not trusted or reused. A small container-local
code-audit environment was created with system packages plus the frozen
`pytest==8.4.1`. This environment is used only for code tests. Model serving
and training will use separately audited, version-pinned environments.

## Preserved logs

- `reports/logs/stage1_pycompile.log`
- `reports/logs/stage1_unittest.log`
- `reports/logs/stage1_pytest.log`

## Decision

The checked-in V6 candidate-construction, selection, materialization, and
directional-training code passes its complete repository test contract.
Proceed to the outcome-independent Pilot registry.

