# V6 Stage 2 — Outcome-Independent Registry

Date: 2026-07-30 (Asia/Shanghai)

Status: **PASS — runtime preflight remains required**

Source commit:
`9f620d438884270f3231101924ff9ef2f6cc5d09`

## Registry result

- Protocol: `v6_candidate_pair_registry_v1`
- Registry SHA-256:
  `453d93e64cec3fbb0de4e99955c4f62da2ca8051038ff074f9e68d42919525ed`
- Structural-eligibility SHA-256:
  `002b5a2c5d83a4dab6dc8ce398d81bb8541bf7bfc6b25c04e62ce9ed179587f7`
- Registry status:
  `STRUCTURALLY_FROZEN_RUNTIME_PREFLIGHT_REQUIRED`
- Selection uses outcomes: false
- Official test used: false
- Official test content exported: false

## Populations

| Population | Tasks | Candidate pairs |
|---|---:|---:|
| Pilot | 24 | 72 |
| Formal pool registry | 50 | 150 |
| Total | 74 choice sets | 222 |

The Pilot identity is the frozen 24-task set with registry hash
`5170b139c6307a3f46e29ab9db2573b864181c7b8ff85e27452e14f591bf27c4`.
It contains 6 airline and 18 retail tasks. The formal registry contains 50
action-identifiable tasks. Pilot and formal generation seed salts do not
overlap.

## Provenance checks

- tau2 commit matched:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Partition SHA-256 matched:
  `c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818`
- Split manifest SHA-256:
  `a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`
- Config SHA-256:
  `960d872287fd92508d6825ee5d807d82351a7042330faa58eaba59f713bda637`
- Official-test identity overlap count: zero.

## Preserved artifacts

- `reports/artifacts/stage2/registry.json`
- `reports/logs/stage2_registry.log`

## Decision

The outcome-independent registry is frozen and internally consistent.
Proceed to the runtime preflight and Pilot candidate generation. Generation
outputs remain unscored and cannot support a scientific claim.

