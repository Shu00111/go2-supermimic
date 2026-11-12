# Physics validation

The RFT implementation and provenance are documented in
`RFT_PROVENANCE_AND_VALIDATION.md`. CPU float64 fixtures validate normal and
tangential wrench, immersion, release, per-environment/per-foot memory reset,
and linear strength scaling. The tracked GPU report records an earlier CUDA
integration and mutually exclusive RFT/rigid support at a single boundary.

Development machine result (2026-11-12): CPU RFT tests pass. CUDA parity is
`NOT RUN — RESOURCE CONSTRAINT` because this Apple Silicon host has no CUDA.

Status by requested capability:

| Capability | Status |
|---|---|
| Flat soft loading/unloading/release | CPU regression PASS |
| Memory isolation/reset | CPU regression PASS |
| Soil-strength scaling | CPU regression PASS |
| Single rigid/soft boundary | historical tracked CUDA evidence; current GPU NOT RUN |
| CPU/CUDA consistency | NOT RUN — RESOURCE CONSTRAINT |
| Rough/irregular soft geometry | BLOCKED — no validated registration |
| Alternating patches / multi-material feet | BLOCKED — no validated registration |

Formal four-stage training is therefore blocked until the two final Stage 3
physics cases are implemented and validated on the GPU simulator. No
spring-damper substitute is permitted.
