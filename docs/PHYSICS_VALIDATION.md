# Physics validation

The RFT implementation and provenance are documented in
`RFT_PROVENANCE_AND_VALIDATION.md`. CPU float64 fixtures validate normal and
tangential wrench, immersion, release, per-environment/per-foot memory reset,
and linear strength scaling. The tracked GPU report records an earlier CUDA
integration and mutually exclusive RFT/rigid support at a single boundary.

Development machine result (2026-11-12): 43 CPU tests pass and one tensor-CUDA
test is skipped when CUDA is intentionally hidden. The host is
an RTX 3090 machine, but it was already running an unrelated formal training
job, which was not touched. No MuJoCo Warp simulator smoke test or benchmark
was run; those are delegated to the RTX 5090. The small tensor-only RFT
CPU/CUDA parity unit in the ordinary test suite passed.

## Implemented pending RTX 5090 simulator validation

`Go2-Chase-RoughSoftTerrain` samples the same cached, bilinear
`target.terrain.HeightField` used by the world-model relief representation.
The foot-centre height defines penetration and unloading memory; every RFT
integration face samples its own `(x,y)` surface height. Flat height input is
bit-for-bit backward compatible. Loading anchors remain per environment and
per foot and acquire the new local height after motion leaves the previous
contact footprint.

`Go2-Chase-AlternatingMaterial` uses a seeded environment-local checkerboard.
The material query is per foot. Cell boundaries are rigid; RFT is active only
when the entire horizontal foot footprint is inside a soft cell. Rigid cells
are explicit collision boxes and soft cells have no rigid foot support. Thus
the RFT coefficient is exactly zero wherever a foot can touch a rigid patch.
Hard-to-soft entry begins with cleared loading memory; soft-to-hard clears it.
`Go2-Chase-MixedMaterialDiagnostic` fixes the reset pose over a known mixed
layout for the 5090 smoke test.

The bounded smoke test accumulates
`soft.any(1) & (~soft).any(1)` at every step, so its mixed-material receipt
requires feet on both materials simultaneously rather than at unrelated times.
It also reads the per-foot `feet_ground_contact` net force. With a `1e-4 N`
support threshold it rejects rigid force on soft rows, nonzero RFT on rigid
rows, and any foot receiving both rigid and RFT support.

Status by requested capability:

| Capability | Status |
|---|---|
| Flat soft loading/unloading/release | CPU regression PASS |
| Memory isolation/reset | CPU regression PASS |
| Soil-strength scaling | CPU regression PASS |
| Single rigid/soft boundary | historical tracked CUDA evidence; current GPU NOT RUN |
| CPU/CUDA RFT tensor consistency | PASS on RTX 3090; simulator parity still NOT RUN |
| Rough/irregular soft geometry | IMPLEMENTED; CPU PASS; Warp CUDA NOT RUN |
| Alternating patches / multi-material feet | IMPLEMENTED; CPU PASS; Warp CUDA NOT RUN |

Formal Stage 3 remains fail-closed until both registrations pass the bounded
RTX 5090 MuJoCo Warp checks. The implementation preserves the RFT equations;
no spring-damper substitute is used.
