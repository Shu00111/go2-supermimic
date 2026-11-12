# Physics validation

The RFT implementation and provenance are documented in
`RFT_PROVENANCE_AND_VALIDATION.md`. CPU float64 fixtures validate normal and
tangential wrench, immersion, release, per-environment/per-foot memory reset,
and linear strength scaling. The tracked GPU report records an earlier CUDA
integration and mutually exclusive RFT/rigid support at a single boundary.

Target-machine result (2026-11-12): all 45 tests pass in the isolated project
environment. The bounded MuJoCo Warp check ran on an RTX 3090 with 16 batched
worlds, 100 control steps per task, and seed 14011. The machine-readable receipt
is `experiments/contact_structured_supermimic/physics_validation/rtx3090_stage0_20261112.json`.

## Validated mixed-material implementation

`Go2-Chase-RoughSoftTerrain` samples the same cached, bilinear
`target.terrain.HeightField` used by the world-model relief representation.
The foot-centre height defines penetration and unloading memory; every RFT
integration face samples its own `(x,y)` surface height. Flat height input is
bit-for-bit backward compatible. Loading anchors remain per environment and
per foot and acquire the new local height after motion leaves the previous
contact footprint.

`Go2-Chase-AlternatingMaterial` uses a seeded environment-local checkerboard.
The material query is per foot. Cell boundaries are rigid; RFT is active only
when the entire horizontal foot footprint plus a 1 cm predictive guard lies
inside a soft cell. Rigid cells are explicit collision boxes and soft cells
have no rigid foot support. Three consecutive eligible physics substeps are
required before activation. Thus the RFT coefficient is zero wherever a foot
can touch a rigid patch, including fast boundary crossings.
Hard-to-soft entry begins with cleared loading memory; soft-to-hard clears it.
`Go2-Chase-MixedMaterialDiagnostic` fixes the reset pose over a known mixed
layout for the validation check.

The bounded smoke test accumulates
`soft.any(1) & (~soft).any(1)` at every step, so its mixed-material receipt
requires feet on both materials simultaneously rather than at unrelated times.
It also reads the per-foot `feet_ground_contact` net force. With a `1e-4 N`
support threshold it rejects rigid force on soft rows, nonzero RFT on rigid
rows, and any foot receiving both rigid and RFT support.

Status by requested capability:

| Capability | Status |
|---|---|
| Flat soft loading/unloading/release | regression PASS |
| Memory isolation/reset | regression PASS |
| Soil-strength scaling | regression PASS |
| Single rigid/soft boundary | Warp CUDA PASS |
| CPU/CUDA RFT tensor consistency | PASS on RTX 3090 |
| Rough/irregular soft geometry | Warp CUDA PASS; max RFT 120.31 N |
| Alternating patches / multi-material feet | Warp CUDA PASS; all 16 worlds mixed |
| Fixed mixed-material diagnostic | Warp CUDA PASS; all 1600 world-steps mixed |

No checked foot received simultaneous rigid and RFT support at the `1e-4 N`
threshold. The implementation preserves the RFT equations; no spring-damper
substitute is used.
