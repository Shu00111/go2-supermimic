# Real Diagnostic Status

## Completed

- All three authoritative checkpoint hashes: PASS.
- Strict Actor/World Model/Critic architecture and normalization-state loading: PASS.
- Existing five-condition matched trajectory hashes and schema: PASS.
- Real M2-v2 fixed-action memory interventions, 30 rows: PASS; forward equivalence is bitwise exact.
- Targeted integration tests: PASS.
- Existing single-environment eight-scenario replay receipt: PASS; existing CUDA mixed-material receipt: FAIL as previously recorded.

## Not run or blocked

- New eight-stratum complete physical trajectory export: BLOCKED by missing complete seeds/snapshots and occupied GPU.
- Real trained-model Jacobian sweep: NOT RUN in this bounded recovery pass.
- Exact `Imagination.rollout` Actor gradients: BLOCKED by missing full rollout seed fields in the preserved dataset.
- Frozen-batch multi-objective conflict by contact phase: NOT RUN.
- Physical policy-update fidelity: BLOCKED/NOT RUN.

## Research readiness

H1 and the memory-path portion of H2 can now be run reproducibly on verified bytes and matched fixed-action physical states. H3 requires the physical policy-update protocol. H4 requires the complete rollout/snapshot dataset. H5 additionally needs historical optimizer traces or a carefully scoped frozen-batch experiment; checkpoint-local gradients cannot reconstruct training-time interaction.

The smallest useful next experiment is a single isolated trajectory export for one rigid and one soft stance, including every `Imagination.rollout` seed field and a validated snapshot. Then run H = 1, 2, 4, 8 on the shared M1 Actor through all three verified World Models before expanding to transition scenarios.
