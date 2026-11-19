# Actor BPTT gradient diagnostics

## Status

`TRAINED-POLICY RESULT: NOT MEASURED.` Frozen actor/world-model bytes and common
physical seed states are unavailable locally. `actor_gradient_horizon.csv` and
`memory_gradient_interventions.csv` contain explicitly labeled untrained-model
smoke results only.

The pipeline uses one common actor initialization for all world models and a
stable parameter order. It records imagined return, Actor-gradient L2 norm,
cosine versus H=1, cosine versus no-memory, per-step contributions, a shared
clip rule, near-zero/non-finite flags and sample counts for H = 1, 2, 4, 8, 16,
32. It also records the gradient norm with respect to the common first action,
which does not depend on Actor parameterization. This smoke objective is
intentionally minimal and is not the formal reward.
The real execution must call the repository's exact `Imagination.rollout` on
frozen replay seeds.

The interventions preserve forward contact values:

- original recurrence;
- loading-memory coordinate blocked only in the backward pass;
- memory-history truncation at K = 2, 4 and 8 boundaries.

Unit tests verify bit-identical forward outputs for the single-step cut and
verify that gradients remain on non-memory contact channels. The smoke receipt
also records identical forward returns. These tests validate implementation
semantics, not whether detachment improves a trained Actor's gradients.

Gradient norm alone is never treated as gradient quality. A trained run must
add the common-first-action derivative and physical finite-difference or paired
policy-update evidence before judging H1 or H3.
