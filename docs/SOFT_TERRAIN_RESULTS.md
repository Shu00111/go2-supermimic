# Soft-terrain results

Formal seed-14011 training and held-out evaluation are in progress. No claim of
successful learning or improvement over B0 is made before iteration 12,000 and
the frozen-checkpoint evaluation complete.

## Available preflight measurement

The released hard-ground Actor (B0) was run directly on uniform soft soil in 32
CUDA environments for five seconds. All values remained finite, no falls were
observed in this short window, and mean planar speed was 1.53 m/s. The same
Actor's hard-ground runtime gate was 1.91 m/s. These short rollouts are not the
required 128-episode formal comparison.

## Planned baseline disposition

- B0: original released hard-ground Actor, to be formally evaluated.
- B1: seed-14011 SuperMimic soft-terrain Actor, currently training.
- B2: NOT COMPLETED. A fair PPO run is outside the current single-seed initial
  training budget; old PPO results from another simulator will not be reused.

Final success, progress, velocity error, falls, slip, sinkage, power/work,
world-model error, and training-cost tables will be added only from actual
held-out measurements. Multi-seed aggregation is intentionally deferred by the
user.
