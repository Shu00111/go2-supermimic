# Real-Checkpoint Diagnostic Integration

Implemented:

- Hash-before-deserialization checkpoint loader with exact method/seed/iteration and key/shape/dtype checks.
- Machine-readable full checkpoint provenance generator.
- Hash-checked frozen-trajectory adapter with nonleaking train/development/held-out partitions.
- Real M2-v2 loading-memory interventions for full gradient, backward-only detach, and truncation K=2/4/8.
- Fail-closed bundle integrity verifier.

The executed real intervention used the verified M2-v2 bytes and held-out environment 8 of the preserved alternating-material trajectory. It produced 30 rows over H = 1, 2, 4, 8, 16, 32. Every gradient-only variant preserved every forward trajectory value bitwise (`max_abs_error = 0`). Results are checkpoint-local fixed-action gradients; they are not historical training gradients and are not the exact Actor objective.

The repository's exact `Imagination.rollout` remains unchanged and authoritative. A real exact-Actor run is blocked because the recovered model-independent P1 dataset lacks the 735-dimensional Actor history, packed target-generator state, encoder bias, and environment-bound robot constants required by that interface. Substituting the earlier synthetic objective would be scientifically invalid.

Tests cover fail-closed hash/schema behavior, dataset schema and partitions, forward-preserving loading-memory detach, JVP/VJP identity, matrix-free spectral norm, finite-difference convergence, checkpoint state round trips, and snapshot mechanics.
