# Single-seed formal result (seed 14011)

M1 and M2 completed the frozen 12,000-iteration, 12.288M-transition contract on one RTX 3090. Scalar-event spans were 1.58 GPU-hours for M1 and 1.81 GPU-hours for M2 (3.39 total, excluding evaluation). Stage 4 completed 4,608 paired first episodes across six OOD conditions. Checkpoint identities are in `frozen_checkpoint_manifest_20261114.json`.

M2 improved forward progress in every Stage-4 condition and reduced measured actuator power, but did not produce a general survival, slip, or world-model prediction advantage. Its only statistically resolved success gain was on rigid-rough OOD. The reduced A2-vs-A1 study favored loading memory on eight-step state prediction, but learned policies were near-static and therefore do not establish a control benefit. Gradient fidelity remains explicitly blocked by the missing validated simulator+soil snapshot/restore adapter.

Failures were preserved: the original Stage-1 pilot numerical failure, the M2 formal collapse rollback, the A0 terminal-checkpoint persistence failure, and repeated A3 reduced-study collapse rollbacks.
