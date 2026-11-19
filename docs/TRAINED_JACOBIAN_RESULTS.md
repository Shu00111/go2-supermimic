# Trained finite-horizon Jacobian results

## Result

The bounded real-checkpoint sweep does **not** support the simple claim that full M2-v2 has larger common physical-state amplification than No-Memory or M1. On the two development trajectory points tested, full M2-v2 had the smallest normalized physical-to-physical spectral gain at every horizon.

| condition | method | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| rigid_rough | M1 | 89.518 | 100.197 | 135.449 | 265.850 |
| rigid_rough | Full M2-v2 | 20.264 | 26.085 | 29.761 | 43.758 |
| rigid_rough | No-Memory | 83.260 | 147.239 | 196.980 | 303.563 |
| soft_rough | M1 | 39.122 | 66.241 | 141.561 | 197.994 |
| soft_rough | Full M2-v2 | 18.024 | 25.829 | 33.761 | 48.416 |
| soft_rough | No-Memory | 52.890 | 72.551 | 149.455 | 247.448 |

These are matrix-free power-iteration estimates in nondimensional coordinates. Physical coordinates use the condition's train-partition standard deviation with a `1e-3` floor; contact coordinates use fixed SI scales; actions use unit scale.

## Memory-specific coupling

For oracle-label initialization, Full M2-v2's normalized loading-memory-to-body gain was `0.404, 0.622, 0.490, 0.525` on rigid and `0.889, 1.894, 4.195, 5.897` on soft terrain for H=`1,2,4,8`. The corresponding No-Memory derivative is exactly zero by construction. Thus the learned memory path is measurable and grows on this soft sample, but it does not make the total physical gain larger than the ablation.

Estimator initialization produced similar Full M2 physical gains: rigid `20.676, 28.320, 34.193, 47.167`; soft `18.376, 25.468, 33.067, 48.492`. The estimator GRU is evaluated once by `initial_contact`; its 84-dimensional recurrent state is not propagated by `Imagination.rollout`. Only the predicted 32-dimensional contact state is recurrent in imagination.

## Numerical checks and scope

- 40 rows: 3 models, two conditions, H=`1,2,4,8`, and both M2 initialization conventions.
- One development trajectory point per condition (`environment=6`, `start=64`); no claim of statistical significance.
- CPU float64, four power iterations, 66.45 s. All gradients were finite.
- JVP/VJP adjoint relative errors were normally near `1e-15`; the worst was approximately `2e-12`.
- Checkpoints were loaded strictly after SHA-256 verification. Results are in `outputs/trained_real_diagnostics/p0_jacobian/`.

Reproduce with `scripts/run_trained_jacobian.py`; its CLI requires all three checkpoint paths and `CONDITION=PATH=SHA256` dataset specifications.
