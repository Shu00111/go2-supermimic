# Frozen loss-gradient conflict

## Result

The exact Full M2-v2 objective was decomposed at the final seed-14011 checkpoint on hash-verified train and development batches from rigid_rough and soft_rough. Of the off-diagonal ordered loss pairs, `30.36%` had negative gradient cosine. The minimum was `-0.5149` between intrusion prediction and contact classification on rigid development data.

Other strong conflicts included loading-memory versus tangential-velocity prediction (`-0.3848`, soft development), tangential velocity versus contact classification (`-0.3643`, rigid train), and loading memory versus contact classification (`-0.3034`, rigid train). These demonstrate local competition among some supervised heads. They do not establish that the same conflicts occurred earlier in training.

The decomposition includes state/pose, normal force, tangential force, intrusion, loading memory, tangential velocity, contact classification and weighted physical anchor. Tangential velocity is retained because it is part of the actual source objective. Each comparison covers 20 shared parameter tensors / 2,428,053 elements. The exact total loss was reconstructed to maximum absolute error `1.421e-14`.

| condition | partition | exact total objective |
|---|---|---:|
| rigid_rough | train | 56.2984 |
| rigid_rough | development | 78.6103 |
| soft_rough | train | 72.5821 |
| soft_rough | development | 76.4606 |

Each stratum uses two trajectories and four transitions per trajectory (8 transitions); H=4; CPU float64; 12.33 s. No held-out or Stage-4 final set was used. Results and heatmap are under `outputs/trained_real_diagnostics/p1_loss_conflict/`; reproduce with `scripts/run_trained_loss_conflict.py`.
