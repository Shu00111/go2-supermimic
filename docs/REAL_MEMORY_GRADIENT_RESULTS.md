# Real memory-gradient interventions

## Result

On a fixed-action terminal normalized physical-state MSE, cutting only the loading-memory backward edge changed Full M2-v2 gradients slightly, not abnormally. Every intervention preserved the predicted physical and contact trajectories bitwise.

| condition | H | full action-grad L2 | detached L2 | cosine(detached, full) | relative difference |
|---|---:|---:|---:|---:|---:|
| rigid_rough | 2 | 6.722 | 6.720 | 1.0000 | 0.0067 |
| rigid_rough | 4 | 12.134 | 12.098 | 0.9999 | 0.0164 |
| rigid_rough | 8 | 54.243 | 54.188 | 0.9999 | 0.0134 |
| soft_rough | 2 | 12.293 | 12.299 | 1.0000 | 0.0050 |
| soft_rough | 4 | 44.515 | 44.581 | 0.9999 | 0.0139 |
| soft_rough | 8 | 65.770 | 65.869 | 0.9998 | 0.0207 |

At H=1 the variants are identical. K=1 equals always-detached under the implemented boundary convention; K=2/4/8 cut the memory edge before steps that are positive multiples of K. K>=H has no cut. Only contact channel 4 (explicit loading memory) is detached; forces, intrusion, tangential velocity, probability, model parameters, actions, rewards and forward values are unchanged.

The cosine with the independently trained No-Memory model ranged from negative to weakly positive (`-0.181` to `0.337` over the reported samples). This is not an intervention comparison: trained parameters differ, so it combines architecture and optimization history.

## Scope

This is a frozen-checkpoint, fixed-action local diagnostic, **not Actor BPTT**. It uses one development point from each of rigid_rough and soft_rough, CPU float64, and H=`1,2,4,8` (48 rows, 1.18 s). Results are in `outputs/trained_real_diagnostics/p0_memory/` and are reproduced by `scripts/run_real_memory_action_gradients.py`.

The evidence does not support H3's proposed large gradient distortion on these two local objectives. Smoother or smaller gradients would not, by themselves, demonstrate better policy optimization.
