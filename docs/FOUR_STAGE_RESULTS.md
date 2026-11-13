# Four-stage results

Status: completed for the user-scoped single training seed 14011. M1 and M2 each consumed 12,288,000 environment transitions. Stages 1--3 were training stages; Stage 4 was frozen before evaluation and did not update any trained state.

Stage 4 used 3 paired evaluation seeds and 128 first episodes per seed and condition (384 episodes per method/condition; 4,608 episodes total). Values below are all-episode means. Episode confidence intervals and paired tests are in the local `stage4_v1` artifacts.

| Stage | Terrain/condition | M1 success | M2 success | M1 progress m | M2 progress m | M1 velocity MAE m/s | M2 velocity MAE m/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Flat rigid + flat soft | trained | trained | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated |
| 2 | Rough rigid + randomized softness | trained | trained | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated |
| 3 | Rough soft + material transitions | trained | trained | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated | not frozen-evaluated |
| 4 | Rigid rough OOD | 0.701 | 0.771 | 27.270 | 33.265 | 0.818 | 0.802 |
| 4 | Soft strength OOD | 0.932 | 0.940 | 37.647 | 39.217 | 0.786 | 0.799 |
| 4 | Hard-to-soft OOD | 0.828 | 0.820 | 37.049 | 40.492 | 0.910 | 0.868 |
| 4 | Soft-to-hard OOD | 0.932 | 0.906 | 37.961 | 41.353 | 0.924 | 0.946 |
| 4 | Initial-state OOD | 0.372 | 0.375 | 17.853 | 21.062 | 1.033 | 1.032 |
| 4 | Command/material OOD | 0.276 | 0.240 | 11.559 | 13.543 | 1.033 | 1.074 |

| Condition | M1 slip m/s | M2 slip m/s | M1 sinkage m | M2 sinkage m | M1 power W | M2 power W | M1 work/path J/m | M2 work/path J/m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rigid rough OOD | 3.006 | 3.362 | NA | NA | 333.236 | 317.653 | 249.292 | 218.049 |
| Soft strength OOD | 1.977 | 1.917 | 0.025 | 0.025 | 282.029 | 267.070 | 140.232 | 132.573 |
| Hard-to-soft OOD | 2.659 | 2.741 | 0.033 | 0.032 | 292.535 | 279.431 | 133.797 | 122.006 |
| Soft-to-hard OOD | 2.547 | 2.644 | 0.026 | 0.025 | 293.019 | 282.839 | 137.099 | 127.243 |
| Initial-state OOD | 2.533 | 2.567 | 0.026 | 0.025 | 341.117 | 306.694 | 212.137 | 179.273 |
| Command/material OOD | 2.298 | 2.277 | 0.033 | 0.032 | 348.666 | 310.273 | 263.565 | 215.993 |

M2 increased progress in all six conditions and significantly increased success only on rigid-rough OOD (+7.03 percentage points, paired 95% CI [+1.48, +12.58]). It did not establish a survival advantage elsewhere and increased rigid/transition slip in several cases. M2 reduced actuator power in all conditions, but absolute joint work is not battery energy.

Across conditions, M1 had lower mean normalized one-step/eight-step errors (0.403/3.539) than M2 (0.476/4.708). M2 contact diagnostics averaged 48.09 N force MAE, 8.40 mm sinkage MAE, 10.21 mm memory MAE, and 0.770 contact accuracy. The evidence does not support a general prediction-fidelity advantage for M2.
