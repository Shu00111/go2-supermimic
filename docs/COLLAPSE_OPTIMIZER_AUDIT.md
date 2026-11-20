# Collapse-Guard and Optimizer Audit

## Observed chronology

The preserved seed-14011 formal-training log contains 23 relevant guard events:

| Method | events | first / last iteration | final LR scale | distinct rollback iterations |
|---|---:|---:|---:|---|
| M1 | 1 | 1800 / 1800 | 0.5 | 1500 |
| Full M2-v2 | **20** | **2450 / 10200** | **9.54e-7** | 2250, 2750, 3000, 3500, 4250, 4500, 5750, 7250, 7500, 10000 |
| No-Memory | 2 | 1900 / 2600 | 0.25 | 1750, 2500 |

The frozen final optimizer states corroborate the log: M1, Full, and No-Memory retain LR scales `0.5`, `9.536743e-7`, and `0.25`, respectively. Their stored Actor Adam learning rates are `4.6541e-5`, `8.8770e-11`, and `2.3271e-5`; each has eight parameter-state entries and zero recorded non-finite steps. Full's Adam step is only 1200, versus 4900 for M1 and No-Memory, which is direct evidence that rollback/freeze behavior substantially reduced accepted Actor optimization.

The first Full M2-v2 guard occurred at iteration 2450 after logged falls rose to 10.2; it rolled back to 2250 and then halved the LR. Each subsequent suppression is emitted by the guard line itself. Therefore repeated physical-collapse detections precede and create the extreme cumulative LR suppression; suppression did not precede the first collapse. By iteration 4000, repeated rollbacks around iteration 3500 had already reduced the scale to 0.00195, and later events continued through iteration 10200.

All 20 Full events have a preceding periodic Actor-gradient record. Those norms range from 1.357 to 17.93 and do not show a monotonic explosion explaining the event sequence. The observational log instead shows recurrent high-fall episodes, rollback, and exponential LR reduction. The guard plausibly amplified persistence by nearly freezing subsequent policy learning, but it was initially a response to collapse and this audit cannot establish causation.

Rollback checkpoints, pre-event world-model loss, imagined return, falls, distance, and Actor-gradient norm are preserved row-by-row in `collapse_guard_events.csv`. No training was resumed and no optimizer state was modified.

Artifacts: `outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1/collapse_optimizer/`.
