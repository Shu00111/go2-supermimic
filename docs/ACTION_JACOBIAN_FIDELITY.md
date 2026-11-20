# Action-Jacobian Fidelity

## Result

The bounded H=1 CPU experiment passed all numerical validity gates: 24/24 action coordinates across rigid seed 82001 and soft seed 82011 were valid. Snapshot replay was bitwise exact, the `1e-3` and `3e-3` central differences agreed within the predeclared 25% tolerance, and no perturbation changed the contact/material/terminal branch. The authentic joint-position action has `clip=None`; values outside `[-1,1]` are therefore not saturation.

| State | Model | cosine | relative error | spectral-norm ratio | valid |
|---|---:|---:|---:|---:|---:|
| rigid stance | M1 | 0.040 | 20.058 | 26.540 | 12/12 |
| rigid stance | Full M2-v2 | **0.141** | **5.127** | **5.516** | 12/12 |
| rigid stance | No-Memory | 0.076 | 9.642 | 12.328 | 12/12 |
| soft stance | M1 | 0.082 | 8.156 | 12.552 | 12/12 |
| soft stance | Full M2-v2 | 0.147 | 4.208 | 5.655 | 12/12 |
| soft stance | No-Memory | **0.202** | **3.299** | **4.839** | 12/12 |

All models have poor whole-state directional agreement and substantial scale error. Full M2-v2 is not globally less accurate: it is best on the rigid state and intermediate on the soft state. The soft-state result is compatible with a smaller Full-versus-No-Memory disadvantage, not a general action-Jacobian-collapse explanation.

Soil-channel relative errors become ill-conditioned on rigid ground because the physical derivative is essentially zero; the raw and state-group columns preserve this rather than treating the ratio as meaningful.

## Scope

This report covers perturbing one action at H=1. H=2/4/8 first-action, whole-sequence, and closed-loop definitions were not mixed into the H=1 table. They remain unexecuted because the exported snapshot contract contains a validated initial relief scan, but no validated matched exogenous relief sequence for divergent physical/model trajectories. Existing exact Actor-BPTT results remain the appropriate closed-loop learned-model evidence, but are not physical derivatives.

Artifacts: `outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1/action_jacobian/`.
