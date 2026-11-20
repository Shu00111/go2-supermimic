# Physical Policy Improvement Results

## Result

Every learned model predicted positive improvement for every candidate update. Only 12/24 nonzero candidate/scenario pairs improved physical return (50% sign agreement), and the sign split was exactly by validation scenario for the small normalized updates: all three methods degraded rigid return and improved soft return. This is direct evidence that local imagined improvement is not a reliable physical-improvement certificate in this pilot. It is not unique to Full M2-v2.

The experiment used the same frozen M1 Actor and critic, the same two generation states, update norms `1e-5`, `3e-5`, and `1e-4`, and two independent CPU validation seeds (83001 rigid, 83011 soft), each for 32 physical steps. Zero-update replay was exact. All 26 physical rollouts were finite, survived 32/32 steps, and had zero authentic action saturation (`clip=None`).

| Update family | Model | mean physical return delta | worst delta | improvement rate |
|---|---:|---:|---:|---:|
| normalized direction | M1 | -0.000147 | -0.001391 | 3/6 |
| normalized direction | Full M2-v2 | -0.000159 | -0.000845 | 3/6 |
| normalized direction | No-Memory | -0.000085 | -0.001063 | 3/6 |
| shared-M1-Adam counterfactual | M1 | +0.016020 | -0.030387 | 1/2 |
| shared-M1-Adam counterfactual | Full M2-v2 | **-0.048581** | **-0.097659** | 1/2 |
| shared-M1-Adam counterfactual | No-Memory | +0.002994 | -0.029819 | 1/2 |

The optimizer-compatible counterfactual is the most discriminating result: Full M2-v2 predicted `+0.01360`, yet produced `+0.00050` on rigid and `-0.09766` on soft, including `-0.02045 m` soft forward-progress change. This candidate uses the shared frozen M1 Adam state so that all directions start at the same parameter vector. It is deliberately not described as Full M2-v2's historical optimizer trajectory.

No confidence interval is defensible with only two independent validation states. The candidate bytes are local/ignored; the committed manifest contains hashes, gradient norms, update norms, clipping statistics, seeds, and provenance.

Artifacts: `outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1/physical_policy_update/`.
