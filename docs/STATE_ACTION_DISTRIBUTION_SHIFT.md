# State and Action Distribution Shift

## Result

Distribution shift is measurable in the retained final replay windows. A deterministic 8,192-transition sample from each frozen checkpoint was compared against the M1 final-replay reference, plus 3,072 development transitions from each independent fixed-policy condition.

| Source | state z-RMS | M1 1–99% coverage | action z-RMS | static fraction (`speed < 0.1`) | mean speed |
|---|---:|---:|---:|---:|---:|
| M1 final replay | 1.000 | 0.980 | 1.000 | 0.004 | 1.489 |
| Full M2-v2 final replay | **1.842** | **0.857** | **1.405** | **0.693** | **0.123** |
| No-Memory final replay | 0.984 | 0.976 | 0.865 | 0.010 | 1.140 |
| fixed-policy rigid development | 1.228 | 0.950 | 1.261 | 0.078 | 1.240 |
| fixed-policy soft development | 2.045 | 0.933 | 1.001 | 0.002 | 2.275 |

The collapsed Full M2-v2 Actor occupies a markedly static region: 69.3% of sampled states are below 0.1 m/s versus 0.4% for M1 and 1.0% for No-Memory. Its action and state distributions are also farther from the M1 retained replay reference. This supports H3 and confirms that static states are rare in the comparison replay.

This is a partial historical reconstruction. Each checkpoint retains a 1,000-step ring per 1,024 environments, not the full training history. Raw final-evaluation trajectories and intermediate feature occupancies were not preserved, so aggregate evaluation summaries cannot supply feature distances or prediction-error-by-distance curves. No single distance is treated as definitive OOD proof.

The environment has no action clip, so authentic saturation is zero; the separate `|action| >= 1` exceedance rate is descriptive, not saturation.

Artifacts: `outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1/distribution_shift/`.
