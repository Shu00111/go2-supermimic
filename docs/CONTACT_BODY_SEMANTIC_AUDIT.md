# Contact-Body Semantic Audit

## Result

Frozen Full M2-v2 was evaluated on 192 development samples per condition (never the held-out split). Native contact, oracle next-contact labels, memory zeroing, a +1 training-standard-deviation memory perturbation, force zeroing, and probability/intrusion zeroing were causal input interventions on the frozen network.

| Condition | Contact input | state RMSE (normalized) | contact semantic RMSE (train std) | action-Jacobian change |
|---|---|---:|---:|---:|
| rigid rough | native | 3.082 | 68.260 | reference |
| rigid rough | oracle label | 3.015 | 0.000 | 0.130 |
| rigid rough | memory zero | 3.063 | 48.260 | 0.010 |
| soft rough | native | 3.047 | 1.494 | reference |
| soft rough | oracle label | 3.044 | 0.000 | 0.160 |
| soft rough | memory zero | 3.019 | 1.510 | 0.011 |
| soft rough | probability/intrusion zero | **2.943** | 1.525 | 0.051 |

Physical-semantic mismatch remains measurable. On rigid data, near-constant physical contact channels make train-standardized native deviations extremely large; this number is evidence of nonzero internal codes where labels are near zero, but its magnitude is ill-conditioned and should not be compared literally with soft terrain. On soft terrain, native semantic error is 1.49 training standard deviations RMS.

Oracle substitution changes action sensitivity by 13.0% (rigid) and 16.0% (soft) but barely improves state RMSE. Zeroing selected channels can improve state error while moving farther from physical semantics. This is consistent with contact/body co-adaptation: the body network uses a learned internal code that is not numerically interchangeable with physical labels. It does not prove that oracle contact is harmful or that the intervention would improve physical control.

Memory alone has only about a 1.1% local action-Jacobian effect in this sample. The stronger evidence is distributed contact-code/body co-adaptation, not loading-memory gradient explosion.

Artifacts: `outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1/contact_body_semantics/`.
