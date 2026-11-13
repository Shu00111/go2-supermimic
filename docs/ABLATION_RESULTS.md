# Ablation results

This is a reduced-budget mechanism study, not a formal-budget comparison. A0/A1/A2/A3 each used seed 14011, 1,024 environments, and exactly 2,048,000 transitions: 1,000 flat-scan iterations followed by 1,000 alternating-material-scan iterations. Evaluation used 128 paired first episodes on command/material OOD.

| Ablation | WM parameters | Success | Progress m | One-step normalized MSE | Eight-step normalized MSE | Contact accuracy | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 generic | 2,370,641 | 1.000 | -0.280 | 0.389 | 6.669 | NA | Policy largely stood still |
| A1 structured, no memory | 2,428,053 | 0.977 | 0.049 | 0.514 | 9.193 | 0.636 | Policy largely stood still |
| A2 full structured | 2,428,053 | 0.945 | -0.375 | 0.277 | 3.369 | 0.574 | Policy largely stood still |
| A3 capacity matched | 2,428,625 | 0.094 | -0.040 | 0.677 | 9.381 | NA | Repeated rollback to iteration-1500 Actor |

A2's eight-step error was substantially lower than A1's under the matched reduced contract, consistent with loading memory helping this narrow prediction diagnostic. It is not causal evidence of better control: none of A0/A1/A2 learned meaningful progress, and A3 collapsed. Contact accuracy favored A1, so the mechanism evidence is mixed.

The initial A0 attempt consumed the transition budget but did not save a terminal checkpoint because its first collapse warning coincided with the last iteration. The failure log was retained. Artifact persistence was fixed to always save a warning-marked terminal state; A0 resumed compatibly from iteration 1750 without changing reward, distribution, or budget.
