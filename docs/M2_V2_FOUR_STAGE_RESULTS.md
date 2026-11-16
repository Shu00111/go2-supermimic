# M2-v2 frozen four-stage results

Status: **Stage 4 complete for the user-approved seed-14011 five-method
matrix.** Ninety method/condition/evaluation-seed jobs produced 11,520 first
episodes without runtime failure.

## Frozen evaluation protocol

The six OOD conditions are rigid rough, soft strength, hard-to-soft,
soft-to-hard, initial-state, and command/material OOD. Each method uses three
paired evaluation seeds (90101, 90103, 90107) and 128 first episodes per seed
and condition, or 384 episodes per method/condition. Policy execution is
deterministic, observation corruption is disabled, and Stage 4 performs no
training or checkpoint selection.

Intervals in the machine-readable tables are episode-level paired intervals
within one training seed. Binary outcomes use exact McNemar tests, continuous
outcomes use paired t tests, and reported corrected probabilities use Holm
adjustment across the declared comparison family. They do **not** estimate
across-training-seed uncertainty.

## Success and forward locomotion

| Condition | M1 success | M2-v1 | M2-v2 | M2-v2 no memory | M3 |
|---|---:|---:|---:|---:|---:|
| rigid rough | 78.65% | 53.12% | 70.05% | 72.92% | 69.53% |
| soft strength | 64.58% | 65.89% | 72.14% | **94.01%** | 88.80% |
| hard-to-soft | **90.10%** | 69.01% | 56.25% | 85.68% | 87.76% |
| soft-to-hard | 78.39% | 51.30% | 61.46% | 83.33% | **85.94%** |
| initial state | 41.67% | 32.55% | **60.42%** | 44.53% | 33.85% |
| command/material | 42.19% | 28.39% | **62.24%** | 41.41% | 23.96% |

| Condition | M1 progress (m) | M2-v1 | M2-v2 | M2-v2 no memory | M3 |
|---|---:|---:|---:|---:|---:|
| rigid rough | 30.58 | 25.64 | **-0.01** | 30.13 | 28.94 |
| soft strength | 28.54 | 30.91 | **0.05** | 36.57 | 35.08 |
| hard-to-soft | 40.64 | 35.85 | **-0.03** | 39.74 | 40.21 |
| soft-to-hard | 36.02 | 30.00 | **0.02** | 38.19 | 37.55 |
| initial state | 19.86 | 19.86 | **0.03** | 21.38 | 17.92 |
| command/material | 17.20 | 14.06 | **-0.00** | 17.44 | 13.60 |

M2-v2's apparent success gains on initial-state and command OOD are survival
without locomotion: mean progress is approximately zero in every condition.
This distinction is why all-episode progress is primary and why low slip or
low instantaneous power cannot be interpreted as improved locomotion.

## Prediction and physical diagnostics

Equal-weight means over the six conditions are shown below; every condition
contains the same 384 episodes.

| Method | One-step normalized MSE | Eight-step normalized MSE | Mean progress | Success | Actuator power |
|---|---:|---:|---:|---:|---:|
| M1 | **0.411** | **4.000** | 28.81 m | 65.93% | 329.85 W |
| M2-v1 | 0.514 | 4.726 | 26.05 m | 50.04% | 314.07 W |
| M2-v2 | 0.696 | 10.204 | **0.01 m** | 63.76% | **28.14 W** |
| M2-v2-no-memory | 0.502 | 4.809 | **30.57 m** | **70.31%** | 278.07 W |
| M3 | 0.503 | 5.341 | 28.88 m | 64.97% | 260.20 W |

M2-v2 reduces mean contact-force MAE from 47.79 N (M2-v1) to 31.71 N,
improves contact accuracy from 0.766 to 0.832, and modestly reduces mean
loading-memory MAE from 10.98 to 10.42 mm. These contact-semantic gains did
not prevent final-training collapse. Its mean actuator work per path is 261.23
J/m, worse than M2-v1 (182.17 J/m), no-memory (181.24 J/m), and M3 (168.64
J/m); the very low absolute power reflects inactivity rather than efficiency.

## Statistical interpretation

Relative to M1, M2-v2 has significantly lower progress in all six conditions
after Holm correction. Its success is lower on both transition conditions and
higher on initial-state and command OOD; rigid-rough and soft-strength success
differences do not survive the full correction family. M2-v2's eight-step
error is higher in five conditions after full correction; soft-strength does
not survive the family-wide adjustment.

All episode rows, aggregates, per-evaluation-seed summaries, exact paired
comparisons, and figures are under
`outputs/contact_structured_supermimic/formal_v2/evaluation/stage4_seed14011_v2/`:

- `aggregate.csv`
- `per_evaluation_seed.csv`
- `paired_vs_m1.csv`
- `paired_all_methods.csv`
- `stage4_v2_comparison.png`
- `stage4_v2_contact_diagnostics.png`
- `summary_manifest.json`

The central Stage-4 result is negative: the frozen M2-v2 diagnostic gains did
not transfer to a useful locomotion policy under the matched full budget.
