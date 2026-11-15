# P1 gate results

Status: **PASS. M2-v2 is frozen for P2 at model source commit
`e8b6d9555cf0df3621c0a7994ab8fb9adcbaf005`.**

This gate evaluates the architecture selected only on the train/development
partitions. The held-out prediction partition and four new physical-gradient
seeds were opened after the model, loss, force contract, anchor weight, and
checkpoint were frozen. No result below was used to tune or replace the
candidate.

## Frozen identities

- M2-v2 checkpoint SHA-256:
  `2d9fce5279af992ecc4ae504d6b3ee5ef86797222deb5820dcba44091f788537`.
- Held-out prediction receipt SHA-256:
  `2b53251fee3ae38cc36eb570f0628c447cf8f605133f588b4b2977ad50cd8b35`.
- Held-out gradient aggregate SHA-256:
  `c9e0577dc159af6b9240603ca385c9debf8d9a1df316add4062318648b62da0c`.
- Prediction split: complete environment ids whose id modulo 10 is 8 or 9;
  15,354 valid transitions and 61,416 foot samples across five conditions.
- Gradient split: previously unseen seeds 15011--15014, rigid stable stance,
  H=32, deterministic isolated CPU replay, fixed epsilons 0.01/0.003/0.001.
  This condition was preregistered because P0 found transition and soft-contact
  H=32 derivatives mathematically unusable rather than silently treating
  censored values as ground truth.

## Held-out prediction

| Method | One step | H2 | H4 | H8 | H16 | H32 |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.31603 | 0.77640 | 1.60061 | 2.90678 | 5.54986 | 12.20205 |
| M2-v1 | 0.33168 | 0.79773 | 1.61557 | 2.97222 | 6.14556 | 13.02757 |
| M2-v2-core | 0.33169 | 0.79792 | 1.61666 | 2.97488 | 6.14809 | 13.02520 |
| M2-v2 | 0.31649 | 0.77402 | 1.59660 | 2.95711 | 6.13587 | 13.24725 |

Relative to M2-v1, M2-v2 improves one-step normalized MSE by 4.58%, H2 by
2.97%, H4 by 1.17%, H8 by 0.51%, and H16 by 0.16%. H32 worsens by 1.69%.
The H8 aggregate gain is not uniform by condition: rigid rough improves while
alternating, soft rough, and both transitions worsen modestly. This is a known
tradeoff, not a hidden claim of universal rollout superiority.

M2-v2 is essentially tied with M1 at one step (+0.15%) and is slightly better
at H2/H4, but is worse at H8/H16/H32. M1 therefore remains the strongest
long-horizon state predictor. P1 does not require M2-v2 to beat M1 on every
metric; the contact representation must instead demonstrate understood,
control-relevant value without collapse.

| Contact metric | M2-v1 | M2-v2 | Change |
|---|---:|---:|---:|
| normal-force MAE | 38.33 N | 37.79 N | -1.43% |
| tangential-vector MAE | 34.09 N | 36.05 N | +5.75% |
| intrusion MAE | 14.49 mm | 12.56 mm | -13.34% |
| loading-memory MAE | 15.82 mm | 13.79 mm | -12.84% |
| contact accuracy | 0.7513 | 0.7608 | +0.95 percentage points |

The intended physical anchoring transfers to unseen data: one-step state,
normal force, intrusion, memory, and classification improve. Tangential force
is the principal negative result and is carried into P2 interpretation.

## Held-out gradient fidelity

All four physical receipts completed without replay error, non-finite values,
action saturation, or accepted contact/constraint discontinuities. The strict
epsilon-convergence censor retained 33 of 48 action-coordinate pairs.

| Method | Cosine | Relative L2 | Sign agreement | Norm ratio |
|---|---:|---:|---:|---:|
| M1 | 0.4785 | 0.9517 | 0.6061 | 0.1114 |
| M2-v1 | 0.3880 | 0.9356 | 0.6667 | 0.2268 |
| M2-v2-core | 0.3867 | 0.9359 | 0.6667 | 0.2274 |
| M2-v2 | 0.5187 | 0.9026 | 0.7879 | 0.2294 |

M2-v2 exceeds M2-v1 on all three agreement metrics: +0.1307 cosine, -0.0331
relative L2 error, and +12.12 percentage points sign agreement. Its cosine and
relative L2 are better on every individual seed, so the aggregate is not
driven by a single state. It also exceeds M1 by +0.0402 cosine, -0.0492
relative L2, and +18.18 percentage points sign agreement.

All learned models still underestimate physical gradient magnitude. M2-v2's
norm ratio is only 0.229, so P1 establishes improved direction and sign, not
calibrated gradient magnitude. Phase-transition H=32 gradient claims remain
censored under the unchanged P0 validity rule.

## Gate audit

- **Regression tests:** pass; 63 tests after the held-out tooling fix.
- **Documented rationale:** pass; the semantic-drift diagnosis, recurrence
  correction, training-only anchor, and rejected force-range variants are
  recorded in `WORLD_MODEL_ERROR_AUDIT_V2.md`, `M2_V2_ARCHITECTURE.md`, and
  `M2_V2_DEVELOPMENT_RESULTS.md`.
- **Behavior understood:** pass; the one-step/contact/gradient gains and the
  tangential/long-horizon costs are conditionally reported.
- **No evaluation leakage:** pass; selection finished before either held-out
  marker existed, splits use complete trajectories, the candidate checkpoint
  is bound to its pre-evaluation source commit, and the one-time tools refuse
  a second opening.
- **No unexplained collapse:** pass; every prediction and model gradient is
  finite, the core variant reproduces M2-v1, and the anchored variant's
  tradeoffs are stable across development and held-out evidence.

P1 is therefore closed with a positive but qualified result. P2 may start with
the exact frozen M2-v2 architecture. Held-out P1 results must not be used for
further hyperparameter or architecture tuning.
