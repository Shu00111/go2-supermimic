# Soft-terrain results

## Protocol

B0 is the released hard-ground Actor (`weights/flat.pt`). B1 is the frozen
seed-14011 Actor at iteration 12,000. Eight conditions use 128 paired held-out
episodes per method (2,048 episodes total), fixed evaluation seeds 91001-91008,
20 s time limits, chase-only targets, identical initialization/domain
randomization within each B0/B1 pair, deterministic Actors, and observation
corruption disabled. Success means reaching 20 s without `fell_over`.

Seen coefficient scales are 0.3, 0.6, and nominal 1.0. Scales 0.2 and 1.2 are
outside the training range. `Delta` is B1 minus B0. Work values below are
all-episode means; each JSON also contains successful-episode-only power/work.

| Condition | Success B0 | Success B1 | Delta pp | Progress B0 m | Progress B1 m | Delta | Work B0 J/m | Work B1 J/m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard | 100.00% | 92.19% | -7.81 | 48.50 | 46.76 | -3.59% | 112.16 | 132.08 |
| Soft 1.0 | 100.00% | 98.44% | -1.56 | 48.21 | 48.02 | -0.40% | 90.06 | 110.96 |
| Seen 0.3 | 100.00% | 100.00% | 0.00 | 48.33 | 51.47 | +6.49% | 90.35 | 88.00 |
| Seen 0.6 | 100.00% | 100.00% | 0.00 | 48.91 | 49.89 | +2.00% | 91.27 | 92.74 |
| Hard-to-soft | 99.22% | 93.75% | -5.47 | 47.20 | 46.90 | -0.64% | 97.20 | 101.66 |
| Soft-to-hard | 97.66% | 95.31% | -2.34 | 46.45 | 48.61 | +4.65% | 98.68 | 105.03 |
| Unseen 0.2 | 100.00% | 100.00% | 0.00 | 46.63 | 51.12 | +9.62% | 92.23 | 86.72 |
| Unseen 1.2 | 100.00% | 98.44% | -1.56 | 47.97 | 46.93 | -2.17% | 89.08 | 103.30 |

## Findings

The result does not support a blanket claim that soft-terrain training improves
the released Actor. It shows a terrain-dependent speed/stability trade-off:

- On soft `cs=0.3`, B1 increased paired forward progress by 3.14 m (95% CI
  2.36-3.91 m), reduced slip by 0.163 m/s, and reduced work/path by 2.35 J/m;
  both methods survived all episodes.
- On unseen softer `cs=0.2`, B1 increased progress by 4.49 m (95% CI
  3.74-5.23 m), reduced slip by 0.099 m/s, and reduced work/path by 5.51 J/m;
  both methods again survived all episodes.
- On hard ground B1 fell in 10 episodes while B0 fell in none (paired McNemar
  exact p=0.00195). On hard-to-soft, B1 lost seven paired successes and gained
  none (p=0.0156).
- Soft-to-hard exhibits a direct trade-off: B1 progresses 4.65% farther but
  falls in 6/128 episodes versus B0's 3/128 and has higher contact-foot slip.
- At nominal 1.0 and unseen stronger 1.2, B1 has no progress benefit and uses
  materially more work per path; it also has two falls in each condition.

The same frozen world model was evaluated on both policies' trajectories.
One-step normalized error is lower on B1 trajectories for both seen randomized
strengths, both transitions, and both unseen strengths. Eight-step error follows
the same direction except that transition confidence intervals are wide;
nominal 1.0 is worse. These are trajectory-conditioned errors, not two separate
world-model baselines.

## Scope and disposition

- B0: COMPLETED.
- B1: COMPLETED for one training seed, as explicitly requested.
- B2 PPO: NOT COMPLETED; no old result from another simulator is reused.
- Multi-seed inference: NOT PERFORMED by user request. Episode dispersion and
  paired CIs do not replace across-training-seed uncertainty.
- Videos: not generated in this evaluation because the request was checkpoint
  freezing and held-out measurement; no video is presented as quantitative
  evidence.

Raw episode CSVs, per-condition JSON summaries, paired tests, plots, protocol,
and SHA-256 checksums are under
`outputs/soft_terrain/seed_14011/evaluation/held_out_v1/`.
