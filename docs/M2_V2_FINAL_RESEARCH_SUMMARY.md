# SuperMimic v2 final research summary

Status: **research pipeline complete under the user-approved single-training-
seed scope; original three-seed P2 acceptance remains PARTIAL.**

## A. P0: gradient fidelity

Deterministic snapshot/restore passes on isolated single-environment CPU
replay. CUDA replay is not equivalent enough to define physical finite-
difference truth around mixed contact. Centered finite differences at epsilon
0.01, 0.003, and 0.001 were characterized for stable stance, aerial,
touchdown, liftoff, soft contact, and both material transitions.

Only 29 action-coordinate pairs from four rigid-stable H=32 states passed all
replay, signal, epsilon-convergence, and contact-topology checks. On those
pairs, M1/M2-v1 cosine was 0.377/0.498, relative L2 error 0.968/0.868, and sign
agreement 0.552/0.690. M2-v1 was directionally better in this constrained
stratum, while both models underestimated magnitude. Soft and transition
H=32 gradients were nonsmooth or numerically unusable and were censored rather
than converted into claims.

## B. P1: M2-v2

The audit identified contact-semantic drift and unstable memory recurrence.
M2-v2 keeps the deployable proprioceptive contact estimator, corrects loading-
memory recurrence, and adds a 0.25-weight training-only physical anchor. It
adds no deployment parameters or privileged actor inputs.

On the one-time held-out P1 gate, M2-v2 improved over M2-v1 in one-step MSE
(4.58%), normal-force MAE (1.43%), intrusion MAE (13.34%), loading-memory MAE
(12.84%), contact accuracy (+0.95 points), gradient cosine (+0.1307), relative
gradient error (-0.0331), and sign agreement (+12.12 points). H32 state error
and tangential-force error worsened. This was a qualified diagnostic success,
not a guarantee of policy improvement.

## C. Retraining and Stage 4

All five methods completed the matched 12.288M-transition seed-14011 budget.
M2-v2 suffered 20 collapse-guard events and ended at a learning-rate scale of
9.54e-7. Frozen Stage 4 completed 11,520 episodes: five methods, six OOD
conditions, three paired evaluation seeds, and 128 first episodes per cell.

M2-v2 improved mean contact-force MAE (31.71 N versus 47.79 N for M2-v1) and
contact accuracy (0.832 versus 0.766), but its actor had approximately zero
progress in every condition. Its mean eight-step error was 10.204 versus 4.000
for M1. Apparent survival successes and 28.14 W actuator power are explained
by inactivity; work per path remained poor at 261.23 J/m. The representation
improvement did not transfer to useful locomotion.

## D. Ablations

M2-v2-no-memory completed with two guard events, 30.57 m mean progress, 70.31%
mean success, and 4.809 eight-step error. It was better than full M2-v2 in
progress and eight-step error in all six conditions, although its force MAE
and contact accuracy were worse. Capacity-matched M3 tracked M1 much more
closely, so model size alone does not explain the collapse.

The measured pattern is a tradeoff: persistent contact structure improves
some physical labels and rigid-state gradient direction, but the chosen
memory/anchor optimization can destabilize full policy training. The current
data do not establish which subcomponent is causal.

## E. Research conclusion

The evidence is **not sufficient for a robotics conference paper claiming
that M2-v2 improves complex/soft-terrain locomotion**. The scientifically
defensible result is negative and potentially useful: improved contact
semantics and local gradient alignment can coexist with worse long-horizon
prediction and catastrophic policy-training dynamics.

A publishable follow-up would require a stable optimization redesign frozen
before evaluation, at least three complete training seeds, replication of the
no-memory advantage, and gradient diagnostics that remain valid beyond rigid
stable contact. The present single-seed full matrix supports mechanism
diagnosis, not a population-level performance claim.

## Acceptance ledger

- P0 snapshot/replay and qualified Gradient Fidelity: **COMPLETE**.
- P1 audit, M2-v2 implementation, tests, held-out prediction/gradient gate:
  **COMPLETE**.
- Full-budget seed14011 five-method matrix: **COMPLETE**.
- Original three-training-seed matrix: **PARTIAL by user scope decision**;
  seed24017 M1/M2 completed, seed24017 M2-v2 interrupted, seed34019 not run.
- Frozen Stage 4 for seed14011: **COMPLETE**, 90 jobs and 11,520 episodes.
- Full-budget ablations: **COMPLETE for seed14011**.
- Statistical comparisons: **COMPLETE for paired evaluation episodes**;
  across-training-seed uncertainty unavailable.
- Checkpoint/source hashes and raw artifacts: **RECORDED**.
- Representative videos: **NOT RUN**; no claim relies on fabricated media.
- Regression suite: **COMPLETE**, 66 tests passed (one upstream deprecation
  warning).
- GitHub publication: **COMPLETE** on
  `research/contact-structured-supermimic-v2`.

## Artifact index

- P0: `docs/GRADIENT_FIDELITY_RESULTS_V2.md`
- P1 audit: `docs/WORLD_MODEL_ERROR_AUDIT_V2.md`
- architecture: `docs/M2_V2_ARCHITECTURE.md`
- development: `docs/M2_V2_DEVELOPMENT_RESULTS.md`
- training: `docs/M2_V2_TRAINING_REPORT.md`
- Stage 4: `docs/M2_V2_FOUR_STAGE_RESULTS.md`
- ablations: `docs/M2_V2_ABLATION_RESULTS.md`
- raw P3 artifacts:
  `outputs/contact_structured_supermimic/formal_v2/evaluation/stage4_seed14011_v2/`
