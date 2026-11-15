# World-model error audit v2

Status: **P1 attribution complete on the development split; M2-v2 controlled
development is in progress and full policy training remains gated.**

This audit is the required diagnostic stage between the completed P0 gradient-fidelity gate and any M2-v2 implementation.  It uses the frozen formal-seed-14011 M1/M2-v1 checkpoints and the matched Stage-4 evaluation protocol.  The existing episode aggregates are evidence for locating failure modes, not a substitute for the transition-level held-out audit described below.

## Frozen evidence and scope

- M1 checkpoint SHA-256: `51aaeacc...` (full digest remains in the formal evaluation manifests).
- M2-v1 checkpoint SHA-256: `29f59646...` (full digest remains in the formal evaluation manifests).
- Stage-4 evaluation: 384 episodes per method/condition, paired seeds, six OOD conditions.
- Existing metrics: one-step normalized state/pose MSE, eight-step normalized state MSE, M2-v1 force/sinkage/loading-memory error and contact accuracy.
- Actor observations, rewards, termination, low-level controller, BPTT horizon and task distribution are frozen for P1 comparisons.

## Preliminary aggregate finding

M2-v1's central contact representation did **not** improve predictive fidelity.  Its one-step normalized MSE is higher than M1 in all six Stage-4 conditions; the paired difference is significant in every condition.  Its eight-step error is also higher in all six conditions, significantly so in hard-to-soft, soft-to-hard, initial-state and command OOD; the rigid-rough and soft-strength confidence intervals include zero.

This is compatible with M2-v1 improving some policy-level outcomes through a useful inductive bias while still being a worse world model.  It is not evidence that the contact pathway is causally responsible for those policy outcomes.

| Condition | M1 one-step | M2-v1 one-step | M1 eight-step | M2-v1 eight-step |
|---|---:|---:|---:|---:|
| rigid rough OOD | 0.285 | 0.338 | 3.154 | 3.366 |
| soft strength OOD | 0.163 | 0.237 | 2.177 | 2.812 |
| hard to soft OOD | 0.375 | 0.471 | 3.035 | 4.282 |
| soft to hard OOD | 0.338 | 0.410 | 2.669 | 3.902 |
| initial state OOD | 0.563 | 0.642 | 4.606 | 6.424 |
| command OOD | 0.691 | 0.758 | 5.593 | 7.463 |

M2-v1 contact diagnostics also degrade at material transitions and demanding commands.  Contact accuracy ranges from 0.721 (hard-to-soft) to 0.821 (rigid rough); force-vector MAE ranges from 37.7 N to 53.2 N.

## Frozen transition-level audit

Five model-independent trajectory sets were collected with the same frozen M1 actor: rigid rough, rough soft, hard-to-soft, soft-to-hard and alternating material. Each contains 32 environment trajectories and 512 transitions per environment. Seeds 71001/71011/71021/71031/71041 are disjoint from the Stage-4 seeds. Complete environment ids, rather than individual transitions, define the split: id modulo 10 in 0–5 is train, 6–7 is development and 8–9 is held out. The held-out rows have not been opened.

Machine-readable results are in `world_model_error_audit_dev_v2.json`/`.csv`; the weighted summary and figure are generated from those files. The audit contains 15,351 valid development transitions (61,404 foot samples). It reports normal/tangential forces, onset/stance/release/flight, near/far boundary, material, soil strength and force-tail strata, plus autoregressive horizons 1/2/4/8/16/32.

### Matched prediction error

The common normalization is estimated once from the train partition and shared by both models. M2-v1 is worse at every measured horizon:

| Horizon | M1 state MSE | M2-v1 state MSE | M2-v1 / M1 |
|---:|---:|---:|---:|
| 1 | 0.227 | 0.369 | 1.62× |
| 2 | 0.470 | 0.809 | 1.72× |
| 4 | 1.057 | 1.785 | 1.69× |
| 8 | 2.130 | 3.465 | 1.63× |
| 16 | 4.621 | 7.979 | 1.73× |
| 32 | 10.466 | 26.234 | 2.51× |

The teacher-forced one-step decomposition is also uniformly worse: base/pose/velocity 0.428 versus 0.266, joints 0.336 versus 0.174, limbs 0.265 versus 0.149 and soil-state channels 0.726 versus 0.603. This rules out an aggregate metric hiding a localized M2-v1 advantage.

### Contact estimator, transition and memory

| Path | Normal-force MAE | Tangential-vector error | Memory MAE | Contact accuracy |
|---|---:|---:|---:|---:|
| zero-state estimator | 47.09 N | 58.12 N | 10.16 mm | 0.619 |
| persistent GRU estimator | 47.16 N | 60.00 N | 10.17 mm | 0.619 |
| transition from predicted current contact | 42.82 N | 26.34 N | 12.03 mm | 0.692 |
| transition from oracle current contact | 81.43 N | 66.63 N | 30.81 mm | 0.514 |

Persisting the estimator GRU across physical steps produces no measurable benefit. The explicit reset implementation is exact and regression-tested; the error is not reset leakage. Contact propagation error remains roughly bounded through horizon 32 (normal-force MAE 42.97 N, memory MAE 11.30 mm), while state MSE grows to 26.23. Therefore long-horizon collapse is dominated by contact-conditioned body dynamics, not by monotonically exploding contact memory alone.

The fixed force decoder is also physically under-ranged. It clips every component to ±150 N, but the development targets have normal-force p95 values of 459 N on rigid rough, 226 N at hard-to-soft and 222 N on alternating material; rare impacts exceed 2 kN. This is a structural saturation, not merely a loss-weight choice.

### Oracle counterfactual and dominant failure

Replacing predicted contact with physical oracle contact does not rescue M2-v1. One-step common-normalized error rises from 0.358 to 0.732 when the transition starts from oracle current contact and to 2.089 when the body receives oracle next contact. The largest oracle-next degradation is in joints (4.288 versus 0.336) and soil state (3.410 versus 0.726).

This counterfactual must not be read as “oracle contact is harmful.” The transition and body networks were trained only on their own predicted-contact distribution. They have co-adapted the nominally physical channels into an internal code whose numerical values are not interchangeable with the simulator labels. The dominant diagnosed failure is therefore **loss of physical semantic anchoring plus contact-conditioned body co-adaptation**, compounded by force clipping and an inconsistent loading-memory recurrence. M2-v1's transition decoder uses the same scalar as both a memory proposal and a retention factor, whereas its estimator uses intrusion plus an independent retention factor.

## M2-v2 change justified by the audit

The controlled candidate is intentionally narrow:

1. replace the uniform ±150 N force range with measured, separately normalized
   ±250 N tangential and ±600 N normal ranges while retaining bounded `tanh`
   decoding for numerical stability;
2. make transition memory use the same intrusion-plus-retention recurrence as the estimator and the intended loading-path semantics;
3. add an auxiliary oracle-anchored contact/body path during world-model fitting while retaining the ordinary fully predicted rollout loss. Oracle labels remain training-only and never enter actor observations or deployment-time estimation;
4. keep actor, critic, observations, rewards, termination, controller, BPTT horizon, task distribution and real-data protocol unchanged.

The auxiliary path directly tests the diagnosed semantic-drift mechanism. A core-only variant (items 1–2) and the anchored candidate (items 1–3) will be compared so any improvement is not attributed to a bundle of untested changes.

## Split and leakage contract

- Use a fixed-policy collection with simulator snapshots and oracle contact labels; do not draw model-selection evidence from the final Stage-4 evaluation episodes.
- Split by complete environment trajectory/seed, never by individual transition, so adjacent windows cannot cross partitions.
- Development data may be used for diagnosis and M2-v2 selection.  The held-out partition is opened once for the P1 gate and then frozen for P2.
- Simulator contact, terrain material and boundary distance are labels/stratifiers only.  They remain unavailable to the actor and to the deployment-time contact estimator.
- Reliability/censoring annotations may describe the audit but may not weight, mask or gate the policy BPTT objective.

## P1 gate (not yet passed)

The attribution and design-rationale portions are complete. P1 remains closed until M1, M2-v1, the M2-v2 core variant and the anchored M2-v2 candidate are trained on the matched train partition, selected on the development partition, and the frozen candidate is evaluated once on held-out prediction and gradient benchmarks. No full policy training is authorized before this gate passes.
