# World-model error audit v2

Status: **P1 in progress — architecture changes and full training remain gated.**

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

M2-v1 contact diagnostics also degrade at material transitions and demanding commands.  Contact accuracy ranges from 0.721 (hard-to-soft) to 0.821 (rigid rough); force-vector MAE ranges from 37.7 N to 53.2 N.  These aggregates do not yet separate normal and tangential force, onset/release timing, rare spikes, soil strength, boundary distance, or systematic bias.

## Required transition-level audit

The next diagnostic runner must operate on a deterministic, checkpoint-independent trajectory set and freeze its split manifest before model selection.

1. **Contact force:** report normal and tangential signed bias, MAE, RMSE and high quantiles; stratify by contact onset, sustained stance, release, soft-soil strength, material boundary distance and target magnitude.  Report both sample-weighted errors and rare-event tails.
2. **Loading memory:** measure reset leakage after episode termination, retention during flight, stance accumulation, release decay, material-transition response and long-horizon drift.  Compare the explicit memory channel with the estimator GRU hidden-state persistence.
3. **Multi-step state:** report teacher-forced one-step and autoregressive horizons 1/2/4/8/16/32, decomposed into body pose/velocity, joints and feet.  Stratify by gait/contact phase and material transition.
4. **Attribution:** replay M2-v1 with oracle current contact, oracle transitioned contact, and predicted contact.  The three paths isolate estimator error, contact-transition error and contact-conditioned body-transition error without retraining.

## Split and leakage contract

- Use a fixed-policy collection with simulator snapshots and oracle contact labels; do not draw model-selection evidence from the final Stage-4 evaluation episodes.
- Split by complete environment trajectory/seed, never by individual transition, so adjacent windows cannot cross partitions.
- Development data may be used for diagnosis and M2-v2 selection.  The held-out partition is opened once for the P1 gate and then frozen for P2.
- Simulator contact, terrain material and boundary distance are labels/stratifiers only.  They remain unavailable to the actor and to the deployment-time contact estimator.
- Reliability/censoring annotations may describe the audit but may not weight, mask or gate the policy BPTT objective.

## P1 gate (not yet passed)

P1 remains closed until the transition-level audit is complete, the dominant failure is identified, and a minimal M2-v2 change is justified against that failure.  M1, M2-v1 and M2-v2 must then be trained on matched train data, selected on the same development split, and evaluated once on the frozen held-out split.  No full policy training is authorized before this gate passes.
