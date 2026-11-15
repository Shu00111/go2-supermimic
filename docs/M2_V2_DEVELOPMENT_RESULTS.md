# M2-v2 controlled development results

Status: **development complete; frozen candidate passed the one-time held-out
prediction and gradient benchmark.**

## Protocol

Five fixed-policy datasets cover rigid rough, soft rough, hard-to-soft,
soft-to-hard, and alternating materials. Each contains 32 complete
environment trajectories of 512 transitions. Environment id modulo 10 defines
disjoint partitions: 0--5 train, 6--7 development, and 8--9 held out. No
held-out row was read during the experiments below.

All methods use seed 14011, 1,200 optimizer updates, batch size 512, eight-step
windows, three 256-unit hidden layers, learning rate 2e-4, and the same common
normalization fitted on the train partition. Checkpoints are selected by the
fixed development score `one-step + eight-step state MSE`.

## Frozen comparison

| Method | One-step MSE | Eight-step MSE | Score | Normal MAE | Tangential MAE | Memory MAE | Contact acc. | Wall time | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.37344 | 3.08991 | 3.46335 | -- | -- | -- | -- | 37.5 s | 199,505 |
| M2-v1 | 0.39517 | 3.19610 | 3.59127 | 41.49 N | 36.77 N | 15.94 mm | 0.74467 | 95.4 s | 232,341 |
| M2-v2-core | 0.39514 | 3.19605 | 3.59119 | 41.57 N | 36.74 N | 15.64 mm | 0.74445 | 89.4 s | 232,341 |
| M2-v2 anchored | 0.37733 | 3.21936 | 3.59669 | 41.57 N | 39.06 N | 13.80 mm | 0.75233 | 133.8 s | 232,341 |

Relative to M2-v1, the anchored candidate improves one-step normalized MSE by
4.51%, loading-memory MAE by 13.44%, and contact accuracy by 0.77 percentage
points. Normal-force MAE changes by only +0.08 N. Eight-step state MSE worsens
by 0.73% and tangential-vector MAE worsens by 6.21%. Relative to the core
variant, the same direction and magnitude remain, attributing the development
change to the training-only anchor rather than the recurrence correction.

The anchor adds no parameters or deployment computation. Its 49% development
training-time increase versus M2-v1 comes from the auxiliary training path.
M2-v2 deployment has the same parameter count and graph structure as the core
variant.

## Preserved negative variants

Every unsuccessful run remains under
`outputs/contact_structured_supermimic/p1_audit_v2/`.

| Attempt | Change | Core score | Anchored score | Anchored normal / tangent MAE | Conclusion |
|---|---|---:|---:|---:|---|
| 1 `linear_force` | unbounded linear force decoder | 3.59086 | 3.58331 | 46.43 / 40.86 N | slightly better aggregate score, but force semantics degrade and predictions are unbounded |
| 2 `wide_coupled_scale` | ±250 N tangent, ±600 N normal; widened normalization | 3.59247 | 3.56737 | 72.45 / 55.13 N | weaker normalized force loss causes severe force degradation |
| 3 `decoupled_force` | wide bounds with original-strength normalized loss | 3.59612 | 3.59224 | 51.62 / 40.29 N | force remains worse and rollout does not improve |
| 4 `original_force_provisional` | original bounds plus recurrence and anchor, dirty source tree | 3.59119 | 3.59669 | 41.57 / 39.06 N | scientifically useful but not accepted as formal evidence because its contract recorded the preceding commit |

Attempt 4 was reproduced bit-for-bit after committing the exact model and test
changes. The canonical `development_v2` contracts bind both v2 checkpoints to
source commit `e8b6d9555cf0df3621c0a7994ab8fb9adcbaf005`; the anchored checkpoint
SHA-256 is
`2d9fce5279af992ecc4ae504d6b3ee5ef86797222deb5820dcba44091f788537`.

## Freeze decision

The frozen M2-v2 candidate is the original force contract plus corrected
memory recurrence and the 0.25 training-only anchor. This decision prioritizes
the diagnosed physical-semantic failure, preserves force accuracy, and avoids
tuning for aggregate MSE alone. The later held-out gate reproduced its
one-step/contact benefits and improved physical-gradient direction and sign;
the complete qualified result is in `P1_GATE_RESULTS.md`.
