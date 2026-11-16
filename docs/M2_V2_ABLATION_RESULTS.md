# M2-v2 full-budget ablation results

Status: **complete for seed14011; across-training-seed replication not run.**

## Matched methods

| Method | Contact structure | Persistent loading memory | Capacity role |
|---|---|---|---|
| M1 | no | no | original generic baseline |
| M2-v1 | yes | yes | historical architecture, fresh matched run |
| M2-v2 | yes | yes | frozen recurrence plus training anchor |
| M2-v2-no-memory | yes | no | structured full-budget ablation |
| M3 | no | no | parameter-matched generic control |

Every method completed the same 12.288M-transition seed-14011 contract and the
same 2,304 Stage-4 episodes. Parameter matching does not imply compute
matching: measured training spans were 1.55 h for M1, 1.74 h for M2-v1, 2.09 h
for M2-v2, 2.12 h for no-memory, and 1.57 h for M3.

## Loading-memory ablation

The no-memory model outperforms full M2-v2 in forward progress by 17.44--39.77
m across all six conditions; all paired intervals exclude zero after Holm
correction. It also raises success on soft strength (+21.88 points),
hard-to-soft (+29.43), and soft-to-hard (+21.88), but lowers success on
initial-state (-15.89) and command OOD (-20.83). The rigid-rough success
difference (+2.86) is inconclusive.

| Quantity, mean over six conditions | M2-v2 | No-memory | Interpretation |
|---|---:|---:|---|
| forward progress | 0.01 m | **30.57 m** | full M2-v2 actor is static |
| success | 63.76% | **70.31%** | survival alone hides static behavior |
| one-step normalized MSE | 0.696 | **0.502** | no-memory better overall |
| eight-step normalized MSE | 10.204 | **4.809** | no-memory better in every condition |
| contact-force MAE | **31.71 N** | 55.62 N | full memory/anchor improves semantics |
| contact accuracy | **0.832** | 0.774 | full memory/anchor improves classification |
| actuator absolute power | 28.14 W | 278.07 W | low M2-v2 power is inactivity |
| work per path | 261.23 J/m | **181.24 J/m** | no efficiency gain for M2-v2 |

Thus the persistent-memory path is associated with better contact labels but
also with the collapse that destroyed useful control in this seed. The data do
not isolate whether recurrence, auxiliary anchoring, or their optimization
interaction is causal; 20 guard events for M2-v2 versus two for no-memory are
strong failure evidence but not a randomized component-level causal proof.

## Capacity and historical controls

M3 reaches 28.88 m mean progress and 64.97% success, close to M1's 28.81 m
and 65.93%. This shows that parameter count alone does not explain the M2-v2
failure. M2-v1 reaches 26.05 m and 50.04%, worse than both generic controls on
this training seed, although unlike M2-v2 it remains locomotory.

M2-v2 has the strongest contact force and classification metrics of the three
structured variants, while M1 remains the strongest aggregate state predictor
(0.411 one-step and 4.000 eight-step normalized MSE). Better contact semantics
therefore do not imply better total rollout prediction or policy optimization.

## Valid conclusion

The full-budget ablation contradicts a simple claim that persistent physical
loading memory improves control. In this run, removing memory preserves the
structured model's ability to train a locomotory actor and sharply improves
rollout error, at the cost of worse contact-force and classification accuracy.
Because there is one complete training seed, this is a strong within-seed
negative result and a hypothesis for replication, not a population-level
estimate.

The complete paired evidence is in `paired_all_methods.csv`; no reduced-budget
or successful-only surrogate is used for this conclusion.
