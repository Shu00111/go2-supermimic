# Loss-gradient conflict diagnostics

## Audited objective

The M2-v2 frozen objective is the sum of autoregressive normalized robot-state
plus pose prediction, predicted-contact supervision, and a weight-0.25 physical
anchor. Predicted-contact supervision is smooth-L1 over normalized force,
intrusion, memory and tangential velocity plus binary cross-entropy for contact
probability. The anchor contains both teacher-forced contact prediction and a
body prediction conditioned on oracle next-contact labels.

The diagnostic decomposition reports robot state, force, classification,
intrusion, loading memory and the complete weighted physical anchor. Pairwise
cosines are calculated only over the intersection of parameter tensors reached
by both losses; the shared-tensor count is recorded in every row. This avoids
padding method-specific parameters with misleading zeros.

## Status

`TRAINED-POLICY RESULT: NOT MEASURED.` The file
`loss_gradient_conflict.csv` and `gradient_conflict_heatmap.png` are
`UNTRAINED_MODEL_SMOKE_TEST` outputs at H = 1, 2, 4 and 8. They prove that loss
decomposition, shared-parameter filtering and cosine calculations execute in
float64. They cannot establish whether auxiliary losses conflicted during the
collapsed training run. Negative cosines in this fixture are not evidence for
H2.

Hessian-vector products remain P3 and `NOT MEASURED`; no full Hessian is
materialized.
