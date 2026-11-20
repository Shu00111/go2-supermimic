# P2 Control-Relevant Fidelity Summary

## Bottom line

The evidence does **not** support a general Full M2-v2 action-Jacobian failure or loading-memory gradient explosion. It supports a coupled failure: contact/body semantic co-adaptation, a policy-induced shift into an unusually static state/action distribution, imperfect imagined-to-physical update transfer, and a collapse guard that reacted to early failures by exponentially suppressing later learning.

| Hypothesis | Evidence | Contradiction / limit | Confidence | Next action |
|---|---|---|---|---|
| H1 action-Jacobian mismatch | all models have poor H=1 cosines and large scale error | Full M2-v2 is best on rigid and intermediate on soft | medium against “M2 uniquely worst” | matched H=2/4/8 relief-sequence export |
| H2 model exploitation | all predicted deltas positive; only 50% physical sign agreement; Full optimizer counterfactual is -0.0977 on soft | only two validation states; M1/No-Memory also fail by scenario | medium | 16–32 paired independent states |
| H3 distribution shift | Full replay z-RMS 1.842, coverage 0.857, static fraction 0.693 | retained final ring, not full history | high for final replay, medium historically | export checkpoint-indexed occupancy summaries |
| H4 optimization dynamics | 20 guards reduce LR scale to 9.54e-7; final Actor LR is 8.88e-11 and Adam step is 1200 versus 4900 for controls | observational; guard may prevent worse collapse | high chronology, low causality | replay one bounded guard/optimizer counterfactual offline |
| H5 contact-conditioned bias | soft native contact semantic RMSE 1.49 std; oracle changes action Jacobian 16%; zero interventions sometimes lower body error | oracle barely improves state RMSE; interventions are network-local | medium-high | constrain a future model with calibrated physical contact bottleneck |

## Explicit answers

1. **Are Full M2-v2 action derivatives less accurate?** Not generally at H=1. It outperforms M1 on both states and No-Memory on rigid, but trails No-Memory on soft.
2. **Do predicted improvements agree with physical improvements?** Only 50% of 24 candidate/scenario comparisons agree in sign. The Full optimizer counterfactual is the worst soft result, but the pilot is too small for a population claim.
3. **Is semantic drift present?** Yes, especially as a distributed contact-code/body mismatch; memory-only sensitivity is small.
4. **Is distribution shift measurable?** Yes in the retained final replay, with a 69.3% static-state fraction for Full versus 0.4% M1 and 1.0% No-Memory.
5. **Did optimizer collapse precede repeated guards?** The first physical fall collapse triggered the first guard; repeated guards then caused the extreme LR suppression. Suppression followed, rather than preceded, the initial collapse.
6. **Best current explanation?** A feedback loop: contact-conditioned co-adaptation permits imperfect control gradients; policy updates move Full M2-v2 into a rare static regime; recurrent physical failures trigger rollback/LR suppression, leaving too little learning rate to escape. This is the best synthesis, not a proven single cause.
7. **Enough evidence for M2-v3?** Enough to specify discriminating safeguards, not enough to claim a solved architecture: calibrated contact bottlenecks, occupancy/velocity guards that do not only freeze LR, paired physical-gradient validation, and matched multistep relief exports should be mandatory.

## Audit totals and limitations

- Independent physical states: 2 generation + 2 disjoint validation states.
- H=1 finite-difference validity: 24/24 coordinates (100%); censored: 0.
- Physical update comparisons: 24 nonzero pairs; 12 positive, 12 negative; all finite, no falls.
- Failed diagnostic attempts: one incorrect saturation gate and one dtype mismatch were quarantined under `/tmp`; neither contributed results. The corrected runs passed.
- Not run: H=2/4/8 physical/model sensitivity because no validated common divergent-trajectory relief-sequence contract was exported; larger paired physical study; raw final-evaluation feature distances.
- Distribution analysis is partial because only final replay rings and aggregate final evaluations survive.

## Reproduction and tests

All numerical commands used `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` and the `go2-supermimic-rft` Python environment. The exact command lines, hashes, paths, seeds, strides, horizons, and output locations are preserved in:

```text
OUTPUT_ROOT=/tmp/p2-reproduction-v1 scripts/run_p2_control_fidelity_v1.sh
```

The output root must not already exist. Set `FORMAL_ROOT` if the preserved formal-v2 log lives elsewhere. Inputs are the three hash-locked `handoff/contact_memory_diagnostics_seed14011/checkpoints/*/model_12000.pt` files, the two complete CPU seeds under `outputs/trained_real_diagnostics/actor_rollout_seeds_v1/`, and the hash-locked rigid/soft trajectories. Source before this P2 change was `c17bb8d2aac4e71cfd41887c9a73420c340f19f2`; the final commit SHA is recorded in Git history rather than retroactively rewriting frozen result receipts.
