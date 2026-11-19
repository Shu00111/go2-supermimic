# Contact-memory mathematical conclusions

## Consolidated evidence

| Metric | M1 | Full M2-v2 | No-Memory | Interpretation |
|---|---:|---:|---:|---|
| rigid physical gain, H=8 | 265.850 | 43.758 | 303.563 | Full M2 is smaller on this point |
| soft physical gain, H=8 | 197.994 | 48.416 | 247.448 | Full M2 is smaller on this point |
| soft memory-to-body gain, H=8 | N/A | 5.897 | 0 | coupling exists and grows, but does not dominate total gain |
| fixed-action detached/full gradient cosine, soft H=8 | N/A | 0.9998 | N/A | backward memory cut causes only a small local change |
| loss-pair negative-cosine frequency | N/A | 30.36% | NOT MEASURED | local multi-objective conflict exists |
| complete physical rollout seeds | shared | PASS (2) | PASS (2) | exact Actor BPTT prerequisite satisfied |
| Actor-gradient L2, rigid H=8 | 2.129 | 5.630 | 2.544 | Full M2 is largest on this seed |
| Actor-gradient L2, soft H=8 | 7.617 | 11.512 | 4.913 | Full M2 is largest on this seed |
| independent physical policy-update test | NOT RUN | NOT RUN | NOT RUN | no physical gradient-superiority claim |

## Answers to the research questions

1. **Does M2-v2 amplify perturbations more than No-Memory?** Not in the measured common physical map. Full M2 had substantially lower gains at all H=`1,2,4,8` on both sampled conditions. This contradicts simple H1, subject to the very small sample and power-iteration approximation.
2. **Is contact-memory-to-body coupling responsible?** A real coupling is present, especially on soft terrain (`0.889` to `5.897` from H=1 to 8), but the total gain result does not make it responsible for abnormal amplification.
3. **Does backward-only detachment reduce abnormal gradients?** No. In fixed-action diagnostics it changed norms by little and kept cosine >=`0.9998`; in exact Actor BPTT at H=8 it slightly *increased* Full-M2 norms (`5.630 -> 5.685` rigid; `11.512 -> 11.638` soft).
4. **Are contact losses in conflict with state prediction?** Some objective pairs conflict locally: 30.36% of off-diagonal ordered pairs were negative. This supports H2 as a local landscape observation, not a historical causal claim.
5. **Can the measured mechanism explain the historical Actor collapse?** No. Full M2 did show larger H=8 Actor gradients on both stance seeds, but neither the common physical Jacobian nor removal of the memory backward edge explains that increase. Distribution mismatch, contact-to-body semantic drift, reward/Actor optimization and collapse-guard interactions remain viable H4/H5 explanations.
6. **Is the evidence sufficient for M2-v3?** No. A new method should not be justified from two local trajectory points and two stance seeds. The most defensible next experiment is a preregistered, matched tiny Actor update followed by restored-state CPU physics evaluation.
7. **What remains unverified?** Variation across training seeds and trajectories; longer horizons; historical (not final-checkpoint) loss gradients; behavior near transitions/aerial states; optimizer/collapse-guard dynamics; and whether any learned-gradient change predicts real return change.

## Exact graph boundary

M1's 78D physical state already includes four simulator RFT plastic-depth coordinates. Full M2 adds a recurrent 32D predicted contact state, including four explicit loading-memory coordinates. The contact estimator's 84D GRU recurrent tensor initializes contact once and is not propagated in imagination. Closed-loop Actor BPTT additionally carries previous action, planar pose, 735D observation history, target-generator state, context, encoder bias and terrain/robot constants.

All findings are exploratory: one frozen training seed, one Jacobian point per condition, two Actor stance seeds, and numerical differentiation variation rather than independent experimental replication.
