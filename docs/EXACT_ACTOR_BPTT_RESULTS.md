# Exact Actor BPTT results

## Protocol

This experiment uses the repository's unmodified `Imagination.rollout`, analytic reward, a shared frozen M1 Actor and shared M1 critic. All methods start from the two validated physical seeds. Full M2 and No-Memory initialize contact with their deployment `initial_contact` contract. H=`1,2,4,8`; CPU float32; identical random state for observation noise.

Full M2 interventions cut only the loading-memory backward edge using always-detached and K=`1,2,4,8` truncation. Their forward imagined returns must be bitwise identical to Full M2 or the run fails. Reported per-step contributions are Actor-parameter gradient norms of each raw analytic step reward. Parameter ordering is exactly the shared Actor's module order.

## Results

At H=8, Full M2 produced the largest Actor-gradient norm on both seeds, although its common physical-state Jacobian had been the smallest:

| seed | metric | M1 | Full M2-v2 | No-Memory |
|---|---|---:|---:|---:|
| rigid | imagined return | 0.5731 | 0.4565 | 0.5052 |
| rigid | Actor gradient L2 | 2.129 | 5.630 | 2.544 |
| rigid | first-action gradient L2 | 0.0156 | 0.0428 | 0.0235 |
| soft | imagined return | 4.7846 | 4.1239 | 4.9615 |
| soft | Actor gradient L2 | 7.617 | 11.512 | 4.913 |
| soft | first-action gradient L2 | 0.1390 | 0.0829 | 0.0498 |

At H=8 the Full-M2/No-Memory Actor-gradient cosine was `0.6758` rigid and `0.6497` soft. The Full-M2/M1 cosine was `0.6859` and `0.5880`. Always-detaching memory did **not** reduce the norms: it changed rigid `5.630 -> 5.685` and soft `11.512 -> 11.638`, while retaining cosine `0.99985` and `0.99997` with the full gradient. All M2 intervention returns were bitwise equal, all gradients finite, and the final run took 32.89 s.

The numerical table and plot are generated at `outputs/trained_real_diagnostics/p1_actor_bptt/`. See `exact_actor_bptt.csv`, `exact_actor_gradient_cosine.csv`, `exact_actor_bptt_summary.json` and `exact_actor_bptt.png` for all 64 method/intervention rows and model/cross-horizon cosines.

This is exact learned-model BPTT, not physical-gradient validation. With two stance seeds it is exploratory and cannot alone explain a historical training collapse or establish physical gradient superiority. The next phase would apply matched tiny hypothetical Actor updates and evaluate them from identical restored CPU states; that physical-policy update is prepared but NOT RUN, so the historical Stage-4 final set remains untouched.
