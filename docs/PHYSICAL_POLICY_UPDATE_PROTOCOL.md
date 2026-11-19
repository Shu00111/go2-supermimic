# Physical Policy-Update Protocol

Status: **PREPARED, NOT RUN**.

The experiment must use one common frozen Actor, paired allocation-bound snapshots, the repository's exact reward/observation/target conventions, a predeclared update-size schedule, and an unchanged-Actor control. For each snapshot, compute one candidate update through a frozen World Model, record predicted imagined-return change, restore the same physical state, and evaluate candidate and control policies with identical target sequences and RNG state. Report physical reward, progress, velocity tracking, termination, and per-condition failures.

Do not tune update sizes on Stage-4 OOD seeds. Reject a sample unless two restore/replays of its fixed action sequence are bitwise exact, including RFT loading memory, contact mode, target generator, observations, reward, and terminal bookkeeping. Run on one isolated environment first.

The prerequisite seed gate now passes for rigid seed 82001 and soft seed 82011 on isolated CPU. Their complete portable seed records and manifests are under `outputs/trained_real_diagnostics/actor_rollout_seeds_v1/`; allocation-bound snapshots were restored twice with bitwise equality for every recorded field. CUDA remains forbidden as physical-gradient truth.

The predeclared candidates are Full-M2, backward-detached Full-M2, K=2 Full-M2, No-Memory and an unchanged control, all starting from the exact same M1 Actor. Normalize each raw candidate direction to a common parameter L2 norm before applying update magnitudes `1e-5`, `3e-5`, and `1e-4`; do not select a magnitude using these two seeds. Use 32 physical steps, one environment, observation corruption disabled, and restore the in-process simulator snapshot before each candidate. The generator and RNG must come from the snapshot; record predicted change, real return change, termination, state/action finiteness, contact/material signature and the maximum replay discrepancy.

Prerequisites and preparation commands are:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/export_actor_rollout_seeds.py \
  --m1 handoff/contact_memory_diagnostics_seed14011/checkpoints/m1/model_12000.pt \
  --m2-v2 handoff/contact_memory_diagnostics_seed14011/checkpoints/m2_v2/model_12000.pt \
  --m2-v2-nomemory handoff/contact_memory_diagnostics_seed14011/checkpoints/m2_v2_nomemory/model_12000.pt \
  --horizon 8 --output outputs/trained_real_diagnostics/actor_rollout_seeds_v1
python scripts/run_exact_actor_bptt.py \
  --m1 handoff/contact_memory_diagnostics_seed14011/checkpoints/m1/model_12000.pt \
  --m2-v2 handoff/contact_memory_diagnostics_seed14011/checkpoints/m2_v2/model_12000.pt \
  --m2-v2-nomemory handoff/contact_memory_diagnostics_seed14011/checkpoints/m2_v2_nomemory/model_12000.pt \
  --seed-bundle outputs/trained_real_diagnostics/actor_rollout_seeds_v1/rigid_stable_stance_seed82001.pt \
  --seed-bundle outputs/trained_real_diagnostics/actor_rollout_seeds_v1/soft_stable_stance_seed82011.pt \
  --output outputs/trained_real_diagnostics/p1_actor_bptt
```

The actual policy mutation and physical comparison remain deliberately NOT RUN in this phase. No physical return or policy-improvement result is claimed, and the historical Stage-4 set remains untouched.
