# Frozen Checkpoint Recovery

Status: **three of three authoritative seed-14011 checkpoints recovered and byte-verified**.

## Source provenance

- Frozen research/reporting commit: `f7e61b6954e7a706e9aeb1f5d15ccbec0896ca17`.
- Completed-run manifests record training source `23acf7c1ae990b14b5dd5030e6e852e05d129edb`.
- The M2-v2 model contract was frozen at `e8b6d9555cf0df3621c0a7994ab8fb9adcbaf005`.
- `git diff 23acf7c..f7e61b6 -- go2_supermimic configs scripts/train.py` is empty: later commits are reporting/diagnostic changes, not changes to the trained implementation.
- Ubuntu integration starts from Mac diagnostic commit `395ffde82c80a9b7b1e6ddda3f2ac8cd4af1da81` on `research/contact-memory-gradient-diagnostics`.

Original implementations are in `go2_supermimic/supermimic/train.py` (Actor, Critic, objective, checkpoint format), `world_model.py` (M1 and running statistics), `contact_model.py` (M2-v2 and No-Memory), and `imagine.py` (`Imagination.rollout`). RFT loading memory is in `go2_supermimic/soft_terrain/rft.py`; complete allocation-bound replay is in `go2_supermimic/diagnostics/simulator_snapshot.py`.

## Verified frozen bytes

| Method | Size (bytes) | SHA-256 | Result |
|---|---:|---|---|
| M1 | 3,164,619,415 | `b3a76b694c7ddad1e5897ceeaa01040b3b2587a11143f36fd7a324813f658179` | PASS |
| M2-v2 | 3,165,322,231 | `3d322d8f7ffe68f50f5b6f5573dd48af8dc8dd15220b4446802691469f117c19` | PASS |
| M2-v2-no-memory | 3,165,322,295 | `023b0eed06eb5d8bfb31f3e76c504ea6cd0ea3ae861e3839e4cfc3c49326cbee` | PASS |

Each is iteration 12,000, seed 14011, task `Go2-Chase-AlternatingMaterial-Scan`, and records 12,308,480 environment transitions. Strict loading verified Actor, World Model, Critic, all running-normalization tensors, all three optimizer states, replay buffer, window archive, curriculum/training state, and RNG states. The full tensor schema is generated in `metadata/checkpoint_provenance.json` in the handoff bundle.

`scripts/verify_frozen_artifacts.py` checks SHA-256 before the historical `torch.save` payload is deserialized and rejects any method, seed, iteration, key, shape, or dtype mismatch.
