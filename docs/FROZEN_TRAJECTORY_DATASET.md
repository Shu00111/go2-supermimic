# Frozen Trajectory Dataset

Status: **existing five-condition P1 dataset recovered and verified; requested eight-condition complete export is BLOCKED/NOT RUN**.

The preserved model-independent P1 collection has 32 environments × 512 transitions for each of `rigid_rough`, `soft_rough`, `hard_to_soft`, `soft_to_hard`, and `alternating`: 160 trajectories and 81,920 transitions total. Actions came from one frozen Actor and are therefore common across every World Model comparison. Environment IDs are partitioned without trajectory leakage: modulo 10 values 0–5 train, 6–7 development, and 8–9 held out.

Every transition includes state, next state, pose delta, action, previous action, domain context, relief scan, contact labels, next-contact labels, terminal flag, episode ID, soil strength, material-boundary distance, and per-foot soft mask. Oracle contact data remain diagnostic labels and are never Actor inputs. `frozen_trajectory.py` validates hashes, schema, shapes, and split selection.

This collection predates the current handoff contract and does **not** include Actor observation history, the full target-generator state, reward breakdown, a portable simulator snapshot, or explicit touchdown/liftoff strata. It therefore cannot be called the requested complete reproducible physical trajectory bundle. No new collection was launched because an unrelated formal 4,096-environment GPU training job was active and exact CPU-backend collection was not established by the existing script. The historical Stage-4 OOD set was not used or modified.

Preserved dataset hashes:

- `rigid_rough.pt`: `474807cbef7a82c8ce53066478c3b844d11561be16fc17a8bf8e8595e5e792ee`
- `soft_rough.pt`: `eefefdc166781ed87b002c6ab0b7fdc64ab97681d4f0e9d5adf58f87a422afa3`
- `hard_to_soft.pt`: `82fa8f7cc182d4ca8b9426bce3e52a78f325d7c618abdcb2b2712f50c0712549`
- `soft_to_hard.pt`: `31dd10e70b4ce83532437f6163f3b5eebaf0e2fe634f095fac27510e574bc5ba`
- `alternating.pt`: `d5f0d32a172605e2816ce740069dd4ce0e5f4c93416152b9e41954907bc8b651`

Historical replay evidence is preserved separately: the isolated single-environment replay receipt passes bitwise equality for all eight requested scenarios; the CUDA receipt fails exactness for mixed/transition scenarios. These receipts are evidence about the snapshot adapter, not substitutes for exported snapshots.
