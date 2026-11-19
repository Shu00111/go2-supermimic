# Complete Actor-rollout seeds

## Status: PASS

Two isolated single-environment CPU seeds were exported under `outputs/trained_real_diagnostics/actor_rollout_seeds_v1/`:

| scenario | task | seed | frames | observation | SHA-256 |
|---|---|---:|---:|---:|---|
| rigid stable stance | Go2-Chase-Flat-Scan | 82001 | 8 | 735D | `6e3c60b1422d7ea2c14c62d3afd92c4591dc088bec2199adac54b0c9ed40b757` |
| soft stable stance | Go2-Chase-SoftTerrain-Scan | 82011 | 8 | 735D | `492bcdfb608097ffd38d1b1258c29a1db1889678b60c2b63114c036dfae761ec` |

Each file contains the exact `Imagination.rollout` seed (`env`, physical state, planar pose, previous action, 735D history, packed generator, context and encoder bias), robot constants, observation schema/history, target config, height field and relief/height scans, full/no-memory initial contact estimates, material/RFT state, fixed M1-Actor action sequence, reference physical frames, rewards/dones and Python/NumPy/Torch RNG states.

The allocation-bound simulator snapshot was captured and restored twice in the same process. Every recorded tensor was bitwise equal on replay. The soft seed additionally checked plastic depth/position/surface, signed distance, soft mask and RFT force; all maximum absolute errors were zero. Rewards and observations were finite. The snapshot itself is deliberately not serialized because the validated adapter rejects restoration into another environment or Warp allocation.

The versioned manifest is `manifest.json`; files are 61,134 and 64,973 bytes. Generate them with `scripts/export_actor_rollout_seeds.py`. The script forces CPU isolation when invoked with an empty `CUDA_VISIBLE_DEVICES` and refuses a nonempty output directory.
