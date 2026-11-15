# Gradient Fidelity Results V2

Status: **P0 PARTIAL — replay Gate A passed on the deterministic CPU backend;
finite-difference and World Model comparisons are not yet complete.**

## Environment and source baseline

- Source baseline: `6febd863851bbd6756e77864bf0e4c50e0183309`.
- Development branch: `research/contact-structured-supermimic-v2`.
- Host GPU: NVIDIA RTX 3090, 24 GiB; driver 580.178.04.
- Python 3.10.22, PyTorch 2.14.1+cu130, MuJoCo 3.11.0,
  MuJoCo-Warp 3.11.0, Warp 1.19.0, mjlab 1.6.0.
- Baseline regression suite before P0 changes: 49 passed.

## Snapshot contents

`go2_supermimic/diagnostics/simulator_snapshot.py` captures the complete
in-process MJWarp `Data` allocation, rather than only `qpos/qvel`. This includes
integration inputs, actuator activation/history, controls, applied forces,
solver warm-start, derived kinematics, contact and constraint arrays, and sensor
data. It additionally captures:

- mjlab action history and processed actions;
- entity actuator targets and latency buffers;
- observation history and delay buffers;
- target-command and ball/obstacle generator state;
- reward, termination, event, metrics, and episode bookkeeping;
- RFT force diagnostics, material activation counters, plastic depth, plastic
  footprint anchor, and local surface-height memory;
- Python, NumPy, Torch CPU, and Torch CUDA RNG state.

Static model and terrain objects are retained by reference. Restore is therefore
intentionally restricted to the exact environment and MJWarp allocation that
created a snapshot. Shape, dtype, allocation identity, and environment count
are checked and mismatches fail closed.

## Replay validation

The validation uses one environment, seed 14011, a 12-control-step fixed action
sequence, observation corruption disabled, and manual terminal handling. Each
snapshot is restored twice and complete trajectories are compared. The recorded
state includes qpos/qvel, base and joint state, observations, rewards,
termination, RFT and rigid forces, material modes, and all RFT loading-memory
fields.

Eight required conditions were exercised:

1. rigid stable stance;
2. soft stable stance;
3. touchdown;
4. liftoff;
5. hard-to-soft transition;
6. soft-to-hard transition;
7. alternating-material contact;
8. simultaneous mixed-foot materials.

### CUDA result (negative finding)

Stable rigid/soft, touchdown, and liftoff replays were bitwise exact. Contact
transitions and mixed-material contact were not. The largest 12-step divergence
occurred on alternating material: joint velocity differed by 4.07 rad/s and RFT
force by 197.15 N after contact-mode divergence. Thresholds were not widened.
CUDA replay is therefore **invalid as physical finite-difference ground truth**
for this study.

Machine receipt:
`experiments/contact_structured_supermimic/reports/snapshot_replay_cuda_seed14011_v2.json`
(SHA-256 `3ef2be1cbe44b712edcc45fe3cf958ee126f5ffd0f1def897796b18cf47282f7`).

### CPU result (Gate A pass)

The identical MuJoCo-Warp/RFT tasks on the single-environment CPU backend were
bitwise exact in all eight conditions. Every recorded tensor had maximum
absolute replay error 0 for all 12 steps. Both transition scenarios changed
material mode during the rollout; the mixed-foot diagnostic simultaneously
reported rigid and soft feet. This is the frozen physical finite-difference
backend for P0.

Machine receipt:
`experiments/contact_structured_supermimic/reports/snapshot_replay_cpu_seed14011_v2.json`
(SHA-256 `fcede2e467f76859ab8270dd0d23824a7eb354eab538a3256355a5c9f074fa06`).

## Gate status and limitations

P0 Gate A passes only for isolated, single-environment CPU replay. GPU replay
remains useful for training but cannot define physical gradient truth around
mixed contact. This backend choice is a measured engineering constraint, not a
claim that CPU and CUDA trajectories are numerically identical to one another.

Finite-difference convergence at epsilon 0.01, 0.003, and 0.001, perturbation
signal-to-replay-noise checks, discontinuity censoring, and paired M1/M2 World
Model gradient comparisons remain **NOT RUN** at this checkpoint.
