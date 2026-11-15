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

## Frozen common objective

`go2_supermimic/diagnostics/gradient_objective.py` defines the scalar that will
be used on both the physical trace and the World Model trace. For horizon
`H=32`, it is

\[
J_R=\sum_{t=0}^{H-1}\gamma^t c_t
  \left(r_{\mathrm{analytic},t}+w_{\mathrm{term}}\,\Delta t\,d(s_{t+1})\right),
\quad c_0=1,\quad c_{t+1}=c_t(1-d(s_{t+1})).
\]

Here `gamma=0.99`, `d` is the existing smooth fall probability, and
`r_analytic` is the existing differentiable SuperMimic reward implementation.
Actions use normalized 12-dimensional policy/action-manager coordinates. The
target, obstacle, and command-generator states are a frozen per-step sequence
shared by every `+epsilon`, `-epsilon`, M1, and M2 rollout. Only the initial
action is differentiated; later actions are fixed to the recorded nominal
sequence. This makes the comparison an open-loop initial-action derivative.

The learned endpoint value from the training objective is explicitly excluded
from `J_R`, because it has no equivalent physical-ground-truth evaluation. It
will be reported separately rather than mixed into the gradient comparison.
The shared evaluator and open-loop BPTT rollout are covered by the regression
suite; the full suite now has 53 passing tests.

The isolated CPU finite-difference runner is implemented in
`scripts/measure_physical_gradient.py`. A two-step soft-contact smoke test had
zero replay-objective error, no saturation, terminal change, contact-mode
change, or non-finite result. For action coordinate 0, the centered estimates
at epsilon 0.01, 0.003, and 0.001 were respectively -6.97e-4, -6.30e-4, and
-6.42e-4. This is an execution-path check only, not a Gate B result; the
required 32-step, all-coordinate, phase-stratified study remains outstanding.

The first 32-step/all-coordinate soft-stable measurement completed with zero
replay-objective error, no saturation, non-finite value, terminal change, or
coarse foot/material-mode change. It did **not** pass the epsilon-convergence
check: 0/12 coordinates were valid. Gradient norms at epsilon 0.01, 0.003, and
0.001 were 1.45, 4.36, and 15.43, and the corresponding gradient-vector cosine
similarities were 0.171 (0.01 vs 0.003), 0.631 (0.003 vs 0.001), and 0.126
(0.01 vs 0.001). Five coordinates changed sign across epsilon. These values
must not be used as physical gradient truth.

This is evidence of a nonsmooth or numerically ill-conditioned 32-step contact
trajectory, not replay noise. The runner now additionally records MuJoCo
constraint counts and complete active contact-set identity, and applies an
explicit epsilon-agreement censor. A rigid two-step control check converged on
the tested coordinate (relative span 0.0055), showing that the measurement path
can recover a stable local derivative away from the problematic long-horizon
soft-contact rollout. Shorter-horizon localization is in progress before any
M1/M2 comparison is attempted.

## Gate status and limitations

P0 Gate A passes only for isolated, single-environment CPU replay. GPU replay
remains useful for training but cannot define physical gradient truth around
mixed contact. This backend choice is a measured engineering constraint, not a
claim that CPU and CUDA trajectories are numerically identical to one another.

Finite-difference convergence at epsilon 0.01, 0.003, and 0.001, perturbation
signal-to-replay-noise checks, discontinuity censoring, and paired M1/M2 World
Model gradient comparisons remain **NOT RUN** at this checkpoint. The common
objective and BPTT execution path are frozen, but this is not a Gate B pass.
