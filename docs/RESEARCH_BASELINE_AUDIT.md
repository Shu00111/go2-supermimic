# Research baseline audit

## Provenance

- Writable fork: `https://github.com/Shu00111/go2-supermimic.git`.
- Upstream: `https://github.com/Zhengtianjin/go2-supermimic.git`.
- Research base: fork branch `feat/soft-terrain-rft`, commit
  `f49ca8365dd930b591e79443c7af54b616e86ca8`.
- Original public SuperMimic import: `d9eee9b3f248e5300e636c433255db2c1c3e086a`.
- GPU RFT implementation commit: `e9615dec86a8fcec66160450193c86f8a893bda0`.
- SCMA persistent reference inspected at
  `cf2e3cf52f293cf2af048b6406022c01bf999060`, especially
  `src/learning/sg_v2_persistent.py`, `contact_adaptation_v2.py`, and
  `contact_adaptation_sg.py`. Commit `d7ba7417` was not present in the fetched
  local history and remains **unverified**; no result table was used as source.

The checkout named `Supermimic` was an empty Git repository and was not used.
No SCMA PPO actor, residual actor, optimizer, or policy weights were imported.

## Verified SuperMimic mechanism

`WorldModel` is a deterministic MLP. Its inputs are state `s` (78), current and
previous 12-D actions, four domain context values, and a 77-point local relief
scan. It predicts normalized state delta (78) plus planar pose delta (3).
Training uses 8-step autoregressive windows. Policy improvement differentiates
a nominal 32-step analytic return through the frozen world model and actor;
the critic uses imagined TD(lambda) targets. Actor and critic are MLPs with
hidden widths 512/256/128. These values are code-confirmed defaults.

The soft-terrain baseline expanded state from the original implementation to
include four oracle RFT plastic depths and four signed material-boundary
distances. Consequently, persistent soil memory is already present in the M1
simulator/world-model state. It is not in the actor observation. This overlap
narrows the M2 contribution to explicit estimated per-foot contact structure
and its differentiable rollout, not the first use of loading history.

Historical B0/B1 checkpoint naming and a B1 initialization receipt were not
found in this branch and are **unverified**. The tracked released policy hashes
are:

| File | SHA-256 |
|---|---|
| `weights/flat.pt` | `574e53a90d8dcb29a398c9b940ca0f2fe9ee5ac90fa6a3b6721ffcc794085220` |
| `weights/obstacles.pt` | `ec4a57a995807512374eacf53cb6b1267cdb0a22750a16a6bd4f4cce9f8ecb91` |
| `weights/terrain.pt` | `fd55d8172b23ff1c9219746a93b0cf4d1ba1ac4f85973219f83b548765c0c995` |

## Shared controller and simulation

- Output: 12 normalized joint-position commands; 50 Hz policy/control rate.
- Soft RFT physics: 2 ms timestep, decimation 10.
- Hip/thigh PD: `kp=17.6918`, `kd=1.12630`; knee PD: `kp=39.8066`,
  `kd=2.53416` (derived directly from tracked actuator configuration).
- Action scale: hip/thigh `0.334901`, knee `0.285317` rad per normalized unit.
- Effort limits: 23.7 Nm hip/thigh, 45.43 Nm knee; actuator delay 0–2 physics
  steps. Soft joint limits use factor 0.9.
- Registered tasks cover flat, rigid rough, uniform soft, randomized soft
  strength, and each direction of a single hard/soft boundary.
- MuJoCo Warp GPU support is inherited from `mjlab>=1.6.0`; exact transitive
  package builds are captured by preflight on the training machine.

Rough deformable terrain and alternating material patches are not registered
in the verified base. The new implementation reviewed the local
`Go2-LunarSoftTerrain` worktree at commit `56cfe10f` for per-face local-height
RFT semantics and the strip material query at `323fe14`; no result table or
policy was imported. The checkerboard implementation is new in this branch.
Both registrations remain a simulator-validation blocker for formal Stage 3
until their RTX 5090 checks pass; neither is silently approximated.
