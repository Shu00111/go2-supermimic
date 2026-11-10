# Apple Silicon macOS deployment audit

Audit date: 2026-11-10 (Asia/Shanghai)

Repository revision: `d9eee9b3f248e5300e636c433255db2c1c3e086a`

## Host

| Item | Observed value |
|---|---|
| OS | macOS 15.6 (24G84), Darwin 24.6.0 |
| Architecture | arm64 |
| CPU/GPU | Apple M2, 10-core integrated GPU, Metal 3 |
| Unified memory | 16 GiB (`17179869184` bytes) |
| Free disk before installation | 61 GiB on `/System/Volumes/Data` |
| Shell | zsh |
| Conda Python | 3.12.2 arm64 |
| Framework Python | 3.13.0 |
| Git | 2.53.0 |
| Homebrew | 6.0.9 |
| uv | Not installed |

## Repository and official entry points

- License: MIT; the vendored Go2 assets retain the MuJoCo Menagerie license.
- No `uv.lock` or requirements file exists. `pyproject.toml` requires Python >=3.10 and `mjlab>=1.6.0`.
- Official evaluator: `scripts/eval.py TASK --checkpoint PATH`.
- Official training entry point: `scripts/train.py`.
- Released checkpoints: `weights/flat.pt`, `weights/obstacles.pt`, and `weights/terrain.pt`.
- Terrain types: `flat`, `stairs`, `stairs_inv`, `boxes`, `rough`, `slope`, and `slope_inv`.
- Registered tasks: `Go2-Chase-Flat`, `Go2-Chase-Obstacles`, and `Go2-Chase-Terrain`.

## Isolated environment

The environment is `/Users/shuyiwei/Projects/go2-supermimic/.venv`, created with Conda's Python 3.12.2 using `python -m venv`. No global packages or unrelated Conda environments were changed.

Installed core versions:

| Package | Version |
|---|---:|
| Python | 3.12.2 |
| go2-supermimic | 0.1.0 (editable) |
| PyTorch | 2.15.0 |
| MuJoCo | 3.11.0 |
| mjlab | 1.6.0 |
| Warp | 1.19.0 |
| MuJoCo Warp | 3.11.0 |
| mediapy | 1.2.7 |
| pytest | 9.2.0 |

PyTorch reports CUDA unavailable and MPS available. Warp reports only the native `cpu` ARM device. MPS was not used because a PyTorch MPS device does not provide a MuJoCo Warp backend.

## Compatibility findings and route

The unmodified evaluator fails at its hard-coded `cuda:0` allocation with `AssertionError: Torch not compiled with CUDA enabled`; see `outputs/macos_validation/official_eval_failure.log`.

The selected route is the project's original mjlab/MuJoCo Warp simulation on Warp's ARM CPU backend, with PyTorch policy inference on CPU. This keeps the original model, dynamics engine, physics configuration, observations, normalization state, action space, rewards, and weights.

Minimal compatibility changes:

1. `scripts/eval.py` accepts `--device` (default remains `cuda:0`) and uses it for mjlab, checkpoint mapping, and the actor. It also prints finite-state and final-pose evidence.
2. `go2_supermimic/env/env_cfg.py` tolerates the MuJoCo Warp 3.11 public-export difference for `OverflowType`. Version 3.11 has no iteration-warning flags, so that mask is skipped while all physics settings remain unchanged.
3. Training accepts a device field (default remains `cuda:0`) so a strictly bounded CPU smoke test is possible.

## Dependency validation

- All core packages import successfully on arm64.
- Warp initializes with `CUDA not enabled in this build` and device `cpu` (`arm`).
- Repository tests: **25 passed**.
- MuJoCo Warp CPU forward simulation and offscreen rendering work.
- Initial compilation is expensive. The first flat rollout took 31.97 s including kernel compilation; the first terrain rollout took 320.91 s, dominated by terrain generation and height-field kernel compilation.

## Checkpoint audit

Each released actor is an MLP with hidden sizes `[512, 256, 128]`, a checkpointed `RunningNorm`, and 12 joint-position outputs.

| Checkpoint | Task input dimension | SHA-256 | Strict load | Zero-observation inference |
|---|---:|---|---|---|
| `flat.pt` | 350 | `574e53a90d8dcb29a398c9b940ca0f2fe9ee5ac90fa6a3b6721ffcc794085220` | PASS | finite, shape `(1, 12)` |
| `obstacles.pt` | 430 | `ec4a57a995807512374eacf53cb6b1267cdb0a22750a16a6bd4f4cce9f8ecb91` | PASS | finite, shape `(1, 12)` |
| `terrain.pt` | 735 | `fd55d8172b23ff1c9219746a93b0cf4d1ba1ac4f85973219f83b548765c0c995` | PASS | finite, shape `(1, 12)` |

There were no missing or unexpected parameters. Normalization statistics are embedded in each actor state dictionary. The released files are actor-only deployment checkpoints; they do not include the critic/world model required to resume training.

## Known blockers and limitations

- The official upstream command is CUDA-only until the local device patch is applied.
- CPU terrain startup is very slow (about 5.3 minutes for the validated first run).
- MuJoCo Warp 3.11 warns that height-field/capsule and height-field/box pairs do not support MULTICCD; at most one contact is generated for those pairs. This is a backend warning, not a local dynamics change.
- Full released-scale training is not practical on a 16 GiB M2: defaults use 4096 environments, large replay/archive tensors, 1024-wide world-model layers, and batches up to 8192. The CPU smoke test does not establish usable convergence speed.
- MuJoCo Warp 3.11 marks major forward/collision modules `enable_backward=False`. This repository does not require physics gradients: its differentiable rollouts are through a learned PyTorch world model.
