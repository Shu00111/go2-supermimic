# Apple Silicon macOS deployment report

Validated on 2026-11-10 at Git revision `d9eee9b3f248e5300e636c433255db2c1c3e086a`.

## Result summary

The primary goal passed. A released pretrained Go2 terrain policy executed a real 5-second (250 control-step / 1000 physics-substep) MuJoCo Warp rollout on pyramid stairs using the ARM CPU backend. Observations and actions stayed finite, the simulation advanced, the robot did not fall, and final pose was recorded. Flat-ground rollout and offscreen MP4 generation also passed.

| Capability | Status | Evidence / limitation |
|---|---|---|
| 1. Repository installation | **PASS** | Editable install in `.venv`; revision recorded |
| 2. Dependency import | **PASS** | PyTorch, MuJoCo, mjlab, Warp, and MuJoCo Warp imported |
| 3. Pretrained checkpoint loading | **PASS** | All three actors strict-loaded with no missing/unexpected keys |
| 4. CPU policy inference | **PASS** | All actors produced finite 12-D actions |
| 5. Headless physics simulation | **PASS** | MuJoCo Warp CPU simulation advanced at 0.005 s physics step |
| 6. Flat-ground rollout | **PASS** | 5 s, 250 control steps, finite state/action, zero falls |
| 7. Complex-terrain rollout | **PASS** | `terrain.pt`, stairs, 5 s, finite state/action, zero falls |
| 8. Video generation | **PASS** | Real flat rollout: 2 s, H.264, 640x360, 50 fps |
| 9. World-model training feasibility | **PARTIAL** | One tiny CPU iteration passed; released scale is impractical |
| 10. Full SuperMimic training feasibility | **PARTIAL** | Technically executes at tiny scale; Linux + NVIDIA is required for practical full training |

## Validated rollouts

### Flat ground

Command:

```bash
.venv/bin/python scripts/eval.py Go2-Chase-Flat --checkpoint weights/flat.pt \
  --mode chase --num-envs 1 --seconds 5 --device cpu
```

Observed:

- 5.0 simulated seconds; control step 0.02 s; physics step 0.005 s.
- Mean speed: 2.41 m/s.
- Falls: 0.
- Observations/actions finite: `True/True`.
- Final root position: `[-1.9372916, 4.202675, 0.34701222]`.
- Final quaternion (wxyz as exposed by mjlab): `[-0.25572982, 0.004624749, 0.018581145, 0.96655864]`.
- Wall clock: 31.97 s on the first compile; peak footprint reported by `/usr/bin/time`: 881,675,968 bytes.
- Log: `outputs/macos_validation/flat_cpu_5s.log`.

### Complex terrain: pyramid stairs

Command:

```bash
.venv/bin/python scripts/eval.py Go2-Chase-Terrain --checkpoint weights/terrain.pt \
  --terrain stairs --mode chase --num-envs 1 --seconds 5 --device cpu
```

Observed:

- 5.0 simulated seconds; 250 control steps and 1000 physics substeps.
- Terrain policy input shape: 735, including the 5-frame 11x7 height scan.
- Mean speed: 1.87 m/s.
- Chase tags: 1 (reported as 12/min over the 5-second interval).
- Time within 0.3 m: 0.29.
- Falls: 0.
- Observations/actions finite: `True/True`.
- Final root position: `[14.159019, -21.757498, 0.84321177]`.
- Final quaternion: `[-0.124823324, 0.031684387, -0.18993816, 0.97331333]`.
- Wall clock: 320.91 s, including CPU terrain generation/compilation; peak footprint: 1,632,015,616 bytes.
- Log: `outputs/macos_validation/terrain_stairs_cpu_5s.log`.

### Video

The validated video command is encapsulated by `scripts/run_macos_eval.sh`. It produced `outputs/macos_validation/flat_cpu_2s.mp4`: H.264 High profile, 640x360, 50 fps, exactly 2.00 s, SHA-256 `6dbcec7c1a3895afa4509342653f43d3378e75620b963d319d60f8d02aa250db`.

## Automated tests

```text
25 passed in 10.96s
```

The tests cover target generation/collision, obstacles, terrain/height scanning, actor expansion, and warm starts. They do not replace the real rollout evidence above.

## Training feasibility

The local CPU smoke command used one environment, a 16-transition buffer, two-step world-model windows and imagined horizon, 32-wide networks, batch size 4, and exactly one iteration. It completed collection, world-model loss/backward, imagined rollout, actor backward, critic backward, and checkpoint save in 5.21 s with about 768 MB peak footprint after kernels were cached.

Results: world-model loss `0.883`, imagined return `0.1613`, finite actor gradient norm `0.03481`, and critic loss `0.004191`. Evidence is in `outputs/macos_validation/training_cpu_smoke.log` and the generated checkpoint under `outputs/macos_validation/training_smoke/`.

Capability distinctions:

- Pretrained policy inference: **PASS** on CPU.
- Forward physics simulation: **PASS** with MuJoCo Warp CPU.
- World-model training: **PASS only as a bounded smoke test**; ordinary batches would be slow and memory-heavy.
- Differentiable rollout: **PASS through the learned PyTorch world model**. MuJoCo Warp physics itself is not differentiated by this implementation and its major kernels disable backward.
- End-to-end policy optimization: **PASS only as a bounded smoke test**; released-scale optimization is not practical on this host.

For meaningful full training at the documented scale and duration, use a Linux machine with a supported NVIDIA GPU and sufficient VRAM (the README suggests reduced terrain settings even for a 16 GB GPU). No CUDA or Linux-only GPU dependencies were installed on this Mac.

## Reproduction

Activate the environment:

```bash
cd /Users/shuyiwei/Projects/go2-supermimic
source .venv/bin/activate
```

Run the validated video reproduction:

```bash
./scripts/run_macos_eval.sh
```

The first execution on a fresh Warp cache compiles kernels and is slower. Complex terrain additionally spends several minutes generating the full 5x7 terrain grid even with one simulated environment.
