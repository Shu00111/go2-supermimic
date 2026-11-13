# GPU training report

## Contact-structured formal seed 14011 (2026-11-13)

The final M1/M2 runs used one RTX 3090 and 1,024 environments, never concurrently. Each consumed 12.288M environment transitions. TensorBoard wall-time spans were 1.58 GPU-hours (M1) and 1.81 GPU-hours (M2), 3.39 total excluding evaluation. Observed training memory was approximately 8--10 GiB; the Stage-4 evaluator used approximately 4--7 GiB. The local 40-minute watchdog recorded process, checkpoint-age, utilization, and disk receipts. The formal service and watchdog were stopped after training, and all later GPU jobs ran sequentially.

Status: implementation, training, and held-out evaluation complete. Formal seed 14011 was launched
at 2026-11-10 12:28:56 Asia/Shanghai from source commit
`e9615dec86a8fcec66160450193c86f8a893bda0` under the user systemd service
`go2-supermimic-seed14011.service`. This document must not be read as a
multi-seed convergence claim.

## Scalability and end-to-end gates

Twenty timed policy steps after warm-up produced the following soft-terrain
physics throughput on the RTX 3090:

| Environments | Physics steps/s | Finite |
|---:|---:|:---:|
| 128 | 12,223 | yes |
| 256 | 23,824 | yes |
| 512 | 46,468 | yes |
| 1,024 | 90,103 | yes |

A full-shape 512-environment preflight used the default 1024x3 world model,
default Actor/Critic shapes, `wm_batch=2048`, and `imag_batch=4096`. Two
uncompiled iterations completed with finite world-model loss, imagined return,
Actor gradients, and Critic loss in 17.9 s total including initialization;
maximum host RSS was 2.57 GiB. The formal run uses compilation and the normal
five Critic epochs, so this timing is a capacity gate rather than a forecast.

A separate small end-to-end transition run updated all three learners, saved
models 1-3, restored model 3 with the preserved iteration counter, and saved
model 4. This verifies replay, world-model fitting, imagined BPTT, Actor/Critic
updates, checkpoint serialization, and continuation with the 78-D state.

## Formal single-seed curriculum

The exact machine-readable configuration is
`experiments/soft_terrain/seed_14011.json`. Total budget is 12,000 iterations:

| Stage | Task | Iterations |
|---|---|---:|
| A | uniform soft | 1-4,000 |
| B | randomized soil strength | 4,001-8,000 |
| C1 | hard-to-soft | 8,001-10,000 |
| C2 | soft-to-hard | 10,001-12,000 |

The launcher discovers the highest valid periodic checkpoint and passes both
`--resume` and `--continue-run true`, preserving the global counter and all
serialized optimizer/model state. No other formal seed is configured.

## Monitoring

`scripts/watch_soft_terrain_training.py` is a local standard-library watchdog.
Every 40 minutes it appends PID liveness, GPU utilization/memory/temperature,
last checkpoint iteration, and checkpoint age to `runtime/watchdog.jsonl`.
It performs no Codex or model calls. The Codex task also has a separate quiet
40-minute heartbeat for user-visible intervention only when state changes or
action is required.

The first post-launch verification observed iteration 50/4,000 in Stage A,
finite loss, both systemd services active, training PID 684464, and 2,424 MiB
owned by the training Python process (3,757 MiB total GPU memory in use). These
are liveness measurements, not outcome metrics.

## Completed cost

- Start: 2026-11-10 12:28:56 Asia/Shanghai.
- Finish: 2026-11-10 14:04:50 Asia/Shanghai.
- Wall/GPU time: 1 h 35 min 54 s (one RTX 3090, no concurrent heavy training).
- Global iterations: 12,000.
- Approximate environment transitions: 6.14 million, plus short replay prefill
  at each stage; each transition contains ten 2 ms physics substeps.
- World-model optimizer updates: 24,000.
- Actor/Critic policy iterations: approximately 11,700 after the initial 300
  world-model warm-up iterations.
- Final checkpoint: `model_12000.pt`, 35,221,167 bytes.
- Frozen SHA-256:
  `49ee17ecd2009499416492d98b1d49e51a8d4ba5b373b3f4be546f45dd8f6067`.

The four stage-ending world-model losses were 0.2559, 0.1499, 0.4624, and
0.2845. Stage-boundary discontinuities are distribution changes visible in
`training_curves.png`; they are not evaluation scores.
