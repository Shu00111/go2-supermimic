# Development validation report

Host audit (2026-11-12): RTX 3090 (compute capability 8.6, 24 GiB), driver
580.178.04; dual Xeon Silver 4210R (40 logical CPUs), 125 GiB RAM and 1.1 TiB
free disk. An unrelated formal training process and a separate Git operation
were active in another repository. They were left untouched. Development used
the isolated worktree `go2-supermimic-contact-validation`.

Validated on CPU:

- source compilation for modified modules;
- M1 and M2 finite forward/backward;
- actor-action gradient through world model;
- gradient through predicted contact state and shared per-foot transition;
- estimator shape, foot identity, reset and oracle-channel exclusion;
- three-step loss alignment and four-step differentiable rollout;
- RFT numerical fixtures, memory isolation, release and strength scaling;
- flat/local-height parity, rough face heights and local penetration;
- seeded checkerboard lookup, deterministic rigid boundary semantics,
  different materials under different feet and exact zero RFT on rigid feet;
- task registration and Sim2Real configuration parsing;
- clean/Sim2Real actuator-delay configuration isolation;
- smooth finite-difference diagnostic agreement;
- shared experimental-contract configuration.

The final command, `CUDA_VISIBLE_DEVICES='' python -m pytest -q`, produced **43
passed, 1 skipped**, one deprecation warning, 0 failed. The skipped test is the
tensor-only CUDA parity test, which had passed in an earlier CUDA-visible run.
Installed development versions were torch 2.14.1 (CUDA 13.0 build),
MuJoCo 3.11.0, mujoco-warp 3.11.0, mjlab 1.6.0, NumPy 2.2.6 and pytest 9.0.2.

MuJoCo Warp task construction/stepping, mixed-material contact forces,
Blackwell extension build, M2 learning smoke, checkpoint resume under the new
tasks, and 128/256/512/1024 throughput are **NOT RUN — delegated to RTX 5090**.
Stage 3 stays blocked and no scientific result is claimed.
