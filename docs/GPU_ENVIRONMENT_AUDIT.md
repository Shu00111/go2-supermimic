# GPU environment audit

Audit date: 2026-11-10 (Asia/Shanghai)

## Host

- OS: Ubuntu 22.04.5 LTS, kernel `6.8.0-138-generic`, x86_64.
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB.
- NVIDIA driver: `580.178.04`.
- Driver-advertised CUDA compatibility: 13.0.
- Standalone CUDA toolkit (`nvcc`): not installed or not on `PATH`. This is not
  required by the wheel-based PyTorch/MuJoCo Warp installation and no system
  CUDA or driver changes were made.
- System memory: 125 GiB total, approximately 113 GiB available at audit time.
- Workspace filesystem: 2.0 TiB total, approximately 1.1 TiB free.

At audit time another user-owned Isaac Lab evaluation process was using about
4.9 GiB of GPU memory. No SuperMimic training was started concurrently with
that job.

## Repository and isolated environment

- Writable fork: `https://github.com/Shu00111/go2-supermimic`.
- Upstream: `https://github.com/Zhengtianjin/go2-supermimic`.
- Fork base commit: `cfd172230e0d26365d3ac89e98b96d635a839df2`.
- Upstream commit inspected: `d9eee9b3f248e5300e636c433255db2c1c3e086a`.
- Feature branch: `feat/soft-terrain-rft`.
- Conda environment: `/home/yiwei/miniconda3/envs/go2-supermimic-rft`.
- Python: 3.10.

The project is installed only in the isolated environment above, with user-site
packages disabled during all recorded commands (`PYTHONNOUSERSITE=1`).

## Runtime versions and gates

- PyTorch: 2.14.1, CUDA runtime build 13.0; `torch.cuda.is_available()` is true.
- MuJoCo: 3.11.0.
- MuJoCo Warp: 3.11.0.
- Warp: 1.19.0.
- mjlab: 1.6.0.
- Upstream test suite before the soft-terrain changes: 25 passed.
- Complete suite after integration: 29 passed.

The released flat-ground Actor was evaluated on CUDA in 32 environments for
five seconds before simulator changes: all observations/actions stayed finite,
no falls were observed, and mean planar speed was 1.91 m/s. This is a runtime
compatibility gate, not a formal policy-quality result.
