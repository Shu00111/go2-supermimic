# RTX 5090 handoff

Use a fresh virtual environment; do not alter system CUDA or drivers. Run
`scripts/check_environment.sh`, which requires CUDA, prints driver/device/free
memory, executes a finite CUDA matmul, checks free disk, and rejects an old
PyTorch build on Blackwell (`sm_120`). Install a PyTorch wheel compiled for the
machine's CUDA, then install this project and `mjlab` in the environment. Any
MuJoCo Warp/custom extension must be rebuilt inside that environment.

Example:

```bash
git clone --branch research/contact-structured-supermimic https://github.com/Shu00111/go2-supermimic.git
cd go2-supermimic
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
scripts/preflight.sh
OUTPUT_ROOT=/data/supermimic NUM_ENVS=128 TRAINING_ITERATIONS=2 python scripts/benchmark_training.py --method m2 --num-envs 128 --iterations 2
```

Formal launch requires the pilot-frozen `TRAINING_ITERATIONS`, `OUTPUT_ROOT`,
`SEED`, and stage `TASK`. Use `train_m1.sh`, `train_m2.sh`, or `train_m3.sh`.
Each run uses a new directory, records source/config hashes and refuses
overwrite. `RESUME` must point to an existing complete checkpoint and preserves
the global curriculum counter. Checkpoints contain actor, critic, world model,
three optimizers, running normalizers, replay buffer, rare-event archive, RNG,
training schedule and configuration. Environment simulator state is not
serialized, so learning resumes with intact data/optimizer state at a fresh
episode boundary; exact same-physics-substep continuation is not claimed.

For the weighted curriculum, use `run_curriculum.py` with `OUTPUT_ROOT`, `SEED`,
and pilot-frozen `STAGE1_ITERATIONS`, `STAGE2_ITERATIONS`, and
`STAGE3_ITERATIONS`. It preserves previous-stage coverage as deterministic task
segments and resumes the complete learning state. Stage 3 refuses to start
while the rough-soft/alternating-patch physics gap remains; the override exists
for simulator development only and is not authorization for formal runs.

Detached example: `tmux new-session -d -s sm-m2 'env OUTPUT_ROOT=/data/runs SEED=14011 TASK=Go2-Chase-SoftTerrain TRAINING_ITERATIONS=<FROZEN> scripts/train_m2.sh'`.
Start the non-LLM watchdog separately with
`scripts/monitor_training.sh /data/runs/<run-directory>`; it samples every 40
minutes. Recommended minimum free disk is 100 GiB. GPU memory must be established
by the four-size pilot; no unmeasured requirement is claimed.
