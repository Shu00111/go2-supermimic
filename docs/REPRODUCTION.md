# Reproduction

## Environment

```bash
conda create -n go2-supermimic-rft python=3.10 -y
conda activate go2-supermimic-rft
PYTHONNOUSERSITE=1 pip install -e .
PYTHONNOUSERSITE=1 python -m pytest -q
```

The audited environment is at
`/home/yiwei/miniconda3/envs/go2-supermimic-rft`; see
`docs/GPU_ENVIRONMENT_AUDIT.md` for exact resolved core versions.

## Formal training

Launch the single formal seed without holding an interactive session:

```bash
mkdir -p outputs/soft_terrain/seed_14011/runtime
nohup scripts/run_soft_terrain_seed14011.sh \
  >>outputs/soft_terrain/seed_14011/runtime/training.log 2>&1 &
echo $! >outputs/soft_terrain/seed_14011/runtime/nohup.pid
```

Launch the independent 40-minute local watchdog:

```bash
nohup python scripts/watch_soft_terrain_training.py \
  --run-root outputs/soft_terrain/seed_14011 --interval-minutes 40 \
  >>outputs/soft_terrain/seed_14011/runtime/watchdog.log 2>&1 &
echo $! >outputs/soft_terrain/seed_14011/runtime/watchdog.pid
```

Both launchers are restart-safe. The training launcher uses a non-blocking file
lock to prevent duplicate seed-14011 jobs and resumes from the greatest
`model_N.pt` it finds. Outputs and checkpoints are intentionally gitignored.

## Quick inspection

```bash
python scripts/watch_soft_terrain_training.py \
  --run-root outputs/soft_terrain/seed_14011 --once
tail -n 50 outputs/soft_terrain/seed_14011/runtime/training.log
```

The formal evaluation command and checkpoint SHA-256 manifest will be recorded
after the frozen iteration-12,000 checkpoint exists.
