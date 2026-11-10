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

Launch the single formal seed as a supervised user service without holding an
interactive session:

```bash
mkdir -p outputs/soft_terrain/seed_14011/runtime
systemd-run --user --unit=go2-supermimic-seed14011 --collect \
  --property=WorkingDirectory="$PWD" \
  --property=StandardOutput="append:$PWD/outputs/soft_terrain/seed_14011/runtime/training.log" \
  --property=StandardError="append:$PWD/outputs/soft_terrain/seed_14011/runtime/training.log" \
  "$PWD/scripts/run_soft_terrain_seed14011.sh"
```

Launch the independent 40-minute local watchdog:

```bash
systemd-run --user --unit=go2-supermimic-watch14011 --collect \
  --property=WorkingDirectory="$PWD" \
  --property=StandardOutput="append:$PWD/outputs/soft_terrain/seed_14011/runtime/watchdog.log" \
  --property=StandardError="append:$PWD/outputs/soft_terrain/seed_14011/runtime/watchdog.log" \
  /home/yiwei/miniconda3/envs/go2-supermimic-rft/bin/python \
  "$PWD/scripts/watch_soft_terrain_training.py" \
  --run-root "$PWD/outputs/soft_terrain/seed_14011" --interval-minutes 40
```

The training launcher is restart-safe: rerunning the service after interruption
uses a non-blocking file lock to prevent duplicates and resumes from the
greatest `model_N.pt` it finds. Outputs and checkpoints are intentionally
gitignored.

## Quick inspection

```bash
python scripts/watch_soft_terrain_training.py \
  --run-root outputs/soft_terrain/seed_14011 --once
tail -n 50 outputs/soft_terrain/seed_14011/runtime/training.log
```

The formal evaluation command and checkpoint SHA-256 manifest will be recorded
after the frozen iteration-12,000 checkpoint exists.
