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

## Freeze and held-out evaluation

The frozen checkpoint is read-only at
`outputs/soft_terrain/seed_14011/frozen/model_12000.pt`; its SHA-256 is
`49ee17ecd2009499416492d98b1d49e51a8d4ba5b373b3f4be546f45dd8f6067`.

Run or resume the complete 2x8 matrix (existing valid summaries are skipped):

```bash
PYTHONNOUSERSITE=1 python scripts/run_soft_terrain_formal_eval.py \
  --episodes 128 --device cuda:0 \
  --output-dir outputs/soft_terrain/seed_14011/evaluation/held_out_v1
PYTHONNOUSERSITE=1 python scripts/summarize_soft_terrain_eval.py \
  --input-dir outputs/soft_terrain/seed_14011/evaluation/held_out_v1
PYTHONNOUSERSITE=1 python scripts/plot_soft_terrain_training.py \
  --training-dir outputs/soft_terrain/seed_14011/training \
  --output outputs/soft_terrain/seed_14011/evaluation/held_out_v1/training_curves.png
```

The evaluation matrix is chase-only so pursuit velocity error has a single
definition. `protocol.json` records condition tasks, fixed seeds, and
coefficient scales. `ARTIFACT_SHA256SUMS.txt` covers all result artifacts.
