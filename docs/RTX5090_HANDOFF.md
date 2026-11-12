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

## Prepared RTX 5090 commands (not executed on the development host)

Set the storage roots first. Replace every angle-bracket placeholder only after
the matched pilot freezes it.

```bash
export OUTPUT_ROOT=/data/supermimic/runs
export DATA_ROOT=/data/supermimic/data
export SEED=14011
export NUM_ENVS=128
```

1. Clone and environment setup:

```bash
git clone --branch research/contact-structured-supermimic https://github.com/Shu00111/go2-supermimic.git
cd go2-supermimic
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

2. CUDA/Blackwell and CPU/GPU preflight:

```bash
OUTPUT_ROOT="$OUTPUT_ROOT" scripts/check_environment.sh
OUTPUT_ROOT="$OUTPUT_ROOT" scripts/preflight.sh
python scripts/validate_physics_tasks.py --num-envs 16 --steps 100 --seed "$SEED" \
  --device cuda:0 --output "$OUTPUT_ROOT/physics-smoke-seed${SEED}.json"
```

3. Bounded 128/256/512/1024 benchmark (2 iterations, not training evidence):

```bash
for NUM_ENVS in 128 256 512 1024; do
  OUTPUT_ROOT="$OUTPUT_ROOT/benchmarks/m2" SEED="$SEED" NUM_ENVS="$NUM_ENVS" \
    python scripts/benchmark_training.py --method m2 --num-envs "$NUM_ENVS" --iterations 2
done
```

4. Matched M1/M2/M3 pilots:

```bash
for METHOD in m1 m2 m3; do
  OUTPUT_ROOT="$OUTPUT_ROOT/pilots/$METHOD" SEED="$SEED" NUM_ENVS="$NUM_ENVS" \
    TASK=Go2-Chase-AlternatingMaterial TRAINING_ITERATIONS=2 \
    "scripts/train_${METHOD}.sh"
done
```

5. Frozen Stage 1--3 single-seed runs (run each only after CUDA validation and
after substituting pilot-frozen budgets):

```bash
for METHOD in m1 m2 m3; do
  OUTPUT_ROOT="$OUTPUT_ROOT/formal" SEED="$SEED" NUM_ENVS="$NUM_ENVS" \
    STAGE1_ITERATIONS=<FROZEN_STAGE1_ITERATIONS> \
    STAGE2_ITERATIONS=<FROZEN_STAGE2_ITERATIONS> \
    STAGE3_ITERATIONS=<FROZEN_STAGE3_ITERATIONS> \
    python scripts/run_curriculum.py --method "$METHOD" --through-stage 3
done
```

6. Three-seed formal launch matrix:

```bash
for SEED in 14011 24017 34019; do
  for METHOD in m1 m2 m3; do
    OUTPUT_ROOT="$OUTPUT_ROOT/formal" SEED="$SEED" NUM_ENVS="$NUM_ENVS" \
      STAGE1_ITERATIONS=<FROZEN_STAGE1_ITERATIONS> \
      STAGE2_ITERATIONS=<FROZEN_STAGE2_ITERATIONS> \
      STAGE3_ITERATIONS=<FROZEN_STAGE3_ITERATIONS> \
      python scripts/run_curriculum.py --method "$METHOD" --through-stage 3
  done
done
```

7. World-model and four-stage evaluation:

```bash
CHECKPOINT="$OUTPUT_ROOT/formal/<RUN>/model_<ITERATION>.pt"
DATA_ROOT="$DATA_ROOT" python scripts/evaluate_world_model.py \
  --predictions "$DATA_ROOT/<FROZEN_PREDICTIONS>.npz" \
  --output "$OUTPUT_ROOT/evaluation/world-model.json"
CHECKPOINT="$CHECKPOINT" OUTPUT_ROOT="$OUTPUT_ROOT/evaluation/four-stage" \
  scripts/evaluate_four_stages.sh
```

8. Gradient-fidelity capture and scoring. Capture intentionally remains
fail-closed until the 5090 run implements and validates full simulator, soil
memory and RNG snapshot restore; do not bypass this receipt:

```bash
python scripts/evaluate_gradient_capture.py \
  --confirm-snapshot-restore --output "$DATA_ROOT/gradient/capture-receipt.json"
python scripts/evaluate_gradient_fidelity.py \
  --input "$DATA_ROOT/gradient/<PAIRED_GRADIENTS>.npz" \
  --output "$OUTPUT_ROOT/evaluation/gradient-fidelity.json"
```

9. Monitor and safe resume:

```bash
MONITOR_INTERVAL_SECONDS=2400 scripts/monitor_training.sh "$OUTPUT_ROOT/formal/<RUN>"
RESUME="$OUTPUT_ROOT/formal/<RUN>/model_<ITERATION>.pt" \
  OUTPUT_ROOT="$OUTPUT_ROOT/resumed" SEED="$SEED" NUM_ENVS="$NUM_ENVS" \
  TASK=Go2-Chase-AlternatingMaterial \
  TRAINING_ITERATIONS=<FROZEN_TOTAL_ITERATIONS> scripts/train_m2.sh
```

The development host executed source/CPU tests only apart from the tiny
tensor-level CUDA parity unit included in `pytest`; none of the commands in
this section were executed. They establish development readiness, not a
scientific result.
