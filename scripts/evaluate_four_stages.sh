#!/usr/bin/env bash
set -euo pipefail
checkpoint="${CHECKPOINT:?set CHECKPOINT to a frozen model.pt}"
output_root="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
[[ -f "$checkpoint" ]] || { echo "checkpoint not found" >&2; exit 2; }
[[ ! -e "$output_root" ]] || { echo "refusing to overwrite $output_root" >&2; exit 2; }
mkdir -p "$output_root"
for seed in 90101 90103 90107; do
  for task in Go2-Chase-Flat Go2-Chase-Terrain Go2-Chase-SoftTerrainLevels Go2-Chase-SoftTransition Go2-Chase-SoftToHard; do
    log="$output_root/${task}-seed${seed}.log"
    python scripts/eval.py "$task" --checkpoint "$checkpoint" --num-envs 128 --seconds 60 \
      --seed "$seed" --device cuda:0 >"$log" 2>&1
  done
done
