#!/bin/zsh
set -euo pipefail

repo_dir=${0:A:h:h}
cd "$repo_dir"
mkdir -p outputs/macos_validation

if [[ ! -x .venv/bin/python ]]; then
  print -u2 "Missing .venv; create the isolated environment before running this script."
  exit 1
fi

.venv/bin/python scripts/eval.py Go2-Chase-Flat \
  --checkpoint weights/flat.pt \
  --mode chase \
  --num-envs 1 \
  --seconds 2 \
  --device cpu \
  --video outputs/macos_validation/flat_cpu_2s.mp4 \
  --width 640 \
  --height 360 \
  2>&1 | tee outputs/macos_validation/run_macos_eval.log
