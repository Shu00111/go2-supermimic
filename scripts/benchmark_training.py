#!/usr/bin/env python3
"""Launch one bounded scalability probe and record wall time/peak VRAM."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=("m1", "m2", "m3"), required=True)
ap.add_argument("--num-envs", type=int, choices=(128, 256, 512, 1024), required=True)
ap.add_argument("--iterations", type=int, default=2)
args = ap.parse_args()
if not 1 <= args.iterations <= 20:
  raise SystemExit("benchmark iterations must be in [1,20]")
env = os.environ.copy()
output_root = Path(env.get(
  "OUTPUT_ROOT", root / "experiments/contact_structured_supermimic/benchmarks"
)).resolve()
output_root.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_id = f"benchmark-{args.num_envs}-{stamp}"
env.update(
  NUM_ENVS=str(args.num_envs), TRAINING_ITERATIONS=str(args.iterations),
  RUN_ID=run_id, OUTPUT_ROOT=str(output_root),
)
run_dir = output_root / f"{args.method}-seed14011-{run_id}"
peak_vram_mib = 0
peak_utilization = 0
started = time.monotonic()
proc = subprocess.Popen([str(root / f"scripts/train_{args.method}.sh")], cwd=root, env=env)
while proc.poll() is None:
  sample = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True, capture_output=True, check=False,
  )
  if sample.returncode == 0:
    try:
      memory, utilization = (int(value.strip()) for value in sample.stdout.splitlines()[0].split(","))
      peak_vram_mib = max(peak_vram_mib, memory)
      peak_utilization = max(peak_utilization, utilization)
    except (IndexError, ValueError):
      pass
  time.sleep(0.25)
elapsed = time.monotonic() - started
summary = {
  "method": args.method,
  "num_envs": args.num_envs,
  "iterations": args.iterations,
  "elapsed_seconds": elapsed,
  "peak_vram_mib": peak_vram_mib,
  "peak_gpu_utilization_percent": peak_utilization,
  "exit_code": proc.returncode,
  "run_dir": str(run_dir),
  "completed_utc": datetime.now(timezone.utc).isoformat(),
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "benchmark_summary.json").write_text(
  json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
if proc.returncode:
  raise SystemExit(proc.returncode)
