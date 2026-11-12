#!/usr/bin/env python3
"""Launch one explicitly bounded scalability probe at a time."""
import argparse
import os
import subprocess
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
env.update(NUM_ENVS=str(args.num_envs), TRAINING_ITERATIONS=str(args.iterations),
           RUN_ID=f"benchmark-{args.num_envs}")
subprocess.run([str(root / f"scripts/train_{args.method}.sh")], cwd=root, env=env, check=True)
