#!/usr/bin/env python3
"""Run matched M1/M2 Stage-1 pilots sequentially on one GPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--output-root", type=Path, required=True)
  ap.add_argument("--iterations", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=14011)
  args = ap.parse_args()
  if args.iterations < 500:
    raise SystemExit("matched pilot requires at least 500 Stage-1 iterations")
  env = os.environ.copy()
  env.update(
    OUTPUT_ROOT=str(args.output_root.resolve()), SEED=str(args.seed),
    STAGE1_ITERATIONS=str(args.iterations), NUM_ENVS="1024",
  )
  for method in ("m1", "m2"):
    subprocess.run(
      [sys.executable, "scripts/run_curriculum.py", "--method", method,
       "--through-stage", "1"],
      cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
  main()
