#!/usr/bin/env python3
"""Run the frozen single-seed M1/M2 curriculum sequentially on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--output-root", type=Path, required=True)
  args = ap.parse_args()
  contract = json.loads((ROOT / "configs/formal_single_seed.json").read_text())
  if contract["seeds"] != [14011] or contract["methods"] != ["m1", "m2"]:
    raise SystemExit("unexpected formal single-seed contract")
  env = os.environ.copy()
  env.update(
    OUTPUT_ROOT=str(args.output_root.resolve()),
    SEED="14011",
    NUM_ENVS=str(contract["num_envs"]),
    **{f"STAGE{stage[-1]}_ITERATIONS": str(value)
       for stage, value in contract["stage_iterations"].items()},
  )
  for method in contract["methods"]:
    subprocess.run(
      [sys.executable, "scripts/run_curriculum.py", "--method", method,
       "--through-stage", "3"],
      cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
  main()
