#!/usr/bin/env python3
"""Run the frozen weighted curriculum as restartable task segments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--method", choices=("m1", "m2", "m3"), required=True)
  ap.add_argument("--through-stage", type=int, choices=(1, 2, 3), default=3)
  args = ap.parse_args()
  output = Path(os.environ["OUTPUT_ROOT"]).resolve()
  seed = int(os.environ["SEED"])
  resume = os.environ.get("RESUME", "")
  cumulative = int(os.environ.get("START_ITERATION", "0"))
  for stage in range(1, args.through_stage + 1):
    key = f"STAGE{stage}_ITERATIONS"
    if key not in os.environ:
      raise SystemExit(f"{key} is required (freeze it from the matched pilot)")
    cfg = json.loads((ROOT / f"configs/stages/stage{stage}.json").read_text())
    if cfg.get("known_gap") and os.environ.get("ALLOW_UNVALIDATED_STAGE3") != "1":
      raise SystemExit(f"Stage {stage} BLOCKED: {cfg['known_gap']}")
    budget = int(os.environ[key])
    allocations = [round(budget * x) for x in cfg["retention"]]
    allocations[-1] += budget - sum(allocations)
    for index, (task, amount) in enumerate(zip(cfg["tasks"], allocations), 1):
      if amount <= 0:
        raise SystemExit(f"non-positive segment allocation for {task}")
      cumulative += amount
      run_id = f"stage{stage}-segment{index}-{task}"
      segment_root = output / f"seed{seed}" / args.method / f"stage{stage}"
      env = os.environ.copy()
      env.update(OUTPUT_ROOT=str(segment_root), RUN_ID=run_id, TASK=task,
                 TRAINING_ITERATIONS=str(cumulative), SEED=str(seed))
      if resume:
        env["RESUME"] = resume
      subprocess.run(["python", "scripts/run_config.py", "--method", args.method],
                     cwd=ROOT, env=env, check=True)
      matches = list(segment_root.glob(f"{args.method}-seed{seed}-{run_id}/**/model_{cumulative}.pt"))
      if len(matches) != 1:
        raise SystemExit(f"expected one checkpoint model_{cumulative}.pt, found {matches}")
      resume = str(matches[0])
  print(resume)


if __name__ == "__main__":
  main()
