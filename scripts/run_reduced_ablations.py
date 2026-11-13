#!/usr/bin/env python3
"""Run the frozen A0/A1/A2/A3 reduced-budget mechanism study sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def iteration(path: Path) -> int:
  try:
    return int(path.stem.rsplit("_", 1)[1])
  except (IndexError, ValueError):
    return -1


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-root", type=Path, required=True)
  args = parser.parse_args()
  contract = json.loads((ROOT / "configs/ablation_reduced_v1.json").read_text())
  output = args.output_root.resolve()
  output.mkdir(parents=True, exist_ok=True)
  saved_contract = output / "contract.json"
  if saved_contract.exists() and json.loads(saved_contract.read_text()) != contract:
    raise RuntimeError("refusing to change an existing ablation contract")
  saved_contract.write_text(json.dumps(contract, indent=2) + "\n")

  seed = int(contract["seed"])
  for method in contract["methods"]:
    resume = ""
    cumulative = 0
    method_root = output / method
    method_root.mkdir(exist_ok=True)
    for segment_index, segment in enumerate(contract["segments"], 1):
      cumulative += int(segment["iterations"])
      run_id = f"segment{segment_index}-{segment['task']}"
      run_root = method_root / f"{method}-seed{seed}-{run_id}"
      complete = list(run_root.glob(f"**/model_{cumulative}.pt"))
      if len(complete) == 1:
        resume = str(complete[0])
        print(f"skip completed {resume}", flush=True)
        continue
      if len(complete) > 1:
        raise RuntimeError(f"ambiguous completed checkpoints: {complete}")
      candidates = sorted(method_root.glob("**/model_*.pt"), key=iteration)
      if candidates:
        resume = str(candidates[-1])
      if run_root.exists():
        retry = 1
        while (method_root / f"{method}-seed{seed}-{run_id}-retry{retry}").exists():
          retry += 1
        run_id = f"{run_id}-retry{retry}"
      env = os.environ.copy()
      env.update({
        "OUTPUT_ROOT": str(method_root), "RUN_ID": run_id, "SEED": str(seed),
        "NUM_ENVS": str(contract["num_envs"]), "TASK": segment["task"],
        "TRAINING_ITERATIONS": str(cumulative),
      })
      if resume:
        env["RESUME"] = resume
      subprocess.run(
        [sys.executable, "scripts/run_config.py", "--method", method],
        cwd=ROOT, env=env, check=True,
      )
      matches = list((method_root / f"{method}-seed{seed}-{run_id}").glob(
        f"**/model_{cumulative}.pt"
      ))
      if len(matches) != 1:
        raise RuntimeError(f"expected model_{cumulative}.pt, found {matches}")
      resume = str(matches[0])
  (output / "TRAINING_COMPLETE").write_text("Reduced A0/A1/A2/A3 training complete\n")


if __name__ == "__main__":
  main()
