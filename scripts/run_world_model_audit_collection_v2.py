#!/usr/bin/env python3
"""Restart-safe launcher for the frozen P1 transition audit collection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from collect_world_model_audit_v2 import CONDITIONS


def valid_receipt(path: Path, condition: str, num_envs: int, steps: int) -> bool:
  try:
    value = json.loads(path.read_text())
    return (value["condition"] == condition and value["num_envs"] == num_envs
            and value["steps"] == steps and len(value["dataset_sha256"]) == 64)
  except (OSError, KeyError, json.JSONDecodeError):
    return False


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=512)
  parser.add_argument("--seed", type=int, default=71001)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--python", default=sys.executable)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  child_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  repo = Path(__file__).resolve().parents[1]

  for index, condition in enumerate(CONDITIONS):
    output = args.output_dir / f"{condition}.pt"
    receipt = output.with_suffix(output.suffix + ".json")
    if output.exists() and valid_receipt(receipt, condition, args.num_envs, args.steps):
      print(f"skip validated {output}", flush=True)
      continue
    if output.exists() or receipt.exists():
      raise RuntimeError(f"incomplete/conflicting frozen artifact: {output}")
    command = [
      args.python, str(repo / "scripts/collect_world_model_audit_v2.py"),
      "--condition", condition,
      "--policy-checkpoint", str(args.policy_checkpoint),
      "--output", str(output),
      "--seed", str(args.seed + 10 * index),
      "--num-envs", str(args.num_envs),
      "--steps", str(args.steps),
      "--device", args.device,
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, env=child_env, check=True)

  (args.output_dir / "COLLECTION_COMPLETE").write_text("P1 audit collection complete\n")


if __name__ == "__main__":
  main()
