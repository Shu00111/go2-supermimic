#!/usr/bin/env python3
"""Restart-safe matched launcher for P1 M1/M2-v1/M2-v2 development."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from train_p1_world_model_v2 import METHODS


def complete(path: Path, expected: dict) -> bool:
  try:
    contract = json.loads((path / "contract.json").read_text())
    return (path / "TRAINING_COMPLETE").exists() and all(
      contract.get(key) == value for key, value in expected.items()
    ) and Path(contract["checkpoint"]).is_file()
  except (OSError, KeyError, json.JSONDecodeError):
    return False


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--seed", type=int, default=14011)
  parser.add_argument("--updates", type=int, default=1200)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--window", type=int, default=8)
  parser.add_argument("--hidden", type=int, nargs="+", default=(256, 256, 256))
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--eval-every", type=int, default=100)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  repo = Path(__file__).resolve().parents[1]
  child_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for method in METHODS:
    output = args.output_dir / method
    expected = {
      "method": method, "seed": args.seed, "updates": args.updates,
      "batch_size": args.batch_size, "window": args.window,
      "hidden": list(args.hidden), "lr": args.lr,
    }
    if complete(output, expected):
      print(f"skip validated {method}", flush=True)
      continue
    if output.exists() and any(output.iterdir()):
      raise RuntimeError(f"incomplete controlled run requires audit: {output}")
    command = [
      args.python, str(repo / "scripts/train_p1_world_model_v2.py"),
      "--method", method, "--dataset-dir", str(args.dataset_dir),
      "--output-dir", str(output), "--device", args.device,
      "--seed", str(args.seed), "--updates", str(args.updates),
      "--batch-size", str(args.batch_size), "--window", str(args.window),
      "--hidden", *(str(value) for value in args.hidden), "--lr", str(args.lr),
      "--eval-every", str(args.eval_every),
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, env=child_env, check=True)
  (args.output_dir / "DEVELOPMENT_COMPLETE").write_text("P1 controlled development complete\n")


if __name__ == "__main__":
  main()
