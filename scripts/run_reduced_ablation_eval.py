#!/usr/bin/env python3
"""Evaluate A0/A1/A2/A3 on one paired reduced-study mechanism condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"A0": "m1", "A1": "m2_nomemory", "A2": "m2", "A3": "m3"}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--study-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--episodes", type=int, default=128)
  parser.add_argument("--seed", type=int, default=90201)
  args = parser.parse_args()
  root = args.study_root.resolve()
  output = root / "evaluation"
  output.mkdir(exist_ok=True)
  checkpoints = {}
  for label, method in METHODS.items():
    matches = list((root / method).glob("**/model_2000.pt"))
    if len(matches) != 1:
      raise RuntimeError(f"expected one {label} final checkpoint, found {matches}")
    checkpoints[label] = matches[0]
  protocol = {
    "label": "reduced-budget mechanism evaluation; not formal Stage 4",
    "condition": "command_ood",
    "evaluation_seed": args.seed,
    "paired_first_episodes": args.episodes,
    "checkpoints": {k: str(v) for k, v in checkpoints.items()},
  }
  protocol_path = output / "protocol.json"
  if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
    raise RuntimeError("refusing to change existing ablation evaluation protocol")
  protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
  env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for label, checkpoint in checkpoints.items():
    summary = output / f"{label}_command_ood_summary.json"
    if summary.exists():
      print(f"skip existing {summary}", flush=True)
      continue
    command = [
      sys.executable, str(ROOT / "scripts/eval_soft_terrain_formal.py"),
      "--method", label, "--condition", "command_ood",
      "--policy-checkpoint", str(checkpoint),
      "--world-model-checkpoint", str(checkpoint),
      "--episodes", str(args.episodes), "--seed", str(args.seed),
      "--output-dir", str(output), "--device", args.device,
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
  (output / "COMPLETE").write_text("Reduced ablation evaluation complete\n")


if __name__ == "__main__":
  main()
