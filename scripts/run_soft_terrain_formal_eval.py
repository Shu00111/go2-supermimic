#!/usr/bin/env python3
"""Run the frozen B0/B1 held-out matrix in isolated subprocesses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from eval_soft_terrain_formal import CONDITIONS


SEEDS = {
  "hard": 91001,
  "soft_nominal_1.0": 91002,
  "soft_seen_0.3": 91003,
  "soft_seen_0.6": 91004,
  "hard_to_soft": 91005,
  "soft_to_hard": 91006,
  "soft_unseen_0.2": 91007,
  "soft_unseen_1.2": 91008,
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
  parser.add_argument("--episodes", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--python", default=sys.executable)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  frozen = args.repo / "outputs/soft_terrain/seed_14011/frozen/model_12000.pt"
  methods = {"B0": args.repo / "weights/flat.pt", "B1": frozen}
  frozen_hash = sha256(frozen)
  policy_hashes = {name: sha256(path) for name, path in methods.items()}
  env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for method, policy in methods.items():
    for condition in CONDITIONS:
      summary = args.output_dir / f"{method}_{condition}_summary.json"
      if summary.exists():
        try:
          existing = json.loads(summary.read_text())
          valid = (
            existing["episodes"] == args.episodes
            and existing["evaluation_seed"] == SEEDS[condition]
            and existing["policy_sha256"] == policy_hashes[method]
            and existing["world_model_sha256"] == frozen_hash
          )
        except (KeyError, json.JSONDecodeError, OSError):
          valid = False
        if valid:
          print(f"skip existing validated {summary}", flush=True)
          continue
        print(f"replace stale or mismatched {summary}", flush=True)
      command = [
        args.python, str(args.repo / "scripts/eval_soft_terrain_formal.py"),
        "--method", method, "--condition", condition,
        "--policy-checkpoint", str(policy),
        "--world-model-checkpoint", str(frozen),
        "--episodes", str(args.episodes), "--seed", str(SEEDS[condition]),
        "--output-dir", str(args.output_dir), "--device", args.device,
      ]
      print("RUN " + " ".join(command), flush=True)
      subprocess.run(command, cwd=args.repo, env=env, check=True)

  summaries = []
  for method in methods:
    for condition in CONDITIONS:
      path = args.output_dir / f"{method}_{condition}_summary.json"
      summaries.append(json.loads(path.read_text()))
  fields = [
    "method", "condition", "evaluation_seed", "episodes", "soil_coefficient_scale",
    "success_rate", "fall_rate", "forward_progress_m", "path_length_m",
    "pursuit_velocity_mae_mps", "slip_speed_mps", "sinkage_m", "soil_force_n",
    "soil_dissipated_power_w", "actuator_abs_power_w",
    "actuator_work_per_path_m_jpm", "contact_transitions_per_min",
    "wm_one_step_normalized_mse", "wm_eight_step_normalized_state_mse",
  ]
  rows = []
  for summary in summaries:
    agg = summary["aggregate_all_episodes"]
    row = {k: summary.get(k) for k in fields[:5]}
    for name in fields[5:]:
      source = {"success_rate": "success", "fall_rate": "fall"}.get(name, name)
      row[name] = agg[source]["mean"]
    rows.append(row)
  with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  (args.output_dir / "protocol.json").write_text(json.dumps({
    "episodes_per_method_condition": args.episodes,
    "methods": {k: str(v.resolve()) for k, v in methods.items()},
    "world_model": str(frozen.resolve()),
    "conditions": {k: {"task": v[0], "soil_scale": v[1], "seed": SEEDS[k]}
                   for k, v in CONDITIONS.items()},
  }, indent=2) + "\n")
  print(f"complete: {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
  main()
