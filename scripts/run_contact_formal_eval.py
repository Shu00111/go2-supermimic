#!/usr/bin/env python3
"""Restart-safe frozen Stage-4 matrix for formal M1/M2 seed 14011."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from eval_soft_terrain_formal import CONDITIONS


STAGE4_CONDITIONS = (
  "rigid_rough_ood",
  "soft_strength_ood",
  "hard_to_soft_ood",
  "soft_to_hard_ood",
  "initial_state_ood",
  "command_ood",
)
EVAL_SEEDS = (90101, 90103, 90107)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def final_checkpoint(repo: Path, method: str) -> Path:
  matches = list((repo / "outputs/contact_structured_supermimic/formal_seed14011/seed14011" /
                  method.lower()).glob("stage3/*/*/model_12000.pt"))
  if len(matches) != 1:
    raise RuntimeError(f"expected one final {method} checkpoint, found {matches}")
  return matches[0]


def valid_summary(path: Path, *, episodes: int, seed: int, checkpoint_hash: str) -> bool:
  try:
    summary = json.loads(path.read_text())
    return (
      summary["episodes"] == episodes
      and summary["evaluation_seed"] == seed
      and summary["policy_sha256"] == checkpoint_hash
      and summary["world_model_sha256"] == checkpoint_hash
    )
  except (OSError, KeyError, json.JSONDecodeError):
    return False


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
  parser.add_argument("--episodes", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--python", default=sys.executable)
  args = parser.parse_args()
  if args.episodes < 128:
    raise ValueError("formal Stage-4 protocol requires at least 128 episodes")

  checkpoints = {method: final_checkpoint(args.repo, method) for method in ("M1", "M2")}
  hashes = {method: sha256(path) for method, path in checkpoints.items()}
  args.output_dir.mkdir(parents=True, exist_ok=True)
  protocol = {
    "manifest": "experiments/held_out_manifest.json",
    "manifest_version": "rss2027-stage4-v1",
    "training_seed": 14011,
    "evaluation_seeds": list(EVAL_SEEDS),
    "episodes_per_method_condition_seed": args.episodes,
    "checkpoint_selection": "final Stage-3 model_12000; frozen before Stage 4",
    "methods": {
      method: {"checkpoint": str(path.resolve()), "sha256": hashes[method]}
      for method, path in checkpoints.items()
    },
    "conditions": {
      name: {"task": CONDITIONS[name][0], "soil_scale": CONDITIONS[name][1]}
      for name in STAGE4_CONDITIONS
    },
    "condition_mapping_note": (
      "The v1 manifest froze condition names but omitted task parameters. This mapping was "
      "recorded before the first Stage-4 rollout and was not tuned on Stage-4 outcomes."
    ),
  }
  protocol_path = args.output_dir / "protocol.json"
  if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
    raise RuntimeError(f"refusing to change existing Stage-4 protocol: {protocol_path}")
  protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

  child_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for seed in EVAL_SEEDS:
    seed_dir = args.output_dir / f"seed{seed}"
    seed_dir.mkdir(exist_ok=True)
    for method, checkpoint in checkpoints.items():
      for condition in STAGE4_CONDITIONS:
        summary = seed_dir / f"{method}_{condition}_summary.json"
        if valid_summary(summary, episodes=args.episodes, seed=seed,
                         checkpoint_hash=hashes[method]):
          print(f"skip validated {summary}", flush=True)
          continue
        command = [
          args.python, str(args.repo / "scripts/eval_soft_terrain_formal.py"),
          "--method", method, "--condition", condition,
          "--policy-checkpoint", str(checkpoint),
          "--world-model-checkpoint", str(checkpoint),
          "--episodes", str(args.episodes), "--seed", str(seed),
          "--output-dir", str(seed_dir), "--device", args.device,
        ]
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, cwd=args.repo, env=child_env, check=True)
  (args.output_dir / "COMPLETE").write_text("Stage-4 matrix complete\n")


if __name__ == "__main__":
  main()
