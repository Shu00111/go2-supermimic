#!/usr/bin/env python3
"""Collect a frozen, model-independent P1 transition audit dataset.

The simulator oracle contact state is stored only as an audit/supervision label.
Actions come from one frozen deterministic actor and are shared by every world
model evaluated later.  Environment id is retained so complete trajectories can
be assigned to train/dev/held-out partitions without window leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import go2_supermimic.env  # noqa: F401
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_labels import oracle_contact_state
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.target import terrain as tr
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


CONDITIONS: dict[str, tuple[str, float | None]] = {
  "rigid_rough": ("Go2-Chase-Terrain", None),
  "soft_rough": ("Go2-Chase-RoughSoftTerrain", 1.0),
  "hard_to_soft": ("Go2-Chase-SoftTransition-Scan", 0.6),
  "soft_to_hard": ("Go2-Chase-SoftToHard-Scan", 0.6),
  "alternating": ("Go2-Chase-AlternatingMaterial-Scan", 0.6),
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def git_head() -> str:
  return subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
  ).strip()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
  parser.add_argument("--policy-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=512)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.seed in {90101, 90103, 90107}:
    raise ValueError("Stage-4 held-out evaluation seeds are forbidden in P1 development")
  if args.num_envs < 3 or args.steps < 2:
    raise ValueError("audit collection needs at least 3 environments and 2 steps")
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite frozen audit dataset: {args.output}")

  torch.manual_seed(args.seed)
  task, soil_scale = CONDITIONS[args.condition]
  cfg = load_env_cfg(task, play=False)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.observations["actor"].enable_corruption = False
  cfg.commands["target"].mode_probs = (1.0, 0.0, 0.0)
  cfg.commands["target"].target.curriculum_steps = 1
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  if soil_scale is not None:
    cfg.actions["rft"].coefficient_scale_range = (soil_scale, soil_scale)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.common_step_counter = 1
  obs, _ = env.reset()
  dev = env.device
  checkpoint = torch.load(
    args.policy_checkpoint, map_location="cpu", weights_only=False, mmap=True
  )
  actor = Actor(
    obs["actor"].shape[-1], tuple(checkpoint["cfg"]["actor_hidden"]), st.NUM_JOINTS
  ).to(dev)
  actor.load_state_dict(checkpoint["actor"])
  actor.eval()
  del checkpoint

  term = env.command_manager.get_term("target")
  field = term.height_field
  context_reader = st.ContextReader(env)
  previous_action = torch.zeros(args.num_envs, st.NUM_JOINTS, device=dev)
  episode = torch.zeros(args.num_envs, dtype=torch.int32, device=dev)
  rows: dict[str, list[torch.Tensor]] = {name: [] for name in (
    "state", "next_state", "pose_delta", "action", "previous_action", "context",
    "relief", "contact", "next_contact", "done", "episode", "soil_strength",
    "boundary_distance", "soft_mask",
  )}

  with torch.inference_mode():
    for _ in range(args.steps):
      s0 = st.robot_state(env, field=field)
      pose0 = st.planar_pose(env)
      relief = tr.relief(pose0[:, :2], pose0[:, 2], field, term.obstacle_layout)
      z0 = oracle_contact_state(env)
      action = actor(obs["actor"])
      context = context_reader.read()
      if "rft" in env.action_manager.active_terms:
        rft = env.action_manager.get_term("rft")
        strength = rft.coefficient_scale.clone()
        boundary = rft.last_signed_distance.clone()
        soft = rft.last_soft_mask.clone()
      else:
        strength = torch.zeros(args.num_envs, device=dev)
        boundary = torch.zeros(args.num_envs, 4, device=dev)
        soft = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=dev)

      obs_next, _, terminated, truncated, _ = env.step(action)
      done = terminated | truncated
      s1 = st.robot_state(env, field=field)
      z1 = oracle_contact_state(env)
      delta = st.pose_delta(pose0, st.planar_pose(env))

      values = {
        "state": s0, "next_state": s1, "pose_delta": delta, "action": action,
        "previous_action": previous_action, "context": context, "relief": relief,
        "contact": z0, "next_contact": z1, "done": done, "episode": episode,
        "soil_strength": strength, "boundary_distance": boundary, "soft_mask": soft,
      }
      for name, value in values.items():
        rows[name].append(value.detach().cpu())

      previous_action = torch.where(done[:, None], torch.zeros_like(action), action)
      episode = episode + done.to(episode.dtype)
      obs = obs_next

  env.close()
  # Environment-major layout keeps each complete trajectory contiguous.
  data = {name: torch.stack(values, dim=0).transpose(0, 1).contiguous()
          for name, values in rows.items()}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  metadata = {
    "schema": 1,
    "purpose": "P1 world-model error attribution; never Stage-4 final evaluation",
    "condition": args.condition,
    "task": task,
    "soil_strength_scale": soil_scale,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "split_rule": "environment_id modulo 10: 0-5 train, 6-7 dev, 8-9 held_out",
    "policy_checkpoint": str(args.policy_checkpoint.resolve()),
    "policy_sha256": sha256(args.policy_checkpoint),
    "source_commit": git_head(),
    "created_utc": datetime.now(timezone.utc).isoformat(),
  }
  torch.save({"metadata": metadata, "data": data}, args.output)
  receipt = args.output.with_suffix(args.output.suffix + ".json")
  receipt.write_text(json.dumps({**metadata, "dataset_sha256": sha256(args.output)},
                                indent=2) + "\n")
  print(json.dumps({"output": str(args.output), "receipt": str(receipt),
                    "dataset_sha256": sha256(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
  main()
