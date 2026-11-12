#!/usr/bin/env python3
"""Bounded MuJoCo Warp smoke test for the new Stage 3 physics tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import go2_supermimic.env  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


TASKS = (
  "Go2-Chase-RoughSoftTerrain",
  "Go2-Chase-AlternatingMaterial",
  "Go2-Chase-MixedMaterialDiagnostic",
)


def validate(task: str, num_envs: int, steps: int, seed: int, device: str) -> dict:
  torch.manual_seed(seed)
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  obs, _ = env.reset()
  rft = env.action_manager.get_term("rft")
  action = torch.zeros((num_envs, env.action_manager.total_action_dim), device=env.device)
  saw_soft = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  saw_rigid = torch.zeros_like(saw_soft)
  max_force = 0.0
  reset_count = 0
  try:
    for _ in range(steps):
      obs, _, terminated, truncated, _ = env.step(action)
      finite = (
        torch.isfinite(obs["actor"]).all()
        & torch.isfinite(rft.last_force).all()
        & torch.isfinite(rft.state.plastic_depth).all()
      )
      if not bool(finite):
        raise RuntimeError(f"non-finite state in {task}")
      soft = rft.last_signed_distance > 0
      saw_soft |= soft.any(dim=1)
      saw_rigid |= (~soft).any(dim=1)
      if rft.last_force[~soft].count_nonzero():
        raise RuntimeError(f"RFT force present on rigid material in {task}")
      max_force = max(max_force, float(rft.last_force.norm(dim=-1).max()))
      reset_count += int((terminated | truncated).sum())
    if "Alternating" in task or "MixedMaterial" in task:
      if not bool((saw_soft & saw_rigid).any()):
        raise RuntimeError(f"no mixed-material environment observed in {task}")
    if rft.cfg.use_local_height:
      feet = rft._entity.data.site_pos_w[:, rft.site_ids]
      centre, faces = rft._surface_height(feet)
      if not (torch.isfinite(centre).all() and faces is not None and torch.isfinite(faces).all()):
        raise RuntimeError("rough local-height query is non-finite")
    return {
      "task": task, "num_envs": num_envs, "steps": steps, "seed": seed,
      "max_rft_force_n": max_force, "episode_resets": reset_count,
      "mixed_material_envs": int((saw_soft & saw_rigid).sum()), "status": "PASS",
    }
  finally:
    env.close()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--steps", type=int, default=100)
  ap.add_argument("--seed", type=int, default=14011)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--output", type=Path)
  args = ap.parse_args()
  if not 1 <= args.num_envs <= 32 or not 1 <= args.steps <= 200:
    raise SystemExit("bounded validation requires 1..32 envs and 1..200 steps")
  report = [validate(t, args.num_envs, args.steps, args.seed, args.device) for t in TASKS]
  text = json.dumps(report, indent=2) + "\n"
  if args.output:
    if args.output.exists():
      raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
  print(text, end="")


if __name__ == "__main__":
  main()
