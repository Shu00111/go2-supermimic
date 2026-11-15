#!/usr/bin/env python3
"""GPU replay-equivalence validation for the full simulator snapshot adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import go2_supermimic.env  # noqa: F401
from go2_supermimic.diagnostics import capture_snapshot, restore_snapshot
from go2_supermimic.supermimic.contact_labels import rigid_contact_force
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


SCENARIOS = (
  ("rigid_stable_stance", "Go2-Chase-Flat", "stable"),
  ("soft_stable_stance", "Go2-Chase-SoftTerrain", "stable"),
  ("touchdown", "Go2-Chase-SoftTerrain", "touchdown"),
  ("liftoff", "Go2-Chase-SoftTerrain", "liftoff"),
  ("hard_to_soft", "Go2-Chase-SoftTransition", "hard_to_soft"),
  ("soft_to_hard", "Go2-Chase-SoftToHard", "soft_to_hard"),
  ("alternating_material", "Go2-Chase-AlternatingMaterial", "stable"),
  ("simultaneous_mixed_feet", "Go2-Chase-MixedMaterialDiagnostic", "stable"),
)


def _sim_tensor(env, name: str) -> torch.Tensor:
  value = getattr(env.sim.data, name)
  return value if torch.is_tensor(value) else value._tensor


def _prepare(env, mode: str, zero: torch.Tensor) -> None:
  if mode in {"stable", "liftoff"}:
    for _ in range(12):
      env.step(zero)
  qpos, qvel = _sim_tensor(env, "qpos"), _sim_tensor(env, "qvel")
  if mode == "touchdown":
    qpos[:, 2] += 0.12
    qvel[:, 2] = -1.0
  elif mode == "liftoff":
    qvel[:, 2] = 1.5
  elif mode in {"hard_to_soft", "soft_to_hard"}:
    direction = 1.0 if mode == "hard_to_soft" else -1.0
    qpos[:, 0] = env.scene.env_origins[:, 0] - direction * 0.25
    qvel[:, 0] = direction * 1.5
  else:
    return
  env.sim.forward()
  env.sim.sense()
  if "rft" in env.action_manager.active_terms:
    env.action_manager.get_term("rft").reset()


def _frame(env, obs, reward, terminated, truncated) -> dict[str, torch.Tensor]:
  robot = env.scene["robot"].data
  out = {
    "qpos": _sim_tensor(env, "qpos").clone(),
    "qvel": _sim_tensor(env, "qvel").clone(),
    "base_pose": robot.root_link_pose_w.clone(),
    "base_velocity": torch.cat(
      [robot.root_link_lin_vel_w, robot.root_link_ang_vel_w], dim=-1
    ).clone(),
    "joint_pos": robot.joint_pos.clone(),
    "joint_vel": robot.joint_vel.clone(),
    "reward": reward.clone(),
    "terminated": terminated.clone(),
    "truncated": truncated.clone(),
    "observation": obs["actor"].clone(),
  }
  if "rft" in env.action_manager.active_terms:
    rft = env.action_manager.get_term("rft")
    out.update({
      "soft_mask": rft.last_soft_mask.clone(),
      "rft_force": rft.last_force.clone(),
      "soil_memory": rft.state.plastic_depth.clone(),
      "soil_position": rft.state.plastic_position.clone(),
      "soil_surface_z": rft.state.plastic_surface_z.clone(),
      "rigid_force": rigid_contact_force(env, rft.last_force).clone(),
    })
  return out


def _rollout(env, actions: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
  frames = []
  for action in actions:
    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(_frame(env, obs, reward, terminated, truncated))
  return frames


def _compare(first, second) -> dict[str, float]:
  maxima: dict[str, float] = {}
  for a, b in zip(first, second, strict=True):
    for name in a:
      error = float((a[name].double() - b[name].double()).abs().max())
      maxima[name] = max(maxima.get(name, 0.0), error)
  return maxima


def validate(name: str, task: str, mode: str, seed: int, horizon: int, device: str):
  torch.manual_seed(seed)
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  cfg.auto_reset = False
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  zero = torch.zeros((1, env.action_manager.total_action_dim), device=device)
  try:
    env.reset()
    _prepare(env, mode, zero)
    snapshot = capture_snapshot(env)
    axis = torch.arange(env.action_manager.total_action_dim, device=device)
    actions = [0.04 * torch.sin(axis + i).unsqueeze(0) for i in range(horizon)]
    restore_snapshot(env, snapshot)
    first = _rollout(env, actions)
    restore_snapshot(env, snapshot)
    second = _rollout(env, actions)
    errors = _compare(first, second)
    exact = all(value == 0.0 for value in errors.values())
    evidence = {}
    if "soft_mask" in first[0]:
      soft = torch.stack([frame["soft_mask"] for frame in first])
      rft = torch.stack([frame["rft_force"].norm(dim=-1) for frame in first])
      rigid = torch.stack([frame["rigid_force"].norm(dim=-1) for frame in first])
      evidence = {
        "soft_fraction_min": float(soft.float().mean((1, 2)).min()),
        "soft_fraction_max": float(soft.float().mean((1, 2)).max()),
        "max_rft_force_n": float(rft.max()),
        "max_rigid_force_n": float(rigid.max()),
        "material_mode_changed": bool((soft[1:] != soft[:-1]).any()) if len(soft) > 1 else False,
      }
    return {
      "scenario": name, "task": task, "mode": mode, "seed": seed,
      "horizon": horizon, "bitwise_exact": exact, "max_abs_error": errors,
      "evidence": evidence, "status": "PASS" if exact else "FAIL",
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=14011)
  parser.add_argument("--horizon", type=int, default=12)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  results = [validate(name, task, mode, args.seed, args.horizon, args.device)
             for name, task, mode in SCENARIOS]
  report = {
    "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
    "backend": "MuJoCo-Warp float32 CUDA",
    "criterion": "bitwise equality for every recorded tensor",
    "scenarios": results,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2))
  if report["status"] != "PASS":
    raise SystemExit("snapshot replay validation failed")


if __name__ == "__main__":
  main()
