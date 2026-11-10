#!/usr/bin/env python3
"""One-condition, one-policy held-out evaluation for the soft-terrain study.

Each CUDA environment supplies exactly its first episode.  The policy is
deterministic, observation corruption is disabled, and the condition seed is
fixed independently of the training seed.  Per-episode data and aggregate
statistics are written rather than inferred from training logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

# Make direct ``python scripts/...`` invocation work from an uninstalled checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import go2_supermimic.env  # noqa: F401
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target import terrain as tr
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


CONDITIONS: dict[str, tuple[str, float | None]] = {
  "hard": ("Go2-Chase-Flat", None),
  "soft_nominal_1.0": ("Go2-Chase-SoftTerrain", 1.0),
  "soft_seen_0.3": ("Go2-Chase-SoftTerrain", 0.3),
  "soft_seen_0.6": ("Go2-Chase-SoftTerrain", 0.6),
  "hard_to_soft": ("Go2-Chase-SoftTransition", 1.0),
  "soft_to_hard": ("Go2-Chase-SoftToHard", 1.0),
  "soft_unseen_0.2": ("Go2-Chase-SoftTerrain", 0.2),
  "soft_unseen_1.2": ("Go2-Chase-SoftTerrain", 1.2),
}

METRIC_DEFINITIONS = {
  "success": "episode reached the 20 s time limit without fell_over termination",
  "forward_progress_m": "integral of planar velocity along instantaneous body heading",
  "path_length_m": "integral of planar speed",
  "net_displacement_m": "planar distance from initial to final pre-reset pose",
  "pursuit_velocity_mae_mps": (
    "absolute error between robot and chase-target velocity components along robot-to-target"
  ),
  "slip_speed_mps": "mean horizontal foot speed while that foot is in rigid or RFT contact",
  "sinkage_m": "mean RFT maximum immersed depth over soft-contact foot samples",
  "soil_force_n": "mean RFT force norm over soft-contact foot samples",
  "soil_dissipated_power_w": "mean max(0,-RFT wrench power) over soft-contact foot samples",
  "actuator_abs_power_w": "mean sum(abs(joint generalized force * joint velocity))",
  "actuator_work_per_path_m_jpm": "actuator absolute work divided by planar path length",
  "contact_transitions_per_min": "per-foot contact-state changes divided by episode minutes",
  "wm_one_step_normalized_mse": "world-model normalized state-plus-pose-delta MSE",
  "wm_eight_step_normalized_state_mse": (
    "8-step autoregressive normalized state MSE under recorded actions and relief"
  ),
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def finite_or_none(value: float) -> float | None:
  return value if math.isfinite(value) else None


def stats(values: list[float]) -> dict[str, float | int | None]:
  clean = torch.tensor([x for x in values if math.isfinite(x)], dtype=torch.float64)
  if clean.numel() == 0:
    return {"n": 0, "mean": None, "std": None, "median": None}
  return {
    "n": int(clean.numel()),
    "mean": float(clean.mean()),
    "std": float(clean.std(unbiased=False)),
    "median": float(clean.median()),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--method", choices=("B0", "B1"), required=True)
  parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
  parser.add_argument("--policy-checkpoint", type=Path, required=True)
  parser.add_argument("--world-model-checkpoint", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=128)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.episodes < 128:
    raise ValueError("formal protocol requires at least 128 episodes per condition")

  torch.manual_seed(args.seed)
  task, soil_scale = CONDITIONS[args.condition]
  cfg = load_env_cfg(task, play=False)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.episodes
  cfg.observations["actor"].enable_corruption = False
  cfg.commands["target"].mode_probs = (1.0, 0.0, 0.0)  # chase-only locomotion protocol
  cfg.commands["target"].target.curriculum_steps = 1
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
  if soil_scale is not None:
    cfg.actions["rft"].coefficient_scale_range = (soil_scale, soil_scale)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.common_step_counter = 1
  obs, _ = env.reset()
  dev, n, dt = env.device, args.episodes, env.step_dt
  robot = env.scene["robot"]
  term = env.command_manager.get_term("target")
  foot_ids, _ = robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)

  policy_ckpt = torch.load(args.policy_checkpoint, map_location=dev, weights_only=False)
  hidden = tuple(policy_ckpt["cfg"]["actor_hidden"])
  actor = Actor(obs["actor"].shape[-1], hidden, st.NUM_JOINTS).to(dev)
  actor.load_state_dict(policy_ckpt["actor"])
  actor.eval()

  wm_ckpt = torch.load(args.world_model_checkpoint, map_location=dev, weights_only=False)
  wm = WorldModel(tuple(wm_ckpt["cfg"]["wm_hidden"])).to(dev)
  wm.load_state_dict(wm_ckpt["world_model"])
  wm.eval()
  wm_scale = wm.out_norm.var.sqrt() + 1.0e-6
  state_scale = wm_scale[: st.STATE_DIM]
  context = st.ContextReader(env).read()
  field = term.height_field

  active = torch.ones(n, dtype=torch.bool, device=dev)
  finished = torch.zeros_like(active)
  steps = torch.zeros(n, device=dev)
  returns = torch.zeros(n, device=dev)
  falls = torch.zeros(n, device=dev)
  successes = torch.zeros(n, device=dev)
  forward = torch.zeros(n, device=dev)
  path = torch.zeros(n, device=dev)
  pursuit_error = torch.zeros(n, device=dev)
  actuator_energy = torch.zeros(n, device=dev)
  actuator_power_sum = torch.zeros(n, device=dev)
  contact_changes = torch.zeros(n, device=dev)
  slip_sum = torch.zeros(n, device=dev)
  slip_count = torch.zeros(n, device=dev)
  sink_sum = torch.zeros(n, device=dev)
  soil_force_sum = torch.zeros(n, device=dev)
  soil_power_sum = torch.zeros(n, device=dev)
  soil_count = torch.zeros(n, device=dev)
  wm1_sum = torch.zeros(n, device=dev)
  wm1_count = torch.zeros(n, device=dev)
  wm8_sum = torch.zeros(n, device=dev)
  wm8_count = torch.zeros(n, device=dev)
  start_pos = robot.data.root_link_pos_w[:, :2].clone()
  final_pos = start_pos.clone()
  previous_contact = torch.zeros((n, 4), dtype=torch.bool, device=dev)
  previous_action = torch.zeros((n, st.NUM_JOINTS), device=dev)
  rollout_state = st.robot_state(env, field=field)
  rollout_age = torch.zeros(n, dtype=torch.long, device=dev)
  max_steps = env.max_episode_length + 2

  with torch.inference_mode():
    for _ in range(max_steps):
      if not active.any():
        break
      data = robot.data
      pos = data.root_link_pos_w[:, :2]
      vel = data.root_link_lin_vel_w[:, :2]
      heading = data.heading_w
      speed = vel.norm(dim=-1)
      fwd = torch.stack([heading.cos(), heading.sin()], dim=-1)
      target_rel = term.gen.chase_pos[:, :2] - pos
      target_dir = target_rel / target_rel.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
      robot_along = (vel * target_dir).sum(-1)
      target_along = (term.gen.chase_vel[:, :2] * target_dir).sum(-1)
      pursuit_error += active * (robot_along - target_along).abs()
      forward += active * (vel * fwd).sum(-1) * dt
      path += active * speed * dt
      final_pos[active] = pos[active]

      joint_power = (data.qfrc_actuator * data.joint_vel).abs().sum(-1)
      actuator_power_sum += active * joint_power
      actuator_energy += active * joint_power * dt

      rigid_contact = env.scene.sensors["feet_ground_contact"].data.found > 0
      contact = rigid_contact
      if "rft" in env.action_manager.active_terms:
        rft = env.action_manager.get_term("rft")
        soft_contact = (rft.last_sinkage > 0.0) & (rft.last_signed_distance > 0.0)
        contact = contact | soft_contact
        soil_mask = active.unsqueeze(-1) & soft_contact
        sink_sum += (rft.last_sinkage * soil_mask).sum(-1)
        soil_force_sum += (rft.last_force.norm(dim=-1) * soil_mask).sum(-1)
        soil_power_sum += ((-rft.last_power).clamp_min(0.0) * soil_mask).sum(-1)
        soil_count += soil_mask.sum(-1)
      active_feet = active.unsqueeze(-1) & contact
      foot_speed = data.site_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
      slip_sum += (foot_speed * active_feet).sum(-1)
      slip_count += active_feet.sum(-1)
      contact_changes += (active.unsqueeze(-1) & (contact != previous_contact)).sum(-1)
      previous_contact = contact.clone()

      s0 = st.robot_state(env, field=field)
      pose0 = st.planar_pose(env)
      relief = tr.relief(pose0[:, :2], pose0[:, 2], field, term.obstacle_layout)
      action = actor(obs["actor"])
      one_state, one_pose = wm(s0, action, previous_action, context, relief)
      roll_state_next, _ = wm(rollout_state, action, previous_action, context, relief)
      obs, reward, terminated, truncated, _ = env.step(action)
      done = terminated | truncated
      valid = active & ~done
      s1 = st.robot_state(env, field=field)
      pose_delta = st.pose_delta(pose0, st.planar_pose(env))
      one_err = torch.cat([one_state - s1, one_pose - pose_delta], dim=-1) / wm_scale
      wm1_sum += valid * one_err.square().mean(-1)
      wm1_count += valid
      next_age = rollout_age + 1
      at_eight = valid & (next_age == 8)
      eight_err = ((roll_state_next - s1) / state_scale).square().mean(-1)
      wm8_sum += at_eight * eight_err
      wm8_count += at_eight
      reset_roll = done | at_eight
      rollout_state = torch.where(reset_roll.unsqueeze(-1), s1, roll_state_next)
      rollout_age = torch.where(reset_roll, torch.zeros_like(next_age), next_age)

      returns += active * reward
      steps += active
      newly_done = active & done
      falls += newly_done & terminated
      successes += newly_done & truncated & ~terminated
      active &= ~newly_done
      finished |= newly_done
      previous_action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)

  if not bool(finished.all()):
    env.close()
    raise RuntimeError(f"only {int(finished.sum())}/{n} episodes completed")

  duration = steps * dt
  displacement = (final_pos - start_pos).norm(dim=-1)
  eps = 1.0e-12
  rows = []
  for i in range(n):
    soil_n = float(soil_count[i])
    slip_n = float(slip_count[i])
    row = {
      "method": args.method,
      "condition": args.condition,
      "evaluation_seed": args.seed,
      "episode": i,
      "success": float(successes[i]),
      "fall": float(falls[i]),
      "duration_s": float(duration[i]),
      "episode_return": float(returns[i]),
      "forward_progress_m": float(forward[i]),
      "path_length_m": float(path[i]),
      "net_displacement_m": float(displacement[i]),
      "pursuit_velocity_mae_mps": float(pursuit_error[i] / steps[i].clamp_min(1)),
      "slip_speed_mps": float(slip_sum[i] / max(slip_n, 1.0)) if slip_n else math.nan,
      "sinkage_m": float(sink_sum[i] / max(soil_n, 1.0)) if soil_n else math.nan,
      "soil_force_n": float(soil_force_sum[i] / max(soil_n, 1.0)) if soil_n else math.nan,
      "soil_dissipated_power_w": (
        float(soil_power_sum[i] / max(soil_n, 1.0)) if soil_n else math.nan
      ),
      "actuator_abs_power_w": float(actuator_power_sum[i] / steps[i].clamp_min(1)),
      "actuator_work_per_path_m_jpm": float(actuator_energy[i] / path[i].clamp_min(1.0e-3)),
      "contact_transitions_per_min": float(contact_changes[i] / duration[i].clamp_min(eps) * 60),
      "wm_one_step_normalized_mse": float(wm1_sum[i] / wm1_count[i].clamp_min(1)),
      "wm_eight_step_normalized_state_mse": (
        float(wm8_sum[i] / wm8_count[i].clamp_min(1)) if wm8_count[i] else math.nan
      ),
    }
    rows.append(row)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  stem = f"{args.method}_{args.condition}"
  episode_csv = args.output_dir / f"{stem}_episodes.csv"
  with episode_csv.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  metric_names = [
    name for name in rows[0]
    if name not in {"method", "condition", "evaluation_seed", "episode"}
  ]
  aggregate = {name: stats([float(row[name]) for row in rows]) for name in metric_names}
  success_rows = [row for row in rows if row["success"] > 0.5]
  success_only = {
    name: stats([float(row[name]) for row in success_rows])
    for name in ("actuator_abs_power_w", "actuator_work_per_path_m_jpm")
  }
  summary = {
    "method": args.method,
    "condition": args.condition,
    "task": task,
    "soil_coefficient_scale": soil_scale,
    "evaluation_seed": args.seed,
    "episodes": n,
    "episode_length_s": cfg.episode_length_s,
    "policy_checkpoint": str(args.policy_checkpoint.resolve()),
    "policy_sha256": sha256(args.policy_checkpoint),
    "world_model_checkpoint": str(args.world_model_checkpoint.resolve()),
    "world_model_sha256": sha256(args.world_model_checkpoint),
    "actor_observation_corruption": False,
    "target_mode": "chase_only",
    "metric_definitions": METRIC_DEFINITIONS,
    "aggregate_all_episodes": aggregate,
    "aggregate_successful_episodes": success_only,
    "episode_csv": str(episode_csv.resolve()),
  }
  summary_path = args.output_dir / f"{stem}_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
  print(json.dumps({
    "method": args.method,
    "condition": args.condition,
    "success_rate": aggregate["success"]["mean"],
    "fall_rate": aggregate["fall"]["mean"],
    "forward_progress_m": aggregate["forward_progress_m"]["mean"],
    "path": str(summary_path),
  }, sort_keys=True), flush=True)
  env.close()


if __name__ == "__main__":
  main()
