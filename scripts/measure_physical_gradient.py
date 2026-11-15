#!/usr/bin/env python3
"""Centered physical finite differences for the frozen P0 common objective.

This script intentionally runs one isolated CPU environment.  CUDA replay did
not pass Gate A around material transitions and must not be used as ground
truth.  Future actions and the target-generator sequence are held fixed; only
one coordinate of the first normalized action is perturbed per branch.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch
import warp as wp

import go2_supermimic.env  # noqa: F401
from go2_supermimic.diagnostics import (
  capture_snapshot,
  common_objective,
  finite_difference_convergence,
  restore_snapshot,
)
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import pack_generator, unpack_generator
from go2_supermimic.supermimic.contact_labels import rigid_contact_force
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination, RewardWeights
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


SCENARIOS = {
  "rigid_stable_stance": ("Go2-Chase-Flat", "stable"),
  "soft_stable_stance": ("Go2-Chase-SoftTerrain", "stable"),
  "touchdown": ("Go2-Chase-SoftTerrain", "touchdown"),
  "liftoff": ("Go2-Chase-SoftTerrain", "liftoff"),
  "aerial": ("Go2-Chase-SoftTerrain", "aerial"),
  "rigid_aerial": ("Go2-Chase-Flat", "aerial"),
  "hard_to_soft": ("Go2-Chase-SoftTransition", "hard_to_soft"),
  "soft_to_hard": ("Go2-Chase-SoftToHard", "soft_to_hard"),
}


def prepare(env, mode: str, zero: torch.Tensor) -> None:
  for _ in range(12):
    env.step(zero)
  data = env.sim.data
  qpos = data.qpos if torch.is_tensor(data.qpos) else data.qpos._tensor
  qvel = data.qvel if torch.is_tensor(data.qvel) else data.qvel._tensor
  if mode == "touchdown":
    qpos[:, 2] += 0.12
    qvel[:, 2] = -1.0
  elif mode == "liftoff":
    qvel[:, 2] = 1.5
  elif mode == "aerial":
    qpos[:, 2] += 0.25
    qvel[:, 2] = 0.2
  elif mode in {"hard_to_soft", "soft_to_hard"}:
    direction = 1.0 if mode == "hard_to_soft" else -1.0
    qpos[:, 0] = env.scene.env_origins[:, 0] - direction * 0.25
    qvel[:, 0] = direction * 1.5
  elif mode != "stable":
    raise ValueError(f"unknown preparation mode {mode!r}")
  else:
    return
  env.sim.forward()
  env.sim.sense()
  if "rft" in env.action_manager.active_terms:
    env.action_manager.get_term("rft").reset()


def nominal_actions(horizon: int, action_dim: int, device: str) -> torch.Tensor:
  """A fixed, nonsaturating excitation in normalized policy coordinates."""
  step = torch.arange(horizon, device=device, dtype=torch.float32).unsqueeze(-1)
  axis = torch.arange(action_dim, device=device, dtype=torch.float32).unsqueeze(0)
  return (0.12 * torch.sin(0.37 * step + 0.71 * axis)).unsqueeze(0)


def warp_tensor(value) -> torch.Tensor:
  return value if torch.is_tensor(value) else wp.to_torch(value)


def generator_sequence(env, snapshot, actions: torch.Tensor):
  """Record the nominal pre-step and post-step generator states."""
  restore_snapshot(env, snapshot)
  term = env.command_manager.get_term("target")
  before, after = [], []
  for step in range(actions.shape[1]):
    before.append(pack_generator(term.gen).clone())
    env.step(actions[:, step])
    after.append({name: value.clone() for name, value in term.gen.state().items()})
  packed_before = torch.stack(before, 1)
  post = {name: torch.stack([state[name] for state in after], 1) for name in after[0]}
  return packed_before, post


def rollout(env, snapshot, actions, generator_before, generator_after, imagination, constants):
  restore_snapshot(env, snapshot)
  term = env.command_manager.get_term("target")
  field = term.height_field
  states = [st.robot_state(env, field)]
  poses = [st.planar_pose(env)]
  previous = [torch.zeros_like(actions[:, 0])]
  terminal, contact, rigid, rft_contact, soft = [], [], [], [], []
  constraints = {name: [] for name in ("ne", "nf", "nl", "nefc", "nisland")}
  contact_set = {name: [] for name in ("worldid", "type", "geom", "dim")}
  for step in range(actions.shape[1]):
    unpack_generator(term.gen, generator_before[:, step])
    _, _, terminated, truncated, _ = env.step(actions[:, step])
    states.append(st.robot_state(env, field))
    poses.append(st.planar_pose(env))
    terminal.append((terminated | truncated).clone())
    contact.append((st.feet_world_z(states[-1]) < 0.035).clone())
    for name in constraints:
      constraints[name].append(warp_tensor(getattr(env.sim.wp_data, name)).clone())
    for name in contact_set:
      contact_set[name].append(warp_tensor(getattr(env.sim.wp_data.contact, name)).clone())
    if "rft" in env.action_manager.active_terms:
      rft = env.action_manager.get_term("rft")
      soft.append(rft.last_soft_mask.clone())
      rigid.append((rigid_contact_force(env, rft.last_force).norm(dim=-1) > 1.0).clone())
      rft_contact.append((rft.last_force.norm(dim=-1) > 1.0).clone())
    if step + 1 < actions.shape[1]:
      previous.append(actions[:, step])
  result = common_objective(
    imagination, torch.stack(states, 1), torch.stack(poses, 1), actions,
    torch.stack(previous, 1), generator_after, constants,
  )
  signature = {
    "terminal": torch.stack(terminal, 1),
    "contact": torch.stack(contact, 1),
    **{f"constraint_{name}": torch.stack(value, 1) for name, value in constraints.items()},
    **{f"contact_set_{name}": torch.stack(value, 0) for name, value in contact_set.items()},
  }
  if soft:
    signature["soft_material"] = torch.stack(soft, 1)
    signature["rigid_contact"] = torch.stack(rigid, 1)
    signature["rft_contact"] = torch.stack(rft_contact, 1)
  return result, signature


def signature_changed(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, bool]:
  return {name: bool((value != b[name]).any()) for name, value in a.items()}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
  parser.add_argument("--seed", type=int, default=14011)
  parser.add_argument("--horizon", type=int, default=32)
  parser.add_argument("--epsilon", type=float, nargs="+", default=(0.01, 0.003, 0.001))
  parser.add_argument("--coordinates", type=int, nargs="+", default=tuple(range(st.NUM_JOINTS)))
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  if any(axis < 0 or axis >= st.NUM_JOINTS for axis in args.coordinates):
    raise SystemExit("coordinates must be in [0, 11]")

  torch.manual_seed(args.seed)
  task, mode = SCENARIOS[args.scenario]
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.auto_reset = False
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    zero = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    prepare(env, mode, zero)
    snapshot = capture_snapshot(env)
    actions = nominal_actions(args.horizon, env.action_manager.total_action_dim, env.device)
    gen_before, gen_after = generator_sequence(env, snapshot, actions)

    term = env.command_manager.get_term("target")
    robot_constants = st.robot_consts(env)
    constants = {
      "default_q": robot_constants.default_q,
      "action_scale": robot_constants.action_scale,
      "soft_limits": robot_constants.soft_limits,
      "hint_weight": torch.tensor(0.0, device=env.device),
    }
    weights = RewardWeights(**{
      field.name: cfg.rewards[field.name].weight
      for field in fields(RewardWeights) if field.name in cfg.rewards
    })
    imagination = Imagination(
      ImagineCfg(horizon=args.horizon, weights=weights, compile=False),
      robot_constants, term.cfg.target, cfg.observations["actor"].history_length,
      env.device, st.obs_terms(cfg), term.height_field,
    )
    nominal, nominal_signature = rollout(
      env, snapshot, actions, gen_before, gen_after, imagination, constants
    )
    repeat, _ = rollout(env, snapshot, actions, gen_before, gen_after, imagination, constants)

    rows = []
    for epsilon in args.epsilon:
      for coordinate in args.coordinates:
        plus_actions, minus_actions = actions.clone(), actions.clone()
        plus_actions[:, 0, coordinate] += epsilon
        minus_actions[:, 0, coordinate] -= epsilon
        plus, plus_signature = rollout(
          env, snapshot, plus_actions, gen_before, gen_after, imagination, constants
        )
        minus, minus_signature = rollout(
          env, snapshot, minus_actions, gen_before, gen_after, imagination, constants
        )
        plus_value, minus_value = float(plus.value[0]), float(minus.value[0])
        rows.append({
          "epsilon": epsilon,
          "coordinate": coordinate,
          "plus_objective": plus_value,
          "minus_objective": minus_value,
          "centered_gradient": (plus_value - minus_value) / (2.0 * epsilon),
          "perturbation_signal": abs(plus_value - minus_value),
          "action_saturated": bool(
            plus_actions[:, 0, coordinate].abs().max() >= 1.0
            or minus_actions[:, 0, coordinate].abs().max() >= 1.0
          ),
          "branch_discontinuity": signature_changed(plus_signature, minus_signature),
          "nominal_discontinuity": signature_changed(plus_signature, nominal_signature),
          "nonfinite": not bool(torch.isfinite(plus.value).all() and torch.isfinite(minus.value).all()),
        })

    convergence = finite_difference_convergence(rows, float(nominal.value[0]))
    report = {
      "schema": 1,
      "status": "COMPLETE" if all(not row["nonfinite"] for row in rows) else "FAIL",
      "backend": "MuJoCo-Warp float32 CPU, isolated one-environment replay",
      "seed": args.seed,
      "scenario": args.scenario,
      "task": task,
      "horizon": args.horizon,
      "action_coordinates": "normalized 12-D policy/action-manager coordinates",
      "future_actions": "fixed nominal deterministic sequence",
      "generator_sequence": "frozen nominal per-step sequence",
      "objective": "reward-only SuperMimic common objective J_R; no learned endpoint value",
      "nominal_objective": float(nominal.value[0]),
      "replay_objective_abs_error": abs(float(nominal.value[0]) - float(repeat.value[0])),
      "convergence_rule": (
        "active signal above float32/replay floor, identical signs, relative gradient span "
        "<=0.25, no saturation, terminal/contact/constraint-set change, or non-finite value"
      ),
      "valid_coordinate_fraction": sum(row["valid"] for row in convergence) / len(convergence),
      "convergence": convergence,
      "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] == "FAIL":
      raise SystemExit("non-finite finite-difference result")
  finally:
    env.close()


if __name__ == "__main__":
  main()
