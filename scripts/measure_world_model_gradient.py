#!/usr/bin/env python3
"""Compare frozen M1/M2 BPTT gradients with a valid physical FD receipt."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import go2_supermimic.env  # noqa: F401
from go2_supermimic.diagnostics import capture_snapshot, restore_snapshot, world_model_open_loop
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  ContactStructuredWorldModel,
  ContactStructuredWorldModelV2,
)
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination, RewardWeights
from go2_supermimic.supermimic.world_model import WorldModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from measure_physical_gradient import SCENARIOS, generator_sequence, nominal_actions, prepare


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(8 * 1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def move_constants(constants: st.RobotConsts, device: str) -> st.RobotConsts:
  return st.RobotConsts(**{
    field: getattr(constants, field).to(device)
    for field in ("default_q", "action_scale", "kp", "kd", "effort", "soft_limits")
  })


def load_world_model(path: Path, device: str):
  checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  cfg = checkpoint["cfg"]
  method = cfg.get("method")
  hidden = tuple(cfg.get("wm_hidden", cfg.get("hidden", ())))
  if not hidden:
    raise ValueError(f"checkpoint {path} does not record world-model hidden widths")
  if method == "m1":
    model = WorldModel(hidden)
  elif method in {"m2", "m2_nomemory"}:
    model = ContactStructuredWorldModel(hidden, use_loading_memory=method == "m2")
  elif method in {"m2_v2", "m2_v2_core", "m2_v2_nomemory"}:
    model = ContactStructuredWorldModelV2(
      hidden,
      use_loading_memory=method != "m2_v2_nomemory",
      anchor_weight=0.0 if method == "m2_v2_core" else 0.25,
    )
  else:
    raise ValueError(f"checkpoint method {method!r} is not a supported world model")
  model.load_state_dict(checkpoint["world_model"])
  model.to(device).eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return method, model


def bptt_gradient(model, imagination, initial_state, initial_pose, actions,
                  initial_action_prev, context, generator, constants):
  differentiable_actions = actions.detach().clone().requires_grad_(True)
  result, _, _ = world_model_open_loop(
    imagination, model, initial_state, initial_pose, differentiable_actions,
    initial_action_prev, context, generator, constants,
  )
  gradient = torch.autograd.grad(result.value.sum(), differentiable_actions)[0][:, 0]
  return float(result.value[0].detach()), gradient[0].detach().cpu()


def comparison(physical: torch.Tensor, predicted: torch.Tensor, valid: torch.Tensor) -> dict:
  p, q = physical[valid].double(), predicted[valid].double()
  p_norm, q_norm = float(p.norm()), float(q.norm())
  if not len(p) or p_norm < 1.0e-12 or q_norm < 1.0e-12:
    return {
      "valid_coordinates": int(len(p)), "cosine_similarity": None,
      "relative_l2_error": None, "active_sign_agreement": None,
      "physical_norm": p_norm, "model_norm": q_norm, "norm_ratio": None,
    }
  return {
    "valid_coordinates": int(len(p)),
    "cosine_similarity": float(torch.dot(p, q) / (p.norm() * q.norm())),
    "relative_l2_error": float((q - p).norm() / p.norm()),
    "active_sign_agreement": float((torch.sign(p) == torch.sign(q)).double().mean()),
    "physical_norm": p_norm,
    "model_norm": q_norm,
    "norm_ratio": q_norm / p_norm,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
  parser.add_argument("--seed", type=int, default=14011)
  parser.add_argument("--horizon", type=int, default=32)
  parser.add_argument("--physical", type=Path, required=True)
  parser.add_argument("--m1", type=Path)
  parser.add_argument("--m2", type=Path)
  parser.add_argument(
    "--model", action="append", default=[], metavar="LABEL=CHECKPOINT",
    help="repeatable frozen candidate; labels must be unique",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  physical_report = json.loads(args.physical.read_text(encoding="utf-8"))
  if physical_report["scenario"] != args.scenario or physical_report["horizon"] != args.horizon:
    raise SystemExit("physical receipt scenario/horizon mismatch")
  if "convergence" not in physical_report:
    raise SystemExit("physical receipt lacks epsilon-convergence classification")

  model_specs = []
  if args.m1 is not None:
    model_specs.append(("M1", args.m1))
  if args.m2 is not None:
    model_specs.append(("M2", args.m2))
  for spec in args.model:
    if "=" not in spec:
      raise SystemExit(f"invalid --model {spec!r}; expected LABEL=CHECKPOINT")
    label, path = spec.split("=", 1)
    if not label or not path:
      raise SystemExit(f"invalid --model {spec!r}; expected LABEL=CHECKPOINT")
    model_specs.append((label, Path(path)))
  labels = [label for label, _ in model_specs]
  if not labels or len(labels) != len(set(labels)):
    raise SystemExit("at least one uniquely labeled model is required")

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
    actions_cpu = nominal_actions(args.horizon, env.action_manager.total_action_dim, env.device)
    _, generator_cpu = generator_sequence(env, snapshot, actions_cpu)
    restore_snapshot(env, snapshot)
    term = env.command_manager.get_term("target")
    if term.height_field is not None:
      raise SystemExit("this runner currently supports only physically valid rigid/no-field samples")
    initial_state = st.robot_state(env, term.height_field).to(args.device)
    initial_pose = st.planar_pose(env).to(args.device)
    context = st.ContextReader(env).read().to(args.device)
    robot_constants = move_constants(st.robot_consts(env), args.device)
    weights = RewardWeights(**{
      field.name: cfg.rewards[field.name].weight
      for field in fields(RewardWeights) if field.name in cfg.rewards
    })
    imagination = Imagination(
      ImagineCfg(horizon=args.horizon, weights=weights, compile=False),
      robot_constants, term.cfg.target, cfg.observations["actor"].history_length,
      args.device, st.obs_terms(cfg), None,
    )
  finally:
    env.close()

  actions = actions_cpu.to(args.device)
  initial_action_prev = torch.zeros_like(actions[:, 0])
  generator = {name: value.to(args.device) for name, value in generator_cpu.items()}
  constants = {
    "default_q": robot_constants.default_q,
    "action_scale": robot_constants.action_scale,
    "soft_limits": robot_constants.soft_limits,
    "hint_weight": torch.tensor(0.0, device=args.device),
  }

  physical = torch.zeros(st.NUM_JOINTS)
  valid = torch.zeros(st.NUM_JOINTS, dtype=torch.bool)
  for row in physical_report["convergence"]:
    coordinate = row["coordinate"]
    physical[coordinate] = median(row["gradients"])
    valid[coordinate] = row["valid"]

  methods = {}
  for label, path in model_specs:
    method, model = load_world_model(path, args.device)
    objective, gradient = bptt_gradient(
      model, imagination, initial_state, initial_pose, actions,
      initial_action_prev, context, generator, constants,
    )
    methods[label] = {
      "checkpoint": str(path.resolve()),
      "checkpoint_sha256": sha256(path),
      "checkpoint_method": method,
      "objective": objective,
      "gradient": gradient.tolist(),
      "finite": bool(torch.isfinite(gradient).all()),
      "comparison": comparison(physical, gradient, valid),
    }
    del model
    torch.cuda.empty_cache()

  report = {
    "schema": 2,
    "status": "COMPLETE" if all(row["finite"] for row in methods.values()) else "FAIL",
    "seed": args.seed,
    "scenario": args.scenario,
    "task": task,
    "horizon": args.horizon,
    "physical_receipt": str(args.physical),
    "physical_receipt_sha256": sha256(args.physical),
    "physical_gradient_reduction": "median across epsilon on each coordinate",
    "physical_gradient": physical.tolist(),
    "physical_valid_coordinates": torch.where(valid)[0].tolist(),
    "physical_valid_fraction": float(valid.float().mean()),
    "methods": methods,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2))
  if report["status"] == "FAIL":
    raise SystemExit("non-finite BPTT gradient")


if __name__ == "__main__":
  main()
