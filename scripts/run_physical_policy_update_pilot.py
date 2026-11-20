#!/usr/bin/env python3
"""Bounded shared-Actor gradient updates and paired CPU physics validation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import go2_supermimic.env  # noqa: F401
from go2_supermimic.diagnostics import capture_snapshot, restore_snapshot
from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, load_frozen_checkpoint
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination
from go2_supermimic.supermimic.train import clip_grads
from go2_supermimic.target.terrain import HeightField
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


ALPHAS = (1e-5, 3e-5, 1e-4)
VALIDATION = (
  ("rigid_validation", "Go2-Chase-Flat-Scan", 83001),
  ("soft_validation", "Go2-Chase-SoftTerrain-Scan", 83011),
)


def git_sha():
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def tensor_state_sha256(state):
  digest = hashlib.sha256()
  for key, value in sorted(state.items()):
    digest.update(key.encode()); digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def load_seed(path):
  bundle = torch.load(path, map_location="cpu", weights_only=False)
  seed = {key: value.clone() for key, value in bundle["imagination_seed"].items()}
  consts = st.RobotConsts(**bundle["robot_consts"])
  raw = bundle["height_field"]
  field = None if raw is None else HeightField(raw["h"], tuple(raw["origin"]), raw["res"])
  imagination = Imagination(
    ImagineCfg(horizon=8, compile=False, sensor_grad=True), consts, bundle["target_cfg"],
    bundle["history_length"], "cpu", tuple(tuple(x) for x in bundle["observation_terms"]), field,
  )
  return bundle, seed, imagination


def parameters_vector(actor):
  return torch.cat([p.detach().reshape(-1) for p in actor.parameters()])


def gradient_for_model(model, actor_source, critic, seeds):
  actor = copy.deepcopy(actor_source)
  for parameter in model.parameters(): parameter.requires_grad_(False)
  torch.manual_seed(20261120)
  values = []
  for _, seed, imagination in seeds:
    value, _ = imagination.rollout(model, actor, critic, seed)
    values.append(value.mean())
  objective = torch.stack(values).mean()
  grads = torch.autograd.grad(objective, tuple(actor.parameters()))
  vector = torch.cat([g.reshape(-1) for g in grads]).detach()
  for parameter in model.parameters(): parameter.requires_grad_(True)
  return float(objective.detach()), vector


def assign_delta(actor, delta):
  offset = 0
  with torch.no_grad():
    for parameter in actor.parameters():
      size = parameter.numel(); parameter.add_(delta[offset:offset + size].reshape_as(parameter))
      offset += size
  assert offset == delta.numel()


def evaluate_imagined(model, actor, critic, seeds):
  torch.manual_seed(20261120)
  values = []
  with torch.no_grad():
    for _, seed, imagination in seeds:
      value, _ = imagination.rollout(model, actor, critic, seed)
      values.append(value.mean())
  return float(torch.stack(values).mean())


def make_candidates(checkpoints, seeds, output):
  base_actor = checkpoints["m1"].actor
  critic = checkpoints["m1"].critic
  candidates = {"zero": copy.deepcopy(base_actor)}
  manifest, predictions = [], []
  base_hash = tensor_state_sha256(base_actor.state_dict())
  for method, checkpoint in checkpoints.items():
    before, gradient = gradient_for_model(checkpoint.world_model, base_actor, critic, seeds)
    gradient_norm = float(gradient.norm())
    for alpha in ALPHAS:
      actor = copy.deepcopy(base_actor)
      delta = alpha * gradient / gradient.norm().clamp_min(1e-15)
      assign_delta(actor, delta)
      candidate_id = f"{method}_normalized_{alpha:.0e}"
      candidates[candidate_id] = actor
      after = evaluate_imagined(checkpoint.world_model, actor, critic, seeds)
      predictions.append({"candidate_id": candidate_id, "method": method,
                          "update_type": "normalized_direction", "alpha": alpha,
                          "imagined_return_before": before, "imagined_return_after": after,
                          "predicted_delta": after - before})
      manifest.append({
        "candidate_id": candidate_id, "method": method, "update_type": "normalized_direction",
        "gradient_norm": gradient_norm, "update_norm": float(delta.norm()),
        "base_actor_sha256": base_hash, "candidate_actor_sha256": tensor_state_sha256(actor.state_dict()),
      })

    actor = copy.deepcopy(base_actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=checkpoint.raw["cfg"]["lr"], betas=(0.9, 0.95))
    # A shared M1 Actor requires one shared optimizer state; this is a clearly
    # labeled counterfactual, not the historical M2 optimizer trajectory.
    optimizer.load_state_dict(copy.deepcopy(checkpoints["m1"].raw["opt_actor"]))
    offset = 0
    for parameter in actor.parameters():
      size = parameter.numel()
      parameter.grad = -gradient[offset:offset + size].reshape_as(parameter).clone()
      offset += size
    unclipped = float(gradient.norm())
    clipped = float(clip_grads(actor.parameters(), 1.0, 0.1))
    before_vector = parameters_vector(actor).clone()
    optimizer.step()
    delta = parameters_vector(actor) - before_vector
    candidate_id = f"{method}_m1_adam_state"
    candidates[candidate_id] = actor
    after = evaluate_imagined(checkpoint.world_model, actor, critic, seeds)
    predictions.append({"candidate_id": candidate_id, "method": method,
                        "update_type": "m1_optimizer_state_counterfactual", "alpha": None,
                        "imagined_return_before": before, "imagined_return_after": after,
                        "predicted_delta": after - before})
    manifest.append({
      "candidate_id": candidate_id, "method": method,
      "update_type": "m1_optimizer_state_counterfactual", "gradient_norm": gradient_norm,
      "pre_clip_global_norm": unclipped, "clip_returned_norm": clipped,
      "update_norm": float(delta.norm()), "optimizer_lr": optimizer.param_groups[0]["lr"],
      "base_actor_sha256": base_hash, "candidate_actor_sha256": tensor_state_sha256(actor.state_dict()),
    })
  output.mkdir(parents=True, exist_ok=True)
  torch.save({name: actor.state_dict() for name, actor in candidates.items()}, output / "candidate_actors.pt")
  return candidates, manifest, predictions


def rollout_physics(env, snapshot, initial_obs, actor, horizon, action_clip):
  restore_snapshot(env, snapshot)
  obs = initial_obs.clone()
  robot = env.scene["robot"]
  term = env.command_manager.get_term("target")
  consts = st.robot_consts(env)
  start = robot.data.root_link_pos_w[:, :2].clone()
  heading0 = robot.data.heading_w.clone()
  returns = torch.zeros(1); velocity_error = torch.zeros(1)
  action_sq = torch.zeros(1); saturated = torch.zeros(1); violations = torch.zeros(1)
  actions = []; survived = 0; fell = False
  for _ in range(horizon):
    with torch.no_grad(): action = actor(obs)
    actions.append(action.clone())
    next_obs, reward, terminated, truncated, _ = env.step(action)
    returns += reward.cpu()
    data = robot.data
    rel = term.gen.chase_pos[:, :2] - data.root_link_pos_w[:, :2]
    direction = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    robot_along = (data.root_link_lin_vel_w[:, :2] * direction).sum(-1)
    target_along = (term.gen.chase_vel[:, :2] * direction).sum(-1)
    velocity_error += (robot_along - target_along).abs().cpu()
    action_sq += action.square().mean(-1).cpu()
    if action_clip is not None:
      saturated += ((action < action_clip[0]) | (action > action_clip[1])).float().mean(-1).cpu()
    q = data.joint_pos
    limits = consts.soft_limits
    violations += ((q < limits[..., 0]) | (q > limits[..., 1])).float().mean(-1).cpu()
    survived += 1
    obs = next_obs["actor"]
    if bool(terminated.any() or truncated.any()):
      fell = bool(terminated.any()); break
  displacement = robot.data.root_link_pos_w[:, :2] - start
  forward = displacement[:, 0] * heading0.cos() + displacement[:, 1] * heading0.sin()
  action_trace = torch.stack(actions, 1)
  return {
    "physical_return": float(returns[0]), "forward_progress_m": float(forward[0]),
    "velocity_mae": float(velocity_error[0] / survived), "fell": fell,
    "survival_steps": survived, "action_rms": float(action_trace.square().mean().sqrt()),
    "action_saturation_fraction": float(saturated[0] / survived),
    "joint_limit_violation_fraction": float(violations[0] / survived),
    "finite": bool(torch.isfinite(action_trace).all() and torch.isfinite(returns).all()),
  }


def physical_validation(candidates, horizon):
  rows = []
  for scenario, task, seed in VALIDATION:
    torch.manual_seed(seed)
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = 1; cfg.seed = seed; cfg.auto_reset = False
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      action_clip = cfg.actions["joint_pos"].clip
      obs, _ = env.reset(); zero = torch.zeros((1, env.action_manager.total_action_dim))
      for _ in range(12): obs, *_ = env.step(zero)
      initial_obs = obs["actor"].clone(); snapshot = capture_snapshot(env)
      baseline = rollout_physics(env, snapshot, initial_obs, candidates["zero"], horizon, action_clip)
      repeat = rollout_physics(env, snapshot, initial_obs, candidates["zero"], horizon, action_clip)
      if baseline != repeat:
        raise RuntimeError(f"{scenario}: zero-update replay is not exactly repeatable")
      for candidate_id, actor in candidates.items():
        result = rollout_physics(env, snapshot, initial_obs, actor, horizon, action_clip)
        rows.append({
          "scenario": scenario, "seed": seed, "candidate_id": candidate_id, **result,
          "physical_return_delta": result["physical_return"] - baseline["physical_return"],
          "forward_progress_delta_m": result["forward_progress_m"] - baseline["forward_progress_m"],
          "velocity_mae_delta": result["velocity_mae"] - baseline["velocity_mae"],
        })
    finally:
      env.close()
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", action="append", required=True, help="METHOD=PATH")
  parser.add_argument("--seed-bundle", type=Path, action="append", required=True)
  parser.add_argument("--physical-horizon", type=int, default=32)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  started = time.perf_counter()
  paths = dict(value.split("=", 1) for value in args.checkpoint)
  checkpoints = {name: load_frozen_checkpoint(name, Path(path)) for name, path in paths.items()}
  if set(checkpoints) != set(EXPECTED_SHA256): raise ValueError("all checkpoints required")
  seeds = [load_seed(path) for path in args.seed_bundle]
  candidates, manifest, predictions = make_candidates(checkpoints, seeds, args.output)
  physical = physical_validation(candidates, args.physical_horizon)
  pred = {row["candidate_id"]: row for row in predictions}
  raw = []
  for row in physical:
    linked = pred.get(row["candidate_id"])
    raw.append({**row, "method": None if linked is None else linked["method"],
                "update_type": "zero" if linked is None else linked["update_type"],
                "predicted_delta": 0.0 if linked is None else linked["predicted_delta"]})
  fields = list(raw[0])
  with (args.output / "physical_policy_update_raw.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(raw)
  for item in manifest: item["source_sha"] = git_sha()
  frozen = {"schema": 1, "source_sha": git_sha(), "base_checkpoint_sha256": EXPECTED_SHA256,
            "generation_seeds": [bundle["seed"] for bundle, _, _ in seeds],
            "validation_seeds": [row[2] for row in VALIDATION], "candidates": manifest}
  (args.output / "candidate_update_manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
  nonzero = [row for row in raw if row["candidate_id"] != "zero"]
  summary = {
    "schema": 1, "status": "PASS", "source_sha": git_sha(),
    "physical_horizon": args.physical_horizon, "generation_state_count": len(seeds),
    "independent_validation_state_count": len(VALIDATION), "candidate_count": len(candidates),
    "runtime_seconds": time.perf_counter() - started,
    "sign_agreement_fraction": sum((r["predicted_delta"] > 0) == (r["physical_return_delta"] > 0)
                                   for r in nonzero) / len(nonzero),
    "physical_improvement_fraction": sum(r["physical_return_delta"] > 0 for r in nonzero) / len(nonzero),
    "predictions": predictions, "physical_rows": raw,
    "evidence_class": "two generation states and two independent physical validation states; exploratory pilot",
  }
  (args.output / "physical_policy_update_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  fig, ax = plt.subplots(figsize=(6, 5))
  for scenario in {r["scenario"] for r in nonzero}:
    selected = [r for r in nonzero if r["scenario"] == scenario]
    ax.scatter([r["predicted_delta"] for r in selected],
               [r["physical_return_delta"] for r in selected], label=scenario)
  ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7)
  ax.set(xlabel="Predicted imagined-return delta", ylabel="Independent physical-return delta")
  ax.legend(); fig.tight_layout(); fig.savefig(args.output / "physical_policy_update_comparison.png", dpi=160)
  print(json.dumps({k: v for k, v in summary.items() if k not in ("predictions", "physical_rows")}, indent=2))


if __name__ == "__main__":
  main()
