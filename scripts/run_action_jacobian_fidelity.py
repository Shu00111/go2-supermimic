#!/usr/bin/env python3
"""CPU physical finite differences versus frozen World Model action Jacobians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import pack_generator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


SCENARIOS = {
  "rigid_stable_stance": ("Go2-Chase-Flat-Scan", 82001, "rigid_rough"),
  "soft_stable_stance": ("Go2-Chase-SoftTerrain-Scan", 82011, "soft_rough"),
}
EPSILONS = (1e-2, 3e-3, 1e-3)
GROUPS = {
  "base_position_orientation": slice(0, 4),
  "base_velocity": slice(4, 10),
  "joint_positions": slice(10, 22),
  "joint_velocities": slice(22, 34),
  "foot_kinematics": slice(34, 70),
  "soil_state": slice(70, 78),
}


def git_sha() -> str:
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def state_scale(dataset_path: Path, digest: str) -> torch.Tensor:
  dataset = load_frozen_trajectory(dataset_path, expected_sha256=digest)
  ids = dataset.environment_ids("train")
  return dataset.data["state"][ids].reshape(-1, st.STATE_DIM).std(
    0, unbiased=False
  ).clamp_min(1e-3)


def sim_tensor(env, name):
  value = getattr(env.sim.data, name)
  return value if torch.is_tensor(value) else value._tensor


def signature(env, terminated, truncated):
  state = st.robot_state(env, env.command_manager.get_term("target").height_field)
  result = {
    "terminated": terminated.clone(), "truncated": truncated.clone(),
    "contact": (st.feet_world_z(state) < 0.035).clone(),
    "qpos_finite": torch.isfinite(sim_tensor(env, "qpos")).all().reshape(1),
    "qvel_finite": torch.isfinite(sim_tensor(env, "qvel")).all().reshape(1),
  }
  if "rft" in env.action_manager.active_terms:
    rft = env.action_manager.get_term("rft")
    result.update({
      "soft_mask": rft.last_soft_mask.clone(),
      "rft_contact": (rft.last_force.norm(dim=-1) > 1.0).clone(),
    })
  return result


def physical_step(env, snapshot, action):
  restore_snapshot(env, snapshot)
  _, _, terminated, truncated, _ = env.step(action)
  term = env.command_manager.get_term("target")
  return st.robot_state(env, term.height_field).clone(), signature(env, terminated, truncated)


def signature_equal(a, b):
  return all(torch.equal(value, b[name]) for name, value in a.items())


def wm_jacobian(model, seed, relief, scale):
  dtype = torch.float64
  model = model.to(dtype)
  s = seed["s"].to(dtype)
  previous = seed["a_prev"].to(dtype)
  xi = seed["xi"].to(dtype)
  relief = relief.to(dtype)
  action0 = seed["fixed_action"].to(dtype)

  def transition(action):
    if hasattr(model, "initial_contact"):
      contact = model.initial_contact(s, previous)[0]
      nxt = model.step(s, contact, action, previous, xi, relief)[0]
    else:
      nxt = model(s, action, previous, xi, relief)[0]
    return nxt / scale.to(dtype)

  columns = []
  for coordinate in range(st.NUM_JOINTS):
    tangent = torch.zeros_like(action0); tangent[:, coordinate] = 1.0
    _, jvp = torch.func.jvp(transition, (action0,), (tangent,))
    columns.append(jvp[0])
  return torch.stack(columns, 1).detach().cpu()


def cosine(a, b):
  b = b.to(dtype=a.dtype, device=a.device)
  denom = a.norm() * b.norm()
  return float((torch.dot(a.flatten(), b.flatten()) / denom.clamp_min(1e-15)).clamp(-1, 1))


def matrix_metrics(physical, learned):
  learned = learned.to(dtype=physical.dtype, device=physical.device)
  diff = learned - physical
  signal = physical.norm().clamp_min(1e-15)
  active = physical.abs() > 1e-8
  sign = float(((physical[active] * learned[active]) > 0).float().mean()) if active.any() else math.nan
  pnorm = float(torch.linalg.matrix_norm(physical, ord=2))
  lnorm = float(torch.linalg.matrix_norm(learned, ord=2))
  return {
    "cosine": cosine(physical, learned),
    "relative_error": float(diff.norm() / signal),
    "directional_sign_agreement": sign,
    "physical_spectral_norm": pnorm,
    "learned_spectral_norm": lnorm,
    "spectral_norm_ratio": lnorm / max(pnorm, 1e-15),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", action="append", required=True, help="METHOD=PATH")
  parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH=SHA256")
  parser.add_argument("--seed-bundle", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  started = time.perf_counter()
  checkpoint_paths = dict(value.split("=", 1) for value in args.checkpoint)
  datasets = {}
  for value in args.dataset:
    name, path, digest = value.split("=", 2)
    datasets[name] = (Path(path), digest)
  bundles = {torch.load(path, map_location="cpu", weights_only=False)["scenario"]: path
             for path in args.seed_bundle}
  checkpoints = {name: load_frozen_checkpoint(name, Path(path))
                 for name, path in checkpoint_paths.items()}
  if set(checkpoints) != set(EXPECTED_SHA256):
    raise ValueError("all three frozen checkpoints are required")

  raw_rows, summaries, validity = [], [], []
  for scenario, (task, seed_value, dataset_name) in SCENARIOS.items():
    bundle = torch.load(bundles[scenario], map_location="cpu", weights_only=False)
    scale = state_scale(*datasets[dataset_name])
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = 1; cfg.seed = seed_value; cfg.auto_reset = False
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      obs, _ = env.reset()
      zero = torch.zeros((1, env.action_manager.total_action_dim))
      for _ in range(12):
        obs, *_ = env.step(zero)
      term = env.command_manager.get_term("target")
      current = {
        "s": st.robot_state(env, term.height_field).clone(),
        "pose": st.planar_pose(env).clone(), "a_prev": zero.clone(),
        "obs": obs["actor"].clone(), "gen": pack_generator(term.gen).clone(),
        "xi": st.ContextReader(env).read().clone(),
        "fixed_action": bundle["fixed_actions"][:, 0].clone(),
      }
      for key in ("s", "pose", "obs", "gen", "xi"):
        expected = bundle["imagination_seed"][key]
        if not torch.equal(current[key], expected):
          raise RuntimeError(f"{scenario}: reconstructed seed mismatch for {key}")
      snapshot = capture_snapshot(env)
      nominal, nominal_signature = physical_step(env, snapshot, current["fixed_action"])
      repeat, repeat_signature = physical_step(env, snapshot, current["fixed_action"])
      replay_error = float(((nominal / scale) - (repeat / scale)).abs().max())
      if replay_error != 0.0 or not signature_equal(nominal_signature, repeat_signature):
        raise RuntimeError(f"{scenario}: nominal replay is not bitwise exact")

      derivatives = {}
      action_clip = cfg.actions["joint_pos"].clip
      for epsilon in EPSILONS:
        cols = []
        for coordinate in range(st.NUM_JOINTS):
          plus = current["fixed_action"].clone(); plus[:, coordinate] += epsilon
          minus = current["fixed_action"].clone(); minus[:, coordinate] -= epsilon
          # The authentic JointPositionActionCfg has clip=None.  Actor outputs
          # are normalized joint offsets but are not bounded to [-1, 1].
          saturated = False if action_clip is None else bool(
            (plus < action_clip[0]).any() or (plus > action_clip[1]).any()
            or (minus < action_clip[0]).any() or (minus > action_clip[1]).any()
          )
          plus_state, plus_signature = physical_step(env, snapshot, plus)
          minus_state, minus_signature = physical_step(env, snapshot, minus)
          derivative = ((plus_state - minus_state) / (2 * epsilon) / scale).squeeze(0)
          cols.append(derivative)
          branch_consistent = (signature_equal(plus_signature, minus_signature)
                               and signature_equal(plus_signature, nominal_signature))
          raw_rows.append({
            "scenario": scenario, "seed": seed_value, "epsilon": epsilon,
            "action_coordinate": coordinate,
            "physical_derivative_l2": float(derivative.norm()),
            "perturbation_signal_l2": float(((plus_state - minus_state) / scale).norm()),
            "branch_consistent": branch_consistent, "action_saturated": saturated,
            "finite": bool(torch.isfinite(derivative).all()),
          })
        derivatives[epsilon] = torch.stack(cols, 1)

      medium, small = derivatives[3e-3], derivatives[1e-3]
      valid = []
      for coordinate in range(st.NUM_JOINTS):
        denom = small[:, coordinate].norm().clamp_min(1e-12)
        relative_change = float((small[:, coordinate] - medium[:, coordinate]).norm() / denom)
        selected = [r for r in raw_rows if r["scenario"] == scenario
                    and r["action_coordinate"] == coordinate]
        signal = min(r["perturbation_signal_l2"] for r in selected)
        noise_floor = max(1e-7, 20 * replay_error)
        status = "VALID"
        if any(not r["branch_consistent"] for r in selected): status = "NONSMOOTH"
        elif any(r["action_saturated"] for r in selected): status = "SATURATED"
        elif any(not r["finite"] for r in selected): status = "NONFINITE"
        elif signal <= noise_floor: status = "BELOW_NOISE"
        elif relative_change > 0.25: status = "NO_EPSILON_CONVERGENCE"
        valid.append(status == "VALID")
        validity.append({
          "scenario": scenario, "seed": seed_value, "action_coordinate": coordinate,
          "status": status, "relative_change_3e-3_to_1e-3": relative_change,
          "relative_change_limit": 0.25, "minimum_signal": signal,
          "signal_noise_floor": noise_floor,
        })
      mask = torch.tensor(valid, dtype=torch.bool)
      physical = small[:, mask]
      relief = bundle["relief_scan"]
      for method, checkpoint in checkpoints.items():
        learned_all = wm_jacobian(checkpoint.world_model, current, relief, scale)
        learned = learned_all[:, mask]
        metrics = matrix_metrics(physical, learned) if mask.any() else {
          key: math.nan for key in ("cosine", "relative_error", "directional_sign_agreement",
                                    "physical_spectral_norm", "learned_spectral_norm",
                                    "spectral_norm_ratio")
        }
        row = {
          "scenario": scenario, "seed": seed_value, "method": method, "horizon": 1,
          "valid_coordinates": int(mask.sum()), "invalid_coordinates": int((~mask).sum()),
          **metrics, "checkpoint_sha256": checkpoint.sha256,
          "dataset_sha256": datasets[dataset_name][1], "dtype_wm": "float64",
          "dtype_physics": "float32", "device": "cpu",
        }
        for group, sl in GROUPS.items():
          group_metrics = matrix_metrics(physical[sl], learned[sl]) if mask.any() else {}
          row[f"{group}_relative_error"] = group_metrics.get("relative_error", math.nan)
          row[f"{group}_cosine"] = group_metrics.get("cosine", math.nan)
        summaries.append(row)
        for coordinate in range(st.NUM_JOINTS):
          raw_rows.append({
            "scenario": scenario, "seed": seed_value, "epsilon": "WM",
            "action_coordinate": coordinate,
            "physical_derivative_l2": float(small[:, coordinate].norm()),
            "perturbation_signal_l2": math.nan,
            "branch_consistent": valid[coordinate], "action_saturated": False,
            "finite": bool(torch.isfinite(learned_all[:, coordinate]).all()),
            "method": method,
            "learned_derivative_l2": float(learned_all[:, coordinate].norm()),
            "column_cosine": (cosine(small[:, coordinate], learned_all[:, coordinate])
                              if valid[coordinate] else math.nan),
            "column_relative_error": (float((learned_all[:, coordinate] - small[:, coordinate]).norm()
                                              / small[:, coordinate].norm().clamp_min(1e-15))
                                      if valid[coordinate] else math.nan),
          })
    finally:
      env.close()

  fields = sorted({key for row in raw_rows for key in row})
  with (args.output / "action_jacobian_raw.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(raw_rows)
  manifest = {
    "schema": 1, "status": "PASS", "source_sha": git_sha(),
    "criterion": "VALID requires exact replay, branch consistency, finite unsaturated action, signal above floor, and <=25% vector derivative change from epsilon 3e-3 to 1e-3",
    "epsilons": list(EPSILONS), "coordinates": validity,
  }
  (args.output / "action_jacobian_validity.json").write_text(json.dumps(manifest, indent=2) + "\n")
  summary = {
    "schema": 1, "status": "PASS", "source_sha": git_sha(),
    "checkpoint_sha256": EXPECTED_SHA256, "scenarios": list(SCENARIOS),
    "runtime_seconds": time.perf_counter() - started, "rows": summaries,
  }
  (args.output / "action_jacobian_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  labels = [f"{r['scenario'].split('_')[0]}:{r['method']}" for r in summaries]
  axes[0].barh(labels, [r["cosine"] for r in summaries]); axes[0].set_xlabel("Jacobian cosine")
  axes[1].barh(labels, [r["relative_error"] for r in summaries]); axes[1].set_xlabel("Relative error")
  fig.tight_layout(); fig.savefig(args.output / "action_jacobian_comparison.png", dpi=160)
  print(json.dumps({"status": "PASS", "runtime_seconds": summary["runtime_seconds"],
                    "rows": summaries}, indent=2))


if __name__ == "__main__":
  main()
