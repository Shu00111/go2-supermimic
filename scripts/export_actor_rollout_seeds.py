#!/usr/bin/env python3
"""Export and validate two minimal, exact Imagination.rollout seeds on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import go2_supermimic.env  # noqa: F401
from go2_supermimic.diagnostics import capture_snapshot, restore_snapshot
from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, load_frozen_checkpoint
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import pack_generator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


SCENARIOS = (
  ("rigid_stable_stance", "Go2-Chase-Flat-Scan", 82001),
  ("soft_stable_stance", "Go2-Chase-SoftTerrain-Scan", 82011),
)


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def sim_tensor(env, name):
  value = getattr(env.sim.data, name)
  return value if torch.is_tensor(value) else value._tensor


def frame(env, obs, reward, terminated, truncated):
  term = env.command_manager.get_term("target")
  out = {
    "qpos": sim_tensor(env, "qpos").clone(),
    "qvel": sim_tensor(env, "qvel").clone(),
    "state": st.robot_state(env, term.height_field).clone(),
    "pose": st.planar_pose(env).clone(),
    "observation": obs["actor"].clone(),
    "generator": pack_generator(term.gen).clone(),
    "reward": reward.clone(),
    "terminated": terminated.clone(),
    "truncated": truncated.clone(),
  }
  if "rft" in env.action_manager.active_terms:
    rft = env.action_manager.get_term("rft")
    out.update({
      "rft_plastic_depth": rft.state.plastic_depth.clone(),
      "rft_plastic_position": rft.state.plastic_position.clone(),
      "rft_plastic_surface_z": rft.state.plastic_surface_z.clone(),
      "rft_signed_distance": rft.last_signed_distance.clone(),
      "rft_soft_mask": rft.last_soft_mask.clone(),
      "rft_force": rft.last_force.clone(),
    })
  return out


def replay(env, snapshot, actions):
  restore_snapshot(env, snapshot)
  frames = []
  for action in actions:
    obs, reward, terminated, truncated, _ = env.step(action)
    frames.append(frame(env, obs, reward, terminated, truncated))
  return frames


def max_errors(first, second):
  errors = {}
  for a, b in zip(first, second, strict=True):
    for name, value in a.items():
      error = float((value.double() - b[name].double()).abs().max())
      errors[name] = max(errors.get(name, 0.0), error)
  return errors


def export_one(name, task, seed, horizon, actor, contact_models, output):
  torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  cfg.auto_reset = False
  cfg.observations["actor"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    zero = torch.zeros((1, env.action_manager.total_action_dim))
    for _ in range(12):
      obs, *_ = env.step(zero)
    snapshot = capture_snapshot(env)
    term = env.command_manager.get_term("target")
    initial_obs = obs["actor"].clone()
    s = st.robot_state(env, term.height_field).clone()
    pose = st.planar_pose(env).clone()
    a_prev = zero.clone()
    xi = st.ContextReader(env).read().clone()
    enc_bias = env.scene["robot"].data.encoder_bias.clone()
    consts = st.robot_consts(env)
    gen = pack_generator(term.gen).clone()
    height_scan = term.scan().clone()
    relief = height_scan - s[:, st.Z]

    actions, live_obs = [], initial_obs
    for _ in range(horizon):
      with torch.no_grad():
        action = actor(live_obs).clone()
      actions.append(action)
      next_obs, *_ = env.step(action)
      live_obs = next_obs["actor"]
    first = replay(env, snapshot, actions)
    second = replay(env, snapshot, actions)
    errors = max_errors(first, second)
    exact = all(value == 0.0 for value in errors.values())
    if not exact:
      raise RuntimeError(f"{name}: snapshot replay was not bitwise exact: {errors}")

    initial_contacts = {
      method: model.initial_contact(s, a_prev)[0].detach().clone()
      for method, model in contact_models.items()
    }
    rft = env.action_manager.get_term("rft") if "rft" in env.action_manager.active_terms else None
    material = {
      "has_rft": rft is not None,
      "soft_mask": None if rft is None else rft.last_soft_mask.clone(),
      "plastic_depth": None if rft is None else rft.state.plastic_depth.clone(),
      "signed_distance": None if rft is None else rft.last_signed_distance.clone(),
    }
    field = None if term.height_field is None else {
      "h": term.height_field.h.cpu(), "origin": term.height_field.origin,
      "res": term.height_field.res,
    }
    bundle = {
      "schema": 1, "scenario": name, "task": task, "seed": seed,
      "backend": "MuJoCo-Warp float32 CPU", "horizon": horizon,
      "imagination_seed": {
        "env": torch.tensor([0]), "s": s, "pose": pose, "a_prev": a_prev,
        "obs": initial_obs, "gen": gen, "xi": xi, "enc_bias": enc_bias,
      },
      "robot_consts": {
        "default_q": consts.default_q, "action_scale": consts.action_scale,
        "kp": consts.kp, "kd": consts.kd, "effort": consts.effort,
        "soft_limits": consts.soft_limits,
      },
      "history_length": cfg.observations["actor"].history_length,
      "observation_terms": st.obs_terms(cfg),
      "target_cfg": term.cfg.target,
      "height_field": field,
      "relief_scan": relief, "height_scan": height_scan,
      "initial_contact": initial_contacts,
      "material": material,
      "fixed_actions": torch.stack(actions, 1),
      "reference_frames": {k: torch.stack([x[k] for x in first], 1) for k in first[0]},
      "rng": {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
      },
      "snapshot": {
        "captured_and_restored_in_process": True,
        "portable": False,
        "reason": "snapshot is bound to the exact environment and Warp allocation",
      },
      "validation": {"bitwise_exact": exact, "max_abs_error": errors},
    }
    path = output / f"{name}_seed{seed}.pt"
    torch.save(bundle, path)
    return path, {
      "scenario": name, "task": task, "seed": seed, "horizon": horizon,
      "path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size,
      "status": "PASS", "bitwise_exact": exact, "max_abs_error": errors,
      "observation_width": initial_obs.shape[-1],
      "physical_frames": horizon, "finite_rewards": bool(torch.isfinite(bundle["reference_frames"]["reward"]).all()),
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--m1", type=Path, required=True)
  parser.add_argument("--m2-v2", type=Path, required=True)
  parser.add_argument("--m2-v2-nomemory", type=Path, required=True)
  parser.add_argument("--horizon", type=int, default=8)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  if any(args.output.iterdir()):
    raise SystemExit(f"refusing nonempty output directory {args.output}")
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  checkpoints = {
    "m1": load_frozen_checkpoint("m1", args.m1),
    "m2_v2": load_frozen_checkpoint("m2_v2", args.m2_v2),
    "m2_v2_nomemory": load_frozen_checkpoint("m2_v2_nomemory", args.m2_v2_nomemory),
  }
  actor = checkpoints["m1"].actor
  results = [export_one(name, task, seed, args.horizon, actor,
                        {k: v.world_model for k, v in checkpoints.items() if k != "m1"},
                        args.output) for name, task, seed in SCENARIOS]
  manifest = {
    "schema": 1, "status": "PASS", "criterion": "bitwise equality for every replay tensor",
    "checkpoint_sha256": EXPECTED_SHA256, "files": [row for _, row in results],
  }
  manifest_path = args.output / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
