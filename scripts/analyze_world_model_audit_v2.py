#!/usr/bin/env python3
"""Matched-data P1 error attribution for frozen M1 and M2-v1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  CONTACT_DIM,
  NUM_FEET,
  ContactStructuredWorldModel,
  normalize_contact,
)
from go2_supermimic.supermimic.world_model import WorldModel


HORIZONS = (1, 2, 4, 8, 16, 32)
SPLIT_REMAINDERS = {"train": range(0, 6), "dev": range(6, 8), "held_out": range(8, 10)}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def load_model(path: Path, device: str):
  checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  method = checkpoint["cfg"]["method"]
  hidden = tuple(checkpoint["cfg"]["wm_hidden"])
  if method in {"m2", "m2_nomemory"}:
    model = ContactStructuredWorldModel(hidden, use_loading_memory=method == "m2")
  else:
    model = WorldModel(hidden)
  model.load_state_dict(checkpoint["world_model"])
  model.to(device).eval()
  del checkpoint
  return model, method


def env_selection(num_envs: int, split: str) -> torch.Tensor:
  allowed = set(SPLIT_REMAINDERS[split])
  return torch.tensor([i for i in range(num_envs) if i % 10 in allowed], dtype=torch.long)


def finite_stats(values: torch.Tensor) -> dict[str, float | int | None]:
  x = values.detach().double().reshape(-1)
  x = x[torch.isfinite(x)]
  if x.numel() == 0:
    return {"n": 0, "mean": None, "std": None, "rmse": None,
            "p50": None, "p95": None, "p99": None}
  return {
    "n": int(x.numel()),
    "mean": float(x.mean()),
    "std": float(x.std(unbiased=False)),
    "rmse": float(x.square().mean().sqrt()),
    "p50": float(x.quantile(0.50)),
    "p95": float(x.quantile(0.95)),
    "p99": float(x.quantile(0.99)),
  }


def append_row(rows: list[dict], *, condition: str, split: str, model: str,
               family: str, metric: str, stratum: str, values: torch.Tensor) -> None:
  rows.append({
    "condition": condition,
    "split": split,
    "model": model,
    "family": family,
    "metric": metric,
    "stratum": stratum,
    **finite_stats(values),
  })


def body_with_contact(model: ContactStructuredWorldModel, s, z_next, a, a_prev, xi, relief):
  base = model._base_inputs(s, a, a_prev, xi, relief)
  y = model.out_norm.unnormalize(model.body(torch.cat([
    model.in_norm(base), normalize_contact(z_next).reshape(s.shape[0], -1)
  ], -1)))
  return s + y[:, :st.STATE_DIM], y[:, st.STATE_DIM:]


def common_scales(datasets: dict[str, dict]) -> tuple[torch.Tensor, torch.Tensor]:
  targets = []
  for loaded in datasets.values():
    data = loaded["data"]
    ids = env_selection(data["state"].shape[0], "train")
    valid = ~data["done"][ids]
    delta = torch.cat([
      data["next_state"][ids] - data["state"][ids], data["pose_delta"][ids]
    ], -1)
    targets.append(delta[valid])
  scale = torch.cat(targets).float().std(0, unbiased=False).clamp_min(1.0e-3)
  return scale[:st.STATE_DIM], scale[st.STATE_DIM:]


def contact_rows(rows: list[dict], condition: str, split: str, label: str,
                 pred: torch.Tensor, target: torch.Tensor, current: torch.Tensor,
                 boundary: torch.Tensor, soft: torch.Tensor,
                 strength: torch.Tensor) -> None:
  c0 = current[..., 7] >= 0.5
  c1 = target[..., 7] >= 0.5
  phases = {
    "all": torch.ones_like(c1),
    "onset": ~c0 & c1,
    "stance": c0 & c1,
    "release": c0 & ~c1,
    "flight": ~c0 & ~c1,
  }
  if strength.ndim == soft.ndim - 1:
    strength = strength[..., None].expand_as(soft)
  elif strength.shape != soft.shape:
    raise ValueError(
      f"soil strength shape {tuple(strength.shape)} is incompatible with {tuple(soft.shape)}"
    )
  strata = dict(phases)
  strata.update({
    "near_boundary": boundary.abs() < 0.08,
    "far_boundary": boundary.abs() >= 0.08,
    "soft_material": soft,
    "rigid_material": ~soft,
    "soil_strength_low": soft & (strength < 0.8),
    "soil_strength_high": soft & (strength >= 0.8),
  })
  force_target = target[..., :3]
  force_error = pred[..., :3] - force_target
  target_norm = force_target.norm(dim=-1)
  nonzero = target_norm[target_norm > 0]
  spike_threshold = (nonzero.quantile(0.95) if nonzero.numel() else target_norm.new_tensor(math.inf))
  strata["force_tail_p95"] = target_norm >= spike_threshold
  for name, mask in strata.items():
    if not bool(mask.any()):
      continue
    append_row(rows, condition=condition, split=split, model=label,
               family="contact_force", metric="normal_signed_error_n", stratum=name,
               values=force_error[..., 2][mask])
    append_row(rows, condition=condition, split=split, model=label,
               family="contact_force", metric="normal_absolute_error_n", stratum=name,
               values=force_error[..., 2].abs()[mask])
    append_row(rows, condition=condition, split=split, model=label,
               family="contact_force", metric="tangential_vector_error_n", stratum=name,
               values=force_error[..., :2].norm(dim=-1)[mask])
    append_row(rows, condition=condition, split=split, model=label,
               family="loading_memory", metric="absolute_error_m", stratum=name,
               values=(pred[..., 4] - target[..., 4]).abs()[mask])
    append_row(rows, condition=condition, split=split, model=label,
               family="contact_classification", metric="incorrect", stratum=name,
               values=((pred[..., 7] >= 0.5) != c1)[mask].float())


def normalized_error(pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale):
  return torch.cat([
    (pred_s - target_s) / state_scale,
    (pred_pose - target_pose) / pose_scale,
  ], -1).square().mean(-1)


def component_error(pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale,
                    component: str):
  if component == "base_pose_velocity":
    slices = (st.Z, st.GRAV, st.V_YAW, st.W_B)
    parts = [(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl] for sl in slices]
    parts.append((pred_pose - target_pose) / pose_scale)
  elif component == "joints":
    parts = [(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
             for sl in (st.Q, st.QD)]
  elif component == "limbs":
    parts = [(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
             for sl in (st.HIPS, st.KNEES, st.FEET)]
  elif component == "soil_state":
    parts = [(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
             for sl in (st.SOIL_MEMORY, st.SOIL_DISTANCE)]
  else:
    raise ValueError(f"unknown state component {component}")
  return torch.cat(parts, -1).square().mean(-1)


@torch.inference_mode()
def analyze_one_step(rows: list[dict], condition: str, split: str, data: dict,
                     models: dict[str, torch.nn.Module], state_scale: torch.Tensor,
                     pose_scale: torch.Tensor, device: str, batch_size: int) -> None:
  ids = env_selection(data["state"].shape[0], split)
  valid = ~data["done"][ids]
  flat = {name: value[ids][valid] for name, value in data.items()
          if name not in {"done", "episode"}}
  output: dict[str, list[torch.Tensor]] = {}
  components = ("base_pose_velocity", "joints", "limbs", "soil_state")

  def keep(name: str, value: torch.Tensor) -> None:
    output.setdefault(name, []).append(value.detach().cpu())
  for lo in range(0, flat["state"].shape[0], batch_size):
    batch = {name: value[lo:lo + batch_size].to(device) for name, value in flat.items()}
    s, a, ap = batch["state"], batch["action"], batch["previous_action"]
    xi, relief = batch["context"], batch["relief"]
    target_s, target_pose = batch["next_state"], batch["pose_delta"]
    m1_s, m1_pose = models["M1"](s, a, ap, xi, relief)
    m2 = models["M2-v1"]
    z0_est, _ = m2.initial_contact(s, ap)
    m2_s, m2_pose, z1_est = m2.step(s, z0_est, a, ap, xi, relief)
    oracle_s, oracle_pose = body_with_contact(
      m2, s, batch["next_contact"], a, ap, xi, relief
    )
    oc_s, oc_pose, z1_oc = m2.step(s, batch["contact"], a, ap, xi, relief)
    paths = {
      "m1": (m1_s, m1_pose),
      "m2": (m2_s, m2_pose),
      "body_oracle": (oracle_s, oracle_pose),
      "body_oracle_current": (oc_s, oc_pose),
    }
    for key, (pred_s, pred_pose) in paths.items():
      keep(f"{key}_error", normalized_error(
        pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale))
      for component in components:
        keep(f"{key}_{component}", component_error(
          pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale, component))
    keep("z0_est", z0_est)
    keep("z1_est", z1_est)
    keep("z1_oracle_current", z1_oc)
  output = {name: torch.cat(values) for name, values in output.items()}
  for model, key in (("M1", "m1_error"), ("M2-v1", "m2_error"),
                     ("M2-v1/oracle-next-contact", "body_oracle_error"),
                     ("M2-v1/oracle-current-contact", "body_oracle_current_error")):
    append_row(rows, condition=condition, split=split, model=model,
               family="world_model", metric="one_step_common_normalized_mse",
               stratum="all", values=output[key])
  for model, key in (("M1", "m1"), ("M2-v1", "m2"),
                     ("M2-v1/oracle-next-contact", "body_oracle"),
                     ("M2-v1/oracle-current-contact", "body_oracle_current")):
    for component in components:
      append_row(rows, condition=condition, split=split, model=model,
                 family="world_model_component", metric="one_step_common_normalized_mse",
                 stratum=component, values=output[f"{key}_{component}"])
  # Contact-estimator and transition attributions use identical physical samples.
  contact_rows(rows, condition, split, "M2-v1/initial-estimator", output["z0_est"],
               flat["contact"], flat["contact"], flat["boundary_distance"],
               flat["soft_mask"], flat["soil_strength"])
  contact_rows(rows, condition, split, "M2-v1/predicted-current", output["z1_est"],
               flat["next_contact"], flat["contact"], flat["boundary_distance"],
               flat["soft_mask"], flat["soil_strength"])
  contact_rows(rows, condition, split, "M2-v1/oracle-current", output["z1_oracle_current"],
               flat["next_contact"], flat["contact"], flat["boundary_distance"],
               flat["soft_mask"], flat["soil_strength"])


@torch.inference_mode()
def analyze_recurrent_estimator(rows: list[dict], condition: str, split: str, data: dict,
                                model: ContactStructuredWorldModel, device: str) -> None:
  ids = env_selection(data["state"].shape[0], split)
  s = data["state"][ids].to(device)
  ap = data["previous_action"][ids].to(device)
  done = data["done"][ids].to(device)
  recurrent = model.estimator.zero_state(len(ids), s)
  predictions = []
  for t in range(s.shape[1]):
    z, recurrent = model.estimator(s[:, t], ap[:, t], recurrent)
    predictions.append(z.cpu())
    recurrent = model.estimator.reset(recurrent, done[:, t])
  pred = torch.stack(predictions, 1)
  valid = ~data["done"][ids]
  contact_rows(rows, condition, split, "M2-v1/recurrent-estimator", pred[valid],
               data["contact"][ids][valid], data["contact"][ids][valid],
               data["boundary_distance"][ids][valid], data["soft_mask"][ids][valid],
               data["soil_strength"][ids][valid])


@torch.inference_mode()
def analyze_horizons(rows: list[dict], condition: str, split: str, data: dict,
                     models: dict[str, torch.nn.Module], state_scale: torch.Tensor,
                     device: str, batch_size: int) -> None:
  ids = env_selection(data["state"].shape[0], split)
  selected = {name: value[ids] for name, value in data.items()}
  e, steps = selected["done"].shape
  for horizon in HORIZONS:
    if steps <= horizon:
      continue
    valid_window = torch.ones((e, steps - horizon), dtype=torch.bool)
    for j in range(horizon):
      valid_window &= ~selected["done"][:, j:j + steps - horizon]
    env_index, time_index = valid_window.nonzero(as_tuple=True)
    for label, model in models.items():
      errors = []
      contact_normal, contact_tangent, contact_memory, contact_incorrect = [], [], [], []
      for lo in range(0, env_index.numel(), batch_size):
        ei = env_index[lo:lo + batch_size]
        ti = time_index[lo:lo + batch_size]
        state = selected["state"][ei, ti].to(device)
        z = None
        if isinstance(model, ContactStructuredWorldModel):
          z, _ = model.initial_contact(
            state, selected["previous_action"][ei, ti].to(device)
          )
        for j in range(horizon):
          index = ti + j
          a = selected["action"][ei, index].to(device)
          ap = selected["previous_action"][ei, index].to(device)
          xi = selected["context"][ei, index].to(device)
          relief = selected["relief"][ei, index].to(device)
          if isinstance(model, ContactStructuredWorldModel):
            state, _, z = model.step(state, z, a, ap, xi, relief)
          else:
            state, _ = model(state, a, ap, xi, relief)
        target = selected["next_state"][ei, ti + horizon - 1].to(device)
        errors.append(((state - target) / state_scale).square().mean(-1).cpu())
        if isinstance(model, ContactStructuredWorldModel):
          target_contact = selected["next_contact"][ei, ti + horizon - 1].to(device)
          contact_normal.append((z[..., 2] - target_contact[..., 2]).abs().cpu())
          contact_tangent.append(
            (z[..., :2] - target_contact[..., :2]).norm(dim=-1).cpu()
          )
          contact_memory.append((z[..., 4] - target_contact[..., 4]).abs().cpu())
          contact_incorrect.append(
            ((z[..., 7] >= 0.5) != (target_contact[..., 7] >= 0.5)).float().cpu()
          )
      append_row(rows, condition=condition, split=split, model=label,
                 family="world_model", metric="autoregressive_state_mse",
                 stratum=f"horizon_{horizon}", values=torch.cat(errors))
      if contact_normal:
        for metric, values in (
          ("normal_absolute_error_n", contact_normal),
          ("tangential_vector_error_n", contact_tangent),
          ("loading_memory_absolute_error_m", contact_memory),
          ("contact_incorrect", contact_incorrect),
        ):
          append_row(rows, condition=condition, split=split, model=label,
                     family="contact_propagation", metric=metric,
                     stratum=f"horizon_{horizon}", values=torch.cat(values))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--m1-checkpoint", type=Path, required=True)
  parser.add_argument("--m2-checkpoint", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  parser.add_argument("--split", choices=("train", "dev", "held_out"), default="dev")
  parser.add_argument("--unlock-held-out", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-size", type=int, default=2048)
  args = parser.parse_args()
  if args.split == "held_out" and not args.unlock_held_out:
    raise RuntimeError("held-out P1 benchmark is locked until the M2-v2 design is frozen")
  if not (args.dataset_dir / "COLLECTION_COMPLETE").exists():
    raise RuntimeError("audit collection is incomplete")
  paths = sorted(args.dataset_dir.glob("*.pt"))
  if not paths:
    raise RuntimeError("no audit datasets found")
  datasets = {path.stem: torch.load(path, map_location="cpu", weights_only=False)
              for path in paths}
  state_scale_cpu, pose_scale_cpu = common_scales(datasets)
  state_scale = state_scale_cpu.to(args.device)
  pose_scale = pose_scale_cpu.to(args.device)
  m1, m1_method = load_model(args.m1_checkpoint, args.device)
  m2, m2_method = load_model(args.m2_checkpoint, args.device)
  if m1_method != "m1" or m2_method != "m2":
    raise RuntimeError(f"expected m1/m2 checkpoints, got {m1_method}/{m2_method}")
  models = {"M1": m1, "M2-v1": m2}
  rows: list[dict] = []
  for condition, loaded in datasets.items():
    data = loaded["data"]
    analyze_one_step(rows, condition, args.split, data, models, state_scale, pose_scale,
                     args.device, args.batch_size)
    analyze_recurrent_estimator(rows, condition, args.split, data, m2, args.device)
    analyze_horizons(rows, condition, args.split, data, models, state_scale,
                     args.device, args.batch_size)
  result = {
    "schema": 1,
    "split": args.split,
    "held_out_unlocked": bool(args.unlock_held_out),
    "datasets": {name: {**loaded["metadata"], "path": str((args.dataset_dir / f"{name}.pt").resolve()),
                         "sha256": sha256(args.dataset_dir / f"{name}.pt")}
                 for name, loaded in datasets.items()},
    "checkpoints": {
      "M1": {"path": str(args.m1_checkpoint.resolve()), "sha256": sha256(args.m1_checkpoint)},
      "M2-v1": {"path": str(args.m2_checkpoint.resolve()), "sha256": sha256(args.m2_checkpoint)},
    },
    "common_state_scale": state_scale_cpu.tolist(),
    "common_pose_scale": pose_scale_cpu.tolist(),
    "rows": rows,
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(result, indent=2) + "\n")
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  with args.output_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(json.dumps({"rows": len(rows), "json": str(args.output_json),
                    "csv": str(args.output_csv), "split": args.split}, indent=2))


if __name__ == "__main__":
  main()
