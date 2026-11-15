#!/usr/bin/env python3
"""One-time held-out prediction gate for frozen P1 world-model candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  ContactStructuredWorldModel,
  ContactStructuredWorldModelV2,
  count_parameters,
)
from go2_supermimic.supermimic.world_model import WorldModel


METHODS = ("m1", "m2", "m2_v2_core", "m2_v2")
HORIZONS = (1, 2, 4, 8, 16, 32)
DATA_KEYS = (
  "state", "next_state", "pose_delta", "action", "previous_action", "context",
  "relief", "contact", "next_contact", "done",
)


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


def frozen_model_unchanged_since(commit: str) -> bool:
  result = subprocess.run(
    [
      "git", "diff", "--quiet", commit, "--",
      "go2_supermimic/supermimic/contact_model.py",
      "go2_supermimic/supermimic/world_model.py",
    ],
    cwd=REPO_ROOT,
  )
  return result.returncode == 0


def heldout_ids(num_envs: int) -> torch.Tensor:
  """Select complete held-out trajectories; never split adjacent transitions."""
  return torch.tensor([i for i in range(num_envs) if i % 10 in {8, 9}])


def load_heldout(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
  loaded = torch.load(path, map_location="cpu", weights_only=False)
  data = loaded["data"]
  ids = heldout_ids(data["state"].shape[0])
  if ids.numel() == 0:
    raise RuntimeError(f"no held-out environments in {path}")
  selected = {key: data[key][ids] for key in DATA_KEYS}
  return selected, loaded["metadata"]


def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
  return {key: torch.cat([part[key] for part in parts]) for key in DATA_KEYS}


def valid_starts(done: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
  valid = ~done.unfold(1, horizon, 1).any(-1)
  return valid.nonzero(as_tuple=True)


def instantiate(method: str, hidden: tuple[int, ...], device: str):
  if method == "m1":
    model = WorldModel(hidden)
  elif method == "m2":
    model = ContactStructuredWorldModel(hidden)
  elif method == "m2_v2_core":
    model = ContactStructuredWorldModelV2(hidden, anchor_weight=0.0)
  elif method == "m2_v2":
    model = ContactStructuredWorldModelV2(hidden)
  else:
    raise ValueError(method)
  return model.to(device).eval()


def load_candidate(path: Path, expected_method: str, device: str):
  checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  cfg = checkpoint["cfg"]
  if cfg["method"] != expected_method:
    raise RuntimeError(f"{path}: method {cfg['method']} != {expected_method}")
  model = instantiate(expected_method, tuple(cfg["hidden"]), device)
  model.load_state_dict(checkpoint["world_model"])
  state_scale = checkpoint["common_state_scale"].float()
  pose_scale = checkpoint["common_pose_scale"].float()
  return model, cfg, state_scale, pose_scale


def component_error(pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale):
  def mse(parts):
    return torch.cat(parts, -1).square().mean(-1)

  return {
    "base_pose_velocity": mse([
      *[(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
        for sl in (st.Z, st.GRAV, st.V_YAW, st.W_B)],
      (pred_pose - target_pose) / pose_scale,
    ]),
    "joints": mse([(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
                    for sl in (st.Q, st.QD)]),
    "limbs": mse([(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
                   for sl in (st.HIPS, st.KNEES, st.FEET)]),
    "soil_state": mse([(pred_s[:, sl] - target_s[:, sl]) / state_scale[sl]
                        for sl in (st.SOIL_MEMORY, st.SOIL_DISTANCE)]),
  }


@torch.inference_mode()
def evaluate(model, data: dict[str, torch.Tensor], state_scale: torch.Tensor,
             pose_scale: torch.Tensor, device: str, batch_size: int) -> dict[str, float | int]:
  state_scale, pose_scale = state_scale.to(device), pose_scale.to(device)
  valid = (~data["done"]).nonzero(as_tuple=True)
  sums = {"one_step_common_normalized_mse": 0.0}
  component_sums = {name: 0.0 for name in
                    ("base_pose_velocity", "joints", "limbs", "soil_state")}
  contact = {
    "normal": 0.0, "tangent": 0.0, "intrusion": 0.0, "memory": 0.0,
    "wrong": 0.0, "n": 0,
  }
  count = 0
  started = time.monotonic()
  for lo in range(0, valid[0].numel(), batch_size):
    e, t = valid[0][lo:lo + batch_size], valid[1][lo:lo + batch_size]
    s = data["state"][e, t].to(device)
    a = data["action"][e, t].to(device)
    ap = data["previous_action"][e, t].to(device)
    xi = data["context"][e, t].to(device)
    relief = data["relief"][e, t].float().to(device)
    if isinstance(model, ContactStructuredWorldModel):
      z, _ = model.initial_contact(s, ap)
      pred_s, pred_pose, z = model.step(s, z, a, ap, xi, relief)
      target_z = data["next_contact"][e, t].to(device)
      contact["normal"] += float((z[..., 2] - target_z[..., 2]).abs().sum())
      contact["tangent"] += float((z[..., :2] - target_z[..., :2]).norm(dim=-1).sum())
      contact["intrusion"] += float((z[..., 3] - target_z[..., 3]).abs().sum())
      contact["memory"] += float((z[..., 4] - target_z[..., 4]).abs().sum())
      contact["wrong"] += float(
        ((z[..., 7] >= 0.5) != (target_z[..., 7] >= 0.5)).sum()
      )
      contact["n"] += int(z[..., 7].numel())
    else:
      pred_s, pred_pose = model(s, a, ap, xi, relief)
    target_s = data["next_state"][e, t].to(device)
    target_pose = data["pose_delta"][e, t].to(device)
    error = torch.cat([
      (pred_s - target_s) / state_scale, (pred_pose - target_pose) / pose_scale,
    ], -1).square().mean(-1)
    sums["one_step_common_normalized_mse"] += float(error.sum())
    for name, values in component_error(
      pred_s, pred_pose, target_s, target_pose, state_scale, pose_scale
    ).items():
      component_sums[name] += float(values.sum())
    count += error.numel()

  metrics: dict[str, float | int] = {
    "valid_transitions": count,
    "one_step_common_normalized_mse": sums["one_step_common_normalized_mse"] / count,
  }
  metrics.update({f"one_step_{name}_mse": value / count
                  for name, value in component_sums.items()})
  if contact["n"]:
    metrics.update({
      "contact_samples": contact["n"],
      "contact_normal_mae_n": contact["normal"] / contact["n"],
      "contact_tangential_vector_mae_n": contact["tangent"] / contact["n"],
      "contact_intrusion_mae_m": contact["intrusion"] / contact["n"],
      "contact_memory_mae_m": contact["memory"] / contact["n"],
      "contact_accuracy": 1.0 - contact["wrong"] / contact["n"],
    })

  for horizon in HORIZONS:
    starts = valid_starts(data["done"], horizon)
    total = 0.0
    n = 0
    for lo in range(0, starts[0].numel(), batch_size):
      e, t = starts[0][lo:lo + batch_size], starts[1][lo:lo + batch_size]
      state = data["state"][e, t].to(device)
      z = None
      if isinstance(model, ContactStructuredWorldModel):
        z, _ = model.initial_contact(
          state, data["previous_action"][e, t].to(device)
        )
      for j in range(horizon):
        index = t + j
        action = data["action"][e, index].to(device)
        action_prev = data["previous_action"][e, index].to(device)
        context = data["context"][e, index].to(device)
        relief = data["relief"][e, index].float().to(device)
        if isinstance(model, ContactStructuredWorldModel):
          state, _, z = model.step(state, z, action, action_prev, context, relief)
        else:
          state, _ = model(state, action, action_prev, context, relief)
      target = data["next_state"][e, t + horizon - 1].to(device)
      value = ((state - target) / state_scale).square().mean(-1)
      total += float(value.sum())
      n += value.numel()
    metrics[f"h{horizon}_common_normalized_state_mse"] = total / n
    metrics[f"h{horizon}_valid_windows"] = n
  elapsed = time.monotonic() - started
  metrics["evaluation_wall_time_s"] = elapsed
  metrics["evaluated_transition_steps_per_s"] = count / elapsed
  return metrics


def flatten_rows(results: dict[str, dict[str, dict]]) -> list[dict]:
  rows = []
  for condition, methods in results.items():
    for method, metrics in methods.items():
      for metric, value in metrics.items():
        rows.append({
          "condition": condition, "method": method, "metric": metric, "value": value,
        })
  return rows


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--development-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-size", type=int, default=1024)
  args = parser.parse_args()

  result_path = args.output_dir / "p1_heldout_prediction_v2.json"
  marker_path = args.output_dir / "HELDOUT_OPENED.json"
  if result_path.exists() or marker_path.exists():
    raise SystemExit("held-out gate already opened; refusing a second evaluation")
  if not (args.development_dir / "DEVELOPMENT_COMPLETE").exists():
    raise RuntimeError("controlled development is incomplete")
  paths = sorted(args.dataset_dir.glob("*.pt"))
  if len(paths) != 5 or not (args.dataset_dir / "COLLECTION_COMPLETE").exists():
    raise RuntimeError("expected the frozen five-condition dataset")

  checkpoint_paths = {
    method: args.development_dir / method / "best.pt" for method in METHODS
  }
  contracts = {
    method: json.loads(
      (args.development_dir / method / "contract.json").read_text(encoding="utf-8")
    ) for method in METHODS
  }
  for method, contract in contracts.items():
    if contract["method"] != method or contract["held_out_opened"] is not False:
      raise RuntimeError(f"invalid development contract for {method}")
    if sha256(checkpoint_paths[method]) != contract["checkpoint_sha256"]:
      raise RuntimeError(f"checkpoint hash mismatch for {method}")
  if not frozen_model_unchanged_since(contracts["m2_v2"]["source_commit"]):
    raise RuntimeError("frozen M2-v2 model source changed after candidate training")

  # Complete every checkpoint/model compatibility check before the single-use
  # marker is written or any held-out tensor is loaded.
  models = {}
  cfgs = {}
  state_scale = pose_scale = None
  for method, path in checkpoint_paths.items():
    model, cfg, candidate_state_scale, candidate_pose_scale = load_candidate(
      path, method, args.device
    )
    if state_scale is None:
      state_scale, pose_scale = candidate_state_scale, candidate_pose_scale
    elif not (torch.equal(state_scale, candidate_state_scale)
              and torch.equal(pose_scale, candidate_pose_scale)):
      raise RuntimeError("candidate checkpoints do not share identical train-fitted scales")
    models[method], cfgs[method] = model, cfg

  manifest = {
    "schema": 1,
    "status": "OPENED",
    "purpose": "single-use P1 held-out prediction gate; never model selection",
    "evaluator_source_commit": git_head(),
    "split_rule": "complete environment ids with id modulo 10 in {8,9}",
    "datasets": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
    "checkpoints": {
      method: {
        "path": str(path.resolve()), "sha256": sha256(path),
        "training_source_commit": contracts[method]["source_commit"],
      } for method, path in checkpoint_paths.items()
    },
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  marker_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

  by_condition = {}
  combined_parts = []
  metadata = {}
  for path in paths:
    data, meta = load_heldout(path)
    name = meta["condition"]
    by_condition[name] = data
    combined_parts.append(data)
    metadata[name] = meta
  by_condition["all"] = concatenate(combined_parts)

  results = {}
  for condition, data in by_condition.items():
    results[condition] = {}
    for method, model in models.items():
      metrics = evaluate(
        model, data, state_scale, pose_scale, args.device, args.batch_size
      )
      metrics["parameters"] = count_parameters(model)
      results[condition][method] = metrics
      print(json.dumps({"condition": condition, "method": method, **metrics}))

  report = {
    **manifest,
    "status": "COMPLETE",
    "held_out_used_for_selection": False,
    "conditions": metadata,
    "results": results,
  }
  result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  csv_path = args.output_dir / "p1_heldout_prediction_v2.csv"
  rows = flatten_rows(results)
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=("condition", "method", "metric", "value"))
    writer.writeheader()
    writer.writerows(rows)
  marker_path.write_text(json.dumps({**manifest, "status": "COMPLETE"}, indent=2) + "\n",
                         encoding="utf-8")
  (args.output_dir / "HELDOUT_PREDICTION_COMPLETE").write_text(
    "one-time P1 held-out prediction evaluation complete\n", encoding="utf-8"
  )


if __name__ == "__main__":
  main()
