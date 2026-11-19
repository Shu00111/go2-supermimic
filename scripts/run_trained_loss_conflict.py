#!/usr/bin/env python3
"""Exact frozen M2-v2 loss decomposition and shared-parameter gradient cosines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.contact_memory import safe_cosine
from go2_supermimic.diagnostics.frozen_artifacts import load_frozen_checkpoint
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET, normalize_contact


LOSS_NAMES = (
  "state_pose_prediction", "normal_force_prediction", "tangential_force_prediction",
  "intrusion_prediction", "loading_memory_prediction", "tangential_velocity_prediction",
  "contact_classification", "weighted_physical_anchor",
)


def batch(dataset, environments, start, horizon, dtype):
  d = dataset.data
  index = torch.tensor(environments)
  sl = slice(start, start + horizon)
  result = {name: d[name][index, sl] for name in (
    "state", "next_state", "pose_delta", "action", "previous_action", "context",
    "relief", "contact", "next_contact", "done",
  )}
  return {key: value.to(dtype) if value.is_floating_point() else value for key, value in result.items()}


def exact_components(model, b):
  s0 = b["state"][:, 0]
  actions, previous_actions = b["action"], b["previous_action"]
  context = b["context"][:, 0]
  z, _ = model.initial_contact(s0, previous_actions[:, 0])
  s = s0
  valid = torch.cumprod(1.0 - b["done"].to(s0.dtype), dim=1)
  denom = valid.sum().clamp_min(1.0)
  losses = {name: s0.new_zeros(()) for name in LOSS_NAMES}
  oracle_current = model._memory_contract(b["contact"][:, 0])
  for step in range(actions.shape[1]):
    s_pred, pose_pred, z = model.step(
      s, z, actions[:, step], previous_actions[:, step], context, b["relief"][:, step]
    )
    weight = valid[:, step]
    state_error = torch.cat(
      [s_pred - b["next_state"][:, step], pose_pred - b["pose_delta"][:, step]], -1
    )
    losses["state_pose_prediction"] += (
      ((state_error / model.out_norm.std()) ** 2).mean(-1) * weight
    ).sum()
    target = model._memory_contract(b["next_contact"][:, step])
    pred_n, target_n = normalize_contact(z, model.scales), normalize_contact(target, model.scales)
    channel = F.smooth_l1_loss(pred_n[..., :7], target_n[..., :7], reduction="none")
    # Preserve the original mean over four feet and all seven physical channels.
    divisor = NUM_FEET * 7
    losses["tangential_force_prediction"] += (
      channel[..., :2].sum((-1, -2)) / divisor * weight
    ).sum()
    losses["normal_force_prediction"] += (channel[..., 2].sum(-1) / divisor * weight).sum()
    losses["intrusion_prediction"] += (channel[..., 3].sum(-1) / divisor * weight).sum()
    losses["loading_memory_prediction"] += (channel[..., 4].sum(-1) / divisor * weight).sum()
    losses["tangential_velocity_prediction"] += (
      channel[..., 5:7].sum((-1, -2)) / divisor * weight
    ).sum()
    classification = F.binary_cross_entropy(
      z[..., 7].clamp(1e-6, 1 - 1e-6), target[..., 7], reduction="none"
    ).mean(-1)
    losses["contact_classification"] += (classification * weight).sum()

    teacher_s = s0 if step == 0 else b["next_state"][:, step - 1]
    anchor_z = model._predict_contact(teacher_s, oracle_current, actions[:, step], context)
    anchor_s, anchor_pose = model._predict_body(
      teacher_s, target, actions[:, step], previous_actions[:, step], context,
      b["relief"][:, step],
    )
    anchor_error = torch.cat(
      [anchor_s - b["next_state"][:, step], anchor_pose - b["pose_delta"][:, step]], -1
    )
    auxiliary = ((anchor_error / model.out_norm.std()) ** 2).mean(-1) + model._contact_loss(
      anchor_z, target
    )
    losses["weighted_physical_anchor"] += (model.anchor_weight * auxiliary * weight).sum()
    oracle_current, s = target, s_pred
  losses = {name: value / denom for name, value in losses.items()}
  exact = model.loss(
    s0, actions, previous_actions, context, b["relief"], b["next_state"],
    b["pose_delta"], valid, contact_targets=b["next_contact"],
    contact_initial=b["contact"][:, 0],
  )
  difference = abs(float(sum(losses.values()).detach() - exact.detach()))
  if difference > 1e-10:
    raise RuntimeError(f"loss decomposition does not reproduce exact objective: {difference}")
  return losses, float(exact.detach()), difference


def gradient_maps(model, losses):
  named = dict(model.named_parameters())
  parameters = tuple(named.values())
  result = {}
  for name, loss in losses.items():
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    result[name] = {key: value.detach() for key, value in zip(named, gradients) if value is not None}
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--dataset", action="append", required=True,
                      help="CONDITION=PATH=SHA256")
  parser.add_argument("--start", type=int, default=64)
  parser.add_argument("--horizon", type=int, default=4)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--skip-checkpoint-rehash", action="store_true")
  args = parser.parse_args()
  torch.set_num_threads(2); torch.set_num_interop_threads(1); dtype = torch.float64
  frozen = load_frozen_checkpoint("m2_v2", args.checkpoint,
                                  verify_hash=not args.skip_checkpoint_rehash)
  model = frozen.world_model.to(dtype); frozen.raw = {}
  datasets = {}
  for item in args.dataset:
    condition, path, digest = item.split("=", 2)
    datasets[condition] = load_frozen_trajectory(Path(path), expected_sha256=digest)
  rows, exact_receipts, started = [], [], time.perf_counter()
  for condition, dataset in datasets.items():
    for partition, environments in (("train", [0, 1]), ("development", [6, 7])):
      b = batch(dataset, environments, args.start, args.horizon, dtype)
      losses, exact, decomposition_error = exact_components(model, b)
      gradients = gradient_maps(model, losses)
      exact_receipts.append({"condition": condition, "partition": partition,
                             "exact_loss": exact, "decomposition_error": decomposition_error,
                             "trajectories": len(environments), "transitions": len(environments) * args.horizon})
      for left in LOSS_NAMES:
        for right in LOSS_NAMES:
          shared = sorted(set(gradients[left]) & set(gradients[right]))
          ga = torch.cat([gradients[left][name].reshape(-1) for name in shared])
          gb = torch.cat([gradients[right][name].reshape(-1) for name in shared])
          norm_a, norm_b = float(ga.norm()), float(gb.norm())
          cosine = safe_cosine(ga, gb)
          rows.append({
            "condition": condition, "partition": partition, "horizon": args.horizon,
            "loss_i": left, "loss_j": right, "loss_value_i": float(losses[left].detach()),
            "loss_value_j": float(losses[right].detach()), "gradient_cosine": cosine,
            "gradient_dot_product": float(torch.dot(ga, gb)),
            "gradient_norm_i": norm_a, "gradient_norm_j": norm_b,
            "gradient_norm_ratio_i_over_j": norm_a / max(norm_b, 1e-300),
            "shared_parameter_tensors": len(shared),
            "shared_parameter_elements": sum(gradients[left][name].numel() for name in shared),
            "negative_cosine": math.isfinite(cosine) and cosine < 0,
            "nonfinite_count": int((~torch.isfinite(ga)).sum() + (~torch.isfinite(gb)).sum()),
            "checkpoint_sha256": frozen.sha256, "dataset_sha256": dataset.sha256,
          })
  args.output.mkdir(parents=True, exist_ok=True)
  csv_path = args.output / "trained_loss_gradient_cosines.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  off_diagonal = [row for row in rows if row["loss_i"] != row["loss_j"]
                  and math.isfinite(row["gradient_cosine"])]
  summary = {
    "schema": 1, "status": "PASS", "horizon": args.horizon,
    "conditions": list(datasets), "partitions": ["train", "development"],
    "trajectories_per_stratum": 2, "transitions_per_stratum": 2 * args.horizon,
    "dtype": "float64", "device": "cpu", "runtime_seconds": time.perf_counter() - started,
    "negative_cosine_frequency_off_diagonal": sum(row["negative_cosine"] for row in off_diagonal)
      / len(off_diagonal),
    "minimum_off_diagonal_cosine": min(row["gradient_cosine"] for row in off_diagonal),
    "maximum_decomposition_error": max(row["decomposition_error"] for row in exact_receipts),
    "exact_objective_receipts": exact_receipts,
    "evidence_class": "final-checkpoint local optimization landscape; not historical training gradients",
  }
  (args.output / "trained_loss_gradient_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )
  aggregate = {}
  for left in LOSS_NAMES:
    for right in LOSS_NAMES:
      values = [row["gradient_cosine"] for row in rows if row["loss_i"] == left
                and row["loss_j"] == right and math.isfinite(row["gradient_cosine"])]
      aggregate[left, right] = sum(values) / len(values)
  matrix = [[aggregate[left, right] for right in LOSS_NAMES] for left in LOSS_NAMES]
  fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
  ax.set_xticks(range(len(LOSS_NAMES)), LOSS_NAMES, rotation=45, ha="right")
  ax.set_yticks(range(len(LOSS_NAMES)), LOSS_NAMES)
  ax.set_title("Trained M2-v2 loss-gradient cosine (mean across 4 strata)")
  fig.colorbar(image, ax=ax); fig.tight_layout()
  fig.savefig(args.output / "trained_loss_gradient_heatmap.png", dpi=180)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
