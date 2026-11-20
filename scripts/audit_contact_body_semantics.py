#!/usr/bin/env python3
"""Frozen M2-v2 contact-to-body interventions on development trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import load_frozen_checkpoint
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import normalize_contact


def git_sha():
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def cosine(a, b):
  denom = a.norm() * b.norm()
  return float(((a.flatten() @ b.flatten()) / denom.clamp_min(1e-15)).clamp(-1, 1))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH=SHA256")
  parser.add_argument("--stride", type=int, default=16)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  started = time.perf_counter()
  frozen = load_frozen_checkpoint("m2_v2", args.checkpoint)
  model = frozen.world_model.eval()
  datasets = {}
  for value in args.dataset:
    name, path, digest = value.split("=", 2)
    datasets[name] = load_frozen_trajectory(Path(path), expected_sha256=digest)
  rows = []
  for condition, dataset in datasets.items():
    data = dataset.data
    train_ids = dataset.environment_ids("train")
    development_ids = dataset.environment_ids("development")
    train_contact = data["next_contact"][train_ids].reshape(-1, 4, 8)
    contact_mean = train_contact.mean((0, 1))
    contact_std = train_contact.std((0, 1), unbiased=False).clamp_min(1e-4)
    state_scale = data["state"][train_ids].reshape(-1, st.STATE_DIM).std(0, unbiased=False).clamp_min(1e-3)
    selector = (development_ids, slice(None, None, args.stride))
    s = data["state"][selector].reshape(-1, st.STATE_DIM)
    target = data["next_state"][selector].reshape(-1, st.STATE_DIM)
    a = data["action"][selector].reshape(-1, st.NUM_JOINTS)
    a_prev = data["previous_action"][selector].reshape(-1, st.NUM_JOINTS)
    xi = data["context"][selector].reshape(-1, st.CONTEXT_DIM)
    relief = data["relief"][selector].reshape(-1, data["relief"].shape[-1])
    oracle = data["next_contact"][selector].reshape(-1, 4, 8)
    with torch.no_grad():
      z0, _ = model.initial_contact(s, a_prev)
      native = model._predict_contact(s, z0, a, xi)
    memory_sigma = contact_std[4]

    def contact_for(action, variant):
      if variant == "oracle_next_label":
        return oracle
      z = model._predict_contact(s, z0, action, xi)
      if variant == "native_predicted": return z
      z = z.clone()
      if variant == "memory_zero": z[..., 4] = 0
      elif variant == "memory_plus_train_std": z[..., 4] += memory_sigma
      elif variant == "force_zero": z[..., :3] = 0
      elif variant == "probability_intrusion_zero": z[..., 3] = 0; z[..., 7] = 0
      else: raise ValueError(variant)
      return z

    variants = ("native_predicted", "oracle_next_label", "memory_zero",
                "memory_plus_train_std", "force_zero", "probability_intrusion_zero")
    jacobians = {}
    for variant in variants:
      def transition(action):
        z = contact_for(action, variant)
        return model._predict_body(s, z, action, a_prev, xi, relief)[0] / state_scale
      with torch.no_grad():
        z = contact_for(a, variant)
        prediction = model._predict_body(s, z, a, a_prev, xi, relief)[0]
      columns = []
      for coordinate in range(st.NUM_JOINTS):
        tangent = torch.zeros_like(a); tangent[:, coordinate] = 1
        _, jvp = torch.func.jvp(transition, (a,), (tangent,))
        columns.append(jvp.detach())
      jacobian = torch.stack(columns, -1)
      jacobians[variant] = jacobian
      semantic = (z - oracle) / contact_std
      label_z = (z - contact_mean) / contact_std
      rows.append({
        "condition": condition, "variant": variant, "development_samples": s.shape[0],
        "normalized_state_rmse": float(((prediction - target) / state_scale).square().mean().sqrt()),
        "contact_semantic_rmse_train_std": float(semantic.square().mean().sqrt()),
        "contact_input_z_rms_vs_train_labels": float(label_z.square().mean().sqrt()),
        "action_jacobian_frobenius": float(jacobian.norm()),
        "action_jacobian_cosine_vs_native": math.nan,
        "action_jacobian_relative_change_vs_native": math.nan,
      })
    native_j = jacobians["native_predicted"]
    for row in rows[-len(variants):]:
      current = jacobians[row["variant"]]
      row["action_jacobian_cosine_vs_native"] = cosine(native_j, current)
      row["action_jacobian_relative_change_vs_native"] = float(
        (current - native_j).norm() / native_j.norm().clamp_min(1e-15)
      )
  fields = list(rows[0])
  with (args.output / "contact_body_semantic_audit.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  receipt = {
    "schema": 1, "status": "PASS", "source_sha": git_sha(),
    "checkpoint_sha256": frozen.sha256,
    "dataset_sha256": {key: value.sha256 for key, value in datasets.items()},
    "split": "development only", "stride": args.stride, "rows": rows,
    "runtime_seconds": time.perf_counter() - started,
    "interpretation_scope": "causal input interventions on a frozen network, not physical interventions",
  }
  (args.output / "contact_body_semantic_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
  main()
