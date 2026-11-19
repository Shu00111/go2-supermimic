#!/usr/bin/env python3
"""Fixed-action real-model gradients for loading-memory path interventions."""

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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.contact_memory import detach_loading_memory, safe_cosine
from go2_supermimic.diagnostics.frozen_artifacts import load_frozen_checkpoint
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory, world_model_window
from go2_supermimic.supermimic import state as st


HORIZONS = (1, 2, 4, 8)
VARIANTS = (
  ("full", None), ("backward_only_detached", "always"), ("truncated_k1", 1),
  ("truncated_k2", 2), ("truncated_k4", 4), ("truncated_k8", 8),
)


def state_scale(dataset, dtype):
  ids = dataset.environment_ids("train")
  return dataset.data["state"][ids].reshape(-1, st.STATE_DIM).to(dtype).std(
    0, unbiased=False
  ).clamp_min(1e-3).unsqueeze(0)


def gradient(model, window, scale, variant):
  state = window["state"][:, 0]
  previous = window["previous_action"][:, 0]
  contact = model.initial_contact(state, previous)[0]
  actions = window["action"].detach().clone().requires_grad_(True)
  states, contacts = [], []
  for step in range(actions.shape[1]):
    if variant == "always" or (isinstance(variant, int) and step > 0 and step % variant == 0):
      contact = detach_loading_memory(contact)
    state, _, contact = model.step(
      state, contact, actions[:, step], previous, window["context"][:, step],
      window["relief"][:, step],
    )
    previous = actions[:, step]
    states.append(state); contacts.append(contact)
  trajectory = torch.stack(states, 1)
  contact_trajectory = torch.stack(contacts, 1)
  target = window["next_state"][:, -1]
  objective = (((state - target) / scale) ** 2).mean()
  action_gradient = torch.autograd.grad(objective, actions)[0]
  return trajectory.detach(), contact_trajectory.detach(), float(objective.detach()), action_gradient.detach()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--full-checkpoint", type=Path, required=True)
  parser.add_argument("--nomemory-checkpoint", type=Path, required=True)
  parser.add_argument("--dataset", action="append", required=True,
                      help="CONDITION=PATH=SHA256")
  parser.add_argument("--environment", type=int, default=6)
  parser.add_argument("--start", type=int, default=64)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--skip-checkpoint-rehash", action="store_true")
  args = parser.parse_args()
  torch.set_num_threads(2); torch.set_num_interop_threads(1); dtype = torch.float64
  full = load_frozen_checkpoint("m2_v2", args.full_checkpoint,
                                verify_hash=not args.skip_checkpoint_rehash)
  nomemory = load_frozen_checkpoint("m2_v2_nomemory", args.nomemory_checkpoint,
                                    verify_hash=not args.skip_checkpoint_rehash)
  full_model, nomemory_model = full.world_model.to(dtype), nomemory.world_model.to(dtype)
  full.raw = {}; nomemory.raw = {}
  datasets = {}
  for item in args.dataset:
    condition, path, digest = item.split("=", 2)
    datasets[condition] = load_frozen_trajectory(Path(path), expected_sha256=digest)
  rows, started = [], time.perf_counter()
  for condition, dataset in datasets.items():
    scale = state_scale(dataset, dtype)
    all_window = {key: value.to(dtype) if value.is_floating_point() else value
                  for key, value in world_model_window(
                    dataset, args.environment, args.start, max(HORIZONS)
                  ).items()}
    for horizon in HORIZONS:
      window = {key: value[:, :horizon] for key, value in all_window.items()}
      _, _, nomemory_objective, nomemory_gradient = gradient(
        nomemory_model, window, scale, None
      )
      reference_trajectory = reference_contact = reference_gradient = None
      for label, semantics in VARIANTS:
        trajectory, contacts, objective, action_gradient = gradient(
          full_model, window, scale, semantics
        )
        if reference_trajectory is None:
          reference_trajectory, reference_contact = trajectory, contacts
          reference_gradient = action_gradient
        state_error = float((trajectory - reference_trajectory).abs().max())
        contact_error = float((contacts - reference_contact).abs().max())
        if state_error != 0.0 or contact_error != 0.0:
          raise RuntimeError("backward-only intervention changed forward values")
        difference = action_gradient - reference_gradient
        rows.append({
          "condition": condition, "partition": "development",
          "environment": args.environment, "start": args.start, "horizon": horizon,
          "variant": label, "objective": objective,
          "nomemory_objective": nomemory_objective,
          "all_action_gradient_l2": float(action_gradient.norm()),
          "initial_action_gradient_l2": float(action_gradient[:, 0].norm()),
          "cosine_vs_full": safe_cosine(action_gradient.flatten(), reference_gradient.flatten()),
          "cosine_vs_nomemory": safe_cosine(action_gradient.flatten(), nomemory_gradient.flatten()),
          "relative_difference_vs_full": float(
            difference.norm() / reference_gradient.norm().clamp_min(1e-15)
          ),
          "near_zero_coordinate_fraction": float((action_gradient.abs() < 1e-12).float().mean()),
          "nonfinite_count": int((~torch.isfinite(action_gradient)).sum()),
          "forward_state_max_abs_error": state_error,
          "forward_contact_max_abs_error": contact_error,
          "scalar_objective": "terminal normalized physical-state prediction MSE",
          "checkpoint_sha256": full.sha256,
          "nomemory_checkpoint_sha256": nomemory.sha256,
          "dataset_sha256": dataset.sha256,
        })
  args.output.mkdir(parents=True, exist_ok=True)
  csv_path = args.output / "real_memory_action_gradients.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  summary = {
    "schema": 1, "status": "PASS", "rows": len(rows), "horizons": list(HORIZONS),
    "conditions": list(datasets), "samples_per_condition": 1, "partition": "development",
    "dtype": "float64", "device": "cpu", "runtime_seconds": time.perf_counter() - started,
    "forward_equivalence": "bitwise exact for predicted physical and contact trajectories",
    "evidence_class": "frozen-checkpoint local fixed-action diagnostic; not Actor BPTT",
    "rows_data": rows,
  }
  (args.output / "real_memory_action_gradient_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps({key: value for key, value in summary.items() if key != "rows_data"}, indent=2))


if __name__ == "__main__":
  main()
