#!/usr/bin/env python3
"""Run bounded loading-memory gradient interventions on real frozen bytes.

This uses matched physical states and fixed actions from a preserved P1
trajectory. It is a frozen-checkpoint local diagnostic, not a claim about
historical training gradients or the exact Actor objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import load_frozen_checkpoint
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory, world_model_window
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET


HORIZONS = (1, 2, 4, 8, 16, 32)
INTERVENTIONS = (("full", None), ("backward_only_detached", 1),
                 ("truncated_k2", 2), ("truncated_k4", 4), ("truncated_k8", 8))


def truncate_memory(z: torch.Tensor) -> torch.Tensor:
  return torch.cat([z[..., :4], z[..., 4:5].detach(), z[..., 5:]], dim=-1)


def rollout(model, window, horizon: int, truncate_every: int | None):
  state = window["state"][:, 0].detach().requires_grad_(True)
  contact = window["contact"][:, 0].detach().requires_grad_(True)
  initial_state, initial_contact = state, contact
  outputs = []
  for step in range(horizon):
    if truncate_every and step % truncate_every == 0:
      contact = truncate_memory(contact)
    state, _, contact = model.step(
      state, contact, window["action"][:, step], window["previous_action"][:, step],
      window["context"][:, step], window["relief"][:, step],
    )
    outputs.append(torch.cat([state, contact.reshape(state.shape[0], NUM_FEET * CONTACT_DIM)], -1))
  trajectory = torch.stack(outputs, 1)
  objective = trajectory.square().mean()
  state_grad, contact_grad = torch.autograd.grad(objective, (initial_state, initial_contact),
                                                  allow_unused=True)
  if contact_grad is None:
    contact_grad = torch.zeros_like(initial_contact)
  return trajectory.detach(), state_grad.detach(), contact_grad.detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
  denominator = left.norm() * right.norm()
  return float((left.flatten() @ right.flatten()) / denominator) if denominator else float("nan")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--trajectory", type=Path, required=True)
  parser.add_argument("--trajectory-sha256", required=True)
  parser.add_argument("--condition", required=True)
  parser.add_argument("--environment", type=int, default=8)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--skip-checkpoint-rehash", action="store_true")
  args = parser.parse_args()
  frozen = load_frozen_checkpoint("m2_v2", args.checkpoint,
                                  verify_hash=not args.skip_checkpoint_rehash)
  dataset = load_frozen_trajectory(args.trajectory, expected_sha256=args.trajectory_sha256)
  if dataset.metadata.get("condition") != args.condition:
    raise ValueError("trajectory condition does not match command line")
  rows = []
  for horizon in HORIZONS:
    window = world_model_window(dataset, args.environment, args.start, horizon)
    reference = None
    for label, truncation in INTERVENTIONS:
      trajectory, state_grad, contact_grad = rollout(frozen.world_model, window, horizon, truncation)
      if reference is None:
        reference = (trajectory, state_grad, contact_grad)
      max_error = float((trajectory - reference[0]).abs().max())
      if max_error != 0.0:
        raise RuntimeError(f"gradient-only intervention changed forward values: {max_error}")
      rows.append({
        "condition": args.condition, "environment": args.environment, "start": args.start,
        "horizon": horizon, "intervention": label, "forward_max_abs_error": max_error,
        "state_gradient_l2": float(state_grad.norm()),
        "contact_gradient_l2": float(contact_grad.norm()),
        "loading_memory_gradient_l2": float(contact_grad[..., 4].norm()),
        "contact_gradient_cosine_vs_full": cosine(contact_grad, reference[2]),
        "evidence_class": "frozen-checkpoint local fixed-action diagnostic",
      })
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
  receipt = {
    "schema": 1, "status": "PASS", "checkpoint_sha256": frozen.sha256,
    "trajectory_sha256": dataset.sha256, "rows": len(rows),
    "forward_equivalence": "bitwise exact", "output": str(args.output.resolve()),
    "limitations": [
      "fixed physical actions; not the exact Imagination.rollout Actor objective",
      "checkpoint-local gradients do not reconstruct historical optimizer gradients",
    ],
  }
  args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
  main()
