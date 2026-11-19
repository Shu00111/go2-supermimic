#!/usr/bin/env python3
"""Exact repository Imagination.rollout Actor-BPTT on validated CPU seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from go2_supermimic.diagnostics.contact_memory import detach_loading_memory, safe_cosine
from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, load_frozen_checkpoint
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination
from go2_supermimic.target.terrain import HeightField


HORIZONS = (1, 2, 4, 8)
VARIANTS = (
  ("full", None), ("backward_only_detached", "always"), ("truncated_k1", 1),
  ("truncated_k2", 2), ("truncated_k4", 4), ("truncated_k8", 8),
)


class MemoryIntervention(nn.Module):
  def __init__(self, model, semantics):
    super().__init__(); self.model = model; self.semantics = semantics; self.step_index = 0

  def initial_contact(self, s, a_prev):
    self.step_index = 0
    return self.model.initial_contact(s, a_prev)

  def step(self, s, z, a, a_prev, xi, relief):
    if self.semantics == "always" or (
      isinstance(self.semantics, int) and self.step_index > 0
      and self.step_index % self.semantics == 0
    ):
      z = detach_loading_memory(z)
    out = self.model.step(s, z, a, a_prev, xi, relief)
    self.step_index += 1
    return out


class RecordingActor(nn.Module):
  def __init__(self, actor):
    super().__init__(); self.actor = actor; self.actions = []

  def forward(self, obs):
    action = self.actor(obs)
    self.actions.append(action)
    return action


def flat_grad(grads, params):
  return torch.cat([
    (torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(grads, params)
  ])


def measure(imagination, model, actor_source, critic, seed, random_seed):
  actor = RecordingActor(copy.deepcopy(actor_source))
  params = tuple(actor.actor.parameters())
  torch.manual_seed(random_seed)
  objective, data = imagination.rollout(model, actor, critic, seed)
  scalar = objective.mean()
  first = torch.autograd.grad(scalar, actor.actions[0], retain_graph=True)[0]
  grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
  vector = flat_grad(grads, params).detach()
  per_step = []
  for step in range(data["rewards"].shape[1]):
    step_grads = torch.autograd.grad(
      data["rewards"][:, step].mean(), params, retain_graph=True, allow_unused=True
    )
    per_step.append(float(flat_grad(step_grads, params).norm()))
  return {
    "objective": float(scalar.detach()), "gradient": vector,
    "first_action_gradient_l2": float(first.norm()),
    "per_step_reward_gradient_l2": per_step,
    "nonfinite_count": int((~torch.isfinite(vector)).sum()),
    "near_zero_fraction": float((vector.abs() < 1e-12).float().mean()),
  }


def load_seed(path):
  bundle = torch.load(path, map_location="cpu", weights_only=False)
  seed = {key: value.clone() for key, value in bundle["imagination_seed"].items()}
  const = bundle["robot_consts"]
  consts = st.RobotConsts(**const)
  raw = bundle["height_field"]
  field = None if raw is None else HeightField(raw["h"], tuple(raw["origin"]), raw["res"])
  return bundle, seed, consts, field


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--m1", type=Path, required=True)
  parser.add_argument("--m2-v2", type=Path, required=True)
  parser.add_argument("--m2-v2-nomemory", type=Path, required=True)
  parser.add_argument("--seed-bundle", type=Path, action="append", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  checkpoints = {
    "m1": load_frozen_checkpoint("m1", args.m1),
    "m2_v2": load_frozen_checkpoint("m2_v2", args.m2_v2),
    "m2_v2_nomemory": load_frozen_checkpoint("m2_v2_nomemory", args.m2_v2_nomemory),
  }
  actor, critic = checkpoints["m1"].actor, checkpoints["m1"].critic
  rows, gradients, started = [], {}, time.perf_counter()
  for seed_path in args.seed_bundle:
    bundle, seed, consts, field = load_seed(seed_path)
    for horizon in HORIZONS:
      imagination = Imagination(
        ImagineCfg(horizon=horizon, compile=False, sensor_grad=True), consts,
        bundle["target_cfg"], bundle["history_length"], "cpu",
        tuple(tuple(x) for x in bundle["observation_terms"]), field,
      )
      random_seed = 20261119 + bundle["seed"] + horizon
      reference = None
      jobs = [("m1", "not_applicable", checkpoints["m1"].world_model),
              ("m2_v2_nomemory", "not_applicable", checkpoints["m2_v2_nomemory"].world_model)]
      jobs += [("m2_v2", label, MemoryIntervention(checkpoints["m2_v2"].world_model, semantics))
               for label, semantics in VARIANTS]
      for method, variant, model in jobs:
        result = measure(imagination, model, actor, critic, seed, random_seed)
        key = (bundle["scenario"], horizon, method, variant)
        gradients[key] = result["gradient"]
        if method == "m2_v2":
          if variant == "full": reference = result
          else:
            if result["objective"] != reference["objective"]:
              raise RuntimeError("backward-only intervention changed forward return")
        rows.append({
          "scenario": bundle["scenario"], "physical_seed": bundle["seed"],
          "horizon": horizon, "method": method, "variant": variant,
          "imagined_return": result["objective"],
          "actor_gradient_l2": float(result["gradient"].norm()),
          "first_action_gradient_l2": result["first_action_gradient_l2"],
          "per_step_reward_gradient_l2": json.dumps(result["per_step_reward_gradient_l2"]),
          "near_zero_coordinate_fraction": result["near_zero_fraction"],
          "nonfinite_count": result["nonfinite_count"],
          "gradient_clip_event_at_1": bool(result["gradient"].norm() > 1.0),
          "forward_return_abs_error_vs_full_m2": (
            0.0 if method == "m2_v2" and variant == "full" else
            abs(result["objective"] - reference["objective"])
            if method == "m2_v2" and reference is not None else None
          ),
          "checkpoint_sha256": EXPECTED_SHA256[method],
        })
  cosine_rows = []
  for bundle_path in args.seed_bundle:
    scenario = torch.load(bundle_path, map_location="cpu", weights_only=False)["scenario"]
    for horizon in HORIZONS:
      base = gradients[(scenario, horizon, "m1", "not_applicable")]
      full = gradients[(scenario, horizon, "m2_v2", "full")]
      nomem = gradients[(scenario, horizon, "m2_v2_nomemory", "not_applicable")]
      for a, av, ga, b, bv, gb in (
        ("m1", "not_applicable", base, "m2_v2", "full", full),
        ("m1", "not_applicable", base, "m2_v2_nomemory", "not_applicable", nomem),
        ("m2_v2", "full", full, "m2_v2_nomemory", "not_applicable", nomem),
      ):
        cosine_rows.append({"scenario": scenario, "horizon": horizon, "method_a": a,
                            "variant_a": av, "method_b": b, "variant_b": bv,
                            "gradient_cosine": safe_cosine(ga, gb)})
      for variant, _ in VARIANTS[1:]:
        g = gradients[(scenario, horizon, "m2_v2", variant)]
        cosine_rows.append({"scenario": scenario, "horizon": horizon, "method_a": "m2_v2",
                            "variant_a": "full", "method_b": "m2_v2", "variant_b": variant,
                            "gradient_cosine": safe_cosine(full, g)})
    for method, variant in (("m1", "not_applicable"), ("m2_v2", "full"),
                            ("m2_v2_nomemory", "not_applicable")):
      for h0, h1 in zip(HORIZONS[:-1], HORIZONS[1:]):
        cosine_rows.append({"scenario": scenario, "horizon": f"{h0}_vs_{h1}",
                            "method_a": method, "variant_a": variant, "method_b": method,
                            "variant_b": variant, "gradient_cosine": safe_cosine(
                              gradients[(scenario, h0, method, variant)],
                              gradients[(scenario, h1, method, variant)])})

  for name, data in (("exact_actor_bptt.csv", rows),
                     ("exact_actor_gradient_cosine.csv", cosine_rows)):
    with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
      writer.writeheader(); writer.writerows(data)
  summary = {
    "schema": 1, "status": "PASS", "device": "cpu", "dtype": "float32",
    "horizons": list(HORIZONS), "physical_seed_count": len(args.seed_bundle),
    "actor": "shared frozen M1 seed-14011 actor", "critic": "shared frozen M1 critic",
    "rollout": "go2_supermimic.supermimic.imagine.Imagination.rollout",
    "forward_equivalence": "bitwise-equal scalar imagined return for all M2 memory interventions",
    "per_step_metric": "L2 norm of Actor-parameter gradient of each raw analytic reward",
    "runtime_seconds": time.perf_counter() - started, "rows_data": rows,
    "cosines": cosine_rows,
  }
  (args.output / "exact_actor_bptt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  plot_rows = [r for r in rows if r["variant"] in ("full", "not_applicable")]
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  for scenario in sorted({r["scenario"] for r in plot_rows}):
    for method in ("m1", "m2_v2", "m2_v2_nomemory"):
      values = [r for r in plot_rows if r["scenario"] == scenario and r["method"] == method]
      axes[0].plot([r["horizon"] for r in values], [r["imagined_return"] for r in values],
                   marker="o", label=f"{scenario}:{method}")
      axes[1].plot([r["horizon"] for r in values], [r["actor_gradient_l2"] for r in values],
                   marker="o", label=f"{scenario}:{method}")
  axes[0].set(xlabel="Horizon", ylabel="Imagined return"); axes[1].set(
    xlabel="Horizon", ylabel="Actor gradient L2", yscale="log")
  axes[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(args.output / "exact_actor_bptt.png", dpi=160)
  print(json.dumps({k: v for k, v in summary.items() if k not in ("rows_data", "cosines")}, indent=2))


if __name__ == "__main__":
  main()
