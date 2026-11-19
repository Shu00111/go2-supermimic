#!/usr/bin/env python3
"""Bounded matrix-free Jacobian diagnostics on verified trained models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.contact_memory import (
  jvp_vjp_identity, nonfinite_count, spectral_norm_matrix_free,
)
from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, load_frozen_checkpoint
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory, world_model_window
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET, ContactStructuredWorldModelV2


HORIZONS = (1, 2, 4, 8)
CONTACT_SCALE = torch.tensor([150.0, 150.0, 150.0, 0.08, 0.08, 1.0, 1.0, 1.0])


def physical_scale(dataset, dtype):
  ids = dataset.environment_ids("train")
  values = dataset.data["state"][ids].reshape(-1, st.STATE_DIM).to(dtype)
  return values.std(0, unbiased=False).clamp_min(1e-3)


def rollout_functions(model, window, horizon, scale, init_mode):
  contact_model = isinstance(model, ContactStructuredWorldModelV2)
  s0 = window["state"][:, 0]
  a_prev0 = window["previous_action"][:, 0]
  if contact_model:
    z_oracle = window["contact"][:, 0]
    z_estimator = model.initial_contact(s0, a_prev0)[0].detach()
    z0 = z_oracle if init_mode == "oracle_label" else z_estimator
    cscale = CONTACT_SCALE.to(dtype=s0.dtype).repeat(NUM_FEET)
    x0 = torch.cat([s0 / scale, z0.reshape(1, -1) / cscale, a_prev0], -1)
  else:
    cscale = None
    x0 = torch.cat([s0 / scale, a_prev0], -1)

  def normalized_map(x):
    s = x[:, :st.STATE_DIM] * scale
    offset = st.STATE_DIM
    if contact_model:
      z = (x[:, offset:offset + NUM_FEET * CONTACT_DIM] * cscale).reshape(
        -1, NUM_FEET, CONTACT_DIM
      )
      offset += NUM_FEET * CONTACT_DIM
    previous = x[:, offset:offset + st.NUM_JOINTS]
    for step in range(horizon):
      action = window["action"][:, step]
      if contact_model:
        s, _, z = model.step(s, z, action, previous, window["context"][:, step],
                             window["relief"][:, step])
      else:
        s, _ = model(s, action, previous, window["context"][:, step],
                     window["relief"][:, step])
      previous = action
    parts = [s / scale]
    if contact_model:
      parts.append(z.reshape(1, -1) / cscale)
    parts.append(previous)
    return torch.cat(parts, -1)

  def physical_map(q):
    s = q * scale
    previous = a_prev0
    if contact_model:
      z = (model.initial_contact(s, previous)[0] if init_mode == "estimator"
           else z_oracle)
    for step in range(horizon):
      action = window["action"][:, step]
      if contact_model:
        s, _, z = model.step(s, z, action, previous, window["context"][:, step],
                             window["relief"][:, step])
      else:
        s, _ = model(s, action, previous, window["context"][:, step],
                     window["relief"][:, step])
      previous = action
    return s / scale

  memory_map = body_contact_map = None
  if contact_model:
    base_memory = z0[..., 4].detach() / CONTACT_SCALE[4].to(s0)

    def memory_map(memory):
      s, previous = s0, a_prev0
      z = torch.cat([z0[..., :4], memory[..., None] * CONTACT_SCALE[4].to(s0),
                     z0[..., 5:]], -1)
      for step in range(horizon):
        action = window["action"][:, step]
        s, _, z = model.step(s, z, action, previous, window["context"][:, step],
                             window["relief"][:, step])
        previous = action
      return s / scale

    def body_contact_map(q):
      s, previous = q * scale, a_prev0
      z = (model.initial_contact(s, previous)[0] if init_mode == "estimator" else z_oracle)
      for step in range(horizon):
        action = window["action"][:, step]
        s, _, z = model.step(s, z, action, previous, window["context"][:, step],
                             window["relief"][:, step])
        previous = action
      return z.reshape(1, -1) / cscale
  return x0, normalized_map, s0 / scale, physical_map, base_memory if contact_model else None, memory_map, body_contact_map


def measure(fn, x, iterations, seed):
  generator = torch.Generator(device=x.device).manual_seed(seed)
  v = torch.randn(x.shape, dtype=x.dtype, generator=generator)
  w = torch.randn(fn(x).shape, dtype=x.dtype, generator=generator)
  check = jvp_vjp_identity(fn, x, v, w)
  gain = spectral_norm_matrix_free(fn, x, iterations=iterations, seed=seed)
  return gain, check.relative_error


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", action="append", required=True,
                      help="METHOD=PATH")
  parser.add_argument("--dataset", action="append", required=True,
                      help="CONDITION=PATH=SHA256")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--environment", type=int, default=6)
  parser.add_argument("--start", type=int, default=64)
  parser.add_argument("--power-iterations", type=int, default=5)
  parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
  parser.add_argument("--skip-checkpoint-rehash", action="store_true")
  args = parser.parse_args()
  torch.set_num_threads(2); torch.set_num_interop_threads(1)
  dtype = getattr(torch, args.dtype)
  checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
  datasets = {}
  for item in args.dataset:
    condition, path, digest = item.split("=", 2)
    datasets[condition] = load_frozen_trajectory(Path(path), expected_sha256=digest)
  if set(checkpoints) != set(EXPECTED_SHA256):
    raise ValueError("all three authoritative checkpoints are required")
  started = time.perf_counter(); rows = []
  for method in ("m1", "m2_v2", "m2_v2_nomemory"):
    frozen = load_frozen_checkpoint(method, Path(checkpoints[method]),
                                    verify_hash=not args.skip_checkpoint_rehash)
    model = frozen.world_model.to(dtype=dtype)
    # Drop mmap references to replay/optimizer tensors after parameters were copied.
    frozen.raw = {}
    init_modes = ("not_applicable",) if method == "m1" else ("oracle_label", "estimator")
    for condition, dataset in datasets.items():
      scale = physical_scale(dataset, dtype).unsqueeze(0)
      window = {key: value.to(dtype=dtype) if value.is_floating_point() else value
                for key, value in world_model_window(
                  dataset, args.environment, args.start, max(HORIZONS)
                ).items()}
      for init_mode in init_modes:
        for horizon in HORIZONS:
          current = {key: value[:, :horizon] for key, value in window.items()}
          x, ext_fn, q, phys_fn, memory, memory_fn, bc_fn = rollout_functions(
            model, current, horizon, scale, init_mode
          )
          seed = 20261119 + horizon + 100 * len(rows)
          ext_gain, adjoint = measure(ext_fn, x, args.power_iterations, seed)
          phys_gain, phys_adjoint = measure(phys_fn, q, args.power_iterations, seed + 1)
          memory_gain = body_contact_gain = None
          if memory_fn is not None:
            memory_gain = spectral_norm_matrix_free(
              memory_fn, memory, iterations=args.power_iterations, seed=seed + 2
            )
            body_contact_gain = spectral_norm_matrix_free(
              bc_fn, q, iterations=args.power_iterations, seed=seed + 3
            )
          values = [ext_gain, phys_gain, adjoint, phys_adjoint]
          if memory_gain is not None: values += [memory_gain, body_contact_gain]
          rows.append({
            "method": method, "condition": condition, "partition": "development",
            "environment": args.environment, "start": args.start, "horizon": horizon,
            "contact_initialization": init_mode, "extended_state_width": x.shape[-1],
            "normalized_extended_gain": ext_gain,
            "normalized_physical_to_physical_gain": phys_gain,
            "log_physical_amplification_rate": math.log(max(phys_gain, 1e-300)) / horizon,
            "normalized_memory_to_body_gain": memory_gain,
            "normalized_body_to_contact_gain": body_contact_gain,
            "jvp_vjp_relative_error": adjoint,
            "physical_jvp_vjp_relative_error": phys_adjoint,
            "nonfinite_count": sum(not math.isfinite(value) for value in values),
            "near_zero_count": sum(abs(value) < 1e-12 for value in values),
            "dtype": args.dtype, "power_iterations": args.power_iterations,
            "checkpoint_sha256": frozen.sha256, "dataset_sha256": dataset.sha256,
          })
          print(json.dumps({"event": "jacobian", **rows[-1]}), flush=True)
    del frozen, model; gc.collect()
  args.output.mkdir(parents=True, exist_ok=True)
  csv_path = args.output / "trained_jacobian_horizon.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  summary = {
    "schema": 1, "status": "PASS", "rows": len(rows), "horizons": list(HORIZONS),
    "conditions": list(datasets), "partition": "development", "samples_per_condition": 1,
    "dtype": args.dtype, "power_iterations": args.power_iterations,
    "runtime_seconds": time.perf_counter() - started, "device": "cpu",
    "python": platform.python_version(), "torch": torch.__version__,
    "scaling": "condition-specific train-partition physical-state std (floor 1e-3); fixed SI contact scales; unit action scale",
    "rows_data": rows,
  }
  (args.output / "trained_jacobian_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
  )
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  styles = {"m1": "o-", "m2_v2": "s-", "m2_v2_nomemory": "^-"}
  for axis, condition in zip(axes, datasets):
    for method in styles:
      selected = [row for row in rows if row["condition"] == condition and row["method"] == method
                  and row["contact_initialization"] in {"not_applicable", "oracle_label"}]
      axis.plot([row["horizon"] for row in selected],
                [row["normalized_physical_to_physical_gain"] for row in selected],
                styles[method], label=method)
    axis.set(xscale="log", yscale="log", xlabel="Horizon",
             ylabel="normalized physical→physical gain", title=condition)
    axis.legend()
  fig.tight_layout(); fig.savefig(args.output / "trained_jacobian_horizon.png", dpi=180)
  print(json.dumps({key: value for key, value in summary.items() if key != "rows_data"}, indent=2))


if __name__ == "__main__":
  main()
