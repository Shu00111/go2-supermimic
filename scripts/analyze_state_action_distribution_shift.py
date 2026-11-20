#!/usr/bin/env python3
"""Compare preserved final replay buffers with independent fixed-policy data."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, sha256_file
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory
from go2_supermimic.supermimic import state as st


def git_sha():
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sample_replay(method, path, sample_count):
  digest = sha256_file(path)
  if digest != EXPECTED_SHA256[method]:
    raise ValueError(f"checkpoint hash mismatch for {method}")
  raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  replay = raw["replay_buffer"]
  total = replay["s"].shape[0] * int(replay["size"])
  generator = torch.Generator().manual_seed(20261120)
  index = torch.randperm(total, generator=generator)[:min(sample_count, total)]
  size = int(replay["size"])
  env, slot = index // size, index % size
  values = {
    "state": replay["s"][env, slot].float().clone(),
    "action": replay["a"][env, slot].float().clone(),
    "contact": replay["contact"][env, slot].float().clone(),
  }
  metadata = {"size_per_environment": size, "num_environments": int(replay["n"]),
              "sample_count": len(index), "checkpoint_sha256": digest}
  del raw, replay
  gc.collect()
  return values, metadata


def metrics(label, values, reference):
  state, action, contact = values["state"], values["action"], values["contact"]
  s_mean, s_std, s_lo, s_hi, a_mean, a_std = reference
  z = (state - s_mean) / s_std
  az = (action - a_mean) / a_std
  return {
    "source": label, "sample_count": state.shape[0],
    "state_z_rms_vs_m1_replay": float(z.square().mean().sqrt()),
    "state_z_l2_mean_vs_m1_replay": float(z.norm(dim=-1).mean()),
    "state_feature_coverage_m1_1_99pct": float(((state >= s_lo) & (state <= s_hi)).float().mean()),
    "action_z_rms_vs_m1_replay": float(az.square().mean().sqrt()),
    "action_rms": float(action.square().mean().sqrt()),
    "action_unit_exceedance_fraction": float((action.abs() >= 1).float().mean()),
    "action_saturation_fraction": 0.0,
    "static_state_fraction_speed_lt_0_1": float((state[:, st.V_YAW].norm(dim=-1) < 0.1).float().mean()),
    "base_speed_mean": float(state[:, st.V_YAW].norm(dim=-1).mean()),
    "contact_code_rms": float(contact.square().mean().sqrt()),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", action="append", required=True, help="METHOD=PATH")
  parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH=SHA256")
  parser.add_argument("--sample-count", type=int, default=8192)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  checkpoints = dict(value.split("=", 1) for value in args.checkpoint)
  if set(checkpoints) != set(EXPECTED_SHA256):
    raise ValueError("all three checkpoints are required")
  samples, provenance = {}, {}
  for method, path in checkpoints.items():
    samples[method], provenance[method] = sample_replay(method, Path(path), args.sample_count)
  m1 = samples["m1"]
  reference = (
    m1["state"].mean(0), m1["state"].std(0, unbiased=False).clamp_min(1e-3),
    torch.quantile(m1["state"], 0.01, dim=0), torch.quantile(m1["state"], 0.99, dim=0),
    m1["action"].mean(0), m1["action"].std(0, unbiased=False).clamp_min(1e-3),
  )
  rows = [metrics(f"final_replay_{method}", values, reference) for method, values in samples.items()]
  dataset_hashes = {}
  for value in args.dataset:
    name, path, digest = value.split("=", 2)
    dataset = load_frozen_trajectory(Path(path), expected_sha256=digest)
    ids = dataset.environment_ids("development")
    data = dataset.data
    values = {
      "state": data["state"][ids].reshape(-1, st.STATE_DIM),
      "action": data["action"][ids].reshape(-1, st.NUM_JOINTS),
      "contact": data["contact"][ids].reshape(-1, 32),
    }
    rows.append(metrics(f"fixed_policy_development_{name}", values, reference))
    dataset_hashes[name] = dataset.sha256
  with (args.output / "state_action_distribution_shift.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
  receipt = {
    "schema": 1, "status": "PARTIAL", "source_sha": git_sha(), "rows": rows,
    "checkpoint_provenance": provenance, "dataset_sha256": dataset_hashes,
    "reference": "deterministic sample of M1 final replay buffer",
    "action_contract": "JointPositionActionCfg.clip=None; saturation is zero by definition; unit exceedance is reported separately",
    "limitations": [
      "Final replay buffers cover only their retained ring windows, not full training history.",
      "Historical final-evaluation raw trajectories were not preserved; aggregate evaluation cannot yield feature distances.",
      "Distances are complementary diagnostics, not definitive OOD tests.",
    ],
  }
  (args.output / "state_action_distribution_shift.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
  main()
