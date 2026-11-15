#!/usr/bin/env python3
"""Aggregate paired gradient-fidelity receipts over valid physical coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import torch


def metrics(physical: torch.Tensor, model: torch.Tensor) -> dict:
  p, q = physical.double(), model.double()
  pn, qn = float(p.norm()), float(q.norm())
  return {
    "pairs": int(len(p)),
    "cosine_similarity": float(torch.dot(p, q) / (p.norm() * q.norm())),
    "relative_l2_error": float((q - p).norm() / p.norm()),
    "sign_agreement": float((torch.sign(p) == torch.sign(q)).double().mean()),
    "physical_norm": pn,
    "model_norm": qn,
    "norm_ratio": qn / pn,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("inputs", type=Path, nargs="+")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
  identity = {(row["scenario"], row["horizon"]) for row in reports}
  if len(identity) != 1:
    raise SystemExit("all inputs must have the same scenario and horizon")

  physical_parts = []
  method_names = tuple(reports[0]["methods"])
  if not method_names or any(tuple(report["methods"]) != method_names for report in reports):
    raise SystemExit("all inputs must contain the same ordered method set")
  model_parts = {method: [] for method in method_names}
  sample_metrics = {method: [] for method in method_names}
  valid_count = 0
  for path, report in zip(args.inputs, reports, strict=True):
    valid = report["physical_valid_coordinates"]
    physical = torch.tensor(report["physical_gradient"])[valid]
    physical_parts.append(physical)
    valid_count += len(valid)
    for method in model_parts:
      gradient = torch.tensor(report["methods"][method]["gradient"])[valid]
      model_parts[method].append(gradient)
      sample_metrics[method].append({
        "source": str(path), "seed": report["seed"], **metrics(physical, gradient),
      })

  physical = torch.cat(physical_parts)
  methods = {}
  for method, parts in model_parts.items():
    aggregate = metrics(physical, torch.cat(parts))
    per_sample = sample_metrics[method]
    methods[method] = {
      "aggregate": aggregate,
      "per_sample": per_sample,
      "mean_sample_cosine": mean(row["cosine_similarity"] for row in per_sample),
      "median_sample_cosine": median(row["cosine_similarity"] for row in per_sample),
      "mean_sample_sign_agreement": mean(row["sign_agreement"] for row in per_sample),
    }
  scenario, horizon = identity.pop()
  summary = {
    "schema": 1,
    "status": "COMPLETE",
    "scenario": scenario,
    "horizon": horizon,
    "samples": len(reports),
    "seeds": [row["seed"] for row in reports],
    "total_coordinates": 12 * len(reports),
    "valid_pairs": valid_count,
    "valid_pair_fraction": valid_count / (12 * len(reports)),
    "methods": methods,
    "sources": [str(path) for path in args.inputs],
  }
  if "M1" in methods:
    baseline = methods["M1"]["aggregate"]
    summary["method_minus_M1"] = {
      method: {
        "cosine_similarity": values["aggregate"]["cosine_similarity"]
        - baseline["cosine_similarity"],
        "relative_l2_error": values["aggregate"]["relative_l2_error"]
        - baseline["relative_l2_error"],
        "sign_agreement": values["aggregate"]["sign_agreement"]
        - baseline["sign_agreement"],
        "norm_ratio": values["aggregate"]["norm_ratio"] - baseline["norm_ratio"],
      }
      for method, values in methods.items() if method != "M1"
    }
  if "M1" in methods and "M2" in methods:
    summary["M2_minus_M1"] = summary["method_minus_M1"]["M2"]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
