#!/usr/bin/env python3
"""Aggregate the five-method, single-training-seed frozen Stage-4 matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from run_contact_formal_eval import EVAL_SEEDS, STAGE4_CONDITIONS
from run_formal_v2_stage4_eval import METHODS


METRICS = (
  "success", "fall", "duration_s", "episode_return", "forward_progress_m",
  "path_length_m", "net_displacement_m", "pursuit_velocity_mae_mps",
  "slip_speed_mps", "sinkage_m", "soil_force_n", "soil_dissipated_power_w",
  "actuator_abs_power_w", "actuator_work_per_path_m_jpm",
  "contact_transitions_per_min", "wm_one_step_normalized_mse",
  "wm_eight_step_normalized_state_mse", "contact_force_mae_n",
  "contact_sinkage_mae_m", "contact_memory_mae_m", "contact_accuracy",
)
BINARY_METRICS = {"success", "fall"}


def load_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as stream:
    return list(csv.DictReader(stream))


def describe(values: np.ndarray) -> dict[str, float | int]:
  values = values[np.isfinite(values)]
  n = len(values)
  if not n:
    return {"n": 0, "mean": math.nan, "std": math.nan,
            "ci95_low": math.nan, "ci95_high": math.nan}
  mean = float(values.mean())
  std = float(values.std(ddof=1)) if n > 1 else 0.0
  radius = float(stats.t.ppf(0.975, n - 1) * stats.sem(values)) if n > 1 else 0.0
  return {"n": n, "mean": mean, "std": std,
          "ci95_low": mean - radius, "ci95_high": mean + radius}


def paired_p(a: np.ndarray, b: np.ndarray, metric: str) -> tuple[str, float]:
  if metric in BINARY_METRICS:
    b01 = int(((a < 0.5) & (b >= 0.5)).sum())
    b10 = int(((a >= 0.5) & (b < 0.5)).sum())
    discordant = b01 + b10
    p = float(stats.binomtest(min(b01, b10), discordant, 0.5).pvalue) if discordant else 1.0
    return "exact_mcnemar", p
  delta = b - a
  if len(delta) < 2 or float(delta.std()) == 0.0:
    return "paired_t", 1.0
  return "paired_t", float(stats.ttest_rel(b, a).pvalue)


def holm_adjust(pvalues: list[float]) -> list[float]:
  order = np.argsort(pvalues)
  adjusted = np.ones(len(pvalues), dtype=float)
  running = 0.0
  m = len(pvalues)
  for rank, index in enumerate(order):
    running = max(running, (m - rank) * pvalues[index])
    adjusted[index] = min(1.0, running)
  return adjusted.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", type=Path, required=True)
  args = parser.parse_args()
  if not (args.input_dir / "COMPLETE").exists():
    raise RuntimeError("Stage-4 matrix is incomplete")

  all_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
  seed_rows = []
  for method in METHODS:
    for condition in STAGE4_CONDITIONS:
      combined = []
      for seed in EVAL_SEEDS:
        rows = load_rows(
          args.input_dir / f"seed{seed}" / f"{method}_{condition}_episodes.csv"
        )
        if len(rows) != 128:
          raise ValueError(f"incomplete {method}/{condition}/seed{seed}: {len(rows)}")
        combined.extend(rows)
        for metric in METRICS:
          values = np.asarray([float(row[metric]) for row in rows])
          seed_rows.append({
            "method": method, "condition": condition, "evaluation_seed": seed,
            "metric": metric, **describe(values),
          })
      all_rows[(method, condition)] = combined

  aggregate_rows = []
  for (method, condition), rows in all_rows.items():
    for metric in METRICS:
      values = np.asarray([float(row[metric]) for row in rows])
      aggregate_rows.append({
        "method": method, "condition": condition, "metric": metric,
        **describe(values),
      })

  paired_rows = []
  for comparator in METHODS:
    if comparator == "M1":
      continue
    for condition in STAGE4_CONDITIONS:
      base = all_rows[("M1", condition)]
      other = all_rows[(comparator, condition)]
      for metric in METRICS:
        a = np.asarray([float(row[metric]) for row in base])
        b = np.asarray([float(row[metric]) for row in other])
        keep = np.isfinite(a) & np.isfinite(b)
        delta = b[keep] - a[keep]
        desc = describe(delta)
        test, p = paired_p(a[keep], b[keep], metric)
        paired_rows.append({
          "baseline": "M1", "comparator": comparator, "condition": condition,
          "metric": metric, "n": desc["n"],
          "baseline_mean": float(a[keep].mean()) if keep.any() else math.nan,
          "comparator_mean": float(b[keep].mean()) if keep.any() else math.nan,
          "comparator_minus_baseline": desc["mean"],
          "ci95_low": desc["ci95_low"], "ci95_high": desc["ci95_high"],
          "test": test, "paired_p": p,
        })
  adjusted = holm_adjust([float(row["paired_p"]) for row in paired_rows])
  for row, value in zip(paired_rows, adjusted):
    row["holm_p_all_stage4_comparisons"] = value

  all_pair_rows = []
  for baseline, comparator in itertools.combinations(METHODS, 2):
    for condition in STAGE4_CONDITIONS:
      base = all_rows[(baseline, condition)]
      other = all_rows[(comparator, condition)]
      for metric in METRICS:
        a = np.asarray([float(row[metric]) for row in base])
        b = np.asarray([float(row[metric]) for row in other])
        keep = np.isfinite(a) & np.isfinite(b)
        delta = b[keep] - a[keep]
        desc = describe(delta)
        test, p = paired_p(a[keep], b[keep], metric)
        all_pair_rows.append({
          "baseline": baseline, "comparator": comparator, "condition": condition,
          "metric": metric, "n": desc["n"],
          "baseline_mean": float(a[keep].mean()) if keep.any() else math.nan,
          "comparator_mean": float(b[keep].mean()) if keep.any() else math.nan,
          "comparator_minus_baseline": desc["mean"],
          "ci95_low": desc["ci95_low"], "ci95_high": desc["ci95_high"],
          "test": test, "paired_p": p,
        })
  adjusted = holm_adjust([float(row["paired_p"]) for row in all_pair_rows])
  for row, value in zip(all_pair_rows, adjusted):
    row["holm_p_all_pairwise_comparisons"] = value

  write_csv(args.input_dir / "aggregate.csv", aggregate_rows)
  write_csv(args.input_dir / "per_evaluation_seed.csv", seed_rows)
  write_csv(args.input_dir / "paired_vs_m1.csv", paired_rows)
  write_csv(args.input_dir / "paired_all_methods.csv", all_pair_rows)

  lookup = {(r["method"], r["condition"], r["metric"]): r for r in aggregate_rows}
  panels = (
    ("success", "Success rate", 100.0),
    ("forward_progress_m", "Forward progress (m)", 1.0),
    ("pursuit_velocity_mae_mps", "Velocity MAE (m/s)", 1.0),
    ("slip_speed_mps", "Contact-foot slip (m/s)", 1.0),
    ("actuator_abs_power_w", "Actuator abs. power (W)", 1.0),
    ("wm_eight_step_normalized_state_mse", "8-step WM normalized MSE", 1.0),
  )
  colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
  fig, axes = plt.subplots(3, 2, figsize=(18, 13), constrained_layout=True)
  x = np.arange(len(STAGE4_CONDITIONS))
  width = 0.16
  offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
  for ax, (metric, title, scale) in zip(axes.flat, panels):
    for offset, (method, color) in zip(offsets, zip(METHODS, colors)):
      rows = [lookup[(method, condition, metric)] for condition in STAGE4_CONDITIONS]
      means = [row["mean"] * scale for row in rows]
      errors = [(row["ci95_high"] - row["mean"]) * scale for row in rows]
      ax.bar(x + offset, means, width, yerr=errors, label=method,
             color=color, capsize=1.5)
    ax.set_title(title)
    ax.set_xticks(x, STAGE4_CONDITIONS, rotation=27, ha="right")
    ax.grid(axis="y", alpha=0.25)
  axes.flat[0].legend(ncol=3, fontsize=8)
  fig.suptitle(
    "Frozen Stage-4 OOD evaluation: one training seed, 3×128 paired episodes",
    fontsize=15,
  )
  fig.savefig(args.input_dir / "stage4_v2_comparison.png", dpi=180)
  plt.close(fig)

  contact_methods = ("M2_V1", "M2_V2", "M2_V2_NOMEMORY")
  contact_metrics = (
    ("contact_accuracy", "Contact accuracy"),
    ("contact_force_mae_n", "Force MAE (N)"),
    ("contact_sinkage_mae_m", "Sinkage MAE (m)"),
    ("contact_memory_mae_m", "Loading-memory MAE (m)"),
  )
  fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
  width = 0.24
  offsets = (-width, 0.0, width)
  for ax, (metric, title) in zip(axes.flat, contact_metrics):
    for offset, method, color in zip(offsets, contact_methods, colors[1:4]):
      rows = [lookup[(method, condition, metric)] for condition in STAGE4_CONDITIONS]
      means = [row["mean"] for row in rows]
      errors = [row["ci95_high"] - row["mean"] for row in rows]
      ax.bar(x + offset, means, width, yerr=errors, label=method,
             color=color, capsize=2)
    ax.set_title(title)
    ax.set_xticks(x, STAGE4_CONDITIONS, rotation=27, ha="right")
    ax.grid(axis="y", alpha=0.25)
  axes.flat[0].legend(fontsize=8)
  fig.suptitle("Contact-structured frozen diagnostics (3×128 episodes/condition)")
  fig.savefig(args.input_dir / "stage4_v2_contact_diagnostics.png", dpi=180)
  plt.close(fig)

  manifest = {
    "schema": 1,
    "training_seeds": [14011],
    "evaluation_seeds": list(EVAL_SEEDS),
    "episodes_per_method_condition": len(EVAL_SEEDS) * 128,
    "methods": list(METHODS),
    "conditions": list(STAGE4_CONDITIONS),
    "aggregate": str((args.input_dir / "aggregate.csv").resolve()),
    "per_evaluation_seed": str((args.input_dir / "per_evaluation_seed.csv").resolve()),
    "paired_vs_m1": str((args.input_dir / "paired_vs_m1.csv").resolve()),
    "paired_all_methods": str((args.input_dir / "paired_all_methods.csv").resolve()),
    "overview_figure": str((args.input_dir / "stage4_v2_comparison.png").resolve()),
    "contact_figure": str((args.input_dir / "stage4_v2_contact_diagnostics.png").resolve()),
    "inference_limit": (
      "Intervals and paired tests quantify episode/evaluation-seed variation for one "
      "training seed; they do not estimate across-training-seed uncertainty."
    ),
  }
  (args.input_dir / "summary_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
  )


if __name__ == "__main__":
  main()
