#!/usr/bin/env python3
"""Aggregate and pair the frozen M1/M2 Stage-4 evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from run_contact_formal_eval import EVAL_SEEDS, STAGE4_CONDITIONS


METRICS = (
  "success", "fall", "forward_progress_m", "pursuit_velocity_mae_mps",
  "slip_speed_mps", "sinkage_m", "actuator_abs_power_w",
  "actuator_work_per_path_m_jpm", "wm_one_step_normalized_mse",
  "wm_eight_step_normalized_state_mse", "contact_force_mae_n",
  "contact_sinkage_mae_m", "contact_memory_mae_m", "contact_accuracy",
)


def load_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as stream:
    return list(csv.DictReader(stream))


def finite(values: np.ndarray) -> np.ndarray:
  return values[np.isfinite(values)]


def describe(values: np.ndarray) -> dict[str, float | int]:
  values = finite(values)
  n = len(values)
  if not n:
    return {"n": 0, "mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
  mean = float(values.mean())
  radius = float(stats.t.ppf(0.975, n - 1) * stats.sem(values)) if n > 1 else 0.0
  return {"n": n, "mean": mean, "ci95_low": mean - radius, "ci95_high": mean + radius}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", type=Path, required=True)
  args = parser.parse_args()
  if not (args.input_dir / "COMPLETE").exists():
    raise RuntimeError("Stage-4 matrix is incomplete")

  all_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
  for method in ("M1", "M2"):
    for condition in STAGE4_CONDITIONS:
      rows = []
      for seed in EVAL_SEEDS:
        part = load_rows(args.input_dir / f"seed{seed}" / f"{method}_{condition}_episodes.csv")
        if len(part) != 128:
          raise ValueError(f"incomplete {method}/{condition}/seed{seed}: {len(part)}")
        for row in part:
          row["evaluation_seed"] = str(seed)
        rows.extend(part)
      all_rows[(method, condition)] = rows

  aggregate_rows = []
  paired_rows = []
  for condition in STAGE4_CONDITIONS:
    m1, m2 = all_rows[("M1", condition)], all_rows[("M2", condition)]
    for method, rows in (("M1", m1), ("M2", m2)):
      for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        aggregate_rows.append({"method": method, "condition": condition, "metric": metric,
                               **describe(values)})
    for metric in METRICS:
      a = np.asarray([float(row[metric]) for row in m1])
      b = np.asarray([float(row[metric]) for row in m2])
      keep = np.isfinite(a) & np.isfinite(b)
      delta = b[keep] - a[keep]
      desc = describe(delta)
      paired_rows.append({
        "condition": condition, "metric": metric, "n": desc["n"],
        "m1_mean": float(a[keep].mean()) if keep.any() else math.nan,
        "m2_mean": float(b[keep].mean()) if keep.any() else math.nan,
        "m2_minus_m1": desc["mean"], "ci95_low": desc["ci95_low"],
        "ci95_high": desc["ci95_high"],
        "paired_p": (float(stats.ttest_rel(b[keep], a[keep]).pvalue)
                     if keep.sum() > 1 and delta.std() > 0 else 1.0),
      })

  def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)

  write_csv(args.input_dir / "aggregate.csv", aggregate_rows)
  write_csv(args.input_dir / "paired_comparison.csv", paired_rows)

  lookup = {(r["method"], r["condition"], r["metric"]): r for r in aggregate_rows}
  panels = (
    ("success", "Success rate", 100.0),
    ("forward_progress_m", "Forward progress (m)", 1.0),
    ("pursuit_velocity_mae_mps", "Velocity MAE (m/s)", 1.0),
    ("slip_speed_mps", "Contact-foot slip (m/s)", 1.0),
    ("actuator_work_per_path_m_jpm", "Actuator work/path (J/m)", 1.0),
    ("wm_eight_step_normalized_state_mse", "8-step WM normalized MSE", 1.0),
  )
  fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
  x = np.arange(len(STAGE4_CONDITIONS))
  width = 0.38
  for ax, (metric, title, scale) in zip(axes.flat, panels):
    for offset, method in ((-width / 2, "M1"), (width / 2, "M2")):
      rows = [lookup[(method, c, metric)] for c in STAGE4_CONDITIONS]
      means = [r["mean"] * scale for r in rows]
      errors = [(r["ci95_high"] - r["mean"]) * scale for r in rows]
      ax.bar(x + offset, means, width, yerr=errors, label=method, capsize=2)
    ax.set_title(title)
    ax.set_xticks(x, STAGE4_CONDITIONS, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
  axes.flat[0].legend()
  fig.suptitle("Frozen Stage-4 OOD evaluation: 3×128 paired episodes per condition")
  fig.savefig(args.input_dir / "stage4_comparison.png", dpi=180)
  plt.close(fig)

  contact_metrics = (
    ("contact_accuracy", "Contact accuracy"),
    ("contact_force_mae_n", "Force MAE (N)"),
    ("contact_sinkage_mae_m", "Sinkage MAE (m)"),
    ("contact_memory_mae_m", "Loading-memory MAE (m)"),
  )
  fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
  for ax, (metric, title) in zip(axes.flat, contact_metrics):
    rows = [lookup[("M2", c, metric)] for c in STAGE4_CONDITIONS]
    means = [r["mean"] for r in rows]
    errors = [r["ci95_high"] - r["mean"] for r in rows]
    ax.bar(x, means, yerr=errors, color="tab:orange", capsize=2)
    ax.set_title(title)
    ax.set_xticks(x, STAGE4_CONDITIONS, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
  fig.suptitle("M2 frozen contact-state diagnostics (3×128 episodes/condition)")
  fig.savefig(args.input_dir / "contact_diagnostics.png", dpi=180)
  plt.close(fig)

  manifest = {
    "episodes_per_method_condition": 3 * 128,
    "aggregate": str((args.input_dir / "aggregate.csv").resolve()),
    "paired": str((args.input_dir / "paired_comparison.csv").resolve()),
    "figure": str((args.input_dir / "stage4_comparison.png").resolve()),
    "contact_figure": str((args.input_dir / "contact_diagnostics.png").resolve()),
  }
  (args.input_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
