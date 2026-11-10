#!/usr/bin/env python3
"""Create paired B1-vs-B0 statistics and figures from formal episode CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


CONDITIONS = (
  "hard", "soft_nominal_1.0", "soft_seen_0.3", "soft_seen_0.6",
  "hard_to_soft", "soft_to_hard", "soft_unseen_0.2", "soft_unseen_1.2",
)
LABELS = {
  "hard": "Hard", "soft_nominal_1.0": "Soft 1.0", "soft_seen_0.3": "Seen 0.3",
  "soft_seen_0.6": "Seen 0.6", "hard_to_soft": "Hard→Soft",
  "soft_to_hard": "Soft→Hard", "soft_unseen_0.2": "Unseen 0.2",
  "soft_unseen_1.2": "Unseen 1.2",
}
METRICS = (
  "success", "forward_progress_m", "pursuit_velocity_mae_mps", "slip_speed_mps",
  "sinkage_m", "soil_force_n", "soil_dissipated_power_w", "actuator_abs_power_w",
  "actuator_work_per_path_m_jpm", "contact_transitions_per_min",
  "wm_one_step_normalized_mse", "wm_eight_step_normalized_state_mse",
)


def load(path: Path) -> dict[str, np.ndarray]:
  rows = list(csv.DictReader(path.open()))
  out = {}
  for key in rows[0]:
    if key in {"method", "condition"}:
      continue
    out[key] = np.array([float(row[key]) for row in rows], dtype=np.float64)
  return out


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
  keep = np.isfinite(a) & np.isfinite(b)
  a, b = a[keep], b[keep]
  delta = b - a
  n = len(delta)
  if n == 0:
    return {"n": 0, "b0_mean": math.nan, "b1_mean": math.nan, "delta": math.nan,
            "delta_percent": math.nan, "ci95_low": math.nan, "ci95_high": math.nan,
            "paired_p": math.nan}
  mean, sem = float(delta.mean()), float(stats.sem(delta)) if n > 1 else 0.0
  radius = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
  p = float(stats.ttest_rel(b, a).pvalue) if n > 1 and delta.std() > 0 else 1.0
  a_mean, b_mean = float(a.mean()), float(b.mean())
  return {
    "n": n, "b0_mean": a_mean, "b1_mean": b_mean, "delta": mean,
    "delta_percent": 100.0 * mean / abs(a_mean) if a_mean else math.nan,
    "ci95_low": mean - radius, "ci95_high": mean + radius, "paired_p": p,
  }


def fmt(value: float, digits: int = 3) -> str:
  return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", type=Path, required=True)
  args = parser.parse_args()
  comparisons = []
  data = {}
  for condition in CONDITIONS:
    b0 = load(args.input_dir / f"B0_{condition}_episodes.csv")
    b1 = load(args.input_dir / f"B1_{condition}_episodes.csv")
    if len(b0["episode"]) != 128 or not np.array_equal(b0["episode"], b1["episode"]):
      raise ValueError(f"unpaired or incomplete condition: {condition}")
    data[condition] = (b0, b1)
    for metric in METRICS:
      row = {"condition": condition, "metric": metric, **paired_stats(b0[metric], b1[metric])}
      if metric == "success":
        b0_only = int(((b0[metric] > 0.5) & (b1[metric] < 0.5)).sum())
        b1_only = int(((b1[metric] > 0.5) & (b0[metric] < 0.5)).sum())
        row["b0_only_successes"] = b0_only
        row["b1_only_successes"] = b1_only
        row["mcnemar_exact_p"] = (
          float(stats.binomtest(min(b0_only, b1_only), b0_only + b1_only, 0.5).pvalue)
          if b0_only + b1_only else 1.0
        )
      comparisons.append(row)

  fields = list(dict.fromkeys(key for row in comparisons for key in row))
  with (args.input_dir / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(comparisons)

  panels = (
    ("success", "Success rate", 100.0),
    ("forward_progress_m", "Forward progress (m)", 1.0),
    ("actuator_work_per_path_m_jpm", "Actuator work/path (J/m)", 1.0),
    ("slip_speed_mps", "Contact foot slip (m/s)", 1.0),
    ("wm_eight_step_normalized_state_mse", "8-step WM normalized MSE", 1.0),
  )
  fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
  x = np.arange(len(CONDITIONS))
  width = 0.38
  for ax, (metric, title, scale) in zip(axes.flat, panels):
    means0, means1, ci0, ci1 = [], [], [], []
    for condition in CONDITIONS:
      a, b = data[condition][0][metric], data[condition][1][metric]
      a, b = a[np.isfinite(a)] * scale, b[np.isfinite(b)] * scale
      means0.append(a.mean() if len(a) else np.nan)
      means1.append(b.mean() if len(b) else np.nan)
      ci0.append(1.96 * stats.sem(a) if len(a) > 1 else 0.0)
      ci1.append(1.96 * stats.sem(b) if len(b) > 1 else 0.0)
    ax.bar(x - width / 2, means0, width, yerr=ci0, label="B0", capsize=2)
    ax.bar(x + width / 2, means1, width, yerr=ci1, label="B1", capsize=2)
    ax.set_title(title)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
  axes.flat[0].legend()
  axes.flat[-1].axis("off")
  fig.suptitle("Held-out soft-terrain evaluation (128 paired episodes/condition)", fontsize=15)
  fig.savefig(args.input_dir / "held_out_comparison.png", dpi=180)
  plt.close(fig)

  lookup = {(r["condition"], r["metric"]): r for r in comparisons}
  lines = [
    "# Held-out paired evaluation", "",
    "B0 is the released hard-ground Actor; B1 is frozen seed-14011 model_12000.",
    "Each cell uses 128 paired, chase-only, 20 s held-out episodes. `Δ` is B1−B0.", "",
    "| Condition | Success B0 | Success B1 | Δ pp | Progress B0 m | Progress B1 m | Δ % | Work B0 J/m | Work B1 J/m | 8-step WM B0 | 8-step WM B1 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for c in CONDITIONS:
    s, p = lookup[(c, "success")], lookup[(c, "forward_progress_m")]
    w, m = lookup[(c, "actuator_work_per_path_m_jpm")], lookup[(c, "wm_eight_step_normalized_state_mse")]
    lines.append(
      f"| {LABELS[c]} | {100*s['b0_mean']:.2f}% | {100*s['b1_mean']:.2f}% | "
      f"{100*s['delta']:+.2f} | {p['b0_mean']:.2f} | {p['b1_mean']:.2f} | "
      f"{p['delta_percent']:+.2f}% | {w['b0_mean']:.2f} | {w['b1_mean']:.2f} | "
      f"{m['b0_mean']:.3f} | {m['b1_mean']:.3f} |"
    )
  lines += [
    "", "## Interpretation", "",
    "- B1 improves forward progress most on softer soil, but does not improve survival.",
    "- B1 regresses survival on hard ground, nominal/strong unseen soil, and both transitions.",
    "- The trained world model's 8-step error is lower under B1 trajectories for both seen randomized strengths, both transitions, and both unseen strengths; nominal scale 1.0 is the exception. Better prediction did not translate into safer boundary crossing.",
    "- This is one training seed; uncertainty shown in the raw summaries is episode dispersion, not across-seed uncertainty.",
  ]
  (args.input_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
