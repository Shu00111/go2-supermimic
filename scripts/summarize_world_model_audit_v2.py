#!/usr/bin/env python3
"""Build weighted P1 audit summaries and a publication-quality diagnostic figure."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def weighted(rows: list[dict], *, family: str, metric: str, model: str,
             stratum: str) -> tuple[float, int]:
  selected = [row for row in rows if row["family"] == family and row["metric"] == metric
              and row["model"] == model and row["stratum"] == stratum and row["n"]]
  n = sum(int(row["n"]) for row in selected)
  return (sum(float(row["mean"]) * int(row["n"]) for row in selected) / n, n) if n else (np.nan, 0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-figure", type=Path, required=True)
  args = parser.parse_args()
  audit = json.loads(args.input.read_text())
  rows = audit["rows"]

  grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
  for row in rows:
    grouped[(row["model"], row["family"], row["metric"], row["stratum"])].append(row)
  aggregate = []
  for (model, family, metric, stratum), values in sorted(grouped.items()):
    n = sum(int(row["n"]) for row in values)
    if not n:
      continue
    aggregate.append({
      "model": model, "family": family, "metric": metric, "stratum": stratum,
      "n": n,
      "weighted_mean": sum(float(row["mean"]) * int(row["n"]) for row in values) / n,
      "conditions": len(values),
    })
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps({
    "schema": 1, "source": str(args.input.resolve()), "split": audit["split"],
    "aggregate": aggregate,
  }, indent=2) + "\n")

  conditions = sorted({row["condition"] for row in rows})
  phases = ["onset", "stance", "release", "flight"]
  contact_models = ["M2-v1/predicted-current", "M2-v1/oracle-current"]
  colors = {"M1": "#3B6FB6", "M2-v1": "#D55E00",
            contact_models[0]: "#D55E00", contact_models[1]: "#009E73"}
  fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

  ax = axes[0, 0]
  for condition in conditions:
    for model in ("M1", "M2-v1"):
      selected = [r for r in rows if r["condition"] == condition and r["model"] == model
                  and r["metric"] == "autoregressive_state_mse"]
      selected.sort(key=lambda r: int(r["stratum"].split("_")[-1]))
      if selected:
        ax.plot([int(r["stratum"].split("_")[-1]) for r in selected],
                [r["mean"] for r in selected], color=colors[model], alpha=0.28,
                marker="o", markersize=2.8, linewidth=1.0)
  for model in ("M1", "M2-v1"):
    x, y = [], []
    for horizon in (1, 2, 4, 8, 16, 32):
      value, n = weighted(rows, family="world_model", metric="autoregressive_state_mse",
                          model=model, stratum=f"horizon_{horizon}")
      if n:
        x.append(horizon); y.append(value)
    ax.plot(x, y, color=colors[model], marker="o", linewidth=2.5, label=model)
  ax.set_xscale("log", base=2)
  ax.set_yscale("log")
  ax.set_xlabel("Autoregressive horizon")
  ax.set_ylabel("Common-normalized state MSE")
  ax.set_title("A. Matched-data rollout error")
  ax.grid(alpha=0.22)
  ax.legend(frameon=False)

  def grouped_bars(ax, family, metric, title, ylabel, transform=lambda x: x):
    x = np.arange(len(phases)); width = 0.36
    for offset, model in zip((-width / 2, width / 2), contact_models):
      values = [transform(weighted(rows, family=family, metric=metric, model=model,
                                   stratum=phase)[0]) for phase in phases]
      ax.bar(x + offset, values, width, label=model.split("/")[-1], color=colors[model])
    ax.set_xticks(x, phases)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)

  grouped_bars(axes[0, 1], "contact_force", "normal_absolute_error_n",
               "B. Normal-force error by phase", "MAE [N]")
  grouped_bars(axes[1, 0], "loading_memory", "absolute_error_m",
               "C. Loading-memory error by phase", "MAE [mm]", lambda x: 1000 * x)
  grouped_bars(axes[1, 1], "contact_classification", "incorrect",
               "D. Contact classification by phase", "Accuracy", lambda x: 1 - x)
  axes[0, 1].legend(frameon=False)
  axes[1, 0].legend(frameon=False)
  axes[1, 1].legend(frameon=False)
  fig.suptitle(f"P1 world-model error audit ({audit['split']} split)", fontsize=14)
  args.output_figure.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output_figure, dpi=220)
  plt.close(fig)
  print(json.dumps({"summary": str(args.output_json), "figure": str(args.output_figure),
                    "aggregate_rows": len(aggregate)}, indent=2))


if __name__ == "__main__":
  main()
