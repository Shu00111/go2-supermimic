#!/usr/bin/env python3
"""Build the P0 phase table and compact diagnostic figure from frozen receipts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PHYSICAL = (
  ("rigid stable 14011", "physical_gradient_rigid_stable_h32_seed14011_v2.json", None),
  ("rigid stable 14012", "physical_gradient_rigid_stable_h32_seed14012_v2.json", None),
  ("rigid stable 14013", "physical_gradient_rigid_stable_h32_seed14013_v2.json", None),
  ("rigid stable 14014", "physical_gradient_rigid_stable_h32_seed14014_v2.json", None),
  ("rigid aerial", "physical_gradient_rigid_aerial_h32_seed14011_v2.json", None),
  ("soft stable", "physical_gradient_soft_stable_seed14011_v2.json",
   "physical_gradient_soft_stable_seed14011_v2_analysis.json"),
  ("touchdown", "physical_gradient_touchdown_h32_seed14011_v2.json", None),
  ("liftoff", "physical_gradient_liftoff_h32_seed14011_v2.json", None),
  ("hard to soft", "physical_gradient_hard_to_soft_h32_seed14011_v2.json", None),
  ("soft to hard", "physical_gradient_soft_to_hard_h32_seed14011_v2.json", None),
)


def load(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report-dir", type=Path, required=True)
  parser.add_argument("--json", type=Path, required=True)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--figure", type=Path, required=True)
  args = parser.parse_args()
  for output in (args.json, args.csv, args.figure):
    if output.exists():
      raise SystemExit(f"refusing to overwrite {output}")

  rows = []
  for label, raw_name, analysis_name in PHYSICAL:
    raw_path = args.report_dir / raw_name
    raw = load(raw_path)
    analysis = load(args.report_dir / analysis_name) if analysis_name else raw
    convergence = analysis["convergence"]
    valid = sum(row["valid"] for row in convergence)
    rows.append({
      "phase": label,
      "source": str(raw_path),
      "seed": raw["seed"],
      "horizon": raw["horizon"],
      "valid_coordinates": valid,
      "valid_fraction": valid / len(convergence),
      "low_signal_coordinates": sum(
        not row.get("active_at_all_epsilons", row.get("active", False))
        for row in convergence
      ),
      "epsilon_unconverged_coordinates": sum(
        not row["epsilon_converged"] for row in convergence
      ),
      "sign_flip_coordinates": sum(not row["sign_consistent"] for row in convergence),
      "discontinuous_coordinates": sum(row["discontinuity"] for row in convergence),
      "replay_objective_abs_error": raw["replay_objective_abs_error"],
    })

  model = load(args.report_dir / "gradient_fidelity_rigid_stable_h32_aggregate_n4_v2.json")
  summary = {
    "schema": 1,
    "status": "COMPLETE",
    "criterion": (
      "valid only when all three epsilon signals clear the float32/replay floor, signs agree, "
      "relative span <=0.25, and no saturation, non-finite, terminal, contact, material, "
      "or MuJoCo constraint-set change occurs"
    ),
    "physical_phase_rows": rows,
    "rigid_stable_model_comparison": model,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  with args.csv.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
  labels = [row["phase"] for row in rows]
  values = [row["valid_fraction"] for row in rows]
  colors = ["#4472C4" if value else "#C0504D" for value in values]
  axes[0].barh(range(len(rows)), values, color=colors)
  axes[0].set_yticks(range(len(rows)), labels)
  axes[0].invert_yaxis()
  axes[0].set_xlim(0, 1)
  axes[0].set_xlabel("valid physical-coordinate fraction")
  axes[0].set_title("H=32 finite-difference validity")
  axes[0].grid(axis="x", alpha=0.25)

  names = ("cosine", "sign agreement", "1 - relative L2 error")
  x = range(len(names))
  width = 0.36
  for index, method in enumerate(("M1", "M2")):
    metric = model["methods"][method]["aggregate"]
    values = (
      metric["cosine_similarity"], metric["sign_agreement"],
      1.0 - metric["relative_l2_error"],
    )
    axes[1].bar([value + (index - 0.5) * width for value in x], values, width,
                label=method, color=("#A5A5A5" if method == "M1" else "#70AD47"))
  axes[1].set_xticks(list(x), names, rotation=15, ha="right")
  axes[1].set_ylim(0, 0.8)
  axes[1].set_ylabel("score (higher is better)")
  axes[1].set_title("Rigid-stable aggregate (4 states, 29 valid pairs)")
  axes[1].legend(frameon=False)
  axes[1].grid(axis="y", alpha=0.25)
  args.figure.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.figure, dpi=180)
  plt.close(fig)
  print(json.dumps({"json": str(args.json), "csv": str(args.csv),
                    "figure": str(args.figure)}, indent=2))


if __name__ == "__main__":
  main()
