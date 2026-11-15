#!/usr/bin/env python3
"""Add conservative epsilon-convergence classification to a raw FD receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2_supermimic.diagnostics import finite_difference_convergence


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
  report = json.loads(args.input.read_text(encoding="utf-8"))
  convergence = finite_difference_convergence(report["rows"], report["nominal_objective"])
  summary = {
    "schema": 1,
    "source": str(args.input),
    "scenario": report["scenario"],
    "horizon": report["horizon"],
    "nominal_objective": report["nominal_objective"],
    "replay_objective_abs_error": report["replay_objective_abs_error"],
    "valid_coordinate_fraction": sum(row["valid"] for row in convergence) / len(convergence),
    "valid_coordinates": [row["coordinate"] for row in convergence if row["valid"]],
    "invalid_coordinates": [row["coordinate"] for row in convergence if not row["valid"]],
    "convergence": convergence,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
