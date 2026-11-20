#!/usr/bin/env python3
"""Reconstruct collapse-guard chronology from the preserved formal-training log."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import EXPECTED_SHA256, sha256_file


METHODS = ("m1", "m2_v2", "m2_v2_nomemory")
METHOD_RE = re.compile(r"/seed14011/(m2_v2_nomemory|m2_v2|m1)/")
ITER_RE = re.compile(r"^it (\d+)/(\d+) \| (.*)$")
GUARD_RE = re.compile(
  r"collapse guard: ([0-9.]+) falls, rolled back to (.*?/model_(\d+)\.pt), lr scale ([0-9.e+-]+)"
)
PAIR_RE = re.compile(r"(?:^| \| )([a-z_]+) ([^ |]+)")


def git_sha(root: Path) -> str:
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def parse_float(value: str):
  try:
    return float(value)
  except ValueError:
    return None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-log", type=Path, required=True)
  parser.add_argument("--checkpoint", action="append", required=True, help="METHOD=PATH")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  current_method = None
  last_metric = {method: {} for method in METHODS}
  events = []
  for line_number, line in enumerate(args.training_log.read_text(errors="replace").splitlines(), 1):
    match = METHOD_RE.search(line)
    if match:
      current_method = match.group(1)
    iteration = ITER_RE.match(line)
    if iteration and current_method in METHODS:
      values = {key: parse_float(value) for key, value in PAIR_RE.findall(iteration.group(3))}
      values.update({"iteration": int(iteration.group(1)), "line": line_number})
      last_metric[current_method] = values
    guard = GUARD_RE.search(line)
    if not guard:
      continue
    method_match = METHOD_RE.search(guard.group(2))
    if not method_match or method_match.group(1) not in METHODS:
      continue
    method = method_match.group(1)
    metric = last_metric[method]
    events.append({
      "method": method, "event_index": 1 + sum(row["method"] == method for row in events),
      "iteration": metric.get("iteration"), "trigger_falls": float(guard.group(1)),
      "rollback_iteration": int(guard.group(3)), "lr_scale_after": float(guard.group(4)),
      "actor_grad_norm_before": metric.get("actor_grad_norm"),
      "wm_loss_before": metric.get("wm_loss"), "imagined_return_before": metric.get("imagined_return"),
      "fell_over_logged": metric.get("fell_over"), "distance_logged": metric.get("dist"),
      "log_line": line_number, "rollback_checkpoint": guard.group(2),
    })
  args.output.mkdir(parents=True, exist_ok=False)
  fields = list(events[0])
  with (args.output / "collapse_guard_events.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(events)
  summaries = {}
  for method in METHODS:
    rows = [row for row in events if row["method"] == method]
    summaries[method] = {
      "guard_events": len(rows),
      "first_guard_iteration": rows[0]["iteration"] if rows else None,
      "last_guard_iteration": rows[-1]["iteration"] if rows else None,
      "final_lr_scale": rows[-1]["lr_scale_after"] if rows else 1.0,
      "guard_events_with_actor_gradient": sum(row["actor_grad_norm_before"] is not None for row in rows),
      "rollback_iterations": sorted(set(row["rollback_iteration"] for row in rows)),
    }
  checkpoint_paths = dict(value.split("=", 1) for value in args.checkpoint)
  if set(checkpoint_paths) != set(EXPECTED_SHA256):
    raise ValueError("all three frozen checkpoints are required")
  final_optimizer = {}
  for method, path_text in checkpoint_paths.items():
    path = Path(path_text)
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256[method]:
      raise ValueError(f"checkpoint hash mismatch for {method}")
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    optimizer = raw["opt_actor"]
    steps = [float(value["step"]) for value in optimizer["state"].values() if "step" in value]
    final_optimizer[method] = {
      "checkpoint_sha256": digest, "training_lr_scale": raw["training_state"]["lr_scale"],
      "actor_optimizer_lr": optimizer["param_groups"][0]["lr"],
      "actor_optimizer_state_entries": len(optimizer["state"]),
      "actor_optimizer_step_min": min(steps) if steps else None,
      "actor_optimizer_step_max": max(steps) if steps else None,
      "nonfinite_steps": raw["training_state"]["nonfinite_steps"],
    }
  receipt = {
    "schema": 1, "status": "PASS", "source_sha": git_sha(root),
    "training_log": str(args.training_log.resolve()), "event_count": len(events),
    "methods": summaries, "final_optimizer_state": final_optimizer,
    "chronology_interpretation": (
      "Each learning-rate suppression is logged as part of and after its triggering guard event; "
      "the repeated guards therefore precede the extreme cumulative suppression."
    ),
    "limitations": [
      "This is observational chronology, not a causal intervention.",
      "Metrics are the most recent periodic log record immediately before each guard.",
      "No training was resumed and no optimizer state was modified.",
    ],
  }
  (args.output / "collapse_optimizer_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
  main()
