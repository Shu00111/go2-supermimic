#!/usr/bin/env python3
"""Restart-safe one-time P1 held-out physical-gradient benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (15011, 15012, 15013, 15014)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def load_complete(path: Path, *, seed: int, methods: tuple[str, ...] | None = None) -> bool:
  if not path.exists():
    return False
  report = json.loads(path.read_text(encoding="utf-8"))
  if report.get("status") != "COMPLETE" or report.get("seed") != seed:
    raise RuntimeError(f"invalid existing receipt {path}")
  if methods is not None and tuple(report.get("methods", ())) != methods:
    raise RuntimeError(f"method mismatch in existing receipt {path}")
  return True


def run(command: list[str], log: Path) -> None:
  print("RUN", " ".join(command), flush=True)
  with log.open("w", encoding="utf-8") as stream:
    completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stream,
                               stderr=subprocess.STDOUT)
  if completed.returncode:
    raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--development-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
  args = parser.parse_args()
  if tuple(args.seeds) != DEFAULT_SEEDS:
    raise SystemExit(f"frozen held-out seeds are exactly {DEFAULT_SEEDS}")
  methods = ("M1", "M2-v1", "M2-v2-core", "M2-v2")
  checkpoints = {
    "M1": args.development_dir / "m1" / "best.pt",
    "M2-v1": args.development_dir / "m2" / "best.pt",
    "M2-v2-core": args.development_dir / "m2_v2_core" / "best.pt",
    "M2-v2": args.development_dir / "m2_v2" / "best.pt",
  }
  for label, path in checkpoints.items():
    if not path.exists():
      raise RuntimeError(f"missing frozen checkpoint for {label}: {path}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  complete = args.output_dir / "HELDOUT_GRADIENT_COMPLETE"
  if complete.exists():
    print("held-out gradient benchmark already complete")
    return

  receipts = []
  for seed in args.seeds:
    physical = args.output_dir / f"physical_rigid_stable_h32_seed{seed}.json"
    if not load_complete(physical, seed=seed):
      run([
        sys.executable, str(REPO_ROOT / "scripts" / "measure_physical_gradient.py"),
        "--scenario", "rigid_stable_stance", "--seed", str(seed),
        "--horizon", "32", "--output", str(physical),
      ], args.output_dir / f"physical_rigid_stable_h32_seed{seed}.log")
    world = args.output_dir / f"world_model_rigid_stable_h32_seed{seed}.json"
    if not load_complete(world, seed=seed, methods=methods):
      command = [
        sys.executable, str(REPO_ROOT / "scripts" / "measure_world_model_gradient.py"),
        "--scenario", "rigid_stable_stance", "--seed", str(seed),
        "--horizon", "32", "--physical", str(physical),
        "--device", args.device, "--output", str(world),
      ]
      for label, path in checkpoints.items():
        command.extend(("--model", f"{label}={path}"))
      run(command, args.output_dir / f"world_model_rigid_stable_h32_seed{seed}.log")
    receipts.append(world)

  aggregate = args.output_dir / "p1_heldout_gradient_aggregate_v2.json"
  if not aggregate.exists():
    run([
      sys.executable, str(REPO_ROOT / "scripts" / "aggregate_gradient_fidelity.py"),
      *map(str, receipts), "--output", str(aggregate),
    ], args.output_dir / "p1_heldout_gradient_aggregate_v2.log")
  protocol = {
    "schema": 1,
    "status": "COMPLETE",
    "purpose": "one-time P1 held-out gradient gate; never model selection",
    "scenario": "rigid_stable_stance",
    "backend": "MuJoCo-Warp float32 CPU physical FD; CUDA forbidden by P0 Gate A",
    "horizon": 32,
    "seeds": list(args.seeds),
    "epsilons": [0.01, 0.003, 0.001],
    "methods": {
      label: {"path": str(path.resolve()), "sha256": sha256(path)}
      for label, path in checkpoints.items()
    },
    "aggregate": str(aggregate.resolve()),
    "aggregate_sha256": sha256(aggregate),
  }
  (args.output_dir / "p1_heldout_gradient_manifest_v2.json").write_text(
    json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
  )
  complete.write_text("one-time P1 held-out gradient benchmark complete\n",
                      encoding="utf-8")


if __name__ == "__main__":
  main()
