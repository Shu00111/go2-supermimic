#!/usr/bin/env python3
"""Fail-closed launcher that records provenance then delegates to train.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--method", choices=("m1", "m2", "m2_nomemory", "m3"), required=True)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  cfg = load(ROOT / "configs/common.json") | load(ROOT / f"configs/methods/{args.method}.json")
  cfg["seed"] = int(os.environ.get("SEED", cfg["seed"]))
  cfg["iterations"] = int(os.environ.get("TRAINING_ITERATIONS", cfg["iterations"]))
  cfg["num_envs"] = int(os.environ.get("NUM_ENVS", cfg["num_envs"]))
  cfg["task"] = os.environ.get("TASK", "Go2-Chase-SoftTerrain")
  resume = os.environ.get("RESUME", "")
  if resume:
    if not Path(resume).is_file():
      raise SystemExit(f"RESUME checkpoint not found: {resume}")
    cfg["resume"] = str(Path(resume).resolve())
    cfg["continue_run"] = True
  source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  output_root = Path(os.environ.get("OUTPUT_ROOT", ROOT / "runs")).resolve()
  run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  run_dir = output_root / f"{args.method}-seed{cfg['seed']}-{run_id}"
  if run_dir.exists():
    raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
  cfg["log_root"] = str(run_dir)
  dirty = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
  ).strip()
  if dirty and os.environ.get("ALLOW_DIRTY_SOURCE") != "1":
    raise SystemExit("refusing to train from a dirty tracked worktree")
  canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
  config_hash = hashlib.sha256(canonical).hexdigest()
  if args.dry_run:
    print(json.dumps({"run_dir": str(run_dir), "source_sha": source_sha,
                      "config_sha256": config_hash, "config": cfg}, indent=2))
    return
  run_dir.mkdir(parents=True)
  manifest = {
    "schema": 1,
    "status": "starting",
    "source_sha": source_sha,
    "config_sha256": config_hash,
    "config": cfg,
    "argv": sys.argv,
    "created_utc": datetime.now(timezone.utc).isoformat(),
  }
  manifest_path = run_dir / "run_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  command = [sys.executable, str(ROOT / "scripts/train.py")]
  for key, value in cfg.items():
    flag = "--" + key.replace("_", "-")
    if isinstance(value, list):
      command.extend([flag, *map(str, value)])
    elif isinstance(value, bool):
      command.extend([flag, str(value).lower()])
    else:
      command.extend([flag, str(value)])
  try:
    subprocess.run(command, cwd=ROOT, check=True)
  except BaseException:
    manifest["status"] = "failed_or_interrupted"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    raise
  manifest["status"] = "completed"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
