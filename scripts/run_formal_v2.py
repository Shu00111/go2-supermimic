#!/usr/bin/env python3
"""Run the frozen P2 method/seed matrix sequentially and restart safely."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  args = parser.parse_args()
  run_root = args.run_root.resolve()
  training_root = run_root / "training"
  runtime = run_root / "runtime"
  training_root.mkdir(parents=True, exist_ok=True)
  runtime.mkdir(parents=True, exist_ok=True)

  contract_path = ROOT / "configs/formal_v2.json"
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  expected_methods = ["m1", "m2", "m2_v2", "m2_v2_nomemory", "m3"]
  expected_seeds = [14011, 24017, 34019]
  if contract["methods"] != expected_methods or contract["seeds"] != expected_seeds:
    raise SystemExit("unexpected P2 method/seed contract")
  if contract["num_envs"] != 1024 or contract["stage_iterations"] != {
    "stage1": 4000, "stage2": 4000, "stage3": 4000,
  }:
    raise SystemExit("unexpected P2 training budget")
  dirty = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
  ).strip()
  if dirty:
    raise SystemExit("refusing formal P2 training from a dirty tracked worktree")
  source_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
  ).strip()
  manifest_path = runtime / "formal_manifest.json"
  manifest = {
    "schema": 1,
    "status": "running",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha": source_sha,
    "contract": str(contract_path.resolve()),
    "contract_sha256": sha256(contract_path),
    "completed": [],
    "active": None,
  }
  if manifest_path.exists():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("source_sha") != source_sha or previous.get("contract_sha256") != manifest["contract_sha256"]:
      raise SystemExit("existing formal run is bound to different source or contract")
    manifest["created_utc"] = previous.get("created_utc", manifest["created_utc"])
    manifest["completed"] = previous.get("completed", [])
  (runtime / "launcher.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
  (runtime / "status").write_text("running\n", encoding="utf-8")

  env_base = os.environ.copy()
  env_base.update(
    PYTHONNOUSERSITE="1",
    PYTHONPATH=str(ROOT) + (os.pathsep + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else ""),
    OUTPUT_ROOT=str(training_root),
    NUM_ENVS=str(contract["num_envs"]),
    STAGE1_ITERATIONS="4000",
    STAGE2_ITERATIONS="4000",
    STAGE3_ITERATIONS="4000",
  )
  try:
    for seed in contract["seeds"]:
      for method in contract["methods"]:
        key = f"seed{seed}/{method}"
        manifest["active"] = key
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        env = env_base | {"SEED": str(seed)}
        subprocess.run(
          [sys.executable, "scripts/run_curriculum.py", "--method", method, "--through-stage", "3"],
          cwd=ROOT, env=env, check=True,
        )
        if key not in manifest["completed"]:
          manifest["completed"].append(key)
        manifest["active"] = None
        write_json(manifest_path, manifest)
  except BaseException:
    manifest["status"] = "failed_or_interrupted"
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    (runtime / "status").write_text("failed_or_interrupted\n", encoding="utf-8")
    raise
  manifest["status"] = "complete"
  manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
  write_json(manifest_path, manifest)
  (runtime / "status").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
  main()
