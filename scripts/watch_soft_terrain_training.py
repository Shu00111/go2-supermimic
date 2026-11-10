#!/usr/bin/env python3
"""Local no-model watchdog for the detached seed-14011 training process."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path) -> str:
  try:
    return path.read_text().strip()
  except OSError:
    return ""


def process_alive(pid: int) -> bool:
  if pid <= 0:
    return False
  try:
    os.kill(pid, 0)
    return True
  except OSError:
    return False


def gpu_status() -> dict[str, object]:
  command = [
    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
  ]
  try:
    fields = subprocess.check_output(command, text=True, timeout=10).strip().split(", ")
    return {
      "utilization_percent": int(fields[0]), "memory_used_mib": int(fields[1]),
      "memory_total_mib": int(fields[2]), "temperature_c": int(fields[3]),
    }
  except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
    return {"error": str(exc)}


def snapshot(run_root: Path) -> dict[str, object]:
  runtime = run_root / "runtime"
  pid_text = read_text(runtime / "launcher.pid")
  pid = int(pid_text) if pid_text.isdigit() else 0
  checkpoints = list((run_root / "training").glob("**/model_*.pt"))
  def iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1
  latest = max(checkpoints, key=iteration, default=None)
  now = time.time()
  return {
    "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    "launcher_pid": pid or None,
    "process_alive": process_alive(pid),
    "status": read_text(runtime / "status") or "unknown",
    "last_iteration": iteration(latest) if latest else 0,
    "latest_checkpoint": str(latest) if latest else None,
    "checkpoint_age_minutes": round((now - latest.stat().st_mtime) / 60, 2) if latest else None,
    "gpu": gpu_status(),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--interval-minutes", type=float, default=40.0)
  parser.add_argument("--once", action="store_true")
  args = parser.parse_args()
  runtime = args.run_root / "runtime"
  runtime.mkdir(parents=True, exist_ok=True)
  output = runtime / "watchdog.jsonl"
  while True:
    state = snapshot(args.run_root)
    with output.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(state, sort_keys=True) + "\n")
    (runtime / "watchdog.latest.json").write_text(
      json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, sort_keys=True), flush=True)
    if args.once or state["status"] == "complete":
      return
    time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
  main()
