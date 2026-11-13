#!/usr/bin/env python3
"""Render representative real MuJoCo-Warp rollouts for frozen formal policies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=15.0)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  training = ROOT / "outputs/contact_structured_supermimic/formal_seed14011/seed14011"
  env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for method in ("m1", "m2"):
    matches = list((training / method).glob("stage3/*/*/model_12000.pt"))
    if len(matches) != 1:
      raise RuntimeError(f"expected one final {method} checkpoint, found {matches}")
    video = args.output_dir / f"{method}_alternating_material_ood.mp4"
    if video.exists() and video.stat().st_size > 0:
      print(f"skip existing {video}", flush=True)
      continue
    command = [
      sys.executable, str(ROOT / "scripts/eval.py"),
      "Go2-Chase-AlternatingMaterial-Scan", "--checkpoint", str(matches[0]),
      "--num-envs", "1", "--seconds", str(args.seconds), "--seed", "90301",
      "--soil-scale", "0.2", "--mode", "chase", "--video", str(video),
      "--width", "960", "--height", "540", "--device", "cuda:0",
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
  main()
