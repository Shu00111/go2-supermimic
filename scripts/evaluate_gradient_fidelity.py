#!/usr/bin/env python3
"""Score frozen WM autograd and paired simulator finite-difference gradients."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2_supermimic.supermimic.diagnostics import gradient_agreement

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True,
                help="NPZ containing g_wm and g_fd_<epsilon>; generation is simulator-specific")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
if args.output.exists():
  raise SystemExit(f"refusing to overwrite {args.output}")
data = np.load(args.input, allow_pickle=False)
if "g_wm" not in data.files:
  raise SystemExit("missing g_wm")
pred = torch.from_numpy(data["g_wm"])
reports = []
for name in sorted(n for n in data.files if n.startswith("g_fd_")):
  epsilon = float(name.removeprefix("g_fd_").replace("p", "."))
  reports.append(gradient_agreement(pred, torch.from_numpy(data[name]), epsilon).__dict__)
if len(reports) < 3:
  raise SystemExit("at least three finite-difference step sizes are required")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
