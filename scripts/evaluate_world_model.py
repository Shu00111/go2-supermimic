#!/usr/bin/env python3
"""Evaluate frozen one/multi-step predictions from an exported NPZ batch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2_supermimic.supermimic.diagnostics import prediction_errors

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", type=Path, required=True,
                help="NPZ with state_pred, state_target and optional boolean stratum_* arrays")
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
if args.output.exists():
  raise SystemExit(f"refusing to overwrite {args.output}")
data = np.load(args.predictions, allow_pickle=False)
required = {"state_pred", "state_target"}
if not required.issubset(data.files):
  raise SystemExit(f"missing arrays: {sorted(required - set(data.files))}")
strata = {name.removeprefix("stratum_"): torch.from_numpy(data[name])
          for name in data.files if name.startswith("stratum_")}
metrics = prediction_errors(torch.from_numpy(data["state_pred"]),
                            torch.from_numpy(data["state_target"]), strata)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
print(json.dumps(metrics, indent=2, allow_nan=True))
