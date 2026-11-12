#!/usr/bin/env python3
"""Protocol guard for simulator-specific paired gradient capture.

The physical capture must restore an identical serialized simulator snapshot
before every +/- perturbation.  This guard records the required conventions;
integration with a validated GPU mjlab snapshot API is intentionally fail-closed.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--confirm-snapshot-restore", action="store_true")
args = ap.parse_args()
if not args.confirm_snapshot_restore:
  raise SystemExit("refusing capture without confirmed simulator+soil-memory snapshot restore")
if args.output.exists():
  raise SystemExit(f"refusing to overwrite {args.output}")
receipt = {"status": "BLOCKED", "reason": "validated mjlab snapshot/restore adapter not implemented",
           "epsilons": [0.01, 0.003, 0.001], "terminal_handling": "paired censoring",
           "future_action_convention": "hold nominal future actions fixed",
           "random_numbers": "restore RNG snapshot for each pair"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(receipt["reason"])
