#!/usr/bin/env python3
"""Verify authoritative checkpoints and emit a machine-readable manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import (
  EXPECTED_SHA256, checkpoint_manifest, load_frozen_checkpoint, write_json,
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", action="append", required=True,
                      help="METHOD=/absolute/path/to/model_12000.pt")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--skip-rehash", action="store_true",
                      help="Only for a second pass after hashes were verified in this session")
  args = parser.parse_args()
  supplied = dict(item.split("=", 1) for item in args.checkpoint)
  if set(supplied) != set(EXPECTED_SHA256):
    raise SystemExit(f"exactly these methods are required: {sorted(EXPECTED_SHA256)}")
  rows = []
  for method in EXPECTED_SHA256:
    checkpoint = load_frozen_checkpoint(
      method, Path(supplied[method]), verify_hash=not args.skip_rehash,
    )
    rows.append(checkpoint_manifest(checkpoint))
    del checkpoint
  manifest = {"schema": 1, "status": "PASS", "checkpoints": rows}
  write_json(args.output, manifest)
  print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
  main()
