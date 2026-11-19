#!/usr/bin/env python3
"""Fail-closed SHA-256 verification for a handoff manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import sha256_file


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("manifest", type=Path)
  args = parser.parse_args()
  manifest_path = args.manifest.resolve(strict=True)
  manifest = json.loads(manifest_path.read_text())
  failures = []
  for item in manifest["files"]:
    path = manifest_path.parent / item["path"]
    if not path.is_file():
      failures.append({"path": item["path"], "error": "missing"}); continue
    actual = sha256_file(path)
    if actual != item["sha256"] or path.stat().st_size != item["size_bytes"]:
      failures.append({"path": item["path"], "error": "hash_or_size_mismatch", "actual": actual})
  result = {"status": "PASS" if not failures else "FAIL", "checked": len(manifest["files"]),
            "failures": failures}
  print(json.dumps(result, indent=2))
  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
