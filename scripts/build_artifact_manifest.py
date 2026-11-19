#!/usr/bin/env python3
"""Generate a deterministic SHA-256/size manifest for a materialized bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.diagnostics.frozen_artifacts import sha256_file


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("bundle", type=Path)
  args = parser.parse_args()
  root = args.bundle.resolve(strict=True)
  manifest_path = root / "manifest.json"
  files = []
  for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest_path):
    files.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                  "sha256": sha256_file(path)})
  payload = {
    "schema": 1, "bundle": root.name, "status": "PARTIAL_WITH_DOCUMENTED_BLOCKERS",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip(),
    "authoritative_training_source": "23acf7c1ae990b14b5dd5030e6e852e05d129edb",
    "frozen_research_commit": "f7e61b6954e7a706e9aeb1f5d15ccbec0896ca17",
    "files": files,
  }
  manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({"manifest": str(manifest_path), "files": len(files)}, indent=2))


if __name__ == "__main__":
  main()
