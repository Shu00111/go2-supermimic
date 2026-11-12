#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
scripts/check_environment.sh
exec python scripts/run_config.py --method m3
