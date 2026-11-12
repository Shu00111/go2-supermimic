#!/usr/bin/env bash
set -euo pipefail
exec python "$(dirname "$0")/evaluate_world_model.py" "$@"
