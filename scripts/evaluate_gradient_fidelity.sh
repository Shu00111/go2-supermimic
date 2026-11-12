#!/usr/bin/env bash
set -euo pipefail
exec python "$(dirname "$0")/evaluate_gradient_fidelity.py" "$@"
