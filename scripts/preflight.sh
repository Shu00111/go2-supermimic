#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
python -m pytest -q tests/test_contact_structured_model.py tests/test_diagnostics.py tests/test_soft_rft.py
if [[ "${ALLOW_CPU_PREFLIGHT:-0}" == "1" ]]; then
  echo "CPU-only preflight requested; CUDA execution remains unverified"
else
  scripts/check_environment.sh
fi
for method in m1 m2 m3; do
  python scripts/run_config.py --method "$method" --dry-run >/dev/null
done
echo "preflight: PASS"
