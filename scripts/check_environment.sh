#!/usr/bin/env bash
set -euo pipefail
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi unavailable" >&2; exit 2; }
command -v python >/dev/null || { echo "ERROR: python unavailable" >&2; exit 2; }
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version --format=csv
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is false")
major, minor = torch.cuda.get_device_capability(0)
print({"python": sys.version.split()[0], "torch": torch.__version__,
       "torch_cuda": torch.version.cuda, "capability": [major, minor]})
if major >= 12 and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 7):
    raise SystemExit("ERROR: Blackwell requires a PyTorch build with sm_120 support (use >=2.7)")
x = torch.randn(64, 64, device="cuda")
y = x @ x
assert torch.isfinite(y).all()
print("CUDA finite matmul: PASS")
PY
required_gb="${MIN_FREE_DISK_GB:-100}"
available_kb="$(df -Pk "${OUTPUT_ROOT:-.}" | awk 'NR==2 {print $4}')"
if (( available_kb < required_gb * 1024 * 1024 )); then
  echo "ERROR: less than ${required_gb} GiB free at OUTPUT_ROOT" >&2
  exit 2
fi
