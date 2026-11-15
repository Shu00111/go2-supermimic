"""Train a chase / dodge policy with SuperMimic.

Flat ground, from scratch:
  python scripts/train.py --task Go2-Chase-Flat --run-name flat

Obstacles or terrain, warm-started from a flat policy (see README.md for the
terrain recipe with the MoE-CTS and flat-policy data drivers):
  python scripts/train.py --task Go2-Chase-Obstacles --resume <flat model.pt> --run-name obstacles

Every TrainCfg field (go2_supermimic/supermimic/train.py) is a flag.
"""

from pathlib import Path
import sys

# Editable installs on research hosts may point at an older checkout.  Formal
# runs must import the source tree containing this launcher.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from go2_supermimic.supermimic.train import main

if __name__ == "__main__":
  torch.set_float32_matmul_precision("high")
  main()
