from pathlib import Path
import importlib.util
import os
import subprocess
import sys

import torch

from go2_supermimic.supermimic.contact_model import ContactStructuredWorldModelV2


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_world_model_gradient.py"
SPEC = importlib.util.spec_from_file_location("measure_world_model_gradient", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gradient_loader_accepts_controlled_m2_v2_checkpoint(tmp_path):
  model = ContactStructuredWorldModelV2((32, 32), (128, 128))
  checkpoint = tmp_path / "m2_v2.pt"
  torch.save({
    "cfg": {"method": "m2_v2", "hidden": [32, 32]},
    "world_model": model.state_dict(),
  }, checkpoint)
  method, loaded = MODULE.load_world_model(checkpoint, "cpu")
  assert method == "m2_v2"
  for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
    torch.testing.assert_close(expected, actual)


def test_gradient_entrypoints_bind_repository_when_launched_by_absolute_path(tmp_path):
  env = {**os.environ, "PYTHONNOUSERSITE": "1"}
  for name in ("measure_physical_gradient.py", "measure_world_model_gradient.py"):
    completed = subprocess.run(
      [sys.executable, str(SCRIPT.parent / name), "--help"], cwd=tmp_path,
      env=env, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
