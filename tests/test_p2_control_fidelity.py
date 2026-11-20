import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
  path = ROOT / "scripts" / name
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_action_jacobian_metrics_accept_physics_and_wm_precision():
  module = load_script("run_action_jacobian_fidelity.py")
  physical = torch.eye(2, dtype=torch.float32)
  learned = torch.eye(2, dtype=torch.float64)
  metrics = module.matrix_metrics(physical, learned)
  assert metrics["cosine"] == 1.0
  assert metrics["relative_error"] == 0.0
  assert metrics["spectral_norm_ratio"] == 1.0


def test_distribution_metrics_respect_unclipped_action_contract():
  module = load_script("analyze_state_action_distribution_shift.py")
  state = torch.zeros(2, 78)
  state[1, 4] = 1.0
  values = {"state": state, "action": torch.full((2, 12), 2.0),
            "contact": torch.zeros(2, 32)}
  reference = (torch.zeros(78), torch.ones(78), -torch.ones(78), torch.ones(78),
               torch.zeros(12), torch.ones(12))
  result = module.metrics("test", values, reference)
  assert result["action_unit_exceedance_fraction"] == 1.0
  assert result["action_saturation_fraction"] == 0.0
  assert result["static_state_fraction_speed_lt_0_1"] == 0.5
