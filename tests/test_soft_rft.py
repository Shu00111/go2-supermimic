"""Parity and state-isolation gates for the vendored validated RFT model."""

import numpy as np
import pytest
import torch

from go2_supermimic.soft_terrain.rft import RftOperator, RftState


POSITIONS = torch.tensor([
  [0.0, 0.0, 0.015], [0.0, 0.0, 0.010],
  [0.0, 0.0, 0.019], [0.0, 0.0, 0.050],
], dtype=torch.float64)
VELOCITIES = torch.tensor([
  [0.0, 0.0, -0.1], [0.2, -0.1, -0.05],
  [-0.3, 0.15, 0.02], [0.1, 0.2, -0.3],
], dtype=torch.float64)
ANGULAR = torch.tensor([
  [0.0, 0.0, 0.0], [1.0, -2.0, 3.0],
  [-2.0, 0.5, -1.0], [4.0, 5.0, 6.0],
], dtype=torch.float64)
PLASTIC = torch.tensor([0.007, 0.015, 0.012, 0.0], dtype=torch.float64)

# Generated directly by Go2-LunarSoftTerrain master@08568c617ed5c40a4b8969b09fa4b9b847772b85
# using its NumPy compute_force reference, Quikrete play sand, Earth gravity,
# radius 0.022 m and subdivisions=1.
EXPECTED_FORCE = np.array([
  [-6.245004513516506e-17, -2.7755575615628914e-17, 10.518095156485325],
  [-5.060303858753693, 1.2564751282741453, 13.028372063852574],
  [0.1835309871986443, -0.06856875109053755, 0.0],
  [0.0, 0.0, 0.0],
])
EXPECTED_TORQUE = np.array([
  [0.0, -4.336808689942018e-19, 0.0],
  [0.0067655544833034155, 0.06900562207984046, -0.00587032989196781],
  [-0.001412601042419102, -0.003781042367844105, -0.00012991349311645896],
  [0.0, 0.0, 0.0],
])


def _compute(device="cpu", dtype=torch.float64):
  op = RftOperator(device=device, dtype=dtype, subdivisions=1)
  return op.compute(
    POSITIONS.to(device=device, dtype=dtype),
    VELOCITIES.to(device=device, dtype=dtype),
    ANGULAR.to(device=device, dtype=dtype),
    plastic_depth=PLASTIC.to(device=device, dtype=dtype),
  )


def test_float64_matches_validated_numpy_fixtures():
  result = _compute()
  np.testing.assert_allclose(result.force.numpy(), EXPECTED_FORCE, rtol=2e-12, atol=2e-12)
  np.testing.assert_allclose(result.torque.numpy(), EXPECTED_TORQUE, rtol=2e-12, atol=2e-12)
  np.testing.assert_array_equal(result.n_immersed.numpy(), [12, 22, 6, 0])
  np.testing.assert_array_equal(result.n_active.numpy(), [12, 22, 6, 0])
  assert torch.isfinite(result.force).all()
  assert torch.isfinite(result.torque).all()
  assert result.force[-1].count_nonzero() == 0  # release above the surface


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float32_matches_cpu_and_is_finite():
  cpu = _compute(dtype=torch.float32)
  gpu = _compute(device="cuda", dtype=torch.float32)
  torch.testing.assert_close(gpu.force.cpu(), cpu.force, rtol=3e-5, atol=3e-5)
  torch.testing.assert_close(gpu.torque.cpu(), cpu.torque, rtol=3e-5, atol=3e-5)
  assert torch.isfinite(gpu.power).all()


def test_loading_memory_isolated_by_environment_and_foot():
  state = RftState(3, 4, device="cpu", dtype=torch.float64)
  position = torch.zeros((3, 4, 3), dtype=torch.float64)
  position[..., 2] = 0.04  # airborne
  position[1, 2, 2] = 0.012
  state.update(position, 0.0, 0.022)
  assert state.plastic_depth.count_nonzero() == 1
  assert state.plastic_depth[1, 2] == pytest.approx(0.010)
  state.reset(torch.tensor([0, 2]))
  assert state.plastic_depth[1, 2] == pytest.approx(0.010)
  position[1, 2, 2] = 0.04
  state.update(position, 0.0, 0.022)
  assert state.plastic_depth.count_nonzero() == 0


def test_strength_multiplier_is_linear_for_static_loading():
  op = RftOperator(device="cpu", dtype=torch.float64, subdivisions=1)
  position = torch.zeros((2, 1, 3), dtype=torch.float64)
  position[..., 2] = 0.015
  zeros = torch.zeros_like(position)
  result = op.compute(
    position, zeros, zeros, coefficient_scale=torch.tensor([[0.3], [0.6]])
  )
  assert (result.force[1, 0, 2] / result.force[0, 0, 2]).item() == pytest.approx(2.0)
