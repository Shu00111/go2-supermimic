from types import SimpleNamespace

import torch

from go2_supermimic.diagnostics.gradient_objective import (
  common_objective,
  finite_difference_convergence,
)


class FakeImagination:
  def __init__(self):
    self.cfg = SimpleNamespace(
      gamma=0.5, dt=0.02, weights=SimpleNamespace(terminated=-10.0)
    )

  def _rewards(self, s0, s1, pose1, gst, action, action_prev, constants):
    del s0, pose1, gst, action_prev, constants
    return s1[:, 0] + action[:, 0], {}

  def fall_prob(self, state):
    return state[:, 1]


def test_common_objective_has_soft_survival_and_no_endpoint_value():
  # Rewards: (1 + 2) - .2*.5 = 2.9, then (3 + 4) - .2*.25 = 6.95.
  # Objective: 2.9 + gamma*.5*6.95 = 4.6375.
  states = torch.tensor([[[0.0, 0.0], [1.0, 0.5], [3.0, 0.25]]])
  poses = torch.zeros(1, 3, 3)
  actions = torch.tensor([[[2.0], [4.0]]])
  result = common_objective(
    FakeImagination(), states, poses, actions, torch.zeros_like(actions),
    {"dummy": torch.zeros(1, 2, 1)}, {},
  )
  torch.testing.assert_close(result.rewards, torch.tensor([[2.9, 6.95]]))
  torch.testing.assert_close(result.survival, torch.tensor([[1.0, 0.5, 0.375]]))
  torch.testing.assert_close(result.value, torch.tensor([4.6375]))


def test_common_objective_rejects_mismatched_trace():
  states = torch.zeros(1, 2, 2)
  poses = torch.zeros(1, 3, 3)
  actions = torch.zeros(1, 2, 1)
  try:
    common_objective(
      FakeImagination(), states, poses, actions, torch.zeros_like(actions),
      {"dummy": torch.zeros(1, 2, 1)}, {},
    )
  except ValueError as exc:
    assert "H+1" in str(exc)
  else:
    raise AssertionError("mismatched trace was accepted")


def test_finite_difference_convergence_censors_sign_flip():
  rows = []
  for epsilon, gradient in ((0.01, 1.0), (0.003, 1.1), (0.001, -0.8)):
    rows.append({
      "epsilon": epsilon, "coordinate": 0, "centered_gradient": gradient,
      "perturbation_signal": 2.0 * epsilon * abs(gradient),
      "action_saturated": False, "nonfinite": False,
      "branch_discontinuity": {}, "nominal_discontinuity": {},
    })
  result = finite_difference_convergence(rows, objective=1.0)
  assert not result[0]["sign_consistent"]
  assert not result[0]["valid"]
