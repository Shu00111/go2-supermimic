import torch

from go2_supermimic.diagnostics.contact_memory import (
  contact_step_with_intervention,
  detach_loading_memory,
  finite_difference_directional,
  jvp_vjp_identity,
  spectral_norm_matrix_free,
)
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  CONTACT_DIM, NUM_FEET, ContactStructuredWorldModelV2,
)
from go2_supermimic.target.terrain import SCAN_POINTS


def test_backward_only_memory_cut_preserves_forward_and_other_channel_gradients():
  torch.manual_seed(4)
  dtype = torch.float64
  model = ContactStructuredWorldModelV2((16,), (8,), anchor_weight=0.0).to(dtype)
  s = torch.randn(1, st.STATE_DIM, dtype=dtype) * 0.01
  a = torch.randn(1, st.NUM_JOINTS, dtype=dtype) * 0.01
  xi = torch.zeros(1, st.CONTEXT_DIM, dtype=dtype)
  relief = torch.zeros(1, SCAN_POINTS, dtype=dtype)
  z = model.initial_contact(s, a)[0].detach().requires_grad_(True)
  regular = contact_step_with_intervention(model, s, z, a, a, xi, relief)
  blocked = contact_step_with_intervention(
    model, s, z, a, a, xi, relief, detach_memory=True,
  )
  for left, right in zip(regular, blocked):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
  blocked[0].sum().backward()
  assert z.grad[..., 4].count_nonzero() == 0
  assert z.grad[..., :4].abs().sum() + z.grad[..., 5:].abs().sum() > 0


def test_jvp_vjp_identity_and_matrix_free_spectral_norm():
  dtype = torch.float64
  matrix = torch.tensor([[3.0, 0.0], [0.0, -2.0]], dtype=dtype)
  fn = lambda value: matrix @ value
  x = torch.tensor([0.2, -0.4], dtype=dtype)
  v = torch.tensor([0.7, 0.1], dtype=dtype)
  w = torch.tensor([-0.3, 0.5], dtype=dtype)
  check = jvp_vjp_identity(fn, x, v, w)
  assert check.relative_error < 1e-14
  assert abs(spectral_norm_matrix_free(fn, x, iterations=12) - 3.0) < 1e-6


def test_finite_difference_directional_converges_on_smooth_fixture():
  dtype = torch.float64
  x = torch.tensor([0.2, -0.4], dtype=dtype)
  direction = torch.tensor([0.7, 0.1], dtype=dtype)
  fn = lambda value: value.sin()
  exact = x.cos() * direction
  estimates = finite_difference_directional(fn, x, direction, (1e-2, 1e-3, 1e-4))
  errors = [(estimate - exact).norm() for estimate in estimates.values()]
  assert errors[2] < errors[1] < errors[0]


def test_detach_loading_memory_keeps_values_and_shape():
  z = torch.randn(2, NUM_FEET, CONTACT_DIM, requires_grad=True)
  detached = detach_loading_memory(z)
  torch.testing.assert_close(detached, z)
  assert detached.shape == z.shape
