import torch

from go2_supermimic.supermimic.diagnostics import central_difference, gradient_agreement


def test_finite_difference_matches_autograd_for_smooth_objective():
  action = torch.tensor([[0.2, -0.3]], requires_grad=True)
  value = (action.square()).sum()
  predicted = torch.autograd.grad(value, action)[0]
  fd = central_difference(lambda a: a.square().sum(), action.detach(), (1e-2, 1e-3))
  for epsilon, estimate in fd.items():
    report = gradient_agreement(predicted, estimate, epsilon)
    assert report.cosine_similarity > 0.9999
    assert report.relative_error < 1e-4
    assert report.sign_agreement == 1.0
