"""Offline prediction and gradient-fidelity diagnostics (never used by training)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class GradientAgreement:
  epsilon: float
  cosine_similarity: float
  relative_error: float
  sign_agreement: float


def gradient_agreement(predicted: torch.Tensor, finite_difference: torch.Tensor,
                       epsilon: float) -> GradientAgreement:
  predicted = predicted.reshape(-1).double()
  finite_difference = finite_difference.reshape(-1).double()
  denom = predicted.norm() * finite_difference.norm()
  cosine = (predicted @ finite_difference / denom.clamp_min(1e-12)).item()
  relative = ((predicted - finite_difference).norm() /
              finite_difference.norm().clamp_min(1e-12)).item()
  active = (predicted.abs() + finite_difference.abs()) > 1e-10
  signs = ((predicted.sign() == finite_difference.sign()) & active).sum()
  sign = (signs / active.sum().clamp_min(1)).item()
  return GradientAgreement(epsilon, cosine, relative, sign)


def central_difference(
  objective: Callable[[torch.Tensor], torch.Tensor],
  action: torch.Tensor,
  epsilons: tuple[float, ...] = (1e-2, 3e-3, 1e-3),
) -> dict[float, torch.Tensor]:
  """Coordinate central differences with all external simulator state held by caller."""
  flat = action.detach().reshape(-1)
  results = {}
  for epsilon in epsilons:
    values = []
    for i in range(flat.numel()):
      basis = torch.zeros_like(flat)
      basis[i] = epsilon
      plus = objective((flat + basis).reshape_as(action)).detach()
      minus = objective((flat - basis).reshape_as(action)).detach()
      values.append((plus - minus) / (2 * epsilon))
    results[epsilon] = torch.stack(values).reshape_as(action)
  return results


def prediction_errors(pred: torch.Tensor, target: torch.Tensor,
                      strata: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
  """RMSE over identical robot-state channels, optionally stratified by masks."""
  if pred.shape != target.shape:
    raise ValueError("prediction and target shapes differ")
  sq = (pred - target).square().mean(-1)
  output = {"all/rmse": float(sq.mean().sqrt())}
  for name, mask in (strata or {}).items():
    mask = mask.bool()
    output[f"{name}/rmse"] = float(sq[mask].mean().sqrt()) if mask.any() else float("nan")
  return output
