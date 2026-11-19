"""Numerical primitives for contact-memory gradient diagnostics.

This module is deliberately independent of mjlab.  It operates on frozen
PyTorch world models and recorded tensors, so the sensitive calculations can
run in CPU float64.  None of the helpers changes model parameters or training
objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET


@dataclass(frozen=True)
class ExtendedStateLayout:
  """Slices for the Markov state used by an offline learned rollout."""

  physical: slice
  contact: slice | None
  previous_action: slice

  @property
  def width(self) -> int:
    return self.previous_action.stop


def extended_state_layout(contact_structured: bool) -> ExtendedStateLayout:
  contact_width = NUM_FEET * CONTACT_DIM if contact_structured else 0
  return ExtendedStateLayout(
    physical=slice(0, st.STATE_DIM),
    contact=(slice(st.STATE_DIM, st.STATE_DIM + contact_width)
             if contact_structured else None),
    previous_action=slice(st.STATE_DIM + contact_width,
                          st.STATE_DIM + contact_width + st.NUM_JOINTS),
  )


def detach_loading_memory(z: torch.Tensor) -> torch.Tensor:
  """Preserve contact values while cutting only the memory-channel history."""

  parts = [z[..., :4], z[..., 4:5].detach(), z[..., 5:]]
  return torch.cat(parts, dim=-1)


def contact_step_with_intervention(model, s, z, a, a_prev, xi, relief,
                                   *, detach_memory: bool = False):
  """Run the original forward path with an optional backward-only cut.

  The detached tensor has exactly the same value, dtype and shape.  Contact
  force, intrusion, tangent and probability channels retain their gradients.
  """

  z_input = detach_loading_memory(z) if detach_memory else z
  return model.step(s, z_input, a, a_prev, xi, relief)


def flatten_grads(parameters: Iterable[torch.nn.Parameter], *, allow_none: bool = True):
  chunks = []
  for parameter in parameters:
    if parameter.grad is None:
      if not allow_none:
        raise RuntimeError("parameter gradient is missing")
      chunks.append(torch.zeros_like(parameter).reshape(-1))
    else:
      chunks.append(parameter.grad.reshape(-1))
  return torch.cat(chunks) if chunks else torch.empty(0)


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-15) -> float:
  denom = a.norm() * b.norm()
  if not torch.isfinite(denom) or denom <= eps:
    return float("nan")
  # Roundoff in float32 can otherwise report impossible values such as 1.00002.
  return float((torch.dot(a, b) / denom).clamp(-1.0, 1.0))


@dataclass(frozen=True)
class JvpVjpCheck:
  lhs: float
  rhs: float
  absolute_error: float
  relative_error: float


def jvp_vjp_identity(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                     v: torch.Tensor, w: torch.Tensor) -> JvpVjpCheck:
  """Check <Jv,w> = <v,J^T w> without materializing J."""

  _, jv = torch.func.jvp(fn, (x,), (v,))
  _, pullback = torch.func.vjp(fn, x)
  jtw = pullback(w)[0]
  lhs = torch.dot(jv.reshape(-1), w.reshape(-1))
  rhs = torch.dot(v.reshape(-1), jtw.reshape(-1))
  absolute = (lhs - rhs).abs()
  scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(x.dtype).eps)
  return JvpVjpCheck(float(lhs.detach()), float(rhs.detach()), float(absolute.detach()),
                     float((absolute / scale).detach()))


def spectral_norm_matrix_free(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                              *, iterations: int = 20, seed: int = 0) -> float:
  """Estimate the largest singular value of d(fn)/dx by power iteration."""

  generator = torch.Generator(device=x.device).manual_seed(seed)
  v = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
  v = v / v.norm().clamp_min(torch.finfo(x.dtype).eps)
  sigma = x.new_zeros(())
  for _ in range(iterations):
    _, jv = torch.func.jvp(fn, (x,), (v,))
    sigma = jv.norm()
    if not torch.isfinite(sigma) or sigma <= torch.finfo(x.dtype).eps:
      return float(sigma.detach())
    w = jv / sigma
    _, pullback = torch.func.vjp(fn, x)
    v = pullback(w)[0]
    norm = v.norm()
    if not torch.isfinite(norm) or norm <= torch.finfo(x.dtype).eps:
      return float(norm.detach())
    v = v / norm
  _, jv = torch.func.jvp(fn, (x,), (v,))
  return float(jv.norm().detach())


def finite_difference_directional(fn: Callable[[torch.Tensor], torch.Tensor],
                                  x: torch.Tensor, direction: torch.Tensor,
                                  epsilons: Iterable[float]) -> dict[float, torch.Tensor]:
  """Centered directional derivatives at multiple perturbation magnitudes."""

  return {
    float(epsilon): (fn(x + epsilon * direction) - fn(x - epsilon * direction))
    / (2.0 * epsilon)
    for epsilon in epsilons
  }


def nonfinite_count(*tensors: torch.Tensor) -> int:
  return sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors)
