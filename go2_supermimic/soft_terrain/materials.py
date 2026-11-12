"""Deterministic, vectorized per-foot hard/soft material maps."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class CheckerboardMaterial:
  """Environment-local checkerboard with deterministic boundary semantics.

  A cell is soft when ``(ix + iy)`` is odd.  Cell boundaries belong to the
  rigid material. The signed clearance additionally accounts for the spherical
  foot radius: RFT is active only when the complete horizontal footprint lies
  inside a soft cell.  This prevents simultaneous RFT and contact with a
  neighbouring rigid patch at an edge.
  """

  cell_size: float = 0.6
  seed: int = 7
  foot_radius: float = 0.022

  def __post_init__(self) -> None:
    if self.cell_size <= 2 * self.foot_radius:
      raise ValueError("cell_size must be larger than the foot diameter")

  @property
  def phase(self) -> tuple[float, float]:
    rng = random.Random(self.seed)
    return tuple(rng.uniform(-0.5, 0.5) * self.cell_size for _ in range(2))

  def query(self, xy_w: torch.Tensor, env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(soft_mask, signed_clearance)`` for ``(N, ..., 2)`` points.

    Positive clearance means the whole foot footprint is inside soft material;
    zero and negative values are rigid by definition.
    """
    if xy_w.ndim < 3 or xy_w.shape[0] != env_origins.shape[0] or xy_w.shape[-1] != 2:
      raise ValueError("xy_w must have shape (num_envs, ..., 2)")
    phase = xy_w.new_tensor(self.phase)
    origin_shape = (xy_w.shape[0],) + (1,) * (xy_w.ndim - 2) + (2,)
    origins = env_origins[:, :2].reshape(origin_shape)
    local = xy_w - origins - phase
    scaled = local / self.cell_size
    cell = torch.floor(scaled).to(torch.int64)
    frac = scaled - cell.to(scaled.dtype)
    edge_distance = torch.minimum(frac, 1.0 - frac).amin(dim=-1) * self.cell_size
    soft_cell = (cell.sum(dim=-1).remainder(2) == 1)
    signed = torch.where(soft_cell, edge_distance, -edge_distance) - self.foot_radius
    return signed > 0.0, signed

  def rigid_cells(self, half_extent: float) -> tuple[tuple[float, float], ...]:
    """Rigid cell centres covering an environment-local square."""
    if half_extent <= 0:
      raise ValueError("half_extent must be positive")
    px, py = self.phase
    lo = int((-half_extent - max(abs(px), abs(py))) // self.cell_size) - 1
    hi = int((half_extent + max(abs(px), abs(py))) // self.cell_size) + 2
    cells = []
    for ix in range(lo, hi + 1):
      for iy in range(lo, hi + 1):
        if (ix + iy) % 2 == 0:
          cells.append(((ix + 0.5) * self.cell_size + px,
                        (iy + 0.5) * self.cell_size + py))
    return tuple(cells)
