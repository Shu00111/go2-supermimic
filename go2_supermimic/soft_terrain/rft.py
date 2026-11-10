"""Batched spherical-foot granular RFT.

This is a small, self-contained port of the validated implementation in
Shu00111/Go2-LunarSoftTerrain at commit
08568c617ed5c40a4b8969b09fa4b9b847772b85.  The equations, integration mesh,
units and elasto-plastic memory match ``src/terramechanics/rft_torch.py`` in
that snapshot.  The nominal Earth material is the measured Quikrete play-sand
coefficient (zeta=3.75 N/cm^3); ``coefficient_scale`` is the source project's
soil-strength/softness multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

FOOT_RADIUS = 0.022
PLAY_SAND_ZETA_N_PER_CM3 = 3.75
GENERIC_COEFFICIENTS = (
  0.206, 0.169, 0.212, 0.358, 0.055,
  -0.124, 0.253, 0.007, 0.088,
)
N_PER_CM3_TO_N_PER_M3 = 1.0e6
YIELD_SPEED = 0.01
QUASI_STATIC_SPEED = 1.0e-6
LEADING_REGULARISATION = 0.05
PLASTIC_RATIO = 0.7


def _icosphere(radius: float, subdivisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return face centroids, outward normals and area for the source icosphere."""
  if radius <= 0.0:
    raise ValueError(f"radius must be positive, got {radius}")
  if subdivisions < 0:
    raise ValueError(f"subdivisions must be non-negative, got {subdivisions}")
  phi = (1.0 + np.sqrt(5.0)) / 2.0
  verts = []
  for s1 in (-1, 1):
    for s2 in (-1, 1):
      verts.append([0.0, s1, s2 * phi])
      verts.append([s1, s2 * phi, 0.0])
      verts.append([s1 * phi, 0.0, s2])
  verts = np.asarray(verts, dtype=np.float64)
  verts /= np.linalg.norm(verts, axis=1, keepdims=True)
  edge = min(
    np.linalg.norm(verts[i] - verts[j])
    for i in range(12) for j in range(i + 1, 12)
  )
  faces = []
  for i in range(12):
    for j in range(i + 1, 12):
      if abs(np.linalg.norm(verts[i] - verts[j]) - edge) > 1.0e-6:
        continue
      for k in range(j + 1, 12):
        if (
          abs(np.linalg.norm(verts[i] - verts[k]) - edge) < 1.0e-6
          and abs(np.linalg.norm(verts[j] - verts[k]) - edge) < 1.0e-6
        ):
          faces.append([i, j, k])
  faces = np.asarray(faces, dtype=np.int64)
  for _ in range(subdivisions):
    vertex_list = list(verts)
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
      key = (min(a, b), max(a, b))
      if key not in cache:
        point = (vertex_list[a] + vertex_list[b]) / 2.0
        point /= np.linalg.norm(point)
        vertex_list.append(point)
        cache[key] = len(vertex_list) - 1
      return cache[key]

    split = []
    for a, b, c in faces:
      ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
      split += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    verts = np.asarray(vertex_list, dtype=np.float64)
    faces = np.asarray(split, dtype=np.int64)
  v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
  flat_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
  normals = (v0 + v1 + v2) / 3.0
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  areas = flat_area * radius**2
  areas *= 4.0 * np.pi * radius**2 / areas.sum()
  return normals * radius, normals, areas


@dataclass
class RftResult:
  force: torch.Tensor
  torque: torch.Tensor
  n_immersed: torch.Tensor
  n_active: torch.Tensor
  max_depth: torch.Tensor
  immersed_area: torch.Tensor
  power: torch.Tensor


class RftOperator:
  """Dense CUDA-compatible RFT operator; leading dimensions are arbitrary."""

  def __init__(
    self,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    radius: float = FOOT_RADIUS,
    subdivisions: int = 1,
    zeta_n_per_cm3: float = PLAY_SAND_ZETA_N_PER_CM3,
    yield_speed: float = YIELD_SPEED,
    plastic_ratio: float = PLASTIC_RATIO,
  ):
    if yield_speed <= 0.0:
      raise ValueError("yield_speed must be positive")
    if not 0.0 <= plastic_ratio < 1.0:
      raise ValueError("plastic_ratio must be in [0, 1)")
    centroids, normals, areas = _icosphere(radius, subdivisions)
    self.device = torch.device(device)
    self.dtype = dtype
    self.radius = float(radius)
    self.num_faces = len(areas)
    self.yield_speed = float(yield_speed)
    self.plastic_ratio = float(plastic_ratio)
    self.centroids = torch.as_tensor(centroids, device=self.device, dtype=dtype)
    self.normals = torch.as_tensor(normals, device=self.device, dtype=dtype)
    self.areas = torch.as_tensor(areas, device=self.device, dtype=dtype)
    self.device = self.centroids.device
    scale = zeta_n_per_cm3 * N_PER_CM3_TO_N_PER_M3
    self.coefficients = torch.tensor(
      GENERIC_COEFFICIENTS, device=self.device, dtype=dtype
    ) * scale

  def _alpha_z(self, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    a00, a10, b11, b01, bm11 = self.coefficients[:5]
    return (a00 + a10 * torch.cos(2 * beta) + b11 * torch.sin(2 * beta + gamma)
            + b01 * torch.sin(gamma) + bm11 * torch.sin(-2 * beta + gamma))

  def _alpha_x(self, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    c11, c01, cm11, d10 = self.coefficients[5:]
    return (c11 * torch.cos(2 * beta + gamma) + c01 * torch.cos(gamma)
            + cm11 * torch.cos(-2 * beta + gamma) + d10 * torch.sin(2 * beta))

  def _angles(self, velocity: torch.Tensor, normal: torch.Tensor):
    horizontal = velocity.clone()
    horizontal[..., 2] = 0.0
    horizontal_speed = torch.linalg.vector_norm(horizontal, dim=-1)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    normal_horizontal = normal.expand_as(velocity).clone()
    normal_horizontal[..., 2] = 0.0
    normal_horizontal_norm = torch.linalg.vector_norm(normal_horizontal, dim=-1)
    numerical_scale = 1.0e-9 * torch.clamp_min(speed, QUASI_STATIC_SPEED)
    use_velocity = horizontal_speed > numerical_scale
    use_normal = (~use_velocity) & (normal_horizontal_norm > 1.0e-12)
    default = torch.zeros_like(velocity)
    default[..., 0] = 1.0
    velocity_direction = horizontal / torch.clamp_min(horizontal_speed[..., None], 1.0e-30)
    normal_direction = normal_horizontal / torch.clamp_min(
      normal_horizontal_norm[..., None], 1.0e-30
    )
    e_h = torch.where(use_velocity[..., None], velocity_direction, default)
    e_h = torch.where(use_normal[..., None], normal_direction, e_h)
    normal = normal.expand_as(velocity)
    n_h = (normal * e_h).sum(-1)
    beta = torch.atan2(n_h, -normal[..., 2])
    v_h = (velocity * e_h).sum(-1)
    v_dot_n = (velocity * normal).sum(-1)
    stationary = v_dot_n.abs() <= numerical_scale
    dynamic = torch.atan2(-velocity[..., 2], v_h)
    gamma = torch.where(stationary, torch.full_like(dynamic, torch.pi / 2), dynamic)
    return beta, gamma, e_h

  def compute(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    surface_z: torch.Tensor | float = 0.0,
    plastic_depth: torch.Tensor | None = None,
    coefficient_scale: torch.Tensor | float = 1.0,
  ) -> RftResult:
    if position.shape != velocity.shape or position.shape != angular_velocity.shape:
      raise ValueError("position, velocity and angular_velocity shapes must match")
    if position.ndim < 2 or position.shape[-1] != 3:
      raise ValueError(f"foot state must have shape (..., 3), got {position.shape}")
    if position.device != self.device:
      raise ValueError(f"input device {position.device} != operator device {self.device}")
    position = position.to(dtype=self.dtype)
    velocity = velocity.to(dtype=self.dtype)
    angular_velocity = angular_velocity.to(dtype=self.dtype)
    batch_shape = position.shape[:-1]
    surface_z = torch.as_tensor(surface_z, device=self.device, dtype=self.dtype).expand(batch_shape)
    coefficient_scale = torch.as_tensor(
      coefficient_scale, device=self.device, dtype=self.dtype
    ).expand(batch_shape)
    singleton = (1,) * len(batch_shape)
    centroids = self.centroids.view(*singleton, self.num_faces, 3)
    normals = self.normals.view(*singleton, self.num_faces, 3)
    areas = self.areas.view(*singleton, self.num_faces)
    facing_down = normals[..., 2] < 0.0
    face_position = position[..., None, :] + centroids
    face_velocity = velocity[..., None, :] + torch.linalg.cross(
      angular_velocity[..., None, :], centroids.expand(*batch_shape, -1, -1), dim=-1
    )
    depth = surface_z[..., None] - face_position[..., 2]
    immersed = depth > 0.0
    speed = torch.linalg.vector_norm(face_velocity, dim=-1)
    lam = torch.clamp(speed / self.yield_speed, 0.0, 1.0)
    v_dot_n = (face_velocity * normals).sum(-1)
    band = torch.clamp_min(LEADING_REGULARISATION * speed, 1.0e-15)
    yield_weight = lam * torch.clamp(v_dot_n / band, 0.0, 1.0)
    active = immersed & (facing_down | (yield_weight > 0.0))
    beta, gamma, e_h = self._angles(face_velocity, normals)
    beta_static = torch.atan2(torch.hypot(normals[..., 0], normals[..., 1]), -normals[..., 2])
    alpha_static = self._alpha_z(beta_static, torch.full_like(beta_static, torch.pi / 2))
    alpha_sliding = self._alpha_z(beta, gamma.abs())
    alpha_capacity = ((1 - lam) * alpha_static + lam * alpha_sliding) * coefficient_scale[..., None]
    alpha_drag = self._alpha_x(beta, gamma) * coefficient_scale[..., None]
    positive_depth = torch.where(active, depth, torch.zeros_like(depth))
    lift = alpha_capacity * positive_depth * areas * facing_down
    drag = yield_weight * alpha_drag * positive_depth * areas
    penetration = surface_z - (position[..., 2] - self.radius)
    if plastic_depth is None:
      plastic_depth = penetration.clamp_min(0.0)
    else:
      plastic_depth = torch.as_tensor(
        plastic_depth, device=self.device, dtype=self.dtype
      ).expand(batch_shape).clamp_min(0.0)
    depth_at_max = depth + (plastic_depth - penetration)[..., None]
    capacity_at_max = (
      alpha_static * coefficient_scale[..., None] * depth_at_max.clamp_min(0.0)
      * areas * facing_down
    ).sum(-1)
    recovery = (1 - self.plastic_ratio) * plastic_depth
    unloading_fraction = torch.clamp(
      1 - (plastic_depth - penetration) / torch.clamp_min(recovery, 1.0e-30), 0.0, 1.0
    )
    unloading_limit = torch.where(
      (plastic_depth > 0) & (recovery > 0), capacity_at_max * unloading_fraction,
      torch.zeros_like(capacity_at_max),
    )
    lift_total = lift.sum(-1)
    scale = torch.where(
      (plastic_depth > penetration) & (lift_total > unloading_limit) & (lift_total > 0),
      unloading_limit / torch.clamp_min(lift_total, 1.0e-30), torch.ones_like(lift_total),
    )
    lift *= scale[..., None]
    face_force = -drag[..., None] * e_h
    face_force[..., 2] += lift
    face_force = torch.where(active[..., None], face_force, torch.zeros_like(face_force))
    force = face_force.sum(-2)
    torque = torch.linalg.cross(centroids.expand_as(face_force), face_force, dim=-1).sum(-2)
    power = (force * velocity).sum(-1) + (torque * angular_velocity).sum(-1)
    return RftResult(
      force=force, torque=torque, n_immersed=immersed.sum(-1), n_active=active.sum(-1),
      max_depth=depth.amax(-1).clamp_min(0.0), immersed_area=(immersed * areas).sum(-1),
      power=power,
    )


class RftState:
  """Independent per-environment, per-foot elasto-plastic loading memory."""

  def __init__(self, num_envs: int, num_feet: int, *, device, dtype=torch.float32):
    self.plastic_depth = torch.zeros((num_envs, num_feet), device=device, dtype=dtype)
    self.plastic_position = torch.zeros((num_envs, num_feet, 2), device=device, dtype=dtype)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = slice(None) if env_ids is None else env_ids
    self.plastic_depth[env_ids] = 0.0
    self.plastic_position[env_ids] = 0.0

  def update(self, foot_position: torch.Tensor, surface_z, radius: float) -> torch.Tensor:
    surface_z = torch.as_tensor(
      surface_z, device=foot_position.device, dtype=foot_position.dtype
    ).expand(foot_position.shape[:-1])
    penetration = surface_z - (foot_position[..., 2] - radius)
    airborne = penetration <= 0.0
    deeper = (~airborne) & (penetration > self.plastic_depth)
    displacement = torch.linalg.vector_norm(
      foot_position[..., :2] - self.plastic_position, dim=-1
    )
    depth = self.plastic_depth.clamp(0.0, radius)
    width = 2 * torch.sqrt(torch.clamp_min(2 * radius * depth - depth.square(), 0.0))
    fresh = (~airborne) & (~deeper) & (displacement > width)
    self.plastic_depth = torch.where(airborne, 0.0, self.plastic_depth)
    update = deeper | fresh
    self.plastic_depth = torch.where(update, penetration, self.plastic_depth)
    self.plastic_position = torch.where(
      update[..., None], foot_position[..., :2], self.plastic_position
    )
    return self.plastic_depth
