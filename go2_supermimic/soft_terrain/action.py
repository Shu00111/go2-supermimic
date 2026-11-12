"""mjlab action term that evaluates and applies RFT every physics substep."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from go2_supermimic.soft_terrain.rft import FOOT_RADIUS, RftOperator, RftState
from go2_supermimic.soft_terrain.materials import CheckerboardMaterial
from go2_supermimic.target import terrain as terrain_query

FOOT_SITES = ("FR", "FL", "RR", "RL")
FOOT_BODIES = tuple(f"{leg}_calf" for leg in FOOT_SITES)


@dataclass(kw_only=True)
class RftActionCfg(ActionTermCfg):
  """Configuration for the zero-dimensional RFT physics action term.

  This term consumes no policy action.  It participates in the action manager
  solely because mjlab calls action terms immediately before every physics
  substep, which is the required force-evaluation order.
  """

  radius: float = FOOT_RADIUS
  subdivisions: int = 1
  zeta_n_per_cm3: float = 3.75
  coefficient_scale_range: tuple[float, float] = (1.0, 1.0)
  surface_z: float = 0.0
  soft_region: str = "all"
  """One of ``all``, ``positive_x`` or ``negative_x`` in env-local coordinates."""
  boundary_x: float = 0.0
  use_local_height: bool = False
  material_cell_size: float = 0.6
  material_seed: int = 7

  def build(self, env):
    return RftAction(self, env)


class RftAction(ActionTerm):
  def __init__(self, cfg: RftActionCfg, env):
    super().__init__(cfg, env)
    self.cfg = cfg
    self.env = env
    if cfg.coefficient_scale_range[0] <= 0.0:
      raise ValueError("RFT coefficient scale must be positive")
    if cfg.coefficient_scale_range[1] < cfg.coefficient_scale_range[0]:
      raise ValueError("invalid RFT coefficient scale range")
    self._raw_actions = torch.empty((env.num_envs, 0), device=env.device)
    self.site_ids, site_names = self._entity.find_sites(FOOT_SITES, preserve_order=True)
    self.body_ids, body_names = self._entity.find_bodies(FOOT_BODIES, preserve_order=True)
    if tuple(site_names) != FOOT_SITES or tuple(body_names) != FOOT_BODIES:
      raise RuntimeError(f"unexpected foot order: sites={site_names}, bodies={body_names}")
    self.operator = RftOperator(
      device=env.device,
      radius=cfg.radius,
      subdivisions=cfg.subdivisions,
      zeta_n_per_cm3=cfg.zeta_n_per_cm3,
    )
    self.state = RftState(env.num_envs, len(FOOT_SITES), device=env.device)
    self.coefficient_scale = torch.empty(
      (env.num_envs, len(FOOT_SITES)), device=env.device
    )
    self.last_force = torch.zeros((env.num_envs, len(FOOT_SITES), 3), device=env.device)
    self.last_torque_at_foot = torch.zeros_like(self.last_force)
    self.last_sinkage = torch.zeros((env.num_envs, len(FOOT_SITES)), device=env.device)
    self.last_power = torch.zeros_like(self.last_sinkage)
    self.last_signed_distance = torch.ones_like(self.last_sinkage)
    self.env_origin_x = env.scene.env_origins[:, 0].unsqueeze(-1)
    self.material = CheckerboardMaterial(
      cell_size=cfg.material_cell_size, seed=cfg.material_seed, foot_radius=cfg.radius
    ) if cfg.soft_region == "checkerboard" else None
    self._resample_scale(slice(None))

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != self._raw_actions.shape:
      raise ValueError(f"RFT action must be empty, got {tuple(actions.shape)}")

  def _resample_scale(self, env_ids) -> None:
    low, high = self.cfg.coefficient_scale_range
    target = self.coefficient_scale[env_ids]
    if low == high:
      target.fill_(low)
    else:
      target.uniform_(low, high)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = slice(None) if env_ids is None else env_ids
    self.state.reset(env_ids)
    self._resample_scale(env_ids)
    self.last_force[env_ids] = 0.0
    self.last_torque_at_foot[env_ids] = 0.0
    self.last_sinkage[env_ids] = 0.0
    self.last_power[env_ids] = 0.0
    self.last_signed_distance[env_ids] = 1.0

  def _soft_mask(self, foot_position: torch.Tensor) -> torch.Tensor:
    local_x = foot_position[..., 0] - self.env_origin_x
    if self.cfg.soft_region == "all":
      self.last_signed_distance.fill_(1.0)
      return torch.ones_like(local_x, dtype=torch.bool)
    if self.cfg.soft_region == "positive_x":
      self.last_signed_distance.copy_(local_x - self.cfg.boundary_x)
    elif self.cfg.soft_region == "negative_x":
      self.last_signed_distance.copy_(self.cfg.boundary_x - local_x)
    elif self.cfg.soft_region == "checkerboard":
      assert self.material is not None
      soft, distance = self.material.query(foot_position[..., :2], self.env.scene.env_origins)
      self.last_signed_distance.copy_(distance)
      return soft
    else:
      raise ValueError(f"unknown soft_region={self.cfg.soft_region!r}")
    return self.last_signed_distance > 0.0

  def _surface_height(self, foot_position: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor | None]:
    if not self.cfg.use_local_height:
      return self.cfg.surface_z, None
    try:
      field = self.env.command_manager.get_term("target").height_field
    except (AttributeError, KeyError, ValueError) as exc:
      raise RuntimeError("rough-soft RFT requires the target command height field") from exc
    if field is None:
      raise RuntimeError("rough-soft RFT received no terrain height field")
    centre = terrain_query.ground(foot_position[..., :2], field)
    faces = self.operator.centroids.to(foot_position)
    face_xy = foot_position[..., None, :2] + faces[:, :2]
    return centre, terrain_query.ground(face_xy, field)

  def apply_actions(self) -> None:
    data = self._entity.data
    foot_position = data.site_pos_w[:, self.site_ids]
    foot_velocity = data.site_lin_vel_w[:, self.site_ids]
    foot_omega = data.site_ang_vel_w[:, self.site_ids]
    soft = self._soft_mask(foot_position)
    surface_z, face_surface_z = self._surface_height(foot_position)
    plastic_depth = self.state.update(
      foot_position, surface_z, self.cfg.radius
    )
    # A rigid contact must not preload the soil memory before the foot crosses
    # the boundary.  Crossing into soft soil begins a fresh loading path.
    self.state.clear_where(~soft, foot_position, surface_z)
    result = self.operator.compute(
      foot_position,
      foot_velocity,
      foot_omega,
      surface_z=surface_z,
      face_surface_z=face_surface_z,
      plastic_depth=plastic_depth,
      coefficient_scale=self.coefficient_scale * soft,
    )
    # MuJoCo's xfrc_applied torque is about the body's COM.  RFT integration
    # returns torque about the spherical foot centre, so include the moment arm.
    body_com = data.body_com_pos_w[:, self.body_ids]
    torque_at_com = result.torque + torch.linalg.cross(
      foot_position - body_com, result.force, dim=-1
    )
    self._entity.write_external_wrench_to_sim(
      result.force, torque_at_com, body_ids=self.body_ids
    )
    self.last_force.copy_(result.force)
    self.last_torque_at_foot.copy_(result.torque)
    self.last_sinkage.copy_(result.max_depth)
    self.last_power.copy_(result.power)
