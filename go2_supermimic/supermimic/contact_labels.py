"""Simulator-oracle contact labels used only by world-model supervision."""

from __future__ import annotations

import torch

from go2_supermimic.robot.go2 import FOOT_GEOM_NAMES
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET


def rigid_contact_force(env, ref: torch.Tensor) -> torch.Tensor:
  """Read rigid contact forces in the canonical FR, FL, RR, RL order.

  mjlab expands a tuple of contact patterns in entity/model order, which is
  not necessarily the tuple order requested by the caller.  The Go2 MJCF uses
  FL, FR, RL, RR model order.  Reordering by ``primary_names`` is therefore
  mandatory before these labels are combined with RFT tensors, whose foot
  order is FR, FL, RR, RL.
  """
  try:
    sensor = env.scene["feet_ground_contact"]
  except (KeyError, TypeError):
    raise RuntimeError("configured feet_ground_contact sensor is unavailable")
  data = getattr(sensor, "data", sensor)
  for name in ("force", "net_force_w", "net_forces_w", "forces_w"):
    value = getattr(data, name, None)
    if value is not None:
      primary_names = tuple(getattr(sensor, "primary_names", ()))
      if len(primary_names) != NUM_FEET or set(primary_names) != set(FOOT_GEOM_NAMES):
        raise RuntimeError(
          "feet_ground_contact primaries do not match canonical feet: "
          f"{primary_names}"
        )
      value = value.reshape(env.num_envs, NUM_FEET, -1, 3).sum(-2)
      order = [primary_names.index(foot) for foot in FOOT_GEOM_NAMES]
      value = value[:, order]
      return value.to(device=ref.device, dtype=ref.dtype)
  raise RuntimeError("feet_ground_contact exposes no supported net-force field")


@torch.no_grad()
def oracle_contact_state(env) -> torch.Tensor:
  """Return ``(N, 4, 8)`` labels in SI units.

  Channels are force xyz [N], intrusion [m], persistent loading memory [m],
  tangential foot velocity xy [m/s], and binary contact.  Rigid contacts use
  zero intrusion and zero loading memory; this avoids imposing a deformable
  constitutive law on rigid ground.
  """
  robot = env.scene["robot"]
  site_ids, names = robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)
  if tuple(names) != ("FR", "FL", "RR", "RL"):
    raise RuntimeError(f"unexpected foot order {names}")
  velocity = robot.data.site_lin_vel_w[:, site_ids]
  force = rigid_contact_force(env, velocity)
  intrusion = velocity.new_zeros(env.num_envs, NUM_FEET)
  memory = intrusion.clone()
  if "rft" in env.action_manager.active_terms:
    rft = env.action_manager.get_term("rft")
    soft = getattr(rft, "last_soft_mask", rft.last_signed_distance > 0.0)
    force = force + rft.last_force
    intrusion = torch.where(soft, rft.last_sinkage, intrusion)
    memory = torch.where(soft, rft.state.plastic_depth, memory)
  contact = (force.norm(dim=-1) > 1.0) | (intrusion > 0.0)
  return torch.cat([
    force, intrusion[..., None], memory[..., None], velocity[..., :2],
    contact.to(velocity.dtype)[..., None],
  ], -1).reshape(env.num_envs, NUM_FEET, CONTACT_DIM)
