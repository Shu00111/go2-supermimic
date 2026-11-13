"""Per-foot contact state and contact-structured SuperMimic world model.

The actor never consumes these values.  Simulator contact values are labels for
world-model fitting only; imagination starts from an estimate made from the
same proprioceptive state and action history available at deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.world_model import RunningNorm, mlp
from go2_supermimic.target.terrain import SCAN_POINTS

NUM_FEET = 4
CONTACT_DIM = 8
CONTACT_STATE_DIM = NUM_FEET * CONTACT_DIM
ESTIMATOR_HIDDEN = 20
ESTIMATOR_STATE_DIM = ESTIMATOR_HIDDEN + 1  # hidden plus loading memory


@dataclass(frozen=True)
class ContactScales:
  """Fixed SI label scales; values are not learned from held-out data."""

  force_n: float = 150.0
  intrusion_m: float = 0.08
  memory_m: float = 0.08
  tangential_velocity_mps: float = 1.0

  def tensor(self, ref: torch.Tensor) -> torch.Tensor:
    return ref.new_tensor([
      self.force_n, self.force_n, self.force_n, self.intrusion_m,
      self.memory_m, self.tangential_velocity_mps,
      self.tangential_velocity_mps, 1.0,
    ])


def per_foot_observables(s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
  """Return 21 deployment-observable geometry/proprioception values per foot.

  Joint ordering is FR, FL, RR, RL with three consecutive joints per leg.  Foot
  positions are base-relative and yaw-aligned, so no global pose is leaked.
  Soil memory, terrain material, contact force and boundary distance in ``s``
  are deliberately excluded.
  """
  if s.shape[-1] != st.STATE_DIM or action.shape[-1] != st.NUM_JOINTS:
    raise ValueError("invalid state/action width")
  b = s.shape[0]
  q = s[:, st.Q].reshape(b, NUM_FEET, 3)
  qd = s[:, st.QD].reshape(b, NUM_FEET, 3)
  feet = s[:, st.FEET].reshape(b, NUM_FEET, 3)
  act = action.reshape(b, NUM_FEET, 3)
  shared = torch.cat([s[:, st.GRAV], s[:, st.V_YAW], s[:, st.W_B]], -1)
  shared = shared[:, None].expand(-1, NUM_FEET, -1)
  return torch.cat([shared, q, qd, feet, act], -1)


def normalize_contact(z: torch.Tensor, scales: ContactScales = ContactScales()) -> torch.Tensor:
  return z / scales.tensor(z)


class PersistentContactEstimator(nn.Module):
  """Shared per-foot recurrent estimator with differentiable loading memory."""

  def __init__(self, scales: ContactScales = ContactScales()):
    super().__init__()
    self.scales = scales
    self.gru = nn.GRUCell(21, ESTIMATOR_HIDDEN)
    # force(3), intrusion proposal, retention, tangential velocity(2), contact logit
    self.head = nn.Linear(ESTIMATOR_HIDDEN, 8)
    with torch.no_grad():
      self.head.bias[3] = -2.0
      self.head.bias[4] = 2.2
      self.head.bias[7] = -0.5

  @staticmethod
  def zero_state(batch: int, ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(batch, NUM_FEET, ESTIMATOR_STATE_DIM)

  def forward(self, s: torch.Tensor, action: torch.Tensor, recurrent: torch.Tensor):
    expected = (s.shape[0], NUM_FEET, ESTIMATOR_STATE_DIM)
    if recurrent.shape != expected:
      raise ValueError(f"recurrent state {tuple(recurrent.shape)} != {expected}")
    x = per_foot_observables(s, action)
    hidden0, old_memory = recurrent[..., :ESTIMATOR_HIDDEN], recurrent[..., -1]
    hidden = self.gru(x.reshape(-1, 21), hidden0.reshape(-1, ESTIMATOR_HIDDEN))
    hidden = hidden.reshape(s.shape[0], NUM_FEET, ESTIMATOR_HIDDEN)
    raw = self.head(hidden)
    probability = raw[..., 7].sigmoid()
    intrusion = self.scales.intrusion_m * raw[..., 3].sigmoid()
    retention = raw[..., 4].sigmoid()
    # Soft contact gating keeps the BPTT path differentiable at touchdown.
    memory = probability * (intrusion + retention * F.relu(old_memory - intrusion))
    force = probability[..., None] * self.scales.force_n * torch.tanh(raw[..., :3])
    tangent = probability[..., None] * self.scales.tangential_velocity_mps * torch.tanh(
      raw[..., 5:7]
    )
    z = torch.cat([
      force, (probability * intrusion)[..., None], memory[..., None], tangent,
      probability[..., None],
    ], -1)
    recurrent_next = torch.cat([probability[..., None] * hidden, memory[..., None]], -1)
    return z, recurrent_next

  @staticmethod
  def reset(recurrent: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    if done.shape != recurrent.shape[:1]:
      raise ValueError("done mask must have one value per environment")
    return torch.where(done[:, None, None], torch.zeros_like(recurrent), recurrent)


class ContactStructuredWorldModel(nn.Module):
  """M2: shared per-foot transition followed by contact-conditioned body delta."""

  def __init__(
    self,
    hidden: tuple[int, ...] = (1024, 1024, 1024),
    foot_hidden: tuple[int, ...] = (128, 128),
    estimator: PersistentContactEstimator | None = None,
    use_loading_memory: bool = True,
  ):
    super().__init__()
    base_in = st.STATE_DIM + 2 * st.NUM_JOINTS + st.CONTEXT_DIM + SCAN_POINTS
    out_dim = st.STATE_DIM + st.POSE_DELTA_DIM
    self.estimator = estimator or PersistentContactEstimator()
    self.use_loading_memory = use_loading_memory
    self.contact_transition = mlp(21 + CONTACT_DIM + st.CONTEXT_DIM, foot_hidden, CONTACT_DIM)
    self.body = mlp(base_in + CONTACT_STATE_DIM, hidden, out_dim)
    self.in_norm = RunningNorm(base_in)
    self.out_norm = RunningNorm(out_dim, std_floor=1e-3)
    self.scales = self.estimator.scales

  def _memory_contract(self, z: torch.Tensor) -> torch.Tensor:
    """Remove the explicit loading-memory channel for the A1 mechanism ablation."""
    if self.use_loading_memory:
      return z
    return torch.cat([z[..., :4], torch.zeros_like(z[..., 4:5]), z[..., 5:]], -1)

  @staticmethod
  def _base_inputs(s, a, a_prev, xi, relief):
    return torch.cat([s, a, a_prev, xi, s[:, st.Z] + relief], -1)

  def initial_contact(self, s: torch.Tensor, a_prev: torch.Tensor):
    recurrent = self.estimator.zero_state(s.shape[0], s)
    z, recurrent = self.estimator(s, a_prev, recurrent)
    if not self.use_loading_memory:
      recurrent = torch.cat([recurrent[..., :-1], torch.zeros_like(recurrent[..., -1:])], -1)
    return self._memory_contract(z), recurrent

  def _decode_transition(self, raw: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    probability = raw[..., 7].sigmoid()
    force = probability[..., None] * self.scales.force_n * torch.tanh(raw[..., :3])
    intrusion = probability * self.scales.intrusion_m * raw[..., 3].sigmoid()
    proposal = self.scales.memory_m * raw[..., 4].sigmoid()
    memory = probability * (proposal + raw[..., 4].sigmoid() * F.relu(old[..., 4] - proposal))
    tangent = probability[..., None] * self.scales.tangential_velocity_mps * torch.tanh(
      raw[..., 5:7]
    )
    return torch.cat([force, intrusion[..., None], memory[..., None], tangent,
                      probability[..., None]], -1)

  def step(self, s, z, a, a_prev, xi, relief):
    z = self._memory_contract(z)
    local = per_foot_observables(s, a)
    xi_foot = xi[:, None].expand(-1, NUM_FEET, -1)
    raw_z = self.contact_transition(torch.cat([local, normalize_contact(z), xi_foot], -1))
    z_next = self._memory_contract(self._decode_transition(raw_z, z))
    base = self._base_inputs(s, a, a_prev, xi, relief)
    y = self.out_norm.unnormalize(self.body(torch.cat([
      self.in_norm(base), normalize_contact(z_next).reshape(s.shape[0], -1)
    ], -1)))
    return s + y[:, :st.STATE_DIM], y[:, st.STATE_DIM:], z_next

  def forward(self, s, a, a_prev, xi, relief):
    z, _ = self.initial_contact(s, a_prev)
    s_next, pose_delta, _ = self.step(s, z, a, a_prev, xi, relief)
    return s_next, pose_delta

  @torch.no_grad()
  def update_stats(self, s, a, a_prev, xi, relief, s_next, pose_delta, *_):
    self.in_norm.update(self._base_inputs(s, a, a_prev, xi, relief))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], -1))

  def loss(self, s0, actions, actions_prev, xi, reliefs, s_targets, pose_targets, valid,
           contact_targets: torch.Tensor | None = None):
    z, _ = self.initial_contact(s0, actions_prev[:, 0])
    s = s0
    scale = self.out_norm.std()
    denom = valid.sum().clamp_min(1.0)
    body_total = s.new_zeros(())
    contact_total = s.new_zeros(())
    for j in range(actions.shape[1]):
      s_pred, pose_pred, z = self.step(s, z, actions[:, j], actions_prev[:, j], xi,
                                       reliefs[:, j])
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], -1)
      body_total = body_total + ((err / scale) ** 2).mean(-1).mul(valid[:, j]).sum()
      if contact_targets is not None:
        target = self._memory_contract(
          contact_targets[:, j].reshape(-1, NUM_FEET, CONTACT_DIM)
        )
        physical = F.smooth_l1_loss(normalize_contact(z)[..., :7],
                                    normalize_contact(target)[..., :7], reduction="none").mean((-1, -2))
        classification = F.binary_cross_entropy(z[..., 7].clamp(1e-6, 1 - 1e-6),
                                                target[..., 7], reduction="none").mean(-1)
        contact_total = contact_total + ((physical + classification) * valid[:, j]).sum()
      s = s_pred
    return body_total / denom + contact_total / denom


def count_parameters(module: nn.Module) -> int:
  return sum(p.numel() for p in module.parameters() if p.requires_grad)
