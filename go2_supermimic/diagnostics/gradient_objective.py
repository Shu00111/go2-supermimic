"""Common open-loop objective for physical finite differences and WM BPTT.

The learned terminal value used by :meth:`Imagination.rollout` is deliberately
excluded: it cannot be evaluated as physical ground truth.  Target-generator
states are supplied as a frozen sequence, so both sides differentiate the same
open-loop scalar instead of mixing in a closed-loop command response.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from go2_supermimic.supermimic import state as st


@dataclass(frozen=True)
class ObjectiveResult:
  """Per-sample common objective and diagnostics."""

  value: torch.Tensor
  rewards: torch.Tensor
  survival: torch.Tensor
  fall_probability: torch.Tensor


def _generator_at(sequence: Mapping[str, torch.Tensor], step: int) -> dict[str, torch.Tensor]:
  return {name: value[:, step] for name, value in sequence.items()}


def common_objective(
  imagination,
  states: torch.Tensor,
  poses: torch.Tensor,
  actions: torch.Tensor,
  action_prev: torch.Tensor,
  generator: Mapping[str, torch.Tensor],
  constants: Mapping[str, torch.Tensor],
) -> ObjectiveResult:
  """Evaluate the reward-only SuperMimic objective on a supplied trajectory.

  Mathematically, for horizon ``H`` this is

  ``J = sum_t gamma^t c_t [r_analytic(t) + w_term*dt*d(s_{t+1})]``

  with ``c_0=1`` and ``c_{t+1}=c_t*(1-d(s_{t+1}))``.  ``actions`` are in the
  normalized 12-D policy/action-manager coordinates.  ``generator[:, t]`` is
  the same frozen target/obstacle/command state for every compared rollout.
  There is no learned critic endpoint in ``J``.
  """
  if states.ndim != 3 or poses.ndim != 3 or actions.ndim != 3:
    raise ValueError("states, poses, and actions must be batched trajectories")
  batch, horizon, _ = actions.shape
  if states.shape[:2] != (batch, horizon + 1) or poses.shape[:2] != (batch, horizon + 1):
    raise ValueError("state and pose trajectories must have H+1 samples")
  if action_prev.shape != actions.shape:
    raise ValueError("action_prev must have the same shape as actions")
  for name, value in generator.items():
    if value.shape[:2] != (batch, horizon):
      raise ValueError(f"generator field {name!r} must start with shape (B, H)")

  survival = states.new_ones(batch)
  objective = states.new_zeros(batch)
  discount = 1.0
  rewards, survivals, falls = [], [survival], []
  cfg = imagination.cfg
  for step in range(horizon):
    gst = _generator_at(generator, step)
    reward, _ = imagination._rewards(
      states[:, step], states[:, step + 1], poses[:, step + 1], gst,
      actions[:, step], action_prev[:, step], constants,
    )
    fall = imagination.fall_prob(states[:, step + 1])
    reward = reward + cfg.weights.terminated * cfg.dt * fall
    objective = objective + discount * survival * reward
    survival = survival * (1.0 - fall)
    discount *= cfg.gamma
    rewards.append(reward)
    falls.append(fall)
    survivals.append(survival)
  return ObjectiveResult(
    value=objective,
    rewards=torch.stack(rewards, 1),
    survival=torch.stack(survivals, 1),
    fall_probability=torch.stack(falls, 1),
  )


def world_model_open_loop(
  imagination,
  world_model,
  initial_state: torch.Tensor,
  initial_pose: torch.Tensor,
  actions: torch.Tensor,
  initial_action_prev: torch.Tensor,
  context: torch.Tensor,
  generator: Mapping[str, torch.Tensor],
  constants: Mapping[str, torch.Tensor],
) -> tuple[ObjectiveResult, torch.Tensor, torch.Tensor]:
  """Roll out a WM under fixed actions and evaluate :func:`common_objective`.

  This is the BPTT counterpart of physical centered differences.  Callers may
  set only ``actions[:, 0]`` to require gradients; all future actions remain
  fixed nominal values.  M2 contact memory is propagated when available.
  """
  batch, horizon, _ = actions.shape
  if initial_state.shape[0] != batch or initial_pose.shape[0] != batch:
    raise ValueError("initial state/pose batch does not match actions")
  state, pose, previous = initial_state, initial_pose, initial_action_prev
  contact = (world_model.initial_contact(state, previous)[0]
             if hasattr(world_model, "initial_contact") else None)
  states, poses, previous_actions = [state], [pose], []
  for step in range(horizon):
    gst = _generator_at(generator, step)
    relief = imagination.relief(pose, gst, imagination.cfg.sensor_grad)
    action = actions[:, step]
    previous_actions.append(previous)
    if contact is None:
      state, delta = world_model(state, action, previous, context, relief)
    else:
      state, delta, contact = world_model.step(
        state, contact, action, previous, context, relief
      )
    pose = st.apply_pose_delta(pose, delta)
    states.append(state)
    poses.append(pose)
    previous = action
  state_trace, pose_trace = torch.stack(states, 1), torch.stack(poses, 1)
  result = common_objective(
    imagination, state_trace, pose_trace, actions,
    torch.stack(previous_actions, 1), generator, constants,
  )
  return result, state_trace, pose_trace
