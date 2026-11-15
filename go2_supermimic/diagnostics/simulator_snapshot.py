"""Full in-process snapshot/restore for the mjlab environments used here.

MuJoCo's generalized coordinates are only one part of the Markov state.  This
adapter also preserves solver warm-starts, controls and applied wrenches, mjlab
manager buffers, actuator latency, command generators, sensors, and the custom
RFT loading path.  Static model/terrain data are deliberately not copied: a
snapshot is tied to the exact environment and Warp data allocation that made it.

The adapter is intentionally conservative.  It fails when the environment or
data allocation does not match instead of pretending that a portable snapshot
has been restored.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


# MuJoCo mjSTATE_INTEGRATION fields, plus sleeping state.  Derived kinematics,
# contacts, constraints, and sensors are reconstructed by forward()/sense().
_SIM_FIELDS = (
  "time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
  "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
  "userdata", "tree_asleep",
)

# References/configuration that are immutable during a replay or lead back to
# the whole environment.  Stateful children (terms, buffers, generators) are
# intentionally not excluded.
_SKIP_ATTRS = {
  "_env", "env", "cfg", "_entity", "entity", "robot", "scene", "sim",
  "model", "data", "indexing", "operator", "material", "height_field",
  "_obstacles", "_spec", "spec", "_model_bridge", "_data_bridge",
  "_struct", "_wrapped_cache", "_data", "_model", "device", "_device", "wp_device",
  "action_manager", "observation_manager", "command_manager", "reward_manager",
  "termination_manager", "event_manager", "metrics_manager",
  "curriculum_manager", "recorder_manager", "manager_visualizers",
  "action_space", "observation_space", "single_action_space",
  "single_observation_space", "_offline_renderer",
  "extras",
}

Path = tuple[tuple[str, Any], ...]


def _torch_view(value: Any) -> torch.Tensor | None:
  """Return the tensor behind mjlab's TorchArray proxy, if applicable."""
  if torch.is_tensor(value):
    return value
  tensor = getattr(value, "_tensor", None)
  return tensor if torch.is_tensor(tensor) else None


@dataclass(frozen=True)
class _TreeState:
  tensors: dict[Path, torch.Tensor]
  scalars: dict[Path, Any]


@dataclass(frozen=True)
class SimulatorSnapshot:
  """An immutable, allocation-bound snapshot.

  Tensor values are cloned.  Mutating the live environment after capture cannot
  mutate this object, which is important for paired finite differences.
  """

  environment_id: int
  warp_data_id: int
  num_envs: int
  sim_shapes: dict[str, tuple[int, ...]]
  sim_state: dict[str, torch.Tensor]
  warp_data_state: dict[tuple[str, ...], torch.Tensor]
  roots: dict[str, _TreeState]
  python_rng_state: object
  numpy_rng_state: tuple
  torch_rng_state: torch.Tensor
  cuda_rng_state: tuple[torch.Tensor, ...]


def _capture_warp_arrays(root: Any) -> dict[tuple[str, ...], torch.Tensor]:
  """Clone every Warp array in an MJWarp Data struct, including nested structs."""
  import warp as wp

  arrays: dict[tuple[str, ...], torch.Tensor] = {}
  seen: set[int] = set()

  def visit(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, wp.array):
      if value.size:
        arrays[path] = wp.to_torch(value).detach().clone()
      return
    identity = id(value)
    if identity in seen:
      return
    seen.add(identity)
    module = type(value).__module__
    if not (module.startswith("mujoco_warp") or module.startswith("warp")):
      return
    for name in dir(value):
      if name.startswith("_"):
        continue
      try:
        child = getattr(value, name)
      except Exception:
        continue
      if callable(child) or isinstance(child, (bool, int, float, str, type(None))):
        continue
      visit(child, path + (name,))

  visit(root, ())
  return arrays


def _restore_warp_arrays(root: Any, state: dict[tuple[str, ...], torch.Tensor]) -> None:
  import warp as wp

  for path, saved in state.items():
    live = root
    for name in path:
      live = getattr(live, name)
    if not isinstance(live, wp.array):
      raise RuntimeError(f"MJWarp data field {path} is no longer a Warp array")
    tensor = wp.to_torch(live)
    if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
      raise RuntimeError(f"MJWarp data field {path} changed shape or dtype")
    tensor.copy_(saved)


def _children(value: Any):
  if isinstance(value, dict):
    for key, child in value.items():
      yield ("key", key), child
  elif isinstance(value, list):
    for index, child in enumerate(value):
      yield ("index", index), child
  elif hasattr(value, "__dict__"):
    for name, child in vars(value).items():
      if name not in _SKIP_ATTRS:
        yield ("attr", name), child


def _capture_tree(root: Any) -> _TreeState:
  tensors: dict[Path, torch.Tensor] = {}
  scalars: dict[Path, Any] = {}
  seen: set[int] = set()

  def visit(value: Any, path: Path) -> None:
    tensor = _torch_view(value)
    if tensor is not None:
      tensors[path] = tensor.detach().clone()
      return
    if value is None or isinstance(value, (bool, int, float, str)):
      scalars[path] = copy.deepcopy(value)
      return
    if isinstance(value, tuple):
      # Tuples in these objects are configuration/static metadata.  Mutable
      # dynamic collections in mjlab are lists or dictionaries.
      return
    if not isinstance(value, (dict, list)) and not hasattr(value, "__dict__"):
      return
    identity = id(value)
    if identity in seen:
      return
    seen.add(identity)
    for step, child in _children(value):
      visit(child, path + (step,))

  visit(root, ())
  return _TreeState(tensors=tensors, scalars=scalars)


def _resolve(root: Any, path: Path) -> Any:
  value = root
  for kind, key in path:
    if kind == "attr":
      value = getattr(value, key)
    elif kind == "key":
      value = value[key]
    else:
      value = value[key]
  return value


def _assign(root: Any, path: Path, value: Any) -> None:
  if not path:
    raise RuntimeError("cannot replace a snapshot root")
  parent = _resolve(root, path[:-1])
  kind, key = path[-1]
  if kind == "attr":
    setattr(parent, key, value)
  elif kind == "key":
    parent[key] = value
  else:
    parent[key] = value


def _restore_tree(root: Any, state: _TreeState) -> None:
  for path, saved in state.tensors.items():
    current = _resolve(root, path)
    tensor = _torch_view(current)
    if tensor is None:
      _assign(root, path, saved.clone())
      continue
    if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
      raise RuntimeError(
        f"snapshot tensor mismatch at {path}: live={tensor.shape}/{tensor.dtype}, "
        f"saved={saved.shape}/{saved.dtype}"
      )
    tensor.copy_(saved)
  for path, saved in state.scalars.items():
    _assign(root, path, copy.deepcopy(saved))


def _state_roots(env: Any) -> dict[str, Any]:
  roots = {
    "env": env,
    "action_manager": env.action_manager,
    "observation_manager": env.observation_manager,
    "command_manager": env.command_manager,
    "reward_manager": env.reward_manager,
    "termination_manager": env.termination_manager,
    "event_manager": env.event_manager,
    "metrics_manager": env.metrics_manager,
    "curriculum_manager": env.curriculum_manager,
    "scene_sensors": env.scene.sensors,
  }
  # EntityData owns actuator targets/encoder bias; custom actuators own bus
  # delay buffers and clipped-velocity state.
  for name, entity in env.scene.entities.items():
    roots[f"entity_data:{name}"] = entity._data
    roots[f"entity_actuators:{name}"] = entity._actuators
  return roots


def capture_snapshot(env: Any) -> SimulatorSnapshot:
  """Capture every replay-relevant mutable state owned by ``env``."""
  import warp as wp

  wp.synchronize_device(env.sim.wp_device)
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  sim_state = {}
  for name in _SIM_FIELDS:
    if not hasattr(env.sim.wp_data, name):
      raise RuntimeError(f"MuJoCo-Warp data lacks required snapshot field {name!r}")
    raw = getattr(env.sim.wp_data, name)
    if not raw.size:
      continue
    value = wp.to_torch(raw)
    sim_state[name] = value.detach().clone()
  cuda_state = tuple(x.clone() for x in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
  return SimulatorSnapshot(
    environment_id=id(env),
    warp_data_id=id(env.sim.wp_data),
    num_envs=env.num_envs,
    sim_shapes={name: tuple(value.shape) for name, value in sim_state.items()},
    sim_state=sim_state,
    warp_data_state=_capture_warp_arrays(env.sim.wp_data),
    roots={name: _capture_tree(root) for name, root in _state_roots(env).items()},
    python_rng_state=copy.deepcopy(random.getstate()),
    numpy_rng_state=copy.deepcopy(np.random.get_state()),
    torch_rng_state=torch.random.get_rng_state().clone(),
    cuda_rng_state=cuda_state,
  )


@torch.no_grad()
def restore_snapshot(env: Any, snapshot: SimulatorSnapshot) -> None:
  """Restore ``snapshot`` into the exact environment that produced it."""
  import warp as wp

  if id(env) != snapshot.environment_id or id(env.sim.wp_data) != snapshot.warp_data_id:
    raise RuntimeError("snapshot belongs to a different environment or Warp allocation")
  if env.num_envs != snapshot.num_envs:
    raise RuntimeError("snapshot environment count changed")

  _restore_warp_arrays(env.sim.wp_data, snapshot.warp_data_state)
  for name, saved in snapshot.sim_state.items():
    live = wp.to_torch(getattr(env.sim.wp_data, name))
    if tuple(live.shape) != snapshot.sim_shapes[name] or live.dtype != saved.dtype:
      raise RuntimeError(f"MuJoCo-Warp field {name!r} changed shape or dtype")
    live.copy_(saved)

  # The complete MJWarp Data allocation, including derived/contact/constraint
  # arrays, was restored above.  Calling forward() here would replace that exact
  # solver state with a numerically close reconstruction and add replay noise.

  live_roots = _state_roots(env)
  if live_roots.keys() != snapshot.roots.keys():
    raise RuntimeError("snapshot root set changed")
  for name, state in snapshot.roots.items():
    _restore_tree(live_roots[name], state)

  random.setstate(copy.deepcopy(snapshot.python_rng_state))
  np.random.set_state(copy.deepcopy(snapshot.numpy_rng_state))
  torch.random.set_rng_state(snapshot.torch_rng_state.cpu())
  if snapshot.cuda_rng_state:
    torch.cuda.set_rng_state_all([x.cpu() for x in snapshot.cuda_rng_state])
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  wp.synchronize_device(env.sim.wp_device)
