"""Simulator-facing diagnostics that are never used by training."""

from .gradient_objective import (
  ObjectiveResult,
  common_objective,
  finite_difference_convergence,
  world_model_open_loop,
)
from .simulator_snapshot import SimulatorSnapshot, capture_snapshot, restore_snapshot
from .frozen_artifacts import load_frozen_checkpoint, verify_file
from .frozen_trajectory import load_frozen_trajectory, world_model_window

__all__ = [
  "ObjectiveResult", "SimulatorSnapshot", "capture_snapshot", "common_objective",
  "finite_difference_convergence", "load_frozen_checkpoint", "load_frozen_trajectory",
  "restore_snapshot", "verify_file", "world_model_open_loop", "world_model_window",
]
