"""Adapters for the preserved model-independent P1 physical trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from go2_supermimic.diagnostics.frozen_artifacts import sha256_file
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import CONTACT_DIM, NUM_FEET
from go2_supermimic.target.terrain import SCAN_POINTS


REQUIRED_FIELDS = {
  "state": (st.STATE_DIM,), "next_state": (st.STATE_DIM,), "pose_delta": (3,),
  "action": (st.NUM_JOINTS,), "previous_action": (st.NUM_JOINTS,),
  "context": (st.CONTEXT_DIM,), "relief": (SCAN_POINTS,),
  "contact": (NUM_FEET, CONTACT_DIM), "next_contact": (NUM_FEET, CONTACT_DIM),
  "done": (), "episode": (), "boundary_distance": (NUM_FEET,), "soft_mask": (NUM_FEET,),
}


@dataclass(frozen=True)
class FrozenTrajectoryDataset:
  path: Path
  sha256: str
  metadata: dict[str, Any]
  data: dict[str, torch.Tensor]

  def environment_ids(self, split: str) -> list[int]:
    if split not in {"train", "development", "held_out"}:
      raise ValueError(f"unknown split {split!r}")
    allowed = {"train": range(0, 6), "development": range(6, 8), "held_out": range(8, 10)}
    return [index for index in range(self.data["state"].shape[0])
            if index % 10 in allowed[split]]

  def transition(self, environment: int, step: int) -> dict[str, torch.Tensor]:
    return {name: value[environment, step] for name, value in self.data.items()}


def load_frozen_trajectory(path: Path, *, expected_sha256: str) -> FrozenTrajectoryDataset:
  path = path.resolve(strict=True)
  actual = sha256_file(path)
  if actual != expected_sha256:
    raise ValueError(f"trajectory SHA-256 mismatch for {path}")
  payload = torch.load(path, map_location="cpu", weights_only=False)
  missing = set(REQUIRED_FIELDS) - set(payload.get("data", {}))
  if missing:
    raise ValueError(f"trajectory missing fields: {sorted(missing)}")
  data, leading = payload["data"], None
  for name, tail in REQUIRED_FIELDS.items():
    value = data[name]
    if not torch.is_tensor(value) or tuple(value.shape[2:]) != tail:
      raise ValueError(f"invalid trajectory field {name}: {getattr(value, 'shape', None)}")
    leading = leading or tuple(value.shape[:2])
    if tuple(value.shape[:2]) != leading:
      raise ValueError(f"trajectory leading dimensions differ for {name}")
  metadata = payload.get("metadata", {})
  if (metadata.get("schema") != 1 or metadata.get("num_envs") != leading[0]
      or metadata.get("steps") != leading[1]):
    raise ValueError("trajectory metadata/shape mismatch")
  return FrozenTrajectoryDataset(path, actual, metadata, data)


def world_model_window(
  dataset: FrozenTrajectoryDataset, environment: int, start: int, horizon: int
) -> dict[str, torch.Tensor]:
  stop = start + horizon
  if stop > dataset.data["state"].shape[1]:
    raise ValueError("window exceeds frozen trajectory")
  d = dataset.data
  names = ("state", "next_state", "action", "previous_action", "context", "relief",
           "contact", "next_contact", "done")
  return {name: d[name][environment, start:stop].unsqueeze(0) for name in names}
