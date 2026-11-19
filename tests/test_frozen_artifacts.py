import hashlib

import pytest
import torch

from go2_supermimic.diagnostics.frozen_artifacts import assert_exact_schema, verify_file
from go2_supermimic.diagnostics.frozen_trajectory import load_frozen_trajectory, world_model_window


def test_verify_file_and_exact_schema_fail_closed(tmp_path):
  path = tmp_path / "artifact.bin"
  path.write_bytes(b"frozen")
  digest = hashlib.sha256(b"frozen").hexdigest()
  assert verify_file(path, digest)["size_bytes"] == 6
  with pytest.raises(ValueError, match="SHA-256 mismatch"):
    verify_file(path, "0" * 64)
  assert_exact_schema("x", {"a": torch.zeros(2)}, {"a": torch.ones(2)})
  with pytest.raises(ValueError, match="architecture mismatch"):
    assert_exact_schema("x", {"a": torch.zeros(2)}, {"a": torch.ones(3)})


def test_frozen_trajectory_adapter_and_partitions(tmp_path):
  n, t = 10, 3
  shapes = {
    "state": (78,), "next_state": (78,), "pose_delta": (3,), "action": (12,),
    "previous_action": (12,), "context": (4,), "relief": (77,), "contact": (4, 8),
    "next_contact": (4, 8), "done": (), "episode": (), "boundary_distance": (4,),
    "soft_mask": (4,),
  }
  data = {name: torch.zeros(n, t, *tail) for name, tail in shapes.items()}
  path = tmp_path / "trajectory.pt"
  torch.save({"metadata": {"schema": 1, "num_envs": n, "steps": t}, "data": data}, path)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  dataset = load_frozen_trajectory(path, expected_sha256=digest)
  assert dataset.environment_ids("train") == [0, 1, 2, 3, 4, 5]
  assert dataset.environment_ids("development") == [6, 7]
  assert dataset.environment_ids("held_out") == [8, 9]
  assert world_model_window(dataset, 8, 0, 2)["state"].shape == (1, 2, 78)
