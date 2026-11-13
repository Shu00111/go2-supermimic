import torch

from go2_supermimic.supermimic.buffer import ReplayBuffer, WindowArchive
from go2_supermimic.supermimic import state as st


def test_replay_and_archive_round_trip():
  replay = ReplayBuffer(2, 5, 7, "cpu", gen_dim=3)
  replay.s.uniform_()
  replay.contact.uniform_()
  replay.ptr, replay.size = 3, 4
  restored = ReplayBuffer(2, 5, 7, "cpu", gen_dim=3)
  restored.load_state_dict(replay.state_dict())
  assert restored.ptr == 3 and restored.size == 4
  torch.testing.assert_close(restored.s, replay.s)
  torch.testing.assert_close(restored.contact, replay.contact)

  archive = WindowArchive(6, 2, st.CONTEXT_DIM, "cpu")
  archive.data["s"].uniform_()
  archive.ptr, archive.size = 4, 5
  archive2 = WindowArchive(6, 2, st.CONTEXT_DIM, "cpu")
  archive2.load_state_dict(archive.state_dict())
  assert archive2.ptr == 4 and archive2.size == 5
  torch.testing.assert_close(archive2.data["s"], archive.data["s"])


def test_task_boundary_terminates_loaded_trajectories():
  replay = ReplayBuffer(2, 5, 7, "cpu", gen_dim=3)
  replay.ptr, replay.size = 3, 3
  replay.mark_task_boundary()
  assert replay.done[:, 2].tolist() == [1.0, 1.0]
  assert replay.done[:, [0, 1, 3, 4]].count_nonzero() == 0
