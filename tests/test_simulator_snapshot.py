from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from go2_supermimic.diagnostics.simulator_snapshot import (
  _capture_tree,
  _restore_tree,
)


class ReplacingState:
  def __init__(self):
    self.tensor = torch.tensor([1.0, 2.0])
    self.counter = 4
    self.child = {"history": [torch.tensor([3.0]), None]}

  def advance(self):
    self.tensor = self.tensor + 10.0
    self.counter += 1
    self.child["history"][0].add_(20.0)
    self.child["history"][1] = "changed"


def test_tree_snapshot_restores_replaced_tensors_and_scalars():
  state = ReplacingState()
  saved = _capture_tree(state)
  state.advance()
  _restore_tree(state, saved)
  torch.testing.assert_close(state.tensor, torch.tensor([1.0, 2.0]))
  torch.testing.assert_close(state.child["history"][0], torch.tensor([3.0]))
  assert state.counter == 4
  assert state.child["history"][1] is None


def test_tree_snapshot_is_independent_of_live_storage():
  state = ReplacingState()
  saved = _capture_tree(state)
  state.tensor.zero_()
  state.child["history"][0].fill_(99.0)
  torch.testing.assert_close(saved.tensors[(("attr", "tensor"),)], torch.tensor([1.0, 2.0]))
  torch.testing.assert_close(
    saved.tensors[(("attr", "child"), ("key", "history"), ("index", 0))],
    torch.tensor([3.0]),
  )
