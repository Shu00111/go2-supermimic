import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_p1_heldout_prediction_v2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_p1_heldout_prediction_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_heldout_ids_select_complete_disjoint_trajectories():
  ids = MODULE.heldout_ids(32)
  assert ids.tolist() == [8, 9, 18, 19, 28, 29]
  assert all(int(value) % 10 in {8, 9} for value in ids)


def test_valid_starts_excludes_any_window_with_terminal_transition():
  done = torch.zeros(2, 6, dtype=torch.bool)
  done[0, 2] = True
  env, start = MODULE.valid_starts(done, 3)
  pairs = set(zip(env.tolist(), start.tolist(), strict=True))
  assert pairs == {(0, 3), (1, 0), (1, 1), (1, 2), (1, 3)}
