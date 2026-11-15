import importlib.util
from pathlib import Path

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import ContactStructuredWorldModel
from go2_supermimic.supermimic.world_model import WorldModel


def load_script(name: str):
  path = Path(__file__).resolve().parents[1] / "scripts" / name
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


audit = load_script("analyze_world_model_audit_v2.py")


def synthetic_data(environments=10, steps=4):
  torch.manual_seed(19)
  data = {
    "state": torch.randn(environments, steps, st.STATE_DIM),
    "next_state": torch.randn(environments, steps, st.STATE_DIM),
    "pose_delta": torch.randn(environments, steps, 3),
    "action": torch.randn(environments, steps, st.NUM_JOINTS),
    "previous_action": torch.randn(environments, steps, st.NUM_JOINTS),
    "context": torch.randn(environments, steps, st.CONTEXT_DIM),
    "relief": torch.randn(environments, steps, 77),
    "contact": torch.randn(environments, steps, 4, 8),
    "next_contact": torch.randn(environments, steps, 4, 8),
    "done": torch.zeros(environments, steps, dtype=torch.bool),
    "episode": torch.zeros(environments, steps, dtype=torch.int32),
    "soil_strength": torch.ones(environments, steps, 4),
    "boundary_distance": torch.randn(environments, steps, 4),
    "soft_mask": torch.rand(environments, steps, 4) > 0.5,
  }
  data["contact"][..., 7] = (torch.rand(environments, steps, 4) > 0.5).float()
  data["next_contact"][..., 7] = (torch.rand(environments, steps, 4) > 0.5).float()
  return data


def test_environment_split_is_disjoint_and_complete():
  selections = {name: set(audit.env_selection(37, name).tolist())
                for name in ("train", "dev", "held_out")}
  assert selections["train"].isdisjoint(selections["dev"])
  assert selections["train"].isdisjoint(selections["held_out"])
  assert selections["dev"].isdisjoint(selections["held_out"])
  assert set.union(*selections.values()) == set(range(37))


def test_matched_analysis_emits_one_step_contact_and_horizon_rows():
  data = synthetic_data()
  models = {
    "M1": WorldModel((16,)),
    "M2-v1": ContactStructuredWorldModel((16,), foot_hidden=(8,)),
  }
  rows = []
  audit.analyze_one_step(
    rows, "synthetic", "dev", data, models,
    torch.ones(st.STATE_DIM), torch.ones(st.POSE_DELTA_DIM), "cpu", 8,
  )
  audit.analyze_recurrent_estimator(
    rows, "synthetic", "dev", data, models["M2-v1"], "cpu"
  )
  audit.analyze_horizons(
    rows, "synthetic", "dev", data, models, torch.ones(st.STATE_DIM), "cpu", 8
  )
  families = {row["family"] for row in rows}
  assert {"world_model", "contact_force", "loading_memory",
          "contact_classification"} <= families
  assert {row["model"] for row in rows} >= {
    "M1", "M2-v1", "M2-v1/oracle-next-contact",
    "M2-v1/oracle-current-contact", "M2-v1/recurrent-estimator",
  }
  assert all(row["split"] == "dev" and row["n"] > 0 for row in rows)


def test_done_transition_is_excluded_from_one_step_counts():
  data = synthetic_data()
  # Dev contains environment ids 6 and 7: exclude one transition from each.
  data["done"][6, 1] = True
  data["done"][7, 2] = True
  models = {
    "M1": WorldModel((16,)),
    "M2-v1": ContactStructuredWorldModel((16,), foot_hidden=(8,)),
  }
  rows = []
  audit.analyze_one_step(
    rows, "synthetic", "dev", data, models,
    torch.ones(st.STATE_DIM), torch.ones(st.POSE_DELTA_DIM), "cpu", 8,
  )
  m1 = next(row for row in rows if row["model"] == "M1")
  assert m1["n"] == 6
