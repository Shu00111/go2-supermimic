import torch
from types import SimpleNamespace

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  CONTACT_DIM,
  ESTIMATOR_STATE_DIM,
  NUM_FEET,
  ContactStructuredWorldModel,
  PersistentContactEstimator,
  per_foot_observables,
)
from go2_supermimic.supermimic.contact_labels import rigid_contact_force


def test_rigid_contact_force_reorders_mjcf_primaries_to_canonical_feet():
  # MJCF/entity order is FL, FR, RL, RR, while all contact-structured tensors
  # use FR, FL, RR, RL.  Give each source foot a distinct x force.
  sensor = SimpleNamespace(
    primary_names=[
      "FL_foot_collision", "FR_foot_collision",
      "RL_foot_collision", "RR_foot_collision",
    ],
    data=SimpleNamespace(force=torch.tensor([[[1., 0., 0.], [2., 0., 0.],
                                              [3., 0., 0.], [4., 0., 0.]]])),
  )
  env = SimpleNamespace(num_envs=1, scene={"feet_ground_contact": sensor})
  force = rigid_contact_force(env, torch.zeros(1, 4, 3))
  assert force[0, :, 0].tolist() == [2.0, 1.0, 4.0, 3.0]
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target.terrain import SCAN_POINTS


def batch(n=4):
  torch.manual_seed(7)
  s = torch.randn(n, st.STATE_DIM) * 0.1
  a = torch.randn(n, st.NUM_JOINTS) * 0.1
  xi = torch.randn(n, st.CONTEXT_DIM) * 0.1
  relief = torch.randn(n, SCAN_POINTS) * 0.02
  return s, a, xi, relief


def test_estimator_shapes_reset_and_no_oracle_inputs():
  s, a, _, _ = batch()
  # Oracle-only channels must not influence estimator inputs.
  x = per_foot_observables(s, a)
  s2 = s.clone()
  s2[:, st.SOIL_MEMORY] = 999
  s2[:, st.SOIL_DISTANCE] = -999
  torch.testing.assert_close(x, per_foot_observables(s2, a))
  estimator = PersistentContactEstimator()
  state = estimator.zero_state(len(s), s)
  z, state1 = estimator(s, a, state)
  assert z.shape == (len(s), NUM_FEET, CONTACT_DIM)
  assert state1.shape == (len(s), NUM_FEET, ESTIMATOR_STATE_DIM)
  reset = estimator.reset(state1, torch.tensor([True, False, True, False]))
  assert reset[[0, 2]].count_nonzero() == 0
  torch.testing.assert_close(reset[[1, 3]], state1[[1, 3]])


def test_m1_and_m2_forward_backward_are_finite():
  s, a, xi, relief = batch(3)
  for model in (WorldModel((32, 32)), ContactStructuredWorldModel((32, 32), (16,))):
    action = a.clone().requires_grad_(True)
    s1, dp = model(s, action, a, xi, relief)
    assert s1.shape == s.shape and dp.shape == (3, st.POSE_DELTA_DIM)
    loss = s1.square().mean() + dp.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0


def test_policy_gradient_flows_through_predicted_contacts():
  s, a, xi, relief = batch(2)
  model = ContactStructuredWorldModel((32, 32), (16,))
  action = a.clone().requires_grad_(True)
  z, _ = model.initial_contact(s, a)
  s1, _, z1 = model.step(s, z, action, a, xi, relief)
  loss = s1[:, st.V_YAW].sum()
  grad_z = torch.autograd.grad(loss, z1, retain_graph=True)[0]
  loss.backward()
  assert torch.isfinite(grad_z).all() and grad_z.abs().sum() > 0
  assert action.grad is not None and action.grad.abs().sum() > 0
  assert model.contact_transition[-1].weight.grad is not None


def test_autoregressive_loss_and_contact_alignment():
  s, a, xi, relief = batch(2)
  k = 3
  model = ContactStructuredWorldModel((32, 32), (16,))
  actions = a[:, None].expand(-1, k, -1).clone()
  targets = s[:, None].expand(-1, k, -1).clone()
  poses = torch.zeros(2, k, st.POSE_DELTA_DIM)
  contacts = torch.zeros(2, k, NUM_FEET, CONTACT_DIM)
  valid = torch.ones(2, k)
  loss = model.loss(s, actions, actions, xi, relief[:, None].expand(-1, k, -1),
                    targets, poses, valid, contacts)
  loss.backward()
  assert torch.isfinite(loss)
  assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_short_imagined_contact_rollout_has_no_detach():
  s, a, xi, relief = batch(2)
  model = ContactStructuredWorldModel((32, 32), (16,))
  action = a.clone().requires_grad_(True)
  z, _ = model.initial_contact(s, a)
  for _ in range(4):
    s, _, z = model.step(s, z, action, a, xi, relief)
  objective = s[:, st.V_YAW].square().mean() + z.square().mean() * 1e-4
  objective.backward()
  assert action.grad is not None and torch.isfinite(action.grad).all()
  assert action.grad.abs().sum() > 0
