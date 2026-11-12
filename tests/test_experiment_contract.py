from dataclasses import asdict

from go2_supermimic.supermimic.train import TrainCfg


def test_shared_contract_differs_only_in_world_model_method_and_capacity():
  common = asdict(TrainCfg(device="cpu", num_envs=2, compile=False))
  configs = []
  for method, hidden in (("m1", (64, 64)), ("m2", (64, 64)), ("m3", (70, 64))):
    value = common | {"method": method, "wm_hidden": hidden}
    configs.append(value)
  ignored = {"method", "wm_hidden", "run_name"}
  baseline = {k: v for k, v in configs[0].items() if k not in ignored}
  assert all({k: v for k, v in c.items() if k not in ignored} == baseline for c in configs[1:])


def test_new_tasks_register_with_shared_action_interface_and_sim2real_contract():
  import go2_supermimic.env  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  names = (
    "Go2-Chase-RoughSoftTerrain",
    "Go2-Chase-AlternatingMaterial",
    "Go2-Chase-MixedMaterialDiagnostic",
  )
  assert set(names).issubset(list_tasks())
  for name in names:
    cfg = load_env_cfg(name)
    assert tuple(cfg.actions) == ("joint_pos", "rft")
    assert cfg.actions["joint_pos"].scale is not None
    assert cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10
  rough = load_env_cfg(names[0])
  assert rough.actions["rft"].use_local_height
  alternating = load_env_cfg(names[1])
  assert alternating.actions["rft"].soft_region == "checkerboard"
  assert "hard_patches" in alternating.scene.entities
  alternating_contact = next(
    sensor for sensor in alternating.scene.sensors if sensor.name == "feet_ground_contact"
  )
  assert alternating_contact.secondary.entity == "hard_patches"
  assert alternating_contact.secondary.mode == "body"
  assert alternating_contact.secondary_policy == "first"
  assert alternating.events["reset_hard_patches"].params["asset_cfg"].name == "hard_patches"
  assert "per_world_contact_modes" not in alternating.events

  transition = load_env_cfg("Go2-Chase-SoftTransition")
  transition_contact = next(
    sensor for sensor in transition.scene.sensors if sensor.name == "feet_ground_contact"
  )
  assert transition_contact.secondary.entity == "hard_ground"
  assert transition.events["reset_hard_ground"].params["asset_cfg"].name == "hard_ground"

  robust = load_env_cfg("Go2-Chase-AlternatingMaterial-Sim2Real")
  assert robust.observations["actor"].terms["base_ang_vel"].delay_max_lag == 1
  assert {"base_inertia", "link_inertia", "joint_friction"}.issubset(robust.events)
  assert robust.events["reset_robot_joints"].params["position_range"] == (-0.05, 0.05)
  clean_actuators = load_env_cfg("Go2-Chase-AlternatingMaterial").scene.entities["robot"].articulation.actuators
  robust_actuators = robust.scene.entities["robot"].articulation.actuators
  assert all(a.delay_hold_prob == 0.0 and a.delay_update_period == 0 for a in clean_actuators)
  assert all(a.delay_hold_prob == 0.2 and a.delay_update_period == 2 for a in robust_actuators)
