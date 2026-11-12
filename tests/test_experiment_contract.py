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
