"""Registers the chase / dodge tasks with mjlab's task registry on import."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from go2_supermimic.env.env_cfg import go2_chase_env_cfg

for task_id, kw in (
  ("Go2-Chase-Flat", {}),
  ("Go2-Chase-Obstacles", {"obstacles": True}),
  ("Go2-Chase-Terrain", {"terrain": True}),
  ("Go2-Chase-SoftTerrain", {"soft_terrain": True}),
  ("Go2-Chase-SoftTerrainLevels", {
    "soft_terrain": True, "soil_strength_range": (0.3, 1.0)
  }),
  ("Go2-Chase-SoftTransition", {"transition": "hard_to_soft"}),
  ("Go2-Chase-SoftToHard", {"transition": "soft_to_hard"}),
  ("Go2-Chase-RoughSoftTerrain", {"terrain": True, "soft_terrain": True}),
  ("Go2-Chase-AlternatingMaterial", {"material": "checkerboard"}),
  ("Go2-Chase-MixedMaterialDiagnostic", {
    "material": "checkerboard", "material_cell_size": 0.6, "mixed_diagnostic": True
  }),
  ("Go2-Chase-RoughSoftTerrain-Sim2Real", {
    "terrain": True, "soft_terrain": True, "sim2real": True
  }),
  ("Go2-Chase-AlternatingMaterial-Sim2Real", {
    "material": "checkerboard", "sim2real": True
  }),
):
  register_mjlab_task(
    task_id=task_id,
    env_cfg=go2_chase_env_cfg(**kw),
    play_env_cfg=go2_chase_env_cfg(play=True, **kw),
    rl_cfg=RslRlOnPolicyRunnerCfg(),  # unused: training is SuperMimic (scripts/train.py)
  )
