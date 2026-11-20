#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/yiwei/miniconda3/envs/go2-supermimic-rft/bin/python}
FORMAL_ROOT=${FORMAL_ROOT:-/home/yiwei/Documents/ChatGPT/go2-supermimic-v2/outputs/contact_structured_supermimic/formal_v2}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/contact_memory_gradient_diagnostics/p2_control_fidelity/v1}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

HANDOFF=handoff/contact_memory_diagnostics_seed14011
M1="$HANDOFF/checkpoints/m1/model_12000.pt"
M2="$HANDOFF/checkpoints/m2_v2/model_12000.pt"
NM="$HANDOFF/checkpoints/m2_v2_nomemory/model_12000.pt"
RIGID="$HANDOFF/trajectories/rigid_rough.pt"
SOFT="$HANDOFF/trajectories/soft_rough.pt"
RIGID_SHA=474807cbef7a82c8ce53066478c3b844d11561be16fc17a8bf8e8595e5e792ee
SOFT_SHA=eefefdc166781ed87b002c6ab0b7fdc64ab97681d4f0e9d5adf58f87a422afa3
RIGID_SEED=outputs/trained_real_diagnostics/actor_rollout_seeds_v1/rigid_stable_stance_seed82001.pt
SOFT_SEED=outputs/trained_real_diagnostics/actor_rollout_seeds_v1/soft_stable_stance_seed82011.pt

"$PYTHON_BIN" scripts/check_artifact_integrity.py "$HANDOFF/manifest.json"

"$PYTHON_BIN" scripts/run_action_jacobian_fidelity.py \
  --checkpoint "m1=$M1" --checkpoint "m2_v2=$M2" --checkpoint "m2_v2_nomemory=$NM" \
  --seed-bundle "$RIGID_SEED" --seed-bundle "$SOFT_SEED" \
  --dataset "rigid_rough=$RIGID=$RIGID_SHA" --dataset "soft_rough=$SOFT=$SOFT_SHA" \
  --output "$OUTPUT_ROOT/action_jacobian"

"$PYTHON_BIN" scripts/run_physical_policy_update_pilot.py \
  --checkpoint "m1=$M1" --checkpoint "m2_v2=$M2" --checkpoint "m2_v2_nomemory=$NM" \
  --seed-bundle "$RIGID_SEED" --seed-bundle "$SOFT_SEED" --physical-horizon 32 \
  --output "$OUTPUT_ROOT/physical_policy_update"

"$PYTHON_BIN" scripts/audit_contact_body_semantics.py --checkpoint "$M2" \
  --dataset "rigid_rough=$RIGID=$RIGID_SHA" --dataset "soft_rough=$SOFT=$SOFT_SHA" \
  --stride 16 --output "$OUTPUT_ROOT/contact_body_semantics"

"$PYTHON_BIN" scripts/analyze_state_action_distribution_shift.py \
  --checkpoint "m1=$M1" --checkpoint "m2_v2=$M2" --checkpoint "m2_v2_nomemory=$NM" \
  --dataset "rigid_rough=$RIGID=$RIGID_SHA" --dataset "soft_rough=$SOFT=$SOFT_SHA" \
  --sample-count 8192 --output "$OUTPUT_ROOT/distribution_shift"

"$PYTHON_BIN" scripts/audit_collapse_optimizer_history.py \
  --training-log "$FORMAL_ROOT/runtime/training.log" \
  --checkpoint "m1=$M1" --checkpoint "m2_v2=$M2" --checkpoint "m2_v2_nomemory=$NM" \
  --output "$OUTPUT_ROOT/collapse_optimizer"
