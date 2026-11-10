#!/usr/bin/env bash
# Restart-safe, single-GPU staged curriculum for the formal seed-14011 run.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SUPERMIMIC_PYTHON:-/home/yiwei/miniconda3/envs/go2-supermimic-rft/bin/python}"
run_root="${SUPERMIMIC_RUN_ROOT:-${repo_dir}/outputs/soft_terrain/seed_14011}"
runtime_dir="${run_root}/runtime"
train_root="${run_root}/training"
mkdir -p "${runtime_dir}" "${train_root}"

exec 9>"${runtime_dir}/launcher.lock"
if ! flock -n 9; then
  echo "A seed-14011 launcher already owns ${runtime_dir}/launcher.lock" >&2
  exit 2
fi

echo "$$" >"${runtime_dir}/launcher.pid"
printf '%q ' "$0" "$@" >"${runtime_dir}/launcher.command"
printf '\n' >>"${runtime_dir}/launcher.command"
date --iso-8601=seconds >"${runtime_dir}/started_at"
echo running >"${runtime_dir}/status"
trap 'code=$?; date --iso-8601=seconds >"${runtime_dir}/stopped_at"; if ((code==0)); then echo complete >"${runtime_dir}/status"; else echo "failed:${code}" >"${runtime_dir}/status"; fi' EXIT

latest_checkpoint() {
  find "${train_root}" -type f -name 'model_*.pt' -printf '%f\t%p\n' 2>/dev/null \
    | sed -E 's/^model_([0-9]+)\.pt\t/\1\t/' \
    | sort -n -k1,1 | tail -1 | cut -f2-
}

checkpoint_iteration() {
  local checkpoint="$1"
  if [[ -z "${checkpoint}" ]]; then echo 0; return; fi
  basename "${checkpoint}" | sed -E 's/^model_([0-9]+)\.pt$/\1/'
}

run_stage() {
  local stage="$1" task="$2" end_iteration="$3"
  local checkpoint iteration resume_args=()
  checkpoint="$(latest_checkpoint || true)"
  iteration="$(checkpoint_iteration "${checkpoint}")"
  if (( iteration >= end_iteration )); then
    echo "Stage ${stage} already complete at iteration ${iteration}."
    return
  fi
  if [[ -n "${checkpoint}" ]]; then
    resume_args=(--resume "${checkpoint}" --continue-run true)
  fi
  echo "Starting stage ${stage}: ${task}, ${iteration} -> ${end_iteration}"
  PYTHONNOUSERSITE=1 "${python_bin}" "${repo_dir}/scripts/train.py" \
    --task "${task}" --device cuda:0 --seed 14011 \
    --num-envs 512 --iterations "${end_iteration}" \
    --wm-batch 2048 --imag-batch 4096 \
    --log-interval 50 --save-interval 250 \
    --run-name "seed14011_stage_${stage}" --log-root "${train_root}" \
    "${resume_args[@]}"
}

run_stage A Go2-Chase-SoftTerrain 4000
run_stage B Go2-Chase-SoftTerrainLevels 8000
run_stage C1 Go2-Chase-SoftTransition 10000
run_stage C2 Go2-Chase-SoftToHard 12000

latest_checkpoint >"${runtime_dir}/final_checkpoint"
date --iso-8601=seconds >"${runtime_dir}/completed_at"
