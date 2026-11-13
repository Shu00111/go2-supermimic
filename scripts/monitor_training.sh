#!/usr/bin/env bash
set -euo pipefail
run_dir="${1:?usage: monitor_training.sh RUN_DIR}"
interval="${MONITOR_INTERVAL_SECONDS:-2400}"
mkdir -p "$run_dir/runtime"
while true; do
  now="$(date -u +%FT%TZ)"
  pid="$(pgrep -f "scripts/train.py.*--log-root .*${run_dir}" | head -1 || true)"
  checkpoint="$(find "$run_dir" -name 'model_*.pt' -type f -print0 2>/dev/null | xargs -0 -r ls -t 2>/dev/null | head -1 || true)"
  checkpoint_age=""
  if [[ -n "$checkpoint" ]]; then
    checkpoint_age="$(( $(date +%s) - $(stat -c %Y "$checkpoint") ))"
  fi
  last_line="$(grep -hE 'it [0-9]+/[0-9]+' "$run_dir"/runtime/*.log 2>/dev/null | tail -1 || true)"
  iteration="$(sed -nE 's/.*it ([0-9]+)\/.*/\1/p' <<<"$last_line")"
  throughput="$(sed -nE 's/.*env_steps ([0-9.]+M).*/\1/p' <<<"$last_line")"
  gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true)"
  disk="$(df -h "$run_dir" | tail -1)"
  bad="$(find "$run_dir" -type f -name '*.log' -mmin -45 -exec grep -Eil 'nan|inf|traceback' {} + 2>/dev/null | head -5 | tr '\n' ';' || true)"
  printf '{"time":"%s","pid":"%s","iteration":"%s","env_steps":"%s","checkpoint":"%s","checkpoint_age_seconds":"%s","gpu":"%s","disk":"%s","alerts":"%s"}\n' \
    "$now" "$pid" "$iteration" "$throughput" "$checkpoint" "$checkpoint_age" \
    "$gpu" "$disk" "$bad" >> "$run_dir/runtime/watchdog.jsonl"
  sleep "$interval"
done
