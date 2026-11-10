#!/usr/bin/env python3
"""Render the four-stage seed-14011 TensorBoard learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
  ("train/wm_loss", "World-model loss"),
  ("train/imagined_return", "Imagined return"),
  ("train/critic_loss", "Critic loss"),
  ("train/actor_grad_norm", "Actor gradient norm"),
  ("Metrics/rft/sinkage_m", "Mean sinkage (m)"),
  ("Metrics/rft/abs_power_w", "Mean |RFT power| (W/foot)"),
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  series = {tag: [] for tag, _ in TAGS}
  for event_file in args.training_dir.glob("*/events.out*"):
    events = EventAccumulator(str(event_file))
    events.Reload()
    available = set(events.Tags()["scalars"])
    for tag in series:
      if tag in available:
        series[tag].extend((e.step, e.value) for e in events.Scalars(tag))
  fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
  for ax, (tag, title) in zip(axes.flat, TAGS):
    values = sorted(series[tag])
    if values:
      ax.plot([x for x, _ in values], [y for _, y in values], linewidth=1.3)
    for boundary in (4000, 8000, 10000):
      ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Global training iteration")
    ax.grid(alpha=0.25)
  fig.suptitle("Seed 14011 staged soft-terrain training", fontsize=15)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=180)
  plt.close(fig)


if __name__ == "__main__":
  main()
