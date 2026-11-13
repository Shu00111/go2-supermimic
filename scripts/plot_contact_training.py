#!/usr/bin/env python3
"""Plot cumulative formal M1/M2 scalar curves across restartable segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
  "train/wm_loss", "train/imagined_return", "train/actor_grad_norm",
  "train/critic_loss", "Episode_Termination/fell_over",
)


def load(root: Path, method: str) -> dict[str, list[tuple[int, float]]]:
  latest: dict[str, dict[int, tuple[float, float]]] = {tag: {} for tag in TAGS}
  for path in root.glob(f"seed14011/{method}/**/events.out.tfevents.*"):
    events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    available = set(events.Tags().get("scalars", ()))
    for tag in TAGS:
      if tag not in available:
        continue
      for event in events.Scalars(tag):
        old = latest[tag].get(event.step)
        if old is None or event.wall_time > old[0]:
          latest[tag][event.step] = (event.wall_time, event.value)
  return {tag: [(step, value) for step, (_, value) in sorted(values.items())]
          for tag, values in latest.items()}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  curves = {method: load(args.training_root, method) for method in ("m1", "m2")}
  fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
  for ax, tag in zip(axes.flat, TAGS):
    for method, color in (("m1", "tab:blue"), ("m2", "tab:orange")):
      points = curves[method][tag]
      if points:
        x, y = zip(*points)
        ax.plot(x, y, label=method.upper(), color=color, linewidth=1.3)
    for boundary in (4000, 8000):
      ax.axvline(boundary, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_title(tag)
    ax.set_xlabel("cumulative training iteration")
    ax.grid(alpha=0.25)
  axes.flat[0].legend()
  axes.flat[-1].axis("off")
  fig.suptitle("Formal seed 14011 training curves (stage boundaries at 4k/8k)")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=180)
  plt.close(fig)


if __name__ == "__main__":
  main()
