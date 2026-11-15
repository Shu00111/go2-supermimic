#!/usr/bin/env python3
"""Train one P1 world-model candidate on the frozen matched audit dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  ContactStructuredWorldModel,
  ContactStructuredWorldModelV2,
  count_parameters,
)
from go2_supermimic.supermimic.world_model import WorldModel


METHODS = ("m1", "m2", "m2_v2_core", "m2_v2")
DATA_KEYS = (
  "state", "next_state", "pose_delta", "action", "previous_action", "context",
  "relief", "contact", "next_contact", "done",
)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def git_head() -> str:
  return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 text=True).strip()


def split_ids(num_envs: int, split: str) -> torch.Tensor:
  allowed = set(range(6)) if split == "train" else {6, 7}
  return torch.tensor([i for i in range(num_envs) if i % 10 in allowed])


def load_split(paths: list[Path], split: str) -> tuple[dict[str, torch.Tensor], list[dict]]:
  parts: dict[str, list[torch.Tensor]] = {}
  metadata = []
  for path in paths:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    data = loaded["data"]
    ids = split_ids(data["state"].shape[0], split)
    for name in DATA_KEYS:
      value = data[name]
      parts.setdefault(name, []).append(value[ids])
    metadata.append({**loaded["metadata"], "path": str(path.resolve()), "sha256": sha256(path)})
  return {name: torch.cat(values) for name, values in parts.items()}, metadata


def valid_starts(done: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
  valid = ~done.unfold(1, horizon, 1).any(-1)
  return valid.nonzero(as_tuple=True)


def gather(data: dict[str, torch.Tensor], env: torch.Tensor, start: torch.Tensor,
           horizon: int, device: str) -> dict[str, torch.Tensor]:
  steps = start[:, None] + torch.arange(horizon, device=start.device)
  # The stored next_state is authoritative across each transition; valid starts
  # guarantee no auto-reset boundary is included.
  return {
    "s0": data["state"][env, start].to(device),
    "targets": data["next_state"][env[:, None], steps].to(device),
    "poses": data["pose_delta"][env[:, None], steps].to(device),
    "actions": data["action"][env[:, None], steps].to(device),
    "actions_prev": data["previous_action"][env[:, None], steps].to(device),
    "xi": data["context"][env, start].to(device),
    "reliefs": data["relief"][env[:, None], steps].float().to(device),
    "contact_initial": data["contact"][env, start].to(device),
    "contacts": data["next_contact"][env[:, None], steps].to(device),
    "valid": torch.ones(env.shape[0], horizon, device=device),
  }


def make_model(method: str, hidden: tuple[int, ...], device: str):
  if method == "m1":
    model = WorldModel(hidden)
  elif method == "m2":
    model = ContactStructuredWorldModel(hidden)
  elif method == "m2_v2_core":
    model = ContactStructuredWorldModelV2(hidden, anchor_weight=0.0)
  elif method == "m2_v2":
    model = ContactStructuredWorldModelV2(hidden)
  else:
    raise ValueError(method)
  return model.to(device)


@torch.no_grad()
def initialize_stats(model, train: dict[str, torch.Tensor], device: str, batch: int) -> None:
  valid = (~train["done"]).nonzero(as_tuple=True)
  for lo in range(0, valid[0].numel(), batch):
    env, step = valid[0][lo:lo + batch], valid[1][lo:lo + batch]
    model.update_stats(
      train["state"][env, step].to(device),
      train["action"][env, step].to(device),
      train["previous_action"][env, step].to(device),
      train["context"][env, step].to(device),
      train["relief"][env, step].float().to(device),
      train["next_state"][env, step].to(device),
      train["pose_delta"][env, step].to(device),
    )


def common_scale(train: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
  valid = ~train["done"]
  target = torch.cat([
    train["next_state"] - train["state"], train["pose_delta"]
  ], -1)[valid]
  scale = target.float().std(0, unbiased=False).clamp_min(1e-3)
  return scale[:st.STATE_DIM], scale[st.STATE_DIM:]


@torch.no_grad()
def evaluate(model, data: dict[str, torch.Tensor], state_scale: torch.Tensor,
             pose_scale: torch.Tensor, device: str, batch_size: int) -> dict[str, float]:
  state_scale, pose_scale = state_scale.to(device), pose_scale.to(device)
  valid = (~data["done"]).nonzero(as_tuple=True)
  one_sum = 0.0
  one_n = 0
  contact = {"normal": 0.0, "tangent": 0.0, "memory": 0.0, "wrong": 0.0, "n": 0}
  for lo in range(0, valid[0].numel(), batch_size):
    e, t = valid[0][lo:lo + batch_size], valid[1][lo:lo + batch_size]
    s = data["state"][e, t].to(device)
    a = data["action"][e, t].to(device)
    ap = data["previous_action"][e, t].to(device)
    xi = data["context"][e, t].to(device)
    relief = data["relief"][e, t].float().to(device)
    if isinstance(model, ContactStructuredWorldModel):
      z, _ = model.initial_contact(s, ap)
      pred_s, pred_pose, z = model.step(s, z, a, ap, xi, relief)
      target_z = data["next_contact"][e, t].to(device)
      contact["normal"] += float((z[..., 2] - target_z[..., 2]).abs().sum())
      contact["tangent"] += float((z[..., :2] - target_z[..., :2]).norm(dim=-1).sum())
      contact["memory"] += float((z[..., 4] - target_z[..., 4]).abs().sum())
      contact["wrong"] += float(((z[..., 7] >= 0.5) != (target_z[..., 7] >= 0.5)).sum())
      contact["n"] += int(z[..., 7].numel())
    else:
      pred_s, pred_pose = model(s, a, ap, xi, relief)
    target_s = data["next_state"][e, t].to(device)
    target_pose = data["pose_delta"][e, t].to(device)
    error = torch.cat([
      (pred_s - target_s) / state_scale, (pred_pose - target_pose) / pose_scale
    ], -1).square().mean(-1)
    one_sum += float(error.sum())
    one_n += error.numel()

  horizon = 8
  starts = valid_starts(data["done"], horizon)
  eight_sum = 0.0
  eight_n = 0
  for lo in range(0, starts[0].numel(), batch_size):
    e, t = starts[0][lo:lo + batch_size], starts[1][lo:lo + batch_size]
    state = data["state"][e, t].to(device)
    z = None
    if isinstance(model, ContactStructuredWorldModel):
      z, _ = model.initial_contact(state, data["previous_action"][e, t].to(device))
    for j in range(horizon):
      index = t + j
      a = data["action"][e, index].to(device)
      ap = data["previous_action"][e, index].to(device)
      xi = data["context"][e, index].to(device)
      relief = data["relief"][e, index].float().to(device)
      if isinstance(model, ContactStructuredWorldModel):
        state, _, z = model.step(state, z, a, ap, xi, relief)
      else:
        state, _ = model(state, a, ap, xi, relief)
    target = data["next_state"][e, t + horizon - 1].to(device)
    error = ((state - target) / state_scale).square().mean(-1)
    eight_sum += float(error.sum())
    eight_n += error.numel()
  metrics = {
    "one_step_common_normalized_mse": one_sum / one_n,
    "eight_step_common_normalized_state_mse": eight_sum / eight_n,
  }
  if contact["n"]:
    metrics.update({
      "contact_normal_mae_n": contact["normal"] / contact["n"],
      "contact_tangential_vector_mae_n": contact["tangent"] / contact["n"],
      "contact_memory_mae_m": contact["memory"] / contact["n"],
      "contact_accuracy": 1.0 - contact["wrong"] / contact["n"],
    })
  return metrics


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--method", choices=METHODS, required=True)
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=14011)
  parser.add_argument("--updates", type=int, default=1200)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--window", type=int, default=8)
  parser.add_argument("--hidden", type=int, nargs="+", default=(256, 256, 256))
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--eval-every", type=int, default=100)
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  paths = sorted(args.dataset_dir.glob("*.pt"))
  if not (args.dataset_dir / "COLLECTION_COMPLETE").exists() or len(paths) != 5:
    raise RuntimeError("expected the complete five-condition P1 dataset")
  train, metadata = load_split(paths, "train")
  dev, _ = load_split(paths, "dev")
  train_starts = valid_starts(train["done"], args.window)
  if not train_starts[0].numel():
    raise RuntimeError("no valid training windows")
  hidden = tuple(args.hidden)
  model = make_model(args.method, hidden, args.device)
  initialize_stats(model, train, args.device, args.batch_size * 2)
  state_scale, pose_scale = common_scale(train)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  generator = torch.Generator().manual_seed(args.seed + 991)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  log_path = args.output_dir / "training.jsonl"
  if log_path.exists():
    raise FileExistsError(f"refusing to overwrite controlled run: {log_path}")
  started = time.monotonic()
  best_score = math.inf
  best_state = None
  history = []
  for update in range(1, args.updates + 1):
    choice = torch.randint(
      0, train_starts[0].numel(), (args.batch_size,), generator=generator
    )
    batch = gather(
      train, train_starts[0][choice], train_starts[1][choice], args.window, args.device
    )
    loss_args = (
      batch["s0"], batch["actions"], batch["actions_prev"], batch["xi"],
      batch["reliefs"], batch["targets"], batch["poses"], batch["valid"],
    )
    if isinstance(model, ContactStructuredWorldModel):
      loss = model.loss(*loss_args, contact_targets=batch["contacts"],
                        contact_initial=batch["contact_initial"])
    else:
      loss = model.loss(*loss_args)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
      raise RuntimeError(f"non-finite update {update}: loss={loss} grad={grad_norm}")
    optimizer.step()
    if update == 1 or update % args.eval_every == 0 or update == args.updates:
      metrics = evaluate(model, dev, state_scale, pose_scale, args.device, args.batch_size * 2)
      score = metrics["one_step_common_normalized_mse"] + metrics[
        "eight_step_common_normalized_state_mse"
      ]
      record = {
        "update": update, "train_loss": float(loss.detach()),
        "grad_norm": float(grad_norm.detach()),
        "elapsed_s": time.monotonic() - started, "selection_score": score, **metrics,
      }
      history.append(record)
      with log_path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
      print(json.dumps(record), flush=True)
      if score < best_score:
        best_score = score
        best_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
  assert best_state is not None
  contract = {
    "method": args.method, "seed": args.seed, "updates": args.updates,
    "batch_size": args.batch_size, "window": args.window, "hidden": list(hidden),
    "lr": args.lr, "selection_rule": "minimum dev(one_step + eight_step state MSE)",
    "held_out_opened": False, "source_commit": git_head(), "datasets": metadata,
    "parameters": count_parameters(model), "wall_time_s": time.monotonic() - started,
    "best_selection_score": best_score,
  }
  checkpoint = args.output_dir / "best.pt"
  torch.save({"cfg": contract, "world_model": best_state, "history": history,
              "common_state_scale": state_scale, "common_pose_scale": pose_scale}, checkpoint)
  (args.output_dir / "contract.json").write_text(json.dumps({
    **contract, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)
  }, indent=2) + "\n")
  (args.output_dir / "TRAINING_COMPLETE").write_text("controlled P1 training complete\n")


if __name__ == "__main__":
  main()
