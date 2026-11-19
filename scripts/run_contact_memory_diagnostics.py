#!/usr/bin/env python3
"""Run offline contact-memory gradient diagnostics.

With --synthetic-fixture this exercises the complete numerical pipeline using
small, untrained instances of the real model classes.  Those rows are marked
UNTRAINED_MODEL_SMOKE_TEST and must never be used as trained-policy evidence.
The same CSV schemas are intended for frozen checkpoint/trajectory execution
once those bytes are supplied locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from go2_supermimic.diagnostics.contact_memory import (
  contact_step_with_intervention,
  extended_state_layout,
  finite_difference_directional,
  flatten_grads,
  jvp_vjp_identity,
  nonfinite_count,
  safe_cosine,
  spectral_norm_matrix_free,
)
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import (
  CONTACT_DIM, NUM_FEET, ContactStructuredWorldModelV2, normalize_contact,
)
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target.terrain import SCAN_POINTS

HORIZONS = (1, 2, 4, 8, 16, 32)
EVIDENCE = "UNTRAINED_MODEL_SMOKE_TEST"


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fields = list(rows[0]) if rows else ["status"]
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows or [{"status": "NOT MEASURED"}])


def git(*args: str) -> str:
  return subprocess.check_output(["git", *args], text=True).strip()


def checkpoint_audit(repo: Path, roots: list[Path]) -> list[dict]:
  protocol = json.loads((repo / "experiments/contact_structured_supermimic/reports/"
                         "stage4_v2_seed14011/protocol.json").read_text())
  candidates = []
  for root in roots:
    if root.exists():
      candidates.extend(root.glob("**/model_12000.pt"))
      candidates.extend(root.glob("**/best.pt"))
  unique = sorted(set(path.resolve() for path in candidates if path.is_file()))
  by_hash = {}
  for path in unique:
    by_hash.setdefault(sha256(path), path)
  rows = []
  for label, item in protocol["methods"].items():
    expected = item["sha256"]
    found = by_hash.get(expected)
    rows.append({
      "method": label.replace("_", "-"),
      "expected_sha256": expected,
      "actual_sha256": expected if found else "NOT AVAILABLE",
      "verified": bool(found),
      "local_path": str(found) if found else "NOT AVAILABLE",
      "recorded_training_path": item["checkpoint"],
      "evidence_class": "TRAINED_POLICY_EVIDENCE" if found else "NOT MEASURED",
    })
  return rows


def fixture(seed: int, dtype: torch.dtype):
  torch.manual_seed(seed)
  batch = 1
  s = torch.randn(batch, st.STATE_DIM, dtype=dtype) * 0.03
  s[:, st.GRAV] = torch.tensor([0.0, 0.0, -1.0], dtype=dtype)
  a_prev = torch.randn(batch, st.NUM_JOINTS, dtype=dtype) * 0.02
  xi = torch.tensor([[0.8, 0.0, 0.0, 0.0]], dtype=dtype)
  relief = torch.randn(batch, SCAN_POINTS, dtype=dtype) * 0.005
  actor = Actor(st.STATE_DIM + st.NUM_JOINTS, (24, 24), st.NUM_JOINTS).to(dtype)
  m1 = WorldModel((32, 32)).to(dtype)
  full = ContactStructuredWorldModelV2((32, 32), (16,), anchor_weight=0.25).to(dtype)
  nomemory = ContactStructuredWorldModelV2(
    (32, 32), (16,), anchor_weight=0.25, use_loading_memory=False,
  ).to(dtype)
  # Matched initialization except for the architecture's deterministic mask.
  nomemory.load_state_dict(full.state_dict())
  return s, a_prev, xi, relief, actor, {"M1": m1, "M2-v2": full,
                                       "No-Memory": nomemory}


def pack(model, s, a_prev):
  if isinstance(model, ContactStructuredWorldModelV2):
    z = model.initial_contact(s, a_prev)[0]
    return torch.cat([s, z.reshape(s.shape[0], -1), a_prev], -1)
  return torch.cat([s, a_prev], -1)


def closed_loop(model, actor, xi, relief, steps: int, *, intervention_k: int | None = None):
  layout = extended_state_layout(isinstance(model, ContactStructuredWorldModelV2))

  def transition(x):
    state = x[:, layout.physical]
    previous = x[:, layout.previous_action]
    action = torch.tanh(actor(torch.cat([state, previous], -1)))
    if layout.contact is None:
      state_next, _ = model(state, action, previous, xi, relief)
      return torch.cat([state_next, action], -1)
    contact = x[:, layout.contact].reshape(-1, NUM_FEET, CONTACT_DIM)
    state_next, _, contact_next = contact_step_with_intervention(
      model, state, contact, action, previous, xi, relief,
      detach_memory=intervention_k == 1,
    )
    return torch.cat([state_next, contact_next.reshape(state.shape[0], -1), action], -1)

  def rollout(x):
    value = x
    for index in range(steps):
      if intervention_k and intervention_k > 1 and index and index % intervention_k == 0:
        # Only the historical loading-memory coordinate is truncated.  The
        # current forward value and every other state coordinate are unchanged.
        z = value[:, layout.contact].reshape(-1, NUM_FEET, CONTACT_DIM)
        z = torch.cat([z[..., :4], z[..., 4:5].detach(), z[..., 5:]], -1)
        value = torch.cat([value[:, layout.physical], z.reshape(value.shape[0], -1),
                           value[:, layout.previous_action]], -1)
      value = transition(value)
    return value
  return transition, rollout


def jacobian_diagnostics(models, actor, s, a_prev, xi, relief, seed: int):
  spectrum, growth = [], []
  for label, model in models.items():
    x = pack(model, s, a_prev).detach()
    one, _ = closed_loop(model, actor, xi, relief, 1)
    check_v = torch.randn_like(x)
    check_w = torch.randn_like(x)
    identity = jvp_vjp_identity(one, x, check_v, check_w)
    local = spectral_norm_matrix_free(one, x, iterations=10, seed=seed)
    spectrum.append({
      "method": label, "phase": "synthetic_smooth_stance", "sample": 0,
      "spectral_norm": local, "jvp_vjp_relative_error": identity.relative_error,
      "precision": str(x.dtype).removeprefix("torch."), "nonfinite_count": 0,
      "evidence_class": EVIDENCE,
    })
    for horizon in HORIZONS:
      _, fn = closed_loop(model, actor, xi, relief, horizon)
      norm = spectral_norm_matrix_free(fn, x, iterations=10, seed=seed + horizon)
      growth.append({
        "method": label, "phase": "synthetic_smooth_stance", "horizon": horizon,
        "jacobian_product_norm": norm,
        "log_amplification_rate": math.log(max(norm, 1e-300)) / horizon,
        "relative_to_one_step_power": norm / max(local ** horizon, 1e-300),
        "sample_count": 1, "failure_count": 0, "censoring_count": 0,
        "evidence_class": EVIDENCE,
      })
  return spectrum, growth


def actor_objective(model, actor, x, xi, relief, horizon, intervention_k=None):
  layout = extended_state_layout(isinstance(model, ContactStructuredWorldModelV2))
  total = x.new_zeros(())
  value = x
  contributions = []
  for index in range(horizon):
    if (layout.contact is not None and intervention_k and intervention_k > 1
        and index and index % intervention_k == 0):
      z = value[:, layout.contact].reshape(-1, NUM_FEET, CONTACT_DIM)
      z = torch.cat([z[..., :4], z[..., 4:5].detach(), z[..., 5:]], -1)
      value = torch.cat([value[:, layout.physical], z.reshape(value.shape[0], -1),
                         value[:, layout.previous_action]], -1)
    _, step = closed_loop(
      model, actor, xi, relief, 1,
      intervention_k=1 if intervention_k == 1 else None,
    )
    value = step(value)
    state = value[:, layout.physical]
    reward = state[:, st.V_YAW.start] - 0.05 * state[:, st.Q].square().mean(-1)
    weighted = (0.99 ** index) * reward.mean()
    contributions.append(weighted)
    total = total + weighted
  return total, contributions


def first_action_gradient(model, actor, x, xi, relief, horizon):
  """Architecture-independent derivative with respect to the common first action."""
  layout = extended_state_layout(isinstance(model, ContactStructuredWorldModelV2))
  state = x[:, layout.physical]
  previous = x[:, layout.previous_action]
  action = torch.tanh(actor(torch.cat([state, previous], -1))).detach().requires_grad_(True)
  if layout.contact is None:
    state_next, _ = model(state, action, previous, xi, relief)
    value = torch.cat([state_next, action], -1)
  else:
    contact = x[:, layout.contact].reshape(-1, NUM_FEET, CONTACT_DIM)
    state_next, _, contact_next = model.step(state, contact, action, previous, xi, relief)
    value = torch.cat([state_next, contact_next.reshape(state.shape[0], -1), action], -1)
  objective = state_next[:, st.V_YAW.start].mean()
  for index in range(1, horizon):
    _, step = closed_loop(model, actor, xi, relief, 1)
    value = step(value)
    next_state = value[:, layout.physical]
    objective = objective + (0.99 ** index) * (
      next_state[:, st.V_YAW.start] - 0.05 * next_state[:, st.Q].square().mean(-1)
    ).mean()
  gradient = torch.autograd.grad(objective, action)[0]
  return float(gradient.norm()), nonfinite_count(gradient)


def actor_gradient_diagnostics(models, actor0, s, a_prev, xi, relief):
  rows, vectors = [], {}
  initial = {label: pack(model, s, a_prev).detach() for label, model in models.items()}
  state0 = {key: value.detach().clone() for key, value in actor0.state_dict().items()}
  for label, model in models.items():
    for horizon in HORIZONS:
      actor = Actor(st.STATE_DIM + st.NUM_JOINTS, (24, 24), st.NUM_JOINTS).to(s.dtype)
      actor.load_state_dict(state0)
      objective, contributions = actor_objective(
        model, actor, initial[label], xi, relief, horizon,
      )
      actor.zero_grad(set_to_none=True)
      objective.backward(retain_graph=True)
      gradient = flatten_grads(actor.parameters()).detach()
      first_norm, first_nonfinite = first_action_gradient(
        model, actor, initial[label], xi, relief, horizon,
      )
      vectors[(label, horizon)] = gradient
      per_time = []
      for contribution in contributions:
        actor.zero_grad(set_to_none=True)
        contribution.backward(retain_graph=True)
        per_time.append(float(flatten_grads(actor.parameters()).norm()))
      rows.append({
        "method": label, "horizon": horizon, "imagined_return": float(objective.detach()),
        "actor_gradient_l2": float(gradient.norm()),
        "first_action_gradient_l2": first_norm,
        "gradient_cosine_vs_h1": safe_cosine(gradient, vectors[(label, 1)]),
        "gradient_cosine_vs_no_memory": "PENDING",
        "per_timestep_gradient_l2": json.dumps(per_time),
        "clip_threshold": 1.0, "would_clip": float(gradient.norm()) > 1.0,
        "near_zero": float(gradient.norm()) < 1e-12,
        "nonfinite_count": nonfinite_count(gradient) + first_nonfinite, "sample_count": 1,
        "evidence_class": EVIDENCE,
      })
  for row in rows:
    key = (row["method"], row["horizon"])
    row["gradient_cosine_vs_no_memory"] = safe_cosine(
      vectors[key], vectors[("No-Memory", row["horizon"])])
  return rows, vectors


def intervention_diagnostics(model, actor0, s, a_prev, xi, relief):
  rows = []
  x = pack(model, s, a_prev).detach()
  state0 = {key: value.detach().clone() for key, value in actor0.state_dict().items()}
  for horizon in HORIZONS:
    reference = None
    for label, k in (("original", None), ("memory_backward_blocked", 1),
                     ("memory_truncated_k2", 2), ("memory_truncated_k4", 4),
                     ("memory_truncated_k8", 8)):
      actor = Actor(st.STATE_DIM + st.NUM_JOINTS, (24, 24), st.NUM_JOINTS).to(s.dtype)
      actor.load_state_dict(state0)
      objective, _ = actor_objective(model, actor, x, xi, relief, horizon, k)
      actor.zero_grad(set_to_none=True)
      objective.backward()
      gradient = flatten_grads(actor.parameters()).detach()
      if reference is None:
        reference = gradient
      rows.append({
        "intervention": label, "horizon": horizon,
        "forward_return": float(objective.detach()), "actor_gradient_l2": float(gradient.norm()),
        "gradient_cosine_vs_original": safe_cosine(gradient, reference),
        "forward_value_preserved_vs_original": "CHECK_IN_TEST",
        "nonfinite_count": nonfinite_count(gradient), "evidence_class": EVIDENCE,
      })
    original_return = rows[-5]["forward_return"]
    for row in rows[-5:]:
      row["forward_value_preserved_vs_original"] = math.isclose(
        row["forward_return"], original_return, rel_tol=0.0, abs_tol=1e-12
      )
  return rows


def term_gradients(model, s, a_prev, xi, relief, horizon: int):
  z, _ = model.initial_contact(s, a_prev)
  state = s
  actions = [torch.tanh(a_prev + 0.01 * (index + 1)) for index in range(horizon)]
  with torch.no_grad():
    oracle_initial = z.detach().clone()
    targets = []
    state_targets = []
    pose_targets = []
    current_z, current_s = z.detach(), s.detach()
    for action in actions:
      current_s, pose, current_z = model.step(current_s, current_z, action, a_prev, xi, relief)
      state_targets.append(current_s.detach() + 0.001)
      pose_targets.append(pose.detach())
      contact_target = current_z.detach().clone()
      contact_target[..., :3] += contact_target.new_tensor([2.0, -1.0, 3.0])
      contact_target[..., 3] += 0.002
      contact_target[..., 4] += 0.001
      contact_target[..., 5:7] += contact_target.new_tensor([0.02, -0.01])
      contact_target[..., 7] = (contact_target[..., 7] + 0.08).clamp(max=0.98)
      targets.append(contact_target)
  losses = {name: s.new_zeros(()) for name in (
    "robot_state_prediction", "contact_force", "contact_classification",
    "intrusion", "loading_memory", "physical_anchoring",
  )}
  oracle_current = oracle_initial
  for index, action in enumerate(actions):
    next_state, pose, z = model.step(state, z, action, a_prev, xi, relief)
    target = targets[index]
    losses["robot_state_prediction"] += (
      torch.cat([next_state - state_targets[index], pose - pose_targets[index]], -1)
      / model.out_norm.std()
    ).square().mean()
    pred_n = normalize_contact(z, model.scales)
    target_n = normalize_contact(target, model.scales)
    losses["contact_force"] += torch.nn.functional.smooth_l1_loss(
      pred_n[..., :3], target_n[..., :3]
    )
    losses["intrusion"] += torch.nn.functional.smooth_l1_loss(pred_n[..., 3], target_n[..., 3])
    losses["loading_memory"] += torch.nn.functional.smooth_l1_loss(
      pred_n[..., 4], target_n[..., 4]
    )
    losses["contact_classification"] += torch.nn.functional.binary_cross_entropy(
      z[..., 7].clamp(1e-6, 1 - 1e-6), target[..., 7]
    )
    teacher_s = s if index == 0 else state_targets[index - 1]
    anchor_z = model._predict_contact(teacher_s, oracle_current, action, xi)
    anchor_s, anchor_pose = model._predict_body(
      teacher_s, target, action, a_prev, xi, relief,
    )
    losses["physical_anchoring"] += model.anchor_weight * (
      (torch.cat([anchor_s - state_targets[index], anchor_pose - pose_targets[index]], -1)
       / model.out_norm.std()).square().mean() + model._contact_loss(anchor_z, target).mean()
    )
    oracle_current, state = target, next_state
  named = dict(model.named_parameters())
  result = {}
  for name, loss in losses.items():
    gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True,
                                    allow_unused=True)
    result[name] = {parameter: gradient.detach() for parameter, gradient in zip(named, gradients)
                    if gradient is not None}
  return losses, result


def gradient_conflict(model, s, a_prev, xi, relief):
  rows = []
  for horizon in (1, 2, 4, 8):
    losses, grads = term_gradients(model, s, a_prev, xi, relief, horizon)
    for left, left_map in grads.items():
      for right, right_map in grads.items():
        shared = sorted(set(left_map) & set(right_map))
        ga = torch.cat([left_map[name].reshape(-1) for name in shared]) if shared else s.new_empty(0)
        gb = torch.cat([right_map[name].reshape(-1) for name in shared]) if shared else s.new_empty(0)
        rows.append({
          "horizon": horizon, "phase": "synthetic_smooth_stance",
          "loss_i": left, "loss_j": right, "shared_parameter_tensors": len(shared),
          "gradient_cosine": safe_cosine(ga, gb), "gradient_norm_i": float(ga.norm()),
          "gradient_norm_j": float(gb.norm()),
          "weighted_loss_i": float(losses[left].detach()),
          "weighted_loss_j": float(losses[right].detach()),
          "negative_cosine": safe_cosine(ga, gb) < 0,
          "evidence_class": EVIDENCE,
        })
  return rows


def numerical_checks(dtype: torch.dtype, seed: int) -> dict:
  torch.manual_seed(seed)
  x = torch.randn(9, dtype=dtype)
  direction = torch.randn_like(x)
  fn = lambda value: torch.sin(value) + 0.2 * value.square()
  _, jvp = torch.func.jvp(fn, (x,), (direction,))
  estimates = finite_difference_directional(fn, x, direction, (1e-2, 1e-3, 1e-4, 1e-5))
  errors = {str(epsilon): float((estimate - jvp).norm() / jvp.norm())
            for epsilon, estimate in estimates.items()}
  return {"fixture": "smooth_elementwise", "dtype": str(dtype), "relative_errors": errors,
          "converges_before_roundoff": min(errors.values()) < errors["0.01"]}


def plots(output: Path, growth: list[dict], actor_rows: list[dict], conflict: list[dict]):
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  for method in ("M1", "M2-v2", "No-Memory"):
    data = [row for row in growth if row["method"] == method]
    axes[0].plot([row["horizon"] for row in data],
                 [row["jacobian_product_norm"] for row in data], marker="o", label=method)
    data = [row for row in actor_rows if row["method"] == method]
    axes[1].plot([row["horizon"] for row in data],
                 [row["actor_gradient_l2"] for row in data], marker="o", label=method)
  for axis, title, ylabel in ((axes[0], "Finite-horizon Jacobian norm", "estimated spectral norm"),
                              (axes[1], "Actor gradient norm", "L2 norm")):
    axis.set_xscale("log", base=2); axis.set_yscale("log")
    axis.set_xlabel("horizon"); axis.set_ylabel(ylabel); axis.set_title(title); axis.legend()
  fig.suptitle("UNTRAINED MODEL SMOKE TEST — NOT SCIENTIFIC EVIDENCE", color="darkred")
  fig.tight_layout(); fig.savefig(output / "jacobian_analysis.png", dpi=180); plt.close(fig)
  # The requested actor plot is kept separate even though the combined plot also shows it.
  fig, ax = plt.subplots(figsize=(6, 4))
  for method in ("M1", "M2-v2", "No-Memory"):
    data = [row for row in actor_rows if row["method"] == method]
    ax.plot([row["horizon"] for row in data], [row["actor_gradient_l2"] for row in data],
            marker="o", label=method)
  ax.set(xscale="log", yscale="log", xlabel="horizon", ylabel="actor gradient L2",
         title="UNTRAINED MODEL SMOKE TEST")
  ax.legend(); fig.tight_layout(); fig.savefig(output / "actor_gradient_analysis.png", dpi=180)
  plt.close(fig)
  names = sorted(set(row["loss_i"] for row in conflict))
  h8 = {(row["loss_i"], row["loss_j"]): row["gradient_cosine"]
        for row in conflict if row["horizon"] == 8}
  matrix = [[h8.get((left, right), float("nan")) for right in names] for left in names]
  fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
  ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
  ax.set_yticks(range(len(names)), names); ax.set_title("H=8 untrained-model gradient cosine")
  fig.colorbar(image, ax=ax); fig.tight_layout()
  fig.savefig(output / "gradient_conflict_heatmap.png", dpi=180); plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
  parser.add_argument("--seed", type=int, default=20261119)
  parser.add_argument("--synthetic-fixture", action="store_true")
  args = parser.parse_args()
  if not args.synthetic_fixture:
    raise SystemExit("real-checkpoint runner is fail-closed until frozen checkpoint bytes and "
                     "recorded trajectories are provided; use --synthetic-fixture to test plumbing")
  start = time.perf_counter(); repo = Path(__file__).resolve().parents[1]
  output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
  dtype = torch.float64
  checkpoints = checkpoint_audit(repo, args.checkpoint_root)
  write_csv(output / "checkpoint_provenance.csv", checkpoints)
  s, a_prev, xi, relief, actor, models = fixture(args.seed, dtype)
  spectrum, growth = jacobian_diagnostics(models, actor, s, a_prev, xi, relief, args.seed)
  actor_rows, _ = actor_gradient_diagnostics(models, actor, s, a_prev, xi, relief)
  interventions = intervention_diagnostics(models["M2-v2"], actor, s, a_prev, xi, relief)
  conflicts = gradient_conflict(models["M2-v2"], s, a_prev, xi, relief)
  write_csv(output / "jacobian_spectrum.csv", spectrum)
  write_csv(output / "jacobian_horizon_growth.csv", growth)
  write_csv(output / "actor_gradient_horizon.csv", actor_rows)
  write_csv(output / "memory_gradient_interventions.csv", interventions)
  write_csv(output / "loss_gradient_conflict.csv", conflicts)
  write_csv(output / "policy_update_fidelity.csv", [{
    "status": "BLOCKED — REQUIRES VALIDATED LINUX BACKEND",
    "reason": "no frozen checkpoint bytes, paired snapshots, or validated full-state restore adapter",
  }])
  plots(output, growth, actor_rows, conflicts)
  receipt = {
    "schema": 1, "evidence_class": EVIDENCE, "source_commit": git("rev-parse", "HEAD"),
    "source_dirty": bool(git("status", "--porcelain")), "seed": args.seed,
    "precision": "float64", "device": "cpu", "torch": torch.__version__,
    "python": platform.python_version(), "platform": platform.platform(),
    "trajectory_sha256": hashlib.sha256(torch.cat([s.flatten(), a_prev.flatten(), xi.flatten(),
                                                     relief.flatten()]).numpy().tobytes()).hexdigest(),
    "numerical_checks": numerical_checks(dtype, args.seed),
    "runtime_seconds": time.perf_counter() - start,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
    "sample_counts": {"synthetic_trajectories": 1, "training_checkpoints": 0},
    "failure_counts": {"nonfinite": sum(row["nonfinite_count"] for row in actor_rows)},
    "censoring_counts": {"nonsmooth_contact": 0},
    "checkpoint_audit": checkpoints,
  }
  (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps({"output": str(output), "runtime_seconds": receipt["runtime_seconds"],
                    "evidence_class": EVIDENCE}, indent=2))


if __name__ == "__main__":
  main()
