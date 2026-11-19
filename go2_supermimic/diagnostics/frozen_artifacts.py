"""Verified loading and provenance helpers for frozen SuperMimic artifacts.

Historical checkpoints use the original ``torch.save`` training format, so
their trusted SHA-256 is checked before deserialization. Exact state-dict
schemas are then checked before any parameters are loaded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.contact_model import ContactStructuredWorldModelV2
from go2_supermimic.supermimic.train import Actor, Critic
from go2_supermimic.supermimic.world_model import WorldModel


EXPECTED_SHA256 = {
  "m1": "b3a76b694c7ddad1e5897ceeaa01040b3b2587a11143f36fd7a324813f658179",
  "m2_v2": "3d322d8f7ffe68f50f5b6f5573dd48af8dc8dd15220b4446802691469f117c19",
  "m2_v2_nomemory": "023b0eed06eb5d8bfb31f3e76c504ea6cd0ea3ae861e3839e4cfc3c49326cbee",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(block_size), b""):
      digest.update(block)
  return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> dict[str, Any]:
  path = path.resolve(strict=True)
  actual = sha256_file(path)
  if actual != expected_sha256:
    raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
  return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def _tensor_schema(state: dict[str, torch.Tensor]) -> dict[str, tuple[tuple[int, ...], str]]:
  return {key: (tuple(value.shape), str(value.dtype)) for key, value in state.items()}


def assert_exact_schema(name: str, expected: dict[str, torch.Tensor], actual: dict[str, Any]) -> None:
  expected_schema, actual_schema = _tensor_schema(expected), _tensor_schema(actual)
  if expected_schema != actual_schema:
    missing = sorted(set(expected_schema) - set(actual_schema))
    extra = sorted(set(actual_schema) - set(expected_schema))
    changed = sorted(key for key in set(expected_schema) & set(actual_schema)
                     if expected_schema[key] != actual_schema[key])
    raise ValueError(
      f"{name} architecture mismatch: missing={missing}, extra={extra}, changed={changed}"
    )


@dataclass
class FrozenCheckpoint:
  method: str
  path: Path
  sha256: str
  raw: dict[str, Any]
  actor: Actor
  world_model: WorldModel | ContactStructuredWorldModelV2
  critic: Critic


def load_frozen_checkpoint(
  method: str,
  path: Path,
  *,
  expected_sha256: str | None = None,
  device: str | torch.device = "cpu",
  verify_hash: bool = True,
) -> FrozenCheckpoint:
  """Verify, strictly instantiate, and load an authoritative checkpoint."""
  if method not in EXPECTED_SHA256:
    raise ValueError(f"unsupported frozen method {method!r}")
  expected_sha256 = expected_sha256 or EXPECTED_SHA256[method]
  path = path.resolve(strict=True)
  actual_sha256 = sha256_file(path) if verify_hash else expected_sha256
  if actual_sha256 != expected_sha256:
    raise ValueError(f"refusing to deserialize {path}: SHA-256 mismatch")
  # Safe here only because the exact trusted bytes were checked above.
  raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  cfg = raw.get("cfg", {})
  if raw.get("supermimic") is not True or cfg.get("method") != method:
    raise ValueError(
      f"checkpoint identity mismatch: requested={method}, stored={cfg.get('method')!r}"
    )
  if int(cfg.get("seed", -1)) != 14011 or int(raw.get("iteration", -1)) != 12000:
    raise ValueError("checkpoint is not the frozen seed-14011 iteration-12000 artifact")

  actor = Actor(int(raw["obs_dim"]), tuple(cfg["actor_hidden"]), st.NUM_JOINTS)
  if method == "m1":
    world_model: WorldModel | ContactStructuredWorldModelV2 = WorldModel(tuple(cfg["wm_hidden"]))
  else:
    world_model = ContactStructuredWorldModelV2(
      tuple(cfg["wm_hidden"]), use_loading_memory=method != "m2_v2_nomemory"
    )
  critic = Critic(int(raw["critic"]["norm.mean"].shape[0]), tuple(cfg["critic_hidden"]))
  for name, module in (("actor", actor), ("world_model", world_model), ("critic", critic)):
    assert_exact_schema(name, module.state_dict(), raw[name])
    module.load_state_dict(raw[name], strict=True)
    module.to(device).eval()
  return FrozenCheckpoint(method, path, actual_sha256, raw, actor, world_model, critic)


def checkpoint_manifest(checkpoint: FrozenCheckpoint) -> dict[str, Any]:
  raw, cfg = checkpoint.raw, checkpoint.raw["cfg"]
  components = {}
  for name in ("actor", "world_model", "critic"):
    state = raw[name]
    components[name] = {
      "tensor_count": len(state),
      "element_count": sum(value.numel() for value in state.values()),
      "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                  for key, value in state.items()},
    }
  return {
    "method": checkpoint.method, "path": str(checkpoint.path),
    "sha256": checkpoint.sha256, "size_bytes": checkpoint.path.stat().st_size,
    "seed": cfg["seed"], "iteration": raw["iteration"],
    "environment_transitions": raw.get("env_steps"), "task": cfg["task"],
    "checkpoint_format": "trusted torch.save full training checkpoint",
    "components": components,
    "optimizer_state_entries": {
      name: len(raw[name].get("state", {})) for name in ("opt_actor", "opt_wm", "opt_critic")
    },
    "normalization_state_present": {
      name: sorted(key for key in raw[name] if "norm." in key)
      for name in ("actor", "world_model", "critic")
    },
    "replay_buffer_schema": raw.get("replay_buffer", {}).get("schema"),
    "window_archive_schema": raw.get("window_archive", {}).get("schema"),
  }


def write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
