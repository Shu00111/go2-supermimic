# Contact-memory gradient diagnostic summary

## Bottom line

The reproducible CPU-float64 analysis pipeline is implemented and tested, but
the requested trained-model diagnosis is **not yet scientifically measurable
on this Mac** because none of the formal checkpoint bytes is present. No
synthetic or untrained result below is promoted to trained-policy evidence.

## Decision questions

1. **Does full M2-v2 show greater finite-horizon Jacobian amplification than
   no-memory?** `NOT MEASURED` on frozen trained models.
2. **Is the problematic path memory recurrence, contact-to-body coupling, or
   another component?** `NOT MEASURED`. The code now isolates the historical
   memory coordinate without changing its forward value or detaching the other
   contact channels.
3. **Are auxiliary contact losses in conflict with state prediction?** `NOT
   MEASURED` on the trained checkpoint. Shared-parameter cosine plumbing is
   validated only on an untrained fixture.
4. **Does memory gradient detachment improve stability?** `NOT MEASURED`.
   Forward-equivalent backward cuts at K = 1, 2, 4, 8 are implemented and unit
   tested.
5. **Do predicted Actor updates improve actual physical returns?** `BLOCKED —
   REQUIRES VALIDATED LINUX BACKEND` and the frozen bytes/snapshots.
6. **Which hypotheses remain unsupported?** H1, H2 and H3 remain untested by
   trained evidence; H4 and H5 remain viable. Existing one-seed outcome and
   rollout-error reports motivate them but do not distinguish their causes.
7. **Most justified M2-v3 mechanism?** No mechanism passes the decision gate
   yet. The next investigation should be the controlled backward-only memory
   cut on the frozen model. Recommend a gradient-path redesign only if it
   improves direction/fidelity without changing forward predictions. If
   gradients are stable but fail physics, prioritize model error/distribution
   shift; if objectives conflict, investigate separation or balancing; if
   physical directions are correct, inspect Actor optimization, reward and
   training distribution.

## Unified evidence table

| Metric | M1 | M2-v2 | No-Memory | Interpretation |
|---|---|---|---|---|
| Trained finite-horizon Jacobian norm | NOT MEASURED | NOT MEASURED | NOT MEASURED | Checkpoint bytes absent |
| Trained Actor-gradient norm/cosine | NOT MEASURED | NOT MEASURED | NOT MEASURED | Common frozen Actor unavailable |
| Trained loss-gradient conflict | N/A | NOT MEASURED | NOT MEASURED | Requires trained world-model bytes and matched data |
| Backward-only memory intervention | N/A | NOT MEASURED | N/A | Implementation semantics tested only |
| Physics update fidelity | NOT MEASURED | NOT MEASURED | NOT MEASURED | Validated snapshot backend absent |
| Formal checkpoint SHA-256 | recorded, unverified | recorded, unverified | recorded, unverified | See `checkpoint_provenance.csv` |

## Provenance and platform

- Source baseline: `f7e61b6954e7a706e9aeb1f5d15ccbec0896ca17`.
- Branch: `research/contact-memory-gradient-diagnostics`.
- Tested host: macOS 15.6, Apple M2 arm64, 16 GB RAM, about 53 GiB free.
- Diagnostic environment: Python 3.12.2, PyTorch 2.15.0, CPU float64; MPS is
  built and available but was not used for numerical sensitivity checks.
- Physical packages: MuJoCo 3.11.0, MuJoCo-Warp 3.11.0, Warp 1.19.0, mjlab
  installed (package does not expose `__version__`).
- Formal checkpoint hashes: M1
  `b3a76b694c7ddad1e5897ceeaa01040b3b2587a11143f36fd7a324813f658179`,
  M2-v2 `3d322d8f7ffe68f50f5b6f5573dd48af8dc8dd15220b4446802691469f117c19`,
  no-memory `023b0eed06eb5d8bfb31f3e76c504ea6cd0ea3ae861e3839e4cfc3c49326cbee`.
  The manifests are present; the corresponding bytes are not, so none is
  locally verified.

The older frozen manifest references M1/M2 checkpoints of roughly 3.15 GB
each. The formal-v2 report supplies hashes and remote Linux paths for the five
matched methods. Searching both local repository working directories found no
formal `model_12000.pt` or P1 `best.pt`; only an unrelated macOS smoke
checkpoint and public policy weights exist.

## Completed work

- isolated frozen worktree and branch;
- source/model/recurrence/anchor audit;
- strict checkpoint provenance table;
- matrix-free JVP/VJP Jacobian and finite-horizon tooling;
- common-Actor gradient comparisons and per-step contributions;
- forward-equivalent memory-gradient interventions;
- shared-parameter multi-objective gradient cosines;
- deterministic seeds, float64 precision, hashes, non-finite/failure/censoring
  fields, runtime and peak-memory receipt;
- 15 targeted tests passing; synthetic run completed without non-finite values.

## Reproduction

```bash
cd /path/to/go2-supermimic-gradient-diagnostics
PYTHONPATH=. /Users/shuyiwei/Projects/go2-supermimic/.venv/bin/python -m pytest -q \
  tests/test_contact_memory_diagnostics.py tests/test_contact_structured_model.py \
  tests/test_diagnostics.py

PYTHONPATH=. /Users/shuyiwei/Projects/go2-supermimic/.venv/bin/python \
  scripts/run_contact_memory_diagnostics.py --synthetic-fixture --seed 20261119 \
  --checkpoint-root /Users/shuyiwei/Projects/go2-supermimic \
  --checkpoint-root /Users/shuyiwei/Documents/ChatGPT/Supermimic/go2-supermimic \
  --output outputs/contact_memory_gradient_diagnostics
```

The script fails closed if invoked without `--synthetic-fixture`; it will not
silently create surrogate trained checkpoints.
