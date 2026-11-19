# Jacobian diagnostics

## Status

`TRAINED-POLICY RESULT: NOT MEASURED.` The formal checkpoint bytes and matched
recorded trajectories are not present locally. The committed results are one
CPU-float64 `UNTRAINED_MODEL_SMOKE_TEST` using small instances of the real M1,
M2-v2 and no-memory classes. They validate plumbing only.

The implementation estimates the largest singular value of a local
finite-horizon map using alternating JVP/VJP power iteration. It never treats
an eigenvalue as a singular value and does not infer asymptotic stability from
a one-step norm. Horizons are 1, 2, 4, 8, 16 and 32. A dot-product identity
checks `<Jv,w> = <v,J^T w>` for every model. A separate smooth fixture checks
centered differences at epsilons 1e-2 through 1e-5.

Files are in `outputs/contact_memory_gradient_diagnostics/`:

- `jacobian_spectrum.csv`
- `jacobian_horizon_growth.csv`
- `jacobian_analysis.png`
- `run_receipt.json`

The smoke run used one smooth synthetic stance, not the required phase-stratified
physical trajectories. Rigid/soft stance, touchdown, liftoff and material
transitions remain `NOT MEASURED`. Contact events are nonsmooth and will be
censored or reported separately rather than analyzed as ordinary derivatives.

## Required real-data command

After copying the exact frozen checkpoints and a hash-verified trajectory
bundle to local storage, run the checkpoint mode (currently fail-closed until
the trajectory adapter is supplied):

```bash
PYTHONPATH=. .venv/bin/python scripts/run_contact_memory_diagnostics.py \
  --output outputs/contact_memory_gradient_diagnostics \
  --checkpoint-root /absolute/path/to/frozen/formal_v2
```

No comparison between full M2-v2 and no-memory is scientifically supported by
the untrained smoke magnitudes.
