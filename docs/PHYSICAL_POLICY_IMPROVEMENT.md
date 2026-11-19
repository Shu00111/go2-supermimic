# Physics-grounded policy improvement

## Status

`BLOCKED — REQUIRES VALIDATED LINUX BACKEND`

The Mac has MuJoCo 3.11.0, MuJoCo-Warp 3.11.0, Warp 1.19.0 and mjlab installed
in the existing Python 3.12 environment. That is insufficient for the formal
test because:

1. the frozen M1/M2-v2/no-memory checkpoint bytes are absent;
2. paired physical initial-state snapshots are absent; and
3. the existing project protocol records no validated full snapshot/restore
   adapter covering MuJoCo state, RFT loading memory, target commands, RNG
   streams and terminal bookkeeping.

`policy_update_fidelity.csv` therefore contains only the blocker receipt. No
learned world model was substituted for independent physics validation.

The eventual execution must start every method from the same Actor checkpoint,
use the same fixed data and controlled step sizes, and evaluate paired,
independent physical scenarios. It must report predicted and physical return
changes, sign agreement, progress, velocity tracking and fall-rate changes.
Update sizes may not be tuned on the frozen evaluation set.
