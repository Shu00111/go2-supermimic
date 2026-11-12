# Stage-0 failure audit — 2026-11-12

Failed validation attempts are retained here rather than erased from the
experimental record.

1. The isolated environment's editable install initially resolved to a second
   checkout. It was rebound to this research worktree before any receipt was
   accepted.
2. Rigid contact labels used MuJoCo entity order (`FL, FR, RL, RR`) while RFT
   tensors use canonical order (`FR, FL, RR, RL`). The label reader now reorders
   by the sensor's reported primary names and has a regression test.
3. Mixed-material contact sensing used an unfiltered secondary match, which
   included robot self-contact. It now matches only the dedicated hard support
   body.
4. The mocap-backed hard support entities were not reset to each batched world
   origin. Per-entity reset events now align physical geometry and material
   queries in every world.
5. Enlarging hard checkerboard patches created a boundary overlap: one failed
   attempt measured 203.18 N rigid and 178.59 N RFT on the same foot. Exact
   patch boundaries plus a predictive 1 cm RFT clearance and a three-substep
   activation debounce removed the overlap. An attempted per-world `geom_gap`
   switch was rejected because it did not reliably disable an already generated
   constraint and is not part of the final implementation.

The final three-task RTX 3090 receipt passed only after these corrections.
