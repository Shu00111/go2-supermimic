# Stage-1 matched-pilot failure record

The first 1000-iteration M1/M2 matched pilot (seed 14011, 1024 environments)
completed but is rejected as a budget-freezing result.

Both methods were finite on the first 500 rigid-flat iterations. After the
curriculum process resumed on flat RFT soil, the world-model loss jumped from
approximately 2.2–2.7 to millions. M1 ended with world-model loss `3.992e6`,
Actor gradient norm `1.312e6`, and Critic loss `5.196e9`. M2 ended with
world-model loss `4.049e6`; its other losses partially recovered but remained
unfit for freezing a formal contract.

Root causes:

1. The replay ring joined the final rigid slot to the first soft slot without a
   terminal marker, creating fictitious cross-task transitions.
2. Rigid-only state channels had output variance near `9.7e-11`. When soil
   channels became active, normalized errors were amplified by millions.
3. Policy updates continued immediately while the world model adapted to the
   changed simulator task.

Corrections add an explicit replay task boundary, a `1e-3` physical per-step
output-scale floor, and 300 world-model-only iterations after each task switch.
The failed checkpoints and full logs remain in the ignored local pilot output;
they are not reused as formal initialization.
