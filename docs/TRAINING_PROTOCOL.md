# Frozen training protocol

Methods are M1 vanilla, M2 contact-structured, and M3 capacity-matched generic.
Formal seeds are 14011, 24017, and 34019. Actor, critic, controller, observation,
reward, termination, task mixture, transitions, initialization rule, checkpoint
selection and evaluation manifest are identical. Only method and world-model
capacity may differ. Historical B1 weights must not seed one method alone.

Stage 1 samples flat rigid and flat soft equally. Stage 2 retains Stage 1 and
adds rough rigid plus randomized soil strength. Stage 3 retains prior coverage
and adds rough soft, both boundary directions, alternating patches, and
simultaneous mixed-foot materials. Stage 4 is frozen evaluation, never training
or hyperparameter selection. The current configs explicitly mark the unverified
rough-soft/alternating-patch gap.

The formal transition budget is `PILOT_REQUIRED`: run matched short pilots on
the RTX 5090 at 128, 256, 512, and 1024 environments, select one budget before
formal runs, then freeze it for every method/seed. Do not inherit 12,000
iterations without that evidence. Track wall time, environment transitions,
optimizer steps and GPU-hours.

Stage 4 uses the frozen manifest, at least 128 independent first episodes per
condition, and reports success, survival, progress, velocity MAE, falls, slip,
sinkage, mechanical power, work/distance, prediction quality, gradient
reliability and training efficiency. Report all-episode metrics; any
successful-only metric gets an explicit suffix and denominator.
