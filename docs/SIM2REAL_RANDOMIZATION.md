# Sim2Real randomization

`configs/robustness/baseline.json` describes the clean ablation baseline.
Tasks ending in `-Sim2Real` enable the implemented shared stressors: existing
angular-velocity/projected-gravity/joint observation noise; 0--1 control-step
(0--20 ms at 50 Hz) independent observation latency; 0--2 physics-step motor
bus latency with correlated lag; physically consistent 0.9--1.1 base and link
mass/inertia scaling; base COM variation; joint dry friction; per-axis foot
friction; and ±0.05 rad reset joint offsets. These distributions are part of
the environment, so M1/M2/M3 receive exactly the same perturbations.

The sampled masses, COM, friction, latency and material identity are not Actor
inputs. The normal actor sensor noise remains enabled only for training
(`play=False`). Each stressor should still be independently ablated before a
formal Sim2Real claim.

Evaluation records zero-command standing, action saturation, sign-flip rate,
high-frequency action PSD, joint-limit violations and command-to-actuation
latency. A world model is not assumed to remove actuator jitter. No real robot
commands are part of this repository.
