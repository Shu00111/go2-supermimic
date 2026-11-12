# Sim2Real randomization

`configs/robustness/baseline.json` disables optional stressors. The separate
`sim2real.json` declares proprioceptive/IMU/joint noise, observation and action
latency, communication jitter, mass/COM, joint/foot friction, initial joint
position, and actuation scale. Every factor must be independently ablated; all
are not enabled together by default, and M1/M2/M3 share distributions.

Evaluation records zero-command standing, action saturation, sign-flip rate,
high-frequency action PSD, joint-limit violations and command-to-actuation
latency. A world model is not assumed to remove actuator jitter. No real robot
commands are part of this repository.
