# Contact-structured method

M1 retains the tracked generic `WorldModel`. M2 uses the same actor, critic,
reward, real-data collection, optimizer schedule and BPTT horizon, changing
only the world-model representation. M3 enlarges the generic first hidden
layer to 1072 and is the capacity control. CPU construction gives M1 2,370,641,
M2 2,428,053, and M3 2,428,625 trainable world-model parameters (M3 is within
0.024% of M2). M2 comprises 2,748 estimator, 21,896 per-foot transition, and
2,403,409 contact-conditioned body parameters.

For each ordered foot `(FR, FL, RR, RL)`, M2 represents
`[force_xyz, intrusion, loading_memory, tangential_velocity_xy, contact_p]`.
Units are N, m, m, m/s, and probability. Fixed normalization scales are
`[150,150,150,.08,.08,1,1,1]`.

The persistent estimator consumes only gravity, yaw-frame base velocity,
angular velocity, that leg's joint position/velocity, base-relative foot
position, and previous action. Oracle soil memory, material identity, force and
boundary distance are excluded and tested. A shared GRU has independent
per-environment/per-foot recurrent rows. Its loading recurrence is
`m=q*(p+rho*relu(m_prev-p))`; soft probability gating keeps touchdown/liftoff
differentiable. Episode completion zeroes hidden state and memory.

The shared per-foot transition predicts the next contact vector. A generic body
delta network conditions on all four independently ordered predictions. During
imagination, predicted contact is threaded from step to step without detach.
The actor never receives oracle or estimated contact directly.

Rigid labels have force and contact semantics but exactly zero intrusion and
loading memory. Soft labels use RFT force, sinkage and plastic depth. Loss is
the original normalized multi-step state/pose loss plus smooth-L1 normalized
physical contact loss and BCE contact classification. No held-out data tunes
these scales.
