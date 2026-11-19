# Contact-memory mathematical model

## Scope and source

This audit describes source commit `f7e61b6954e7a706e9aeb1f5d15ccbec0896ca17`.
It distinguishes the learned transition used in imagination from the physical
MuJoCo/RFT simulator. The two must not be conflated.

## State and input dimensions

| Quantity | Width | Persistent across learned steps | Source meaning |
|---|---:|---|---|
| Robot/world-model state `s` | 78 | yes | base height 1, gravity 3, yaw-frame velocity 3, angular velocity 3, joint position 12, joint velocity 12, hips/knees/feet 36, RFT soil plastic depth 4, material-boundary signed distance 4 |
| Planar pose `p` | 3 | yes, outside `s` | world x, y, yaw; updated from predicted local pose delta |
| Contact state `z` | 4 × 8 = 32 | M2 only | force 3, intrusion 1, loading memory 1, tangential velocity 2, contact probability 1 per foot |
| Previous action | 12 | yes | actor observation and world-model input |
| Simulator context `xi` | 4 | episode constant | foot friction 1 and base COM offset 3 |
| Terrain relief scan | 77 | recomputed | 11 × 7 base-relative height grid |
| Actor observation history | 735 in Stage 2/3 | yes | five frames of 147 values when height scan is enabled |
| Estimator recurrent state | 4 × 21 = 84 | only in the explicit recurrent-estimator audit | per-foot GRU hidden 20 plus loading memory 1 |
| Target-generator state | structured dictionary | yes | all fields in `target/generator.py:STATE_DIMS`, including chase/dodge dynamics and obstacle state |

Crucially, M1 does **not** lack all soil history: `s[70:74]` contains the RFT
per-foot plastic depth. M2 adds a learned contact-state memory at `z[...,4]`;
that is distinct from the physical RFT soil state already present in `s`.

## True learned transition

For M1,

\[
u_t=\pi_\theta(h_t),\qquad
(s_{t+1},\Delta p_t)=F^{M1}_\phi(s_t,u_t,u_{t-1},\xi,r_t).
\]

For M2-v2,

\[
\begin{aligned}
u_t &= \pi_\theta(h_t),\\
z_{t+1} &= C_\phi(s_t,z_t,u_t,\xi),\\
(s_{t+1},\Delta p_t) &= B_\phi(s_t,z_{t+1},u_t,u_{t-1},\xi,r_t),\\
p_{t+1} &= P(p_t,\Delta p_t),\\
h_{t+1} &= H(h_t,s_{t+1},p_{t+1},u_t,g_{t+1}).
\end{aligned}
\]

Here `g` is target-generator state and `r` is the terrain/obstacle relief scan.
Therefore the closed-loop Markov state is not merely `s`. It includes at least
`(s,p,z,u_previous,h,g)`; `xi`, encoder bias, terrain and robot constants are
fixed conditioning variables. For M1, omit `z` but retain every other term.

The deployed imagination path initializes `z` once with
`initial_contact(s,u_previous)`, whose GRU state is zero, then propagates `z`
with `contact_transition`. The 84-dimensional GRU recurrent state is propagated
only by the separate recurrent-estimator evaluation path; it is not part of
`Imagination.rollout` after initialization. Any Jacobian study that silently
adds that GRU state to every imagined step analyzes a different system.

## Loading-memory recurrence

For each foot, the transition produces raw intrusion and retention codes:

\[
q_t=0.08\,\sigma(\rho^{intr}_t),\quad
\alpha_t=\sigma(\rho^{ret}_t),\quad
c_t=\sigma(\rho^{contact}_t),
\]

\[
m_{t+1}=c_t\left[q_t+\alpha_t\,\mathrm{ReLU}(m_t-q_t)\right].
\]

The no-memory variant calls `_memory_contract` at estimator and transition
boundaries, replacing the memory coordinate by an exact zero. The remaining
seven contact channels and the structured transition/body networks are kept.

## Computation and differentiation paths

```text
physical s_t ──┬─> per-foot observables ─> contact transition ─> z_{t+1}
action u_t  ───┤                                      │              │
context xi  ───┤                                      │              v
z_t (m_t) ─────┘                                      └────> body network ─> s_{t+1}
                                                                    │
actor history <─ observation map <─ pose/target/scan <──────────────┘
      │
      └─> actor u_{t+1} ─────────────────────────────────────────────>
```

The memory-to-body derivative has a direct same-step branch
`m_{t+1} -> normalize_contact -> body`, and a recurrent branch
`m_t -> contact_transition/_decode_transition -> m_{t+1}`. Memory can also
affect future force, intrusion, tangential velocity and contact probability
because all eight transition outputs share the contact-transition network.

The backward-only intervention passes the numerically identical `z_t` forward
but substitutes `stop_gradient(m_t)` for its memory coordinate. It does not
detach force, intrusion, tangent, probability, `s`, or the newly computed
same-step `m_{t+1}`. Time truncation applies that cut only at boundaries K =
1, 2, 4 or 8. These are diagnostics, not new trained methods.

## Training-only anchor

M2-v2's ordinary predicted-contact rollout supplies the primary body and
contact losses. A weight-0.25 teacher-forced branch separately starts from the
oracle current contact, predicts the oracle next contact, and feeds the oracle
next contact to the body model. Oracle contact is absent from `forward`,
`step`, the actor observation and deployment. Consequently the anchor changes
parameter learning but is not a recurrent state or deployment-time input.

## Scaling

Contact coordinates are normalized by 150 N force, 0.08 m intrusion, 0.08 m
memory, 1 m/s tangential velocity and unit probability. World-model deltas use
the frozen `RunningNorm` statistics with a 1e-3 output standard-deviation
floor. Reported Jacobians must use a documented normalized coordinate system;
raw condition numbers across M1's 90-dimensional diagnostic state and M2's
122-dimensional state are not directly comparable.
