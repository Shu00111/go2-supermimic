# M2-v2 architecture freeze candidate

Status: **development architecture frozen at source commit
`e8b6d9555cf0df3621c0a7994ab8fb9adcbaf005`; held-out evaluation pending.**

## Scope

M2-v2 keeps the central M2 method: a deployable recurrent per-foot contact
estimator, a contact transition, contact-conditioned body dynamics, and
differentiable autoregressive imagination. It does not alter the actor, critic,
actor observation, reward, termination, controller, BPTT horizon, task
distribution, or real-data protocol. Simulator contact labels are used only to
fit the world model and are never inputs to the deployed actor or estimator.

## Frozen changes from M2-v1

1. **Consistent loading-memory recurrence.** The transition now predicts an
   intrusion proposal and a separate retention factor. During contact it uses
   `proposal + retention * relu(old_memory - proposal)` and contact probability
   differentiably resets the state at release. This matches the estimator's
   intended loading-path contract. M2-v1 reused one scalar as both proposal and
   retention.
2. **Training-only physical anchoring.** The ordinary fully predicted rollout
   remains the primary loss and the only deployment path. A weight-0.25
   auxiliary path starts each transition from the physical current contact
   label and presents the physical next-contact label to the body model. This
   makes the learned contact channels numerically compatible with their SI
   semantics. Oracle contact never enters `forward`, `step`, the actor, or a
   deployed observation.
3. **The M2-v1 force contract is preserved.** Forces retain the bounded
   `150 N * tanh(raw)` decoder and the same fixed normalization. Development
   trials with wider ranges increased force error without improving rollout
   quality; they are recorded as negative results rather than included in the
   frozen method.

The core causal variant, `m2_v2_core`, applies item 1 with anchoring weight
zero. `m2_v2_nomemory` retains the frozen architecture and masks the persistent
loading-memory channel for the required P2 mechanism ablation.

## Causal interpretation

The development comparison isolates the effects:

- M2-v2-core is numerically indistinguishable from M2-v1 on state rollout,
  force, and contact classification. The recurrence correction is therefore a
  semantic/software correction, not the source of the observed development
  gain.
- Adding the training-only anchor improves one-step prediction, loading-memory
  error, and contact classification. Its eight-step state error and tangential
  force error are slightly worse. This is not a claim of uniform superiority;
  it is the expected tradeoff for restoring physical meaning to an otherwise
  co-adapted latent contact code.
- M1 remains the strongest development model on the aggregate state-rollout
  score. M2-v2 is retained because P1 evaluates contact-relevant behavior and
  gradients as well as aggregate MSE, and does not require beating M1 on every
  metric.

## Invariants and tests

- `per_foot_observables` excludes oracle force, material, boundary, and soil
  memory fields.
- The deployment API has no oracle-contact argument.
- The anchor fails closed if training labels are absent.
- The no-memory variant masks memory in both estimator and transition paths.
- Force bounds and the corrected memory recurrence have explicit regression
  tests.
- The full suite passed 59 tests at the freeze commit.

The architecture may be rejected after the one-time held-out prediction and
gradient benchmark, but it must not be modified in response to those results.
Any later modification would require a new, explicitly versioned development
cycle and a new held-out set.
