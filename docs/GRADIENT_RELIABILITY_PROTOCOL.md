# Gradient reliability protocol

This is an offline diagnostic and never changes training. Freeze checkpoint,
initial full simulator state, RFT memory, terrain, commands, RNG state, terminal
logic and future nominal actions. For each chosen action coordinate, restore
the identical snapshot and evaluate centered perturbations at epsilon 0.01,
0.003 and 0.001. Hold all future actions at their nominal values.

Compare world-model autograd `g_WM` with paired physical finite differences
using cosine similarity, relative L2 error, and active-coordinate sign
agreement. Report each epsilon, method, contact phase and terrain separately.
Strata are stable stance, liftoff, touchdown, aerial, rigid, soft, hard-to-soft,
and soft-to-hard. If either member crosses a different terminal event, flag the
pair and report both censored and uncensored summaries. Do not interpret exact
gradient agreement at discontinuous contact events.

`diagnostics.py` and `evaluate_gradient_fidelity.py` implement numerical
scoring. Simulator capture is deliberately fail-closed until a validated
mjlab+RFT snapshot/restore adapter exists; `evaluate_gradient_capture.py`
records this blocker rather than generating incomparable data.
