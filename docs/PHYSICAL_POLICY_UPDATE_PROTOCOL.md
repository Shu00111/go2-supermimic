# Physical Policy-Update Protocol

Status: **NOT RUN**.

The experiment must use one common frozen Actor, paired allocation-bound snapshots, the repository's exact reward/observation/target conventions, a predeclared update-size schedule, and an unchanged-Actor control. For each snapshot, compute one candidate update through a frozen World Model, record predicted imagined-return change, restore the same physical state, and evaluate candidate and control policies with identical target sequences and RNG state. Report physical reward, progress, velocity tracking, termination, and per-condition failures.

Do not tune update sizes on Stage-4 OOD seeds. Reject a sample unless two restore/replays of its fixed action sequence are bitwise exact, including RFT loading memory, contact mode, target generator, observations, reward, and terminal bookkeeping. Run on one isolated environment first.

Execution is blocked because no complete paired snapshot dataset with Actor history and target state was recovered, and the GPU was occupied by an unrelated formal experiment. The existing CUDA replay failure also forbids using mixed-material CUDA states as finite-difference references. No physical return or policy-improvement result is claimed.
