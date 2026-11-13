# Corrected matched Stage-1 pilot

Source `e7a8dcd`, seed 14011, and 1024 environments were used for both methods.
Each method received 500 rigid-flat and 500 flat-RFT iterations. At the task
switch, the policy paused through iteration 800 while the world model adapted.

| Method | Final WM loss | Imagined return | Actor grad norm | Critic loss | Finite |
|---|---:|---:|---:|---:|:---:|
| M1 | 1.097 | -1.214 | 7.861 | 0.1361 | yes |
| M2 | 1.606 | -1.532 | 5.186 | 0.09562 | yes |

Unlike the rejected first pilot, losses remained bounded across the rigid to
soft switch. Both runs completed 1,024,000 recorded environment transitions
and produced resumable `model_1000.pt` checkpoints. Pilot checkpoints are not
used to initialize formal training; the formal methods start independently
from the shared seed contract.
