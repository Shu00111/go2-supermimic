# Development validation report

Host audit: Apple Silicon arm64, 8 logical CPUs, 16 GiB RAM, 53 GiB available
on the root volume, no CUDA. No Python training, MuJoCo, Warp, or experiment
process was active. Existing repositories and untracked user files were left
untouched; development occurred in an isolated clone.

Validated on CPU:

- source compilation for modified modules;
- M1 and M2 finite forward/backward;
- actor-action gradient through world model;
- gradient through predicted contact state and shared per-foot transition;
- estimator shape, foot identity, reset and oracle-channel exclusion;
- three-step loss alignment and four-step differentiable rollout;
- RFT numerical fixtures, memory isolation, release and strength scaling;
- smooth finite-difference diagnostic agreement;
- shared experimental-contract configuration.

CUDA simulator integration, Blackwell extension build, full checkpoint
save/resume, throughput, rough-soft physics, alternating patches, and paired
simulator gradient capture are not run or blocked as documented. These are not
reported as passing. No formal training or held-out evaluation was executed.
