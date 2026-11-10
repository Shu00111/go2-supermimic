# Soft-terrain implementation

## Physics coupling

`go2_supermimic.soft_terrain.RftOperator` implements the validated spherical
3D RFT surface integration as batched PyTorch tensor operations on the same CUDA
device as MuJoCo Warp. `RftAction` is a zero-dimensional mjlab action term, so
it receives no policy output but is called before every 2 ms physics substep.
It applies the integrated force and the torque shifted from the foot centre to
the calf-body centre of mass through MuJoCo's external-wrench buffer.

Every environment and foot owns independent maximum-depth and plastic-position
memory. Reset and release clear only the relevant entries. Soil strength scales
the validated constitutive coefficients; it does not replace them with a
spring-damper model.

## Environments

- `Go2-Chase-SoftTerrain`: uniform reference soil.
- `Go2-Chase-SoftTerrainLevels`: per-environment coefficient scale in [0.3, 1.0].
- `Go2-Chase-SoftTransition`: hard-to-soft boundary at local x=0.
- `Go2-Chase-SoftToHard`: soft-to-hard boundary at local x=0.

All retain the original 12-dimensional joint-position action, PD control,
50 Hz Actor rate, observations, Actor architecture, rewards, and SuperMimic
optimizer. Only physics runs at 500 Hz. Soft foot geoms do not rigidly collide;
transition foot geoms collide only with their dedicated hard half-plane.

## World-model state boundary

The Actor observation is unchanged and contains neither soil coefficients,
RFT forces, nor the soil-memory state. The privileged world-model state is 78
dimensions: the original 70 robot dimensions, four per-foot plastic depths,
and four signed distances to the hard/soft boundary. This is the minimal
Markov state for loading-history dynamics and transition prediction. Uniform
soft tasks use a constant positive distance; original hard tasks use zeros.

TensorBoard additionally records sinkage, RFT force magnitude, absolute
mechanical power, plastic depth, soft-foot fraction, and boundary-band fraction.
