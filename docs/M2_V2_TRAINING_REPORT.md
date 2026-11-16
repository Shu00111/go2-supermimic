# M2-v2 formal training report

Status: **PARTIAL relative to the original three-training-seed contract; complete
for the user-approved single-seed five-method comparison.**

## Frozen contract and scope

Training used source commit `23acf7c1ae990b14b5dd5030e6e852e05d129edb`,
M2-v2 model freeze `e8b6d9555cf0df3621c0a7994ab8fb9adcbaf005`, 1,024
parallel environments, 4,000 iterations per stage, and 12,000 iterations
(12.288 million environment transitions) per completed method. Actor,
observation, reward, termination, BPTT horizon, curriculum, and checkpoint
selection were held fixed. The final Stage-3 `model_12000.pt` was selected
without Stage-4 selection.

The complete matched seed-14011 matrix contains M1, fresh M2-v1, M2-v2,
M2-v2-no-memory, and capacity-matched M3. M1 and M2-v1 also completed seed
24017. The user ended further P2 expansion on 2026-11-16; seed24017/M2-v2 was
interrupted after its first checkpoints and seed34019 was not run. Those
partial artifacts are preserved and excluded from formal comparisons.

## Completed seed-14011 runs

| Method | Transitions | Wall span | Collapse-guard events | Final LR scale | Outcome |
|---|---:|---:|---:|---:|---|
| M1 | 12.288M | 1.545 h | 1 | 0.5 | completed |
| M2-v1 | 12.288M | 1.741 h | 1 | 0.5 | completed |
| M2-v2 | 12.288M | 2.090 h | 20 | 9.54e-7 | completed, severe instability |
| M2-v2-no-memory | 12.288M | 2.121 h | 2 | 0.25 | completed |
| M3 capacity control | 12.288M | 1.573 h | 2 | 0.25 | completed |

Wall span is filesystem-observed elapsed time from the first run directory to
the final checkpoint; it includes task transitions and checkpoint I/O and is
not a pure kernel benchmark. M2-v2 repeatedly tripped the preregistered fall
guard from Stage 1 onward. Its final learning-rate scale is approximately one
millionth of the initial value. This is a measured failure, not a successful
recovery: the final actor survived many Stage-4 episodes but was effectively
static.

Seed24017 provides limited stability replication only. M1 completed with one
guard event; M2-v1 completed with seven guard events. No cross-method formal
mean is computed from seed24017 because the remaining methods are absent.

## Frozen final checkpoint identities

| Method | SHA-256 |
|---|---|
| M1 | `b3a76b694c7ddad1e5897ceeaa01040b3b2587a11143f36fd7a324813f658179` |
| M2-v1 | `9c6c4436aa59f435ef70f5b38b5176fd36b14ec8f3b3b27502818020de7cc9f3` |
| M2-v2 | `3d322d8f7ffe68f50f5b6f5573dd48af8dc8dd15220b4446802691469f117c19` |
| M2-v2-no-memory | `023b0eed06eb5d8bfb31f3e76c504ea6cd0ea3ae861e3839e4cfc3c49326cbee` |
| M3 | `ce0fd2f4158b4fc381fe98a21fdcfcd781db1d583b5c6323812746ab236d5455` |

The paths and hashes are also frozen in
`outputs/contact_structured_supermimic/formal_v2/evaluation/stage4_seed14011_v2/protocol.json`.

## Preservation and reproducibility

Intermediate completed-segment checkpoints were compressed with zstd only
after archive integrity and decompressed SHA-256 verification. Segment-final
checkpoints, `model_1500.pt`, every actual rollback checkpoint, failed states,
and the interrupted current segment were retained. The audit ledger is
`outputs/contact_structured_supermimic/formal_v2/runtime/checkpoint_archive_manifest.tsv`.

The original three-seed acceptance item is marked **PARTIAL by explicit user
scope decision**, not silently treated as complete. All Stage-4 conclusions
therefore concern one independent training seed.
