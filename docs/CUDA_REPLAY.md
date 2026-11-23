# Go2 CUDA controller and Warp validation

The full CUDA path runs action decoding, gait, planning, rigid-body dynamics,
task projections, the float force QP, and motor commands on the device. It
keeps Eigen for supported matrix and vector operations. The only replacements
for unavailable Eigen decompositions on the GPU are:

- cuBLAS `SgetrfBatched` with partial pivoting, then `SgetriBatched`, for
  the 18-by-18 mass-matrix inverse.
- cuSOLVER `SgesvdjBatched` for the kinematic and inertia-weighted
  pseudo-inverses. Kinematic inputs are factored as transposes so the library
  receives tall matrices. The pseudo-inverse undoes that transpose. The
  singular values are retained only when strictly above the ULA cutoffs
  (`0.001` for kinematics, `0.0001` for the weighted inverse). cuSOLVER's
  Jacobi implementation need not use Eigen's rotation or pivoted-QR order.

Both calls work on the caller's CUDA stream. Solver buffers are allocated at
batch creation. Actual contact matrices have 3, 6, 9, or 12 rows; zero contacts
means no contact solve. No active matrix is padded. The stock gait always has
two planned contacts, so it needs no count readback. Inactive slots in its
fixed batch (for example the first tick after reset) are ignored and receive
dummy identities, not altered real problems. Other schedules compact worlds
by contact count, read back five counts once per tick, and batch each exact
contact shape. Task stages stay ordered, but different contact groups share
each three-row task solve when they have that stage.

## What the checks mean

The bundled 1,092-tick trace is a policy-free replay of two episodes from
pinned ULA, including reset priming, forward motion and turns. The CUDA
controller gets only recorded inputs and config, never reference outputs.
It checks 65 identical worlds per tick, then reruns the trace and compares the
output files. It expects 1,090 controller solves (one startup skip per episode).

Gait phase, planned contacts and the WBIC-run flag must match exactly.
Continuous outputs are compared to ULA in double precision. For each field,
the agreed review flag is four times ULA float's maximum absolute error
against that same double reference. A flag is not an automatic hard failure.
The original tighter CPU-vs-ULA comparison remains separate. See
[the CPU replay contract](CONTROLLER_REPLAY.md) for capture points and limits.

A light diagnostic samples the mass and SVD matrices already submitted by
the full CUDA replay. Eigen gets the same float inputs. The report prints
the inverse and SVD reconstruction residuals, nearest cutoff and any
sampled singular-value cutoff disagreements. It does not compare raw
singular vectors, which may differ in sign or basis. Nonconvergence and
failed solves still fail the CUDA replay.

Controller-only timing keeps inputs resident on the GPU and measures full
ticks, including memory packing, kernel launches and any contact-group wait.
It compares the NVIDIA solver path and the previous fused draft at 64, 256
and 1,024 worlds. The mixed benchmark spreads the gait phases across an
alternate schedule to exercise all five contact counts. It is not a
closed-loop gait test. The reported count wait includes previously queued
stream work that must finish before the counts are available; it is not
just transfer time. The separate Warp test uses the stock gait, two
1 ms physics substeps per 2 ms controller tick and five ticks per scripted
action. It uses the older cached joint sample for WBIC and the planner, the
fresh sample for the leg commands, and recalculates motor PD from actuator
length and velocity at every substep. None of these tests train a policy.

## Run on Colab or another NVIDIA host

Prefer the official Colab CLI for unattended checks. An optional embedded
notebook can be generated under `bench/artifacts/notebooks/`; choose a GPU
and **Run all**. For CLI or a local NVIDIA host, generate the
`bench/artifacts/bundles/ula-go2-vendor.tar.gz` archive.
After extracting it, install Eigen >= 3.4, CMake, a supported C++17 compiler,
the CUDA toolkit (including cuBLAS/cuSOLVER), and CUDA PyTorch, then run:

```bash
python3 -m pip install mujoco==3.11.0 mujoco-warp==3.11.0 warp-lang==1.19.0
bash bench/run_cuda_replay.sh
python3 bench/verify_warp_loop.py --device cuda --num-envs 64,256,1024 --steps 24
```

For a CPU check, build `ula_batch` without CUDA and use
`python3 bench/verify_warp_loop.py --device cpu`. This deliberately uses
the host implementation and does not prove CUDA behavior.

To make a fresh Colab bundle, use the original recorded trace (not an
output of `cpu_replay`):

```bash
bash bench/make_cuda_replay_tarball.sh /path/to/go2_reference.jsonl
python3 bench/make_colab_cuda_replay_standalone.py --validation
```

Older generated notebooks and their embedded duplicate archives are not kept.
Those experiments tested hybrid paths, where ULA's Eigen task projections
still ran on the CPU. The expanded
hybrid replay passed on A100 before fixed-size dynamics was added. On the
subsequent A100 dynamics run, the isolated 256-state test passed; the hybrid
controller exceeded the old CPU-vs-ULA limits. These runs are separate from
the full controller and Warp tests above.

## Local verification

The local CUDA 12.8 build compiles and links the controller for `sm_75` (T4).
The laptop has no NVIDIA device. All 16 CPU CTest checks passed, including
the reference replays. The CPU Warp smoke advanced 1.8 s, walked 0.169 m,
and turned 0.080 rad without a fall or controller status error. A separate
three-world CPU check reset only one world; the two unchanged worlds and
their next policy step were bit-identical to a run without that reset. An
instrumented CPU step checked the actual call order: five controller calls
at 2 ms intervals and two PD updates per controller call at 1 ms intervals.

## Colab T4 results

One Tesla T4 session ran the bundle with driver 580.82.07, CUDA toolkit
13.0, Eigen 3.4.0, Torch 2.14.1+cu130, MuJoCo Warp 3.11.0 and Warp 1.19.0.
The session was terminated after downloading the results.

All 1,092 replay ticks ran on CUDA; 1,090 ran WBIC. All 65 copies agreed
per tick, and both full replays emitted identical output files. Gait phase,
planned contacts and the WBIC-run flag matched ULA exactly.

| Maximum error vs ULA double | ULA float | NVIDIA path | 4x review flag? |
| --- | ---: | ---: | :--- |
| Foot position target (m) | 0 | 1.490e-8 | Yes: float CPU baseline is exactly zero |
| Joint position target (rad) | 5.964e-6 | 5.728e-6 | No |
| Joint velocity target (rad/s) | 7.481e-5 | 7.630e-5 | No |
| Feedforward torque (N·m) | 1.808e-5 | 3.219e-5 | No |
| Motor torque (N·m) | 2.241e-4 | 2.274e-4 | No |

The foot-target difference comes before LU or SVD in the tick; do not use it
to attribute a numerical error to either solver. The review rule is unchanged.
On 35 sampled mass matrices, the largest inverse residual was `4.80e-6`
on cuBLAS vs `3.87e-6` on Eigen. On 350 sampled SVDs, the worst relative
reconstruction residual was `2.58e-6`, with no cutoff crossings. An additional
20 SVDs from the mixed-gait check also had no cutoff crossings.

| Worlds | Controller only, trot: inline / NVIDIA (ticks/s) | Controller only, mixed: inline / NVIDIA (ticks/s) | NVIDIA + Warp, trot (steps/s at 100 Hz) |
| ---: | ---: | ---: | ---: |
| 64 | 5,279 / 6,923 | 3,652 / 4,891 | 1,233 |
| 256 | 22,409 / 21,318 | 14,415 / 12,459 | 3,886 |
| 1,024 | 82,504 / 44,981 | 53,716 / 32,807 | 7,661 |

The NVIDIA path won at 64 worlds but was slower at 1,024, even in the
stock trot where it did not wait for contact counts. In the 1,024-world
mixed test it took 31.21 ms per batch, with 29.85 ms of wall time spent
waiting for the CUDA stream to deliver the counts. That time includes
waiting for previously queued GPU work. The inline implementation is a
controller-only speed comparison, not a validated Warp backend.

The T4 Warp test walked 0.169 m and turned 0.080 rad in 1.8 simulated
seconds, with no falls or controller errors. Its timings include the
controller, both physics substeps and environment output calculations, with
physics captured in a CUDA graph. The 1,024-world run reached 38,306
controller ticks/s; it does not include policy inference, optimizer updates
or episode resets. PPO was outside this run; see [PPO smoke](PPO_SMOKE.md)
for the later wiring check.
