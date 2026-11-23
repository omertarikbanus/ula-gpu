# ula-gpu: context and decisions

Experimental workspace. Goal (owner, 2026-11-19): train the quadruped policy of
`~/ula` very fast and efficiently on a large NVIDIA GPU. MJX/JAX is not a
requirement; whatever gives the most throughput wins.

## Where this comes from

`~/ula` trains a PPO policy that sits on top of a model-based whole-body
controller (gait scheduling, footstep planning, WBIC). Each EnvPool env runs the
complete C++ controller at 500 Hz inside MuJoCo C, on CPU threads (about 7,400
env steps/s on a 56-core workstation, 2,400 on a 6-core laptop; the paper's
85M-transition run took about 3.2 h). This repository is standalone: it holds a
copy of the Cheetah-derived controller core (no link to ula), the Go2 URDF and
scene, and the GPU work. Evaluation and the robot stay on MuJoCo C and the C++
controller in ula; policies trained here are validated on ula's harness.

## Target architecture

Everything for N robots stays on the GPU, with no per-tick host round trip:
GPU physics (MuJoCo Warp or MJX, chosen by benchmark), the env logic batched on
the GPU (observations, reward, termination, commands, curriculum, disturbances),
the controller core as a batched kernel (one robot per thread or warp), and PPO
on the GPU.

## Decisions

- Float precision throughout the controller is fine (the original MIT
  Cheetah-Software was templated on the scalar type and ran float on the
  robot). Consumer GPUs are far slower in FP64, so the GPU path is float32.
- One controller implementation where possible: the core is refactored into an
  allocation-free, fixed-size form that compiles for CPU and GPU, rather than a
  rewrite in another language.
- The QP (WBIC uses Goldfarb QuadProg++, double only, heap-allocating) gets a
  fixed-storage, bounded-iteration float version of the same dual active-set
  algorithm, compiled for CPU and GPU.
- GPU access: Google Colab, with the code uploaded as the small package made
  by `bench/make_cuda_replay_tarball.sh` (no repository push). Bundle short
  controller and Warp checks into one official Colab CLI session on an
  inexpensive GPU where possible, and stop it afterward. The development
  laptop can build CUDA code but cannot run it.
- Repository hygiene: keep reusable benchmark source in `bench/` and tests
  in `tests/`. Write generated bundles, notebooks, results and local policies
  beneath the ignored `bench/artifacts/` directory. Retain one copy of final
  checkpoints and compact evidence; discard temporary session logs, snapshots
  and retired one-off launchers. Never hide all of `bench/` with `.gitignore`.
- Keep Eigen for operations supported on CUDA. Evaluate NVIDIA's batched
  partial-pivot LU and Jacobi SVD only for the decomposition calls Eigen
  cannot run on-device. Keep the QP, singular-value cutoffs, and actual
  contact/task sizes. Check both numerical behavior and whole-tick speed
  before settling on a backend; see [CUDA replay](CUDA_REPLAY.md).
- Training uses the Go2 with simple collision shapes (`models/go2/scene_gpu.xml`:
  box body, capsule legs, unchanged foot spheres, identical masses and
  inertias). MJX needs 24 GB for one step of the mesh scene and 0.5 GB for this
  one. Trained policies are checked on ula's detailed mesh scene.
- Compare training against the three final ICRA Epsilon policies and their
  matched normalization statistics. First test the frozen policies separately
  from short PPO plumbing checks. For a training comparison, match the
  interaction budget, evaluate in the ICRA setup, and report policy quality
  separately from wall time; see [PPO smoke](PPO_SMOKE.md).

## Findings so far

- The "zero mass matrix" was a demo bug: `FloatingBaseModel::getMassMatrix()`
  returns a cache that only `massMatrix()` fills; the WBC's `_UpdateModel()`
  calls `massMatrix()`. The parser and model are fine (total mass 14.857 kg).
- The URDF parse failed intermittently at `-O0` and passed under gdb:
  nondeterministic, so probably uninitialized memory in the parser or model.
  ula's pass-1 CMake controller also crashed at startup unless built with ASan
  (`MdlLegController::init`); possibly the same bug.
- `QuadrupedKinematics` already returns body-frame foot positions here (the fix
  from ula main); the earlier round-trip residual came from joint-limit
  clamping on the rear legs and the IK branch choice, not from frames.
- Both MJX and MuJoCo Warp accept `models/go2/scene.xml`; both support
  `qfrc_applied` (needed for disturbances); `mujoco_warp` steps on CPU here with
  a larger `njmax`. `mujoco-mjx` 4.x requires `mujoco` 4.x, so pin both at
  3.11.0 (ula's engine version); the `mujoco-core` package exists only for 4.x.
- The first MJX batch benchmark tried to allocate about 710 GB: to be bisected
  (mesh collision geometry in MJX is a likely cause; GPU-oriented Go2 models
  use primitive collision geoms).
- Custom-kernel hooks exist in both engines: JAX `jax.ffi` (XLA FFI custom
  call on the stream), Warp `wp.func_native` / native CUDA kernels.
