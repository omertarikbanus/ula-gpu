# GPU benchmark: MuJoCo Warp vs MJX

Goal: pick the physics engine for training the Go2 policy with the C++ controller
ticking every physics step. Harness: `bench/mjx_bench.py`, `bench/warp_bench.py`
(N robots, fixed random ctrl, K steps; physics-only and "ctrl-ph" = physics plus a
placeholder controller tick every step). Timing basis of the real system: physics
dt 2 ms, controller every physics step (500 Hz), policy every 5 steps (100 Hz).
`steps/s` is aggregate (N x physics steps / wall s); realtime factor is aggregate
simulated seconds per wall second.

## Scene: why `scene_gpu.xml`

MJX with the original mesh scene (`models/go2/scene.xml`) fails on a single robot:
`RESOURCE_EXHAUSTED: Out of memory allocating 24,606,162,792 bytes` (N=1, one
step; the batched draft asked for ~710 GB). Bisect: marking the 13 mesh geoms
(about 96k vertices, 192k faces) non-colliding makes the same step run in 0.74 GB
RSS; the primitive scene needs 0.55 GB. So the cause is mesh collision (mesh-plane
and mesh-mesh buffers sized from the vertex/face counts), not njmax/nconmax or
vmap. `models/go2/scene_gpu.xml` + `robot_gpu.xml` replace collision meshes with
primitives; original files are untouched.

Physical differences from the original (contacts change dynamics, so a policy
trained on this scene must be validated on the mesh scene in ula):

| Item | original | GPU scene |
|---|---|---|
| Base collision | base.stl mesh | box half-sizes 0.1881 x 0.04675 x 0.057 (Menagerie-like) |
| Thigh / calf | meshes | capsules along the link (0 -> -0.213 z), radius 0.025 / 0.012 |
| Abad (hip) bodies | mesh | no collision (visual mesh removed too) |
| Foot | sphere r=0.022 | unchanged |
| Collision pairs | everything vs everything (minus excludes) | only robot geoms vs floor (robot geoms contype 1, conaffinity 0; floor 1/1): no self-collision |
| Visuals | meshes | none |
| Mass, inertia, joints, actuators, friction, floor | | unchanged (sum of body masses 15.017 kg in both) |

Harness choices: `--timestep` defaults to 0.002 (the scene file says 0.001), initial
pose = standing (abad 0, hip -0.8, knee 1.6, base z 0.32), float32.

## Colab first

1. On the laptop: `bench/make_colab_tarball.sh` -> `/tmp/opencode/ula-gpu-bench.tar.gz`
   (small, ~12 KB, no meshes needed), or push the repo and use `git clone` (a private
   repo needs a token in the URL).
2. Colab: Runtime > GPU, open `bench/colab_bench.ipynb`, set `REPO_URL` or upload the
   tarball (option B), Run all. Needs Python >= 3.12 (jax 0.12.x pin); if the Colab
   image is older, lower the jax pin (and keep mujoco/mjx/mujoco_warp equal).
3. Download `results_colab.json`, put it in `bench/`.

## Owner's GPU machine

```bash
cd ~/ula-gpu                      # repo root
docker build -f docker/Dockerfile.gpu -t ula-gpu-bench .
docker run --rm --gpus all -v "$PWD/bench:/work/ula-gpu/bench" ula-gpu-bench bash -c '
  nvidia-smi -L &&
  python /work/ula-gpu/bench/mjx_bench.py  --device gpu --num-envs 1024,4096,16384,65536 --steps 200 --results /work/ula-gpu/bench/results_gpu.json &&
  python /work/ula-gpu/bench/warp_bench.py --device gpu --num-envs 1024,4096,16384,65536 --steps 200 --results /work/ula-gpu/bench/results_gpu.json 2>&1 | grep -v "^Module"'
```
Requires the NVIDIA driver plus nvidia-container-toolkit on the host (driver must
support CUDA 12.6). Memory: watch `nvidia-smi --query-gpu=memory.used --format=csv -l 1`
in a second terminal during the largest N (MJX/XLA preallocates 75% of the GPU by default;
set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to see real use). Without Docker: use the same
pip pins as the Dockerfile in a Python 3.12 venv.

### Record

GPU model, driver, CUDA, N, steps/s for `physics` and `ctrl-ph`, compile_s, GPU memory
(peak), whether `finite` is true. Results JSON already contains engine, N, steps/s,
realtime factor, GPU name, host, versions.

| GPU | driver | engine | variant | N | steps/s | RT factor | peak mem | notes |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

## Calling a custom CUDA C++ controller kernel per robot per physics step

The placeholder (`ctrl-ph`) marks the spot: it reads state, writes `ctrl` (or
`qfrc_applied`) before each physics step. A real controller is one thread (or warp)
per robot, fixed-size float32 state in global memory, no heap.

**MJX (jax.ffi).** Build the kernel as a shared library exposing an XLA FFI handler
(`XLA_FFI_DEFINE_HANDLER_SYMBOL`, platform "CUDA", taking `cudaStream_t` from
`xla::ffi::PlatformStream<cudaStream_t>`); in Python `jax.ffi.register_ffi_target(name,
jax.ffi.pycapsule(lib.Handler), platform="CUDA")` and call
`jax.ffi.ffi_call(name, out_shape, vmap_method="broadcast_all")(qpos, qvel, state, ...)`
inside the scan body before `mjx.step`. The kernel runs on XLA's stream, in the same
compiled program, with no host sync. Per-robot controller state must be threaded
through the scan carry (functional; the FFI call returns new state, `input_output_aliases`
avoids copies). Risks: XLA cannot fuse across the opaque call (extra memory round trips
of the state arrays every step); `vmap` semantics of the call must be declared right;
a 500 Hz tick means 5 calls per policy step, each launch has fixed overhead (graphs help
as XLA already uses CUDA graphs only in some configurations); toolchain coupling to the
jaxlib's XLA FFI header version.

**MuJoCo Warp.** Either `wp.func_native` (inline CUDA snippet / `.cu` source) inside a
`@wp.kernel` that operates on `d.qpos`, `d.qvel`, `d.ctrl` (zero-copy, same arrays
MuJoCo Warp reads), or launch an externally built kernel on `wp.get_stream()` through
ctypes. Launch it before `mjw.step(m, d)` inside the same `wp.ScopedCapture()`; the
whole 500 Hz loop (controller + step, K repetitions) is one CUDA graph replay, so launch
overhead is amortized. Risks: everything in a captured graph must be allocation-free
and have static launch dims (no host decisions per step); the native code must be built
with the same CUDA arch and compile flags as Warp's NVRTC; large per-thread stacks
(QP solver) may need `cudaDeviceSetLimit(cudaLimitStackSize)`; register pressure of a
fat controller kernel limits occupancy; Warp's API is younger and changes between releases.

## Engine comparison (what is known so far)

* CPU smoke numbers (laptop, N<=64, 50 steps, not representative of GPU) are in
  the results JSON files; they only prove the code paths run.
* MJX: pure JAX, needs primitive collisions; fixed-size contact buffers; mature.
  Per-step kernel launches are many unless XLA fuses well.
* MuJoCo Warp: the engine MuJoCo upstream is investing in for GPU; native kernels,
  CUDA-graph capture, mesh collision handled (not tested on GPU here); needs
  sensible `nconmax`/`njmax` per world (`--nconmax 16 --njmax 96` defaults).
* Decision belongs after the GPU numbers above; the expectation (unmeasured) is that
  Warp with graph capture beats MJX for a controller-in-the-loop design.
