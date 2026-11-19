# ula-gpu

Standalone controller core for GPU training with MJX (MuJoCo XLA). The context
and decisions behind this repository are in [`docs/CONTEXT.md`](docs/CONTEXT.md).

## What is here

```
src/include/, src/{dynamics,controllers,estimators,input,parameters,utilities}/
    The Cheetah-derived controller core from ula's controller (templated on the
    scalar type), minus the robot runtime (modules, supervisor, hardware).
src/core_demo.cc
    Checks the core: builds the Go2 floating-base model from the URDF and runs a
    forward/inverse kinematics round trip.
third-party/{Goldfarb_Optimizer,tinyxml2}
    Vendored QP solver and XML parser used by the core.
models/go2/   Go2 URDF and meshes used by the demo.
```

## Build and run

Needs CMake, a C++17 compiler and Eigen 3 (headers only):

```bash
cmake -S . -B build && cmake --build build -j6
./build/core_demo models/go2/go2.urdf
```

The demo is being extended into a WBC tick and a parity dump; see
`docs/CONTEXT.md` for the current findings.

## Next steps

1. A WBC tick from a fixed stance command, printed as `desJpos`/`desJvel`, and a
   parity harness comparing those numbers with ula's controller on the same
   inputs.
2. Make the core allocation-free and fixed-size, and replace the QuadProg++
   QP (double only) with a fixed-storage, bounded-iteration float version of the
   same algorithm.
3. The batched GPU kernel: fixed-size, allocation-free per-robot state, one
   thread per robot, called from JAX through the XLA FFI next to MJX.
