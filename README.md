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

Expected output at the moment:

```
model: nDof=18	nq=19	mass=6.921	hip_links=0.095/0.213/0.235
model: max |H_ii| after one FK pass = 0.000000
kinematics: max round-trip error (hip frame) 2.333e-01
```

Two things that are not yet right and are the next steps:

- **The URDF-built model has no mass matrix** after `setState`/`forwardKinematics`
  (all `H_ii` zero). The WBC tick therefore stops in QuadProg++ with a singular
  matrix. ula populates the model the same way, so the missing piece is how ula's
  runtime first advances the model (find it in `MdlLegController`/`MdlSimDriver`
  and mirror it here).
- **Forward kinematics returns hip-relative foot positions while inverse
  kinematics expects a body-frame target.** The round-trip figure above is
  therefore measured in the hip frame (and is not zero: the IK's joint-limit
  clamping and branch choice account for the residual). Decide the convention in
  the GPU core and make both directions agree.

## Next steps

1. A WBC tick from a fixed stance command, printed as `desJpos`/`desJvel`, and a
   parity harness comparing those numbers with ula's controller on the same
   inputs.
2. Replace the double-typed additions where the GPU core needs them
   (`QuadrupedKinematics` is already in hip-relative/double form) and replace the
   QuadProg++ QP (double only) with a bounded-iteration float QP.
3. The batched GPU kernel: fixed-size, allocation-free per-robot state, one
   thread per robot, called from JAX through the XLA FFI next to MJX.
