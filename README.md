# ula-gpu

Experimental workspace for training ula's quadruped policy fast on a large
NVIDIA GPU. It holds a standalone copy of ula's controller core, a benchmark of
GPU physics engines, and the reference data a GPU version of the controller will
be checked against. Context, goal and decisions: [`docs/CONTEXT.md`](docs/CONTEXT.md).

## What is here

```
src/                 the Cheetah-derived controller core copied from ula
                     (templated on the scalar type), without the robot runtime
src/core/WbcHarness.hh   builds the controller inputs the way ula's runtime does
src/core_demo.cc     one controller tick from a standing pose, printed
tools/parity_dump.cc deterministic sweep of controller ticks written to a
                     binary file: the reference for a GPU build
tools/alloc_audit.cc counts heap allocations per controller tick
src/qp/gi_qp.hh      fixed-size float QP solver (same algorithm as QuadProg++),
                     ready to compile for the GPU; docs/QP_SOLVER.md
tools/qp_dump.cc, qp_check.cc, qp_cuda_check.cu, parity_compare.cc
                     record the controller's QPs, check the new solver on CPU
                     and GPU, compare two parity files
bench/               GPU physics benchmarks (MJX and MuJoCo Warp), Colab notebook
docker/Dockerfile.gpu    image for the GPU machine
models/go2/          Go2 URDF and scenes (scene_gpu.xml: simple collision shapes)
third-party/         vendored Goldfarb QuadProg++ and tinyxml2
docs/                CONTEXT.md, CORE_STATUS.md (controller), GPU_BENCH.md (engines),
                     QP_SOLVER.md (the new QP solver)
```

## Controller core

Needs CMake, a C++17 compiler and Eigen 3 (headers). Run from the repo root
(the URDF path is relative):

```bash
cmake -S . -B build && cmake --build build -j6
./build/core_demo models/go2/go2.urdf
./build/parity_dump models/go2/go2.urdf /tmp/parity.bin   # fnv1a64 58a2b3ac27ab48cd
./build/alloc_audit models/go2/go2.urdf
./build/qp_dump models/go2/go2.urdf /tmp/wbic.qp && ./build/qp_check /tmp/wbic.qp
```

`-DULA_QP_SOLVER=gi` builds the controller with the new float QP solver instead
of QuadProg++ (default).

## GPU physics benchmark

See [`docs/GPU_BENCH.md`](docs/GPU_BENCH.md): the Colab notebook
`bench/colab_bench.ipynb` (upload the package made by
`bench/make_colab_tarball.sh`) or the Docker image on the GPU machine.

## Next steps

1. Run the engine benchmark on a GPU and pick the engine.
2. Make the rest of the controller core allocation-free and fixed-size (see
   `docs/CORE_STATUS.md`). The QP solver part is done (`src/qp/gi_qp.hh`); it
   still needs a GPU run on Colab and a broader sweep of harder QPs.
3. Compile the core as a batched GPU kernel (one robot per thread) and check it
   against `tools/parity_dump.cc`.
