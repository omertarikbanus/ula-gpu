#!/usr/bin/env bash
# Pack everything bench/colab_bench.ipynb needs into /tmp/opencode/ula-gpu-bench.tar.gz:
# the physics benchmarks with the simple-shape Go2 scene, and the GPU check of the
# float QP solver with the QPs recorded from the controller (made by qp_dump; build it
# first: cmake -S . -B build && cmake --build build -t qp_dump).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/opencode/ula-gpu-bench.tar.gz
QP_DUMP=${QP_DUMP:-build/qp_dump}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
[ -x "$QP_DUMP" ] || { echo "$QP_DUMP missing: build it first (see the header of this script)"; exit 1; }
mkdir -p "$(dirname "$OUT")" "$STAGE/ula-gpu/data"
"$QP_DUMP" models/go2/go2.urdf "$STAGE/ula-gpu/data/wbic.qp" | tail -1
cp --parents bench/_common.py bench/mjx_bench.py bench/warp_bench.py bench/colab_bench.ipynb \
  models/go2/scene_gpu.xml models/go2/robot_gpu.xml docs/GPU_BENCH.md \
  src/qp/gi_qp.hh tools/qp_cuda_check.cu tools/qp_file.hh docs/QP_SOLVER.md "$STAGE/ula-gpu/"
tar czf "$OUT" -C "$STAGE" ula-gpu
ls -l "$OUT"
