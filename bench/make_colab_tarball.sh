#!/usr/bin/env bash
# Pack exactly what bench/colab_bench.ipynb needs into /tmp/opencode/ula-gpu-bench.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=/tmp/opencode/ula-gpu-bench.tar.gz
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" --transform 's,^,ula-gpu/,' \
  bench/_common.py bench/mjx_bench.py bench/warp_bench.py bench/colab_bench.ipynb \
  models/go2/scene_gpu.xml models/go2/robot_gpu.xml docs/GPU_BENCH.md
ls -l "$OUT"; tar tzf "$OUT"
