#!/usr/bin/env bash
# Build and run on an NVIDIA host (including a Colab GPU runtime).
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
ref=$(realpath -e -- "${1:-$root/data/go2_reference.jsonl}")
build=${BUILD_DIR:-$root/build-cuda}
python=${PYTHON:-python3}

if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo 'CUDA compiler or NVIDIA driver tools missing. Use a GPU runtime with nvcc.' >&2
  exit 2
fi
if ! nvidia-smi -L; then
  echo 'NVIDIA device unavailable. No CPU fallback will be used.' >&2
  exit 2
fi
arch=${CUDA_ARCHITECTURES:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '. ' | sort -u | paste -sd ';')}
if [[ ! $arch =~ ^[0-9]+(\;[0-9]+)*$ ]]; then
  echo 'Unable to select a CUDA architecture; set CUDA_ARCHITECTURES, e.g. 75 for a T4.' >&2
  exit 2
fi
nvcc --version
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DULA_ENABLE_CUDA_REPLAY=ON -DULA_QP_SOLVER=quadprog \
  -DCMAKE_CUDA_ARCHITECTURES="$arch" -DULA_REPLAY_REFERENCE="$ref" \
  -DPython3_EXECUTABLE="$(command -v "$python")"
cmake --build "$build" --parallel "${BUILD_JOBS:-2}" --target cpu_replay cuda_replay ula_batch

# Keep the original CPU tolerance check. GPU precision has its own comparison
# against the double run and the stated 4x review flag.
status=0
ctest --test-dir "$build" --no-tests=error -V \
  -R '^go2_(cpu_parity|host_stages_parity|full_(double_precision|host_precision|cuda_(precision|repeatability)))$' || status=$?
"$python" "$root/tools/replay_cpu.py" "$ref" --executable "$build/cuda_replay" \
  --urdf "$root/models/go2/go2.urdf" --mode=--cuda-bench || status=$?
exit "$status"
