#!/usr/bin/env bash
# Validate the fused inline CUDA solver at the same bar as the NVIDIA path.
#
# Runs the full 1,092-tick CUDA replay twice with `--full-cuda-inline`
# (65 replicated worlds, same pipeline as --full-cuda), requires the two
# outputs to be byte-identical, then applies the frozen GPU review rule:
# tools/compare_gpu.py DOUBLE_CANDIDATE INLINE_CANDIDATE --cpu-float
# CPU_CANDIDATE, i.e. the 4x CPU-float flag plus exact gait/contact/WBIC
# parity. Prints PASS/REVIEW lines like the existing frozen checks.
#
# Usage (on an NVIDIA host with the CUDA tree already built):
#   BUILD_DIR=build-cuda bash bench/compare_inline.sh [REFERENCE.jsonl]
# BUILD_DIR defaults to build-cuda (override for build-vendor-cuda etc.).
# The double and CPU-float baselines are reused from the CTest candidates
# in the build tree and regenerated from the CPU replay when absent.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
ref=$(realpath -e -- "${1:-$root/data/go2_reference.jsonl}")
build=${BUILD_DIR:-$root/build-cuda}
python=${PYTHON:-python3}
urdf="$root/models/go2/go2.urdf"
inline="$build/go2_full_cuda_inline_candidate.jsonl"
repeat="$build/go2_full_cuda_inline_repeat_candidate.jsonl"
double="$build/go2_double_reference_candidate.jsonl"
cpu="$build/go2_cpu_candidate.jsonl"

[[ -x $build/cuda_replay ]] || { echo "cuda_replay missing in $build (set BUILD_DIR)" >&2; exit 2; }
[[ -x $build/cpu_replay ]] || { echo "cpu_replay missing in $build (set BUILD_DIR)" >&2; exit 2; }

# Baselines for the review rule: ULA in double precision and ULA float.
[[ -s $double ]] || "$python" "$root/tools/replay_cpu.py" "$ref" "$double" \
  --executable "$build/cpu_replay" --urdf "$urdf" --mode=--reference
[[ -s $cpu ]] || "$python" "$root/tools/replay_cpu.py" "$ref" "$cpu" \
  --executable "$build/cpu_replay" --urdf "$urdf"

# The inline candidate, twice: the replay must be file-identical to repeat.
"$python" "$root/tools/replay_cpu.py" "$ref" "$inline" \
  --executable "$build/cuda_replay" --urdf "$urdf" --mode=--full-cuda-inline
"$python" "$root/tools/replay_cpu.py" "$ref" "$repeat" \
  --executable "$build/cuda_replay" --urdf "$urdf" --mode=--full-cuda-inline
if cmp -s "$inline" "$repeat"; then
  echo "PASS: full-cuda-inline replay is byte-identical across runs"
else
  echo "FAIL: full-cuda-inline replay outputs differ between runs" >&2
  exit 1
fi

# Same review rule as the frozen vendor checks (PASS/REVIEW lines).
exec "$python" "$root/tools/compare_gpu.py" "$double" "$inline" --cpu-float "$cpu"
