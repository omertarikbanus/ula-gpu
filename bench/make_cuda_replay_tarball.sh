#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 REFERENCE.jsonl [OUTPUT.tar.gz]" >&2
  exit 2
fi
ref=$(realpath -e -- "$1")
out=$(realpath -m -- "${2:-$root/bench/artifacts/bundles/ula-go2-vendor.tar.gz}")
temp=${TMPDIR:-/tmp}
if [[ -d /tmp/opencode && -z ${TMPDIR:-} ]]; then temp=/tmp/opencode; fi
stage=$(mktemp -d "$temp/ula-cuda-bundle.XXXXXX")
trap 'rm -r -- "$stage"' EXIT
mkdir -p "$stage/ula-gpu/data" "$(dirname -- "$out")"
cd "$root"
# Ensure the included trace is actually consumable by the CUDA replay before
# packaging; no traces or model assets are downloaded by the Colab notebook.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/tools" python3 - "$ref" <<'PY'
import sys
from pathlib import Path
from controller_replay import load_and_validate_trace
from replay_cpu import wire_config, wire_tick

trace = load_and_validate_trace(Path(sys.argv[1]))
wire_config(trace.manifest, Path('models/go2/go2.urdf'))
for rec in trace.records:
    if rec['type'] == 'reset':
        if rec.get('priming_ticks') != 6 or rec.get('priming_action_count') != 0:
            raise ValueError('expected the six cleared-action priming ticks')
    else:
        wire_tick(rec, 6)
print(f'Packaging {trace.tick_count} ticks from {trace.episode_count} episodes')
PY

cp --parents -r CMakeLists.txt src third-party \
  models/go2/{go2.urdf,scene_gpu.xml,robot_gpu.xml} \
  tests/*.cc tests/*.py \
  tools/{alloc_audit.cc,compare_gpu.py,controller_replay.py,cpu_replay.cc,parity_compare.cc,parity_dump.cc,qp_check.cc,qp_dump.cc,qp_file.hh,replay_cpu.py,sweep.hh} \
  tools/reference/compare_cpu.sh \
  bench/*.py bench/*.sh bench/requirements*.txt bench/colab_cuda_replay.ipynb \
  ula_gpu/*.py \
  docs/{CONTROLLER_REPLAY.md,CUDA_REPLAY.md} "$stage/ula-gpu/"
cp -- "$ref" "$stage/ula-gpu/data/go2_reference.jsonl"
tar -czf "$out" -C "$stage" ula-gpu
ls -lh -- "$out"
