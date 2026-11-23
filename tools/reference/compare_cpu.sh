#!/usr/bin/env bash
# Frozen Go2 WBIC replay limits, calibrated on the 412-tick CPU trace.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
if (( $# != 2 )); then
  echo "Usage: $0 REFERENCE.jsonl CANDIDATE.jsonl" >&2
  exit 2
fi
exec python3 "$root/tools/controller_replay.py" compare "$1" "$2" \
  --abs-tol 0 --rel-tol 0 \
  --field-tol foot_position_target=2e-7,0 \
  --field-tol joint_position_target=1e-6,0 \
  --field-tol joint_velocity_target=1e-5,0 \
  --field-tol feedforward_torque=3e-5,0 \
  --field-tol motor_torque=3e-5,0
