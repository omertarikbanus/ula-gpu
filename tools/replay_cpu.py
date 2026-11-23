#!/usr/bin/env python3
"""Run the CPU or CUDA-hybrid WBIC replay on a pinned Go2 JSONL trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import controller_replay as cr


REFERENCE = "29c3dfccde133ffe3bbf56608bd4b0a96777de2e"
GAINS = ("Kp_body", "Kd_body", "Kp_ori", "Kd_ori", "Kp_foot", "Kd_foot", "Kp_joint", "Kd_joint")
FIELDS = (
    ("action", 25), ("base_command", 3), ("position_w", 3), ("orientation_wxyz", 4),
    ("linear_velocity_b", 3), ("angular_velocity_b", 3), ("rotation_wb", 9),
    ("linear_velocity_w", 3), ("yaw", 1), ("foot_position_leg", 12),
    ("joint_position", 12), ("joint_velocity", 12), ("leg_controller_joint_position", 12),
    ("leg_controller_joint_velocity", 12), ("actuator_position", 12),
    ("actuator_velocity", 12), ("actuator_enabled", 12),
)


def wire_config(manifest, urdf):
    if manifest["reference_commit"] != REFERENCE or manifest.get("replay_profile") != "go2_envpool_script_v1":
        raise ValueError("unsupported recorder profile or reference revision")
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != manifest["urdf_sha256"]:
        raise ValueError("URDF hash differs from the reference")
    if manifest["dt_s"] != 0.002:
        raise ValueError("expected a 500 Hz controller trace")
    values = dict(line.removeprefix("PARITY ").split("=", 1)
                  for line in manifest["effective_parameters"].splitlines() if line.startswith("PARITY "))
    out = [str(manifest["dt_s"])]
    for name, length in [("floating_base_weight", 1), ("body_height", 1), ("contact_mu", 1),
                         ("max_torque", 3), *((g, 3) for g in GAINS)]:
        tokens = values[name].split()
        if len(tokens) != length or not all(math.isfinite(float(v)) for v in tokens):
            raise ValueError(f"invalid effective parameter: {name}")
        out.extend(tokens)
    return " ".join(out)


def wire_tick(rec, priming_ticks):
    inp = rec["input"]
    count = inp.get("action_count")
    if type(count) is not int or not 0 <= count <= 25 or (rec["tick"] < priming_ticks and count != 0):
        raise ValueError("invalid action_count or missing cleared-action priming tick")
    out = ["T", str(count)]
    for name, length in FIELDS:
        value = inp.get(name)
        if name == "yaw":
            value = [value]
        valid = (lambda x: type(x) is bool) if name == "actuator_enabled" else (
            lambda x: type(x) in (int, float) and math.isfinite(x))
        if not isinstance(value, list) or len(value) != length or not all(map(valid, value)):
            raise ValueError(f"episode {rec['episode']} tick {rec['tick']}: invalid input.{name}")
        out.extend(str(int(x)) if type(x) is bool else str(x) for x in value)
    return " ".join(out)


def replay(source, target, executable, urdf, host_stages=False, extra_args=()):
    if target is not None and source.resolve() == target.resolve():
        raise ValueError("candidate output must not replace the reference")
    ref = cr.load_and_validate_trace(source)
    wire = [wire_config(ref.manifest, urdf)]
    priming_ticks = 0
    for rec in ref.records:
        if rec["type"] == "reset":
            priming_ticks = rec.get("priming_ticks")
            if priming_ticks != 6 or rec.get("priming_action_count") != 0:
                raise ValueError("expected six captured cleared-action priming ticks per reset")
            wire.append("R")
        else:
            wire.append(wire_tick(rec, priming_ticks))
    command = [str(executable.resolve()), str(urdf.resolve())]
    if host_stages:
        command.append("--host-stages")
    command.extend(extra_args)
    result = subprocess.run(command, input="\n".join(wire),
                            stdout=subprocess.PIPE, text=True, check=True)
    if target is None:
        return  # benchmark reports to stderr; it does not generate a trace
    outputs = [json.loads(line, parse_constant=cr._reject_nonfinite_constant)
               for line in result.stdout.splitlines()]
    ticks = sum(rec["type"] == "tick" for rec in ref.records)
    if len(outputs) != ticks:
        raise ValueError(f"expected {ticks} candidate outputs, got {len(outputs)}")
    records = [ref.manifest]
    it = iter(outputs)
    for rec in ref.records:
        if rec["type"] == "tick":
            out = next(it)
            cr._validate_output_block(out, False, f"candidate episode {rec['episode']} tick {rec['tick']}")
            rec = {**rec, "output": out}
        records.append(rec)
    target.write_text("".join(json.dumps(rec, allow_nan=False) + "\n" for rec in records), encoding="utf8")
    print(f"Replayed {ticks} ticks with {executable.name}: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--executable", type=Path, default=Path("build/cpu_replay"))
    parser.add_argument("--urdf", type=Path, default=Path("models/go2/go2.urdf"))
    parser.add_argument("--host-stages", action="store_true", help="exercise the staged dispatch on CPU")
    parser.add_argument("--mode", action="append", default=[],
                        help="extra executable flag, e.g. --mode=--reference (double-precision ULA)")
    args = parser.parse_args()
    if (args.candidate is None) != (args.mode == ["--cuda-bench"]):
        parser.error("provide a candidate output except with --mode=--cuda-bench")
    try:
        replay(args.reference, args.candidate, args.executable, args.urdf, args.host_stages, args.mode)
    except (OSError, ValueError, KeyError, cr.TraceError, subprocess.CalledProcessError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
