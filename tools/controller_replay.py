#!/usr/bin/env python3
"""JSONL trace validator and comparator for the ULA v1 controller replay contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA_NAME = "ula-controller-replay"
SCHEMA_VERSION = 1

DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-5

# Required full-controller input fields and expected vector lengths.
INPUT_VECTOR_LENGTHS: dict[str, int] = {
    "action": 25,
    "base_command": 3,
    "position_w": 3,
    "orientation_wxyz": 4,
    "linear_velocity_b": 3,
    "angular_velocity_b": 3,
    "joint_position": 12,
    "joint_velocity": 12,
}
INPUT_REQUIRED_FIELDS = tuple(INPUT_VECTOR_LENGTHS.keys()) + ("measured_contact",)
INPUT_OPTIONAL_VECTOR_LENGTHS = {"actuator_position": 12, "actuator_velocity": 12}

# Required full-controller output fields.
OUTPUT_CONTINUOUS_VECTOR_LENGTHS: dict[str, int] = {
    "foot_position_target": 12,
    "joint_position_target": 12,
    "joint_velocity_target": 12,
    "feedforward_torque": 12,
    "motor_torque": 12,
}
OUTPUT_CONTINUOUS_FIELDS = ("gait_phase",) + tuple(OUTPUT_CONTINUOUS_VECTOR_LENGTHS.keys())
OUTPUT_REQUIRED_FIELDS = ("gait_phase", "planned_contact") + tuple(
    OUTPUT_CONTINUOUS_VECTOR_LENGTHS.keys()
)


class TraceError(ValueError):
    """Raised when a trace violates the v1 replay contract."""


@dataclass(frozen=True)
class FieldTolerance:
    abs_tol: float = DEFAULT_ABS_TOL
    rel_tol: float = DEFAULT_REL_TOL


@dataclass
class FieldErrorStats:
    max_abs_err: float = 0.0
    max_abs_loc: str = "-"
    max_rel_err: float = 0.0
    max_rel_loc: str = "-"


@dataclass
class Divergence:
    episode: int | None
    tick: int | None
    category: str
    field: str
    message: str


@dataclass
class LoadedTrace:
    path: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    partial: bool
    episode_count: int
    tick_count: int


@dataclass
class CompareResult:
    passed: bool
    partial: bool
    episodes: int
    ticks: int
    first_divergence: Divergence | None = None
    field_stats: dict[str, FieldErrorStats] = field(default_factory=dict)
    divergence_count: int = 0


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _reject_nonfinite_constant(token: str) -> Any:
    raise TraceError(f"Non-finite JSON constant {token!r} is not allowed")


def _check_finite_recursive(val: Any, context: str) -> None:
    if _is_number(val):
        if not math.isfinite(float(val)):
            raise TraceError(f"{context}: non-finite number {val!r}")
    elif isinstance(val, list):
        for idx, item in enumerate(val):
            _check_finite_recursive(item, f"{context}[{idx}]")
    elif isinstance(val, dict):
        for key, item in val.items():
            _check_finite_recursive(item, f"{context}.{key}")


def _validate_num_vector(val: Any, length: int, name: str) -> None:
    if not isinstance(val, list) or len(val) != length:
        actual = len(val) if isinstance(val, list) else type(val).__name__
        raise TraceError(f"{name}: expected list of length {length}, got {actual}")
    for idx, item in enumerate(val):
        if not _is_number(item) or not math.isfinite(float(item)):
            raise TraceError(f"{name}[{idx}]: expected finite number, got {item!r}")


def _validate_bool_vector(val: Any, length: int, name: str) -> None:
    if not isinstance(val, list) or len(val) != length:
        actual = len(val) if isinstance(val, list) else type(val).__name__
        raise TraceError(f"{name}: expected list of {length} booleans, got {actual}")
    for idx, item in enumerate(val):
        if not isinstance(item, bool):
            raise TraceError(f"{name}[{idx}]: expected bool, got {item!r}")


def _validate_measured_contact(val: Any, name: str) -> None:
    if not isinstance(val, list) or len(val) != 4:
        actual = len(val) if isinstance(val, list) else type(val).__name__
        raise TraceError(f"{name}: expected list of length 4, got {actual}")
    all_bool = all(isinstance(x, bool) for x in val)
    all_num = all(_is_number(x) and math.isfinite(float(x)) for x in val)
    if not (all_bool or all_num):
        raise TraceError(f"{name}: expected 4 booleans or 4 finite numbers, got {val!r}")


def _validate_manifest(rec: dict[str, Any], allow_partial: bool) -> bool:
    if rec.get("type") != "manifest":
        raise TraceError(f"Line 1: first record must have type='manifest', got {rec.get('type')!r}")
    if rec.get("schema") != SCHEMA_NAME:
        raise TraceError(
            f"Line 1: manifest schema must be {SCHEMA_NAME!r}, got {rec.get('schema')!r}"
        )
    version = rec.get("version")
    if not _is_int(version) or version != SCHEMA_VERSION:
        raise TraceError(f"Line 1: manifest version must be {SCHEMA_VERSION}, got {version!r}")

    for key in ("reference_commit", "scene_sha256", "urdf_sha256", "config_sha256"):
        val = rec.get(key)
        if not isinstance(val, str) or not val.strip():
            raise TraceError(f"Line 1: manifest.{key} must be a non-empty string")

    dt_s = rec.get("dt_s")
    if not _is_number(dt_s) or not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise TraceError(f"Line 1: manifest.dt_s must be a finite positive number, got {dt_s!r}")

    partial = rec.get("partial", False)
    if not isinstance(partial, bool):
        raise TraceError(f"Line 1: manifest.partial must be a boolean if present, got {partial!r}")
    if partial and not allow_partial:
        raise TraceError(
            "Partial trace rejected for full parity (manifest.partial=true); "
            "pass --allow-partial for diagnostics only"
        )
    return partial


def _validate_input_block(inp: Any, partial: bool, ctx: str) -> None:
    if not isinstance(inp, dict):
        raise TraceError(f"{ctx}: 'input' must be a JSON object")
    _check_finite_recursive(inp, f"{ctx}.input")

    if not partial:
        missing = [k for k in INPUT_REQUIRED_FIELDS if k not in inp]
        if missing:
            raise TraceError(f"{ctx}: missing required input fields: {', '.join(missing)}")

    for fields in (INPUT_VECTOR_LENGTHS, INPUT_OPTIONAL_VECTOR_LENGTHS):
        for name, length in fields.items():
            if name in inp:
                _validate_num_vector(inp[name], length, f"{ctx}.input.{name}")
    if "measured_contact" in inp:
        _validate_measured_contact(inp["measured_contact"], f"{ctx}.input.measured_contact")


def _validate_output_block(out: Any, partial: bool, ctx: str) -> None:
    if not isinstance(out, dict):
        raise TraceError(f"{ctx}: 'output' must be a JSON object")
    _check_finite_recursive(out, f"{ctx}.output")

    if not partial:
        missing = [k for k in OUTPUT_REQUIRED_FIELDS if k not in out]
        if missing:
            raise TraceError(f"{ctx}: missing required output fields: {', '.join(missing)}")

    if "gait_phase" in out:
        gp = out["gait_phase"]
        if not _is_number(gp) or not math.isfinite(float(gp)):
            raise TraceError(f"{ctx}.output.gait_phase: expected finite scalar number, got {gp!r}")
    if "planned_contact" in out:
        _validate_bool_vector(out["planned_contact"], 4, f"{ctx}.output.planned_contact")
    if "wbic_ran" in out and type(out["wbic_ran"]) is not bool:
        raise TraceError(f"{ctx}.output.wbic_ran: expected boolean")
    for name, length in OUTPUT_CONTINUOUS_VECTOR_LENGTHS.items():
        if name in out:
            _validate_num_vector(out[name], length, f"{ctx}.output.{name}")


def load_and_validate_trace(path: Path, allow_partial: bool = False) -> LoadedTrace:
    """Load and strictly validate a JSONL controller trace against the v1 contract."""
    if not path.is_file():
        raise FileNotFoundError(f"Trace file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise TraceError(f"{path}: trace file is empty")

    parsed_records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            raise TraceError(f"{path}:{line_no}: empty line in JSONL trace")
        try:
            obj = json.loads(raw, parse_constant=_reject_nonfinite_constant)
        except json.JSONDecodeError as exc:
            raise TraceError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise TraceError(f"{path}:{line_no}: record must be a JSON object")
        _check_finite_recursive(obj, f"{path}:{line_no}")
        parsed_records.append(obj)

    manifest = parsed_records[0]
    partial = _validate_manifest(manifest, allow_partial=allow_partial)

    if len(parsed_records) == 1:
        raise TraceError(f"{path}: manifest-only trace has no reset or tick records")

    current_episode: int | None = None
    expected_tick = 0
    ticks_in_episode = 0
    total_ticks = 0
    episode_count = 0
    body_records: list[dict[str, Any]] = []

    for idx, rec in enumerate(parsed_records[1:], start=2):
        rec_type = rec.get("type")
        if rec_type == "manifest":
            raise TraceError(f"{path}:{idx}: unexpected duplicate manifest record")
        if rec_type == "reset":
            ep = rec.get("episode")
            if not _is_int(ep) or ep < 0:
                raise TraceError(f"{path}:{idx}: reset.episode must be a non-negative int, got {ep!r}")
            expected_ep = 0 if current_episode is None else current_episode + 1
            if ep != expected_ep:
                raise TraceError(
                    f"{path}:{idx}: expected reset episode {expected_ep}, got {ep}"
                )
            if current_episode is not None and ticks_in_episode == 0:
                raise TraceError(
                    f"{path}:{idx}: episode {current_episode} has no ticks before next reset"
                )
            current_episode = ep
            expected_tick = 0
            ticks_in_episode = 0
            episode_count += 1
            body_records.append(rec)
        elif rec_type == "tick":
            if current_episode is None:
                raise TraceError(f"{path}:{idx}: tick encountered before initial reset record")
            ep = rec.get("episode")
            tick = rec.get("tick")
            if not _is_int(ep) or ep != current_episode:
                raise TraceError(
                    f"{path}:{idx}: expected tick.episode={current_episode}, got {ep!r}"
                )
            if not _is_int(tick) or tick != expected_tick:
                raise TraceError(
                    f"{path}:{idx}: expected tick={expected_tick} in episode {current_episode}, "
                    f"got {tick!r}"
                )
            ctx = f"{path}:{idx} (ep={ep}, tick={tick})"
            _validate_input_block(rec.get("input"), partial=partial, ctx=ctx)
            _validate_output_block(rec.get("output"), partial=partial, ctx=ctx)
            expected_tick += 1
            ticks_in_episode += 1
            total_ticks += 1
            body_records.append(rec)
        else:
            raise TraceError(f"{path}:{idx}: unknown record type {rec_type!r}")

    if total_ticks == 0 or ticks_in_episode == 0:
        raise TraceError(f"{path}: trace has episode reset without any tick records")

    return LoadedTrace(
        path=path,
        manifest=manifest,
        records=body_records,
        partial=partial,
        episode_count=episode_count,
        tick_count=total_ticks,
    )


def _exact_value_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_exact_value_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact_value_equal(a[k], b[k]) for k in a)
    return a == b


def _check_manifest_compatibility(ref: LoadedTrace, cand: LoadedTrace) -> Divergence | None:
    for key in ("schema", "version", "reference_commit", "scene_sha256", "urdf_sha256", "config_sha256"):
        if ref.manifest.get(key) != cand.manifest.get(key):
            return Divergence(
                episode=None,
                tick=None,
                category="manifest",
                field=key,
                message=(
                    f"manifest.{key} mismatch: ref={ref.manifest.get(key)!r} "
                    f"cand={cand.manifest.get(key)!r}"
                ),
            )
    ref_dt = float(ref.manifest["dt_s"])
    cand_dt = float(cand.manifest["dt_s"])
    if ref_dt != cand_dt:
        return Divergence(
            episode=None,
            tick=None,
            category="manifest",
            field="dt_s",
            message=f"manifest.dt_s mismatch: ref={ref_dt} cand={cand_dt}",
        )
    return None


def _compare_inputs(
    ref_in: dict[str, Any],
    cand_in: dict[str, Any],
    episode: int,
    tick: int,
) -> Divergence | None:
    ref_keys = set(ref_in.keys())
    cand_keys = set(cand_in.keys())
    if ref_keys != cand_keys:
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)
        return Divergence(
            episode=episode,
            tick=tick,
            category="input",
            field="*",
            message=f"input field set mismatch (missing={missing}, extra={extra})",
        )
    for key in sorted(ref_keys):
        if not _exact_value_equal(ref_in[key], cand_in[key]):
            return Divergence(
                episode=episode,
                tick=tick,
                category="input",
                field=key,
                message=f"input.{key} mismatch: ref={ref_in[key]!r} cand={cand_in[key]!r}",
            )
    return None


def _update_continuous_field(
    field_name: str,
    ref_val: Any,
    cand_val: Any,
    tol: FieldTolerance,
    episode: int,
    tick: int,
    stats: FieldErrorStats,
) -> Divergence | None:
    if isinstance(ref_val, list):
        pairs = [(idx, float(r), float(c)) for idx, (r, c) in enumerate(zip(ref_val, cand_val))]
    else:
        pairs = [(None, float(ref_val), float(cand_val))]

    first_div: Divergence | None = None
    for idx, r, c in pairs:
        abs_err = abs(c - r)
        rel_err = abs_err / abs(r) if r != 0.0 else (0.0 if abs_err == 0.0 else math.inf)
        loc = f"ep={episode},tick={tick}" + (f"[{idx}]" if idx is not None else "")
        if abs_err > stats.max_abs_err:
            stats.max_abs_err = abs_err
            stats.max_abs_loc = loc
        if rel_err > stats.max_rel_err:
            stats.max_rel_err = rel_err
            stats.max_rel_loc = loc

        allowed = tol.abs_tol + tol.rel_tol * abs(r)
        if abs_err > allowed and first_div is None:
            field_label = f"{field_name}[{idx}]" if idx is not None else field_name
            first_div = Divergence(
                episode=episode,
                tick=tick,
                category="output",
                field=field_label,
                message=(
                    f"output.{field_label} exceeded tolerance: "
                    f"ref={r:.9g} cand={c:.9g} abs_err={abs_err:.3e} "
                    f"rel_err={rel_err:.3e} allowed={allowed:.3e} "
                    f"(abs_tol={tol.abs_tol:.3e}, rel_tol={tol.rel_tol:.3e})"
                ),
            )
    return first_div


def compare_traces(
    ref: LoadedTrace,
    cand: LoadedTrace,
    tolerances: dict[str, FieldTolerance] | None = None,
    default_tol: FieldTolerance | None = None,
    output_fields: list[str] | None = None,
) -> CompareResult:
    """Compare a candidate trace against a reference trace."""
    is_partial = ref.partial or cand.partial or bool(output_fields)
    tol_map = tolerances or {}
    fallback_tol = default_tol or FieldTolerance()

    manifest_div = _check_manifest_compatibility(ref, cand)
    if manifest_div is not None:
        return CompareResult(
            passed=False,
            partial=is_partial,
            episodes=ref.episode_count,
            ticks=ref.tick_count,
            first_divergence=manifest_div,
            divergence_count=1,
        )

    min_len = min(len(ref.records), len(cand.records))
    stats: dict[str, FieldErrorStats] = {}
    first_div: Divergence | None = None
    divergence_count = 0
    compared_output = False

    def record_div(div: Divergence) -> None:
        nonlocal first_div, divergence_count
        divergence_count += 1
        if first_div is None:
            first_div = div

    for i in range(min_len):
        r_rec = ref.records[i]
        c_rec = cand.records[i]
        r_type = r_rec["type"]
        c_type = c_rec["type"]
        if (
            r_type != c_type
            or r_rec.get("episode") != c_rec.get("episode")
            or r_rec.get("tick") != c_rec.get("tick")
        ):
            record_div(
                Divergence(
                    episode=r_rec.get("episode"),
                    tick=r_rec.get("tick"),
                    category="sequence",
                    field="record",
                    message=(
                        f"record {i} mismatch: "
                        f"ref=({r_type}, ep={r_rec.get('episode')}, tick={r_rec.get('tick')}) vs "
                        f"cand=({c_type}, ep={c_rec.get('episode')}, tick={c_rec.get('tick')})"
                    ),
                )
            )
            break

        if r_type != "tick":
            if not _exact_value_equal(r_rec, c_rec):
                record_div(Divergence(r_rec["episode"], None, "sequence", "reset",
                                      "reset metadata differs"))
                break
            continue

        ep = int(r_rec["episode"])
        tk = int(r_rec["tick"])

        in_div = _compare_inputs(r_rec["input"], c_rec["input"], ep, tk)
        if in_div is not None:
            record_div(in_div)
            break

        r_out = r_rec["output"]
        c_out = c_rec["output"]

        fields = output_fields or OUTPUT_REQUIRED_FIELDS
        if not output_fields and ("wbic_ran" in r_out or "wbic_ran" in c_out):
            fields = (*fields, "wbic_ran")
        for req_field in fields:
            in_ref = req_field in r_out
            in_cand = req_field in c_out
            if in_ref != in_cand or (output_fields and not in_ref):
                record_div(
                    Divergence(
                        episode=ep,
                        tick=tk,
                        category="output",
                        field=req_field,
                        message=(
                            f"output.{req_field} presence mismatch: "
                            f"ref={in_ref} cand={in_cand}"
                        ),
                    )
                )
                continue
            if not in_ref:
                continue

            compared_output = True
            if req_field in ("planned_contact", "wbic_ran"):
                if r_out[req_field] != c_out[req_field]:
                    record_div(
                        Divergence(
                            episode=ep,
                            tick=tk,
                            category="output",
                            field=req_field,
                            message=(
                                f"output.{req_field} exact mismatch: "
                                f"ref={r_out[req_field]} cand={c_out[req_field]}"
                            ),
                        )
                    )
            else:
                f_stats = stats.setdefault(req_field, FieldErrorStats())
                tol = tol_map.get(req_field, fallback_tol)
                out_div = _update_continuous_field(
                    req_field, r_out[req_field], c_out[req_field], tol, ep, tk, f_stats
                )
                if out_div is not None:
                    record_div(out_div)

    if first_div is None and len(ref.records) != len(cand.records):
        next_rec = ref.records[min_len] if len(ref.records) > min_len else cand.records[min_len]
        record_div(
            Divergence(
                episode=next_rec.get("episode"),
                tick=next_rec.get("tick"),
                category="sequence",
                field="length",
                message=(
                    f"trace record count mismatch: ref={len(ref.records)} cand={len(cand.records)}"
                ),
            )
        )
    if first_div is None and not compared_output:
        record_div(Divergence(None, None, "output", "*", "no contract output field was compared"))

    return CompareResult(
        passed=(first_div is None),
        partial=is_partial,
        episodes=ref.episode_count,
        ticks=ref.tick_count,
        first_divergence=first_div,
        field_stats=stats,
        divergence_count=divergence_count,
    )


def _parse_field_tol(spec: str) -> tuple[str, FieldTolerance]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --field-tol {spec!r}; expected FIELD=ABS,REL"
        )
    field_name, vals = spec.split("=", 1)
    if field_name not in OUTPUT_CONTINUOUS_FIELDS:
        raise argparse.ArgumentTypeError(
            f"Unknown continuous output field {field_name!r}; "
            f"valid fields: {', '.join(OUTPUT_CONTINUOUS_FIELDS)}"
        )
    parts = vals.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid tolerance values {vals!r} for {field_name}; expected ABS,REL"
        )
    try:
        abs_tol = float(parts[0])
        rel_tol = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Non-numeric tolerance in {spec!r}") from exc
    if abs_tol < 0.0 or rel_tol < 0.0 or not (math.isfinite(abs_tol) and math.isfinite(rel_tol)):
        raise argparse.ArgumentTypeError(f"Tolerances must be finite and >= 0 in {spec!r}")
    return field_name, FieldTolerance(abs_tol=abs_tol, rel_tol=rel_tol)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and compare ULA v1 policy-free controller replay JSONL traces."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a single JSONL trace.")
    p_val.add_argument("trace", type=Path, help="Path to JSONL trace file.")
    p_val.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow manifest.partial=true traces for diagnostics only (not full parity).",
    )

    p_cmp = sub.add_parser("compare", help="Compare candidate trace against reference trace.")
    p_cmp.add_argument("ref", type=Path, help="Reference JSONL trace file.")
    p_cmp.add_argument("cand", type=Path, help="Candidate JSONL trace file.")
    p_cmp.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow manifest.partial=true traces for diagnostics only (not full parity).",
    )
    p_cmp.add_argument(
        "--abs-tol",
        type=float,
        default=DEFAULT_ABS_TOL,
        help=f"Default absolute tolerance for continuous outputs (default: {DEFAULT_ABS_TOL}).",
    )
    p_cmp.add_argument(
        "--rel-tol",
        type=float,
        default=DEFAULT_REL_TOL,
        help=f"Default relative tolerance for continuous outputs (default: {DEFAULT_REL_TOL}).",
    )
    p_cmp.add_argument(
        "--field-tol",
        action="append",
        default=[],
        metavar="FIELD=ABS,REL",
        help="Per-field continuous output tolerance override (repeatable).",
    )
    p_cmp.add_argument(
        "--output-field", action="append", choices=OUTPUT_REQUIRED_FIELDS,
        help="Stage diagnostic: compare only this output (repeatable; requires --allow-partial).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            trace = load_and_validate_trace(args.trace, allow_partial=args.allow_partial)
            prefix = "[PARTIAL DIAGNOSTIC - NOT FULL PARITY] " if trace.partial else ""
            print(
                f"{prefix}VALID: {trace.path} "
                f"(episodes={trace.episode_count}, ticks={trace.tick_count})"
            )
            return 0

        if args.command == "compare":
            if args.output_field and not args.allow_partial:
                parser.error("--output-field requires --allow-partial")
            if args.abs_tol < 0.0 or args.rel_tol < 0.0 or not (
                math.isfinite(args.abs_tol) and math.isfinite(args.rel_tol)
            ):
                parser.error("--abs-tol and --rel-tol must be finite and >= 0")

            tol_map: dict[str, FieldTolerance] = {}
            for spec in args.field_tol:
                try:
                    fname, ftol = _parse_field_tol(spec)
                except argparse.ArgumentTypeError as exc:
                    parser.error(str(exc))
                tol_map[fname] = ftol

            ref = load_and_validate_trace(args.ref, allow_partial=args.allow_partial)
            cand = load_and_validate_trace(args.cand, allow_partial=args.allow_partial)
            res = compare_traces(
                ref,
                cand,
                tolerances=tol_map,
                default_tol=FieldTolerance(abs_tol=args.abs_tol, rel_tol=args.rel_tol),
                output_fields=args.output_field,
            )

            prefix = "[PARTIAL DIAGNOSTIC - NOT FULL PARITY] " if res.partial else ""
            status = "PASS" if res.passed else "FAIL"
            print(
                f"{prefix}{status}: compared {args.ref} vs {args.cand} "
                f"(episodes={res.episodes}, ticks={res.ticks})"
            )
            for fname, st in res.field_stats.items():
                print(
                    f"  {fname}: max_abs_err={st.max_abs_err:.3e} ({st.max_abs_loc}), "
                    f"max_rel_err={st.max_rel_err:.3e} ({st.max_rel_loc})"
                )
            if res.first_divergence is not None:
                d = res.first_divergence
                loc = f"ep={d.episode}, tick={d.tick}" if d.episode is not None else "manifest"
                print(
                    f"  FIRST DIVERGENCE ({loc}, {d.category}:{d.field}): {d.message}",
                    file=sys.stderr,
                )
                return 1
            return 0

    except (OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except TraceError as exc:
        print(f"INVALID TRACE: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
