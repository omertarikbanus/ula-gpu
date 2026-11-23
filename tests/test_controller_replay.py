#!/usr/bin/env python3
"""Unit tests for tools/controller_replay.py using temporary synthetic traces."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import controller_replay as cr  # noqa: E402


def _make_manifest(*, partial: bool | None = None, dt_s: float = 0.002) -> dict:
    rec: dict = {
        "type": "manifest",
        "schema": "ula-controller-replay",
        "version": 1,
        "reference_commit": "29c3dfccde133ffe3bbf56608bd4b0a96777de2e",
        "dt_s": dt_s,
        "scene_sha256": "abc123",
        "urdf_sha256": "def456",
        "config_sha256": "ghi789",
    }
    if partial is not None:
        rec["partial"] = partial
    return rec


def _make_input(offset: float = 0.0, *, extra: dict | None = None) -> dict:
    inp: dict = {
        "action": [0.1 + offset] * 25,
        "base_command": [0.5, 0.0, 0.1],
        "position_w": [0.0, 0.0, 0.3],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_b": [0.5, 0.0, 0.0],
        "angular_velocity_b": [0.0, 0.0, 0.1],
        "joint_position": [0.0, 0.8, -1.6] * 4,
        "joint_velocity": [0.0] * 12,
        "measured_contact": [True, True, False, False],
    }
    if extra:
        inp.update(extra)
    return inp


def _make_output(offset: float = 0.0, *, contact: list[bool] | None = None) -> dict:
    return {
        "gait_phase": 0.25 + offset,
        "planned_contact": contact if contact is not None else [True, False, True, False],
        "foot_position_target": [0.2, -0.1, -0.3] * 4,
        "joint_position_target": [0.0, 0.8, -1.6] * 4,
        "joint_velocity_target": [0.1] * 12,
        "feedforward_torque": [1.0 + offset] * 12,
        "motor_torque": [2.0 + offset] * 12,
    }


def _make_tick(episode: int, tick: int, inp: dict | None = None, out: dict | None = None) -> dict:
    return {
        "type": "tick",
        "episode": episode,
        "tick": tick,
        "input": inp if inp is not None else _make_input(),
        "output": out if out is not None else _make_output(),
    }


def _write_jsonl(path: Path, records: list[dict | str]) -> Path:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


class ControllerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_and_compare_pass(self) -> None:
        records = [
            _make_manifest(),
            {"type": "reset", "episode": 0},
            _make_tick(0, 0, _make_input(extra={"imu_seq": 10})),
            _make_tick(0, 1, _make_input(0.05, extra={"imu_seq": 11})),
            {"type": "reset", "episode": 1},
            _make_tick(1, 0, _make_input(extra={"imu_seq": 12})),
        ]
        ref = _write_jsonl(self.tmp_dir / "ref.jsonl", records)
        cand = _write_jsonl(self.tmp_dir / "cand.jsonl", records)

        self.assertEqual(cr.main(["validate", str(ref)]), 0)
        self.assertEqual(cr.main(["compare", str(ref), str(cand)]), 0)

    def test_empty_and_manifest_only_fail_even_with_allow_partial(self) -> None:
        empty = self.tmp_dir / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        manifest_only = _write_jsonl(
            self.tmp_dir / "manifest_only.jsonl", [_make_manifest(partial=True)]
        )
        reset_only = _write_jsonl(
            self.tmp_dir / "reset_only.jsonl",
            [_make_manifest(partial=True), {"type": "reset", "episode": 0}],
        )

        for p in (empty, manifest_only, reset_only):
            self.assertEqual(cr.main(["validate", str(p)]), 1)
            self.assertEqual(cr.main(["validate", str(p), "--allow-partial"]), 1)

    def test_partial_trace_rejected_for_full_parity_and_allowed_for_diagnostics(self) -> None:
        partial_tick = {
            "type": "tick",
            "episode": 0,
            "tick": 0,
            "input": {"action": [0.0] * 25},
            "output": {"motor_torque": [1.0] * 12},
        }
        p_ref = _write_jsonl(
            self.tmp_dir / "p_ref.jsonl",
            [_make_manifest(partial=True), {"type": "reset", "episode": 0}, partial_tick],
        )
        p_cand = _write_jsonl(
            self.tmp_dir / "p_cand.jsonl",
            [_make_manifest(partial=True), {"type": "reset", "episode": 0}, partial_tick],
        )

        self.assertEqual(cr.main(["validate", str(p_ref)]), 1)
        self.assertEqual(cr.main(["compare", str(p_ref), str(p_cand)]), 1)
        self.assertEqual(cr.main(["validate", str(p_ref), "--allow-partial"]), 0)
        self.assertEqual(cr.main(["compare", str(p_ref), str(p_cand), "--allow-partial"]), 0)

    def test_missing_full_fields_rejected_when_not_partial(self) -> None:
        bad_tick = _make_tick(0, 0)
        del bad_tick["output"]["motor_torque"]
        p = _write_jsonl(
            self.tmp_dir / "missing_field.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, bad_tick],
        )
        self.assertEqual(cr.main(["validate", str(p)]), 1)
        self.assertEqual(cr.main(["validate", str(p), "--allow-partial"]), 1)

    def test_stage_comparison_requires_the_selected_output(self) -> None:
        reset = {"type": "reset", "episode": 0}
        ref = _write_jsonl(self.tmp_dir / "ref.jsonl", [_make_manifest(), reset, _make_tick(0, 0)])
        cand = _write_jsonl(self.tmp_dir / "cand.jsonl", [
            _make_manifest(partial=True), reset, _make_tick(0, 0, out={"gait_phase": 0.25}),
        ])
        cmd = ["compare", str(ref), str(cand), "--allow-partial", "--output-field"]
        self.assertEqual(cr.main(cmd + ["gait_phase"]), 0)
        self.assertEqual(cr.main(cmd + ["motor_torque"]), 1)

    def test_empty_partial_outputs_are_not_a_comparison(self) -> None:
        records = [_make_manifest(partial=True), {"type": "reset", "episode": 0},
                   _make_tick(0, 0, out={})]
        trace = _write_jsonl(self.tmp_dir / "partial.jsonl", records)
        self.assertEqual(cr.main(["compare", str(trace), str(trace), "--allow-partial"]), 1)

    def test_nonfinite_rejected(self) -> None:
        raw_nan_tick = json.dumps(_make_tick(0, 0)).replace("0.25", "NaN", 1)
        p_nan = _write_jsonl(
            self.tmp_dir / "nan.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, raw_nan_tick],
        )
        self.assertEqual(cr.main(["validate", str(p_nan)]), 1)

        raw_inf_tick = json.dumps(_make_tick(0, 0, _make_input(extra={"diag": 999.0}))).replace(
            "999.0", "Infinity", 1
        )
        p_inf = _write_jsonl(
            self.tmp_dir / "inf.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, raw_inf_tick],
        )
        self.assertEqual(cr.main(["validate", str(p_inf)]), 1)

    def test_reset_and_tick_order_errors(self) -> None:
        no_reset = _write_jsonl(
            self.tmp_dir / "no_reset.jsonl", [_make_manifest(), _make_tick(0, 0)]
        )
        self.assertEqual(cr.main(["validate", str(no_reset)]), 1)

        skipped_tick = _write_jsonl(
            self.tmp_dir / "skipped_tick.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, _make_tick(0, 0), _make_tick(0, 2)],
        )
        self.assertEqual(cr.main(["validate", str(skipped_tick)]), 1)

        bad_ep = _write_jsonl(
            self.tmp_dir / "bad_ep.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0),
                {"type": "reset", "episode": 2},
                _make_tick(2, 0),
            ],
        )
        self.assertEqual(cr.main(["validate", str(bad_ep)]), 1)

    def test_field_length_and_type_errors(self) -> None:
        bad_len = _make_tick(0, 0)
        bad_len["input"]["action"] = [0.0] * 24
        p_len = _write_jsonl(
            self.tmp_dir / "bad_len.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, bad_len],
        )
        self.assertEqual(cr.main(["validate", str(p_len)]), 1)

        bad_bool = _make_tick(0, 0)
        bad_bool["output"]["motor_torque"][0] = True
        p_bool = _write_jsonl(
            self.tmp_dir / "bad_bool.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, bad_bool],
        )
        self.assertEqual(cr.main(["validate", str(p_bool)]), 1)

        bad_contact = _make_tick(0, 0)
        bad_contact["output"]["planned_contact"] = [1, 0, 1, 0]
        p_contact = _write_jsonl(
            self.tmp_dir / "bad_contact.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, bad_contact],
        )
        self.assertEqual(cr.main(["validate", str(p_contact)]), 1)

    def test_input_mismatch_including_additional_field(self) -> None:
        ref = _write_jsonl(
            self.tmp_dir / "ref.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, _make_input(extra={"timing_us": 100})),
            ],
        )
        cand_diff_extra = _write_jsonl(
            self.tmp_dir / "cand1.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, _make_input(extra={"timing_us": 101})),
            ],
        )
        cand_missing_extra = _write_jsonl(
            self.tmp_dir / "cand2.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, _make_input()),
            ],
        )
        self.assertEqual(cr.main(["compare", str(ref), str(cand_diff_extra)]), 1)
        self.assertEqual(cr.main(["compare", str(ref), str(cand_missing_extra)]), 1)

    def test_planned_contact_exact_and_tolerance_edge(self) -> None:
        ref_out = _make_output(0.0, contact=[True, False, True, False])
        ref = _write_jsonl(
            self.tmp_dir / "ref.jsonl",
            [_make_manifest(), {"type": "reset", "episode": 0}, _make_tick(0, 0, out=ref_out)],
        )

        cand_contact_out = _make_output(0.0, contact=[True, True, True, False])
        cand_contact = _write_jsonl(
            self.tmp_dir / "cand_contact.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, out=cand_contact_out),
            ],
        )
        self.assertEqual(cr.main(["compare", str(ref), str(cand_contact)]), 1)

        # Tolerance edge: motor_torque ref is 2.0; use exact binary float 1/1024
        # so abs_err == abs_tol passes and slightly above fails.
        tol_edge = 1.0 / 1024.0
        edge_pass_out = _make_output(0.0)
        edge_pass_out["motor_torque"][3] = 2.0 + tol_edge
        cand_edge_pass = _write_jsonl(
            self.tmp_dir / "cand_edge_pass.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, out=edge_pass_out),
            ],
        )
        self.assertEqual(
            cr.main(
                [
                    "compare",
                    str(ref),
                    str(cand_edge_pass),
                    "--abs-tol",
                    "0",
                    "--rel-tol",
                    "0",
                    "--field-tol",
                    f"motor_torque={tol_edge},0",
                ]
            ),
            0,
        )

        edge_fail_out = _make_output(0.0)
        edge_fail_out["motor_torque"][3] = 2.0 + tol_edge + 1e-6
        cand_edge_fail = _write_jsonl(
            self.tmp_dir / "cand_edge_fail.jsonl",
            [
                _make_manifest(),
                {"type": "reset", "episode": 0},
                _make_tick(0, 0, out=edge_fail_out),
            ],
        )
        self.assertEqual(
            cr.main(
                [
                    "compare",
                    str(ref),
                    str(cand_edge_fail),
                    "--abs-tol",
                    "0",
                    "--rel-tol",
                    "0",
                    "--field-tol",
                    f"motor_torque={tol_edge},0",
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
