#!/usr/bin/env python3
"""Evaluate existing checkpoints only, with complete 5.5-second scoring windows.

All 12 direction/magnitude cells share a 384-world physics batch, with the
same per-cell onset RNG streams as the serial evaluator. No training occurs.
The published seed-0 Epsilon checkpoint is evaluated by the same code.
"""

import json
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import run_overnight as job


def main():
    job.OUT.mkdir(exist_ok=True)
    job.STATE.update(
        run_id=os.environ["ULA_RUN_ID"], started=time.time(), current=None,
        phases={}, elapsed_s={}, purpose="correct incomplete scoring windows; no training",
        seed=0, cap_steps=1200, episodes_per_cell=32, cells=12, batch_worlds=384)
    job.atomic_json(job.OUT / "outcomes.json", {"status": "running", "run_id": job.STATE["run_id"]})
    inputs = job.REPO / "bench/eval_inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = job.REPO.parent / name
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"evaluation input hash mismatch: {name}")
    shutil.copyfile(inputs / "manifest.json", job.OUT / "input_manifest.json")
    for checkpoint in inputs.glob("*.pt"):
        shutil.copyfile(checkpoint, job.OUT / checkpoint.name)
    worker = threading.Thread(target=job.measurements, daemon=True)
    worker.start()
    status = "failed"
    try:
        library = job.setup()
        if job.phase("scoring_regression", [
                "env", "RECOVERY_TEST_DEVICE=cuda", f"RECOVERY_TEST_LIBRARY={library}",
                sys.executable, "tests/test_recovery_scoring.py"], 180):
            raise RuntimeError("CUDA scoring regression failed before evaluation")
        evaluations = [
            ("untrained", ["--untrained"]),
            ("A_16128x8", ["--checkpoint", str(job.OUT / "ckpt_A_16128x8.pt")]),
            ("B_16128x16", ["--checkpoint", str(job.OUT / "ckpt_B_16128x16.pt")]),
            ("C_16128x16_lr3e5", ["--checkpoint", str(job.OUT / "ckpt_C_16128x16_lr3e5.pt")]),
            ("icra_epsilon_s0", ["--icra-root", str(inputs)]),
        ]
        for tag, source in evaluations:
            command = [
                sys.executable, "-u", "bench/eval_ours.py", "--device", "cuda",
                "--library", library, "--backend", "inline", "--batched",
                "--seed", "0", "--dirs", "fwd,back,left,right,up,down",
                "--mags", "2.5,4.0", "--episodes", "32", "--cap-steps", "1200",
                "--results", str(job.OUT / f"eval_{tag}.json"), *source]
            if job.phase(f"eval_{tag}", command, 900):
                raise RuntimeError(f"{tag} failed a protocol gate; scores are not yet valid")
        status = "finished"
    except Exception:
        (job.OUT / "failure.log").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        job.STATE.update(status=status, finished=time.time(), current=None)
        job.progress()
        job.STOP.set()
        worker.join(timeout=45)
        job.atomic_json(job.OUT / "outcomes.json", job.STATE)
        job.snapshot()
    return 0 if status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
