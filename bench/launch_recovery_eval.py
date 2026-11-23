#!/usr/bin/env python3
"""Foreground Colab entry point for the corrected, evaluation-only job."""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO = Path("/content/ula-gpu")
Path("/tmp/opencode").mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    with tempfile.TemporaryDirectory(prefix="ula-evaluation-") as tmp:
        with tarfile.open("/content/ula-go2-vendor.tar.gz") as bundle:
            bundle.extractall(tmp, filter="data")
        Path(tmp, "ula-gpu").rename(REPO)
child = subprocess.Popen(
    [sys.executable, "-u", "bench/run_recovery_eval.py"], cwd=REPO,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in child.stdout:
    print(line, end="", flush=True)
if child.wait():
    raise RuntimeError("corrected recovery evaluation failed; see saved logs")
