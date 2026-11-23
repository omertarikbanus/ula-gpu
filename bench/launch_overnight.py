#!/usr/bin/env python3
"""Colab notebook entry point: unpack, then keep the kernel executing the job.

Run through the official CLI with --timeout 180. The supervisor relays actual
child output and resource measurements. Do not detach it or leave the kernel
idle, and do not interpret a dropped client as proof of VM termination.
"""

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO = Path("/content/ula-gpu")
Path("/tmp/opencode").mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    with tempfile.TemporaryDirectory(prefix="ula-launch-") as tmp:
        with tarfile.open("/content/ula-go2-vendor.tar.gz") as bundle:
            bundle.extractall(tmp, filter="data")
        Path(tmp, "ula-gpu").rename(REPO)
    print("bundle extracted", flush=True)

child = subprocess.Popen(
    [sys.executable, "-u", str(REPO / "bench/run_overnight.py")],
    cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in child.stdout:
    # Explicitly print through IPython's stream; inheriting OS stdout alone does
    # not reliably publish a subprocess's output as notebook IOPub messages.
    print(line, end="", flush=True)
code = child.wait()
if code:
    raise RuntimeError(f"training supervisor exited {code}")
