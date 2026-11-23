#!/usr/bin/env python3
"""Official-Colab-CLI job driver: stream, verify live backups, release the GPU.

This process owns a bounded foreground notebook execution, plus one downloader
thread using the file API. It never queues extra execution cells behind training.
No GPU is released until the final archive has been downloaded and verified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import traceback


def restore_archive(archive_path, destination, run_id):
    """Check the entire archive before publishing any files or trusting status."""
    with tarfile.open(archive_path) as archive:
        files = {}
        for member in archive:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError(f"unexpected snapshot member: {member.name}")
            files[member.name] = archive.extractfile(member).read()
    state = json.loads(files["progress.json"])
    if state["run_id"] != run_id:
        raise ValueError("snapshot belongs to a different experiment")
    hashes = json.loads(files["checkpoint_hashes.json"])
    for name, digest in hashes.items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"checkpoint digest mismatch: {name}")
    for name, data in files.items():
        temporary = destination / (name + ".download")
        temporary.write_bytes(data)
        temporary.replace(destination / name)
    return state, hashes


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    events = args.output / "driver.jsonl"
    stop = threading.Event()
    final_verified = threading.Event()
    state = {}
    state_lock = threading.Lock()
    config = str(args.output / "sessions.json")

    def cli(*command, timeout=90, log=None):
        return subprocess.run(
            ["colab", "--config", config, *command], stdin=subprocess.DEVNULL,
            stdout=log if log else subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout)

    def event(kind, **details):
        value = dict(time=time.time(), kind=kind, **details)
        with state_lock:
            with events.open("a") as output:
                output.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    def pull():
        temporary = args.output / "incoming.tar"
        result = cli("download", "-s", args.session,
                     "/content/ula-go2-overnight/snapshot.tar", str(temporary), timeout=60)
        if result.returncode:
            event("download_unavailable", detail=result.stdout[-500:])
            return False
        latest, hashes = restore_archive(temporary, args.output, args.session)
        temporary.replace(args.output / "snapshot.tar")
        state.update(latest)
        event("backup_verified", current=latest.get("current"), hashes=hashes,
              phases=latest.get("phases", {}))
        if "ckpt_smoke.pt" in hashes and latest.get("current") == "awaiting_verified_local_backup":
            ack = args.output / "backup_ack.json"
            ack.write_text(json.dumps({"run_id": args.session, "ckpt_smoke.pt": hashes["ckpt_smoke.pt"]}))
            uploaded = cli("upload", "-s", args.session, str(ack),
                           "/content/ula-go2-overnight/backup_ack.json")
            if uploaded.returncode:
                raise RuntimeError(f"backup acknowledgement upload failed: {uploaded.stdout}")
            event("backup_acknowledged")
        outcomes = json.loads((args.output / "outcomes.json").read_text())
        if outcomes.get("status") in ("finished", "failed"):
            final_verified.set()
        return True

    def download_loop():
        while not stop.wait(45):
            try:
                pull()
            except Exception:
                event("backup_error", detail=traceback.format_exc())

    created = False
    try:
        allocation = cli("new", "-s", args.session, "--gpu", "A100", timeout=480)
        (args.output / "allocation.log").write_text(allocation.stdout)
        if allocation.returncode:
            raise RuntimeError(f"A100 allocation failed: {allocation.stdout[-1500:]}")
        created = True
        uploaded = cli("upload", "-s", args.session, str(args.bundle),
                       "/content/ula-go2-vendor.tar.gz", timeout=120)
        if uploaded.returncode:
            raise RuntimeError(uploaded.stdout)
        downloader = threading.Thread(target=download_loop, daemon=True)
        downloader.start()
        event("executing", session=args.session)
        with (args.output / "notebook.log").open("w", buffering=1) as log:
            result = cli(
                "exec", "-s", args.session, "-f", str(args.entry),
                "--timeout", "180", "--env", f"ULA_RUN_ID={args.session}",
                timeout=4 * 3600, log=log)
        event("execution_returned", code=result.returncode)
        stop.set()
        downloader.join(timeout=100)
        # Capture the final snapshot even if the stream failed. Preserve the VM
        # for inspection/reconnection unless completion/failure artifacts exist.
        for _ in range(3):
            try:
                if pull() and final_verified.is_set():
                    break
            except Exception:
                event("final_backup_error", detail=traceback.format_exc())
        if not final_verified.is_set():
            raise RuntimeError("no final snapshot: runtime is preserved for diagnosis; do not assume it died")
        event("completed", status=state.get("status"), phases=state.get("phases"))
        return 0 if state.get("status") == "finished" else 1
    finally:
        stop.set()
        if created and final_verified.is_set():
            released = cli("stop", "-s", args.session, timeout=90)
            event("runtime_release", code=released.returncode, detail=released.stdout)
        elif created:
            event("runtime_preserved", session=args.session,
                  reason="execution did not produce a verified final archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--entry", type=Path, default=Path(__file__).with_name("launch_overnight.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.bundle.is_file() or not args.entry.is_file():
        parser.error("bundle and notebook entry point must exist")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
