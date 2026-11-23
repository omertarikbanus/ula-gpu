#!/usr/bin/env python3
"""Bounded, single-seed Colab experiment with streamed output and live backups.

The notebook remains busy running this supervisor. The local driver downloads
an atomic archive during training, not just when the final result appears.
Checkpoints support a declared fresh-episode restart, not exact simulator
continuation. No retry can recover files that were never copied off the VM.
"""

from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

REPO = Path("/content/ula-gpu")
OUT = Path("/content/ula-go2-overnight")
BUDGET = 84_897_792  # 658 x (16128 x 8) == 329 x (16128 x 16).
CONFIGS = (
    ("A_16128x8", 16128, 8, 1e-5),
    ("B_16128x16", 16128, 16, 1e-5),
    ("C_16128x16_lr3e5", 16128, 16, 3e-5),
)
STATE = {}
LOCK = threading.Lock()
STOP = threading.Event()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def progress():
    atomic_json(OUT / "progress.json", STATE)
    print(json.dumps(STATE), flush=True)


def snapshot():
    """Publish only closed/atomically replaced files; checkpoint bytes are hashed.

    Holding each checkpoint open pins the inode if training replaces it while
    this archive is being written. The manifest describes exactly those bytes.
    """
    with LOCK:
        temporary = OUT / "snapshot.tar.tmp"
        manifest = {}
        with tarfile.open(temporary, "w") as archive:
            for path in sorted(OUT.iterdir()):
                if path.suffix not in (".pt", ".json", ".csv", ".log", ".jsonl"):
                    continue
                with path.open("rb") as source:
                    info = archive.gettarinfo(str(path), arcname=path.name, fileobj=source)
                    if path.suffix == ".pt":
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                        source.seek(0)
                        manifest[path.name] = digest
                    archive.addfile(info, source)
            payload = json.dumps(manifest).encode()
            import io
            info = tarfile.TarInfo("checkpoint_hashes.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        temporary.replace(OUT / "snapshot.tar")


def measurements():
    while not STOP.wait(30):
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            memory = {line.split(":")[0]: line.split(":")[1].strip()
                      for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:"))}
            processes = subprocess.check_output(
                ["ps", "-eo", "pid,ppid,rss,comm", "--sort=-rss"], text=True, timeout=10)
            sample = dict(time=time.time(), phase=STATE.get("current"),
                          gpu_util_mem_mib_total_mib_watts=gpu,
                          host_memory=memory, top_rss_processes=processes.splitlines()[:8])
            with (OUT / "resources.jsonl").open("a") as log:
                log.write(json.dumps(sample) + "\n")
            print("resources " + json.dumps(sample), flush=True)
            snapshot()
        except Exception as exc:
            print(f"measurement/backup error: {exc!r}", flush=True)


def phase(name, command, budget=7200):
    STATE["current"] = name
    STATE["phases"][name] = "running"
    progress()
    started = time.monotonic()
    with (OUT / f"{name}.log").open("a", buffering=1) as log:
        child = subprocess.Popen(
            command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)

        def terminate():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)

        deadline = threading.Timer(budget, terminate)
        deadline.start()
        try:
            for line in child.stdout:
                log.write(line)
                print(f"[{name}] {line}", end="", flush=True)
            code = child.wait()
        finally:
            deadline.cancel()
            if child.poll() is None:
                terminate()
                child.wait()
    STATE["phases"][name] = "ok" if code == 0 else f"exit {code}"
    STATE["elapsed_s"][name] = round(time.monotonic() - started, 2)
    STATE["current"] = None
    progress()
    snapshot()
    return code


def setup():
    gpu, cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        text=True, timeout=15).strip().split(", ")
    STATE["gpu"] = gpu
    progress()
    if not Path("/usr/include/eigen3/Eigen/Core").is_file():
        if phase("apt_update", ["apt-get", "-qq", "update"], 180):
            raise RuntimeError("apt update failed")
        if phase("eigen", ["apt-get", "-qq", "install", "-y", "libeigen3-dev"], 180):
            raise RuntimeError("Eigen installation failed")
    build = REPO / "build-cuda"
    commands = (
        ("configure", ["cmake", "-S", str(REPO), "-B", str(build),
                       "-DCMAKE_BUILD_TYPE=Release", "-DULA_ENABLE_CUDA_REPLAY=ON",
                       f"-DULA_REPLAY_REFERENCE={REPO / 'data/go2_reference.jsonl'}",
                       f"-DCMAKE_CUDA_ARCHITECTURES={cc.replace('.', '')}"]),
        ("build", ["cmake", "--build", str(build), "--parallel", "8"]),
        ("ctest", ["ctest", "--test-dir", str(build), "--output-on-failure"]),
        ("dependencies", [sys.executable, "-m", "pip", "install", "mujoco==3.11.0",
                          "mujoco-warp==3.11.0", "warp-lang==1.19.0",
                          "-r", str(REPO / "bench/requirements-ppo.txt")]),
    )
    for name, command in commands:
        if phase(name, command, 900):
            raise RuntimeError(f"{name} failed")
    if phase("training_job_tests", [sys.executable, "tests/test_training_job.py"], 120):
        raise RuntimeError("training/checkpoint transport regression checks failed")
    return str(build / "libula_batch.so")


def train_command(lib, tag, n_envs, n_steps, rate, total=BUDGET):
    return [
        sys.executable, "-u", "bench/pilot_epsilon.py", "--device", "cuda",
        "--library", lib, "--backend", "inline", "--n-envs", str(n_envs),
        "--n-steps", str(n_steps), "--batch-size", "1024" if n_envs > 64 else "512",
        "--total-steps", str(total), "--seed", "0", "--update-graph",
        "--learning-rate", str(rate), "--target-kl", "0.01",
        "--checkpoint-every", "10" if n_envs > 64 else "1",
        "--checkpoint", str(OUT / f"ckpt_{tag}.pt"),
        "--results", str(OUT / f"train_{tag}.json"),
    ]


def evaluation(lib, tag, checkpoint=None, smoke=False):
    command = [
        sys.executable, "-u", "bench/eval_ours.py", "--device", "cuda",
        "--library", lib, "--backend", "inline", "--seed", "0",
        "--dirs", "fwd,down" if smoke else "fwd,back,left,right,up,down",
        "--mags", "1.0" if smoke else "2.5,4.0",
        "--episodes", "4" if smoke else "32",
        "--results", str(OUT / f"eval_{tag}.json"),
    ]
    command += ["--checkpoint", str(checkpoint)] if checkpoint else ["--untrained"]
    return phase(f"eval_{tag}", command, 2400)


def main():
    OUT.mkdir(exist_ok=True)
    STATE.update(run_id=os.environ.get("ULA_RUN_ID", "colab-validated"),
                 started=time.time(), current=None, phases={}, elapsed_s={},
                 seed=0, interaction_budget=BUDGET, configs=CONFIGS)
    atomic_json(OUT / "outcomes.json", {"status": "running", "run_id": STATE["run_id"]})
    worker = threading.Thread(target=measurements, daemon=True)
    worker.start()
    try:
        lib = setup()
        if phase("smoke", train_command(lib, "smoke", 64, 16, 1e-5, total=2048), 600):
            raise RuntimeError("CUDA training preflight failed")
        if evaluation(lib, "smoke", OUT / "ckpt_smoke.pt", smoke=True):
            raise RuntimeError("CUDA checkpoint evaluation preflight failed")
        STATE["current"] = "awaiting_verified_local_backup"
        progress()
        snapshot()
        # Full training is forbidden until the local driver has downloaded and
        # hash-verified the CUDA smoke checkpoint and result files.
        until = time.monotonic() + 180
        while not (OUT / "backup_ack.json").exists():
            if time.monotonic() > until:
                raise RuntimeError("no acknowledgement of a verified local checkpoint backup")
            time.sleep(2)
        ack = json.loads((OUT / "backup_ack.json").read_text())
        expected = hashlib.sha256((OUT / "ckpt_smoke.pt").read_bytes()).hexdigest()
        if ack != {"run_id": STATE["run_id"], "ckpt_smoke.pt": expected}:
            raise RuntimeError("local backup acknowledgement does not match this job")
        if evaluation(lib, "untrained"):
            raise RuntimeError("untrained control failed protocol sanity checks")
        for tag, n_envs, n_steps, rate in CONFIGS:
            code = phase(f"train_{tag}", train_command(lib, tag, n_envs, n_steps, rate), 5400)
            if code:
                raise RuntimeError(f"training {tag} failed; see streamed child/resource logs")
            # A weak trained policy can fail pre-onset walking. Keep that result
            # rather than suppressing it or calling it a successful sanity gate.
            evaluation(lib, tag, OUT / f"ckpt_{tag}.pt")
        status = "finished"
    except Exception:
        (OUT / "failure.log").write_text(traceback.format_exc())
        traceback.print_exc()
        status = "failed"
    finally:
        STATE["finished"] = time.time()
        STATE["status"] = status
        progress()
        STOP.set()
        worker.join(timeout=45)
        atomic_json(OUT / "outcomes.json", STATE)
        snapshot()
    return 0 if status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
