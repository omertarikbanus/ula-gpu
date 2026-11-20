"""Helpers shared by mjx_bench.py and warp_bench.py (no jax / warp imports here)."""

import json
import platform
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = str(REPO / 'models/go2/scene_gpu.xml')
DEFAULT_RESULTS = str(REPO / 'bench/results.json')

# Real-system timing: physics dt = 2 ms, controller ticks every physics step
# (500 Hz), policy acts every 5 physics steps (100 Hz).
PHYSICS_DT = 0.002
POLICY_EVERY = 5

# Standing pose (hip, knee per leg, abad 0): feet roughly under the hips.
STAND_JOINTS = (0.0, -0.8, 1.6)
STAND_BASE_Z = 0.32  # lowest foot-sphere centre at about its radius (0.022)


def parse_int_list(s):
    return [int(x) for x in str(s).split(',') if x.strip()]


def gpu_info():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ''
    except Exception:
        return ''


def report(engine, device, variant, n, total_steps, best_s, dt, extra=None, results=None):
    """Print one compact line and return the result dict.

    steps/s is aggregate (N * physics steps / wall seconds). The realtime factor
    is aggregate simulated seconds per wall second (N * steps * dt / wall); the
    single-robot realtime factor is that divided by N.
    """
    sps = n * total_steps / best_s
    rtf = n * total_steps * dt / best_s
    print(f'{engine:6s} {device:4s} {variant:10s} N={n:<7d} {sps:>14,.0f} steps/s  '
          f'realtime x{rtf:,.1f} (per-robot x{rtf / n:.3f})', flush=True)
    rec = dict(engine=engine, device=device, variant=variant, num_envs=n, steps=total_steps,
               best_wall_s=best_s, steps_per_s=sps, realtime_factor=rtf, dt=dt,
               gpu=gpu_info(), host=platform.node(), time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    if extra:
        rec.update(extra)
    return rec


def append_results(path, recs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = []
    data.extend(recs)
    p.write_text(json.dumps(data, indent=1))


def failure(engine, device, variant, n, err):
    """Record for a run that raised (e.g. out of GPU memory at a large N)."""
    msg = f'{type(err).__name__}: {err}'.splitlines()[0][:300]
    print(f'{engine:6s} {device:4s} {variant:10s} N={n:<7d} FAILED  {msg}', flush=True)
    return dict(engine=engine, device=device, variant=variant, num_envs=n, error=msg,
                gpu=gpu_info(), host=platform.node(), time=time.strftime('%Y-%m-%dT%H:%M:%S'))
