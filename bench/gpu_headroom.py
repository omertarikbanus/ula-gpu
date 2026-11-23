#!/usr/bin/env python3
"""GPU load and saturation measurement for the frozen-policy closed loop.

Runs the same full closed loop as bench/icra_walk_gpu.py (frozen seed-0 Epsilon
actor, command 0.7 m/s, evaluation reset, physics graph on CUDA) at increasing
world counts to answer: "we ran x worlds but the GPU could have run y."

Per n in --worlds: warm up 30 policy steps, then time --reps x --steps policy
steps with zero host syncs inside; record mean ms per policy step, env
steps/s (n*steps/wall) and controller world-ticks/s (5 ticks per policy step
per world). Decomposition: CUDA-event pairs around (i) the policy forward
alone on a stored obs batch and (ii) the full env.step, plus untimed
micro-benchmarks of the controller ABI loop (tick + act, the env's own
tensors) and env._physics() alone (physics + PD, act included). nvidia-smi
utilization is sampled every 200 ms during the timed reps only (mean, p95,
peak memory; skipped gracefully when absent). Per-n failures (e.g. CUDA OOM)
are recorded and the sweep continues. The JSON carries, per n, the four
components -- policy forward, controller, physics+PD, unattributed remainder
(contacts + obs assembly + Python; note act is counted in both the controller
and physics+PD numbers, the remainder absorbs the overlap) -- plus marginal
ms per added world, the saturation n (env steps/s growth below 10%), and a
linear memory extrapolation to the 15 GB T4 budget.
"""

import argparse
import gc
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from icra_walk_gpu import (DEFAULT_POLICY_ROOT, close_env, make_env,  # noqa: E402
                           step_loop, sync)
from ula_gpu.icra_inference import EpsilonActor  # noqa: E402

WARMUP_STEPS, DECOMP_STEPS, MICRO_ITERS = 30, 20, 20
GPU_BUDGET_BYTES = 15 * 1024 ** 3          # T4 15 GB memory budget


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--library", type=Path, help="libula_batch.so path (default: build/ or build-cuda/)")
    p.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT,
                   help="directory containing epsilon_s{0,1,2}")
    p.add_argument("--worlds", default="64,256,1024,2048,4096", help="comma list of world counts")
    p.add_argument("--steps", type=int, default=150, help="policy steps per timed rep")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--results", type=Path, default=REPO / "bench/artifacts/results_gpu_headroom.json")
    p.add_argument("--quick", action="store_true", help="local CPU smoke: worlds 1,8, steps 50, reps 1")
    a = p.parse_args(argv)
    if a.quick:
        a.worlds, a.steps, a.reps = "1,8", 50, 1
    a.worlds_list = [int(x) for x in str(a.worlds).split(",") if x.strip()]
    if not a.worlds_list or a.steps < 1 or a.reps < 1:
        p.error("need --steps/--reps >= 1 and one world count")
    return a


def gpu_sampler_start():
    """nvidia-smi -> temp csv every 200 ms; returns (proc, file, path) or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="ula-smi-")
        out = open(fd, "w")
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader", "-lms", "200"], stdout=out, stderr=subprocess.DEVNULL)
        return proc, out, path
    except Exception:
        return None


def gpu_sampler_stop(handle):
    if handle is None:
        return None
    proc, out, path = handle
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    out.close()
    utils, mems = [], []
    try:
        for line in Path(path).read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                try:
                    utils.append(float(parts[0].rstrip(" %")))
                    mems.append(float(parts[1].rstrip(" MiB")))
                except ValueError:
                    pass
    finally:
        Path(path).unlink(missing_ok=True)
    if not utils:
        return {"error": "no nvidia-smi samples parsed"}
    utils.sort()
    return {"samples": len(utils),
            "mean_util_pct": statistics.fmean(utils),
            "p95_util_pct": utils[min(len(utils) - 1, round(0.95 * len(utils)))],
            "peak_mem_mib": max(mems) if mems else None}


def decompose(env, actor, obs):
    """Event-paired (perf_counter on CPU) ms per policy step: (i) the policy
    forward alone, replaying a stored obs batch; (ii) the full env.step,
    driven by on-policy actions computed outside the event pair so the step
    measurement stays on a healthy walking gait (a stale-obs action loop makes
    the robot stumble and skews the contact cost)."""
    stored = obs[:, :54].clone()
    cuda = env.device.type == "cuda"

    def mean_ms(fn):
        if not cuda:
            sync(env)
            t0 = time.perf_counter()
            for _ in range(DECOMP_STEPS):
                fn()
            sync(env)
            return 1000 * (time.perf_counter() - t0) / DECOMP_STEPS
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(DECOMP_STEPS)]
        for s, e in pairs:
            s.record()
            fn()
            e.record()
        sync(env)
        return statistics.fmean(s.elapsed_time(e) for s, e in pairs)

    policy_ms = mean_ms(lambda: actor(stored))
    if cuda:
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(DECOMP_STEPS)]
        for s, e in pairs:
            action = actor(obs[:, :54])       # on-policy action, outside the pair
            s.record()
            obs, *_ = env.step(action, auto_reset=False)
            e.record()
        sync(env)
        step_ms = statistics.fmean(s.elapsed_time(e) for s, e in pairs)
    else:
        step_ms = 0.0
        for _ in range(DECOMP_STEPS):
            action = actor(obs[:, :54])
            t0 = time.perf_counter()
            obs, *_ = env.step(action, auto_reset=False)
            step_ms += time.perf_counter() - t0
        step_ms = 1000 * step_ms / DECOMP_STEPS
    return {"policy_forward_ms": policy_ms, "env_step_ms": step_ms,
            "method": "cuda-events" if cuda else "perf-counter (cpu)"}


def micro(env):
    """Untimed micro-benchmarks: controller ABI loop and physics+PD per policy step."""
    def clock(fn):
        sync(env)
        t0 = time.perf_counter()
        for _ in range(MICRO_ITERS):
            fn()
        sync(env)
        return 1000 * (time.perf_counter() - t0) / MICRO_ITERS   # ms per call

    c = env.controller
    tick = clock(lambda: c.tick(env.q, env.v, env.previous, env.command, env.noise,
                                None, env.actor, env.vw, env.status))
    act = clock(lambda: c.act(env.length, env.speed, env.ctrl))
    phys = clock(env._physics)
    tick, act, phys = tick * env.policy_ticks, act * env.policy_ticks * env.substeps, \
        phys * env.policy_ticks
    return {"controller_tick_ms": tick, "controller_act_ms": act,
            "controller_ms": tick + act,          # the tick+act controller-only loop
            "physics_pd_ms": phys,                # env._physics(): act/PD included
            "micro_iters": MICRO_ITERS}


def measure_n(args, actor, n):
    env = make_env(n, args.device, args.library, 100, graph=True)
    try:
        obs = env.reset(command=(0.7, 0., 0.))
        if env.use_graph:
            env.capture_physics()
        obs = step_loop(env, actor, obs, WARMUP_STEPS)
        handle = gpu_sampler_start() if env.device.type == "cuda" else None
        walls = []
        for _ in range(args.reps):
            sync(env)
            t0 = time.perf_counter()
            obs = step_loop(env, actor, obs, args.steps)
            sync(env)
            walls.append(time.perf_counter() - t0)
        gpu = gpu_sampler_stop(handle)
        wall = statistics.fmean(walls)
        rec = {
            "n": n, "reps": args.reps, "steps": args.steps, "graph": bool(env.use_graph),
            "host_syncs": 0,  # by construction inside the timed reps (see icra_walk_gpu)
            "mean_ms_per_policy_step": 1000 * wall / args.steps,
            "env_steps_per_s": n * args.steps / wall,
            "controller_world_ticks_per_s": n * args.steps * env.policy_ticks / wall,
            "gpu": gpu,
        }
        rec["decomposition"] = {**decompose(env, actor, obs), **micro(env)}
        m = rec["decomposition"]
        # remainder = env.step total - policy - controller tick - physics+PD
        # (physics+PD already contains act; the shared act cost shows in both
        #  controller_ms and physics_pd_ms and is absorbed here once).
        m["remainder_ms"] = m["env_step_ms"] - m["policy_forward_ms"] - m["controller_tick_ms"] \
            - m["physics_pd_ms"]
        m["note"] = ("components come from separate passes (events: policy forward on a stored "
                     "batch, on-policy env.step; untimed micro: controller tick+act, _physics). "
                     "remainder = env_step - policy - tick - physics+PD, where physics+PD already "
                     "contains act (controller_ms double-counts it; the remainder absorbs that "
                     "overlap and pass-to-pass noise, so it can go slightly negative on a "
                     "contended CPU)")
        return rec
    finally:
        close_env(env)


def saturation(runs):
    """Marginal cost per added world, saturation n, memory-extrapolated worlds."""
    ok = [r for r in runs if "error" not in r]
    for prev, cur in zip(ok, ok[1:]):
        cur["marginal_ms_per_added_world"] = \
            (cur["mean_ms_per_policy_step"] - prev["mean_ms_per_policy_step"]) / (cur["n"] - prev["n"])
    sat = next((cur["n"] for prev, cur in zip(ok, ok[1:])
                if cur["env_steps_per_s"] <= 1.10 * prev["env_steps_per_s"]), None)
    fit, worlds = None, None
    pts = [(r["n"], r["gpu"]["peak_mem_mib"] * 2 ** 20) for r in ok
           if r.get("gpu") and r["gpu"].get("peak_mem_mib")]
    if len(pts) >= 2:
        nn = [p[0] for p in pts]
        mm = [p[1] for p in pts]
        nbar, mbar = statistics.fmean(nn), statistics.fmean(mm)
        b = statistics.fmean((x - nbar) * (y - mbar) for x, y in pts) / \
            statistics.fmean((x - nbar) ** 2 for x in nn)
        a = mbar - b * nbar
        fit = {"intercept_bytes": a, "per_world_bytes": b, "budget_bytes": GPU_BUDGET_BYTES,
               "samples": pts}
        if b > 0:
            worlds = max(0, int((GPU_BUDGET_BYTES - a) / b))
    return {"marginal_ms_per_added_world": {r["n"]: r["marginal_ms_per_added_world"]
                                            for r in ok[1:]},
            "saturation_n": sat, "memory_fit": fit, "worlds_memory_allows": worlds}


def main(argv=None):
    args = parse_args(argv)
    actor = EpsilonActor(args.policy_root, 0, args.device)   # frozen seed-0 policy
    runs = []
    for n in args.worlds_list:
        try:
            rec = measure_n(args, actor, n)
            d = rec["decomposition"]
            gpu = rec.get("gpu") or {}
            util = f"{gpu.get('mean_util_pct', float('nan')):.0f}%/" \
                   f"{gpu.get('p95_util_pct', float('nan')):.0f}%@{gpu.get('peak_mem_mib')}MiB" \
                if gpu else "n/a"
            print(f"headroom {args.device} n={n:<6d} step={rec['mean_ms_per_policy_step']:7.2f}ms "
                  f"eps={rec['env_steps_per_s']:>11,.0f}/s ticks={rec['controller_world_ticks_per_s']:>11,.0f}/s "
                  f"policy={d['policy_forward_ms']:6.2f}ms ctrl={d['controller_ms']:6.2f}ms "
                  f"phys+PD={d['physics_pd_ms']:6.2f}ms rest={d['remainder_ms']:+5.2f}ms "
                  f"util={util}", flush=True)
        except Exception as exc:   # e.g. CUDA OOM at a large n: record and continue
            rec = {"n": n, "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:300]}
            print(f"headroom {args.device} n={n:<6d} FAILED  {rec['error']}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        runs.append(rec)
    sat = saturation(runs)
    largest = max((r["n"] for r in runs if "error" not in r), default=0)
    print(f"saturation: ran {largest} worlds; saturated at "
          f"{sat['saturation_n'] if sat['saturation_n'] else 'none within range'}; "
          f"memory allows ~{sat['worlds_memory_allows'] if sat['worlds_memory_allows'] else '?'} worlds",
          flush=True)
    result = {
        "harness": "gpu_headroom", "device": args.device, "steps": args.steps,
        "reps": args.reps, "worlds": args.worlds_list, "host": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "command_mps": 0.7, "policy_seed": 0,
        "warmup_steps": WARMUP_STEPS, "runs": runs, "saturation": sat,
        "verdict": f"ran {largest} worlds; saturated at "
                   f"{sat['saturation_n'] if sat['saturation_n'] else 'none within range'}; "
                   f"memory allows ~{sat['worlds_memory_allows'] if sat['worlds_memory_allows'] else '?'} worlds",
    }
    path = args.results
    if str(path) != "-":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"results json: {path}", flush=True)
    else:
        print(json.dumps(result, indent=1))
    return 0 if all("error" not in r for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
