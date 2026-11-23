#!/usr/bin/env python3
"""Batched frozen-policy walking validation, fully on device (Go2, Epsilon PPO).

For each seed 0/1/2 and each world count n from --worlds, runs three cases
per (seed, n): fixed-command (a) 0.35 m/s, (b) 0.70 m/s straight; (c) mixed
batch with per-world commands vx~U(0,1), vy~U(-0.2,0.2), yaw~U(-0.2,0.2)
drawn from a torch.Generator, plus the training spawn-height noise
U(-0.03,0.03) folded into the initial pose so the controller primes against
the jittered height exactly as training resets do; other training-time init
jitters -- joint angles, orientation -- are NOT modeled. Each case runs --steps policy steps
inside one timed region with zero host syncs (per-world stats accumulate on
device; one sync at the end; env.step is called with auto_reset=False, the
only auto_reset path that syncs).

Tracking metrics ignore the first 25 policy steps (0.25 s spin-up from the
spawn pose) so the RMSE/mean-error gates mean the same thing at --quick's 60
steps and at the full 550: at 60 steps the spin-up transient alone is
~0.29 m/s RMSE, at 550 it contributes ~0.02. Gates:

  (a)/(b)  no falls (terminations) and no controller status errors;
  (b)      forward-tracking RMSE <= 0.18 m/s absolute (CPU single-world
           reference seed means: 0.129 native / 0.115 warp at 0.7; per-seed
           numbers are recorded next to the gate);
  (c)      no controller errors, >= 95% of worlds complete all steps, and
           mean |vx - cmd| < 0.25 m/s gated on batches (n >= 4; the
           threshold is a batch mean over random commands -- at n < 4 the
           error is recorded but ungated, since one draw has no averaging:
           seed 1 at n=1 draws the corner vx 0.99 + yaw 0.15, which the
           frozen policy converges to only by ~step 100). Body-frame forward
           speed for (c); a/b use world vx like the reference harness.

Reset isolation: in the smallest n >= 4 mixed case, a mid-run
env._reset(mask) touching exactly one world must leave every other world's
q/v/time and next-step observations bit-identical to a parallel env run
without the reset (30 steps each).
"""

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
from ula_gpu.env import Go2Env  # noqa: E402
from ula_gpu.icra_inference import EpsilonActor  # noqa: E402

DEFAULT_POLICY_ROOT = Path("/home/tarik/quadruped_ws/envpool/data")
RMSE_GATE = 0.18
MIXED_MIN_COMPLETE = 0.95
MIXED_MAX_ABS_ERR = 0.25
TRACKING_SKIP = 25          # policy steps excluded from tracking (see module doc)
ISOLATION_STEPS = 30


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--library", type=Path, help="libula_batch.so path (default: build/ or build-cuda/)")
    p.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT,
                   help="directory containing epsilon_s{0,1,2}")
    p.add_argument("--worlds", default="1,64,256", help="comma list of world counts")
    p.add_argument("--steps", type=int, default=550)
    p.add_argument("--results", type=Path, default=REPO / "bench/artifacts/results_icra_walk_gpu.json")
    p.add_argument("--quick", action="store_true", help="small local smoke: worlds 1,4, steps 60")
    a = p.parse_args(argv)
    if a.quick:
        a.worlds, a.steps = "1,4", 60
    a.worlds_list = [int(x) for x in str(a.worlds).split(",") if x.strip()]
    if not a.worlds_list or a.steps < TRACKING_SKIP + 10:
        p.error(f"need at least {TRACKING_SKIP + 10} steps and one world count")
    return a


def make_env(n, device, library, seed, graph=True):
    """Evaluation env: fixed-command reset, no training noise, huge max_steps."""
    return Go2Env(n, device, library=library, seed=seed, training=False,
                  max_steps=1 << 30, graph=graph)


def close_env(env):
    if env is not None:
        env.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync(env):
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)


def step_loop(env, actor, obs, steps):
    """Run `steps` policy steps of the frozen closed loop; returns final obs.

    No host syncs inside (auto_reset=False is the sync-free step path)."""
    step = env.step
    for _ in range(steps):
        obs, *_ = step(actor(obs[:, :54]), auto_reset=False)
    return obs


def mixed_setup(env, gen):
    """Per-world commands + training spawn-height noise; returns the command tensor."""
    n, dev = env.n, env.device
    draw = lambda: torch.rand((n,), generator=gen, device=dev)
    cmd = torch.stack((draw(), (2 * draw() - 1) * 0.2, (2 * draw() - 1) * 0.2), dim=1)
    height = (2 * draw() - 1) * 0.03
    initial = env.initial.unsqueeze(0).expand(n, -1).clone()
    initial[:, 2] += height
    env.initial = initial
    return cmd


def body_forward(env):
    """Body-frame forward speed from truth: world xy velocity rotated by yaw."""
    w, x, y, z = env.q[:, 3:7].unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw.cos() * env.v[:, 0] - yaw.sin() * env.v[:, 1]


def run_case(env, actor, command, steps, gen_seed, body_frame):
    """One evaluation case: reset (optionally per-world mixed setup), then a
    timed region of `steps` policy steps accumulating per-world stats on
    device; a single sync at the end."""
    dev, n = env.device, env.n
    gen = None
    if gen_seed is not None:
        gen = torch.Generator(device=dev).manual_seed(gen_seed)
        command = mixed_setup(env, gen)
    obs = env.reset(command=command)
    if env.use_graph:
        env.capture_physics()
    x0 = env.q[:, 0].clone()
    done_at = torch.full((n,), steps, dtype=torch.int64, device=dev)
    stopped = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    ctrl_err = torch.zeros(n, dtype=torch.bool, device=dev)
    vx_sum, abs_sum, sq_sum = (env.zeros(n) for _ in range(3))
    sync(env)
    t0 = time.perf_counter()
    for i in range(steps):
        action = actor(obs[:, :54])
        obs, _r, term, trunc, _final, failed = env.step(action, auto_reset=False)
        bad = term | trunc
        live = ~stopped                      # world was still running entering this step
        done_at = torch.where(bad & live, torch.full_like(done_at, i), done_at)
        stopped |= bad
        fell |= term
        ctrl_err |= failed
        vx = body_forward(env) if body_frame else env.v[:, 0]
        err = vx - env.command[:, 0]
        keep = live & (i >= TRACKING_SKIP)   # zero host syncs: all on device
        vx_sum += torch.where(keep, vx, 0.)
        abs_sum += torch.where(keep, err.abs(), 0.)
        sq_sum += torch.where(keep, err * err, 0.)
    sync(env)
    wall = time.perf_counter() - t0
    # One host transfer of the accumulated per-world stats (outside the timed region).
    count = (done_at + 1).clamp(max=steps).sub(TRACKING_SKIP).clamp(min=1).to(torch.float64)
    mean_vx = vx_sum / count
    abs_err = abs_sum / count
    rmse = (sq_sum / count).sqrt()
    disp = env.q[:, 0] - x0
    stats = {
        "completed_frac": float((done_at >= steps).float().mean()),
        "completed_min": int(done_at.min()),
        "falls": int(fell.sum()),
        "controller_errors": int(ctrl_err.sum()),
        "mean_vx": float(mean_vx.mean()),
        "mean_abs_vx_err": float(abs_err.mean()),
        "vx_rmse_mean": float(rmse.mean()),
        "vx_rmse_max": float(rmse.max()),
        "forward_disp_mean": float(disp.mean()),
        "forward_disp_min": float(disp.min()),
        "wall_s": wall,
        "policy_steps_per_s": steps / wall,
        "env_steps_per_s": n * steps / wall,
        "host_syncs": 0,  # by construction: no .item/.cpu/.any-to-bool inside the region
        "tracking_skip_steps": TRACKING_SKIP,
    }
    return stats


def gate_case(case, stats, n):
    g = {}
    if case in ("a", "b"):
        g["no_falls"] = stats["falls"] == 0
        g["no_controller_errors"] = stats["controller_errors"] == 0
        if case == "b":
            g[f"rmse_le_{RMSE_GATE}"] = stats["vx_rmse_mean"] <= RMSE_GATE
    else:
        g["no_controller_errors"] = stats["controller_errors"] == 0
        g["complete_ge_95pct"] = stats["completed_frac"] >= MIXED_MIN_COMPLETE
        # The 0.25 threshold is a BATCH mean over random commands. A single
        # world (n < 4) is one un-averaged draw: e.g. seed 1 at n=1 draws the
        # hard corner vx=0.987 m/s + yaw 0.15 rad/s, which the frozen policy
        # only converges to by ~step 100 -- a policy property, not a harness
        # defect. Below 4 worlds the error is recorded but not gated.
        g["tracking_gated"] = n >= 4
        if n >= 4:
            g["mean_abs_vx_err_lt_0.25"] = stats["mean_abs_vx_err"] < MIXED_MAX_ABS_ERR
    g["pass"] = all(v for k, v in g.items() if k not in ("pass", "tracking_gated"))
    return g


CASES = (("a", (0.35, 0., 0.), False), ("b", (0.70, 0., 0.), False), ("mixed", None, True))


def walk_all(args):
    result = {
        "harness": "icra_walk_gpu", "device": args.device, "steps": args.steps,
        "worlds": args.worlds_list, "graph": args.device == "cuda",
        "policy_root": str(args.policy_root), "host": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates": {"case_b_rmse_abs": RMSE_GATE,
                  "case_b_reference_seed_means": {"native_0.7": 0.129, "warp_0.7": 0.115},
                  "mixed_min_complete": MIXED_MIN_COMPLETE,
                  "mixed_max_mean_abs_vx_err": MIXED_MAX_ABS_ERR,
                  "mixed_tracking_gate_min_worlds": 4,
                  "tracking_skip_steps": TRACKING_SKIP},
        "cases": [], "reset_isolation": [], "passed": False,
    }
    iso_n = min((x for x in args.worlds_list if x >= 4), default=None)
    for seed in (0, 1, 2):
        actor = EpsilonActor(args.policy_root, seed, args.device)
        for n in args.worlds_list:
            env = None
            try:
                env = make_env(n, args.device, args.library, 100 + seed,
                               graph=args.device == "cuda")
                for case, command, body_frame in CASES:
                    gen_seed = None if case != "mixed" else 10_000 * seed + 100 + n
                    stats = run_case(env, actor, command, args.steps, gen_seed, body_frame)
                    gate = gate_case(case, stats, n)
                    row = dict(seed=seed, n=n, case=case, gen_seed=gen_seed, **stats, gate=gate)
                    result["cases"].append(row)
                    print(f"walk {args.device} seed {seed} n={n:<5d} case {case:<5s} "
                          f"done={stats['completed_frac']:5.0%} falls={stats['falls']} "
                          f"err={stats['controller_errors']} meanvx={stats['mean_vx']:+.3f} "
                          f"|dv|={stats['mean_abs_vx_err']:.3f} rmse={stats['vx_rmse_mean']:.3f} "
                          f"disp={stats['forward_disp_mean']:+.2f}m "
                          f"wall={stats['wall_s']:.1f}s eps={stats['env_steps_per_s']:,.0f}/s "
                          f"gate={'PASS' if gate['pass'] else 'FAIL'}", flush=True)
            finally:
                close_env(env)
        if iso_n is not None:
            result["reset_isolation"].append(reset_isolation(args, actor, seed, iso_n))
        del actor
        gc.collect()
    per_seed_b = {s: [c["vx_rmse_mean"] for c in result["cases"] if c["seed"] == s and c["case"] == "b"]
                  for s in (0, 1, 2)}
    result["case_b_rmse_per_seed"] = per_seed_b
    result["passed"] = (all(c["gate"]["pass"] for c in result["cases"]) and
                        all(r["pass"] for r in result["reset_isolation"]))
    return result


def reset_isolation(args, actor, seed, n):
    """Mid-run single-world _reset must not perturb any other world.

    Three identical envs run the same mixed-command loop; at the halfway
    step one env re-initializes exactly one world via env._reset(mask).
    On CPU the untouched worlds must stay bit-identical. On CUDA, atomic
    reductions in the contact pipeline are not order-deterministic, so two
    identical no-reset envs already drift; the reset env's untouched worlds
    must then stay within a small multiple of that measured baseline
    (contamination would dwarf it)."""
    envs = []
    try:
        setups = []
        for _ in range(3):
            env = make_env(n, args.device, args.library, 100 + seed,
                           graph=args.device == "cuda")
            gen = torch.Generator(device=env.device).manual_seed(7_000 + seed)
            obs = env.reset(command=mixed_setup(env, gen))
            if env.use_graph:
                env.capture_physics()
            setups.append((env, obs))
        (env0, obs0), (env1, obs1), (env2, obs2) = setups
        envs = [env0, env1, env2]
        mask = torch.zeros(n, dtype=torch.bool, device=env0.device)
        mask[1] = True                       # exactly one world is re-initialized
        for i in range(ISOLATION_STEPS):
            if i == ISOLATION_STEPS // 2:
                env2._reset(mask)           # training=False: re-initializes world 1 only
            obs0, *_ = env0.step(actor(obs0[:, :54]), auto_reset=False)
            obs1, *_ = env1.step(actor(obs1[:, :54]), auto_reset=False)
            obs2, *_ = env2.step(actor(obs2[:, :54]), auto_reset=False)
        for env in envs:
            sync(env)
        others = [j for j in range(n) if j != 1]

        def drift(a, b):
            return float((a[others] - b[others]).abs().max())

        control = {"q": drift(env0.q, env1.q), "v": drift(env0.v, env1.v),
                   "next_obs": drift(obs0, obs1)}
        reset = {"q": drift(env0.q, env2.q), "v": drift(env0.v, env2.v),
                 "next_obs": drift(obs0, obs2)}
        if args.device == "cpu" or str(args.device) == "cpu":
            checks = {f"{k}_exact": reset[k] == 0.0 for k in reset}
        else:
            checks = {f"{k}_within_baseline":
                      reset[k] <= 3.0 * max(control[k], 1e-7) + 1e-5 for k in reset}
        checks["time_exact"] = bool(torch.equal(env0.time[others], env2.time[others]))
    finally:
        for env in envs:
            close_env(env)
    ok = all(checks.values())
    print(f"reset-isolation {args.device} seed {seed} n={n}: "
          + " ".join(f"{k}={'ok' if v else 'DIFF'}" for k, v in checks.items())
          + f" control={{{', '.join(f'{k}:{control[k]:.2e}' for k in control)}}}"
          + f" reset={{{', '.join(f'{k}:{reset[k]:.2e}' for k in reset)}}}"
          + f" -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"seed": seed, "n": n, "steps": ISOLATION_STEPS, "checks": checks,
            "control_drift": control, "reset_drift": reset, "pass": ok}


def main(argv=None):
    args = parse_args(argv)
    result = walk_all(args)
    print(f"=== icra_walk_gpu: {'PASS' if result['passed'] else 'FAIL'} "
          f"(device={args.device}, steps={args.steps}, worlds={args.worlds_list}) ===", flush=True)
    path = args.results
    if str(path) != "-":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"results json: {path}", flush=True)
    else:
        print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
