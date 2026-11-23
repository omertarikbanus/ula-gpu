#!/usr/bin/env python3
"""Policy-free Warp closed-loop verification and throughput harness (Go2, pre-PPO).

Script, via normalized actions only (env.command stays zero), 60 policy steps
(~0.6 s at 100 Hz) per phase: per-leg Fz -0.4 in action slots [5,8,11,14] in
every phase, action[0]=0.6 during walk and turn, action[2]=0.4 during turn
only; controller parity is not retested (the 1092-tick replay covers it).
smoke (n=2) gates: walk forward > 0.05 m, turn yaw > +0.02 rad (CPU reference
+0.17 m, +0.079 rad), no terminations or controller status hits, finite,
control_dt 0.002 with exactly 5x2 substeps, sim time within float32 error;
throughput: per N in --num-envs, warmed wall time of the full env step, --reps
timed regions, syncs only at endpoints, term/failed accumulated on device.
Results JSON (--results, default bench/artifacts/results_verify_warp_loop.json or '-');
exit 0 pass, 1 failed check, 2 unusable environment.
"""
import argparse, gc, json, platform, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PHASE_STEPS, FZ_SLOTS, FZ = 60, (5, 8, 11, 14), -0.4
A_WALK, A_TURN = 0.6, 0.4           # normalized action[0] walk+turn, action[2] turn
FORWARD_MIN, TURN_MIN = 0.05, 0.02  # smoke gates, not controller parity
PHASES = ("stand", "walk", "turn")
DEFAULT_RESULTS = str(Path(__file__).resolve().parent / "artifacts/results_verify_warp_loop.json")


def action_for(phase):
    a = [0.0] * 25
    for s in FZ_SLOTS:
        a[s] = FZ
    if phase >= 1:
        a[0] = A_WALK
    if phase >= 2:
        a[2] = A_TURN
    return a


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--library", default=None, help="libula_batch.so path (default: build/ or build-cuda/)")
    p.add_argument("--num-envs", default="1", help="comma list of world counts for throughput")
    p.add_argument("--steps", type=int, default=200, help="policy steps per timed region")
    p.add_argument("--reps", type=int, default=2, help="timed regions per N; best kept")
    p.add_argument("--graph", action="store_true", help="capture physics via Go2Env.capture_physics (CUDA)")
    p.add_argument("--results", default=DEFAULT_RESULTS, help="result JSON path or '-'")
    args = p.parse_args(argv)
    try:
        args.num_envs_list = [int(x) for x in args.num_envs.split(",") if x.strip()]
        assert args.steps >= 1 and args.reps >= 1 and args.num_envs_list and min(args.num_envs_list) >= 1
    except Exception:
        p.error("invalid --steps/--reps/--num-envs")
    return args


def fail(result, stage, msg):
    print((line := f"FAIL [{stage}] {msg}"), file=sys.stderr, flush=True)
    result["failures"].append(line)


def load_stack(args):
    try:
        import torch, warp, mujoco, mujoco_warp  # noqa: F401
        from ula_gpu.env import Go2Env
    except Exception as exc:
        raise RuntimeError(f"cannot load the stack ({type(exc).__name__}: {exc}); "
                           f"needs the bench/requirements.txt venv; nothing is installed here")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but torch.cuda.is_available() is False")
    if args.library and not Path(args.library).exists():
        raise RuntimeError(f"--library {args.library} does not exist")
    if not args.library and not (REPO / ("build-cuda" if args.device == "cuda" else "build")
                                 / "libula_batch.so").exists():
        raise RuntimeError("controller library missing under build/ or build-cuda/; "
                           "build it or pass --library")
    warp.config.quiet = True
    return torch, Go2Env


def sync(torch, env):
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)


def make_env(torch, Go2Env, args, n):
    env = Go2Env(n, args.device, library=args.library, seed=1, training=False,
                 max_steps=1 << 30, graph=args.graph)
    env.reset()                      # warm-up; also primes the controller
    if args.graph and env.use_graph:
        env.capture_physics()        # stream choice stays inside Go2Env
    sync(torch, env)
    return env


def close_env(torch, env):
    if env is not None:
        env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def script_actions(torch, env):
    return [torch.as_tensor(action_for(p), dtype=torch.float32, device=env.device).repeat(env.n, 1)
            for p in range(3)]


def run_script(env, acts, start, steps, bad):
    # Drive `steps` policy steps from global index `start`; bad stays on device.
    step = env.step
    for i in range(start, start + steps):
        _o, _r, term, _t, _f, failed = step(acts[(i // PHASE_STEPS) % 3], auto_reset=False)
        bad |= term | failed         # no host sync inside the region


def yaw(torch, q):
    return torch.atan2(2.0 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
                       1.0 - 2.0 * (q[:, 5] ** 2 + q[:, 6] ** 2))


def smoke(torch, Go2Env, args, result):
    stage, n = "smoke", 2
    out = {"ran": True, "num_envs": n, "passed": False}
    env = None
    try:
        env = make_env(torch, Go2Env, args, n)
        out["timing"] = {"control_dt_s": float(env.control_dt), "policy_ticks": int(env.policy_ticks),
                         "substeps": int(env.substeps), "graph": bool(env.use_graph)}
        if not (env.control_dt == 0.002 and env.policy_ticks == 5 and env.substeps == 2):
            fail(result, stage, f"expected control_dt 0.002 with exactly 5x2 substeps, got {out['timing']}")
        acts = script_actions(torch, env)
        bad = torch.zeros(n, dtype=torch.bool, device=env.device)
        t_reset, marks = float(env.time.detach().cpu()[0]), []
        for k in range(3):
            run_script(env, acts, k * PHASE_STEPS, PHASE_STEPS, bad)
            sync(torch, env)                       # phase endpoints only
            marks.append(env.q.detach().cpu().clone())
        finite = bool(torch.isfinite(marks[-1]).all())
        if bool(bad.any()) or not finite:
            fail(result, stage, f"fall/status/non-finite in the script: bad={bad.tolist()} finite={finite}")
        heading = torch.stack((yaw(torch, marks[0]).cos(), yaw(torch, marks[0]).sin()), 1)
        fwd = float(((marks[1][:, :2] - marks[0][:, :2]) * heading).sum(1).min())
        dyaw = float((yaw(torch, marks[2]) - yaw(torch, marks[1])).min())
        out.update(forward_min_m=fwd, dyaw_min_rad=dyaw)
        if fwd <= FORWARD_MIN:
            fail(result, stage, f"walk forward {fwd:.3f} m <= {FORWARD_MIN} m (CPU reference +0.17 m)")
        if dyaw <= TURN_MIN:
            fail(result, stage, f"turn yaw {dyaw:+.3f} rad <= {TURN_MIN} rad (CPU reference +0.079 rad)")
        t_end, expected = float(env.time.detach().cpu()[0]), PHASE_STEPS * 3 * 0.01
        out["time_advance_s"] = t_end - t_reset
        if abs((t_end - t_reset) - expected) > 2e-3 + 900 * 2e-6:
            fail(result, stage, f"sim time advance {t_end - t_reset:.4f} s != {expected:.4f} s (float32 error)")
        print(f"smoke  device={env.device} n={n} forward={fwd:.3f}m dyaw={dyaw:+.3f}rad "
              f"time={t_end - t_reset:.4f}s (expected {expected:.4f}s) bad={int(bad.sum())}/{n}",
              flush=True)
    finally:
        close_env(torch, env)
        env = None
    out["passed"] = not any(f.startswith("FAIL [smoke]") for f in result["failures"])
    print(f"smoke: {'PASS' if out['passed'] else 'FAIL'}", flush=True)
    return out


def throughput(torch, Go2Env, args, result):
    stage = "throughput"
    out = {"ran": True, "steps": args.steps, "reps": args.reps, "runs": [], "passed": False}
    for n in args.num_envs_list:
        env = None
        try:
            env = make_env(torch, Go2Env, args, n)
            acts = script_actions(torch, env)
            bad = torch.zeros(n, dtype=torch.bool, device=env.device)
            run_script(env, acts, 0, PHASE_STEPS * 3, bad)   # warm-up + sanity script
            sync(torch, env)
            t_ref, best, index = float(env.time.detach().cpu()[0]), float("inf"), PHASE_STEPS * 3
            for _ in range(args.reps):
                sync(torch, env)                 # region endpoints only
                w = time.perf_counter()
                run_script(env, acts, index, args.steps, bad)  # timed, no syncs inside
                sync(torch, env)
                best, index = min(best, time.perf_counter() - w), index + args.steps
            if bool(bad.any()):
                fail(result, stage, f"N={n}: fall/controller-status hit in the run; timing not representative")
            if not bool(torch.isfinite(env.q.detach().cpu()).all()):
                fail(result, stage, f"N={n}: non-finite qpos after the run")
            drift = (float(env.time.detach().cpu()[0]) - t_ref) - args.reps * args.steps * 0.01
            if abs(drift) > 2e-3 + args.reps * args.steps * 5 * 2e-6:
                fail(result, stage, f"N={n}: sim time drift {drift:+.4f} s over the timed region")
            eps = n * args.steps / best
            run = {"num_envs": n, "graph": bool(env.use_graph), "best_wall_s": best,
                   "env_steps_per_s": eps, "controller_ticks_per_s": eps * env.policy_ticks,
                   "physics_steps_per_s": eps * env.policy_ticks * env.substeps,
                   "realtime_factor": eps * env.control_dt * env.policy_ticks,
                   "per_robot_realtime": eps * env.control_dt * env.policy_ticks / n}
            out["runs"].append(run)
            print(f"verify  {args.device:4s} N={n:<7d} {eps:>11,.0f} env/s "
                  f"{run['controller_ticks_per_s']:>11,.0f} tick/s {run['physics_steps_per_s']:>11,.0f} "
                  f"phys/s realtime x{run['realtime_factor']:,.1f} wall={best:.3f}s graph={run['graph']}",
                  flush=True)
        except Exception as exc:  # e.g. out of GPU memory at a large N
            fail(result, stage, f"N={n}: {type(exc).__name__}: {exc}")
            out["runs"].append({"num_envs": n, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            close_env(torch, env)
            env = None
    out["passed"] = not any(f.startswith("FAIL [throughput]") for f in result["failures"])
    print(f"throughput: {'PASS' if out['passed'] else 'FAIL'}", flush=True)
    return out


def write_results(args, result):
    if args.results == "-":
        print(json.dumps(result, indent=1))
        return
    p = Path(args.results)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1))
    print(f"results json: {p}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    result = {"harness": "verify_warp_loop", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "host": platform.node(), "config": vars(args).copy(),
              "stages": {}, "passed": False, "failures": []}
    try:
        torch, Go2Env = load_stack(args)
    except Exception as exc:
        result["failures"].append(f"setup: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        write_results(args, result)
        return 2
    for name, fn in (("smoke", smoke), ("throughput", throughput)):
        try:
            result["stages"][name] = fn(torch, Go2Env, args, result)
        except Exception as exc:
            fail(result, name, f"{type(exc).__name__}: {exc}")
            result["stages"][name] = {"ran": True, "passed": False, "error": str(exc),
                                      "traceback": traceback.format_exc()[-2000:]}
    result["passed"] = not result["failures"]
    write_results(args, result)
    print(f"=== verify_warp_loop: {'PASS' if result['passed'] else 'FAIL'} "
          f"(device={args.device}) ===", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
