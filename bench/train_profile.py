#!/usr/bin/env python3
"""Profile the training loop per region and check the captured PPO update.

Regions timed with CUDA events, all read back once at the end (no per-step
syncs): policy forward + sampling, the env step (controller + physics +
contacts + obs), the per-step reset pass, the timeout value patch, GAE, and
the PPO update (eager, and captured when --update-graph is set).

Also validates the captured update against the eager update: two identical
policies consume the same fixed buffer with the same minibatch order; their
losses must agree to float tolerance and the weight drift must be tiny.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.env import Go2Env
from ula_gpu.ppo_epsilon import EpsilonPolicy, Normalizer, PPO, Rollout


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--library", type=Path)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--update-graph", action="store_true")
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    if a.device == "cuda" and a.n_envs * a.n_steps % a.batch_size:
        p.error("n-envs * n-steps must be a multiple of batch-size")
    return a


class Events:
    """CUDA-event pairs per region, accumulated on the host at the end."""

    def __init__(self, device):
        self.device = device
        self.pairs = {}

    def region(self, name, function, *args):
        if self.device.type != "cuda":
            return function(*args)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = function(*args)
        end.record()
        self.pairs.setdefault(name, []).append((start, end))
        return out

    def report(self):
        rows = {}
        for name, pairs in self.pairs.items():
            total = sum(start.elapsed_time(end) for start, end in pairs)
            rows[name] = {"ms_total": total, "calls": len(pairs),
                          "ms_per_call": total / len(pairs)}
        return rows


def collect_and_update(args, use_graph):
    """One training-loop profile run at the requested geometry."""
    device = torch.device(args.device)
    events = Events(device)
    env = Go2Env(args.n_envs, args.device, library=args.library, seed=args.seed,
                 training=True, max_steps=1000, graph=device.type == "cuda")
    policy = EpsilonPolicy().to(device)
    normalizer = Normalizer(args.n_envs, 54, device)
    ppo = PPO(policy, batch_size=args.batch_size, device=device, use_graph=use_graph)
    rollout = Rollout(args.n_steps, args.n_envs, 54, 25, device)
    sampler = torch.Generator(device=device).manual_seed(args.seed)
    clock = time.perf_counter()

    obs_raw = env.reset()
    env.capture_physics()
    obs_norm = normalizer.observe(obs_raw[:, :54])
    episode_start = torch.ones(args.n_envs, dtype=torch.bool, device=device)
    for update in range(args.updates):
        t_update = time.perf_counter()
        for _ in range(args.n_steps):
            def policy_step():
                with torch.no_grad():
                    mean, std = policy.distribution(obs_norm)
                    noise = torch.randn(mean.shape, generator=sampler, device=device)
                    actions = mean + std * noise
                    log_prob = (-0.5 * noise.square() - std.log()
                                - 0.5 * 1.8378770664093453).sum(-1)
                    value = policy.value(obs_norm)
                return actions, log_prob, value
            actions, log_prob, value = events.region("policy", policy_step)
            def env_step():
                return env.step(actions.clamp(-1, 1), auto_reset=False)
            obs_next, reward, term, trunc, final, _failed = events.region("env_step", env_step)
            done = term | trunc
            def reward_fold():
                r_norm = normalizer.reward(reward)
                with torch.no_grad():
                    patch = 0.995 * policy.value(normalizer.normalize_obs(final[:, :54]))
                return torch.where(trunc & ~term, r_norm + patch, r_norm)
            reward_norm = events.region("timeout_patch", reward_fold)
            rollout.add(obs_norm, actions, reward_norm, value, log_prob, episode_start)
            def reset_pass():
                return env._reset(done, full=False)
            fresh = events.region("reset", reset_pass)
            obs_raw = torch.where(done[:, None], fresh, final)
            def observe_next():
                nonlocal obs_norm
                obs_norm = normalizer.observe(obs_raw[:, :54])
                normalizer.episode_ends(done)
            events.region("normalize", observe_next)
            episode_start = done
        def finish():
            with torch.no_grad():
                rollout.finish(policy, obs_norm, episode_start)
        events.region("gae", finish)
        stats = ppo.update(rollout)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"update {update + 1}: {time.perf_counter() - t_update:.1f}s "
              f"kl {stats['approx_kl']:.4f} reward-ish loss {stats['loss']:.3f}", flush=True)
    wall = time.perf_counter() - clock
    env.close()
    rate = args.updates * args.n_envs * args.n_steps / wall
    return {"regions_ms": events.report(), "wall_s": wall,
            "env_steps_per_s": rate, "use_graph": bool(use_graph and ppo.use_graph)}


def update_equivalence(args):
    """Captured vs eager update on identical inputs must match to tolerance."""
    device = torch.device(args.device)
    if device.type != "cuda":
        return {"skipped": "cuda only"}
    torch.manual_seed(3)
    count, batch = 4096, 1024
    base = EpsilonPolicy().to(device)
    with torch.no_grad():
        obs = torch.randn(count, 54, device=device)
        actions = torch.randn(count, 25, device=device)
    rollout = Rollout(1, count, 54, 25, device)
    rollout.pos = 0
    with torch.no_grad():
        mean, std = base.distribution(obs)
        noise = torch.randn(count, 25, device=device)
        log_prob = (-0.5 * noise.square() - std.log() - 0.9189385332046727).sum(-1)
        rollout.add(obs, actions, torch.randn(count, device=device), base.value(obs),
                    log_prob, torch.zeros(count, dtype=torch.bool, device=device))
        rollout.finish(base, obs, torch.zeros(count, dtype=torch.bool, device=device))
    results = {}
    for mode in ("eager", "graph"):
        policy = EpsilonPolicy().to(device)
        policy.load_state_dict(base.state_dict())
        ppo = PPO(policy, batch_size=batch, device=device, use_graph=(mode == "graph"),
                  n_epochs=3)
        torch.manual_seed(5)
        stats = ppo.update(rollout)
        results[mode] = {"stats": stats,
                         "weights": {k: v.detach().clone() for k, v in policy.state_dict().items()}}
    eager, graph = results["eager"], results["graph"]
    deltas = {k: float((eager["weights"][k] - graph["weights"][k]).abs().max())
              for k in eager["weights"]}
    stat_diff = {k: abs(eager["stats"][k] - graph["stats"][k]) for k in eager["stats"]}
    ok = max(deltas.values()) <= 1e-5 and max(stat_diff.values()) <= 1e-4
    out = {"max_weight_delta": max(deltas.values()), "weight_deltas": deltas,
           "stat_differences": stat_diff, "pass": bool(ok)}
    print(f"update equivalence: max weight delta {max(deltas.values()):.2e}, "
          f"max stat diff {max(stat_diff.values()):.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return out


def main():
    args = parse_args()
    result = {"equivalence": update_equivalence(args),
              "profile_eager": collect_and_update(args, use_graph=False)}
    if args.update_graph:
        result["profile_graphed_update"] = collect_and_update(args, use_graph=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    ok = result["equivalence"].get("pass", True)
    print(json.dumps({k: v["env_steps_per_s"] for k, v in result.items()
                      if isinstance(v, dict) and "env_steps_per_s" in v}, indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
