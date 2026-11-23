#!/usr/bin/env python3
"""Bounded Epsilon-recipe PPO pilot on the Warp Go2 env, fully on device.

This supports pipeline validation and bounded learning experiments, not a
paper-equivalent run: environment and PPO deviations are documented, while
everything (normalization, policy,
buffers, GAE, the update) stays on one torch device. The host reads scalars
once per PPO update. See docs/PPO_SMOKE.md for the recorded deviations.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.env import Go2Env
from ula_gpu.ppo_epsilon import EpsilonPolicy, Normalizer, PPO, Rollout

EPSILON_STEPS = 85_065_728
EPSILON_BOUNDARIES = (8_100_000, 24_300_000)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--library", type=Path)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=4_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=Path("bench/artifacts/pilot/checkpoint.pt"))
    p.add_argument("--results", type=Path, default=Path("bench/artifacts/pilot/results.json"))
    p.add_argument("--quick", action="store_true", help="tiny wiring check: 4 envs, 2 updates")
    p.add_argument("--update-graph", action="store_true",
                   help="capture the PPO update in a CUDA graph (validated separately)")
    p.add_argument("--backend", choices=("vendor", "inline"), default="vendor",
                   help="solver backend: NVIDIA batched LU/SVD, or the fused inline kernels")
    p.add_argument("--overlap", action="store_true",
                   help="run each PPO update on a second stream while the next rollout "
                        "collects; rollouts then use weights one update stale (declared "
                        "deviation) and the KL early stop is disabled")
    p.add_argument("--target-kl", type=float, default=0.01,
                   help="KL early-stop threshold; 0 or negative disables it")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="save a resumable checkpoint every N updates (0 = only at the end)")
    p.add_argument("--resume", action="store_true",
                   help="restore training state with fresh episodes (not exact simulator continuation)")
    a = p.parse_args()
    if a.quick:
        a.n_envs, a.n_steps, a.batch_size, a.total_steps = 4, 16, 32, 128
    if not (1 <= a.batch_size <= a.n_envs * a.n_steps and a.total_steps % (a.n_envs * a.n_steps) == 0):
        p.error("total-steps must be a multiple of n-envs * n-steps")
    if a.total_steps <= 0 or a.checkpoint_every < 0 or a.learning_rate <= 0:
        p.error("steps and learning-rate must be positive; checkpoint-every must be nonnegative")
    if a.resume and (not a.checkpoint.is_file() or a.overlap):
        p.error("--resume needs an existing checkpoint and does not support --overlap")
    return a


def make_env(a, device):
    """Build the training env. The kick curriculum is the ICRA schedule,
    scaled to the pilot horizon; interaction-space boundaries follow the
    documented Epsilon ramp, compressed by the same factor as the horizon."""
    scale = a.total_steps / EPSILON_STEPS
    boundaries = [round(b * scale) for b in EPSILON_BOUNDARIES]
    torch.manual_seed(a.seed)
    # `curriculum` wiring is finalized against ula_gpu/kicks; the env owns
    # per-world interaction counting (one per policy step).
    return Go2Env(a.n_envs, device, library=a.library, seed=a.seed, training=True,
                  max_steps=1000, graph=(a.device == "cuda"),
                  kick="velocity", curriculum=(a.total_steps, boundaries),
                  backend=a.backend)


def walk_check(a, device, state):
    """Deterministic frozen-policy walk of the pilot checkpoint, 3 s at 0.7 m/s."""
    policy = EpsilonPolicy().to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    mean, var = state["normalizer"]["obs_rms"][:2]
    mean, var = mean.to(device), var.to(device)
    env = Go2Env(8, device, library=a.library, seed=7, training=False,
                  max_steps=1 << 30, graph=(a.device == "cuda"), kick=None,
                  backend=a.backend)
    try:
        obs = env.reset(command=(0.7, 0., 0.))
        env.capture_physics()
        fallen = torch.zeros(8, dtype=torch.bool, device=device)
        start = env.q[:, 0].clone()
        with torch.no_grad():
            for _ in range(300):
                normalized = ((obs[:, :54] - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
                action = policy.distribution(normalized)[0].clamp(-1, 1)
                obs, _r, term, _t, _f, failed = env.step(action, auto_reset=False)
                fallen |= term | failed
        torch.cuda.synchronize() if device.type == "cuda" else None
        forward = float((env.q[:, 0] - start).mean())
        return dict(forward_m=forward, falls=int(fallen.sum()),
                    mean_vx=float(env.v[:, 0].mean()))
    finally:
        env.close()


def kl_setting(a):
    return None if a.target_kl <= 0 else a.target_kl


def collect(a, env, policy, normalizer, rollout, sampler, obs_norm, episode_start, device):
    """One rollout into `rollout`, using `policy` for every inference.

    Returns (obs_norm, episode_start, reward_sum_on_device, falls_on_device).
    No host synchronisation: every quantity stays on device until the caller
    reads it once per update."""
    reward_sum = torch.zeros((), device=device)
    falls = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(a.n_steps):
            mean, std = policy.distribution(obs_norm)
            noise = torch.randn(mean.shape, generator=sampler, device=device)
            actions = mean + std * noise
            log_prob = (-0.5 * noise.square() - std.log()
                        - 0.5 * math.log(2 * math.pi)).sum(-1)
            value = policy.value(obs_norm)
            _next, reward, term, trunc, final, _failed = \
                env.step(actions.clamp(-1, 1), auto_reset=False)
            done = term | trunc
            reward_norm = normalizer.reward(reward)
            reward_norm = torch.where(
                trunc & ~term,
                reward_norm + 0.995 * policy.value(normalizer.normalize_obs(final[:, :54])),
                reward_norm)
            rollout.add(obs_norm, actions, reward_norm, value, log_prob, episode_start)
            reward_sum += reward.mean()
            falls += term.sum()
            fresh = env._reset(done, full=False)
            obs_raw = torch.where(done[:, None], fresh, final)
            obs_norm = normalizer.observe(obs_raw[:, :54])
            normalizer.episode_ends(done)
            episode_start = done
        rollout.finish(policy, obs_norm, episode_start)
    return obs_norm, episode_start, reward_sum, falls


def main_overlap(a, device):
    """Overlapped training: the PPO update runs on a second CUDA stream while
    the next rollout collects.

    Declared deviations, both required for a genuine overlap: (1) the rollout
    uses a frozen copy of the weights that is one update stale, so the update
    never fights the rollout over the parameters; (2) the KL early stop is
    disabled, because checking it forces a host stall mid-update. The A100
    measurements put approx-KL at ~0.0025 against the 0.01 budget, so (2) does
    not bind in practice."""
    env = make_env(a, device)
    policy = EpsilonPolicy().to(device)
    infer = EpsilonPolicy().to(device)
    infer.load_state_dict(policy.state_dict())
    infer.eval()
    normalizer = Normalizer(a.n_envs, 54, device)
    ppo = PPO(policy, learning_rate=a.learning_rate, batch_size=a.batch_size, device=device,
              use_graph=a.update_graph, target_kl=None)
    buffers = [Rollout(a.n_steps, a.n_envs, 54, 25, device) for _ in range(2)]
    sampler = torch.Generator(device=device).manual_seed(a.seed)
    updates = a.total_steps // (a.n_envs * a.n_steps)
    upd_stream = torch.cuda.Stream(device=device)
    done_event = torch.cuda.Event()

    obs_raw = env.reset()
    env.capture_physics()
    obs_norm = normalizer.observe(obs_raw[:, :54])
    episode_start = torch.ones(a.n_envs, dtype=torch.bool, device=device)
    rows, checkpoints = [], []
    interactions = 0
    pending = False
    t_loop = time.perf_counter()
    for update in range(updates):
        buf = buffers[update % 2]
        t_rollout = time.perf_counter()
        obs_norm, episode_start, reward_sum, falls = \
            collect(a, env, infer, normalizer, buf, sampler, obs_norm, episode_start, device)
        rollout_s = time.perf_counter() - t_rollout
        # The update launched last round has had the whole rollout to finish;
        # waiting here costs nothing, and only now may the inference copy take
        # the new weights.
        if pending:
            torch.cuda.current_stream(device).wait_event(done_event)
            infer.load_state_dict(policy.state_dict())
            stats = ppo.read_stats()
            stats.update(interactions=interactions,
                         mean_raw_reward=float(prev_reward) / a.n_steps,
                         fall_rate=float(prev_falls) / (a.n_steps * a.n_envs),
                         rollout_s=prev_rollout_s, update_s=prev_update_s,
                         loop_wall_s=None)
            rows.append(stats)
            print(f"update {len(rows)}/{updates}: reward {stats['mean_raw_reward']:.3f} "
                  f"fall {stats['fall_rate']:.4f} kl {stats['approx_kl']:.4f} "
                  f"std {stats['std']:.3f} rollout {rollout_s:.1f}s (overlapped)", flush=True)
        prev_reward, prev_falls = reward_sum, falls
        prev_rollout_s = rollout_s
        t_update = time.perf_counter()
        upd_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(upd_stream):
            ppo.update(buf, defer=True)
            done_event.record()
        prev_update_s = time.perf_counter() - t_update
        pending = True
        interactions += a.n_envs * a.n_steps
    # Flush the last update.
    torch.cuda.current_stream(device).wait_event(done_event)
    stats = ppo.read_stats()
    stats.update(interactions=interactions, mean_raw_reward=float(prev_reward) / a.n_steps,
                 fall_rate=float(prev_falls) / (a.n_steps * a.n_envs),
                 rollout_s=prev_rollout_s, update_s=prev_update_s, loop_wall_s=None)
    rows.append(stats)
    loop_wall = time.perf_counter() - t_loop
    for row in rows:
        row["loop_wall_s"] = loop_wall
    state = dict(policy=policy.state_dict(), normalizer=normalizer.state_dict(),
                 interactions=interactions)
    torch.save(state, a.checkpoint)
    walk = walk_check(a, device, state)
    result = dict(device=a.device, seed=a.seed, interactions=interactions,
                  n_envs=a.n_envs, n_steps=a.n_steps, batch_size=a.batch_size,
                  updates=updates, checkpoints=checkpoints, walk=walk,
                  backend=a.backend, overlap=True, target_kl=None,
                  loop_wall_s=loop_wall, env_steps_per_s=interactions / loop_wall,
                  final=rows[-1], first=rows[0])
    a.results.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    print(f"overlapped: {interactions} steps in {loop_wall:.1f}s "
          f"= {interactions / loop_wall:,.0f} env/s", flush=True)
    print(json.dumps(result["walk"]), flush=True)


def main():
    a = parse_args()
    device = torch.device(a.device)
    torch.set_num_threads(1)
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    a.results.parent.mkdir(parents=True, exist_ok=True)
    if a.overlap:
        if a.device != "cuda":
            raise SystemExit("--overlap needs --device cuda (two streams)")
        return main_overlap(a, device)
    env = make_env(a, device)
    policy = EpsilonPolicy().to(device)
    normalizer = Normalizer(a.n_envs, 54, device)
    ppo = PPO(policy, learning_rate=a.learning_rate, batch_size=a.batch_size, device=device, use_graph=a.update_graph,
              target_kl=kl_setting(a))
    rollout = Rollout(a.n_steps, a.n_envs, 54, 25, device)
    sampler = torch.Generator(device=device).manual_seed(a.seed)
    interactions = 0
    rows, checkpoints = [], []
    updates = a.total_steps // (a.n_envs * a.n_steps)
    log_path = a.results.with_suffix(".csv")
    start_update = 0
    saved = None
    prior_wall = 0.0
    config = {key: getattr(a, key) for key in
              ("seed", "n_envs", "n_steps", "batch_size", "total_steps", "backend",
               "learning_rate", "target_kl", "update_graph")}

    if a.resume:
        saved = torch.load(a.checkpoint, map_location=device, weights_only=False)
        if saved.get("config") != config:
            raise ValueError("resume configuration differs from the saved training run")
        policy.load_state_dict(saved["policy"])
        ppo.optimizer.load_state_dict(saved["optimizer"])
        normalizer.load_state_dict(saved["normalizer"])
        start_update = int(saved["update"]) + 1
        interactions = int(saved["interactions"])
        rows = saved.get("rows", [])
        prior_wall = saved.get("loop_wall_s", 0.0)
        if not 0 <= start_update <= updates or interactions != start_update * a.n_envs * a.n_steps:
            raise ValueError("checkpoint update/interaction counters are inconsistent")
        print(f"resuming from {a.checkpoint} at update {start_update}/{updates} "
              f"({interactions} interactions)", flush=True)
        if rows:
            with log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def save_state(update_done, state_rows):
        state = dict(policy=policy.state_dict(),
                        optimizer=ppo.optimizer.state_dict(),
                        normalizer=normalizer.state_dict(),
                        update=update_done, interactions=interactions,
                        rows=state_rows, config=config, format_version=2,
                        resume_mode="fresh_episodes",
                        loop_wall_s=prior_wall + time.perf_counter() - t_loop,
                        sampler_rng=sampler.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                        env_rng=env.rng.get_state(),
                        kick_rng=env.kicks.rng.get_state(),
                        kick_interactions=env.kicks.interactions,
                        kicks_total=env.kicks.total_delivered)
        temporary = a.checkpoint.with_suffix(a.checkpoint.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(a.checkpoint)
        return state

    obs_raw = env.reset()
    env.capture_physics()
    if saved is not None:
        # Simulator/controller internals are not serialised. Start fresh episodes,
        # retain learned statistics and the lifetime curriculum clock, and do not
        # reuse discounted returns belonging to the abandoned episodes.
        normalizer.returns.zero_()
        env.kicks.interactions.copy_(saved["kick_interactions"])
        env.kicks.total_delivered.copy_(saved["kicks_total"])
        env.rng.set_state(saved["env_rng"].cpu())
        env.kicks.rng.set_state(saved["kick_rng"].cpu())
        sampler.set_state(saved["sampler_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
    obs_norm = normalizer.observe(obs_raw[:, :54])
    episode_start = torch.ones(a.n_envs, dtype=torch.bool, device=device)
    raw_reward_sum = torch.zeros((), device=device)
    falls = torch.zeros((), device=device)

    # One host scalar per update keeps the loop itself sync-free.
    t_loop = time.perf_counter()
    for update in range(start_update, updates):
        t_rollout = time.perf_counter()
        with torch.no_grad():
            for _ in range(a.n_steps):
                mean, std = policy.distribution(obs_norm)
                noise = torch.randn(mean.shape, generator=sampler, device=device)
                actions = mean + std * noise
                log_prob = (-0.5 * noise.square() - std.log()
                            - 0.5 * math.log(2 * math.pi)).sum(-1)
                value = policy.value(obs_norm)
                obs_next, reward, term, trunc, final, _failed = \
                    env.step(actions.clamp(-1, 1), auto_reset=False)
                done = term | trunc
                reward_norm = normalizer.reward(reward)
                reward_norm = torch.where(
                    trunc & ~term,
                    reward_norm + 0.995 * policy.value(normalizer.normalize_obs(final[:, :54])),
                    reward_norm)
                rollout.add(obs_norm, actions, reward_norm, value, log_prob, episode_start)
                raw_reward_sum += reward.mean()
                falls += term.sum()
                fresh = env._reset(done, full=False)
                obs_raw = torch.where(done[:, None], fresh, final)
                obs_norm = normalizer.observe(obs_raw[:, :54])
                normalizer.episode_ends(done)
                episode_start = done
        with torch.no_grad():
            rollout.finish(policy, obs_norm, episode_start)
        rollout_s = time.perf_counter() - t_rollout
        stats = ppo.update(rollout)
        stats.update(interactions=interactions + a.n_envs * a.n_steps,
                     mean_raw_reward=float(raw_reward_sum) / a.n_steps,
                     fall_rate=float(falls) / (a.n_steps * a.n_envs),
                      rollout_s=rollout_s, update_s=time.perf_counter() - t_rollout - rollout_s)
        if not all(math.isfinite(value) for value in stats.values() if isinstance(value, (int, float))):
            raise FloatingPointError(f"nonfinite PPO statistics at update {update + 1}: {stats}")
        stats["kicks_total"] = int(env.kicks.total_delivered)
        stats["loop_wall_s"] = prior_wall + time.perf_counter() - t_loop
        raw_reward_sum.zero_(); falls.zero_()
        rows.append(stats)
        interactions = stats["interactions"]
        if a.checkpoint_every and (update + 1) % a.checkpoint_every == 0:
            save_state(update, rows)
            with log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"update {update + 1}/{updates}: reward {stats['mean_raw_reward']:.3f} "
              f"fall {stats['fall_rate']:.3f} kl {stats['approx_kl']:.4f} "
              f"std {stats['std']:.3f} rollout {stats['rollout_s']:.1f}s", flush=True)

    state = save_state(updates - 1, rows)
    loop_wall = prior_wall + time.perf_counter() - t_loop
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    walk = walk_check(a, device, state)
    result = dict(device=a.device, seed=a.seed, interactions=interactions,
                  n_envs=a.n_envs, n_steps=a.n_steps, batch_size=a.batch_size,
                  updates=updates, checkpoints=checkpoints, walk=walk,
                  backend=a.backend, overlap=False, target_kl=kl_setting(a),
                  learning_rate=a.learning_rate, resume_mode="fresh_episodes" if a.resume else None,
                  loop_wall_s=loop_wall, env_steps_per_s=interactions / loop_wall,
                  final=rows[-1], first=rows[0])
    a.results.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    print(json.dumps(result["walk"]), flush=True)


if __name__ == "__main__":
    main()
