#!/usr/bin/env python3
"""Small frozen Epsilon comparison: fixed pose, clean observations, no kicks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.policy import Go2Policy
from ula_gpu.icra_inference import ICRA_HASHES  # published digests, single source of truth


def load_policy(root, seed):
    model = root / f"epsilon_s{seed}" / "quadruped_ppo_model.zip"
    stats = model.with_name(f"{model.stem}_vecnormalize.pkl")
    for path, expected in zip((model, stats), ICRA_HASHES[seed]):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"not the published Epsilon artifact: {path}")
    # The Python 3.10 learning-rate closure is not needed for frozen inference.
    agent = PPO.load(model, device="cpu",
                     custom_objects={"policy_class": Go2Policy, "lr_schedule": None})
    # VecNormalize.load requires a VecEnv. Only apply inference-time obs normalization.
    with stats.open("rb") as f:
        norm = pickle.load(f)
    if (agent.num_timesteps != 85_065_728 or agent.observation_space.shape != (54,) or
            agent.action_space.shape != (25,) or not isinstance(norm, VecNormalize) or
            not norm.norm_obs or norm.obs_rms.mean.shape != (54,)):
        raise ValueError("unexpected Epsilon policy or normalization dimensions")
    norm.training = norm.norm_reward = False
    return lambda obs: agent.predict(norm.normalize_obs(obs), deterministic=True)[0]


class Native:
    def __init__(self, args, speed, seed):
        import envpool
        import ula_envs

        self.library_hash = sha256(Path(ula_envs._quadruped.__file__).read_bytes()).hexdigest()
        overlay = f"""
[simulation]
external_force_enable = false
external_force_mode = "off"
sensor_noise_enable = false
[state_estimator]
state_source = "truth"
[rl_command_source]
policy_random_velocity_enabled = false
policy_base_velocity = [{speed}, 0.0, 0.0]
desired_velocity = [{speed}, 0.0, 0.0]
[logserver]
enabled = false
"""
        self.env = envpool.make("QuadrupedWBC-v0", env_type="gymnasium", num_envs=1,
                                seed=seed, sim_config_path=str(args.native_config),
                                sim_config_overlay=overlay, wbc_randomize_init=False,
                                wbc_randomize_friction=False, max_episode_steps=args.steps + 1)

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action.astype(self.env.action_space.dtype))
        return (obs, bool((term | trunc | info["fall"].astype(bool))[0]), False,
                float(info["x_velocity"][0]), float(info["x_position"][0]),
                float(info["sim_time"][0]))

    def close(self):
        self.env.close()


class Warp:
    def __init__(self, args, speed, seed):
        import torch
        import warp as wp
        from ula_gpu.env import Go2Env

        wp.config.quiet = True
        self.torch, self.speed = torch, speed
        self.env = Go2Env(1, args.device, library=args.library, seed=seed, training=False,
                          max_steps=args.steps + 1, graph=args.device == "cuda")
        self.library_hash = sha256(Path(self.env.controller.lib._name).read_bytes()).hexdigest()

    def reset(self):
        obs = self.env.reset(command=(self.speed, 0., 0.))
        self.env.capture_physics()
        return obs[:, :54].cpu().numpy()

    def step(self, action):
        env = self.env
        obs, _, term, trunc, _, failed = env.step(env.as_tensor(action), auto_reset=False)
        status = self.torch.stack((term[0], trunc[0], failed[0])).cpu().tolist()
        state = self.torch.stack((env.v[0, 0], env.q[0, 0], env.time[0])).cpu().tolist()
        return obs[:, :54].cpu().numpy(), status[0] or status[1], status[2], *state

    def close(self):
        self.env.close()


def run_case(env, predict, seed, speed, steps):
    try:
        obs = env.reset()
        if obs.shape != (1, 54) or abs(float(obs[0, 0]) - speed) > 1e-6:
            raise RuntimeError(f"reset did not report the requested command {speed}")
        initial = obs[0].tolist()
        vx, pos, time = [], [], []
        for _ in range(steps):
            if not np.isfinite(obs).all():
                raise RuntimeError("nonfinite actor input")
            obs, stopped, failed, v, x, t = env.step(predict(obs))
            if not np.isfinite((v, x, t)).all():
                raise RuntimeError("nonfinite simulator truth")
            vx.append(v)
            pos.append(x)
            time.append(t)
            if stopped or failed:
                break
        if not np.allclose(np.diff(time), .01, rtol=0., atol=5e-6):
            raise RuntimeError("simulator did not advance at 100 Hz")
        return dict(policy_seed=seed, eval_seed=100 + seed, command_mps=speed,
                    completed=len(vx) == steps and not (stopped or failed),
                    samples=len(vx), controller_failed=failed, stopped=stopped,
                    mean_vx_mps=float(np.mean(vx)), vx_rmse_mps=float(np.mean((np.array(vx) - speed)**2)**.5),
                    forward_m=pos[-1], initial_obs=initial, library_sha256=env.library_hash)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=("native", "warp"))
    p.add_argument("--policy-root", type=Path, required=True, help="directory containing epsilon_s{0,1,2}")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--native-config", type=Path, default=Path("/app/src/controller/config/robots/sim/envpool.toml"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--library", type=Path)
    p.add_argument("--speed", type=float, action="append", help="repeat for several forward commands (default 0.7)")
    p.add_argument("--steps", type=int, default=550)
    a = p.parse_args()
    speeds = a.speed or [0.7]
    if not 1 <= a.steps <= 1000 or not all(0 <= speed <= 1 for speed in speeds):
        p.error("expected 1–1000 steps and forward commands between 0 and 1 m/s")
    backend = Warp if a.backend == "warp" else Native
    rows = []
    for seed in range(3):
        predict = load_policy(a.policy_root, seed)
        for speed in speeds:
            row = run_case(backend(a, speed, 100 + seed), predict, seed, speed, a.steps)
            rows.append(row)
            print(f"{a.backend} seed {seed}, command {speed:.2f}: "
                  f"{'completed' if row['completed'] else 'stopped'} {row['samples']}/{a.steps}, "
                  f"mean vx {row['mean_vx_mps']:.3f}, RMSE {row['vx_rmse_mps']:.3f} m/s", flush=True)
    a.results.write_text(json.dumps(dict(backend=a.backend, device=a.device, steps=a.steps, cases=rows), indent=2) + "\n")
    return 0 if all(r["completed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
