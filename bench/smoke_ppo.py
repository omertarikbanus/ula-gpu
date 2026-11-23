#!/usr/bin/env python3
"""Two PPO rollouts on Go2 Warp. Tests the wiring, not policy quality or speed."""

import argparse
import json
from pathlib import Path
import sys

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.env import Go2Env
from ula_gpu.policy import Go2Policy


class SmokeVecEnv(VecEnv):
    """SB3's NumPy bridge, not the final GPU-resident training path."""

    render_mode = None

    def __init__(self, plant):
        self.plant = plant
        self.terminations = self.timeouts = self.failures = 0
        super().__init__(plant.n, spaces.Box(-np.inf, np.inf, (plant.obs_dim,), np.float32),
                         spaces.Box(-1, 1, (plant.action_dim,), np.float32))

    def reset(self):
        # Use the same seed whether SB3 queues a seed or the caller supplies it at construction.
        if self._seeds[0] is not None:
            self.plant.rng.manual_seed(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        obs = self.plant.reset()
        self.plant.capture_physics()
        return obs.cpu().numpy()

    def step_async(self, actions):
        self.actions = self.plant.as_tensor(actions)

    def step_wait(self):
        obs, reward, terminated, truncated, final, failed = self.plant.step(self.actions)
        ob, rew = obs.cpu().numpy(), reward.cpu().numpy()
        term, trunc, bad = torch.stack((terminated, truncated, failed)).cpu().numpy()
        self.terminations += int(term.sum())
        self.timeouts += int(trunc.sum())
        self.failures += int(bad.sum())
        done = term | trunc
        infos = [{"TimeLimit.truncated": bool(t)} for t in trunc]
        if done.any():
            last = final.cpu().numpy()
            for i in np.flatnonzero(done):
                infos[i]["terminal_observation"] = last[i].copy()
        return ob, rew, done, infos

    def close(self):
        self.plant.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("no per-world attributes in the smoke adapter")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("no per-world methods in the smoke adapter")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def check_inputs(policy, env):
    obs = torch.zeros(2, env.obs_dim, device=policy.device)
    obs[1, env.actor_dim:] = 1
    with torch.no_grad():
        actions = policy.get_distribution(obs).mode()
        values = policy.predict_values(obs)
    if (not torch.allclose(actions[0], actions[1], atol=1e-7, rtol=0)
            or torch.allclose(values[0], values[1])):
        raise RuntimeError("privileged inputs did not reach only the critic")


def check_step_contract(raw):
    raw.reset()
    actions = np.zeros((raw.num_envs, raw.plant.action_dim), dtype=np.float32)
    obs, reward, done, _ = raw.step(actions)
    if done.any() or not np.isfinite(obs).all() or not np.isfinite(reward).all():
        raise RuntimeError("unexpected termination on the first policy step")
    # A bad action must terminate only its own world, not produce a timeout.
    actions[0, 0] = np.nan
    _, reward, done, info = raw.step(actions)
    if (not done[0] or done[1:].any() or info[0]["TimeLimit.truncated"]
            or "terminal_observation" not in info[0] or reward[0] != -10
            or raw.failures or raw.plant.steps.cpu().tolist() != [0] + [2] * (raw.num_envs - 1)):
        raise RuntimeError("termination or partial reset is not isolated")
    raw.terminations = raw.timeouts = raw.failures = 0


def run(args):
    torch.set_num_threads(1)
    wp.config.quiet = True
    n, horizon, batches = 4, 16, 2
    plant = Go2Env(n, args.device, library=args.library, seed=args.seed,
                   max_steps=8, graph=args.device == "cuda")
    raw = SmokeVecEnv(plant)
    try:
        check_step_contract(raw)
        env = VecNormalize(raw, gamma=.995, norm_obs=True, norm_reward=True, clip_reward=10.)
        model = PPO(Go2Policy, env, device=args.device, seed=args.seed, n_steps=horizon,
                    batch_size=32, n_epochs=2, learning_rate=1e-5, ent_coef=.05, gamma=.995,
                    gae_lambda=.97, clip_range=.1, clip_range_vf=.2, vf_coef=1.,
                    max_grad_norm=.1, target_kl=.01,
                    policy_kwargs=dict(actor_obs_dim=plant.actor_dim, log_std_init=-3.,
                                       net_arch=dict(pi=[256, 128], vf=[256, 128])))
        check_inputs(model.policy, plant)
        before = {k: v.detach().clone() for k, v in model.policy.named_parameters()}
        print(f"Reset and policy checks passed; collecting {n * horizon * batches} transitions.", flush=True)
        model.learn(total_timesteps=n * horizon * batches)
        actor_changed = not torch.equal(model.policy.action_net.weight, before["action_net.weight"])
        critic_changed = not torch.equal(model.policy.value_net.weight, before["value_net.weight"])
        losses = {k: float(model.logger.name_to_value[k]) for k in
                  ("train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/approx_kl")}
        rollout = model.rollout_buffer
        if (model.num_timesteps != n * horizon * batches or raw.failures or not raw.timeouts
                or not actor_changed or not critic_changed or not np.isfinite(list(losses.values())).all()
                or not np.isfinite(rollout.advantages).all()
                or not np.isfinite(rollout.returns).all()
                or not all(torch.isfinite(v).all() for v in model.policy.parameters())):
            raise RuntimeError("PPO smoke failed: missing updates/resets, nonfinite state, or controller error")
        return dict(device=args.device, interactions=model.num_timesteps, rollouts=batches,
                    terminations=raw.terminations, timeouts=raw.timeouts, controller_errors=raw.failures,
                    actor_changed=actor_changed, critic_changed=critic_changed, losses=losses)
    finally:
        raw.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--library", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=Path, default=Path("bench/artifacts/results_ppo_smoke.json"))
    args = p.parse_args()
    result = run(args)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print("PPO smoke PASS:", json.dumps(result), flush=True)
