"""Device-resident PPO matching the ICRA Epsilon recipe.

Everything runs on one torch device: the running observation/reward
normalization, the actor-critic, the rollout buffers, GAE, and the clipped
surrogate update. The host launches kernels and reads update statistics.
The formulas follow stable-baselines3 2.7.1, but this is not an exact SB3
implementation: running statistics use float32 rather than float64, and the
captured path checks KL at epoch rather than minibatch boundaries.

- VecNormalize order: update obs RMS with the raw batch, then normalize;
  reward RMS follows discounted returns; observations and rewards are
  normalized before the buffer sees them.
- TimeLimit bootstrapping: a truncated episode's stored reward gets
  ``gamma * V(terminal_observation)`` added (SB3 issue #633), and GAE cuts
  at episode starts, exactly like ``RolloutBuffer.compute_returns_and_advantage``.
- PPO: clipped surrogate (0.1), clipped value loss (0.2), analytic Gaussian
  entropy, the k3 approximate KL with early stop at 1.5x target_kl,
  per-minibatch advantage normalization, grad-norm clip 0.1, and a
  log_std clamp to [0.05, 0.30] after every optimizer step.

Known deviations from the native Epsilon environment, deliberate and
documented in docs/PPO_SMOKE.md: the Warp env does not perturb joint
sensor readings, and the disturbance curriculum follows the documented
interaction-space schedule rather than the native driver's internal clock.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RunningMeanStd:
    """SB3 RunningMeanStd on device, float32 statistics."""

    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, batch):
        batch = batch.reshape(-1, *self.mean.shape)
        n = batch.shape[0]
        if n == 0:
            return
        delta = batch.mean(dim=0) - self.mean
        total = self.count + n
        new_mean = self.mean + delta * n / total
        m_a = self.var * self.count
        m_b = batch.var(dim=0, unbiased=False) * n
        m2 = m_a + m_b + delta.square() * self.count * n / total
        self.mean, self.var = new_mean, m2 / total
        self.count = total


class Normalizer:
    """VecNormalize equivalent: online obs, discounted-return reward RMS."""

    def __init__(self, n_envs, obs_dim, device, gamma=0.995, epsilon=1e-8):
        self.obs_rms = RunningMeanStd((obs_dim,), device)
        self.ret_rms = RunningMeanStd((), device)
        self.returns = torch.zeros(n_envs, device=device)
        self.gamma, self.epsilon = gamma, epsilon

    @torch.no_grad()
    def observe(self, obs):
        """Update obs statistics with a raw batch, then normalize it."""
        self.obs_rms.update(obs)
        return self.normalize_obs(obs)

    def normalize_obs(self, obs):
        return ((obs - self.obs_rms.mean) / (self.obs_rms.var + self.epsilon).sqrt()).clamp(-10, 10)

    @torch.no_grad()
    def reward(self, raw):
        """Update the return statistics with raw rewards, return normalized."""
        self.returns = self.returns * self.gamma + raw
        self.ret_rms.update(self.returns)
        return (raw / (self.ret_rms.var + self.epsilon).sqrt()).clamp(-10, 10)

    @torch.no_grad()
    def episode_ends(self, done):
        self.returns = torch.where(done, torch.zeros_like(self.returns), self.returns)

    def state_dict(self):
        return {"returns": self.returns.detach().clone(),
                "obs_rms": [self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count],
                "ret_rms": [self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count]}

    def load_state_dict(self, saved):
        self.returns.copy_(saved["returns"])
        mean, var, count = saved["obs_rms"]
        self.obs_rms.mean.copy_(mean); self.obs_rms.var.copy_(var); self.obs_rms.count = count
        mean, var, count = saved["ret_rms"]
        self.ret_rms.mean.copy_(mean); self.ret_rms.var.copy_(var); self.ret_rms.count = count


class EpsilonPolicy(nn.Module):
    """54 inputs, Tanh 256-128 trunks for both nets, state-independent std."""

    def __init__(self, obs_dim=54, action_dim=25, log_std_init=-3.0):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(),
                                nn.Linear(256, 128), nn.Tanh())
        self.vf = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(),
                                nn.Linear(256, 128), nn.Tanh())
        self.action_net = nn.Linear(128, action_dim)
        self.value_net = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        for net in (self.pi, self.vf):
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, math.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.action_net.weight, 0.01)
        nn.init.zeros_(self.action_net.bias)
        nn.init.orthogonal_(self.value_net.weight, 1.0)
        nn.init.zeros_(self.value_net.bias)

    def distribution(self, obs):
        mean = self.action_net(self.pi(obs))
        return mean, self.log_std.exp().expand_as(mean)

    def value(self, obs):
        return self.value_net(self.vf(obs)).squeeze(-1)

    def evaluate(self, obs, actions):
        mean, std = self.distribution(obs)
        noise = (actions - mean) / std
        log_prob = (-0.5 * noise.square() - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        entropy = (std.log() + 0.5 + 0.5 * math.log(2 * math.pi)).sum(-1)
        return self.value(obs), log_prob, entropy


class Rollout:
    """Fixed device buffers, n_steps x n_envs, filled in order."""

    def __init__(self, n_steps, n_envs, obs_dim, action_dim, device):
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=device, dtype=dtype)
        self.obs = zeros(n_steps, n_envs, obs_dim)
        self.actions = zeros(n_steps, n_envs, action_dim)
        self.rewards = zeros(n_steps, n_envs)      # already normalized
        self.values = zeros(n_steps, n_envs)
        self.log_probs = zeros(n_steps, n_envs)
        self.episode_starts = zeros(n_steps, n_envs, dtype=torch.bool)
        self.advantages = zeros(n_steps, n_envs)
        self.returns = zeros(n_steps, n_envs)
        self.pos = 0

    def add(self, obs, actions, reward, value, log_prob, episode_start):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, actions, reward
        self.values[i], self.log_probs[i] = value, log_prob
        self.episode_starts[i] = episode_start
        self.pos = i + 1

    @torch.no_grad()
    def finish(self, policy, next_obs, next_done, gamma=0.995, gae_lambda=0.97):
        """GAE mirroring SB3's RolloutBuffer, on stored normalized rewards."""
        next_value = policy.value(next_obs)
        next_non_terminal = (~next_done).float()
        last_gae = torch.zeros_like(next_value)
        for step in range(len(self.rewards) - 1, -1, -1):
            if step == len(self.rewards) - 1:
                non_terminal = next_non_terminal
            else:
                non_terminal = (~self.episode_starts[step + 1]).float()
            delta = self.rewards[step] + gamma * (next_value if step == len(self.rewards) - 1
                                                 else self.values[step + 1]) * non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[step] = last_gae
            next_value = self.values[step]
        self.returns = self.advantages + self.values
        self.pos = 0


class PPO:
    """Clipped-surrogate update with Epsilon's hyperparameters.

    On CUDA the per-minibatch iteration can be captured into a CUDA graph
    (use_graph=True): the Adam optimizer runs in capturable mode and each
    replay reads its minibatch indices from a device buffer, so an update
    replays ~1,300 ready-made iterations instead of launching ~20,000 tiny
    kernels (the measured launch-latency bottleneck). The eager path stays
    the default and the CPU fallback; a captured update must reproduce the
    eager losses and KL within float tolerance before use."""

    def __init__(self, policy, learning_rate=1e-5, n_epochs=10, batch_size=1024,
                 clip_range=0.1, clip_range_vf=0.2, ent_coef=0.05, vf_coef=1.0,
                 max_grad_norm=0.1, target_kl=0.01, std_clamp=(0.05, 0.30),
                 device="cpu", use_graph=False):
        self.policy = policy
        self.cuda = str(device) != "cpu" and torch.cuda.is_available()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate,
                                          capturable=bool(use_graph and self.cuda))
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.clip_range, self.clip_range_vf = clip_range, clip_range_vf
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.std_clamp = tuple(math.log(bound) for bound in std_clamp)
        self.device = device
        self.use_graph = bool(use_graph and self.cuda)
        self.graph, self.graph_count = None, 0
        self.static_full, self.static_idx = None, None
        self.stats_acc, self.kl_acc = None, None
        self.last = {}

    def _clamp_std(self):
        with torch.no_grad():
            self.policy.log_std.clamp_(*self.std_clamp)

    def _iteration(self, full, idx):
        """One PPO minibatch iteration on static buffers; graph-capturable."""
        policy = self.policy
        obs, actions, old_values, old_log_probs, advantages, returns = full
        values, log_prob, entropy = policy.evaluate(obs[idx], actions[idx])
        adv = advantages[idx]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        ratio = (log_prob - old_log_probs[idx]).exp()
        policy_loss = -torch.min(adv * ratio, adv * ratio.clamp(1 - self.clip_range,
                                                                1 + self.clip_range)).mean()
        clipped = old_values[idx] + (values - old_values[idx]).clamp(-self.clip_range_vf,
                                                                     self.clip_range_vf)
        value_loss = F.mse_loss(returns[idx], clipped)
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        with torch.no_grad():
            log_ratio = log_prob - old_log_probs[idx]
            kl = ((log_ratio.exp() - 1) - log_ratio).mean()
            kl_over = (kl > 1.5 * self.target_kl) if self.target_kl is not None \
                else torch.zeros((), dtype=torch.bool, device=kl.device)
            acc = torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(),
                               entropy_loss.detach(), kl.detach(),
                               (ratio - 1).abs().gt(self.clip_range).float().mean().detach()))
        self.stats_acc += acc
        self.kl_acc |= kl_over
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            policy.log_std.clamp_(*self.std_clamp)
        return loss

    def _flat(self, rollout):
        flat = lambda *t: [x.reshape(-1, *x.shape[2:]) for x in t]
        return flat(rollout.obs, rollout.actions, rollout.values, rollout.log_probs,
                    rollout.advantages, rollout.returns)

    def _ensure_graph(self, count):
        """Allocate static full-batch buffers and capture one iteration.

        The warmup iterations would pollute the weights and Adam state, so
        both are snapshotted and restored around the capture. All buffers the
        captured iteration writes (stats, KL flag) are allocated here once
        and only zeroed in place afterwards, so their addresses stay bound to
        the graph."""
        policy = self.policy
        obs_dim, act_dim = policy.pi[0].in_features, policy.action_net.out_features
        dev = torch.device(self.device)
        self.static_full = tuple(torch.zeros(count, *shape, device=dev) for shape in
                                 ((obs_dim,), (act_dim,), (), (), (), ()))
        self.static_idx = torch.zeros(self.batch_size, dtype=torch.long, device=dev)
        self.stats_acc = torch.zeros(6, device=dev)
        self.kl_acc = torch.zeros((), device=dev, dtype=torch.bool)
        weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        saved_optimizer = {}
        for g, group in self.optimizer.state_dict()["state"].items():
            saved_optimizer[g] = {k: (v.detach().clone() if torch.is_tensor(v)
                                      else ([t.detach().clone() if torch.is_tensor(t) else t
                                             for t in v] if isinstance(v, list) else v)
                                      if v is not None else v)
                                  for k, v in group.items()}
        # Warm up the exact code path (allocator, gradients, optimizer
        # state) first on scratch data, then capture.
        self.static_idx.copy_(torch.arange(self.batch_size, device=dev))
        for _ in range(3):
            self._iteration(self.static_full, self.static_idx)
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._iteration(self.static_full, self.static_idx)
        torch.cuda.current_stream(dev).wait_stream(stream)
        # Undo the warmup's parameter and moment drift. State the warmup
        # created from scratch (fresh optimizer) is zeroed in place rather
        # than deleted: the captured graph writes to these exact addresses.
        with torch.no_grad():
            for name, value in policy.state_dict().items():
                value.copy_(weights[name])
            for g, group in self.optimizer.state_dict()["state"].items():
                saved = saved_optimizer.get(g)
                for k, v in group.items():
                    if torch.is_tensor(v):
                        v.copy_(saved[k] if saved is not None and torch.is_tensor(saved[k])
                                else torch.zeros_like(v))
                    elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                        source = saved[k] if saved is not None else None
                        for t, s in zip(v, source if source else [None] * len(v)):
                            t.copy_(s if torch.is_tensor(s) else torch.zeros_like(t))
        self.graph = graph
        self.graph_count = count

    def update(self, rollout, defer=False):
        """Run the PPO update. With defer=True the result is left on device
        (see read_stats) so the caller can overlap the next rollout without
        a host stall -- the update is enqueued and returns immediately."""
        policy = self.policy
        flat_data = self._flat(rollout)
        count = len(flat_data[0])
        use_graph = self.use_graph and count % self.batch_size == 0
        if use_graph and (self.graph is None or self.graph_count != count):
            self._ensure_graph(count)
        dev = torch.device(self.device)
        if not use_graph and self.stats_acc is None:
            self.stats_acc = torch.zeros(6, device=dev)
            self.kl_acc = torch.zeros((), device=dev, dtype=torch.bool)
        self.stats_acc.zero_()
        self.kl_acc.zero_()
        acc_count = 0
        stop = False
        if use_graph:
            for full, static in zip(flat_data, self.static_full):
                static.copy_(full)
        for _ in range(self.n_epochs):
            if stop:
                break
            order = torch.randperm(count, device=dev)
            for start in range(0, count, self.batch_size):
                idx = order[start:start + self.batch_size]
                if use_graph:
                    self.static_idx.copy_(idx)
                    self.graph.replay()
                else:
                    self._iteration(flat_data, idx)
                acc_count += 1
            if self.target_kl is not None and bool(self.kl_acc):
                stop = True
        self._raw = (self.stats_acc.clone() / max(1, acc_count),
                     rollout.values.flatten().clone(),
                     rollout.returns.flatten().clone(),
                     self.policy.log_std.detach().clone())
        if defer:
            return None
        return self.read_stats()

    def read_stats(self):
        """Host values for the most recent update (one readback)."""
        names = ("loss", "policy_gradient_loss", "value_loss", "entropy_loss",
                 "approx_kl", "clip_fraction")
        means, values_flat, returns_flat, log_std = self._raw
        self.last = {n: float(means[i]) for i, n in enumerate(names)}
        self.last["explained_variance"] = float(1 - (returns_flat - values_flat).var() / returns_flat.var())
        self.last["std"] = float(log_std.exp().mean())
        return self.last

    def state(self):
        return dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict())
