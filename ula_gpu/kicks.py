"""Device-resident additive base-velocity kick schedules (ICRA disturbance protocol).

Native ground truth: MdlSimDriver.cc `_updateExternalForce` with
external_force_type = "velocity" -- ONE instantaneous kick per ON window,
delivered as an add-semantics delta to the base's world-frame linear velocity
(qvel[0:3], all three components) inside the control tick, before physics
integration; xfrc_applied stays zero.  Frame "body_latched_at_onset": the
configured body-frame vector is rotated into the world by the base
orientation at the kick tick.  Onset: after `onset_cycle` (3) whole gait
cycles, when the scheduler's full-cycle phase crosses a per-episode uniformly
drawn target phase.
"""

import torch

# Eval fixed cell: one kick, then an off window longer than any episode.
FIXED_ON_DURATION = 0.2
FIXED_OFF_DURATION = 100.0
# Training curriculum windows (native random mode): 0.2 s on / 1.4 s off.
TRAIN_ON_DURATION = 0.2
TRAIN_OFF_DURATION = 1.4
TRAIN_NOFORCE_PROBABILITY = 0.2
TRAIN_CEILINGS = (1.0, 2.0, 3.0)
# Documented interaction-space ramp: 1.0 m/s to 8.1M interactions, 2.0 to
# 24.3M, 3.0 over the rest of an 85M-interaction run.
DEFAULT_TOTAL_INTERACTIONS = 85_000_000
DEFAULT_RAMP_BOUNDARIES = (8_100_000, 24_300_000)
ONSET_CYCLE = 3

# +X forward, +Y left, +Z up in the base frame (forces.py DIRS).
DIRS = {
    'fwd': (1.0, 0.0, 0.0), 'back': (-1.0, 0.0, 0.0),
    'left': (0.0, 1.0, 0.0), 'right': (0.0, -1.0, 0.0),
    'up': (0.0, 0.0, 1.0), 'down': (0.0, 0.0, -1.0),
}


def rotation_matrix(q):
    """Batched body->world rotation from wxyz quaternions, MuJoCo convention."""
    w, x, y, z = q.unbind(1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=1).reshape(-1, 3, 3)


class KickSchedule:
    """Per-world velocity-kick state, entirely as torch tensors on the device.

    Fixed mode (``configure``) reproduces the evaluation cell: every world
    gets the same body-frame kick, one delivery after the latched onset.
    Training mode (``configure_training``) reproduces the random-cardinal
    curriculum: one Bernoulli(0.8) kick per 1.6 s window, direction uniform
    over the six signed cardinals, magnitude = fraction * ramped ceiling.
    """

    def __init__(self, env, onset_cycle=ONSET_CYCLE):
        self.env, self.n, self.device = env, env.n, env.device
        self.mode = None
        self.onset_cycle = onset_cycle
        self.on_duration, self.off_duration = FIXED_ON_DURATION, FIXED_OFF_DURATION
        self.period = self.on_duration + self.off_duration
        self.body = env.zeros(env.n, 3)            # unit direction, body frame
        self.mag = env.zeros(env.n)                # kick magnitude, m/s
        self.target = env.zeros(env.n)             # onset phase target, U(0,1)
        self.prev_phase = env.zeros(env.n)
        self.cycles = env.zeros(env.n, dtype=torch.int32)
        self.onset = torch.full((env.n,), -1.0, device=env.device)
        self.last_window = torch.full((env.n,), -1, device=env.device, dtype=torch.int64)
        self.count = env.zeros(env.n, dtype=torch.int32)     # kicks delivered
        self.total_delivered = env.zeros(dtype=torch.int64)  # lifetime, not reset
        self.delivered = env.zeros(env.n, 3)      # world-frame delta, last kick
        self.interactions = env.zeros(env.n, dtype=torch.int64)
        self.draw = env.zeros(env.n, 4)           # curriculum uniforms, per policy step
        self.rng = torch.Generator(device=env.device)
        self.streams = []                          # fixed mode: per-world phase streams
        self.fractions = torch.zeros(0, device=env.device)
        self.per_env = 1.0
        self.ceilings = torch.tensor(TRAIN_CEILINGS, device=env.device)

    # -- configuration ---------------------------------------------------

    def configure(self, env, dirs, mags, seed, onset_cycle=ONSET_CYCLE, phase_seeds=None):
        """Fixed-cell evaluation: one pre-specified kick per world.

        `dirs`/`mags` are a single direction key / float (broadcast to every
        world) or exactly one per world.  The onset target phase is drawn per
        world at every reset from a dedicated torch.Generator seeded
        `seed + world_id`, mirroring the native per-env stream
        (``external_force_seed + env_id``).  Attach before the episode reset
        so the draw happens there, as the native driver does.
        """
        self.mode = 'fixed'
        self.onset_cycle = onset_cycle
        self.on_duration, self.off_duration = FIXED_ON_DURATION, FIXED_OFF_DURATION
        self.period = self.on_duration + self.off_duration
        one = not isinstance(dirs, (list, tuple))
        keys = [dirs] * env.n if one else list(dirs)
        body = torch.as_tensor([DIRS[k] for k in keys], dtype=torch.float32)
        mags = torch.as_tensor([mags] * env.n if one else mags, dtype=torch.float32).reshape(-1)
        if body.shape != (env.n, 3) or mags.shape != (env.n,) or bool((mags < 0).any()):
            raise ValueError('dirs/mags must be a scalar or exactly one (nonnegative) per world')
        self.body.copy_(body.to(env.device))
        self.mag.copy_(mags.to(env.device))
        seeds = [int(seed) + i for i in range(env.n)] if phase_seeds is None else list(phase_seeds)
        if len(seeds) != env.n:
            raise ValueError("phase_seeds must contain exactly one seed per world")
        self.streams = [torch.Generator(device=env.device).manual_seed(int(s)) for s in seeds]
        env.kicks = self

    def configure_training(self, env, total_timesteps=DEFAULT_TOTAL_INTERACTIONS,
                           boundaries=DEFAULT_RAMP_BOUNDARIES, seed=1,
                           onset_cycle=ONSET_CYCLE):
        """Random-cardinal velocity-kick curriculum (native training recipe).

        Every 1.6 s window after the latched onset, draw with probability 0.8
        a kick: direction uniform over the six signed cardinals (0..5 ->
        +x,-x,+y,-y,+z,-z), magnitude = fraction * ceiling with fraction drawn
        75% from U(0.5, 1.0) and 25% from U(0, 0.5).

        Ramp clock: INTERACTION space, as the training configs document it --
        level boundaries at `boundaries` of `total_timesteps` total training
        interactions, each world's share being total / n worlds (native:
        total_timesteps / num_envs).  The native driver instead counts its own
        internal per-callback clock (control ticks since process start, never
        reset between episodes), which may differ; the documented
        interaction-space schedule is the target.
        """
        boundaries = tuple(boundaries)
        if len(boundaries) != len(TRAIN_CEILINGS) - 1 or any(b <= 0 for b in boundaries) \
                or sorted(boundaries) != list(boundaries):
            raise ValueError(f'boundaries must be {len(TRAIN_CEILINGS) - 1} ascending positives')
        self.mode = 'train'
        self.onset_cycle = onset_cycle
        self.on_duration, self.off_duration = TRAIN_ON_DURATION, TRAIN_OFF_DURATION
        self.period = self.on_duration + self.off_duration
        self.per_env = float(total_timesteps) / env.n
        self.fractions = torch.tensor([b / float(total_timesteps) for b in boundaries],
                                      device=env.device)
        self.rng.manual_seed(seed)
        env.kicks = self
        self.reset(torch.ones(env.n, dtype=torch.bool, device=env.device))

    # -- per-episode and per-step hooks ----------------------------------

    def reset(self, mask):
        """Episode start for the masked worlds: clear kick state, redraw phase.

        The stale-window-index clear mirrors the native reset guard: without
        it the next episode's first ON window can match the previous
        episode's index and swallow its kick.
        """
        self.onset.masked_fill_(mask, -1)
        self.cycles.masked_fill_(mask, 0)
        self.prev_phase.masked_fill_(mask, 0)
        self.last_window.masked_fill_(mask, -1)
        self.count.masked_fill_(mask, 0)
        self.delivered.masked_fill_(mask[:, None], 0)
        if self.mode == 'fixed':
            # One draw per world from its own stream, so a world's onset is
            # independent of how many worlds share the batch.
            drawn = torch.stack([torch.rand((), generator=g, device=self.device)
                                 for g in self.streams])
            self.target.copy_(torch.where(mask, drawn, self.target))
        elif self.mode == 'train':
            self.target.copy_(torch.where(mask, torch.rand((self.n,), generator=self.rng,
                                                          device=self.device), self.target))

    def count_step(self):
        """One policy step elapsed: advance the ramp clock, refresh the draws.

        The per-world interaction counter is the documented ramp clock
        (native notes its own per-callback clock may differ; see
        configure_training).  Each world starts at most one ON window per
        policy step, so four fresh uniforms per world per step serve every
        window draw without a host sync.
        """
        self.interactions += 1
        if self.mode == 'train':
            self.draw = torch.rand((self.n, 4), generator=self.rng, device=self.device)

    def delivered_audit(self):
        """Per-world |world delta| of the most recent delivered kick (device)."""
        return self.delivered.norm(dim=1)

    # -- per-tick update ---------------------------------------------------

    def tick(self, active=None):
        """One control tick: latch the onset, then deliver at most one kick.

        No host syncs: wrap detection, the curriculum draw and the kick all
        run as batched torch ops on the env's device.
        """
        env = self.env
        phase = env.vw[:, 3]
        wrapped = phase < self.prev_phase
        crossed = torch.where(wrapped, (self.target > self.prev_phase) | (self.target <= phase),
                              (self.target > self.prev_phase) & (self.target <= phase))
        self.cycles.add_(wrapped if active is None else wrapped & active)
        begin = (self.onset < 0) & (self.cycles >= self.onset_cycle) & crossed
        if active is not None:
            begin &= active
        self.prev_phase.copy_(phase if active is None else torch.where(active, phase, self.prev_phase))
        self.onset.copy_(torch.where(begin, env.time, self.onset))
        elapsed = (env.time - self.onset).clamp_min(0)
        window = (elapsed / self.period).long()
        on = (self.onset >= 0) & (elapsed.remainder(self.period) < self.on_duration)
        start = on & (window != self.last_window)
        if active is not None:
            start &= active
        if self.mode == 'train':
            # A window draw consumes the window whether or not it kicks.
            self._draw(start)
            consume = start
        else:
            consume = start & (self.mag > 0)
        fire = consume & (self.mag > 0)
        delta = torch.bmm(rotation_matrix(env.q[:, 3:7]),
                          (self.body * self.mag[:, None]).unsqueeze(2)).squeeze(2)
        env.v[:, :3] += delta * fire[:, None]
        self.delivered.copy_(torch.where(fire[:, None], delta, self.delivered))
        self.count.add_(fire.to(torch.int32))
        self.total_delivered.add_(fire.sum())
        self.last_window.copy_(torch.where(consume, window, self.last_window))

    def _draw(self, start):
        """Curriculum draw for worlds entering an ON window (native random mode).

        `self.draw` is refreshed once per policy step by count_step(), so
        every window start consumes four fresh uniforms; the native driver
        instead draws from one per-env std::mt19937 stream in draw order.
        The distributions match (i.i.d. uniforms either way); only stream
        identity differs.
        """
        u_fire, u_dir, u_choice, u_val = self.draw.unbind(1)
        kick = (u_fire >= TRAIN_NOFORCE_PROBABILITY) & start
        direction = (u_dir * 6).long()
        axis = torch.div(direction, 2, rounding_mode='floor').clamp(0, 2)
        sign = torch.where(direction % 2 == 0, 1.0, -1.0)
        body = torch.zeros_like(self.body)
        body.scatter_(1, axis[:, None], sign[:, None])
        fraction = torch.where(u_choice < 0.75, 0.5 + 0.5 * u_val, 0.5 * u_val)
        self.body.copy_(torch.where(kick[:, None], body, self.body))
        self.mag.copy_(torch.where(kick, fraction * self._ceiling(), torch.zeros_like(self.mag)))

    def _ceiling(self):
        """Ramp ceiling per world from its interaction share of training."""
        progress = (self.interactions.to(torch.float32) / self.per_env).clamp(0., 1.)
        level = (progress.unsqueeze(1) >= self.fractions).sum(1)
        return self.ceilings[level]
