"""Go2 Warp plant and ULA Gamma5 policy interface (flat ground)."""
import math

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .controller import BatchController, ROOT
from .kicks import KickSchedule


def tensor(a):
    # Warp CPU -> Torch may use NumPy; on CUDA this shares device memory.
    return wp.to_torch(a)


def reward_terms(obs, vw, action, previous, has_action, accel, qw, y):
    penalties = torch.stack((
        3 * (vw[:, 0] - obs[:, 0]).abs(), 3 * vw[:, 2].abs(),
        20 * (obs[:, 52] - obs[:, 53]).abs(), 50 * (1 - qw.clamp(-1, 1)),
        5 * y.abs() + 3 * vw[:, 1].abs(), accel.square().sum(-1), obs[:, 4:7].square().sum(-1),
        3 * (action - previous).square().sum(-1) * has_action,
        (action[:, 24] + 1).clamp_min(0),
        action[:, :3].square().sum(-1) + action[:, 3:15].reshape(-1, 4, 3)[:, :, :2].square().sum((1, 2))
        + action[:, 15:24].square().sum(-1),
    ), dim=1)
    return penalties


class Go2Env:
    control_dt = 0.002
    policy_ticks = 5
    action_dim = 25
    actor_dim = 54
    obs_dim = 61

    def __init__(self, n=1, device='cpu', *, library=None, seed=1, max_steps=1000,
                 training=True, graph=False, kick='auto', curriculum=None, backend='vendor'):
        # kick="velocity" (default when training=True) turns on the ICRA
        # additive base-velocity kick curriculum in place of the old Gamma5
        # xfrc force-pulse block, which is deleted; kick=None keeps training
        # disturbances off.  `curriculum` is (total_timesteps, boundaries),
        # the interaction-space ramp handed to KickSchedule.configure_training.
        if kick == 'auto':
            kick = 'velocity' if training else None
        if kick not in (None, 'velocity'):
            raise ValueError(f'unknown kick mode: {kick!r}')
        if curriculum is not None and kick != 'velocity':
            raise ValueError('curriculum requires kick="velocity"')
        wp.init()
        self.controller = BatchController(n, device, library, backend)
        self.device = self.controller.device
        self.n, self.max_steps, self.training = n, max_steps, training
        self.use_graph = graph and self.device.type == 'cuda'
        self.physics_graph = None
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.wp_device = wp.get_device(str(self.device))
        mj = mujoco.MjModel.from_xml_path(str(ROOT / 'models/go2/scene_gpu.xml'))
        def name(kind, value):
            i = mujoco.mj_name2id(mj, kind, value)
            if i < 0:
                raise ValueError(f'missing Go2 element: {value}')
            return i

        if (mj.nq, mj.nv, mj.nu) != (19, 18, 12) or mj.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError('expected free-base Go2 and Euler integration')
        joint_ids = [name(mujoco.mjtObj.mjOBJ_JOINT, f'go2/{j}_{l}_joint')
                     for l in range(1, 5) for j in ('abad', 'hip', 'knee')]
        if list(mj.jnt_qposadr[joint_ids]) != list(range(7, 19)) or list(mj.actuator_trnid[:, 0]) != joint_ids:
            raise ValueError('Go2 joint and motor order must be FR, FL, HR, HL')
        self.substeps = round(self.control_dt / mj.opt.timestep)
        if abs(self.substeps * mj.opt.timestep - self.control_dt) > 1e-10:
            raise ValueError('physics step must divide the 2 ms controller step')
        self.base = name(mujoco.mjtObj.mjOBJ_BODY, 'go2/base_link_inertia')
        self.feet = self.as_tensor([name(mujoco.mjtObj.mjOBJ_GEOM, f'go2/foot_{l}') for l in range(1, 5)], torch.int32)
        self.geom_body = self.as_tensor(mj.geom_bodyid, torch.int32)
        self.acc_adr = int(mj.sensor_adr[name(mujoco.mjtObj.mjOBJ_SENSOR, 'acc')])
        with wp.ScopedDevice(self.wp_device):
            self.model = mjw.put_model(mj)
            self.data = mjw.make_data(mj, nworld=n, nconmax=32, njmax=160)
            self.ids = wp.array(torch.arange(self.data.contact.geom.shape[0], dtype=torch.int32).numpy(), dtype=int)
            self.contact_force = wp.zeros(self.ids.shape, dtype=wp.spatial_vector)
        d = self.data
        self.q, self.v = tensor(d.qpos), tensor(d.qvel)
        self.length, self.speed, self.ctrl = map(tensor, (d.actuator_length, d.actuator_velocity, d.ctrl))
        self.force = tensor(d.xfrc_applied)[:, self.base, :3]
        self.contact_world, self.contact_geom, self.contact_n = map(tensor, (d.contact.worldid, d.contact.geom, d.nacon))
        self.contact_force_t = tensor(self.contact_force)
        self.ids_t = tensor(self.ids)
        self.frame = tensor(d.contact.frame)
        self.time = tensor(d.time)
        self.sensor = tensor(d.sensordata)
        self.actor, self.vw = self.zeros(n, 54), self.zeros(n, 4)
        self.status = self.zeros(n, dtype=torch.int32)
        self.failed = self.zeros(n, dtype=torch.bool)
        self.command = self.zeros(n, 3)
        self.steps = self.zeros(n, dtype=torch.int64)
        self.previous, self.zero = self.zeros(n, 25), self.zeros(n, 25)
        self.noise_range = self.as_tensor([0.05]*3 + [0.1]*3 + [0.2]*3) if training else self.zeros(9)
        self.noise = self.zeros(n, 9)
        self.truth_qw, self.acc, self.base_y = self.zeros(n), self.zeros(n, 3), self.zeros(n)
        self.weights = self.as_tensor([.25, .1, .35, .15, .15, .025, .025, .1, .05, .05])
        q = [0, 0, .3, 1, 0, 0, 0] + [a * math.pi / 180 for l in range(4)
                                              for a in (10 if l % 2 else -10, -50, 90)]
        self.initial = self.as_tensor(q)
        self.kicks = None
        if kick == 'velocity':
            self.kicks = KickSchedule(self)
            if curriculum is not None:
                total, boundaries = curriculum
                self.kicks.configure_training(self, total_timesteps=total,
                                              boundaries=boundaries, seed=seed)
            else:
                self.kicks.configure_training(self, seed=seed)

    def zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(shape, device=self.device, dtype=dtype)

    def as_tensor(self, values, dtype=torch.float32):
        return torch.as_tensor(values, device=self.device, dtype=dtype)

    def uniform(self, *shape):
        return torch.rand(shape, generator=self.rng, device=self.device)

    def scoped(self):
        if self.device.type == 'cpu':
            return wp.ScopedDevice(self.wp_device)
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _physics(self):
        for _ in range(self.substeps):
            mjw.step1(self.model, self.data)
            self.controller.act(self.length, self.speed, self.ctrl)
            mjw.step2(self.model, self.data)

    def capture_physics(self):
        if self.use_graph and self.physics_graph is None:
            # Warm-up happened during reset; capture only the fixed-buffer
            # physics and actuator loop, not Python reset branches or RNG.
            caller = torch.cuda.current_stream(self.device)
            stream = caller
            if caller.cuda_stream == 0:  # the legacy default stream cannot be captured
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                warp_stream = wp.stream_from_torch(stream)
                with wp.ScopedStream(warp_stream), wp.ScopedCapture(stream=warp_stream) as capture:
                    self._physics()
            self.physics_graph = capture.graph

    def _tick(self, action, active=None, prime=False):
        if self.training:
            self.noise.copy_((2 * self.uniform(self.n, 9) - 1) * self.noise_range)
        self.controller.tick(self.q, self.v, action, self.command, self.noise, active,
                             self.actor, self.vw, self.status, prime)
        hit = self.status != 0
        if active is None:
            self.failed.logical_or_(hit)
        else:
            self.failed.logical_or_(hit & active)
        # Reward samples the same (pre-physics) estimator state as the actor.
        half = self.actor[:, 7:10] * 0.5
        sr, sp, sy = half.sin().unbind(1)
        cr, cp, cy = half.cos().unbind(1)
        qw = cr*cp*cy + sr*sp*sy if self.training else self.q[:, 3]
        self.truth_qw.copy_(qw if active is None else torch.where(active, qw, self.truth_qw))
        self.base_y.copy_(self.q[:, 1] if active is None else torch.where(active, self.q[:, 1], self.base_y))
        # The reward only needs squared acceleration; use the IMU rotation
        # to express world gravity in the sensor's body frame.
        w, x, y, z = self.q[:, 3:7].unbind(1)
        g = torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=1)
        accel = self.sensor[:, self.acc_adr:self.acc_adr+3] - 9.81*g
        self.acc.copy_(accel if active is None else torch.where(active[:, None], accel, self.acc))

        # ICRA disturbance protocol: additive base-velocity kicks.  Replaces
        # the old Gamma5 xfrc force-pulse block; xfrc stays zero (native
        # "velocity" kicks deliver entirely through qvel[0:3]).
        if self.kicks is not None:
            self.kicks.tick(active)

        with self.scoped():
            if self.physics_graph is None:
                self._physics()
            else:
                wp.capture_launch(self.physics_graph)

    def _contacts(self):
        with self.scoped():
            mjw.contact_force(self.model, self.data, self.ids, False, self.contact_force)
        valid = self.ids_t < self.contact_n[0]
        world = torch.where(valid, self.contact_world, 0).long()
        geom = self.contact_geom.clamp(0, self.geom_body.numel()-1)
        body = self.geom_body[geom.long()]
        sign = (body[:, 1] == self.base).float() - (body[:, 0] == self.base).float()
        f = (self.contact_force_t[:, :3, None] * self.frame).sum(dim=1)
        f = torch.where((valid & (sign != 0))[:, None], f * sign[:, None], 0)
        trunk = self.zeros(self.n, 3).index_add_(0, world, f).norm(dim=1)
        foot = (self.contact_geom[:, :, None] == self.feet).any(dim=1) & valid[:, None]
        contacts = self.zeros(self.n, 4).index_add_(0, world, foot.float()).clamp_(0, 1)
        return trunk, contacts

    def _reset(self, mask, command=None, full=True):
        # Full resets (the initial reset and every evaluation reset) prime the
        # controller with six zero-action ticks, exactly like the recorded
        # ULA trace, and roll the untouched worlds' integration state back.
        # Incremental training resets take the lean path instead: no priming
        # physics, no save/restore. This is a performance tradeoff: the full
        # path spends a whole grid's physics on the handful of worlds that
        # actually fell (measured ~18% of training-loop time). Native runtime
        # reset internals have not been traced to prove equivalent priming. The lean
        # path still refreshes the reset worlds' observations with one gated
        # controller tick so the stored post-reset observation is real.
        d = self.data
        if full:
            saved = [self.q, self.v, self.time, tensor(d.qacc_warmstart), self.force, self.sensor]
            old = [a.clone() for a in saved]
        with self.scoped():
            mjw.reset_data(self.model, self.data, wp.from_torch(mask, dtype=wp.bool))
        q = self.initial.expand(self.n, -1).clone()
        cmd = self.zeros(self.n, 3) if command is None else command
        if self.training:
            q[:, 2] += (2 * self.uniform(self.n) - 1) * .03
            cmd.copy_(self.uniform(self.n, 3))
            cmd[:, 1:] = (cmd[:, 1:] - .5) * .4
        self.q.copy_(torch.where(mask[:, None], q, self.q))
        self.command.copy_(torch.where(mask[:, None], cmd, self.command))
        self.steps.masked_fill_(mask, 0)
        self.previous.masked_fill_(mask[:, None], 0)
        self.failed.masked_fill_(mask, False)
        if self.kicks is not None:
            self.kicks.reset(mask)
        self.controller.reset(self.q, self.v, mask)
        if full:
            for _ in range(6):
                self._tick(self.zero, mask, prime=True)
        else:
            # One gated tick (no physics) so the reset worlds' controller
            # output and gait phase are post-reset, matching what the next
            # real step will consume.
            self.controller.tick(self.q, self.v, self.zero, self.command, self.noise,
                                 mask, self.actor, self.vw, self.status, True)
        _, feet = self._contacts()
        fresh = torch.cat((self.actor, self.force, feet), dim=1)
        if full:
            for a, original in zip(saved, old):
                select = mask.reshape(self.n, *([1]*(a.ndim-1)))
                a.copy_(torch.where(select, a, original))
        return fresh

    @torch.no_grad()
    def reset(self, *, command=None):
        """Reset all worlds; optionally set their fixed (vx, vy, yaw) evaluation commands."""
        if command is not None:
            if self.training:
                raise ValueError("fixed reset commands are for evaluation only")
            command = self.as_tensor(command).broadcast_to((self.n, 3))
        return torch.nan_to_num(self._reset(torch.ones(self.n, device=self.device, dtype=torch.bool), command))

    @torch.no_grad()
    def step(self, actions, *, auto_reset=True):
        valid_action = torch.isfinite(actions).all(dim=1)
        actions = torch.nan_to_num(actions).clamp(-1, 1).contiguous()
        if self.kicks is not None:
            self.kicks.count_step()
        for _ in range(self.policy_ticks):
            self._tick(actions)
        trunk, feet = self._contacts()
        obs = torch.cat((self.actor, self.force, feet), dim=1)
        failed = self.failed.clone()
        self.failed.zero_()
        terminated = (failed | ~valid_action | (trunk > 1) | (obs[:, 7].abs() > .8) |
                      (obs[:, 8].abs() > 1.) | ~torch.isfinite(obs).all(dim=1) |
                      ~torch.isfinite(self.q).all(dim=1) | ~torch.isfinite(self.v).all(dim=1))
        p = reward_terms(obs, self.vw, actions, self.previous, self.steps > 0, self.acc, self.truth_qw, self.base_y)
        reward = ((-p).exp() * self.weights).sum(dim=1)
        terminated |= ~torch.isfinite(reward)
        reward = torch.where(terminated, -10., reward)
        self.previous.copy_(actions)
        self.steps += 1
        truncated = (self.steps >= self.max_steps) & ~terminated
        done = terminated | truncated
        final = torch.nan_to_num(obs)
        # One host scalar to skip an expensive reset pass when nobody ended.
        obs = torch.where(done[:, None], self._reset(done), final) if auto_reset and bool(done.any()) else final
        return torch.nan_to_num(obs), reward, terminated, truncated, final, failed

    def close(self):
        self.physics_graph = None
        self.controller.close()
