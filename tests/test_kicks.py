#!/usr/bin/env python3
"""Unit tests for ula_gpu/kicks.py: kick rotation math and onset latch logic.

Runs on CPU with no pytest dependency (plain asserts, exit 0 on pass):

    python tests/test_kicks.py

Covers the two pieces the kick protocol stands on:
  1. rotation_matrix (batched quaternion -> body->world) against a manual
     matrix construction for random orientations;
  2. the per-tick onset wrap-crossing latch against a scalar port of
     MdlSimDriver.cc _updateExternalForceOnsetLatch, plus the single-kick
     window guard of the fixed cell and the curriculum draw/ceiling math.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.kicks import (DIRS, KickSchedule, TRAIN_CEILINGS, TRAIN_OFF_DURATION,
                            TRAIN_ON_DURATION, rotation_matrix)

GAIT_PERIOD = 0.4      # s, scheduler full-cycle period used by the scripts
DT = 0.002             # s, one control tick


class FakeEnv:
    """Just the tensor surface KickSchedule touches, with identity attitude."""

    def __init__(self, n=4):
        self.n, self.device = n, torch.device('cpu')
        self.vw = torch.zeros(n, 4)
        self.q = torch.zeros(n, 19)
        self.q[:, 3] = 1.0
        self.v = torch.zeros(n, 18)
        self.time = torch.zeros(n)

    def zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(shape, device=self.device, dtype=dtype)


def phases_and_times(ticks, period=GAIT_PERIOD, dt=DT):
    t = torch.arange(ticks, dtype=torch.float32) * dt
    return (t / period).remainder(1.0), t


def native_scalar_latch(phases, times, target, onset_cycle):
    """Scalar port of MdlSimDriver::_updateExternalForceOnsetLatch.

    Runs until the onset latches; returns the latched time (or -1).
    """
    cycles, prev, had_prev, onset = 0, 0.0, False, -1.0
    for phase, t in zip(phases, times):
        cur, had = prev, had_prev
        prev, had_prev = float(phase), True
        if not had:
            continue
        wrapped = phase < cur
        if wrapped:
            cycles += 1
        if cycles < onset_cycle:
            continue
        crossed = ((target > cur) or (target <= phase)) if wrapped \
            else ((target > cur) and (target <= phase))
        if crossed:
            onset = float(t)
            break
    return onset


def run_all(schedule, env, phases, times):
    """Feed the whole scripted sequence; return (onset, ticks fed)."""
    for k, (phase, t) in enumerate(zip(phases, times)):
        env.vw[:, 3] = phase
        env.time[:] = t
        schedule.tick()
    return schedule.onset.clone(), k


def run_ticks(schedule, env, phases, times, world=0):
    """Feed the scripted gait phase tick by tick; return the latched onset."""
    for phase, t in zip(phases, times):
        env.vw[:, 3] = phase
        env.time[:] = t
        schedule.tick()
        if schedule.onset[world] >= 0:
            return float(schedule.onset[world])
    return -1.0


def test_rotation_matrix():
    g = torch.Generator().manual_seed(7)
    q = torch.randn(256, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    v = torch.randn(256, 3, generator=g)
    got = torch.bmm(rotation_matrix(q), v.unsqueeze(2)).squeeze(2)
    # Manual quaternion -> matrix, MuJoCo wxyz body->world convention.
    w, x, y, z = q.unbind(1)
    manual = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=1).reshape(-1, 3, 3)
    ref = torch.bmm(manual, v.unsqueeze(2)).squeeze(2)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
    # Rotation preserves norms, so |delivered| == requested magnitude.
    assert torch.allclose(got.norm(dim=1), v.norm(dim=1), atol=1e-6)
    # Identity quaternion is the identity map (body frame == world frame).
    ident = rotation_matrix(torch.tensor([[1.0, 0, 0, 0]]))
    assert torch.allclose(ident.squeeze(0), torch.eye(3), atol=1e-7)
    print('rotation_matrix vs manual quaternion->matrix: OK')


def test_onset_latch_matches_native():
    g = torch.Generator().manual_seed(11)
    phases, times = phases_and_times(1200)      # 2.4 s > 4 gait cycles
    for trial, target in enumerate((0.13, 0.5, 0.97, 0.003)):
        env = FakeEnv(2)
        schedule = KickSchedule(env, onset_cycle=3)
        schedule.mode = 'fixed'
        schedule.target.fill_(target)           # pin the drawn phase target
        got = run_ticks(schedule, env, phases, times)
        ref = native_scalar_latch(phases.tolist(), times.tolist(), target, 3)
        assert got == ref, f'trial {trial}: schedule onset {got} != native {ref} (target {target})'
        # No magnitude is configured in this test, so the latch fires but the
        # kick must not: zero-magnitude windows never deliver (native `nonzero`).
        assert schedule.onset.min() >= 1.2 - 1e-6, 'onset before 3 whole cycles'
        assert schedule.count.tolist() == [0, 0], schedule.count.tolist()
        assert torch.allclose(schedule.delivered_audit(), torch.full((2,), 0.0)), \
            'no magnitude configured: nothing should have been delivered'
        print(f'onset latch vs native scalar port, target {target}: OK (onset {got:.3f}s)')


def test_first_tick_cannot_latch():
    # prev_phase starts at 0 after reset, so a naive first-tick crossing
    # (target <= phase) is possible -- the cycles >= onset_cycle gate must
    # swallow it exactly like the native had_prev guard does.
    env = FakeEnv(1)
    schedule = KickSchedule(env, onset_cycle=3)
    schedule.mode = 'fixed'
    schedule.target.fill_(0.01)
    env.vw[:, 3] = 0.5
    env.time[:] = DT
    schedule.tick()
    assert schedule.onset[0] < 0, 'latched on the first tick after reset'
    print('first-tick spurious crossing gated: OK')


def test_fixed_cell_single_kick():
    env = FakeEnv(2)
    schedule = KickSchedule(env)
    schedule.configure(env, 'fwd', 2.5, seed=100)
    schedule.reset(torch.ones(2, dtype=torch.bool))
    target = float(schedule.target[0])
    phases, times = phases_and_times(1200)
    onset, _ = run_all(schedule, env, phases, times)
    ref = native_scalar_latch(phases.tolist(), times.tolist(), target, 3)
    assert float(onset[0]) == ref and ref > 0
    assert schedule.count.tolist() == [1, 1], schedule.count.tolist()
    # All three components delivered: +x only here, identity attitude.
    assert torch.allclose(env.v[:, :3], torch.tensor([[2.5, 0.0, 0.0]] * 2), atol=1e-6)
    assert torch.allclose(schedule.delivered_audit(), torch.full((2,), 2.5), atol=1e-6)
    # Continue long past the window: the window-index guard keeps it at one.
    for phase, t in zip(*phases_and_times(1200, dt=DT)):
        env.vw[:, 3] = phase
        env.time[:] = t + 2.4
        schedule.tick()
    assert schedule.count.tolist() == [1, 1], 'fixed cell fired more than one kick'
    print(f'fixed cell: single kick of 2.5 m/s at onset {ref:.3f}s: OK')


def test_curriculum_draw_and_ceiling():
    env = FakeEnv(4)
    schedule = KickSchedule(env)
    schedule.configure_training(env, total_timesteps=100, boundaries=(40, 70), seed=5)
    assert schedule.period == TRAIN_ON_DURATION + TRAIN_OFF_DURATION
    assert torch.allclose(schedule.fractions, torch.tensor([0.4, 0.7]))
    # Level boundaries in interaction space: 10/25 = 0.4 -> level 1 (2.0 m/s).
    schedule.interactions.fill_(10)
    assert torch.allclose(schedule._ceiling(), torch.full((4,), TRAIN_CEILINGS[1]))
    schedule.interactions.fill_(20)
    assert torch.allclose(schedule._ceiling(), torch.full((4,), TRAIN_CEILINGS[2]))
    schedule.interactions.fill_(1)
    assert torch.allclose(schedule._ceiling(), torch.full((4,), TRAIN_CEILINGS[0]))

    # Hand-picked uniforms: kick (u_fire > 0.2), direction 0 -> +x, 75% branch
    # with u_val = 0.5 -> fraction 0.75 -> magnitude 0.75 * ceiling.
    schedule.draw[:] = torch.tensor([0.9, 0.0, 0.5, 0.5])
    phases, times = phases_and_times(900)          # 1.8 s: every world latches
    onset, _ = run_all(schedule, env, phases, times)
    assert (onset >= 1.2 - 1e-6).all() and schedule.count.tolist() == [1] * 4
    expect = 0.75 * TRAIN_CEILINGS[0]
    assert torch.allclose(env.v[:, :3], torch.tensor([[expect, 0.0, 0.0]] * 4), atol=1e-6)
    assert torch.allclose(schedule.delivered_audit(), torch.full((4,), expect), atol=1e-6)
    # Same draw on the next 1.6 s window: a second, identical kick per world.
    for phase, t in zip(*phases_and_times(900, dt=DT)):
        env.vw[:, 3] = phase
        env.time[:] = t + 1.8
        schedule.tick()
    assert schedule.count.tolist() == [2] * 4, 'no second window kick'
    assert torch.allclose(env.v[:, :3], torch.tensor([[2 * expect, 0.0, 0.0]] * 4), atol=1e-6)
    # A no-force draw (u_fire < 0.2) consumes its window without kicking.
    schedule.draw[:, 0] = 0.1
    for phase, t in zip(*phases_and_times(900, dt=DT)):
        env.vw[:, 3] = phase
        env.time[:] = t + 3.6
        schedule.tick()
    assert schedule.count.tolist() == [2] * 4, 'no-force window still kicked'
    print(f'curriculum draw/ceiling/window guard (kick {expect} m/s): OK')


def test_direction_decode():
    # Six signed cardinals, one hot axis with the right sign.
    env = FakeEnv(6)
    schedule = KickSchedule(env)
    schedule.configure_training(env, total_timesteps=100, boundaries=(40, 70), seed=9)
    schedule.draw[:] = torch.tensor([0.9, 0.0, 0.5, 0.5])
    expect = {0: (1., 0., 0.), 1: (-1., 0., 0.), 2: (0., 1., 0.),
              3: (0., -1., 0.), 4: (0., 0., 1.), 5: (0., 0., -1.)}
    for d, vec in expect.items():
        schedule.draw[:, 1] = d / 6.0 + 1e-3   # floor(u*6) == d
        schedule._draw(torch.ones(6, dtype=torch.bool))
        assert torch.allclose(schedule.body[d], torch.tensor(vec), atol=1e-6), (d, schedule.body[d])
    print('six signed cardinal directions decoded: OK')


def test_fixed_phase_streams():
    # Per-world streams seeded seed + world_id: reproducible, world-distinct.
    env = FakeEnv(8)
    schedule = KickSchedule(env)
    schedule.configure(env, 'up', 1.0, seed=100)
    schedule.reset(torch.ones(8, dtype=torch.bool))
    first = schedule.target.clone()
    again = KickSchedule(env)
    again.configure(env, 'up', 1.0, seed=100)
    again.reset(torch.ones(8, dtype=torch.bool))
    assert torch.equal(first, again.target), 'phase streams are not reproducible'
    assert len(set(first.tolist())) == 8, 'phase draws are not world-distinct'
    assert DIRS['up'] == (0.0, 0.0, 1.0)
    print('fixed per-world onset phase streams: OK')


def main():
    test_rotation_matrix()
    test_onset_latch_matches_native()
    test_first_tick_cannot_latch()
    test_fixed_cell_single_kick()
    test_curriculum_draw_and_ceiling()
    test_direction_decode()
    test_fixed_phase_streams()
    print('ALL KICK MATH TESTS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
