#!/usr/bin/env python3
"""Warp-harness evaluation of the ICRA additive base-velocity kick protocol.

Per cell (policy seed, direction, magnitude) one Go2Env batch of --episodes
worlds walks at the pinned command (0.8, 0, 0); after --onset-cycle (3) whole
gait cycles the schedule latches a per-world onset phase drawn from a stream
seeded 100 + cell + world, and delivers ONE additive world-frame velocity kick
R(base) @ (body direction * magnitude) to qvel[0:3].  Scoring follows
evaluation/core/_rl_eval.py: survived = no fall (the Go2Env.step termination
rule: trunk contact > 1 N, |roll| > 0.8, |pitch| > 1.0, nonfinite or
controller failure) anywhere in [onset, onset + 5.5] AND the window completed
(t >= onset + 5.5).  Worlds that finish keep simulating (auto_reset=False)
but are masked out of scoring; all scoring state stays on the device as masks
and is read once after the loop.  The only per-step host transfer is the
observation the frozen CPU policy consumes.

Policies are the hash-pinned Epsilon checkpoints loaded through
bench/compare_icra_policy.load_policy (ICRA_HASHES imported from there).

--quick is a wiring smoke test: 1 seed, 2 cells, 4 worlds, 120 steps, and
onset_cycle 1 so every world is kicked inside the short budget (the full
protocol keeps onset_cycle 3, which lands onsets near 1.3-1.8 s).
"""

import argparse
import json
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))       # compare_icra_policy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # ula_gpu
# Hash-pinned Epsilon loader; ICRA_HASHES imported (not duplicated) so the
# digests have one source of truth shared with the other ICRA harnesses.
from compare_icra_policy import ICRA_HASHES, load_policy  # noqa: E402

T_EVAL = 5.5                    # s, post-onset scoring window (_rl_eval.py)
COMMAND = (0.8, 0.0, 0.0)       # m/s, pinned forward command for kick cells
DIRECTIONS = ('fwd', 'back', 'left', 'right', 'up', 'down')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--library', type=Path)
    p.add_argument('--policy-root', type=Path, required=True,
                   help='directory containing epsilon_s{0,1,2}')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--dirs', default=','.join(DIRECTIONS))
    p.add_argument('--mags', default='1.0,2.5,4.0', help='kick magnitudes, m/s')
    p.add_argument('--episodes', type=int, default=48, help='worlds (episodes) per cell')
    p.add_argument('--cap-steps', type=int, default=740, help='policy steps per episode')
    p.add_argument('--onset-cycle', type=int, default=3)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--quick', action='store_true',
                   help='1 seed, 2 cells, 4 worlds, 120 steps, onset_cycle 1')
    a = p.parse_args(argv)
    a.seed_list = [int(s) for s in a.seeds.split(',') if s.strip()]
    a.dir_list = [d for d in a.dirs.split(',') if d.strip()]
    a.mag_list = [float(m) for m in a.mags.split(',') if m.strip()]
    bad = [d for d in a.dir_list if d not in DIRECTIONS]
    if bad or not a.seed_list or not a.mag_list:
        p.error(f'expected seeds, magnitudes and directions among {DIRECTIONS}')
    if a.quick:
        a.seed_list, a.dir_list, a.mag_list = a.seed_list[:1], ['fwd', 'down'], [1.0]
        a.episodes, a.cap_steps, a.onset_cycle = 4, 120, 1
    return a


def run_cell(a, predict, seed, direction, magnitude, cell_index):
    import torch
    import warp as wp
    wp.config.quiet = True
    from ula_gpu.env import Go2Env
    from ula_gpu.kicks import DIRS, KickSchedule

    n, cap = a.episodes, a.cap_steps
    env = Go2Env(n, a.device, library=a.library, seed=seed, training=False,
                 max_steps=cap, graph=a.device == 'cuda',
                 backend=getattr(a, "backend", "vendor"))
    try:
        schedule = KickSchedule(env)
        # Per-world onset phase streams seeded (100 + cell) + world, mirroring
        # the native external_force_seed + env_id.
        schedule.configure(env, direction, magnitude, seed=100 + cell_index,
                           onset_cycle=a.onset_cycle)
        obs = env.reset(command=COMMAND)
        if a.device == 'cuda':
            env.capture_physics()
        onset = schedule.onset
        fell = torch.zeros(n, dtype=torch.bool, device=env.device)
        pre_fall = torch.zeros(n, dtype=torch.bool, device=env.device)
        completed = torch.zeros(n, dtype=torch.bool, device=env.device)
        for _ in range(cap):
            actor_obs = obs[:, :54]
            action = predict(actor_obs if getattr(a, "tensor_policy", False)
                             else actor_obs.cpu().numpy())
            obs, _r, term, _t, _f, _failed = env.step(env.as_tensor(action), auto_reset=False)
            t = env.time
            window = (onset >= 0) & (t <= onset + T_EVAL)
            fell |= term & window
            pre_fall |= term & (onset < 0)
            completed |= (onset >= 0) & (t >= onset + T_EVAL)
        survived = completed & ~fell
        # One readback after the loop; nothing above touched the host.
        onset_np = onset.cpu()
        latched = onset_np >= 0
        count = schedule.count.cpu()
        audit = schedule.delivered_audit().cpu()
        kicked = count > 0
        err = float((audit[kicked] - magnitude).abs().max()) if bool(kicked.any()) else float('nan')
        row = dict(
            seed=seed, direction=direction, magnitude=magnitude, episodes=n, cap_steps=cap,
            onset_cycle=a.onset_cycle, cell_index=cell_index,
            survival_pct=round(100.0 * float(survived.float().mean()), 1),
            survived=int(survived.cpu().sum()),
            completed=int(completed.cpu().sum()),
            falls_in_window=int(fell.cpu().sum()),
            incomplete=int((~completed).cpu().sum()),
            mean_onset_s=round(float(onset_np[latched].mean()), 3) if bool(latched.any()) else None,
            onset_min_s=round(float(onset_np[latched].min()), 3) if bool(latched.any()) else None,
            onset_max_s=round(float(onset_np[latched].max()), 3) if bool(latched.any()) else None,
            kick_mag_err_max=err, kicks_per_world_max=int(count.max()),
            falls_pre_onset=int(pre_fall.cpu().sum()),
        )
        row['library_sha256'] = sha256(Path(env.controller.lib._name).read_bytes()).hexdigest()
        gates = dict(kicked_once=bool((count == 1).all()), kick_err=err < 1e-4,
                     no_pre_fall=int(pre_fall.cpu().sum()) == 0, onset_all=bool(latched.all()),
                     window_complete=bool(completed.all()))
        return row, gates
    finally:
        env.close()


def main(argv=None):
    a = parse_args(argv)
    cells = [(seed, d, m) for seed in a.seed_list for d in a.dir_list for m in a.mag_list]
    print(f'kick eval: {len(cells)} cells x {a.episodes} worlds x {a.cap_steps} steps '
          f'on {a.device}, command {COMMAND}, window {T_EVAL}s, onset_cycle {a.onset_cycle}', flush=True)
    rows, failed_gates = [], []
    t0 = time.time()
    for index, (seed, direction, magnitude) in enumerate(cells):
        predict = load_policy(a.policy_root, seed)
        row, gates = run_cell(a, predict, seed, direction, magnitude, index)
        rows.append(row)
        if not all(gates.values()):
            failed_gates.append((row['seed'], row['direction'], row['magnitude'], gates))
        print(f"seed {seed} {direction:>5s} {magnitude:3.1f} m/s: "
              f"survival {row['survival_pct']:5.1f}%  onset {row['mean_onset_s']!s:>6s}s  "
              f"kick err {row['kick_mag_err_max']:.2e}  pre-falls {row['falls_pre_onset']}"
              + ('' if all(gates.values()) else f'  GATES {gates}'), flush=True)
    payload = dict(
        backend='warp', device=a.device, command_mps=list(COMMAND), window_s=T_EVAL,
        onset_cycle=a.onset_cycle, cap_steps=a.cap_steps, episodes=a.episodes,
        policy_hashes=[h[0] for h in ICRA_HASHES],
        quick=a.quick, host=platform.node(), time=time.strftime('%Y-%m-%dT%H:%M:%S'),
        wall_s=round(time.time() - t0, 1), cells=rows)
    if a.results != Path('-'):
        a.results.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'done in {payload["wall_s"]}s -> {a.results}', flush=True)
    if failed_gates:
        print(f'SANITY GATE FAILURES: {failed_gates}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
