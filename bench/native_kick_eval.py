#!/usr/bin/env python3
"""NATIVE baseline for the ICRA additive base-velocity kick protocol.

Runs INSIDE the ula-dev container (python3.12, envpool, the cached ula_envs
wheel extracted onto PYTHONPATH) and mirrors bench/icra_kick_eval.py: the
same grid (policy seeds x six directions x kick magnitudes), the same pinned
(0.8, 0, 0) command, the same scoring window as evaluation/core/_rl_eval.py
-- onset from info["force_onset"], fall = info["fall"] within
[onset, onset + 5.5], survived = no in-window fall AND t >= onset + 5.5, one
episode per env of one envpool batch per cell.

The kick is the native driver's own semantics, configured purely through the
TOML overlay: external_force_mode="fixed", external_force_type="velocity",
external_push_semantics="add", external_force=[dx,dy,dz] m/s in the BODY
frame, external_force_frame="body_latched_at_onset", on 0.2 s / off 100 s
(the window only; a single kick fires), onset_cycle 3, onset_phase -1 (drawn
per episode from the per-env RNG seeded external_force_seed + env_id, here
100 + cell index), external_force_body="go2/base_link_inertia".

Matched-pool deviations from the paper's serial protocol, both deliberate:
  - wbc_randomize_init=False: the serial path perturbs the initial state per
    episode (initstate.py); one pooled config serves the whole batch, so this
    harness pins the nominal reset instead, exactly like the vectorised
    _rl_eval path it mirrors.
  - sensor noise stays ON and state_source keeps its native default
    ("noisy_truth" with the declared truth_noise_* bands): this is the honest
    native pipeline, unlike compare_icra_policy.py's clean-observation cells.
Policies are the hash-pinned Epsilon checkpoints via
compare_icra_policy.load_policy (ICRA_HASHES imported from there).
"""

import argparse
import json
import os
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # compare_icra_policy
# Hash-pinned Epsilon loader; ICRA_HASHES imported (not duplicated) so the
# digests have one source of truth shared with the other ICRA harnesses.
from compare_icra_policy import ICRA_HASHES, load_policy  # noqa: E402

T_EVAL = 5.5                    # s, post-onset scoring window (metrics.py)
SPEED = 0.8                     # m/s, pinned forward command
NATIVE_CONFIG = Path('/app/src/controller/config/robots/sim/envpool.toml')
DIRECTIONS = ('fwd', 'back', 'left', 'right', 'up', 'down')
DIR_VEC = {'fwd': (1., 0., 0.), 'back': (-1., 0., 0.), 'left': (0., 1., 0.),
           'right': (0., -1., 0.), 'up': (0., 0., 1.), 'down': (0., 0., -1.)}


def overlay(direction, magnitude, force_seed, speed):
    """Fixed-cell velocity-kick [simulation] override + pinned command."""
    dx, dy, dz = DIR_VEC[direction]
    return f"""[simulation]
external_force_enable = true
external_force_mode = "fixed"
external_force_type = "velocity"
external_push_semantics = "add"
external_force = [{dx * magnitude}, {dy * magnitude}, {dz * magnitude}]
external_force_body = "go2/base_link_inertia"
external_force_frame = "body_latched_at_onset"
external_force_on_duration = 0.2
external_force_off_duration = 100.0
external_force_onset_cycle = 3
external_force_onset_phase = -1.0
external_force_seed = {force_seed}
[rl_command_source]
policy_random_velocity_enabled = false
policy_base_velocity = [{speed}, 0.0, 0.0]
desired_velocity = [{speed}, 0.0, 0.0]
[logserver]
enabled = false
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu',), default='cpu')
    p.add_argument('--policy-root', type=Path, required=True,
                   help='directory containing epsilon_s{0,1,2}')
    p.add_argument('--native-config', type=Path, default=NATIVE_CONFIG)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--dirs', default=','.join(DIRECTIONS))
    p.add_argument('--mags', default='1.0,2.5,4.0', help='kick magnitudes, m/s')
    p.add_argument('--episodes', type=int, default=48, help='envs (episodes) per cell')
    p.add_argument('--cap-steps', type=int, default=740, help='policy steps per episode')
    p.add_argument('--speed', type=float, default=SPEED)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--quick', action='store_true',
                   help='1 seed, 2 cells, 8 envs, 300-step cap (onset + kick, no window)')
    a = p.parse_args(argv)
    a.seed_list = [int(s) for s in a.seeds.split(',') if s.strip()]
    a.dir_list = [d for d in a.dirs.split(',') if d.strip()]
    a.mag_list = [float(m) for m in a.mags.split(',') if m.strip()]
    bad = [d for d in a.dir_list if d not in DIRECTIONS]
    if bad or not a.seed_list or not a.mag_list:
        p.error(f'expected seeds, magnitudes and directions among {DIRECTIONS}')
    if a.quick:
        a.seed_list, a.dir_list, a.mag_list = a.seed_list[:1], ['fwd', 'down'], [1.0]
        a.episodes, a.cap_steps = 8, 300
    return a


def run_cell(a, predict, seed, direction, magnitude, cell_index):
    import envpool
    import ula_envs  # noqa: F401  (registers the Quadruped* task ids)

    n, cap = a.episodes, a.cap_steps
    env = envpool.make(
        'QuadrupedWBC-v0', env_type='gymnasium', num_envs=n, seed=seed,
        num_threads=min(n, os.cpu_count() or 1),
        sim_config_path=str(a.native_config),
        sim_config_overlay=overlay(direction, magnitude, 100 + cell_index, a.speed),
        wbc_randomize_init=False, wbc_randomize_friction=False,
        terminate_when_unhealthy=False, max_episode_steps=cap)
    try:
        obs, _ = env.reset()
        onsets = np.full(n, -1.0)
        times = np.zeros(n)
        steps = np.zeros(n, dtype=np.int64)
        delivered = np.zeros(n)
        fell = np.zeros(n, dtype=bool)
        pre_fall = np.zeros(n, dtype=bool)
        completed = np.zeros(n, dtype=bool)
        finished = np.zeros(n, dtype=bool)
        while not finished.all():
            action = predict(np.asarray(obs))
            obs, _rew, term, trunc, info = env.step(action.astype(env.action_space.dtype))
            done = np.logical_or(term, trunc)
            t = np.asarray(info['sim_time'], dtype=float)
            on = np.asarray(info['force_onset'], dtype=float)
            fall = np.asarray(info['fall']).astype(bool)
            applied = np.asarray(info['force_applied'], dtype=float)
            live = ~finished
            onsets[live] = on[live]
            times[live] = t[live]
            steps[live] += 1
            delivered[live] = np.maximum(delivered[live],
                                         np.linalg.norm(applied[live], axis=1))
            fell |= live & (on >= 0) & (on <= t) & (t <= on + T_EVAL) & fall
            pre_fall |= live & (on < 0) & fall
            completed |= live & (on >= 0) & (t >= on + T_EVAL)
            finished |= done | completed
        survived = completed & ~fell
        latched = onsets >= 0
        kicked = delivered > 0
        err = float(np.abs(delivered[kicked] - magnitude).max()) if kicked.any() else float('nan')
        survival_time = np.clip(times - onsets, 0.0, T_EVAL)
        row = dict(
            seed=seed, direction=direction, magnitude=magnitude, episodes=n, cap_steps=cap,
            onset_cycle=3, cell_index=cell_index,
            survival_pct=round(100.0 * float(survived.mean()), 1),
            survived=int(survived.sum()), completed=int(completed.sum()),
            mean_onset_s=round(float(onsets[latched].mean()), 3) if latched.any() else None,
            onset_min_s=round(float(onsets[latched].min()), 3) if latched.any() else None,
            onset_max_s=round(float(onsets[latched].max()), 3) if latched.any() else None,
            mean_survival_s=round(float(survival_time[latched].mean()), 3) if latched.any() else None,
            kick_mag_err_max=err, kicked=int(kicked.sum()),
            falls_pre_onset=int(pre_fall.sum()),
        )
        gates = dict(onset_all=bool(latched.all()), kick_err=bool(kicked.all()) and err < 1e-4,
                     no_pre_fall=int(pre_fall.sum()) == 0)
        return row, gates
    finally:
        env.close()


def main(argv=None):
    a = parse_args(argv)
    import ula_envs
    library_hash = sha256(Path(ula_envs._quadruped.__file__).read_bytes()).hexdigest()
    cells = [(seed, d, m) for seed in a.seed_list for d in a.dir_list for m in a.mag_list]
    print(f'native kick eval: {len(cells)} cells x {a.episodes} envs x {a.cap_steps} steps, '
          f'command {a.speed} m/s, window {T_EVAL}s, onset_cycle 3', flush=True)
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
    # Survival table: one block per seed, directions x magnitudes.
    for seed in a.seed_list:
        block = [r for r in rows if r['seed'] == seed]
        print(f'\nseed {seed} survival % (dirs x mags {sorted(set(r["magnitude"] for r in block))}):')
        for d in DIRECTIONS:
            if any(r['direction'] == d for r in block):
                line = '  '.join(f"{r['survival_pct']:5.1f}" for r in block if r['direction'] == d)
                print(f'  {d:>5s}  {line}')
    payload = dict(
        backend='native', device='cpu', command_mps=[a.speed, 0.0, 0.0], window_s=T_EVAL,
        cap_steps=a.cap_steps, episodes=a.episodes, quick=a.quick,
        sensor_noise='on (native default)', state_source='default (noisy_truth)',
        wbc_randomize_init=False, terminate_when_unhealthy=False,
        library_sha256=library_hash, policy_hashes=[h[0] for h in ICRA_HASHES],
        host=platform.node(),
        time=time.strftime('%Y-%m-%dT%H:%M:%S'), wall_s=round(time.time() - t0, 1), cells=rows)
    if a.results != Path('-'):
        a.results.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'\ndone in {payload["wall_s"]}s -> {a.results}', flush=True)
    if failed_gates:
        print(f'SANITY GATE FAILURES: {failed_gates}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
