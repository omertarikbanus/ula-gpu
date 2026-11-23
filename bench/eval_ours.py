#!/usr/bin/env python3
"""Evaluate a policy trained by bench/pilot_epsilon.py (our own .pt format).

Reuses bench/icra_kick_eval.run_cell, so the kick protocol is identical to the
frozen-policy harness: additive base-velocity kick, onset after whole gait
cycles, common_fall_v1 window. The pilot's observation normalization is part
of the checkpoint and is applied exactly as during training -- evaluating a
normalized policy without its statistics would silently measure a different
controller.

--untrained runs a freshly initialised policy with as-new normalization
statistics (mean 0, var 1). That is the control: it walks because WBIC does
the locomotion, so the discriminating number is survival under hard kicks.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ula_gpu.ppo_epsilon import EpsilonPolicy          # noqa: E402


def make_predict(checkpoint, device):
    """Return a device-tensor predictor with frozen observation statistics."""
    policy = EpsilonPolicy().to(device)
    if checkpoint is None:
        mean, var = torch.zeros(54, device=device), torch.ones(54, device=device)
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(state["policy"])
        mean, var = state["normalizer"]["obs_rms"][:2]
        mean = torch.as_tensor(mean, device=device, dtype=torch.float32)
        var = torch.as_tensor(var, device=device, dtype=torch.float32)
    policy.eval()

    def predict(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, device=device, dtype=torch.float32)
            x = ((x - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)
            action = policy.distribution(x)[0].clamp(-1, 1)
            return action
    return predict


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="omit with --untrained")
    p.add_argument("--untrained", action="store_true",
                   help="evaluate a fresh policy as the no-training control")
    p.add_argument("--icra-root", type=Path, help="hash-pinned published Epsilon baseline")
    p.add_argument("--batched", action="store_true", help="evaluate all cells in one physics batch")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--library", type=Path)
    p.add_argument("--backend", choices=("vendor", "inline"), default="inline")
    p.add_argument("--dirs", default="fwd,down")
    p.add_argument("--mags", default="2.5,4.0")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--cap-steps", type=int, default=1200)
    p.add_argument("--onset-cycle", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args(argv)
    if sum((a.untrained, a.checkpoint is not None, a.icra_root is not None)) != 1:
        p.error("pass exactly one of --checkpoint, --untrained or --icra-root")
    if a.episodes <= 0 or a.cap_steps <= 0:
        p.error("episodes and cap-steps must be positive")
    a.tensor_policy = True
    return a


def run_batched(a, predict):
    """Keep the original per-cell phase streams while sharing one physics batch."""
    from hashlib import sha256
    from icra_kick_eval import COMMAND, DIRECTIONS, T_EVAL
    from ula_gpu.env import Go2Env
    from ula_gpu.kicks import KickSchedule
    cells = [(d, float(m)) for d in a.dirs.split(",") for m in a.mags.split(",")]
    if any(d not in DIRECTIONS or m <= 0 for d, m in cells):
        raise ValueError("expected known directions and positive magnitudes")
    n = a.episodes * len(cells)
    env = Go2Env(n, a.device, library=a.library, seed=a.seed, training=False,
                 max_steps=a.cap_steps, graph=a.device == "cuda", backend=a.backend)
    try:
        schedule = KickSchedule(env)
        schedule.configure(
            env, [d for d, _ in cells for _ in range(a.episodes)],
            [m for _, m in cells for _ in range(a.episodes)], seed=100,
            onset_cycle=a.onset_cycle,
            phase_seeds=[100 + i + world for i in range(len(cells))
                         for world in range(a.episodes)])
        obs = env.reset(command=COMMAND)
        env.capture_physics()
        fell = torch.zeros(n, dtype=torch.bool, device=env.device)
        pre_fall = torch.zeros_like(fell)
        completed = torch.zeros_like(fell)
        for _ in range(a.cap_steps):
            action = predict(obs[:, :54])
            obs, _r, term, _t, _f, _failed = env.step(action, auto_reset=False)
            onset = schedule.onset
            fell |= term & (onset >= 0) & (env.time <= onset + T_EVAL)
            pre_fall |= term & (onset < 0)
            completed |= (onset >= 0) & (env.time >= onset + T_EVAL)
        # One readback of scoring state; no policy/physics host bridge per step.
        onset, count, audit, fell, pre_fall, completed = (
            value.cpu().reshape(len(cells), a.episodes) for value in
            (schedule.onset, schedule.count, schedule.delivered_audit(),
             fell, pre_fall, completed))
        library_hash = sha256(Path(env.controller.lib._name).read_bytes()).hexdigest()
        rows = []
        for i, (direction, magnitude) in enumerate(cells):
            latched = onset[i] >= 0
            kicked = count[i] > 0
            err = float((audit[i][kicked] - magnitude).abs().max()) if bool(kicked.any()) else float("nan")
            gates = dict(kicked_once=bool((count[i] == 1).all()), kick_err=err < 1e-4,
                         no_pre_fall=not bool(pre_fall[i].any()), onset_all=bool(latched.all()),
                         window_complete=bool(completed[i].all()))
            survived = completed[i] & ~fell[i]
            rows.append(dict(
                seed=a.seed, direction=direction, magnitude=magnitude, episodes=a.episodes,
                cap_steps=a.cap_steps, onset_cycle=a.onset_cycle, cell_index=i,
                survived=int(survived.sum()), survival_pct=round(100 * float(survived.float().mean()), 1),
                completed=int(completed[i].sum()), incomplete=int((~completed[i]).sum()),
                falls_in_window=int(fell[i].sum()), falls_pre_onset=int(pre_fall[i].sum()),
                mean_onset_s=round(float(onset[i][latched].mean()), 3) if bool(latched.any()) else None,
                onset_min_s=round(float(onset[i][latched].min()), 3) if bool(latched.any()) else None,
                onset_max_s=round(float(onset[i][latched].max()), 3) if bool(latched.any()) else None,
                kick_mag_err_max=err, kicks_per_world_max=int(count[i].max()),
                library_sha256=library_hash, gates=gates))
        return rows
    finally:
        env.close()


def main(argv=None):
    a = parse_args(argv)
    from icra_kick_eval import DIRECTIONS, run_cell
    device = torch.device(a.device)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    label = "icra_epsilon" if a.icra_root else "untrained" if a.untrained else "trained"
    if a.icra_root:
        from ula_gpu.icra_inference import EpsilonActor
        predict = EpsilonActor(a.icra_root, a.seed, device)
    else:
        predict = make_predict(None if a.untrained else a.checkpoint, device)
    rows = run_batched(a, predict) if a.batched else []
    if not a.batched:
        cell_index = 0
        for direction in a.dirs.split(","):
            for magnitude in (float(m) for m in a.mags.split(",")):
                if direction not in DIRECTIONS:
                    raise ValueError(f"unknown kick direction: {direction}")
                row, gates = run_cell(a, predict, a.seed, direction, magnitude, cell_index)
                cell_index += 1
                row["gates"] = gates
                rows.append(row)
    for row in rows:
        row["label"] = label
        gates = row["gates"]
        print(f"  {label:<12} {row['direction']:<5} {row['magnitude']:.1f} m/s: "
              f"survival {row['survival_pct']:>5.1f}%  onset "
              f"{row['mean_onset_s']}s  completed {row['completed']}/{row['episodes']}  "
              f"gates {'ok' if all(gates.values()) else gates}",
              flush=True)
    a.results.parent.mkdir(parents=True, exist_ok=True)
    valid = all(all(row["gates"].values()) for row in rows)
    a.results.write_text(json.dumps({"label": label, "batched": a.batched,
                                     "seed": a.seed, "backend": a.backend,
                                     "gates_passed": valid,
                                     "checkpoint": str(a.checkpoint), "cells": rows}, indent=1) + "\n")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
