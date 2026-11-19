#!/usr/bin/env python
"""MuJoCo Warp batched physics benchmark for the Go2 scene.

Same measurements as mjx_bench.py: N worlds, a fixed random ctrl per robot, K
physics steps per timed rollout. Two variants per N:

  physics  mujoco_warp.step only.
  ctrl-ph  a PLACEHOLDER per-robot controller kernel (wp.kernel, one thread per
           robot) launched before every physics step: --ctrl-ops small dense
           float32 matmuls of --ctrl-dim and --ctrl-dim2 square matrices per
           robot, result folded into ctrl so it has an effect. It is NOT the real
           controller; it is where a native CUDA kernel would go (docs/GPU_BENCH.md).

On CUDA the K-step loop (controller kernel + step) is captured once into a CUDA
graph and replayed (--no-graph disables it). On CPU Warp has no graph capture,
so the same loop runs eagerly; the code path is otherwise identical.

CPU smoke test (run from outside the repo, MuJoCo may write MUJOCO_LOG.TXT):
    python bench/warp_bench.py --num-envs 1,8,64 --steps 50
GPU:
    python bench/warp_bench.py --num-envs 1024,4096,16384 --steps 200 --device gpu
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default=C.DEFAULT_SCENE)
    p.add_argument('--num-envs', default='32', help='comma list of world counts')
    p.add_argument('--steps', type=int, default=50, help='physics steps per timed rollout')
    p.add_argument('--device', default='cpu', choices=('cpu', 'gpu'))
    p.add_argument('--timestep', type=float, default=C.PHYSICS_DT, help='physics dt override (s); real system 0.002')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--no-ctrl', action='store_true', help='skip the controller-placeholder variant')
    p.add_argument('--no-graph', action='store_true', help='do not use CUDA graph capture (GPU only)')
    p.add_argument('--ctrl-dim', type=int, default=18)
    p.add_argument('--ctrl-dim2', type=int, default=12)
    p.add_argument('--ctrl-ops', type=int, default=4, help='matmuls of each size per physics step')
    p.add_argument('--nconmax', type=int, default=16, help='contacts allocated per world')
    p.add_argument('--njmax', type=int, default=96, help='constraint rows allocated per world')
    p.add_argument('--disturb', action='store_true', help='also apply a fixed random qfrc_applied per robot')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--results', default=C.DEFAULT_RESULTS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import numpy as np
    import mujoco
    import warp as wp
    import mujoco_warp as mjw

    wp.config.quiet = True
    wp.init()
    device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    use_graph = args.device == 'gpu' and not args.no_graph

    @wp.kernel
    def controller_placeholder(
        w1: wp.array2d(dtype=float), w2: wp.array2d(dtype=float),
        a1: wp.array3d(dtype=float), b1: wp.array3d(dtype=float),
        a2: wp.array3d(dtype=float), b2: wp.array3d(dtype=float),
        d1: int, d2: int, ops: int, nu: int,
        ctrl: wp.array2d(dtype=float)):
        # PLACEHOLDER controller tick: one thread = one robot.
        # ping-pong a <-> b so the matmul never reads what it writes.
        w = wp.tid()
        for o in range(ops):
            for i in range(d1):
                for j in range(d1):
                    s = float(0.0)
                    for k in range(d1):
                        s += a1[w, i, k] * w1[k, j]
                    b1[w, i, j] = wp.tanh(s) + 0.01
            for i in range(d1):
                for j in range(d1):
                    a1[w, i, j] = b1[w, i, j]
            for i in range(d2):
                for j in range(d2):
                    s = float(0.0)
                    for k in range(d2):
                        s += a2[w, i, k] * w2[k, j]
                    b2[w, i, j] = wp.tanh(s) + 0.01
            for i in range(d2):
                for j in range(d2):
                    a2[w, i, j] = b2[w, i, j]
        tiny = 1.0e-6 * (a1[w, 0, 0] + a2[w, 0, 0])
        for u in range(nu):
            ctrl[w, u] = ctrl[w, u] + tiny

    print(f'# warp: warp {wp.config.version} mujoco {mujoco.__version__} device {device} '
          f'graph={use_graph}', flush=True)

    mjm = mujoco.MjModel.from_xml_path(args.scene)
    mjm.opt.timestep = args.timestep
    mjd = mujoco.MjData(mjm)
    mjd.qpos[2] = C.STAND_BASE_Z
    for leg in range(4):
        mjd.qpos[7 + 3 * leg:10 + 3 * leg] = C.STAND_JOINTS
    mujoco.mj_forward(mjm, mjd)
    print(f'# scene {os.path.basename(args.scene)}: nq={mjm.nq} nv={mjm.nv} nu={mjm.nu} '
          f'ngeom={mjm.ngeom} dt={mjm.opt.timestep:g}', flush=True)

    limits = (mjm.actuator_ctrlrange if mjm.actuator_ctrllimited.any() else mjm.actuator_forcerange)[:, 1]
    K = args.steps
    d1, d2, ops = args.ctrl_dim, args.ctrl_dim2, args.ctrl_ops
    rng = np.random.default_rng(args.seed)
    recs = []

    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        w1 = wp.array((rng.standard_normal((d1, d1)) / d1 ** 0.5).astype(np.float32), dtype=float)
        w2 = wp.array((rng.standard_normal((d2, d2)) / d2 ** 0.5).astype(np.float32), dtype=float)

        for n in C.parse_int_list(args.num_envs):
            d = mjw.put_data(mjm, mjd, nworld=n, nconmax=args.nconmax, njmax=args.njmax)
            ctrl_np = (rng.uniform(-1, 1, (n, mjm.nu)) * limits).astype(np.float32)
            d.ctrl.assign(ctrl_np)
            if args.disturb:
                d.qfrc_applied.assign(rng.uniform(-1, 1, (n, mjm.nv)).astype(np.float32))
            a1 = wp.zeros((n, d1, d1), dtype=float)
            b1 = wp.zeros((n, d1, d1), dtype=float)
            a2 = wp.zeros((n, d2, d2), dtype=float)
            b2 = wp.zeros((n, d2, d2), dtype=float)

            def run_loop(with_ctrl):
                for _ in range(K):
                    if with_ctrl:
                        wp.launch(controller_placeholder, dim=n,
                                  inputs=[w1, w2, a1, b1, a2, b2, d1, d2, ops, mjm.nu, d.ctrl])
                    mjw.step(m, d)

            variants = [('physics', False)] + ([] if args.no_ctrl else [('ctrl-ph', True)])
            for name, with_ctrl in variants:
                d.ctrl.assign(ctrl_np)  # undo drift from the placeholder
                t0 = time.perf_counter()
                run_loop(with_ctrl)  # warmup / kernel compile / contact-buffer touch
                wp.synchronize()
                if use_graph:
                    with wp.ScopedCapture() as cap:
                        run_loop(with_ctrl)
                    graph = cap.graph
                    wp.capture_launch(graph)
                    wp.synchronize()
                compile_s = time.perf_counter() - t0

                best = float('inf')
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    if use_graph:
                        wp.capture_launch(graph)
                    else:
                        run_loop(with_ctrl)
                    wp.synchronize()
                    best = min(best, time.perf_counter() - t0)
                qpos = d.qpos.numpy()
                ok = bool(np.isfinite(qpos).all())
                recs.append(C.report('warp', args.device, name, n, K, best, mjm.opt.timestep,
                                     extra=dict(compile_s=compile_s, finite=ok, graph=use_graph,
                                                scene=os.path.basename(args.scene),
                                                warp=wp.config.version, mujoco=mujoco.__version__,
                                                nconmax=args.nconmax, njmax=args.njmax,
                                                ctrl_dim=d1, ctrl_dim2=d2, ctrl_ops=ops if with_ctrl else 0)))
                if not ok:
                    print('  warning: non-finite qpos', flush=True)
                if use_graph:
                    del graph
            del d, a1, b1, a2, b2
    C.append_results(args.results, recs)
    return recs


if __name__ == '__main__':
    main(sys.argv[1:])
