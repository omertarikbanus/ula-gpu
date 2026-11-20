#!/usr/bin/env python
"""MJX (MuJoCo XLA) batched physics benchmark for the Go2 scene.

N robots (vmap), a fixed random ctrl per robot, K physics steps inside one
jax.jit + lax.scan. Two variants are timed per N:

  physics  pure mjx.step.
  ctrl     the same plus an in-graph PLACEHOLDER for the 500 Hz controller tick,
           run every physics step: a few batched small dense float32 matmuls per
           robot (--ctrl-dim x --ctrl-dim and --ctrl-dim2 x --ctrl-dim2, --ctrl-ops
           of each) whose result feeds ctrl so XLA cannot drop it. It is NOT the
           real controller; it only shows what extra per-step work costs.

Default scene is models/go2/scene_gpu.xml (primitive collision geoms). The
original mesh scene (models/go2/scene.xml) tries to allocate ~25 GB per robot in
MJX (mesh-plane and mesh-mesh collision buffers), see docs/GPU_BENCH.md.

CPU smoke test (run from outside the repo so MUJOCO_LOG.TXT is not created here):
    python bench/mjx_bench.py --num-envs 1,8,64 --steps 50
GPU:
    python bench/mjx_bench.py --num-envs 1024,4096,16384 --steps 200 --device gpu
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
    p.add_argument('--num-envs', default='32', help='comma list of batch sizes')
    p.add_argument('--steps', type=int, default=50, help='physics steps per timed rollout')
    p.add_argument('--device', default='cpu', choices=('cpu', 'gpu'))
    p.add_argument('--timestep', type=float, default=C.PHYSICS_DT, help='physics dt override (s); real system 0.002')
    p.add_argument('--reps', type=int, default=3, help='timed repetitions; best is reported')
    p.add_argument('--no-ctrl', action='store_true', help='skip the controller-placeholder variant')
    p.add_argument('--ctrl-dim', type=int, default=18, help='placeholder matrix size 1 (per robot)')
    p.add_argument('--ctrl-dim2', type=int, default=12, help='placeholder matrix size 2 (per robot)')
    p.add_argument('--ctrl-ops', type=int, default=4, help='matmuls of each size per physics step')
    p.add_argument('--disturb', action='store_true', help='also apply a fixed random qfrc_applied per robot')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--results', default=C.DEFAULT_RESULTS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.device == 'gpu' else 'cpu'

    import warnings
    warnings.filterwarnings('ignore', message='.*coplanar face.*')
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    dev = jax.devices()[0]
    print(f'# mjx: jax {jax.__version__} mujoco {mujoco.__version__} device {dev}', flush=True)

    m = mujoco.MjModel.from_xml_path(args.scene)
    m.opt.timestep = args.timestep
    d0 = mujoco.MjData(m)
    d0.qpos[2] = C.STAND_BASE_Z
    for leg in range(4):
        d0.qpos[7 + 3 * leg:10 + 3 * leg] = C.STAND_JOINTS
    mujoco.mj_forward(m, d0)
    print(f'# scene {os.path.basename(args.scene)}: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} '
          f'dt={m.opt.timestep:g}', flush=True)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d0)
    limits = jnp.asarray(m.actuator_ctrlrange if m.actuator_ctrllimited.any()
                         else m.actuator_forcerange)[:, 1]
    K = args.steps
    step = jax.vmap(mjx.step, in_axes=(None, 0))

    def make_rollout(with_ctrl):
        d1, d2, ops = args.ctrl_dim, args.ctrl_dim2, args.ctrl_ops

        def rollout(data, w1, w2):
            def body(carry, _):
                data, a1, a2 = carry
                if with_ctrl:
                    # PLACEHOLDER controller tick (per robot, every physics step).
                    for _ in range(ops):
                        a1 = jnp.tanh(jnp.matmul(a1, w1)) + 0.01
                        a2 = jnp.tanh(jnp.matmul(a2, w2)) + 0.01
                    tiny = 1e-6 * (a1[:, 0, 0] + a2[:, 0, 0])
                    data = data.replace(ctrl=data.ctrl + tiny[:, None])
                data = step(mx, data)
                return (data, a1, a2), None

            n = data.qpos.shape[0]
            a1 = jnp.zeros((n, d1, d1), jnp.float32)
            a2 = jnp.zeros((n, d2, d2), jnp.float32)
            (data, a1, a2), _ = jax.lax.scan(body, (data, a1, a2), None, length=K)
            return data.qpos, a1[:, 0, 0] + a2[:, 0, 0]

        return jax.jit(rollout)

    recs = []

    def keep(rec):
        recs.append(rec)
        C.append_results(args.results, [rec])  # written per run: a later failure loses nothing

    for n in C.parse_int_list(args.num_envs):
      variants = [('physics', False)] + ([] if args.no_ctrl else [('ctrl-ph', True)])
      try:
        key = jax.random.PRNGKey(args.seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ctrl = jax.random.uniform(k1, (n, m.nu), minval=-1.0, maxval=1.0) * limits
        batch = jax.vmap(lambda _: dx)(jnp.zeros(n))
        batch = batch.replace(ctrl=ctrl)
        if args.disturb:
            batch = batch.replace(qfrc_applied=jax.random.uniform(k2, (n, m.nv), minval=-1.0, maxval=1.0))
        w1 = jax.random.normal(k3, (args.ctrl_dim, args.ctrl_dim), jnp.float32) / args.ctrl_dim ** 0.5
        w2 = jax.random.normal(k4, (args.ctrl_dim2, args.ctrl_dim2), jnp.float32) / args.ctrl_dim2 ** 0.5

      except Exception as e:  # e.g. out of memory while building the batch
        for name, _ in variants:
            keep(C.failure('mjx', args.device, name, n, e))
        continue
      for name, with_ctrl in variants:
        try:
            fn = make_rollout(with_ctrl)
            t0 = time.perf_counter()
            jax.block_until_ready(fn(batch, w1, w2))
            compile_s = time.perf_counter() - t0
            best = float('inf')
            for _ in range(args.reps):
                t0 = time.perf_counter()
                qpos, _ = jax.block_until_ready(fn(batch, w1, w2))
                best = min(best, time.perf_counter() - t0)
            ok = bool(jnp.all(jnp.isfinite(qpos)))
            keep(C.report('mjx', args.device, name, n, K, best, m.opt.timestep,
                          extra=dict(compile_s=compile_s, finite=ok, scene=os.path.basename(args.scene),
                                     jax=jax.__version__, mujoco=mujoco.__version__,
                                     ctrl_dim=args.ctrl_dim, ctrl_dim2=args.ctrl_dim2,
                                     ctrl_ops=args.ctrl_ops if with_ctrl else 0)))
            if not ok:
                print('  warning: non-finite qpos', flush=True)
        except Exception as e:  # e.g. out of GPU memory at this N
            keep(C.failure('mjx', args.device, name, n, e))
    return recs


if __name__ == '__main__':
    main(sys.argv[1:])
