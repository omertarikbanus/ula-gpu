"""Run one short Go2 replay and Warp validation on a Colab GPU, without PPO."""
import argparse
from pathlib import Path
import os
import subprocess as sp
import sys
import tarfile
import tempfile


def run(repo, results=None):
    repo = repo.resolve()
    sp.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
            '--format=csv,noheader'], check=True)
    if not Path('/usr/include/eigen3/Eigen/Core').exists():
        sp.run(['apt-get', '-qq', 'update'], check=True)
        sp.run(['apt-get', '-qq', 'install', '-y', 'libeigen3-dev'], check=True)
    build = repo / 'build-cuda'
    env = {**os.environ, 'BUILD_DIR': str(build), 'PYTHON': sys.executable, 'PYTHONUNBUFFERED': '1'}
    print('Running controller replay and controller-only timings.', flush=True)
    replay = sp.run(['bash', 'bench/run_cuda_replay.sh'], env=env, cwd=repo)
    if not (build / 'libula_batch.so').is_file():
        replay.check_returncode()
        raise FileNotFoundError('missing controller library')

    # Use Colab's existing CUDA PyTorch. Avoid installing JAX or the MJX
    # plugin: neither is needed for these Warp tests.
    sp.run([sys.executable, '-m', 'pip', 'install', '-q', 'mujoco==3.11.0',
            'mujoco-warp==3.11.0', 'warp-lang==1.19.0'], check=True)
    sp.run([sys.executable, '-c', 'import torch, mujoco, mujoco_warp, warp; '
            'assert torch.cuda.is_available(), "this runner needs CUDA PyTorch"; '
            'assert mujoco.__version__ == mujoco_warp.__version__ == "3.11.0"; '
            'print("torch", torch.__version__, "mujoco", mujoco.__version__, "warp", warp.__version__)'], check=True)
    print('Running Warp closed loop and controller-plus-physics timings.', flush=True)
    sp.run([sys.executable, 'bench/verify_warp_loop.py', '--device', 'cuda',
            '--library', str(build / 'libula_batch.so'), '--num-envs', '64,256,1024',
            '--steps', '24', '--reps', '2', '--graph',
            '--results', str(results.resolve() if results else repo / 'bench/artifacts/results_warp_cuda.json')],
           env=env, cwd=repo, check=True)
    replay.check_returncode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, help='use an already extracted bundle')
    p.add_argument('--archive', type=Path, default=Path('/content/ula-go2-vendor.tar.gz'))
    p.add_argument('--results', type=Path, help='keep the Warp benchmark JSON here')
    args = p.parse_args()
    if args.repo:
        run(args.repo, args.results)
    else:
        with tempfile.TemporaryDirectory(prefix='ula-go2-') as tmp:
            with tarfile.open(args.archive, mode='r:gz') as bundle:
                bundle.extractall(tmp, filter='data')
            run(Path(tmp) / 'ula-gpu', args.results)


if __name__ == '__main__':
    main()
