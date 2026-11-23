"""Pure-Torch, device-resident inference for the frozen ICRA Epsilon actor.

Loads the three published Epsilon PPO checkpoints without SB3 on the forward
path: the actor weights come from policy.pth inside the SB3 zip (via zipfile
+ torch.load weights_only=True) and the observation statistics from the
matching vecnormalize pickle. forward() replays the exact deterministic SB3
inference path on the target device:

    float64 obs normalization -- VecNormalize keeps float64 statistics and
    clips (clip 10, epsilon 1e-8) before the float32 cast -- then
    Linear(54,256)-Tanh, Linear(256,128)-Tanh, Linear(128,25), clip [-1, 1].

No NumPy, no host syncs, no .item()/.cpu() anywhere in forward(). The
checkpoint files are hash-pinned to the published SHA-256 digests (the single
source of truth, shared with bench/compare_icra_policy.py). Running this
module (or --self-test) compares forward() against the SB3 loader in
bench/compare_icra_policy.py on random inputs, moving through CPU/NumPy for
the comparison only, and asserts identical outputs.
"""

import argparse
import io
import json
import pickle
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

import torch
import torch.nn.functional as F

OBS_DIM, HIDDEN, ACTION_DIM = 54, (256, 128), 25
CLIP_OBS, NORM_EPS, TIMESTEPS = 10.0, 1e-8, 85_065_728

# Final reported checkpoints and their matching observation statistics, in seed order.
ICRA_HASHES = (
    ("fdc9cbee4cd5f3299910c26b77f8e51f7a47db9df2384134b6ca74a680e6e369",
     "dbe26118ef4ad206e24db33fe4f556b4ee5dd2198ab71b0e531e773bce013f38"),
    ("606bb391e49596ac712cc5f58c35b4738bb24d4d812e4e20be21d5276b401616",
     "4b279d0e8d35522c93bf9ec38627c4df26daa0e7645e3d343e76d8846ac8fb5d"),
    ("3b3d6ff5a22823560a546e0e353d35a2d4a59fbc585f4340bafd97e9824a315f",
     "9a6ae74098d4e2378ddb21ea3714f1bf203690d2347fed797f100ce65a6bbb3c"),
)


def epsilon_paths(root, seed):
    root = Path(root)
    model = root / f"epsilon_s{seed}" / "quadruped_ppo_model.zip"
    return model, model.with_name(f"{model.stem}_vecnormalize.pkl")


def verify_epsilon_files(seed, model, stats):
    """Raise ValueError unless both artifacts match the published digests."""
    for path, expected in zip((model, stats), ICRA_HASHES[seed]):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"not the published Epsilon artifact: {path}")


class EpsilonActor:
    """Frozen deterministic Epsilon actor; all tensors live on one device."""

    def __init__(self, root, seed, device="cpu", dtype=torch.float32):
        if not 0 <= seed < len(ICRA_HASHES):
            raise ValueError(f"seed must be one of {tuple(range(len(ICRA_HASHES)))}")
        model, stats = epsilon_paths(root, seed)
        verify_epsilon_files(seed, model, stats)
        with zipfile.ZipFile(model) as z:
            meta = json.loads(z.read("data"))
            state = torch.load(io.BytesIO(z.read("policy.pth")),
                               map_location="cpu", weights_only=True)
        self._check_hyperparams(meta, state)
        mean, var, clip, eps = self._load_norm_stats(stats)
        if abs(clip - CLIP_OBS) > 1e-12 or abs(eps - NORM_EPS) > 1e-12:
            raise ValueError(f"unexpected normalization constants: clip={clip} eps={eps}")
        device = torch.device(device)
        self.device, self.dtype, self.seed = device, dtype, seed
        # float64 normalization buffers: SB3 normalizes in float64 and casts
        # to float32 only after clipping, so we must too for bit-exact parity.
        self.obs_mean = mean.to(device=device, dtype=torch.float64)
        self.obs_std = torch.sqrt(var.to(device=device, dtype=torch.float64) + eps)
        self.w0 = state["mlp_extractor.policy_net.0.weight"].to(device=device, dtype=dtype)
        self.b0 = state["mlp_extractor.policy_net.0.bias"].to(device=device, dtype=dtype)
        self.w1 = state["mlp_extractor.policy_net.2.weight"].to(device=device, dtype=dtype)
        self.b1 = state["mlp_extractor.policy_net.2.bias"].to(device=device, dtype=dtype)
        self.w2 = state["action_net.weight"].to(device=device, dtype=dtype)
        self.b2 = state["action_net.bias"].to(device=device, dtype=dtype)

    @staticmethod
    def _check_hyperparams(meta, state):
        kwargs = meta.get("policy_kwargs", {})
        arch = kwargs.get("net_arch")
        pi = arch[0].get("pi") if isinstance(arch, list) and arch else arch
        shapes = {k: tuple(state[k].shape) for k in
                  ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight",
                   "action_net.weight")}
        problems = []
        if meta.get("num_timesteps") != TIMESTEPS:
            problems.append(f"num_timesteps {meta.get('num_timesteps')} != {TIMESTEPS}")
        if kwargs.get("actor_obs_dim") != OBS_DIM or \
                meta.get("observation_space", {}).get("_shape") != [OBS_DIM]:
            problems.append("actor input is not 54 observations")
        if list(pi or ()) != list(HIDDEN):
            problems.append(f"actor net arch {pi} != {list(HIDDEN)}")
        if meta.get("action_space", {}).get("_shape") != [ACTION_DIM]:
            problems.append("action dimension is not 25")
        if shapes["mlp_extractor.policy_net.0.weight"] != (HIDDEN[0], OBS_DIM) or \
                shapes["mlp_extractor.policy_net.2.weight"] != (HIDDEN[1], HIDDEN[0]) or \
                shapes["action_net.weight"] != (ACTION_DIM, HIDDEN[1]):
            problems.append(f"actor weight shapes {shapes}")
        if problems:
            raise ValueError("unexpected Epsilon policy hyperparameters: " + "; ".join(problems))

    @staticmethod
    def _load_norm_stats(stats_path):
        # Unpickling a VecNormalize needs the SB3 class itself (bench venv).
        try:
            from stable_baselines3.common.vec_env import VecNormalize
        except ImportError as exc:
            raise RuntimeError("the Epsilon vecnormalize pickle needs stable_baselines3 "
                               "(bench venv); nothing else in this module does") from exc
        with open(stats_path, "rb") as f:
            norm = pickle.load(f)
        if not norm.norm_obs or norm.obs_rms.mean.shape != (OBS_DIM,):
            raise ValueError("unexpected Epsilon normalization statistics")
        return (torch.as_tensor(norm.obs_rms.mean), torch.as_tensor(norm.obs_rms.var),
                float(norm.clip_obs), float(norm.epsilon))

    @torch.no_grad()
    def forward(self, obs):
        """(n, 54) torch tensor on self.device -> deterministic (n, 25) actions."""
        x = ((obs.to(torch.float64) - self.obs_mean) / self.obs_std)
        x = x.clamp_(-CLIP_OBS, CLIP_OBS).to(self.dtype)
        x = torch.tanh(F.linear(x, self.w0, self.b0))
        x = torch.tanh(F.linear(x, self.w1, self.b1))
        return F.linear(x, self.w2, self.b2).clamp_(-1.0, 1.0)

    __call__ = forward


def self_test(root, device="cpu", seeds=(0, 1, 2), batches=4, batch=64):
    """Compare forward() against the SB3 loader on random inputs; assert identical.

    The comparison moves through CPU/NumPy (that is where the SB3 loader
    lives); on CPU the outputs must match element for element, on CUDA only
    kernel-order noise (<= 2e-5) is tolerated.
    """
    import numpy as np
    bench = str(Path(__file__).resolve().parents[1] / "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    from compare_icra_policy import load_policy
    tolerance = 0.0 if torch.device(device).type == "cpu" else 2e-5
    report = {}
    for seed in seeds:
        mine = EpsilonActor(root, seed, device)
        predict = load_policy(Path(root), seed)
        rng = np.random.default_rng(0xC0FFEE + seed)
        worst, identical = 0.0, True
        for _ in range(batches):
            obs = (rng.normal(size=(batch, OBS_DIM)) * rng.choice([0.2, 1.0, 4.0])).astype(np.float32)
            got = mine(torch.as_tensor(obs, device=device)).cpu().numpy()
            ref = predict(obs)
            identical &= bool((got == ref).all())
            worst = max(worst, float(np.abs(got - ref).max()))
        report[seed] = dict(bitexact=bool(identical), max_abs_diff=worst)
        assert (identical if tolerance == 0.0 else worst <= tolerance), \
            f"seed {seed}: actor disagrees with the SB3 loader (max abs diff {worst:g})"
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true",
                   help="compare against the SB3 loader (this is the default action)")
    p.add_argument("--policy-root", type=Path, default=Path("/home/tarik/quadruped_ws/envpool/data"))
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--batches", type=int, default=4)
    a = p.parse_args(argv)
    report = self_test(a.policy_root, a.device, batches=a.batches)
    for seed, r in sorted(report.items()):
        print(f"self-test seed {seed}: bitexact={r['bitexact']} max_abs_diff={r['max_abs_diff']:g}")
    print("icra_inference self-test: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
