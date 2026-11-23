"""CPU-only regression checks for checkpoint transport and evaluation inputs."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
TEMP = "/tmp/opencode" if Path("/tmp/opencode").is_dir() else None


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job = load("overnight_job", "bench/run_overnight.py")
driver = load("training_driver", "bench/colab_training_driver.py")
evaluate = load("evaluate_ours", "bench/eval_ours.py")
from ula_gpu.ppo_epsilon import EpsilonPolicy, Normalizer


class TrainingJobTests(unittest.TestCase):
    def test_budget_matches_and_requested_world_count(self):
        for _, n_envs, n_steps, _ in job.CONFIGS:
            self.assertEqual(n_envs, 16128)
            self.assertEqual(job.BUDGET % (n_envs * n_steps), 0)

    def test_normalizer_roundtrip(self):
        original = Normalizer(4, 54, "cpu")
        original.observe(torch.randn(4, 54))
        original.reward(torch.randn(4))
        restored = Normalizer(4, 54, "cpu")
        restored.load_state_dict(original.state_dict())
        obs = torch.randn(4, 54)
        self.assertTrue(torch.equal(original.normalize_obs(obs), restored.normalize_obs(obs)))
        self.assertTrue(torch.equal(original.returns, restored.returns))
        self.assertEqual(original.obs_rms.count, restored.obs_rms.count)

    def test_evaluator_uses_frozen_stats_on_device(self):
        torch.manual_seed(0)
        policy = EpsilonPolicy()
        normalizer = Normalizer(4, 54, "cpu")
        normalizer.observe(torch.randn(4, 54) * 3 + 2)
        obs = torch.randn(4, 54)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(dict(policy=policy.state_dict(), normalizer=normalizer.state_dict()), path)
            predict = evaluate.make_predict(path, torch.device("cpu"))
            actual = predict(obs)
        with torch.no_grad():
            expected = policy.distribution(normalizer.normalize_obs(obs))[0].clamp(-1, 1)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(actual.device, obs.device)
        self.assertFalse(actual.requires_grad)

    def test_snapshot_roundtrip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            remote, local = Path(tmp) / "remote", Path(tmp) / "local"
            remote.mkdir()
            local.mkdir()
            torch.save({"policy": {"weight": torch.ones(2)}}, remote / "ckpt_smoke.pt")
            (remote / "ckpt_smoke.pt.tmp").write_bytes(b"unfinished")
            (remote / "progress.json").write_text(json.dumps({"run_id": "unit-test"}))
            with patch.object(job, "OUT", remote):
                job.snapshot()
            state, hashes = driver.restore_archive(remote / "snapshot.tar", local, "unit-test")
            self.assertEqual(state["run_id"], "unit-test")
            self.assertIn("ckpt_smoke.pt", hashes)
            self.assertFalse((local / "ckpt_smoke.pt.tmp").exists())
            saved = torch.load(local / "ckpt_smoke.pt", weights_only=False)
            self.assertTrue(torch.equal(saved["policy"]["weight"], torch.ones(2)))
            with self.assertRaisesRegex(ValueError, "different experiment"):
                driver.restore_archive(remote / "snapshot.tar", local, "wrong")
            import tarfile
            broken = remote / "broken.tar"
            with tarfile.open(remote / "snapshot.tar") as source, tarfile.open(broken, "w") as target:
                for member in source:
                    data = source.extractfile(member).read()
                    if member.name == "ckpt_smoke.pt":
                        data = b"x" * len(data)
                    target.addfile(member, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                driver.restore_archive(broken, local, "unit-test")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
