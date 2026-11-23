"""Short CPU regression: censoring must fail validity, and batching keeps seeds."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from eval_ours import make_predict, run_batched
from icra_kick_eval import run_cell


class RecoveryScoringTests(unittest.TestCase):
    def test_batch_matches_serial_and_rejects_short_window(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        a = SimpleNamespace(
            device=os.environ.get("RECOVERY_TEST_DEVICE", "cpu"),
            library=os.environ.get("RECOVERY_TEST_LIBRARY"), seed=0,
            backend="inline", episodes=2,
            cap_steps=80, onset_cycle=1, dirs="fwd,down", mags="1.0", tensor_policy=True)
        predict = make_predict(None, torch.device(a.device))
        batched = run_batched(a, predict)
        for i, direction in enumerate(("fwd", "down")):
            serial, gates = run_cell(a, predict, a.seed, direction, 1.0, i)
            self.assertFalse(gates["window_complete"])
            self.assertFalse(batched[i]["gates"]["window_complete"])
            self.assertEqual(serial["completed"], 0)
            self.assertEqual(batched[i]["completed"], 0)
            for key in ("mean_onset_s", "onset_min_s", "onset_max_s",
                        "kicks_per_world_max", "falls_pre_onset", "survived"):
                self.assertEqual(serial[key], batched[i][key], key)
        a.cap_steps = 650
        full = run_batched(a, predict)
        for row in full:
            self.assertEqual(row["completed"], a.episodes)
            self.assertEqual(row["incomplete"], 0)
            self.assertTrue(all(row["gates"].values()), row["gates"])


if __name__ == "__main__":
    unittest.main()
