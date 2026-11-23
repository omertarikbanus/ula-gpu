# Benchmarks and experiments

Run commands from the repository root. This directory holds reusable source,
requirements and small notebook templates, not a permanent dump of GPU sessions.
One-off T4/A100 launchers from completed optimization probes have been retired.

## Tools

- Physics comparison: `mjx_bench.py`, `warp_bench.py`, `_common.py`.
- Controller validation: `run_cuda_replay.sh`, `compare_inline.sh`,
  `verify_warp_loop.py`; `run_go2_colab.py` combines replay and Warp checks.
- Training: `pilot_epsilon.py`; `smoke_ppo.py` checks the SB3 environment bridge.
- Frozen-policy checks: `compare_icra_policy.py`, `icra_walk_gpu.py`.
- Kick evaluation: `eval_ours.py` for project checkpoints or published Epsilon;
  `icra_kick_eval.py` is the shared protocol, `native_kick_eval.py` the native
  simulator counterpart.
- Profiling: `gpu_headroom.py`, `train_profile.py`.
- Colab: `colab_training_driver.py` owns a bounded official-CLI execution,
  downloads verified backups and releases the runtime. `launch_overnight.py`
  / `run_overnight.py` reproduce the fixed-seed training comparison;
  `launch_recovery_eval.py` / `run_recovery_eval.py` evaluate its saved policies.
- Packaging: `make_cuda_replay_tarball.sh`, `make_recovery_eval_bundle.py`.
  Optional notebook workflows use the small `colab_bench.ipynb` and
  `colab_cuda_replay.ipynb` templates. Generate embedded notebooks with
  `make_colab_cuda_replay_standalone.py`; do not keep embedded archives in source.
- Results: `summarize_learning.py` validates and summarizes the completed study.

Tests belong in `tests/`, including `test_kicks.py`, `test_training_job.py`
and `test_recovery_scoring.py`.

## Generated files

All new outputs default to, or should be explicitly written beneath,
`bench/artifacts/`, which is ignored by Git:

```
artifacts/
  checkpoints/    one copy of each of the three final policies and their hashes
  training/       final training JSON/CSV and resource measurements
  evaluation/     corrected kick scores, input hashes and learning_summary.json
  history/        compact measurements from older probes, without session logs
  bundles/        disposable upload archives (generate when needed)
  notebooks/      disposable self-contained notebooks (generate when needed)
  runs/           temporary downloads and logs for future Colab sessions
```

Keep final policies and compact evidence locally; remove downloaded snapshots,
duplicate checkpoints, obsolete logs and generated bundles after verification.
The retained source-level result and caveats are in
[`../docs/PPO_SMOKE.md`](../docs/PPO_SMOKE.md). Local artifacts are not bundled
into the source package; the recovery packager adds only explicitly selected
checkpoint inputs.

Regenerate the summary without duplicating checkpoint files:

```bash
python3 bench/summarize_learning.py
```

For a new Colab job, supply a fresh `--output bench/artifacts/runs/NAME`
directory to `colab_training_driver.py`. Do not provision a GPU just to test
packaging or cleanup.
