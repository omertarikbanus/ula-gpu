# PPO validation and single-seed training results

## Reference and protocol

The baseline is the final **ICRA Epsilon** policy, not Gamma5 or an intermediate
checkpoint. Each published policy used 85,065,728 interactions: 54 actor/critic
inputs, 25 actions and Tanh 256/128 networks. `ula_gpu/icra_inference.py` pins
the three weight/normalization pairs by SHA-256 and preserves float64
observation normalization before float32 actor inference.

Training and evaluation keep physics at **1 ms**, two substeps per 500 Hz
controller tick and five controller ticks per 100 Hz policy step. Controller,
physics, policy inference, buffers, normalization and PPO updates run on GPU.
Vendor LU/SVD remains the default; the measured training comparison selected
the replay-validated inline backend, KL stopping on, no asynchronous overlap.

The kick test commands 0.8 m/s forward and applies one body-frame additive
velocity kick after three complete gait cycles at a randomized phase.
Survival means no fall during the next **5.5 seconds** and a completed scoring
window. Six directions and two magnitudes (2.5/4.0 m/s), with 32 phase draws per
cell, give 384 episodes per policy. Every policy uses the same scorer, controller
build and per-cell onset-seed convention.

## Completed from-scratch comparison

All runs used **seed 0**, exactly **16,128 worlds** and **84,897,792 interactions**.
The shared budget is 658 eight-step buffers or 329 sixteen-step buffers.
Each trained independently from fresh weights and finished without a restart
on a Colab A100-SXM4-40GB.

| Rollout steps | Learning rate | Full training loop | Steps/s | Training kicks |
|---|---:|---:|---:|---:|
| 8 | 1e-5 | 1,768.79 s | 47,998 | 380,407 |
| 16 | 1e-5 | 1,762.10 s | 48,180 | 398,924 |
| 16 | 3e-5 | 1,753.77 s | 48,409 | 399,605 |

Rates include periodic checkpoint writes, not setup or post-training evaluation.
Sampled GPU utilization had a 100% median; peak sampled GPU memory was
2,506/2,590/2,590 MiB. Those measurements do not establish maximum world
capacity or full arithmetic saturation.

All 1,920 evaluation episodes completed their scoring windows, with exactly
one correctly sized kick and no pre-onset fall:

| Policy | 2.5 m/s | 4.0 m/s | All cells |
|---|---:|---:|---:|
| Untrained seed-0 control | 108/192 (56.3%) | 46/192 (24.0%) | 154/384 (40.1%) |
| 8 steps, LR 1e-5 | 191/192 (99.5%) | 130/192 (67.7%) | 321/384 (83.6%) |
| 16 steps, LR 1e-5 | 186/192 (96.9%) | 113/192 (58.9%) | 299/384 (77.9%) |
| 16 steps, LR 3e-5 | 192/192 (100%) | 132/192 (68.8%) | 324/384 (84.4%) |
| Published Epsilon seed 0, in Warp | 192/192 (100%) | 89/192 (46.4%) | 281/384 (73.2%) |

**Next candidate: 16 steps, LR 3e-5.** Its aggregate differs from the eight-step
run by only three episodes, not a statistically established ranking. It also
preserves nominal walking better: at a 0.7 m/s command, the three-second
normalized checkpoint check moved 2.23 m with no falls, versus 1.71 m for the
slower-gait eight-step policy and 2.13 m for the default-LR sixteen-step policy.
These are short checks, not comprehensive tracking tests.

Recovery is not uniformly better. The recommended policy survived only
19/32 (59.4%) downward 4 m/s kicks, versus 23/32 (71.9%) for both the untrained
control and published baseline. Upward 4 m/s kicks remain difficult (6/32).

This demonstrates improved kick survival after training **in Warp, for one
training seed**. It does not establish native-engine transfer or superiority
to the paper's multi-seed result. The aggregate weights the 12 cells equally,
not an unknown real-world kick distribution. The untrained control has
mean-zero/unit-variance normalization, so this compares the full learned
policy-plus-normalization stack, not a weights-only ablation.

## Retained evidence and reproduction

Reusable tools are indexed in [`../bench/README.md`](../bench/README.md).
Local generated data is ignored rather than committed:

- `bench/artifacts/checkpoints/`: one copy of each final policy and its hash.
- `bench/artifacts/training/`: final training JSON/CSV and resource samples.
- `bench/artifacts/evaluation/`: corrected per-cell scores, input digests,
  final job status and `learning_summary.json`.
- `bench/artifacts/history/measurements.json`: compact older probe measurements.

Run `python3 bench/summarize_learning.py` to regenerate the summary. It checks
equal budgets, common controller identity, final checkpoint/input hashes,
cell coverage and completed scoring windows. Old session logs, duplicate
checkpoints, replay output traces, embedded notebooks and upload archives
are disposable and are not retained.

`pilot_epsilon.py` trains or resumes a checkpoint; `eval_ours.py --batched`
evaluates project checkpoints, an untrained control or hash-pinned Epsilon.
`make_cuda_replay_tarball.sh` packages source and the explicitly supplied
reference trace, not local experiment outputs. `make_recovery_eval_bundle.py`
adds only the selected final policies and published baseline inputs.

## Safeguards and known differences

- All 16 local frozen CPU CTests and 21 Colab CPU/CUDA CTests passed.
  The existing `1.49e-8 m` pre-solver foot-target review flag against a zero
  CPU baseline remains; test success does not erase that review item.
- CPU/CUDA scoring regression compares serial/batched onset streams, rejects
  deliberately incomplete windows and accepts complete windows.
- The first evaluation used 740 steps, censoring all eight-step-policy windows
  and some sixteen-step windows. Its reported survival scores were invalid
  and were discarded. Corrected evaluation uses 1,200 steps and a mandatory
  `window_complete` gate. Evaluation applies saved observation normalization.
- Two earlier launch attempts left no downloaded checkpoint. The VM-loss
  cause was not established. The replacement runs the notebook in the
  foreground, relays actual child output and resource measurements, and
  downloads atomic, hash-verified snapshots while executing.
- Full training is gated on a verified local CUDA smoke checkpoint. Final
  artifacts are verified before releasing the GPU. Both completed runtimes
  were released; no active assignments remained.
- Checkpoints include optimizer, normalization, RNG and lifetime curriculum
  state. Resume checks configuration equality but starts fresh episodes:
  simulator/controller internals are not serialized and abandoned discounted
  returns are cleared. It is not bit-exact continuation.
- Custom PPO uses float32 RMS rather than SB3's float64 statistics and checks
  KL at epoch rather than minibatch boundaries. Short rollout horizons are
  a credit-assignment change, not recipe-equivalent geometry.
- Incremental training resets skip six-tick priming; initial/evaluation
  resets retain it. Native reset internals were not traced to prove
  equivalence. A four-world CUDA isolation follow-up does not resolve the
  original 64-world mismatch.
- Training omits native joint-sensor noise and uses the documented
  interaction-space velocity-kick curriculum, not the native internal clock.
  Recovery transfer still differs by direction: in the earlier 4 m/s grids,
  native-to-Warp left survival was 66.7% to 22.9%, down 18.8% to 66.0%.
  Noise and collision-shape explanations remain untested.

## Optimization findings worth keeping

CUDA-graph PPO updates reduced a 131,072-transition update from about
5.7 s to 0.8 s. Lean incremental resets removed a measured ~18% rollout cost.
In short A100 probes at 16,384 x 8, vendor achieved 30,601 steps/s,
inline-only 46,445, and inline plus overlap 46,907. Overlap added little and
introduced stale policy weights, so it was not selected.

These short probes were throughput evidence, not learning tests. The completed
comparison above supplies the longer-run measurement. Full-step graph capture
did not improve T4 throughput. A separate padded-SVD merging experiment was
never implemented: sequential nullspace dependencies were identified, but
neither universal impossibility nor an inline performance ceiling was proven.
Do not infer a tested memory limit from earlier linear capacity extrapolations.
