# Policy-free controller replay

The reference is ULA commit `29c3dfccde133ffe3bbf56608bd4b0a96777de2e`.
Build it in a scratch checkout with a pinned Go2 model and an explicit RL
WBIC configuration. Never use or edit the live ULA working tree. Keep the
verified frozen trace at `data/go2_reference.jsonl` (ignored, but inside the
repository so it survives `/tmp` cleanup); scratch work may use `/tmp`. The
frozen trace's SHA-256 is
`5e2e59f52b55983f8d61dbd2ea68f149026bdf34b791d84c37a89db7ddc55a24`, so a
regeneration can be checked rather than trusted.

## Trace contract, version 1

Use newline-delimited JSON. The first record is a manifest:

```json
{"type":"manifest","schema":"ula-controller-replay","version":1,"reference_commit":"29c3dfccde133ffe3bbf56608bd4b0a96777de2e","dt_s":0.002,"scene_sha256":"...","urdf_sha256":"...","config_sha256":"..."}
```

The hashes refer to the MJCF scene entry file, controller URDF, and a
documented effective configuration snapshot, respectively. Record the actual
config paths, seed, and any overrides alongside the trace. The test uses
MuJoCo's CPU simulator only as a source of plant inputs, not as an oracle for
the new controller's closed-loop trajectories.

There are two subsequent record types:

- `{"type":"reset","episode":0}` starts a fresh episode. Count reset
  priming ticks explicitly; don't seed the new controller from the reference's
  gait or swing state. The Go2 recorder emits `priming_ticks: 6` and includes
  those ticks at indices 0 through 5 with `input.action_count: 0`.
- A `tick` contains `episode`, `tick` (zero-based within the episode), `input`,
  and `output`. Capture input at the stage that consumes it, not simply by
  reading the simulator state after `stepFrame()`.

Required full-controller inputs are `action` (25 held normalized actions),
`base_command` (forward velocity, lateral velocity, yaw rate), `position_w`
(3), `orientation_wxyz` (4), `linear_velocity_b` (3),
`angular_velocity_b` (3), `joint_position` (12), `joint_velocity` (12),
and `measured_contact` (4). Use the estimator sample actually consumed by
the module, and record any extra sensor or timing inputs under additional
named fields. Body and world frames are marked `_b` and `_w`.

Required full-controller outputs are `gait_phase` (scalar in cycles),
`planned_contact` (4 booleans), `foot_position_target` (12),
`joint_position_target` (12), `joint_velocity_target` (12),
`feedforward_torque` (12), and `motor_torque` (12, Nm, at the final actuator
boundary, before MuJoCo applies the actuator's force range). ULA computes
the limited leg command from the joint measurement, then the simulator PD
callback reads actuator length and velocity before writing `ctrl`. Record
those as `actuator_position` and `actuator_velocity` (12 each). Record actual
torque limits, action overrides, task frames and
other stage diagnostics as additional fields when available. Joint and foot
vectors are leg-major `FR, FL, HR, HL`; each joint triple is `abad, hip,
knee`, with ULA controller signs. Convert MuJoCo joint/actuator order at the
boundary instead of assuming it matches.

If an initial recorder cannot observe a required field at the right point,
omit it and explicitly mark the manifest `partial: true`. A partial trace is
useful for development but cannot count as full controller parity.

## Comparison rules

Replay supplies identical plant input and commands. Each controller owns its
own persistent state across ticks. A candidate trace has the same input,
episode and tick identifiers and emits its own outputs. Reject a changed
input, missing/reordered tick or non-finite value. Compare contact decisions
exactly. Compare continuous outputs with named tolerances in the units above;
report the first differing tick/field and the worst error. Establish and freeze
acceptance tolerances from repeatable CPU measurements, not by widening them
until a port passes. Reset, moving gait, swing/stance transitions and torque
outputs are required before calling it full parity.

The existing `parity_dump` and `qp_check` are narrower WBIC and QP regression
checks. Keep them intact. No CUDA, Warp integration, or PPO changes are part
of this CPU milestone.

Use the dependency-free checker from the repo root:

```sh
python3 tools/controller_replay.py validate /tmp/opencode/reference.jsonl
python3 tools/controller_replay.py compare /tmp/opencode/reference.jsonl /tmp/opencode/candidate.jsonl
```

For incremental work, pass `--allow-partial --output-field gait_phase`
(repeat `--output-field` as needed). Selected fields must be present on every
tick of both traces. The checker still verifies identical inputs and resets,
but labels the result as a partial diagnostic, not full parity.

## Reproduce the Go2 CPU check

[`tools/reference/record.sh`](../tools/reference/record.sh) archives the pinned
ULA source and its pinned rtrobot and EnvPool submodules into `/tmp/opencode`.
It applies recorder instrumentation only there and uses the `ula:dev` Docker
image, an existing Bazel 9.2.0 binary, and its repository download cache. The
live working tree is not used as source. The reference runs the CPU MuJoCo Go2
scene with `envpool.toml`, random command sampling and external pushes disabled,
and scripted actions starting at rest, then moving forward and turning. The
output manifest identifies the scene, URDF, config and effective gains.

```sh
bash tools/reference/record.sh /path/to/ula /path/to/bazel-9.2.0 /path/to/bazel-repo-cache /tmp/opencode/ula-recording
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python3 tools/replay_cpu.py /tmp/opencode/ula-recording/reference-a.jsonl /tmp/opencode/ula-recording/candidate.jsonl
bash tools/reference/compare_cpu.sh /tmp/opencode/ula-recording/reference-a.jsonl /tmp/opencode/ula-recording/candidate.jsonl
```

Configure with `-DULA_REPLAY_REFERENCE=/absolute/path/reference-a.jsonl` to
include the real trace replay and comparison in CTest. For the float QP,
configure a separate build with `-DULA_QP_SOLVER=gi` and the same trace.

The candidate reads ULA's recorded estimator and plant samples, not any of its
gait, desired body motion, swing state or torque outputs. The planner consumes
the cached leg-frame foot positions (`foot_position_leg`), the supplied world
velocity and body-to-world rotation matrix (row-major) from the estimator.
WBIC consumes the older cached joint sample (`joint_position` and
`joint_velocity`). Torque limiting uses a fresher sample
(`leg_controller_joint_position` and `leg_controller_joint_velocity`), and
`motor_torque` is the actual `ctrl[]` write from the first simulator callback
of each controller tick. Later physics substeps and MuJoCo's force clamp are
not compared. `foot_position_target` is the command after the WBIC liftoff
override when WBIC runs, and the unmodified request on the first tick of each
reset, when ULA sends zero joint commands rather than running WBIC.

Two separate 412-tick captures were byte-identical. On those captures the
largest torque difference across both solvers was `1.001e-5 Nm`. The limits
in `compare_cpu.sh` were then frozen as absolute tolerances: gait phase,
contact, and WBIC startup decision exact; foot positions `2e-7 m`; joint
positions `1e-6 rad`; joint velocities `1e-5 rad/s`; both torques `3e-5 Nm`.

Both builds passed on a longer trace (`720 360`, 1,092 ticks including
priming and nine switches in the planned support legs). Maximum errors with
the fixed-storage float QP were `5.960e-8 m`
for requested foot position, `2.384e-7 rad` for joint position, `2.861e-6
rad/s` for joint velocity, and `1.144e-5 Nm` for both torque fields. All
discrete decisions and the gait phase matched exactly. These are same-input
CPU controller results for this Go2 profile, not closed-loop or CUDA results.
The CPU bridge still uses the existing dynamic-matrix WBIC code; the next port
must remove those allocations and replay the same inputs on CUDA.
