# Core status: WBC tick, parity dump, allocations, CUDA blockers

State of the standalone controller core after the first working WBC tick.
Numbers are from this laptop (GCC 13.3, Eigen system, float build, `-O2` unless
noted). Reproduce with the commands in each section.

## 1. The nondeterministic parse failure: NOT reproduced

What was tried (from `/home/tarik/ula-gpu`, `models/go2/go2.urdf`):

| Check | Result |
|---|---|
| 200 sequential runs at `-O0`, 200 at `-O2` | 0 failures |
| 1,500 parallel runs (`xargs -P12`) each of `-O0`, `-O2`, default build | 0 failures |
| 100 runs each with `MALLOC_PERTURB_` set (poisons fresh/freed heap) | 0 failures |
| ASan + UBSan (`-O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer`): `core_demo` and the full `parity_dump` sweep (384 WBC ticks) | no reports |
| `-O0` vs `-O2` vs ASan output of the sweep | bit-identical (same FNV hash) |
| valgrind | not installed |

So no defect was found in `ParseURDFtoQuadruped.h` or the model, and **no source
change was made for it**. Since the parse path is deterministic, `urdf parse
failed` earlier is most likely not memory garbage in this code. Candidates, in
order: the demo was run from a different working directory (the path is
relative, `models/go2/go2.urdf`, and `LoadFile` failure silently gives
`!robot -> ok=false`); or the URDF/model files were being rewritten by the other
agent while it ran. If it comes back, run the failing command under
`build-asan` and keep the stderr.

Latent hazards in the parser that are real but deterministic (not the intermittent bug),
for whoever hardens it: `stof(attr->Attribute(...))` on a missing attribute passes
a null `const char*` (UB / `std::logic_error`); `hip_2_joint->FirstChildElement("limit")`
and `"dynamics"` (lines 276, 283, 284) are dereferenced unchecked; the many
`...->FirstChildElement("inertial")->FirstChildElement("inertia")->Attribute()`
chains (lines 293-401) are unchecked; the abad/hip/knee/body link inertia reads
assume `bodyLink` non-null (guarded at 186 only for the mass).

**ula**: `src/controller/include/dynamics/ParseURDFtoQuadruped.h` is
byte-identical to this copy (diff empty), so every line above is the same there.
Call sites: `src/controller/src/modules/MdlLegController.cc:45` (checks `urdf_ok`
and wraps in try/catch), `MdlStateEstimator.cc:59`, and
`src/controller/src/dynamics/QuadrupedSingleton.cc:23,29`, which call the parser
**without the `ok` flag**, so a failed parse there silently yields the
uninitialised fallback `Quadruped` (the likeliest source of a "garbage `this`"
style crash if the `Singleton` path is used before `MdlLegController::init`).
Also `MdlLegController.cc:47-53` calls `_mgr->fatalError` in the catch block
without checking `_mgr` (it is checked just above in `if (_mgr && ...)` for the config
table, but not here). ula was not edited.

## 2. The demo: one real WBC tick

`./build/core_demo models/go2/go2.urdf` now builds the model, runs one
`LocomotionCtrl<float>::run`. Inputs mirror
ula: `LocomotionCtrlData` as `SupervisorStates` BalanceStand builds it
(`Fr_des.z = total_mass*9.81/4`, all four `contact_state = 1`, `pBody_des.z =
0.32`), `StateEstimate` with identity orientation at z = 0.32 m and zero
velocity, leg data q/qd, `MIT_UserParameters` defaults, floating-base weight 10
(the envpool training config). Stance joints come from a bisection so the feet
touch z = 0 under the hips: q = (0, -0.7752, 1.5504). Shared code is
`src/core/WbcHarness.hh`. No algorithm, gain or QP change.

```
model: nDof=18 nq=19 mass=6.921 hip_links=0.095/0.213/0.235
wbc: stance leg q = (0.0000, -0.7752, 1.5504), total mass 14.857 kg, Fz_des/leg 36.437 N
wbc FR: qDes 0.0000 -0.7752 1.5504 | qdDes 0 0 0 | tauFF  1.8908 -0.5095 -5.5088 | Fr -1.229 -0.000 35.432
wbc FL: ...                                          tauFF -1.8908 -0.5095 -5.5088 | Fr -1.229 -0.000 35.432
wbc HR: ...                                          tauFF  2.0831 -0.5355 -5.8188 | Fr -1.229 -0.000 37.311
wbc HL: ...                                          tauFF -2.0831 -0.5355 -5.8188 | Fr -1.229 -0.000 37.311
wbc: sum Fz = 145.487 N (weight 145.747 N), all finite = yes, checksum = 125.795593
model: max |H_ii| after massMatrix() = 14.857000
```

Sanity: the reaction forces sum to the body weight within 0.2 %, left/right
feed-forward torques are mirror images, all outputs finite. `desJpos` equals the
measured stance pose and `desJvel` is 0, as expected at rest. The demo's earlier
`max |H_ii| = 0` was the cache issue (`getMassMatrix()` before `massMatrix()`);
also fixed the demo's `state.q` size (12 leg joints, not 19).

## 3. Parity dump

`tools/parity_dump.cc`, target `parity_dump`:

```
./build/parity_dump models/go2/go2.urdf out.bin [n_states=128] [seed=12345]
```

Deterministic sweep: 128 states x 3 commands (stand, 0.5 m/s forward, 0.2 rad/s
yaw) = 384 records of float32 (77 inputs + 66 outputs), fixed `mt19937` seed.
States: joints = stance pose +- noise clamped inside the URDF limits, body
z in 0.26..0.36 m, roll/pitch +-0.15 rad, any yaw, velocities +-0.4,
contact pattern all-stance (half of the states) or one swing leg / a diagonal
pair (swing foot target under the hip, world frame). A fresh controller is built
per record, so records are independent. The record layout is documented at the
top of the file (header of eight uint32, then records; leg order FR FL HR HL).

```
parity_dump: 384 records (128 states x 3 cmds), 0 non-finite, seed 12345
parity_dump: file out.bin, 219680 bytes, fnv1a64 58a2b3ac27ab48cd, output sum 84481.053608
```

Same hash from `-O0`, `-O2` and ASan builds (outputs are bit-identical across
optimisation levels here). Max |tauFF| 17.0 Nm, max |Fr| 85.1 N, max |qddot|
745.5 (the base/joint acceleration slack in the QP, large only for the
swing-leg/yawing cases). A GPU build should match within a float tolerance, not
bitwise (different FMA/associativity, and the QP active set may differ on
degenerate cases: compare also the active-set choice if mismatches appear).
Cost: 384 ticks including re-parsing the URDF each time take 0.13 s in total.

## 4. Allocation audit

`tools/alloc_audit.cc`, target `alloc_audit` (hooks the C allocator, because
Eigen's aligned allocator calls `malloc` directly and bypasses `operator new`;
run `./build/alloc_audit models/go2/go2.urdf [site_depth]` for per-phase call
sites). Heap allocations per WBC tick, steady state (2nd tick), stance pose:

| Phase | Allocs | Bytes | Main source |
|---|---:|---:|---|
| model update (`_UpdateModel`) | 25 | 5,472 | 16 x 1,152 B in `contactJacobians()`, `_A = massMatrix()`, `_Ainv = _A.inverse()`, grav/coriolis temporaries: dynamic `DMat`/`DVec` |
| task / contact update | 9 | 108 | `std::vector<Task*>`/`<ContactSpec*>` re-fill after `clear()`, `(DVec)` temp in `setRFDesired` |
| KinWBC | 121 | 30,840 | `FindConfiguration`, `_BuildProjectionMatrix`, `pseudoInverse` (`JacobiSVD::allocate` + result matrices): dynamic Eigen |
| WBIC setup | 0 | 0 | |
| WBIC QP + torque | 94 | 30,282 | `_SetOptimizationSize` (dynamic Eigen resize each call), `MakeTorque` temporaries, `solve_quadprog` internals (`GVect/GMatr<double>` working arrays, 16 allocs / 7.4 kB), `pseudoInverse` |
| leg command | 0 | 0 | |
| **Total** | **249** | **66,702** | |

With one swing leg: 322 allocs / 74.8 kB (extra swing-task Jacobian and
projection work in KinWBC). The first tick adds lazy initialisation (279 allocs
/ 82.5 kB). By type, in steady state: dynamic Eigen `DVec`/`DMat` (including
`JacobiSVD`) ~ 75 %; QuadProg++ `GVect/GMatr` ~ 7 %; `std::vector` of
task/contact pointers < 1 % by bytes (a handful of calls). No hidden
`std::string`/map allocation was seen in the tick. Frees match allocations
(no leak per tick).

## 5. What blocks compiling this core for CUDA (ranked)

1. **Dynamic Eigen sizes everywhere in the hot path** (`DVec<T>`, `DMat<T>`: the
   mass matrix, contact Jacobians, KinWBC null-space projections, WBIC
   `_dyn_CE/_dyn_CI`, the whole `WBIC_ExtraData`, plus `std::vector<Vec3/Mat6>`
   in `FloatingBaseModel`). This is the root of the ~250 allocations per tick
   and of the per-call `resize`. Eigen's dynamic matrices cannot live in
   registers/shared memory and device-side `malloc` is slow and per-thread
   limited. *Remedy*: fix every size at compile time: the base+12 joints give
   n = 18, 4 feet give contact rows <= 12, tasks <= 2 + 4 x 3; use
   `Eigen::Matrix<float, 18, 18>` etc. and, for the variable
   contact/task counts, fixed-capacity matrices with an active-size integer
   (padding with zero rows), as the QP and KinWBC are already index-based.
   Replace `A.inverse()` and `pseudoInverse` by fixed-size solves/Cholesky
   (the mass matrix is SPD) and a fixed-iteration SVD or damped normal
   equations. Eigen compiles in CUDA device code for fixed sizes, but the
   `JacobiSVD` should be avoided.
2. **Heap allocation and `std::vector`** (above table; also `new`/`delete` in
   `WBC_Ctrl` and `LocomotionCtrl` constructors, and `std::vector<ContactSpec*>`
   / `<Task*>` rebuilt every tick). *Remedy*: a per-robot POD struct holding
   all state and workspaces (one allocation made once on the host, one slice per
   thread), `std::array`-style fixed-capacity lists with a count, no
   `new` after init.
3. **QuadProg++ (Goldfarb-Idnani) in `double`, heap-backed, with a
   data-dependent iteration count and `goto` control flow** (`QuadProg++.cc`:
   `iter` loops until the active set is optimal, throws on infeasible /
   linearly dependent constraints, `exit(-1)` on one path, `GMatr/GVect`
   allocate). Divergent per-thread iteration counts are acceptable on a GPU if
   bounded, but the doubles, throws and allocation are not. *Remedy*: the
   planned fixed-storage float port of the same dual active-set algorithm:
   template on (n=dim_opt, m_eq, m_ineq) maxima, workspaces as members, an
   iteration cap (e.g. 3 x (n + m)), return a status code instead of
   throwing, and fall back to the previous torque / zero correction when
   infeasible. Verify against `parity_dump` with the active-set sequence
   logged. (Tuning the QP is out of scope: same algorithm, different storage.)
4. **Virtual dispatch and heap-owned polymorphic objects**: `WBC_Ctrl` is
   abstract with virtual `_ContactTaskUpdate`; `Task<T>` and `ContactSpec<T>`
   are polymorphic, held through base-class pointers in vectors. CUDA allows
   virtual functions only for objects constructed on the device, and it is slow
   anyway. *Remedy*: there are only 4 concrete types (`BodyPosTask`,
   `BodyOriTask`, `LinkPosTask`, `SingleContact`) and one controller:
   store them by value in a fixed struct and call them directly (CRTP or plain
   functions); the list becomes a compile-time-known sequence with enable flags.
5. **Exceptions and I/O**: `std::stof` throws in the URDF parser, QuadProg++
   throws `std::logic_error`, `std::cout`/`printf` appear in 86 places under
   `src/controllers` and `src/dynamics` (mostly debug prints). Device code has
   no exceptions or `std::string`. *Remedy*: parse the URDF once on the host into
   a flat `ModelParams` POD (all inertias, joint origins, limits, gains), copy
   it to GPU constant memory, and delete I/O and `throw` from the device path
   (the QP returns a status).
6. **Globals, singletons and shared mutable state**: `QuadrupedSingleton`
   (excluded from this build), `std::mutex`/`unique_ptr` statics; inside the
   core, `FloatingBaseModel` caches (`_biasAccelerationsUpToDate`, `_H`, ...) are
   members and therefore per-robot-safe, and `WBC_Ctrl::_iter` is a member too. *Remedy*: keep
   all of it per-robot in the POD state; read-only model parameters in
   constant memory; nothing static.
7. **Double-only and mixed-precision code**: `WBIC_ExtraData::_Fr` etc. are
   `DVec<T>` but the QP buffers are `double` (`GMatr<double>`), `MIT_UserParameters`
   is `double`, kinematics (`QuadrupedKinematics`) is `Eigen::Vector3d`. *Remedy*:
   cast parameters to float once at upload; float QP (item 3); keep
   `QuadrupedKinematics` out of the kernel (it is not on the WBC path).
8. **Framework coupling that is already outside the core but must stay out**:
   modules/supervisor (`rtcore`), `LogServer`, the `ConfigTable`, threads.
   Nothing to do beyond not pulling it in; the core already builds without it.
9. **Branch divergence and numerical robustness** (not a compile blocker):
   contact-pattern-dependent task/contact lists and QP active sets diverge between
   threads in a warp. *Remedy*: group envs by gait phase if throughput suffers;
   keep masked fixed-size problems.

Suggested order: (1)+(2)+(4) as one refactor guided by `alloc_audit` reaching
zero allocations (the CPU build stays the parity oracle through
`parity_dump`), then (3), then (5)-(7).
