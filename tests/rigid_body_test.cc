#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>

#include "dynamics/ParseURDFtoQuadruped.h"
#include "port/rigid_body_adapter.hh"
#ifdef ULA_CUDA_DYNAMICS_TEST
#include "port/replay_stages.hh"
#endif

namespace {

struct Error {
  const char* name;
  float limit;
  float worst = 0;
  int sample = 0, index = 0;

  void add(float measured, float expected, int s, int i) {
    const float distance = std::isfinite(measured) && std::isfinite(expected)
                               ? std::abs(measured - expected) : INFINITY;
    if (distance > worst) {
      worst = distance;
      sample = s;
      index = i;
    }
  }
};

}  // namespace

int run(int argc, char** argv) {
  if (argc != 2) return 2;
  bool loaded = false;
  auto quad = ParseURDFtoQuadruped<float>(argv[1], RobotType::MINI_CHEETAH, &loaded);
  if (!loaded) return 2;
  auto reference = quad.buildModel();
  const auto model = ula::rigid::pack_model(reference);
#ifdef ULA_CUDA_DYNAMICS_TEST
  const auto device = ula::port::make_cuda_replay_stages();
  device->set_model(model);
#endif
  std::mt19937 rng(533);
  const auto noise = [&rng](float range) {
    return std::uniform_real_distribution<float>(-range, range)(rng);
  };

  std::array<Error, 7> err{{{"mass matrix", 2e-6f}, {"gravity", 2e-5f}, {"Coriolis", 2e-5f},
       {"foot position", 3e-6f}, {"foot velocity", 5e-6f}, {"foot Jacobian", 1e-6f}, {"foot bias", 2e-5f}}};
  for (int sample = 0; sample < 256; ++sample) {
    FBModelState<float> state;
    const Vec3<float> rpy(noise(.45f), noise(.45f), noise(3.14f));
    state.bodyOrientation = ori::rpyToQuat(rpy);
    state.bodyPosition << noise(2.f), noise(2.f), .32f + noise(.12f);
    for (int a = 0; a < 6; ++a) state.bodyVelocity[a] = noise(2.f);
    state.q.resize(12);
    state.qd.resize(12);
    for (int i = 0; i < 12; ++i) {
      constexpr float crouch[] = {0.f, -.7752f, 1.5504f};
      state.q[i] = crouch[i % 3] + noise(.55f);
      state.qd[i] = noise(4.f);
    }
    reference.setState(state);
    reference.contactJacobians();
    reference.massMatrix();
    reference.generalizedGravityForce();
    reference.generalizedCoriolisForce();

    ula::rigid::Result<float> result;
    const auto packed = ula::rigid::pack_state(state);
#ifdef ULA_CUDA_DYNAMICS_TEST
    device->dynamics(packed, result);
#else
    ula::rigid::calculate(model, packed, result);
#endif
    for (int i = 0; i < 18; ++i) {
      for (int j = 0; j < 18; ++j)
        err[0].add(result.mass.data[i][j], reference.getMassMatrix()(i, j), sample, i*18+j);
      err[1].add(result.gravity.data[i], reference.getGravityForce()[i], sample, i);
      err[2].add(result.coriolis.data[i], reference.getCoriolisForce()[i], sample, i);
    }
    for (int leg = 0; leg < 4; ++leg) {
      const auto idx = ula::rigid::foot_index[leg];
      const auto& foot = result.feet[leg];
      for (int i = 0; i < 3; ++i) {
        err[3].add(foot.position.data[i], reference._pGC[idx][i], sample, leg*3+i);
        err[4].add(foot.velocity.data[i], reference._vGC[idx][i], sample, leg*3+i);
        for (int j = 0; j < 18; ++j)
          err[5].add(foot.jacobian.data[i][j], reference._Jc[idx](i,j), sample, (leg*3+i)*18+j);
        err[6].add(foot.bias.data[i], reference._Jcdqd[idx][i], sample, leg*3+i);
      }
    }
  }
  bool ok = true;
  for (const auto& value : err) {
    std::printf("%s: max |error| %.9g (limit %.9g, sample %d, index %d)\n",
                value.name, value.worst, value.limit, value.sample, value.index);
    ok &= value.worst <= value.limit;
  }
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
