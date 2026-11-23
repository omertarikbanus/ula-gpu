#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "port/replay_stages.hh"

namespace ula::port {
namespace {

void check(cudaError_t code, const char* operation) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

// Replay is synchronous by design. The slots are allocated once, and each
// result copy waits for the preceding kernel and checks execution errors.
template <class T>
class DeviceSlot {
 public:
  static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
  DeviceSlot() { check(cudaMalloc(&ptr, sizeof(T)), "cudaMalloc replay slot"); }
  ~DeviceSlot() { if (ptr) cudaFree(ptr); }
  DeviceSlot(const DeviceSlot&) = delete;
  DeviceSlot& operator=(const DeviceSlot&) = delete;

  void upload(const T& value) {
    check(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice), "upload replay input");
  }
  void download(T& value) const {
    check(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost), "download replay output");
  }
  T download() const {
    T value{};
    download(value);
    return value;
  }

  T* ptr = nullptr;
};

__global__ void reset_planner(ControllerState* state, const ControllerConfig* config) {
  ula::port::reset(*state, *config);
}

__global__ void plan_tick(ControllerState* state, const ControllerConfig* config,
                          const ControllerInput* in, ControllerOutput* out) {
  *out = ula::port::step(*state, *config, *in);
}

__global__ void solve_tick(const ReplayQp* in, ReplayQpSolution* out) {
  qp::solve_wbic_torque(*in, *out);
}

__global__ void dynamics_tick(const rigid::Model<float>* model, const rigid::State<float>* in,
                              rigid::Result<float>* out) {
  rigid::calculate(*model, *in, *out);
}

__global__ void act_tick(const MotorTickInput* in, MotorTickOutput* out) {
  *out = motor_tick(*in);
}

class CudaReplayStages final : public ReplayStages {
 public:
  const char* name() const override { return "cuda-hybrid"; }

  void reset(const ControllerConfig& config) override {
    config_.upload(config);
    reset_planner<<<1, 1>>>(state_.ptr, config_.ptr);
    check(cudaGetLastError(), "launch CUDA planner reset");
    check(cudaDeviceSynchronize(), "CUDA planner reset");
  }

  ControllerOutput plan(const ControllerInput& in) override {
    planner_in_.upload(in);
    plan_tick<<<1, 1>>>(state_.ptr, config_.ptr, planner_in_.ptr, planner_out_.ptr);
    check(cudaGetLastError(), "launch CUDA planner tick");
    return planner_out_.download();
  }

  void set_model(const rigid::Model<float>& model) override { model_.upload(model); }

  void dynamics(const rigid::State<float>& in, rigid::Result<float>& out) override {
    dynamics_in_.upload(in);
    dynamics_tick<<<1, 1>>>(model_.ptr, dynamics_in_.ptr, dynamics_out_.ptr);
    check(cudaGetLastError(), "launch CUDA dynamics tick");
    dynamics_out_.download(out);
  }

  void solve(const ReplayQp& in, ReplayQpSolution& out) override {
    qp_in_.upload(in);
    solve_tick<<<1, 1>>>(qp_in_.ptr, qp_out_.ptr);
    check(cudaGetLastError(), "launch CUDA QP solve");
    qp_out_.download(out);
  }

  MotorTickOutput act(const MotorTickInput& in) override {
    motor_in_.upload(in);
    act_tick<<<1, 1>>>(motor_in_.ptr, motor_out_.ptr);
    check(cudaGetLastError(), "launch CUDA motor tick");
    return motor_out_.download();
  }

 private:
  DeviceSlot<ControllerConfig> config_;
  DeviceSlot<ControllerState> state_;
  DeviceSlot<ControllerInput> planner_in_;
  DeviceSlot<ControllerOutput> planner_out_;
  DeviceSlot<rigid::Model<float>> model_;
  DeviceSlot<rigid::State<float>> dynamics_in_;
  DeviceSlot<rigid::Result<float>> dynamics_out_;
  DeviceSlot<ReplayQp> qp_in_;
  DeviceSlot<ReplayQpSolution> qp_out_;
  DeviceSlot<MotorTickInput> motor_in_;
  DeviceSlot<MotorTickOutput> motor_out_;
};

}  // namespace

std::unique_ptr<ReplayStages> make_cuda_replay_stages() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "CUDA device discovery");
  if (count < 1) throw std::runtime_error("no CUDA device is available");
  cudaDeviceProp props{};
  int device = 0;
  check(cudaGetDevice(&device), "get CUDA device");
  check(cudaGetDeviceProperties(&props, device), "get CUDA device properties");
  std::fprintf(stderr, "CUDA device %d: %s (sm_%d%d)\n", device, props.name, props.major, props.minor);
  return std::make_unique<CudaReplayStages>();
}

}  // namespace ula::port
