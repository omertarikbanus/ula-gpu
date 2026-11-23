// Plain numeric transport for replay_cpu.py. JSONL is validated by Python.
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include "port/cpu_controller.hh"
#include "port/full_config.hh"
#include "port/batch_backend.hh"

namespace {
template <class T, std::size_t N>
bool read(T (&v)[N]) {
  for (auto& x : v)
    if (!(std::cin >> x)) return false;
  return true;
}

template <class T>
void field(const char* name, const T* v, int n) {
  std::cout << ",\"" << name << "\":[";
  for (int i = 0; i < n; ++i) std::cout << (i ? "," : "") << v[i];
  std::cout << ']';
}

template <class T>
void field(const char* name, const T (&v)[4][3]) {
  T flat[12];
  for (int i = 0; i < 12; ++i) flat[i] = v[i / 3][i % 3];
  field(name, flat, 12);
}

bool read_tick(ula::port::CpuInput& in) {
  auto& p = in.planner;
  return (std::cin >> p.action_count) && read(p.action) && read(p.base_command) &&
         read(p.plant.position_w) && read(in.orientation_wxyz) &&
         read(in.linear_velocity_b) && read(in.angular_velocity_b) &&
         read(p.plant.rotation_wb[0]) && read(p.plant.rotation_wb[1]) &&
         read(p.plant.rotation_wb[2]) && read(p.plant.linear_velocity_w) &&
         (std::cin >> p.plant.yaw) && read(p.plant.foot_position_leg[0]) &&
         read(p.plant.foot_position_leg[1]) && read(p.plant.foot_position_leg[2]) &&
         read(p.plant.foot_position_leg[3]) && read(in.joint_position) && read(in.joint_velocity) &&
         read(in.motor_joint_position) && read(in.motor_joint_velocity) &&
         read(in.actuator_position) && read(in.actuator_velocity) && read(in.actuator_enabled);
}

template <class Out>
void emit(const Out& out) {
  const auto& gait = out.planner.gait;
  std::cout << "{\"gait_phase\":" << gait.gait_phase << std::boolalpha;
  std::cout << ",\"wbic_ran\":" << out.wbic_ran;
  field("planned_contact", gait.planned_contact, 4);
  const auto& command = out.planner.wbic;
  field("foot_position_target", command.foot.foot_position_target);
  field("foot_velocity_target", command.foot.foot_velocity_target);
  field("foot_acceleration_target", command.foot.foot_acceleration_target);
  field("footstep_planner_pFoot_des", out.planner.footstep.foot_position_target);
  field("fr_des", command.force_w);
  field("body_pos_des", command.position_w, 3);
  field("body_vel_des", command.velocity_w, 3);
  field("body_rpy_des", command.orientation_rpy, 3);
  field("joint_position_target", out.joint_position_target, 12);
  field("joint_velocity_target", out.joint_velocity_target, 12);
  field("feedforward_torque", out.feedforward_torque, 12);
  field("leg_controller_tau_ff_sent", out.motor_feedforward, 12);
  field("motor_torque", out.motor_torque, 12);
  std::cout << "}\n";
}

bool read_config(ula::port::CpuConfig& config) {
  auto& u = config.wbic;
  if (!(std::cin >> config.planner.gait.dt_s >> config.floating_base_weight >>
        u.body_height >> u.contact_mu) || !read(config.max_torque)) return false;
  for (auto* gain : {&u.Kp_body, &u.Kd_body, &u.Kp_ori, &u.Kd_ori, &u.Kp_foot,
                     &u.Kd_foot, &u.Kp_joint, &u.Kd_joint}) {
    for (int i = 0; i < 3; ++i) if (!(std::cin >> (*gain)[i])) return false;
  }
  return true;
}

// Controller: reset() and step(CpuInput) returning a CpuOutputT-like record.
template <class Controller>
int replay(Controller& controller, const char* name, bool report) {
  std::cout << std::setprecision(17);
  char type;
  int episode = -1, tick = 0, ticks = 0, solves = 0;
  while (std::cin >> type) {
    if (type == 'R') {
      // init already created a fresh controller, including stage state.
      if (++episode > 0) controller.reset();
      tick = 0;
      continue;
    }
    ula::port::CpuInput in;
    if (episode < 0 || type != 'T' || !read_tick(in)) {
      std::cerr << "invalid CPU replay transport record\n";
      return 2;
    }
    try {
      const auto out = controller.step(in);
      solves += out.wbic_ran;
      emit(out);
    } catch (const std::exception& e) {
      std::cerr << "episode " << episode << " tick " << tick << ": " << e.what() << '\n';
      return 1;
    }
    ++tick;
    ++ticks;
  }
  if (!std::cin.eof() || ticks == 0) return 2;
  if (report)
    std::cerr << name << ": " << ticks << " planner ticks, " << solves
              << " dynamics ticks, " << solves << " QP solves, " << ticks << " motor ticks\n";
  return 0;
}

}  // namespace

int run(int argc, char** argv) {
  const std::string_view mode = argc == 3 ? std::string_view(argv[2]) : std::string_view();
  const bool host_stages = mode == "--host-stages";
  const bool reference = mode == "--reference";
  const bool full = mode == "--full";
  const bool full_double = mode == "--full-double";
#ifdef ULA_CUDA_REPLAY
  const bool audit_cuda = mode == "--full-cuda-audit";
  const bool inline_cuda = mode == "--full-cuda-inline";
  const bool full_cuda = mode == "--full-cuda" || audit_cuda || inline_cuda;
  const bool bench_cuda = mode == "--cuda-bench";
#else
  const bool full_cuda = false, bench_cuda = false, inline_cuda = false;
#endif
  if (argc != 2 && !host_stages && !reference && !full && !full_double && !full_cuda && !bench_cuda) {
    std::cerr << "usage: " << argv[0]
#ifdef ULA_CUDA_REPLAY
              << " GO2_URDF [--full-cuda|--full-cuda-audit|--full-cuda-inline|--cuda-bench] (use tools/replay_cpu.py)\n";
#else
              << " GO2_URDF [--host-stages|--reference|--full|--full-double] (use tools/replay_cpu.py)\n";
#endif
    return 2;
  }
  std::unique_ptr<ula::port::ReplayStages> stages;
#ifdef ULA_CUDA_REPLAY
  if (host_stages || reference || full || full_double) {
    std::cerr << "CUDA replay does not fall back to CPU; use cpu_replay for host tests\n";
    return 2;
  }
  if (!full_cuda && !bench_cuda) stages = ula::port::make_cuda_replay_stages();
#else
  if (host_stages) stages = std::make_unique<ula::port::HostReplayStages>();
#endif
  std::cin.imbue(std::locale::classic());
  std::cout.imbue(std::locale::classic());
  ula::port::CpuConfig config;
  if (!read_config(config)) return 2;
  if (reference || full_double) {
    ula::port::ReferenceController controller;
    if (!controller.init(argv[1], config)) return 2;
    if (full_double) {
      ula::port::FullHostReplay<double> port(ula::port::pack_full_config(controller));
      return replay(port, "full-double", true);
    }
    return replay(controller, "reference-double", false);
  }
  ula::port::CpuController controller;
  if (!controller.init(argv[1], config, stages.get())) return 2;
#ifdef ULA_CUDA_REPLAY
  if (full_cuda) {
    // --full-cuda-inline runs the same 65-world replay pipeline through the
    // fused inline solver instead of cuBLAS/cuSOLVER, into its own candidate.
    ula::port::FullCudaReplay port(ula::port::pack_full_config(controller),audit_cuda,!inline_cuda);
    const int result=replay(port, inline_cuda ? "full-cuda-inline (fused inline, 65 replicated worlds)"
                                             : "full-cuda (cuBLAS/cuSOLVER, 65 replicated worlds)", true);
    if (result==0) port.report();
    return result;
  }
  if (bench_cuda) {
    std::vector<ula::port::CpuInput> inputs;
    char type;
    while (std::cin >> type) {
      if (type=='R') continue;
      ula::port::CpuInput in;
      if (type!='T' || !read_tick(in)) return 2;
      // Start with the controller already receiving an action. Reuse the
      // recorded inputs on-device, with different states across the batch.
      if (in.planner.action_count==ula::port::kPolicyActionDim) inputs.push_back(in);
    }
    if (!std::cin.eof() || inputs.empty()) return 2;
    ula::port::benchmark_cuda(ula::port::pack_full_config(controller),inputs);
    return 0;
  }
#endif
  if (full) {
    ula::port::FullHostReplay<float> port(ula::port::pack_full_config(controller));
    return replay(port, "full-host", true);
  }
  return replay(controller, stages ? stages->name() : "cpu", stages != nullptr);
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "replay setup failed: " << e.what() << '\n';
    return 1;
  }
}
