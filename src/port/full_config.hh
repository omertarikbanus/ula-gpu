#ifndef ULA_GPU_FULL_CONFIG_HH
#define ULA_GPU_FULL_CONFIG_HH
#include "port/cpu_controller.hh"
#include "port/full_controller.hh"
#include "port/rigid_body_adapter.hh"

namespace ula::port {
template<class T> FullConfig<T> pack_full_config(const CpuControllerT<T>& cpu) {
  const auto& cfg=cpu.config();
  const auto& p=cfg.wbic;
  FullConfig<T> out{};
  out.planner=cfg.planner;
  out.model=rigid::pack_model(cpu.model());
  out.mu=p.contact_mu; out.base_weight=cfg.floating_base_weight;
  for (int i=0;i<3;++i) {
    out.kp[0][i]=p.Kp_ori[i]; out.kd[0][i]=p.Kd_ori[i];
    out.kp[1][i]=p.Kp_body[i]; out.kd[1][i]=p.Kd_body[i];
    out.kp[2][i]=p.Kp_foot[i]; out.kd[2][i]=p.Kd_foot[i];
    out.kp_joint[i]=p.Kp_joint[i]; out.kd_joint[i]=p.Kd_joint[i];
    out.limit[i]=cfg.max_torque[i];
  }
  return out;
}

template<class T> class FullHostReplay {
 public:
  explicit FullHostReplay(const FullConfig<T>& cfg) : cfg_(cfg) { reset(); }
  void reset() { port::reset(state_,cfg_.planner); }
  FullOutput<T> step(const CpuInput& in) {
    FullOutput<T> out;
    full_tick(cfg_,state_,in,out,work_,scratch_);
    if (out.status) throw std::runtime_error("full controller status="+std::to_string(out.status));
    return out;
  }
 private:
  FullConfig<T> cfg_;
  ControllerState state_;
  FullWorkspace<T> work_{};
  InlineSolverScratch<T> scratch_{};
};
}
#endif
