#ifndef ULA_GPU_BATCH_HH
#define ULA_GPU_BATCH_HH
#include "port/full_controller.hh"

namespace ula::port {
struct World {
  ControllerState state;
  float q[12], qd[12];
  FullWorkspace<float> work;
  FullOutput<float> out;
};
struct BatchTick {
  const float *qpos, *qvel, *action, *command, *noise;
  const bool *active;
  float *obs, *estimate;  // world velocity (3), then gait phase
  int *status;
  bool prime;
};

ULA_HD inline void reset_world(const FullConfig<float>& cfg, World& w, const float* q, const float* qd) {
  reset(w.state,cfg.planner);
  for (int i=0;i<12;++i) { w.q[i]=q[7+i]; w.qd[i]=qd[6+i]; w.out.motor[i]={}; }
}

ULA_HD inline CpuInput plant_input(const FullConfig<float>& cfg,World& w,const BatchTick& b,int id) {
  const float *q=b.qpos+19*id, *v=b.qvel+18*id, *noise=b.noise+id*9;
  CpuInput in;
  auto& p=in.planner;
  auto& plant=p.plant;
  p.action_count=b.prime ? 0 : kPolicyActionDim;
  for (int i=0;i<kPolicyActionDim;++i) p.action[i]=b.action[id*kPolicyActionDim+i];
  rigid::Vec<float,4> quat{{q[3],q[4],q[5],q[6]}};
  const auto truth=rigid::detail::body_rotation(quat);
  const auto& r=truth.data;
  float roll=std::atan2(r[1][2],r[2][2])+noise[0];
  float pitch=std::asin(detail::clamp_f(-r[0][2],-1,1))+noise[1];
  float yaw=std::atan2(r[0][1],r[0][0])+noise[2];
  if (noise[0] || noise[1] || noise[2]) {
    const float cr=std::cos(roll/2), sr=std::sin(roll/2), cp=std::cos(pitch/2), sp=std::sin(pitch/2);
    const float cy=std::cos(yaw/2), sy=std::sin(yaw/2);
    quat={{cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy}};
  }
  const auto rot=rigid::detail::body_rotation(quat);
  for (int i=0;i<4;++i) in.orientation_wxyz[i]=quat.data[i];
  plant.yaw=yaw;
  for (int i=0;i<3;++i) {
    plant.position_w[i]=q[i];
    p.base_command[i]=b.command[3*id+i];
    plant.linear_velocity_w[i]=v[i]+noise[3+i];
    in.angular_velocity_b[i]=v[3+i]+noise[6+i];
    for (int j=0;j<3;++j) plant.rotation_wb[i][j]=rot.data[j][i];
  }
  detail::mat3_transpose_mul_vec3(plant.rotation_wb,plant.linear_velocity_w,in.linear_velocity_b);
  for (int i=0;i<12;++i) {
    in.joint_position[i]=w.q[i]; in.joint_velocity[i]=w.qd[i];
    in.motor_joint_position[i]=w.q[i]=q[7+i];
    in.motor_joint_velocity[i]=w.qd[i]=v[6+i];
    in.actuator_enabled[i]=false;  // torque goes to Warp separately at every substep
  }
  for (int l=0;l<4;++l) {
    const float cached[3]{in.joint_position[3*l],in.joint_position[3*l+1],in.joint_position[3*l+2]};
    compute_leg_foot_position(cfg.planner.footstep.kinematics,l,cached,plant.foot_position_leg[l]);
  }
  float* obs=b.obs+54*id;
  for (int i=0;i<3;++i) {
    obs[1+i]=in.linear_velocity_b[i]; obs[4+i]=in.angular_velocity_b[i];
    b.estimate[4*id+i]=plant.linear_velocity_w[i];
  }
  obs[7]=roll; obs[8]=pitch; obs[9]=yaw;
  for (int l=0;l<4;++l) {
    float foot[3],hip[3];
    const float fresh[]{w.q[3*l],w.q[3*l+1],w.q[3*l+2]};
    compute_leg_foot_position(cfg.planner.footstep.kinematics,l,fresh,foot);
    get_hip_location(cfg.planner.footstep.kinematics,l,hip);
    for (int j=0;j<3;++j) {
      const int i=3*l+j;
      obs[10+i]=w.q[i]; obs[22+i]=w.qd[i]; obs[38+i]=foot[j]+hip[j];
    }
  }
  obs[52]=q[2]; obs[53]=cfg.planner.body_height;
  return in;
}

ULA_HD inline void plant_output(const World& w,const BatchTick& b,int id) {
  b.status[id]=w.out.status;
  float* obs=b.obs+54*id;
  obs[0]=w.out.planner.decoded.desired_velocity[0];
  float phase=0;
  for (int l=0;l<4;++l) {
    obs[34+l]=w.out.planner.gait.planned_contact[l] ? 1.f : 0.f;
    phase+=w.out.planner.gait.swing_phase[l];
  }
  phase*=6.28318530717958647692f/4.f;
  obs[50]=std::sin(phase); obs[51]=std::cos(phase);
  b.estimate[4*id+3]=w.out.planner.gait.gait_phase;
}

ULA_HD inline void plant_tick(const FullConfig<float>& cfg,World& w,const BatchTick& b,int id,
                              InlineSolverScratch<float>& scratch) {
  if (b.active && !b.active[id]) return;
  const auto in=plant_input(cfg,w,b,id);
  full_tick(cfg,w.state,in,w.out,w.work,scratch);
  plant_output(w,b,id);
}
}
#endif
