#ifndef ULA_GPU_FULL_CONTROLLER_HH
#define ULA_GPU_FULL_CONTROLLER_HH

#include "port/small_linalg.hh"
#include "port/tick_input.hh"
#include "actuation/sim_motor.hh"
#include "qp/wbic_problem.hh"

namespace ula::port {

template<class T> struct FullConfig {
  ControllerConfig planner;
  rigid::Model<T> model;
  T kp[3][3], kd[3][3];  // orientation, body position, swing foot
  T kp_joint[3], kd_joint[3], limit[3];
  T mu, base_weight;
};

template<class T> struct FullOutput {
  ControllerOutput planner;
  bool wbic_ran;
  int status;  // QP status, or -1 for invalid dynamics / task math
  T joint_position_target[12], joint_velocity_target[12], feedforward_torque[12];
  T motor_feedforward[12];
  double motor_torque[12];
  actuation::MotorCommandT<T> motor[12];
};

template<class T> struct FullWorkspace {
  struct Task { small::Mat<T> jac; T error[3], vel[3], acc[3], bias[3]; };
  Task tasks[6];
  int task_count;
  T contact_bias[12], delta[rigid::kDof], vel[rigid::kDof];
  rigid::Workspace<T> spatial;
  rigid::Result<T> dynamics;
  small::Mat<T> contact, inv_mass, nullspace, projected, pinv, correction, next, left, lambda, linv;
  qp::WbicTorqueProblem<T> problem;
  qp::WbicTorqueSolution<T> solution;
  qp::WbicWorkspace<T> qp;
};

// Kept for the host replay and as a benchmark baseline for the library path.
template<class T> struct InlineSolverScratch {
  small::Mat<T> lu;
  small::Svd<T> svd;
};

namespace full_detail {
constexpr int N = rigid::kDof;

template<class T> ULA_HD auto desired_quat(T yaw) {
  // ULA converts an RPY rotation matrix to a quaternion. Roll/pitch are zero.
  const T c=std::cos(yaw), s=std::sin(yaw), trace=(c+c)+T(1);
  if (trace>0) {
    const T d=T(2*std::sqrt(double(trace)+1));
    return rigid::Vec<T,4>{{T(0.25)*d,0,0,(s+s)/d}};
  }
  const T d=T(2*std::sqrt(2.0-double(c)-double(c)));
  return rigid::Vec<T,4>{{(s+s)/d,0,0,T(0.25)*d}};
}

template<class T> ULA_HD auto orientation_error(const rigid::Vec<T,4>& measured, T yaw) {
  const auto desired=desired_quat(yaw);
  const T* a=desired.data;
  const T b[]{-measured.data[1],-measured.data[2],-measured.data[3]};
  const auto cross=rigid::detail::cross(a+1,b);
  const T w=a[0]*measured.data[0]-rigid::detail::dot<3>(a+1,b);
  rigid::Vec<T,3> out;
  auto& v=out.data;
  for (int i=0;i<3;++i) v[i]=((a[0]*b[i]+measured.data[0]*a[i+1])+cross.data[i])*(w<0?-1:1);
  const double norm=std::sqrt(double(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]));
  const T theta=T(2*std::asin(norm));
  if (small::abs(theta)<1e-7) return rigid::Vec<T,3>{};
  const T factor=T(std::sin(double(theta)/2));
  for (int i=0;i<3;++i) v[i]=(v[i]/factor)*theta;
  return out;
}

inline constexpr double kKinematicCutoff = 0.001;
inline constexpr double kWeightedCutoff = 0.0001;

template<class T> ULA_HD const small::Mat<T>& inverse_input(bool weighted, const small::Mat<T>& j,
                                                           int rows, FullWorkspace<T>& w) {
  if (!weighted) return j;
  using namespace small;
  multiply(j,w.inv_mass,w.left,rows,N,N);
  multiply(w.left,j,w.lambda,rows,N,rows,true);
  return w.lambda;
}

template<class T> ULA_HD void finish_weighted_inverse(const small::Mat<T>& j, int rows,
                                                     FullWorkspace<T>& w) {
  using namespace small;
  multiply(w.inv_mass,j,w.left,N,N,rows,true);
  multiply(w.left,w.linv,w.pinv,N,rows,rows);
}

template<class T> ULA_HD void project(const small::Mat<T>& j, int rows, FullWorkspace<T>& w,
                                     bool initial) {
  small::multiply(w.pinv,j,w.correction,N,rows,N);
  auto correction=small::view(w.correction,N,N);
  correction=-correction;
  correction.diagonal().array()+=T(1);
  if (initial) w.nullspace=w.correction;
  else {
    small::multiply(w.nullspace,w.correction,w.next,N,N,N);
    w.nullspace=w.next;
  }
}

template<class T> ULA_HD const small::Mat<T>& hierarchy_inverse_input(bool weighted, int task,
                                                                     FullWorkspace<T>& w) {
  if (task < 0) return inverse_input(weighted,w.contact,w.problem.contact_rows,w);
  small::multiply(w.tasks[task].jac,w.nullspace,w.projected,3,N,N);
  return inverse_input(weighted,w.projected,3,w);
}

template<class T> ULA_HD void apply_hierarchy_inverse(bool weighted, int task, FullWorkspace<T>& w) {
  T* x = weighted ? w.problem.acceleration_pre : w.delta;
  const int rows = task < 0 ? w.problem.contact_rows : 3;
  const auto& j = task < 0 ? w.contact : w.projected;
  if (weighted) finish_weighted_inverse(j,rows,w);
  if (task < 0) {
    if (weighted)
      rigid::detail::vec<N>(x).noalias()=small::view(w.pinv,N,rows).lazyProduct(small::vec(w.contact_bias,rows));
    project(j,rows,w,true);
    return;
  }
  const auto& t=w.tasks[task];
  T ex[3], ev[3];
  for (int i=0;i<3;++i) {
    ex[i]=(weighted ? t.acc[i]-t.bias[i] : t.error[i])-small::vec(t.jac.data[i],N).dot(small::vec(x,N));
    if (!weighted) ev[i]=t.vel[i]-small::vec(t.jac.data[i],N).dot(small::vec(w.vel,N));
  }
  for (int i=0;i<N;++i) {
    x[i]+=rigid::detail::dot<3>(w.pinv.data[i],ex);
    if (!weighted) w.vel[i]+=rigid::detail::dot<3>(w.pinv.data[i],ev);
  }
  if (task+1<w.task_count) project(j,rows,w,false);
}

template<class T> ULA_HD bool hierarchy(bool weighted, FullWorkspace<T>& w, small::Svd<T>& scratch) {
  if (!w.problem.contact_rows) small::identity(w.nullspace,N,N);
  for (int task=w.problem.contact_rows ? -1 : 0; task<w.task_count; ++task) {
    const auto& a=hierarchy_inverse_input(weighted,task,w);
    const int rows=task < 0 ? w.problem.contact_rows : 3;
    if (!small::pinverse(a,rows,weighted ? rows : N,weighted ? kWeightedCutoff : kKinematicCutoff,
                        weighted ? w.linv : w.pinv,scratch)) return false;
    apply_hierarchy_inverse(weighted,task,w);
  }
  return true;
}

template<class T> ULA_HD void prepare_wbic(const FullConfig<T>& cfg,const CpuInput& in,
                                          const FullOutput<T>& out,FullWorkspace<T>& w) {
  rigid::State<T> s;
  for (int i=0;i<4;++i) s.quaternion.data[i]=in.orientation_wxyz[i];
  for (int i=0;i<3;++i) {
    s.position.data[i]=in.planner.plant.position_w[i];
    s.velocity.data[i]=in.angular_velocity_b[i];
    s.velocity.data[i+3]=in.linear_velocity_b[i];
  }
  for (int i=0;i<12;++i) { s.q[i]=in.joint_position[i]; s.qd[i]=in.joint_velocity[i]; }
  rigid::calculate(cfg.model,s,w.dynamics,w.spatial);

  const auto& target=out.planner.wbic;
  const auto rot=rigid::detail::trans(rigid::detail::body_rotation(s.quaternion));
  const auto err=orientation_error(s.quaternion,T(target.orientation_rpy[2]));
  rigid::Vec<T,3> omega_err, vb;
  for (int i=0;i<3;++i) {
    omega_err.data[i]=T(target.angular_velocity[i])-s.velocity.data[i];
    vb.data[i]=s.velocity.data[i+3];
  }
  const auto dw=rigid::detail::mul(rot,omega_err), vw=rigid::detail::mul(rot,vb);
  for (int k=0;k<2;++k) {
    auto& t=w.tasks[k];
    for (int i=0;i<3;++i) {
      for (int j=0;j<N;++j) t.jac.data[i][j]=(j/3==k) ? rot.data[i][j%3] : T(0);
      t.error[i]=k==0 ? err.data[i] : T(target.position_w[i])-s.position.data[i];
      t.vel[i]=k==0 ? T(target.angular_velocity[i]) : T(target.velocity_w[i]);
      t.bias[i]=0;
      const T dv=k==0 ? dw.data[i] : t.vel[i]-vw.data[i];
      t.acc[i]=cfg.kp[k][i]*t.error[i]+cfg.kd[k][i]*dv;
    }
  }
  int nc=0,nt=2;
  auto& p=w.problem;
  for (int a=0;a<24;++a) for (int i=0;i<12;++i) p.force_constraint[a][i]=0;
  for (int l=0;l<4;++l) {
    const auto& f=w.dynamics.feet[l];
    if (out.planner.gait.planned_contact[l]) {
      for (int i=0;i<3;++i) {
        p.desired_force[nc+i]=target.force_w[l][i];
        w.contact_bias[nc+i]=-f.bias.data[i];
        for (int j=0;j<N;++j) p.contact_transpose[j][nc+i]=w.contact.data[nc+i][j]=f.jacobian.data[i][j];
      }
      const int b=2*nc;
      const T constraints[6][3]{{0,0,1},{1,0,cfg.mu},{-1,0,cfg.mu},{0,1,cfg.mu},{0,-1,cfg.mu},{0,0,-1}};
      for (int a=0;a<6;++a) {
        p.force_lower_bound[b+a]=a==5 ? T(-650) : T(0);
        for (int i=0;i<3;++i) p.force_constraint[b+a][nc+i]=constraints[a][i];
      }
      nc+=3;
    } else {
      auto& t=w.tasks[nt++];
      for (int i=0;i<3;++i) {
        for (int j=0;j<N;++j) t.jac.data[i][j]=f.jacobian.data[i][j];
        t.error[i]=T(target.foot.foot_position_target[l][i])-f.position.data[i];
        t.vel[i]=target.foot.foot_velocity_target[l][i];
        t.bias[i]=f.bias.data[i];
        t.acc[i]=(cfg.kp[2][i]*t.error[i]+cfg.kd[2][i]*(t.vel[i]-f.velocity.data[i]))
            +T(target.foot.foot_acceleration_target[l][i]);
      }
    }
  }
  w.task_count=nt;
  p.contact_rows=nc; p.inequality_rows=2*nc;
  for (int i=0;i<N;++i) {
    w.delta[i]=w.vel[i]=0;
    p.acceleration_pre[i]=0; p.cost[i]=i<6 ? cfg.base_weight : T(1);
    p.gravity[i]=w.dynamics.gravity.data[i]; p.coriolis[i]=w.dynamics.coriolis.data[i];
    for (int j=0;j<N;++j) p.mass[i][j]=w.dynamics.mass.data[i][j];
  }
}

template<class T> ULA_HD int finish_wbic(const CpuInput& in, FullOutput<T>& out, FullWorkspace<T>& w) {
  qp::solve_wbic_torque(w.problem,w.solution,w.qp);
  const int status=w.solution.qp.result.status;
  if (status!=qp::kOk) return status;
  if (!std::isfinite(w.solution.qp.result.f)) return -1;
  for (int i=0;i<12;++i) {
    const T knee=in.joint_position[i];
    T q=T(in.joint_position[i])+w.delta[6+i], tau=w.solution.torque[i];
    if (i%3==2) {
      if (q<0.3) q=T(0.3);
      if (knee<0.3) tau=T(1.0/(knee*knee+0.02));
    }
    if (!std::isfinite(q)||!std::isfinite(tau)||!std::isfinite(w.vel[6+i])) return -1;
    out.joint_position_target[i]=q;
    out.joint_velocity_target[i]=w.vel[6+i];
    out.feedforward_torque[i]=tau;
  }
  return 0;
}
}  // namespace full_detail

template<class T> ULA_HD void begin_tick(const FullConfig<T>& cfg,ControllerState& state,
                                        const CpuInput& in,FullOutput<T>& out,FullWorkspace<T>& w) {
  out.planner=step(state,cfg.planner,in.planner);
  out.wbic_ran=out.planner.wbic.run_wbic;
  out.status=0;
  if (out.wbic_ran) full_detail::prepare_wbic(cfg,in,out,w);
}

template<class T> ULA_HD void finish_tick(const FullConfig<T>& cfg,const CpuInput& in,
                                         FullOutput<T>& out,FullWorkspace<T>& w) {
  if (out.wbic_ran && out.status==0) out.status=full_detail::finish_wbic(in,out,w);
  const bool ready=out.wbic_ran && out.status==0;
  for (int l=0;l<4;++l) {
    actuation::LegMeasurementT<T> meas{};
    actuation::LegTargetT<T> cmd{};
    for (int j=0;j<3;++j) {
      const int i=3*l+j;
      meas.joint_position[j]=in.motor_joint_position[i];
      meas.joint_velocity[j]=in.motor_joint_velocity[i];
      if (ready) {
        cmd.joint_position[j]=out.joint_position_target[i];
        cmd.joint_velocity[j]=out.joint_velocity_target[i];
        cmd.torque_ff[j]=out.feedforward_torque[i];
        cmd.joint_kp[j][j]=cfg.kp_joint[j]; cmd.joint_kd[j][j]=cfg.kd_joint[j];
      } else out.joint_position_target[i]=out.joint_velocity_target[i]=out.feedforward_torque[i]=0;
    }
    const auto m=actuation::prepare_leg_command(meas,cmd,cfg.limit);
    for (int j=0;j<3;++j) {
      const int i=3*l+j;
      out.motor[i]=m.command[j];
      out.motor_feedforward[i]=out.motor[i].feedforward;
      out.motor_torque[i]=actuation::sim_motor_torque(out.motor[i],in.actuator_position[i],in.actuator_velocity[i],in.actuator_enabled[i]);
    }
  }
}

template<class T> ULA_HD void full_tick(const FullConfig<T>& cfg,ControllerState& state,
                                       const CpuInput& in,FullOutput<T>& out,FullWorkspace<T>& w,
                                       InlineSolverScratch<T>& scratch) {
  begin_tick(cfg,state,in,out,w);
  if (out.wbic_ran) {
    scratch.lu=w.dynamics.mass;
    if (!small::invert(scratch.lu,w.inv_mass,rigid::kDof) ||
        !full_detail::hierarchy(false,w,scratch.svd) ||
        !full_detail::hierarchy(true,w,scratch.svd)) out.status=-1;
  }
  finish_tick(cfg,in,out,w);
}
}
#endif
