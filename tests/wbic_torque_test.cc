#include <cmath>
#include <iostream>

#include "qp/wbic_problem.hh"

int main() {
  using namespace ula::qp;
  using B = WbicTorqueProblem<float>;
  B in;
  for (int i = 0; i < B::Dof; ++i) {
    in.mass[i][i] = 1.0f;
    in.cost[i] = 1.0f;
  }
  in.acceleration_pre[0] = 2.0f;
  in.acceleration_pre[6] = 3.0f;
  in.mass[6][6] = 2.0f;
  in.coriolis[6] = 1.0f;
  in.gravity[6] = 2.0f;

  const auto check = [](bool ok, const char* message) {
    if (!ok) std::cerr << message << '\n';
    return ok;
  };
  const auto near = [](float a, float b) { return std::abs(a - b) < 1e-6f; };
  WbicTorqueSolution<float> sol;
  solve_wbic_torque(in, sol);
  if (!check(sol.qp.result.status == kOk && near(sol.acceleration[0], 0.0f) &&
             near(sol.acceleration[6], 3.0f) && near(sol.torque[0], 9.0f),
             "no-contact solve must cancel the base acceleration but keep the joint acceleration")) return 1;

  // One point contact. Ask for a normal force of 3 N, but cap it at 2 N.
  // That force acts on the first base component and first joint; the latter
  // needs 9 - 2 = 7 N*m after subtracting the support term.
  in.contact_rows = 3;
  in.inequality_rows = 2;
  in.desired_force[2] = 3.0f;
  in.contact_transpose[0][2] = 1.0f;
  in.contact_transpose[6][2] = 1.0f;
  in.force_constraint[0][2] = 1.0f;
  in.force_constraint[1][2] = -1.0f;
  in.force_lower_bound[1] = -2.0f;
  solve_wbic_torque(in, sol);
  if (!check(sol.qp.result.status == kOk && near(sol.force[2], 2.0f) &&
             near(sol.acceleration[0], 2.0f) && near(sol.torque[0], 7.0f),
             "contact solve must respect force limits and subtract the contact torque")) return 1;

  in.contact_rows = B::MaxContact + 1;
  solve_wbic_torque(in, sol);
  return check(sol.qp.result.status == kBadSize && sol.torque[0] == 0.0f,
               "out-of-range contact count must fail without reusing stale torque") ? 0 : 1;
}
