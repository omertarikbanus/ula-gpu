#ifndef ULA_GPU_PORT_RIGID_BODY_HH
#define ULA_GPU_PORT_RIGID_BODY_HH

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#ifdef __CUDACC__
#define ULA_RIGID_HD __host__ __device__
#else
#define ULA_RIGID_HD
#endif

namespace ula::rigid {

// Same generalized coordinate order as ULA: angular base, linear base, then
// twelve joints. Body indices 5..17 match the original model's tree indices.
inline constexpr int kBase = 5, kFirstJoint = 6, kDof = 18, kFeet = 4;

template <class T, int N> struct Vec { T data[N]{}; };
template <class T, int R, int C> struct Mat { T data[R][C]{}; };

template <class T> struct Model {
  Mat<T, 6, 6> body_inertia[kDof]{}, rotor_inertia[kDof]{};
  Mat<T, 6, 6> tree[kDof]{}, rotor_tree[kDof]{};
  T gear[kDof]{};
  int parent[kDof]{}, axis[kDof]{};
  Vec<T, 3> gravity{}, foot_local[kFeet]{};
  int foot_parent[kFeet]{};
};

template <class T> struct State {
  Vec<T, 4> quaternion{};  // wxyz, use ULA's rounded quaternion without renormalizing
  Vec<T, 3> position{};
  Vec<T, 6> velocity{};    // body frame, angular then linear
  T q[12]{}, qd[12]{};
};

template <class T> struct Foot {
  Vec<T, 3> position{}, velocity{}, bias{};
  Mat<T, 3, kDof> jacobian{};
};

template <class T> struct Result {
  Mat<T, kDof, kDof> mass{};
  Vec<T, kDof> gravity{}, coriolis{};
  Foot<T> feet[kFeet]{};  // FR, FL, HR, HL only (unused collision points omitted)
};

template <class T> struct Dispatch {
  void* context{};
  void (*run)(void*, const State<T>&, Result<T>&){};
};

namespace detail {

template <class T, int R, int C>
using EigenMat = Eigen::Matrix<T, R, C, C == 1 ? Eigen::ColMajor : Eigen::RowMajor>;

template <class T, int R, int C>
ULA_RIGID_HD inline auto view(const Mat<T, R, C>& a) {
  return Eigen::Map<const EigenMat<T, R, C>>(&a.data[0][0]);
}

template <class T, int R, int C>
ULA_RIGID_HD inline auto view(Mat<T, R, C>& a) {
  return Eigen::Map<EigenMat<T, R, C>>(&a.data[0][0]);
}

template <int N, class T>
ULA_RIGID_HD inline auto vec(const T* a) { return Eigen::Map<const EigenMat<T, N, 1>>(a); }

template <int N, class T>
ULA_RIGID_HD inline auto vec(T* a) { return Eigen::Map<EigenMat<T, N, 1>>(a); }

template <int N, class T> ULA_RIGID_HD inline T dot(const T* a, const T* b) {
  return vec<N>(a).dot(vec<N>(b));
}

template <class T, int R, int C>
ULA_RIGID_HD inline Mat<T, C, R> trans(const Mat<T, R, C>& a) {
  Mat<T, C, R> b;
  view(b) = view(a).transpose();
  return b;
}

template <class T, int R, int N, int C>
ULA_RIGID_HD inline Mat<T, R, C> mul(const Mat<T, R, N>& a, const Mat<T, N, C>& b) {
  Mat<T, R, C> out;
  view(out).noalias() = view(a).lazyProduct(view(b));
  return out;
}

template <class T, int R, int C>
ULA_RIGID_HD inline Vec<T, R> mul(const Mat<T, R, C>& a, const Vec<T, C>& b) {
  Vec<T, R> out;
  vec<R>(out.data).noalias() = view(a).lazyProduct(vec<C>(b.data));
  return out;
}

template <class T, int N> ULA_RIGID_HD inline void add(Vec<T, N>& a, const Vec<T, N>& b) {
  vec<N>(a.data) += vec<N>(b.data);
}

template <class T, int R, int C> ULA_RIGID_HD inline void add(Mat<T, R, C>& a, const Mat<T, R, C>& b) {
  view(a) += view(b);
}

template <class T> ULA_RIGID_HD inline Mat<T, 3, 3> body_rotation(const Vec<T, 4>& q) {
  const T w = q.data[0], x = q.data[1], y = q.data[2], z = q.data[3];
  return {{{1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)},
           {2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)},
           {2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)}}};
}

template <class T> ULA_RIGID_HD inline Mat<T, 3, 3> joint_rotation(int axis, T angle) {
  const T s = std::sin(angle), c = std::cos(angle);
  if (axis == 0) return {{{1, 0, 0}, {0, c, s}, {0, -s, c}}};
  if (axis == 1) return {{{c, 0, -s}, {0, 1, 0}, {s, 0, c}}};
  return {{{c, s, 0}, {-s, c, 0}, {0, 0, 1}}};
}

template <class T> ULA_RIGID_HD inline Mat<T, 3, 3> rotation(const Mat<T, 6, 6>& x) {
  Mat<T, 3, 3> r;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r.data[i][j] = x.data[i][j];
  return r;
}

template <class T> ULA_RIGID_HD inline Mat<T, 6, 6> transform(const Mat<T, 3, 3>& r, const Vec<T, 3>& p) {
  const auto* v = p.data;
  const Mat<T, 3, 3> skew = {{{0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0}}};
  const auto tr = mul(r, skew);
  Mat<T, 6, 6> out;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    out.data[i][j] = out.data[i+3][j+3] = r.data[i][j];
    out.data[i+3][j] = -tr.data[i][j];
  }
  return out;
}

template <class T> ULA_RIGID_HD inline Vec<T, 3> translation(const Mat<T, 6, 6>& x) {
  Mat<T, 3, 3> lower;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) lower.data[i][j] = x.data[i+3][j];
  const auto a = mul(trans(rotation(x)), lower);
  return {{T(-0.5) * (a.data[2][1] - a.data[1][2]), T(-0.5) * (a.data[0][2] - a.data[2][0]),
           T(-0.5) * (a.data[1][0] - a.data[0][1])}};
}

template <class T> ULA_RIGID_HD inline Vec<T, 3> cross(const T* a, const T* b) {
  Vec<T, 3> out;
  vec<3>(out.data) = vec<3>(a).cross(vec<3>(b));
  return out;
}

template <class T> ULA_RIGID_HD inline Vec<T, 6> motion_cross(const Vec<T, 6>& av, const Vec<T, 6>& bv) {
  const auto* a = av.data;
  const auto* b = bv.data;
  const auto w = cross(a, b);
  return {{w.data[0], w.data[1], w.data[2],
           a[1]*b[5]-a[2]*b[4]+a[4]*b[2]-a[5]*b[1],
           a[2]*b[3]-a[0]*b[5]-a[3]*b[2]+a[5]*b[0],
           a[0]*b[4]-a[1]*b[3]+a[3]*b[1]-a[4]*b[0]}};
}

template <class T> ULA_RIGID_HD inline Vec<T, 6> force_cross(const Vec<T, 6>& av, const Vec<T, 6>& bv) {
  const auto* a = av.data;
  const auto* b = bv.data;
  const auto f = cross(a, b + 3);
  return {{b[2]*a[1]-b[1]*a[2]-b[4]*a[5]+b[5]*a[4],
           b[0]*a[2]-b[2]*a[0]+b[3]*a[5]-b[5]*a[3],
           b[1]*a[0]-b[0]*a[1]-b[3]*a[4]+b[4]*a[3], f.data[0], f.data[1], f.data[2]}};
}

}  // namespace detail

// Direct port of the floating-base model passes used by WBIC. Preserve the
// rotor terms and carry the same frames as ULA; this is not simulator dynamics.
template <class T> struct Workspace {
  Mat<T, 6, 6> up[kDof], rotor[kDof], absolute[kDof], inertia[kDof];
  Vec<T, 6> v[kDof], bias[kDof], f[kDof], rotor_f[kDof], grav[kDof];
};

template <class T>
ULA_RIGID_HD inline void calculate(const Model<T>& model, const State<T>& s, Result<T>& out,
                                  Workspace<T>& work) {
  using namespace detail;
  out = Result<T>{};
  auto& [up, rotor, absolute, inertia, v, bias, f, rotor_f, grav] = work;

  bias[kBase] = {};
  up[kBase] = absolute[kBase] = transform(body_rotation(s.quaternion), s.position);
  v[kBase] = s.velocity;
  f[kBase] = force_cross(v[kBase], mul(model.body_inertia[kBase], v[kBase]));
  inertia[kBase] = model.body_inertia[kBase];
  for (int j = kFirstJoint; j < kDof; ++j) {
    const int p = model.parent[j], axis = model.axis[j], i = j - kFirstJoint;
    const T gear = model.gear[j];
    up[j] = mul(transform(joint_rotation(axis, s.q[i]), Vec<T, 3>{}), model.tree[j]);
    rotor[j] = mul(transform(joint_rotation(axis, s.q[i] * gear), Vec<T, 3>{}), model.rotor_tree[j]);
    absolute[j] = mul(up[j], absolute[p]);

    Vec<T, 6> vj{}, vrj{};
    vj.data[axis] = s.qd[i];
    vrj.data[axis] = gear * s.qd[i];
    v[j] = mul(up[j], v[p]);
    auto vr = mul(rotor[j], v[p]);
    add(v[j], vj);
    add(vr, vrj);
    bias[j] = mul(up[j], bias[p]);
    auto br = mul(rotor[j], bias[p]);
    add(bias[j], motion_cross(v[j], vj));
    add(br, motion_cross(vr, vrj));

    inertia[j] = model.body_inertia[j];
    f[j] = mul(inertia[j], bias[j]);
    rotor_f[j] = mul(model.rotor_inertia[j], br);
    add(f[j], force_cross(v[j], mul(inertia[j], v[j])));
    add(rotor_f[j], force_cross(vr, mul(model.rotor_inertia[j], vr)));
  }

  for (int j = kDof - 1; j >= kFirstJoint; --j) {
    const int p = model.parent[j], axis = model.axis[j];
    const auto ut = trans(up[j]), rt = trans(rotor[j]);
    add(inertia[p], mul(mul(ut, inertia[j]), up[j]));
    add(inertia[p], mul(mul(rt, model.rotor_inertia[j]), rotor[j]));
    out.coriolis.data[j] = f[j].data[axis] + model.gear[j] * rotor_f[j].data[axis];
    add(f[p], mul(ut, f[j]));
    add(f[p], mul(rt, rotor_f[j]));
  }

  Vec<T, 6> gravity{};
  for (int i = 0; i < 3; ++i) gravity.data[i+3] = model.gravity.data[i];
  grav[kBase] = mul(up[kBase], gravity);
  const auto base_g = mul(inertia[kBase], grav[kBase]);
  for (int i = 0; i < 6; ++i) {
    out.gravity.data[i] = -base_g.data[i];
    out.coriolis.data[i] = f[kBase].data[i];
    for (int j = 0; j < 6; ++j) out.mass.data[i][j] = inertia[kBase].data[i][j];
  }

  for (int j = kFirstJoint; j < kDof; ++j) {
    const int p = model.parent[j], axis = model.axis[j];
    const T gear = model.gear[j];
    grav[j] = mul(up[j], grav[p]);
    const auto gr = mul(rotor[j], grav[p]);
    out.gravity.data[j] = -dot<6>(inertia[j].data[axis], grav[j].data)
                          - gear * dot<6>(model.rotor_inertia[j].data[axis], gr.data);

    Vec<T, 6> unit{}, rotor_unit{};
    for (int i = 0; i < 6; ++i) {
      unit.data[i] = inertia[j].data[i][axis];
      rotor_unit.data[i] = model.rotor_inertia[j].data[i][axis] * gear;
    }
    out.mass.data[j][j] = unit.data[axis] + gear * rotor_unit.data[axis];
    auto force = mul(trans(up[j]), unit);
    add(force, mul(trans(rotor[j]), rotor_unit));
    for (int i = p; i > kBase; i = model.parent[i]) {
      out.mass.data[i][j] = out.mass.data[j][i] = force.data[model.axis[i]];
      force = mul(trans(up[i]), force);
    }
    for (int i = 0; i < 6; ++i) out.mass.data[i][j] = out.mass.data[j][i] = force.data[i];
  }

  for (int leg = 0; leg < kFeet; ++leg) {
    auto& foot = out.feet[leg];
    const int j = model.foot_parent[leg];
    const auto r = rotation(absolute[j]);
    const auto rt = trans(r);
    auto neg = mul(r, translation(absolute[j]));
    for (int i = 0; i < 3; ++i) neg.data[i] = -neg.data[i];
    const auto xa_inv = transform(rt, neg);
    auto local = model.foot_local[leg];
    const auto shift = translation(xa_inv);
    for (int i = 0; i < 3; ++i) local.data[i] -= shift.data[i];
    foot.position = mul(rt, local);
    const auto world_v = mul(xa_inv, v[j]);
    foot.velocity = cross(world_v.data, foot.position.data);
    for (int i = 0; i < 3; ++i) foot.velocity.data[i] = world_v.data[i+3] + foot.velocity.data[i];

    const auto xc = transform(rt, model.foot_local[leg]);
    const auto ac = mul(xc, bias[j]), vc = mul(xc, v[j]);
    foot.bias = cross(vc.data, vc.data + 3);
    for (int i = 0; i < 3; ++i) foot.bias.data[i] = ac.data[i+3] + foot.bias.data[i];
    Mat<T, 3, 6> x;
    for (int i = 0; i < 3; ++i) for (int c = 0; c < 6; ++c) x.data[i][c] = xc.data[i+3][c];
    for (int i = j; i > kBase; i = model.parent[i]) {
      for (int a = 0; a < 3; ++a) foot.jacobian.data[a][i] = x.data[a][model.axis[i]];
      // This product is dynamic in ULA. Keep the inner extent dynamic in
      // Eigen too; the maps still refer to the same fixed-size buffers.
      Mat<T, 3, 6> next;
      const Eigen::Map<const Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor, 3, 6>>
          jac(&x.data[0][0], 3, 6);
      view(next).noalias() = jac.lazyProduct(view(up[i]));
      x = next;
    }
    for (int i = 0; i < 3; ++i) for (int c = 0; c < 6; ++c) foot.jacobian.data[i][c] = x.data[i][c];
  }
}

template <class T>
ULA_RIGID_HD inline void calculate(const Model<T>& model, const State<T>& s, Result<T>& out) {
  Workspace<T> work;
  calculate(model, s, out, work);
}

}  // namespace ula::rigid

#endif
