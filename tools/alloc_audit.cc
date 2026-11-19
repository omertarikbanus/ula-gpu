// alloc_audit: counts heap allocations (global operator new/delete, plus
// malloc-free C++ containers) during one WBC tick, by phase, with the
// allocation call sites. Run on the steady-state tick (second tick) and on the
// very first tick.
//
// usage: alloc_audit <urdf> [site_depth=5]
#include <execinfo.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "core/WbcHarness.hh"

// ---- allocation recorder ---------------------------------------------------
struct Rec { int phase; size_t bytes; void* bt[10]; int n; };
static const size_t MAXREC = 1 << 18;
static Rec* g_recs;                 // allocated with malloc, never via new
static size_t g_nrecs = 0, g_frees = 0;
static bool g_on = false;
static int g_phase = 0;
static bool g_in_hook = false;

extern "C" {
void* __libc_malloc(size_t);
void __libc_free(void*);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void* __libc_memalign(size_t, size_t);
}
static void record(size_t n) {
  if (g_on && !g_in_hook && g_nrecs < MAXREC) {
    g_in_hook = true;
    Rec& r = g_recs[g_nrecs++];
    r.phase = g_phase; r.bytes = n;
    r.n = backtrace(r.bt, 10);
    g_in_hook = false;
  }
}
// Eigen's aligned allocator calls malloc directly (not operator new), so the
// hook sits on the C allocator, which also sees every operator new.
extern "C" {
void* malloc(size_t n) { record(n); return __libc_malloc(n); }
void* calloc(size_t a, size_t b) { record(a * b); return __libc_calloc(a, b); }
void* realloc(void* p, size_t n) { record(n); return __libc_realloc(p, n); }
void* memalign(size_t al, size_t n) { record(n); return __libc_memalign(al, n); }
int posix_memalign(void** out, size_t al, size_t n) { record(n); *out = __libc_memalign(al, n); return *out ? 0 : 12; }
void* aligned_alloc(size_t al, size_t n) { record(n); return __libc_memalign(al, n); }
void free(void* p) { if (g_on && p && !g_in_hook) ++g_frees; __libc_free(p); }
}

// ---- WBC driven phase by phase (members are protected) ---------------------
struct AuditCtrl : LocomotionCtrl<float> {
  using LocomotionCtrl<float>::LocomotionCtrl;
  enum { MODEL = 1, TASKS, KINWBC, WBIC_SETUP, WBIC_QP, LEGCMD };
  void runPhased(LocomotionCtrlData<float>* cmd, const quadruped::StateEstimate<float>& st,
                 LegController<float>* lc, const quadruped::MIT_UserParameters* up) {
    ++_iter;
    g_phase = MODEL;      _UpdateModel(st, lc->datas);
    g_phase = TASKS;      _ContactTaskUpdate(cmd, up);
    g_phase = KINWBC;     _kin_wbc->FindConfiguration(_full_config, _task_list, _contact_list, _des_jpos, _des_jvel);
    g_phase = WBIC_SETUP; _wbic->UpdateSetting(_A, _Ainv, _coriolis, _grav);
    g_phase = WBIC_QP;    _wbic->MakeTorque(_tau_ff, _wbic_data);
    g_phase = LEGCMD;     _UpdateLegCMD(lc);
    g_phase = 0;
  }
};

static std::string symbolize(void* addr) {
  char** s = backtrace_symbols(&addr, 1);
  std::string raw = s ? s[0] : "?";
  std::free(s);
  // "./alloc_audit(_ZN...+0x1f) [0x...]" -> demangled name
  size_t a = raw.find('('), b = raw.find('+', a);
  if (a == std::string::npos || b == std::string::npos || b <= a + 1) return raw;
  std::string mangled = raw.substr(a + 1, b - a - 1);
  int st = 0;
  char* d = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &st);
  std::string out = (st == 0 && d) ? d : mangled;
  std::free(d);
  if (out.size() > 110) out = out.substr(0, 107) + "...";
  return out;
}

static const char* kPhase[] = {"-", "model update", "task/contact update", "KinWBC",
                               "WBIC setup", "WBIC QP (QuadProg++)", "leg command"};

static void report(const char* title, int depth) {
  std::printf("\n== %s ==\n", title);
  size_t cnt[7] = {0}, bytes[7] = {0};
  for (size_t i = 0; i < g_nrecs; ++i) { cnt[g_recs[i].phase]++; bytes[g_recs[i].phase] += g_recs[i].bytes; }
  size_t tc = 0, tb = 0;
  for (int p = 1; p <= 6; ++p) {
    std::printf("  %-24s %6zu allocs %10zu bytes\n", kPhase[p], cnt[p], bytes[p]);
    tc += cnt[p]; tb += bytes[p];
  }
  std::printf("  %-24s %6zu allocs %10zu bytes (frees seen: %zu)\n", "TOTAL", tc, tb, g_frees);

  // sites: first `depth` frames after the operator new frames, per phase
  for (int p = 1; p <= 6; ++p) {
    std::map<std::string, std::pair<size_t, size_t>> sites;
    for (size_t i = 0; i < g_nrecs; ++i) {
      if (g_recs[i].phase != p) continue;
      std::string key;
      int shown = 0;
      for (int f = 2; f < g_recs[i].n && shown < depth; ++f) {  // skip record() and the malloc hook
        std::string s = symbolize(g_recs[i].bt[f]);
        if (s.find("__libc_start") != std::string::npos || s.find("main") == 0) break;
        key += (shown ? "  <-  " : "") + s;
        ++shown;
      }
      auto& e = sites[key];
      e.first++; e.second += g_recs[i].bytes;
    }
    if (sites.empty()) continue;
    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> v(sites.begin(), sites.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
    std::printf("  -- %s: top sites (count, bytes)\n", kPhase[p]);
    for (size_t k = 0; k < v.size() && k < 6; ++k)
      std::printf("     %5zu x %8zu B  %s\n", v[k].second.first, v[k].second.second, v[k].first.c_str());
  }
}

int main(int argc, char** argv) {
  g_recs = static_cast<Rec*>(std::malloc(sizeof(Rec) * MAXREC));
  { void* w[4]; backtrace(w, 4); }  // warm up libgcc unwinder before hooking
  const std::string urdf = argc > 1 ? argv[1] : "models/go2/go2.urdf";
  const int depth = argc > 2 ? std::atoi(argv[2]) : 5;

  wbch::Harness h;
  if (!h.init(urdf)) { std::printf("urdf parse failed\n"); return 1; }
  // Second controller, phase-instrumented, same model.
  AuditCtrl ctrl(h.quad.buildModel());
  ctrl.setFloatingBaseWeight(10.0f);

  wbch::T leg[3];
  h.stanceJoints(0.32f, leg);
  quadruped::StateEstimate<float> st;
  st.orientation = Vec4<float>(1, 0, 0, 0);
  st.position = Vec3<float>(0, 0, 0.32f);
  for (int l = 0; l < 4; ++l)
    for (int j = 0; j < 3; ++j) h.leg->datas[l].q[j] = leg[j];
  LocomotionCtrlData<float> cmd{};
  cmd.setZero();
  cmd.pBody_des[2] = 0.32f;
  for (int l = 0; l < 4; ++l) { cmd.contact_state[l] = 1.f; cmd.Fr_des[l][2] = h.total_mass * 9.81f / 4; }

  g_on = true; g_nrecs = 0; g_frees = 0;
  ctrl.runPhased(&cmd, st, h.leg.get(), &h.user);
  g_on = false;
  report("first tick (includes lazy initialisation)", depth);

  g_on = true; g_nrecs = 0; g_frees = 0;
  ctrl.runPhased(&cmd, st, h.leg.get(), &h.user);
  g_on = false;
  report("steady-state tick (2nd)", depth);

  // Same tick, 3 legs in contact + 1 swing, to show the swing-task path
  cmd.contact_state[1] = 0.f; cmd.Fr_des[1].setZero();
  cmd.pFoot_des[1] = Vec3<float>(0.19f, 0.15f, 0.05f);
  g_on = true; g_nrecs = 0; g_frees = 0;
  ctrl.runPhased(&cmd, st, h.leg.get(), &h.user);
  g_on = false;
  report("tick with one swing leg", depth);
  return 0;
}
