#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "port/batch_backend.hh"
#include "port/solver_audit.hh"

namespace ula::port {
namespace {
constexpr int kBlock=64;
constexpr int N=rigid::kDof;
constexpr int kBadLu=-2, kBadSvd=-3, kBadMatrix=-4;
using Clock=std::chrono::steady_clock;
using Counts=std::array<int,5>;
using ColMatrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,N,N>;

int blocks(int n) { return (n+kBlock-1)/kBlock; }

void check(cudaError_t e,const char* where) {
  if (e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
void check(cublasStatus_t e,const char* where) {
  if (e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where)+": "+cublasGetStatusString(e));
}
void check(cusolverStatus_t e,const char* where) {
  if (e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(where)+": cuSOLVER status "+std::to_string(e));
}
void check_launch(const char* where) { check(cudaGetLastError(),where); }

template<class T> struct DeviceBuffer {
  T* data=nullptr;
  DeviceBuffer()=default;
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&)=delete;
  DeviceBuffer& operator=(const DeviceBuffer&)=delete;
  void alloc(size_t count) { check(cudaMalloc(&data,count*sizeof(T)),"allocate controller device buffer"); }
};

// Buffers use packed, real extents (column-major for the libraries). No
// allocation, device-to-host copy, or synchronization in the uniform tick.
struct VendorSolvers {
  cublasHandle_t blas=nullptr;
  cusolverDnHandle_t svid=nullptr;
  gesvdjInfo_t params=nullptr;
  DeviceBuffer<float> matrix, inverse, u, v, sigma, scratch;
  DeviceBuffer<float*> inverse_ptr;
  DeviceBuffer<float*> matrix_ptr;
  DeviceBuffer<int> pivots, lu_info, info;
  int work_size=0;

  ~VendorSolvers() {
    if (params) cusolverDnDestroyGesvdjInfo(params);
    if (svid) cusolverDnDestroy(svid);
    if (blas) cublasDestroy(blas);
  }
  void init(int count,bool mixed) {
    check(cublasCreate(&blas),"cublasCreate");
    check(cusolverDnCreate(&svid),"cusolverDnCreate");
    check(cusolverDnCreateGesvdjInfo(&params),"cusolverDnCreateGesvdjInfo");
    check(cusolverDnXgesvdjSetSortEig(params,1),"sort singular values");
    const size_t matrices=static_cast<size_t>(count)*N*N;
    matrix.alloc(matrices);
    inverse.alloc(matrices);
    u.alloc(matrices);
    v.alloc(static_cast<size_t>(count)*12*12);  // at most 12 columns (rows of J)
    sigma.alloc(static_cast<size_t>(count)*12);
    matrix_ptr.alloc(count); inverse_ptr.alloc(count); pivots.alloc(static_cast<size_t>(count)*N);
    info.alloc(count); lu_info.alloc(count);
    for (int rows=3;rows<=12;rows+=3) {
      if (!mixed && rows!=3 && rows!=6) continue;
      for (int m : {rows,N}) {
        int need=0;
        check(cusolverDnSgesvdjBatched_bufferSize(svid,CUSOLVER_EIG_MODE_VECTOR,m,rows,
              matrix.data,m,sigma.data,u.data,m,v.data,rows,&need,params,count),"SVD workspace size");
        work_size=std::max(work_size,need);
      }
    }
    scratch.alloc(std::max(1,work_size));
  }
  void set_stream(cudaStream_t stream) {
    check(cublasSetStream(blas,stream),"set LU stream");
    check(cusolverDnSetStream(svid,stream),"set SVD stream");
  }
  void lu(int count) {
    check(cublasSgetrfBatched(blas,N,matrix_ptr.data,N,pivots.data,lu_info.data,count),"batched partial-pivot LU");
    check(cublasSgetriBatched(blas,N,matrix_ptr.data,N,pivots.data,inverse_ptr.data,N,info.data,count),"batched inverse");
  }
  void svd(int m,int n,int count) {
    check(cusolverDnSgesvdjBatched(svid,CUSOLVER_EIG_MODE_VECTOR,m,n,matrix.data,m,
          sigma.data,u.data,m,v.data,n,scratch.data,work_size,info.data,params,count),"batched Jacobi SVD");
  }
};

struct TickWorld { CpuInput in; bool active; };
struct TickSource {
  BatchTick plant{};
  const CpuInput* recorded=nullptr;
  int samples=0, offset=0;
  bool shared=false;
};

// The default trot always has 2 planned contacts, even when worlds are on
// opposite diagonals. Other schedules use the same math with an exact-size
// routing pass that reads back only five counts (never controller state).
bool fixed_trot(const GaitSchedulerConfig& gait) {
  constexpr GaitSchedulerConfig trot{};
  for (int i=0;i<4;++i)
    if (gait.phase_offsets[i]!=trot.phase_offsets[i] || gait.contact_durations[i]!=0.5f) return false;
  return true;
}

struct DeviceBatch {
  int size;
  bool fixed;
  bool use_vendor;  // cuBLAS/cuSOLVER, or the fused inline kernels
  DeviceBuffer<FullConfig<float>> cfg;
  DeviceBuffer<World> world;
  DeviceBuffer<TickWorld> tick;
  DeviceBuffer<int> indices, slots, counts;
  DeviceBuffer<InlineSolverScratch<float>> scratch;
  int* host_counts=nullptr;
  VendorSolvers vendor;
  Counts last_counts{};
  int group_waits=0;
  double group_wait_ms=0;

  DeviceBatch(const FullConfig<float>& config,int n,bool vendor_path)
      :size(n),fixed(fixed_trot(config.planner.gait)),use_vendor(vendor_path) {
    cfg.alloc(1); world.alloc(n);
    check(cudaMemcpy(cfg.data,&config,sizeof(config),cudaMemcpyHostToDevice),"upload Go2 config");
    if (vendor_path) {
      vendor.init(n,!fixed);
      tick.alloc(n);
      if (!fixed) {
        indices.alloc(n); slots.alloc(n); counts.alloc(5);
        check(cudaMallocHost(&host_counts,5*sizeof(int)),"allocate contact count readback");
      }
    } else scratch.alloc(n);
  }
  ~DeviceBatch() { if (host_counts) cudaFreeHost(host_counts); }
};

__global__ void reset_kernel(const FullConfig<float>* cfg,World* w,int n,
                             const float* q,const float* v,const bool* mask,bool spread) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n && (!mask || mask[i])) {
    if (q) reset_world(*cfg,w[i],q+19*i,v+18*i);
    else reset(w[i].state,cfg->planner);
    if (spread) w[i].state.gait.theta=(i+0.5f)/n;
  }
}

__device__ const CpuInput& replay_input(TickSource src,int i) {
  return src.recorded[src.shared ? 0 : (i+src.offset)%src.samples];
}

__global__ void begin_kernel(const FullConfig<float>* cfg,World* worlds,int n,TickSource src,
                             TickWorld* ticks,int* counts,int* slots) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  auto& t=ticks[i];
  t.active=src.recorded || !src.plant.active || src.plant.active[i];
  if (slots) slots[i]=-1;
  if (!t.active) return;
  auto& w=worlds[i];
  t.in=src.recorded ? replay_input(src,i) : plant_input(*cfg,w,src.plant,i);
  begin_tick(*cfg,w.state,t.in,w.out,w.work);
  if (!w.out.wbic_ran) return;
  const int feet=w.work.problem.contact_rows/3;
  if (counts) slots[i]=atomicAdd(counts+feet,1);
  else if (feet!=2) w.out.status=kBadMatrix;  // guard the fixed-trot shortcut
}

__device__ bool ready(const World& w,const TickWorld& tick) {
  return tick.active && w.out.wbic_ran && w.out.status==0;
}
__device__ int world_index(const int* indices,int i) { return indices ? indices[i] : i; }

__global__ void scatter_kernel(const World* w,int n,const int* counts,const int* slots,int* indices) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n || slots[i]<0) return;
  const int feet=w[i].work.problem.contact_rows/3;
  int dest=slots[i];
  for (int g=0;g<feet;++g) dest+=counts[g];
  indices[dest]=i;
}

__global__ void pack_lu(World* w,const TickWorld* t,const int* indices,int count,
                        float* in,float* out,float** in_ptr,float** out_ptr) {
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if (b>=count) return;
  const int i=world_index(indices,b);
  in_ptr[b]=in+size_t(b)*N*N; out_ptr[b]=out+size_t(b)*N*N;
  Eigen::Map<Eigen::Matrix<float,N,N>> a(in_ptr[b]);
  if (ready(w[i],t[i])) {
    a=rigid::detail::view(w[i].work.dynamics.mass);
    if (a.array().isFinite().all()) return;
    w[i].out.status=kBadMatrix;
  }
  // Fixed-trot batch slots with no work (startup/reset mask or prior
  // failure) are never consumed. Do not resize or pad any actual problem.
  a.setIdentity();
}

__global__ void unpack_lu(World* w,const TickWorld* t,const int* indices,int count,
                          const float* inverse,const int* factor_info,const int* inverse_info) {
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if (b>=count) return;
  const int i=world_index(indices,b);
  if (!ready(w[i],t[i])) return;
  const Eigen::Map<const Eigen::Matrix<float,N,N>> a(inverse+size_t(b)*N*N);
  if (factor_info[b] || inverse_info[b] || !a.array().isFinite().all()) { w[i].out.status=kBadLu; return; }
  rigid::detail::view(w[i].work.inv_mass)=a;
}

__global__ void pack_svd(World* worlds,const TickWorld* t,const int* indices,int start,int count,
                         int m,int n,bool weighted,int task,float* in) {
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if (b>=count) return;
  const int i=world_index(indices,start+b);
  auto& w=worlds[i];
  Eigen::Map<ColMatrix> a(in+size_t(b)*m*n,m,n);
  if (ready(w,t[i])) {
    if (task==0 && w.work.problem.contact_rows==0) small::identity(w.work.nullspace,N,N);
    const auto& src=full_detail::hierarchy_inverse_input(weighted,task,w.work);
    // cuSOLVER takes a tall J^T in the unweighted hierarchy. Swapping
    // left/right factors when rebuilding J^+ preserves ULA's formula.
    if (weighted) a=small::view(src,n,n);
    else a=small::view(src,n,m).transpose();
    if (a.array().isFinite().all()) return;
    w.out.status=kBadMatrix;
  }
  a.setIdentity();
}

__global__ void unpack_svd(World* worlds,const TickWorld* t,const int* indices,int start,int count,
                           int m,int n,bool weighted,int task,float* u_data,float* v_data,
                           const float* sigma,const int* info) {
  const int b=blockIdx.x*blockDim.x+threadIdx.x;
  if (b>=count) return;
  const int i=world_index(indices,start+b);
  auto& w=worlds[i];
  if (!ready(w,t[i])) return;
  Eigen::Map<ColMatrix> u(u_data+size_t(b)*m*m,m,m),v(v_data+size_t(b)*n*n,n,n);
  const auto s=small::vec(sigma+size_t(b)*n,n);
  if (info[b] || !s.array().isFinite().all()) { w.out.status=kBadSvd; return; }
  const double cutoff=weighted ? full_detail::kWeightedCutoff : full_detail::kKinematicCutoff;
  for (int j=0;j<n;++j) {
    const float factor=s(j)>cutoff ? static_cast<float>(1.0/s(j)) : 0.f;
    if (weighted) v.col(j)*=factor;
    else u.col(j)*=factor;
  }
  if (weighted) small::view(w.work.linv,n,n).noalias()=v.lazyProduct(u.transpose());
  else small::view(w.work.pinv,m,n).noalias()=u.leftCols(n).lazyProduct(v.transpose());
  full_detail::apply_hierarchy_inverse(weighted,task,w.work);
}

__global__ void finish_kernel(const FullConfig<float>* cfg,World* w,int n,TickSource src,const TickWorld* t) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n || !t[i].active) return;
  finish_tick(*cfg,t[i].in,w[i].out,w[i].work);
  if (!src.recorded) plant_output(w[i],src.plant,i);
}

__global__ void inline_kernel(const FullConfig<float>* cfg,World* w,int n,TickSource src,
                               InlineSolverScratch<float>* scratch) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  if (src.recorded) full_tick(*cfg,w[i].state,replay_input(src,i),w[i].out,w[i].work,scratch[i]);
  else plant_tick(*cfg,w[i],src.plant,i,scratch[i]);
}

template<class T> void copy_sample(T* dst,const T* dev,int size,cudaStream_t stream) {
  check(cudaMemcpyAsync(dst,dev,size*sizeof(T),cudaMemcpyDeviceToHost,stream),"copy solver audit sample");
}

void run_vendor_tick(DeviceBatch& b,TickSource src,cudaStream_t stream,SolverAudit* audit=nullptr) {
  auto& lib=b.vendor;
  lib.set_stream(stream);
  auto* w=b.world.data;
  auto* t=b.tick.data;
  const auto* indices=b.indices.data;
  const int size=b.size;
  if (!b.fixed) check(cudaMemsetAsync(b.counts.data,0,5*sizeof(int),stream),"clear contact counts");
  begin_kernel<<<blocks(size),kBlock,0,stream>>>(b.cfg.data,w,size,src,t,b.counts.data,b.slots.data);
  check_launch("prepare controller tick");
  Counts counts{};
  if (b.fixed) counts[2]=size;
  else {
    scatter_kernel<<<blocks(size),kBlock,0,stream>>>(w,size,b.counts.data,b.slots.data,b.indices.data);
    check_launch("gather exact contact groups");
    const auto begin=Clock::now();
    check(cudaMemcpyAsync(b.host_counts,b.counts.data,5*sizeof(int),cudaMemcpyDeviceToHost,stream),"read contact counts");
    check(cudaStreamSynchronize(stream),"wait for contact counts");
    b.group_wait_ms+=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    ++b.group_waits;
    std::copy_n(b.host_counts,5,counts.begin());
  }
  b.last_counts=counts;
  const int total=std::accumulate(counts.begin(),counts.end(),0);
  if (total) {
    // The audit is optional; its host buffers are filled by the copies below.
    std::array<float,N*N> sample, inverse, u;
    std::array<float,12*12> v;
    std::array<float,12> sigma;
    pack_lu<<<blocks(total),kBlock,0,stream>>>(w,t,indices,total,
        lib.matrix.data,lib.inverse.data,lib.matrix_ptr.data,lib.inverse_ptr.data);
    check_launch("pack mass matrices");
    if (audit) copy_sample(sample.data(),lib.matrix.data,N*N,stream);
    lib.lu(total);
    if (audit) {
      copy_sample(inverse.data(),lib.inverse.data,N*N,stream);
      check(cudaStreamSynchronize(stream),"read LU audit");
      // The controller separately checks factor and solve info per world.
      audit->lu(sample.data(),inverse.data(),N);
    }
    unpack_lu<<<blocks(total),kBlock,0,stream>>>(w,t,indices,total,lib.inverse.data,lib.lu_info.data,lib.info.data);
    check_launch("unpack inverse mass matrices");

    const auto svd=[&](bool weighted,int task,int rows,int start,int count) {
      const int m=weighted ? rows : N;
      pack_svd<<<blocks(count),kBlock,0,stream>>>(w,t,indices,start,count,m,rows,weighted,task,lib.matrix.data);
      check_launch("pack pseudo-inverse inputs");
      if (audit) copy_sample(sample.data(),lib.matrix.data,m*rows,stream);
      lib.svd(m,rows,count);
      if (audit) {
        copy_sample(u.data(),lib.u.data,m*m,stream);
        copy_sample(v.data(),lib.v.data,rows*rows,stream);
        copy_sample(sigma.data(),lib.sigma.data,rows,stream);
        check(cudaStreamSynchronize(stream),"read SVD audit");
        audit->svd(sample.data(),u.data(),v.data(),sigma.data(),m,rows,weighted,task,
                   weighted ? full_detail::kWeightedCutoff : full_detail::kKinematicCutoff);
      }
      unpack_svd<<<blocks(count),kBlock,0,stream>>>(w,t,indices,start,count,m,rows,weighted,task,
          lib.u.data,lib.v.data,lib.sigma.data,lib.info.data);
      check_launch("apply task projection");
    };
    // The hierarchy stays sequential within each world. The contacts have
    // one shape per group. Each subsequent task has three rows, so all
    // worlds that have that task can share a single call.
    for (bool weighted : {false,true}) {
      int start=counts[0];
      for (int feet=1;feet<=4;++feet) {
        if (counts[feet]) svd(weighted,-1,3*feet,start,counts[feet]);
        start+=counts[feet];
      }
      int count=total;
      for (int k=0;k<6;++k) {
        if (k>=2) count-=counts[6-k];
        if (count) svd(weighted,k,3,0,count);
      }
    }
  }
  finish_kernel<<<blocks(size),kBlock,0,stream>>>(b.cfg.data,w,size,src,t);
  check_launch("finish controller tick");
}

__global__ void act_kernel(const World* w,int n,const float* q,const float* v,float* ctrl) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<12*n) ctrl[i]=actuation::sim_motor_torque(w[i/12].out.motor[i%12],q[i],v[i],true);
}

__device__ bool same(const FullOutput<float>& a,const FullOutput<float>& b) {
  if (a.status!=b.status || a.wbic_ran!=b.wbic_ran || a.planner.gait.gait_phase!=b.planner.gait.gait_phase) return false;
  for (int i=0;i<12;++i) {
    const int l=i/3,j=i%3;
    if (a.planner.gait.planned_contact[l]!=b.planner.gait.planned_contact[l] ||
        a.planner.wbic.foot.foot_position_target[l][j]!=b.planner.wbic.foot.foot_position_target[l][j] ||
        a.joint_position_target[i]!=b.joint_position_target[i] || a.joint_velocity_target[i]!=b.joint_velocity_target[i] ||
        a.motor_torque[i]!=b.motor_torque[i] || a.feedforward_torque[i]!=b.feedforward_torque[i]) return false;
  }
  return true;
}
__global__ void check_kernel(const World* w,int n,FullOutput<float>* out,int* bad) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i==0) *out=w[0].out;
  if (i>0 && i<n && !same(w[0].out,w[i].out)) atomicExch(bad,1);
}
__global__ void bench_status_kernel(const World* w,int n,int* status) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n && w[i].out.status!=0) atomicCAS(status,0,w[i].out.status);
}
}  // namespace

void* make_device_batch(const FullConfig<float>& cfg,int n,bool use_vendor) {
  return new DeviceBatch(cfg,n,use_vendor);
}
void free_device_batch(void* b) { delete static_cast<DeviceBatch*>(b); }
int device_reset(void* ptr,int n,const float* q,const float* v,const bool* mask,void* stream) {
  const auto& b=*static_cast<DeviceBatch*>(ptr);
  reset_kernel<<<blocks(n),kBlock,0,static_cast<cudaStream_t>(stream)>>>(b.cfg.data,b.world.data,n,q,v,mask,false);
  return cudaGetLastError();
}
int device_tick(void* ptr,int n,BatchTick in,void* stream) {
  auto& b=*static_cast<DeviceBatch*>(ptr);
  if (n!=b.size) throw std::invalid_argument("CUDA controller batch size changed");
  if (b.use_vendor) {
    run_vendor_tick(b,TickSource{in},static_cast<cudaStream_t>(stream));
  } else {
    inline_kernel<<<blocks(n),kBlock,0,static_cast<cudaStream_t>(stream)>>>(
        b.cfg.data,b.world.data,n,TickSource{in},b.scratch.data);
    check_launch("inline controller tick");
  }
  return cudaSuccess;
}
int device_act(void* ptr,int n,const float* q,const float* v,float* ctrl,void* stream) {
  const auto& b=*static_cast<DeviceBatch*>(ptr);
  act_kernel<<<blocks(12*n),kBlock,0,static_cast<cudaStream_t>(stream)>>>(b.world.data,n,q,v,ctrl);
  return cudaGetLastError();
}

struct FullCudaReplay::Impl {
  static constexpr int n=65; // exercises a non-block-aligned batch
  DeviceBatch batch;
  DeviceBuffer<CpuInput> in;
  DeviceBuffer<FullOutput<float>> out;
  DeviceBuffer<int> bad;
  SolverAudit audit;
  bool check_solvers;
  bool use_vendor;
  int tick=0, episode_tick=0;
  Impl(const FullConfig<float>& cfg,bool check_solvers,bool use_vendor)
      :batch(cfg,n,use_vendor),check_solvers(check_solvers),use_vendor(use_vendor) {
    in.alloc(1); out.alloc(1); bad.alloc(1);
  }
};

FullCudaReplay::FullCudaReplay(const FullConfig<float>& cfg,bool audit,bool use_vendor) : impl_(nullptr) {
  auto p=std::make_unique<Impl>(cfg,audit,use_vendor);
  check(static_cast<cudaError_t>(device_reset(&p->batch,Impl::n,nullptr,nullptr,nullptr,nullptr)),"reset replay worlds");
  impl_=p.release();
}
FullCudaReplay::~FullCudaReplay() { delete impl_; }
void FullCudaReplay::reset() {
  impl_->episode_tick=0;
  check(static_cast<cudaError_t>(device_reset(&impl_->batch,Impl::n,nullptr,nullptr,nullptr,nullptr)),"reset replay worlds");
}
FullOutput<float> FullCudaReplay::step(const CpuInput& in) {
  auto& p=*impl_;
  check(cudaMemcpy(p.in.data,&in,sizeof(in),cudaMemcpyHostToDevice),"upload replay input");
  check(cudaMemset(p.bad.data,0,sizeof(int)),"clear replay check");
  p.audit.tick=p.tick;
  // The audit samples the vendor libraries' own submissions; the fused
  // inline tick has no separate calls to audit.
  const bool sample=p.use_vendor && p.check_solvers && p.episode_tick>0 && (p.episode_tick-1)%32==0;
  const TickSource source{{},p.in.data,1,0,true};
  if (p.use_vendor)
    run_vendor_tick(p.batch,source,nullptr,sample ? &p.audit : nullptr);
  else {
    inline_kernel<<<blocks(Impl::n),kBlock>>>(p.batch.cfg.data,p.batch.world.data,Impl::n,
        source,p.batch.scratch.data);
    check_launch("inline replay tick");
  }
  check_kernel<<<blocks(Impl::n),kBlock>>>(p.batch.world.data,Impl::n,p.out.data,p.bad.data);
  check_launch("verify replicated worlds");
  FullOutput<float> out;
  int bad;
  check(cudaMemcpy(&out,p.out.data,sizeof(out),cudaMemcpyDeviceToHost),"read replay output");
  check(cudaMemcpy(&bad,p.bad.data,sizeof(bad),cudaMemcpyDeviceToHost),"read repeatability check");
  if (bad) throw std::runtime_error("replicated worlds disagree");
  if (out.status) throw std::runtime_error("CUDA controller status="+std::to_string(out.status)
       +" (-2: LU, -3: SVD, -4: invalid matrix/contact count; positive: QP)");
  ++p.tick; ++p.episode_tick;
  return out;
}
void FullCudaReplay::report() const {
  if (impl_->check_solvers && impl_->use_vendor) impl_->audit.report();
}

void benchmark_cuda(const FullConfig<float>& config,const std::vector<CpuInput>& input) {
  if (input.empty()) throw std::invalid_argument("no plant input for benchmark");
  cudaDeviceProp props{};
  int device=0;
  check(cudaGetDevice(&device),"get benchmark device");
  check(cudaGetDeviceProperties(&props,device),"get benchmark GPU name");
  std::fprintf(stderr,"Controller benchmark on %s, timing includes launches and any contact count wait\n",props.name);
  DeviceBuffer<CpuInput> samples;
  samples.alloc(input.size());
  check(cudaMemcpy(samples.data,input.data(),input.size()*sizeof(CpuInput),cudaMemcpyHostToDevice),"upload benchmark trace");
  DeviceBuffer<int> bad;
  bad.alloc(1);
  constexpr int timed=30, warm=6;
  for (int n : {64,256,1024}) for (bool mixed : {false,true}) for (bool vendor : {false,true}) {
    auto cfg=config;
    if (mixed) {
      // Alternate workload only, not the stock training gait. Phases are
      // also spread across the batch so the same tick has 0..4 contacts.
      for (int l=0;l<4;++l) cfg.planner.gait.phase_offsets[l]=0.1f*l;
    }
    DeviceBatch b(cfg,n,vendor);
    auto run=[&](int step,SolverAudit* audit=nullptr) {
      const TickSource source{{},samples.data,static_cast<int>(input.size()),step*7,false};
      if (vendor) run_vendor_tick(b,source,nullptr,audit);
      else {
        inline_kernel<<<blocks(n),kBlock>>>(b.cfg.data,b.world.data,n,source,b.scratch.data);
        check_launch("inline benchmark tick");
      }
    };
    reset_kernel<<<blocks(n),kBlock>>>(b.cfg.data,b.world.data,n,nullptr,nullptr,nullptr,true);
    check_launch("reset benchmark worlds");
    check(cudaMemset(bad.data,0,sizeof(int)),"clear benchmark status");
    SolverAudit mixed_audit;
    const bool audit_groups=vendor && mixed && n==64;
    for (int step=0;step<warm;++step) {
      run(step,audit_groups && step==warm-1 ? &mixed_audit : nullptr);
      bench_status_kernel<<<blocks(n),kBlock>>>(b.world.data,n,bad.data);
      check_launch("benchmark warmup status");
    }
    check(cudaDeviceSynchronize(),"benchmark warmup");
    if (audit_groups) {
      std::fprintf(stderr,"Mixed-gait solver audit (one matrix per active shape and stage):\n");
      mixed_audit.report();
    }
    b.group_waits=0; b.group_wait_ms=0;
    const auto begin=Clock::now();
    for (int step=warm;step<warm+timed;++step) run(step);
    check(cudaDeviceSynchronize(),"benchmark timed ticks");
    const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    bench_status_kernel<<<blocks(n),kBlock>>>(b.world.data,n,bad.data);
    check_launch("benchmark final status");
    int status;
    check(cudaMemcpy(&status,bad.data,sizeof(status),cudaMemcpyDeviceToHost),"benchmark controller status");
    std::fprintf(stderr,"CONTROLLER %-6s %-5s worlds=%4d %.0f ticks/s %.3f ms/batch count-wait=%.3f ms/tick (%d waits) status=%d",
                 vendor?"NVIDIA":"inline",mixed?"mixed":"trot",n,1e3*n*timed/elapsed,elapsed/timed,
                 b.group_wait_ms/timed,b.group_waits,status);
    if (mixed && vendor)
      std::fprintf(stderr,"  final groups 0/1/2/3/4 contacts: %d/%d/%d/%d/%d",
                   b.last_counts[0],b.last_counts[1],b.last_counts[2],b.last_counts[3],b.last_counts[4]);
    std::fputc('\n',stderr);
    if (status) throw std::runtime_error("controller benchmark reported a failed tick");
  }
}

}  // namespace ula::port
