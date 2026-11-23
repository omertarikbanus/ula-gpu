#ifndef ULA_GPU_BATCH_BACKEND_HH
#define ULA_GPU_BATCH_BACKEND_HH
#include <vector>
#include "port/batch.hh"

namespace ula::port {
#ifdef ULA_BATCH_CUDA
// use_vendor: NVIDIA cuBLAS/cuSOLVER path (validated default); false runs the
// fused inline kernels with per-world scratch.
void* make_device_batch(const FullConfig<float>&,int,bool use_vendor);
void free_device_batch(void*);
int device_reset(void*,int,const float*,const float*,const bool*,void*);
int device_tick(void*,int,BatchTick,void*);
int device_act(void*,int,const float*,const float*,float*,void*);
#endif

class FullCudaReplay {
 public:
  explicit FullCudaReplay(const FullConfig<float>&,bool audit=false,bool use_vendor=true);
  ~FullCudaReplay();
  FullCudaReplay(const FullCudaReplay&)=delete;
  FullCudaReplay& operator=(const FullCudaReplay&)=delete;
  void reset();
  FullOutput<float> step(const CpuInput&);
  void report() const;
 private:
  struct Impl;
  Impl* impl_;
};

// Full ticks on prerecorded plant inputs already resident on the GPU.
// Measures the vendor and inline draft paths, then runs a mixed-gait load.
void benchmark_cuda(const FullConfig<float>&,const std::vector<CpuInput>&);
}
#endif
