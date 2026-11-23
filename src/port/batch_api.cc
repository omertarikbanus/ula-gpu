#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "port/batch_backend.hh"
#include "port/full_config.hh"

namespace {
using namespace ula::port;
struct Batch {
  int size;
  FullConfig<float> cfg;
  std::vector<World> host;
  std::vector<InlineSolverScratch<float>> scratch;
  void* device=nullptr;
  ~Batch() {
#ifdef ULA_BATCH_CUDA
    if (device) free_device_batch(device);
#endif
  }
};
}

// Contiguous row-major arrays. On CUDA all array pointers must be on the
// current device, and all operations are enqueued on the caller's stream.
// backend: null or "vendor" = NVIDIA cuBLAS/cuSOLVER path (validated
// default); "inline" = the fused inline kernels. The CPU path always uses
// the built-in inline solver and ignores the choice.
extern "C" {
void* ula_batch_create(const char* urdf,int size,int cuda,const char* backend) {
  try {
    if (!urdf || size<=0 || size>INT32_MAX/12) throw std::runtime_error("invalid batch size or URDF");
    bool use_vendor=true;
    if (backend && std::string_view(backend)!="vendor") {
      if (std::string_view(backend)=="inline") use_vendor=false;
      else throw std::runtime_error("unknown backend \""+std::string(backend)+"\" (expected \"vendor\" or \"inline\")");
    }
    auto b=std::make_unique<Batch>();
    b->size=size;
    CpuConfig cfg;
    cfg.planner.decoder=ActionDecoderConfig::gamma5_defaults();
    cfg.floating_base_weight=50.f;
    cfg.wbic.Kp_foot.setConstant(550.); cfg.wbic.Kd_foot.setConstant(10.);
    cfg.wbic.Kd_ori.setConstant(10.); cfg.wbic.Kd_body.setConstant(30.);
    cfg.wbic.Kp_joint.setConstant(5.); cfg.wbic.Kd_joint.setConstant(3.);
    CpuController cpu;
    if (!cpu.init(urdf,cfg)) throw std::runtime_error("cannot load Go2 URDF");
    b->cfg=pack_full_config(cpu);
    if (cuda) {
#ifdef ULA_BATCH_CUDA
      b->device=make_device_batch(b->cfg,size,use_vendor);
#else
      throw std::runtime_error("rebuild with ULA_ENABLE_CUDA_REPLAY=ON for CUDA");
#endif
    } else {
      b->host.resize(size);
      b->scratch.resize(size);
    }
    return b.release();
  } catch (const std::exception& e) { std::fprintf(stderr,"ula_batch_create: %s\n",e.what()); return nullptr; }
}
void ula_batch_delete(void* handle) { delete static_cast<Batch*>(handle); }
int ula_batch_reset(void* handle,const float* q,const float* v,const bool* mask,void* stream) {
  auto& b=*static_cast<Batch*>(handle);
#ifdef ULA_BATCH_CUDA
  if (b.device) return device_reset(b.device,b.size,q,v,mask,stream);
#endif
  for (int i=0;i<b.size;++i) if (!mask || mask[i]) reset_world(b.cfg,b.host[i],q+19*i,v+18*i);
  return 0;
}
int ula_batch_tick(void* handle,const float* q,const float* v,const float* action,const float* command,
                   const float* noise,const bool* active,float* obs,float* vw,int* status,int prime,void* stream) {
  auto& b=*static_cast<Batch*>(handle);
  BatchTick tick{q,v,action,command,noise,active,obs,vw,status,prime!=0};
#ifdef ULA_BATCH_CUDA
  if (b.device) {
    try { return device_tick(b.device,b.size,tick,stream); }
    catch (const std::exception& e) { std::fprintf(stderr,"ula_batch_tick: %s\n",e.what()); return -1; }
  }
#endif
  for (int i=0;i<b.size;++i) plant_tick(b.cfg,b.host[i],tick,i,b.scratch[i]);
  return 0;
}
int ula_batch_act(void* handle,const float* length,const float* velocity,float* ctrl,void* stream) {
  auto& b=*static_cast<Batch*>(handle);
#ifdef ULA_BATCH_CUDA
  if (b.device) return device_act(b.device,b.size,length,velocity,ctrl,stream);
#endif
  for (int j=0;j<12*b.size;++j)
    ctrl[j]=ula::actuation::sim_motor_torque(b.host[j/12].out.motor[j%12],length[j],velocity[j],true);
  return 0;
}
}
