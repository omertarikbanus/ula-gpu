#ifndef WBC_CONTROLLER_H
#define WBC_CONTROLLER_H

#include "estimators/StateEstimatorTypes.hh"
#include "controllers/LegController.h"
#include <dynamics/FloatingBaseModel.h>
#include <dynamics/Quadruped.h>
#include "types/cppTypes.h"
#include <controllers/wbc/WBIC/WBIC.hpp>
#include <controllers/wbc/WBIC/KinWBC.hpp>
#include "parameters/MIT_UserParameters.h"


#define WBCtrl WBC_Ctrl<T>

template<typename T>
class WBC_Ctrl{
  public:
    WBC_Ctrl(FloatingBaseModel<T> model);
    virtual ~WBC_Ctrl();

    void run(void* input,
             const quadruped::StateEstimate<T>& state_est,
             ::LegController<T>* leg_controller,
             const MIT_UserParameters* user_params);
    void setFloatingBaseWeight(const T & weight){
      _wbic_data->_W_floating = DVec<T>::Constant(6, weight);
    }
    const FBModelState<T>& getState() const { return _state; }
    const FloatingBaseModel<T>& getModel() const { return _model; }
    const DVec<T>& getDesJpos() const { return _des_jpos; }
    const DVec<T>& getDesJvel() const { return _des_jvel; }
    const WBIC_ExtraData<T>* getWBICData() const { return _wbic_data; }

  protected:
    virtual void _ContactTaskUpdate(void * input,
                                    const MIT_UserParameters* user_params) = 0;
    virtual void _ContactTaskUpdateTEST(void * input,
                                        const MIT_UserParameters* user_params){
      (void)input;
      (void)user_params;
    }
    virtual void _LCM_PublishData(){}
    void _UpdateModel(const quadruped::StateEstimate<T> & state_est,
                      const LegControllerData<T> * leg_data);
    void _UpdateLegCMD(::LegController<T>* leg_controller);
    void _ComputeWBC();

    KinWBC<T>* _kin_wbc;
    WBIC<T>* _wbic;
    WBIC_ExtraData<T>* _wbic_data;

    FloatingBaseModel<T> _model;
    std::vector<ContactSpec<T> * > _contact_list;
    std::vector<Task<T> * > _task_list;

    DMat<T> _A;
    DMat<T> _Ainv;
    DVec<T> _grav;
    DVec<T> _coriolis;

    FBModelState<T> _state;

    DVec<T> _full_config;
    DVec<T> _tau_ff;
    DVec<T> _des_jpos;
    DVec<T> _des_jvel;

    std::vector<T> _Kp_joint, _Kd_joint;
    //std::vector<T> _Kp_joint_swing, _Kd_joint_swing;

    unsigned long long _iter;
};
#endif
