#include "controllers/wbc/WBC_Ctrl/WBC_Ctrl.hpp"
#include <utilities/Utilities_print.h>
#include "port/rigid_body_adapter.hh"

template<typename T>
WBC_Ctrl<T>::WBC_Ctrl(FloatingBaseModel<T> model):
  _full_config(cheetah::num_act_joint + 7),
  _tau_ff(cheetah::num_act_joint),
  _des_jpos(cheetah::num_act_joint),
  _des_jvel(cheetah::num_act_joint)
{
  _iter = 0;
  _full_config.setZero();
  _tau_ff.setZero();
  _des_jpos.setZero();
  _des_jvel.setZero();

  _model = model;
  _kin_wbc = new KinWBC<T>(cheetah::dim_config);

  _wbic = new WBIC<T>(cheetah::dim_config, &(_contact_list), &(_task_list));
  _wbic_data = new WBIC_ExtraData<T>();

  _wbic_data->_W_floating = DVec<T>::Constant(6, 0.1);
  //_wbic_data->_W_floating = DVec<T>::Constant(6, 50.);
  //_wbic_data->_W_floating[5] = 0.1;
  _wbic_data->_W_rf = DVec<T>::Constant(12, 1.);

  _Kp_joint.resize(cheetah::num_leg_joint, 5.);
  _Kd_joint.resize(cheetah::num_leg_joint, 1.5);

  //_Kp_joint_swing.resize(cheetah::num_leg_joint, 10.);
  //_Kd_joint_swing.resize(cheetah::num_leg_joint, 1.5);

  _state.q = DVec<T>::Zero(cheetah::num_act_joint);
  _state.qd = DVec<T>::Zero(cheetah::num_act_joint);
}

template<typename T>
WBC_Ctrl<T>::~WBC_Ctrl(){
  delete _kin_wbc;
  delete _wbic;
  delete _wbic_data;

  // Task/contact pointers are owned by higher-level controllers.
  _task_list.clear();
  _contact_list.clear();
}

template <typename T>
void WBC_Ctrl<T>::_ComputeWBC() {
  // TEST
  _kin_wbc->FindConfiguration(_full_config, _task_list, _contact_list,
                              _des_jpos, _des_jvel);

  // WBIC
  _wbic->UpdateSetting(_A, _Ainv, _coriolis, _grav);
  _wbic->MakeTorque(_tau_ff, _wbic_data);
}

template<typename T>
void WBC_Ctrl<T>::run(void* input,
                      const quadruped::StateEstimate<T>& state_est,
                      ::LegController<T>* leg_controller,
                      const MIT_UserParameters* user_params){
  ++_iter;

  // Update Model
  _UpdateModel(state_est, leg_controller->datas);

  // Task & Contact Update
  _ContactTaskUpdate(input, user_params);

  // WBC Computation
  _ComputeWBC();
  
  // TEST
  //T dt(0.002);
  //for(size_t i(0); i<12; ++i){
    //_des_jpos[i] = _state.q[i] + _state.qd[i] * dt + 0.5 * _wbic_data->_qddot[i+6] * dt * dt;
    //_des_jvel[i] = _state.qd[i] + _wbic_data->_qddot[i+6]*dt;
  //}

  //_ContactTaskUpdateTEST(input, data);
  //_ComputeWBC();
  // END of TEST

  // Update Leg Command
  _UpdateLegCMD(leg_controller);

}



template<typename T>
void WBC_Ctrl<T>::_UpdateLegCMD(::LegController<T>* leg_controller){
  LegControllerCommand<T> * cmd = leg_controller->commands;
  //Vec4<T> contact = data._stateEstimator->getResult().contactEstimate;

  for (size_t leg(0); leg < cheetah::num_leg; ++leg) {
    cmd[leg].zero();
    for (size_t jidx(0); jidx < cheetah::num_leg_joint; ++jidx) {
      cmd[leg].tauFeedForward[jidx] = _tau_ff[cheetah::num_leg_joint * leg + jidx];
      cmd[leg].qDes[jidx] = _des_jpos[cheetah::num_leg_joint * leg + jidx];
      cmd[leg].qdDes[jidx] = _des_jvel[cheetah::num_leg_joint * leg + jidx];

        cmd[leg].kpJoint(jidx, jidx) = _Kp_joint[jidx];
        cmd[leg].kdJoint(jidx, jidx) = _Kd_joint[jidx];
       
       //if(contact[leg] > 0.){ // Contact
        //cmd[leg].kpJoint(jidx, jidx) = _Kp_joint[jidx];
        //cmd[leg].kdJoint(jidx, jidx) = _Kd_joint[jidx];
      //}else{
        //cmd[leg].kpJoint(jidx, jidx) = _Kp_joint_swing[jidx];
        //cmd[leg].kdJoint(jidx, jidx) = _Kd_joint_swing[jidx];
      //}

    }
  }


  // Knee joint non flip barrier
  for(size_t leg(0); leg<4; ++leg){
    if(cmd[leg].qDes[2] < 0.3){
      cmd[leg].qDes[2] = 0.3;
    }
    if(leg_controller->datas[leg].q[2] < 0.3){
      T knee_pos = leg_controller->datas[leg].q[2];
      cmd[leg].tauFeedForward[2] = 1./(knee_pos * knee_pos + 0.02);
    }
  }
}

template<typename T>
void WBC_Ctrl<T>::_UpdateModel(const quadruped::StateEstimate<T> & state_est,
                               const LegControllerData<T> * leg_data){

  _state.bodyOrientation = state_est.orientation;
  _state.bodyPosition = state_est.position;
  for(size_t i(0); i<3; ++i){
    _state.bodyVelocity[i] = state_est.omegaBody[i];
    _state.bodyVelocity[i+3] = state_est.vBody[i];

    for(size_t leg(0); leg<4; ++leg){
      _state.q[3*leg + i] = leg_data[leg].q[i];
      _state.qd[3*leg + i] = leg_data[leg].qd[i];

      _full_config[3*leg + i + 6] = _state.q[3*leg + i];
    }
  }
  _model.setState(_state);

  if (_dynamics_dispatch.run) {
    // Populate the four feet and dynamics fields read by the remaining CPU
    // tasks. Do not mark the full legacy model's caches as up to date.
    using namespace ula::rigid;
    Result<T> dynamics{};
    _dynamics_dispatch.run(_dynamics_dispatch.context, pack_state(_state), dynamics);
    for (int i = 0; i < kDof; ++i) {
      _model._G[i] = dynamics.gravity.data[i];
      _model._Cqd[i] = dynamics.coriolis.data[i];
      for (int j = 0; j < kDof; ++j) _model._H(i, j) = dynamics.mass.data[i][j];
    }
    for (int leg = 0; leg < kFeet; ++leg) {
      const size_t i = foot_index[leg];
      const auto& ft = dynamics.feet[leg];
      for (int a = 0; a < 3; ++a) {
        _model._pGC[i][a] = ft.position.data[a];
        _model._vGC[i][a] = ft.velocity.data[a];
        _model._Jcdqd[i][a] = ft.bias.data[a];
        for (int j = 0; j < kDof; ++j) _model._Jc[i](a, j) = ft.jacobian.data[a][j];
      }
    }
  } else {
    _model.contactJacobians();
    _model.massMatrix();
    _model.generalizedGravityForce();
    _model.generalizedCoriolisForce();
  }
  _A = _model.getMassMatrix();
  _grav = _model.getGravityForce();
  _coriolis = _model.getCoriolisForce();
  _Ainv = _A.inverse();
}


template class WBC_Ctrl<float>;
template class WBC_Ctrl<double>;
