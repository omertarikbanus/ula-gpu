#ifndef LOCOMOTION_CONTROLLER
#define LOCOMOTION_CONTROLLER


#include <controllers/wbc/WBC_Ctrl/WBC_Ctrl.hpp>

#include "control/LocomotionCtrlData.hh"

template<typename T>
class LocomotionCtrl: public WBC_Ctrl<T>{
  public:
    LocomotionCtrl(FloatingBaseModel<T> model);
    virtual ~LocomotionCtrl();

  protected:
    virtual void _ContactTaskUpdate(
        void * input, const MIT_UserParameters* user_params);
    virtual void _ContactTaskUpdateTEST(void * input,
                                        const MIT_UserParameters* user_params);
    void _ParameterSetup(const MIT_UserParameters* param);
    void _CleanUp();
    LocomotionCtrlData<T>* _input_data;

    Task<T>* _body_pos_task;
    Task<T>* _body_ori_task;

    Task<T>* _foot_task[4];
    ContactSpec<T>* _foot_contact[4];

    Vec3<T> pre_foot_vel[4];

    Vec3<T> _Fr_result[4];
    Quat<T> _quat_des;
};

#endif
