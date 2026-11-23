cmake_minimum_required(VERSION 3.18)
project(ula_controller_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CORE_SCALAR_FLOAT "Instantiate the controller in single precision" ON)
if(CORE_SCALAR_FLOAT)
  add_compile_definitions(ULA_CORE_SCALAR_FLOAT)
endif()

# QP solver used by WBIC: quadprog = Goldfarb QuadProg++ (default, double,
# heap), gi = src/qp/gi_qp.hh (fixed storage, bounded iterations, scalar type
# of the controller).
set(ULA_QP_SOLVER "quadprog" CACHE STRING "WBIC QP solver: quadprog or gi")
set_property(CACHE ULA_QP_SOLVER PROPERTY STRINGS quadprog gi)
if(ULA_QP_SOLVER STREQUAL "gi")
  add_compile_definitions(ULA_QP_SOLVER_GI)
elseif(NOT ULA_QP_SOLVER STREQUAL "quadprog")
  message(FATAL_ERROR "ULA_QP_SOLVER must be quadprog or gi")
endif()

find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
  src/dynamics/*.cpp src/dynamics/*.cc
  src/controllers/*.cpp src/controllers/*.cc
  src/estimators/*.cc
  src/input/*.cpp
  src/parameters/*.cc
  src/utilities/*.cpp
)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/QuadrupedSingleton\\.cc$")

add_library(tinyxml2 STATIC third-party/tinyxml2/tinyxml2.cpp)
target_include_directories(tinyxml2 PUBLIC third-party/tinyxml2)

add_library(goldfarb STATIC
  third-party/Goldfarb_Optimizer/Array.cc
  third-party/Goldfarb_Optimizer/QuadProg++.cc)
target_include_directories(goldfarb PUBLIC third-party/Goldfarb_Optimizer)
target_link_libraries(goldfarb PUBLIC Eigen3::Eigen)

add_library(ula_core STATIC ${CORE_SOURCES})
target_include_directories(ula_core PUBLIC src/include src third-party)
target_include_directories(ula_core SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ula_core PUBLIC goldfarb tinyxml2 Eigen3::Eigen)
target_compile_options(ula_core PUBLIC -DEIGEN_DONT_VECTORIZE)

add_executable(core_demo src/core_demo.cc)
target_link_libraries(core_demo PRIVATE ula_core)

add_executable(parity_dump tools/parity_dump.cc)
target_include_directories(parity_dump PRIVATE tools)
target_link_libraries(parity_dump PRIVATE ula_core)

add_executable(alloc_audit tools/alloc_audit.cc)
target_link_libraries(alloc_audit PRIVATE ula_core)
target_link_options(alloc_audit PRIVATE -rdynamic)

add_executable(qp_dump tools/qp_dump.cc)
target_include_directories(qp_dump PRIVATE tools)
target_link_libraries(qp_dump PRIVATE ula_core)

add_executable(parity_compare tools/parity_compare.cc)

add_executable(qp_check tools/qp_check.cc)
target_include_directories(qp_check PRIVATE src third-party tools)
target_link_libraries(qp_check PRIVATE goldfarb)

add_library(ula_cpu_controller STATIC src/port/cpu_controller.cc)
target_link_libraries(ula_cpu_controller PUBLIC ula_core)
add_executable(cpu_replay tools/cpu_replay.cc)
target_link_libraries(cpu_replay PRIVATE ula_cpu_controller)
add_library(ula_batch SHARED src/port/batch_api.cc)
target_link_libraries(ula_batch PRIVATE ula_cpu_controller)

option(ULA_ENABLE_CUDA_REPLAY "Build the Go2 CUDA-hybrid parity executable" OFF)
if(ULA_ENABLE_CUDA_REPLAY)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(ula_cuda_replay STATIC src/port/cuda_replay_stages.cu)
  set_target_properties(ula_cuda_replay PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(ula_cuda_replay PRIVATE src)
  target_link_libraries(ula_cuda_replay PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver Eigen3::Eigen)
  target_compile_definitions(ula_cuda_replay PRIVATE EIGEN_DONT_VECTORIZE)
  # Keep the replay arithmetic non-fused. Eigen's device headers also use
  # constexpr functions from the host standard library.
  target_compile_options(ula_cuda_replay PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--expt-relaxed-constexpr>)
  target_sources(ula_cuda_replay PRIVATE src/port/batch_cuda.cu src/port/solver_audit.cc)
  target_compile_definitions(ula_cuda_replay PRIVATE ULA_BATCH_CUDA)
  target_compile_definitions(ula_batch PRIVATE ULA_BATCH_CUDA)
  target_link_libraries(ula_batch PRIVATE ula_cuda_replay)
  add_executable(cuda_replay tools/cpu_replay.cc)
  target_compile_definitions(cuda_replay PRIVATE ULA_CUDA_REPLAY)
  target_link_libraries(cuda_replay PRIVATE ula_cpu_controller ula_cuda_replay)
endif()

include(CTest)
if(BUILD_TESTING)
  set(GO2_URDF "${CMAKE_CURRENT_SOURCE_DIR}/models/go2/go2.urdf")
  add_test(NAME wbc_stance COMMAND core_demo "${GO2_URDF}")
  add_test(NAME wbc_reference_sweep COMMAND parity_dump "${GO2_URDF}"
           "${CMAKE_CURRENT_BINARY_DIR}/wbc_reference_sweep.bin")

  add_executable(sim_motor_test tests/sim_motor_test.cc)
  target_include_directories(sim_motor_test PRIVATE src)
  add_test(NAME sim_motor COMMAND sim_motor_test)

  add_executable(wbic_torque_test tests/wbic_torque_test.cc)
  target_include_directories(wbic_torque_test PRIVATE src)
  add_test(NAME wbic_torque COMMAND wbic_torque_test)

  add_executable(rigid_body_test tests/rigid_body_test.cc)
  target_link_libraries(rigid_body_test PRIVATE ula_core)
  add_test(NAME rigid_body COMMAND rigid_body_test "${GO2_URDF}")
  if(ULA_ENABLE_CUDA_REPLAY)
    add_executable(cuda_rigid_body_test tests/rigid_body_test.cc)
    target_compile_definitions(cuda_rigid_body_test PRIVATE ULA_CUDA_DYNAMICS_TEST)
    target_link_libraries(cuda_rigid_body_test PRIVATE ula_core ula_cuda_replay)
    add_test(NAME go2_cuda_dynamics COMMAND cuda_rigid_body_test "${GO2_URDF}")
  endif()

  add_executable(cpu_controller_test tests/cpu_controller_test.cc)
  target_link_libraries(cpu_controller_test PRIVATE ula_cpu_controller)
  add_test(NAME cpu_controller COMMAND cpu_controller_test "${GO2_URDF}")

  find_package(Python3 3.10 QUIET COMPONENTS Interpreter)
  set(ULA_REPLAY_REFERENCE "" CACHE FILEPATH "Optional recorded Go2 JSONL trace for the CPU parity tests")
  if(Python3_Interpreter_FOUND)
    add_test(NAME controller_replay_checker
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_controller_replay.py")
    if(ULA_REPLAY_REFERENCE)
      function(add_go2_replay_check backend executable)
        set(candidate "${CMAKE_CURRENT_BINARY_DIR}/go2_${backend}_candidate.jsonl")
        add_test(NAME go2_${backend}_replay
                 COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/replay_cpu.py"
                         "${ULA_REPLAY_REFERENCE}" "${candidate}"
                         --executable $<TARGET_FILE:${executable}> --urdf "${GO2_URDF}" ${ARGN})
        set_tests_properties(go2_${backend}_replay PROPERTIES FIXTURES_SETUP go2_${backend}_candidate)
        add_test(NAME go2_${backend}_parity
                 COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tools/reference/compare_cpu.sh"
                         "${ULA_REPLAY_REFERENCE}" "${candidate}")
        set_tests_properties(go2_${backend}_parity PROPERTIES FIXTURES_REQUIRED go2_${backend}_candidate)
      endfunction()
      add_go2_replay_check(cpu cpu_replay)
      add_go2_replay_check(host_stages cpu_replay --host-stages)
      function(add_precision_replay backend executable mode)
        add_test(NAME go2_${backend}_replay COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tools/replay_cpu.py" "${ULA_REPLAY_REFERENCE}"
                 "${CMAKE_CURRENT_BINARY_DIR}/go2_${backend}_candidate.jsonl"
                 --executable $<TARGET_FILE:${executable}> --urdf "${GO2_URDF}" "--mode=${mode}")
        set_tests_properties(go2_${backend}_replay PROPERTIES FIXTURES_SETUP go2_${backend}_candidate)
      endfunction()
      function(add_precision_check backend arg)
        add_test(NAME go2_${backend}_precision COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_gpu.py"
                 "${CMAKE_CURRENT_BINARY_DIR}/go2_double_reference_candidate.jsonl"
                 "${CMAKE_CURRENT_BINARY_DIR}/go2_${backend}_candidate.jsonl" "${arg}" ${ARGN})
        set_tests_properties(go2_${backend}_precision PROPERTIES FIXTURES_REQUIRED
                             "go2_${backend}_candidate;go2_double_reference_candidate;go2_cpu_candidate")
      endfunction()
      add_precision_replay(double_reference cpu_replay --reference)
      add_precision_replay(full_double cpu_replay --full-double)
      add_precision_replay(full_host cpu_replay --full)
      add_precision_check(full_double --double-check)
      add_precision_check(full_host --cpu-float "${CMAKE_CURRENT_BINARY_DIR}/go2_cpu_candidate.jsonl")
      if(ULA_ENABLE_CUDA_REPLAY)
        add_precision_replay(full_cuda cuda_replay --full-cuda-audit)
        add_precision_replay(full_cuda_repeat cuda_replay --full-cuda)
        add_precision_check(full_cuda --cpu-float "${CMAKE_CURRENT_BINARY_DIR}/go2_cpu_candidate.jsonl")
        add_test(NAME go2_full_cuda_repeatability COMMAND "${CMAKE_COMMAND}" -E compare_files
                 "${CMAKE_CURRENT_BINARY_DIR}/go2_full_cuda_candidate.jsonl"
                 "${CMAKE_CURRENT_BINARY_DIR}/go2_full_cuda_repeat_candidate.jsonl")
        set_tests_properties(go2_full_cuda_repeatability PROPERTIES FIXTURES_REQUIRED
                             "go2_full_cuda_candidate;go2_full_cuda_repeat_candidate")
      endif()
    endif()
  elseif(ULA_ENABLE_CUDA_REPLAY AND ULA_REPLAY_REFERENCE)
    message(FATAL_ERROR "Python 3.10+ is required to replay the recorded CUDA test trace")
  endif()
endif()
