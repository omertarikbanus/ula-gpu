cmake_minimum_required(VERSION 3.16)
project(ula_controller_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CORE_SCALAR_FLOAT "Instantiate the controller in single precision" ON)
if(CORE_SCALAR_FLOAT)
  add_compile_definitions(ULA_CORE_SCALAR_FLOAT)
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
