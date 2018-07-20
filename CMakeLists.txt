#Minimum CMake version
cmake_minimum_required(VERSION 3.2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(PASS_TO_DEVICE_COMPILER ${CMAKE_CXX_FLAGS_RELEASE})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(PASS_TO_DEVICE_COMPILER ${CMAKE_CXX_FLAGS_DEBUG})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
  set(PASS_TO_DEVICE_COMPILER ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "MinSizeRel")
  set(PASS_TO_DEVICE_COMPILER ${CMAKE_CXX_FLAGS_MINSIZEREL})
else()
  message(ERROR " Unknown value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_MODULE_PATH
${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

find_package(PythonInterp 3.2 REQUIRED)
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

#setup
include(FindCXX11)
include(FindOpenCL)

# find pthread
find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  set_property(TARGET my_app PROPERTY COMPILE_OPTIONS "pthread")
  set_property(TARGET my_app PROPERTY INTERFACE_COMPILE_OPTIONS "pthread")
endif()

if(NOT VISIONCPP_DOC_ONLY)
  include(FindComputeCpp)
  #internal dep
  include(testgen)
  #projects
  include(tests)
  include(examples)
endif()

#docs
include(doxygen)
