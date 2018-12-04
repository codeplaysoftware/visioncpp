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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(PythonInterp 3.2 REQUIRED)
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

#setup the minimum required cxx version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#find pthread
find_package(Threads REQUIRED)

if(NOT VISIONCPP_DOC_ONLY)
  find_package(ComputeCpp REQUIRED)
  #internal dep
  include(testgen)
  #projects
  include(tests)
  include(examples)
endif()

#docs
include(doxygen)
