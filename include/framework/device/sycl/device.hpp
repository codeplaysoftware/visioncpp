// This file is part of VisionCpp, a lightweight C++ template library
// for compute vision and image processing.
//
// Copyright (C) 2016 Codeplay Software Limited. All Rights Reserved.
//
// Contact: visioncpp@codeplay.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \file sycl/device.hpp
/// \brief This file contains the include headers for sycl devices
#pragma once

namespace visioncpp {
namespace internal {
/// \struct DeviceSelector
/// \brief this class is used to define different types of device for sycl
template <device DV>
struct DeviceSelector;

/// \brief specialisation of the device_selector for sycl when the device type
/// is cpu
template <>
struct DeviceSelector<device::cpu> {
  using Type = cl::sycl::intel_selector;
};
/// \brief specialisation of the device_selector for sycl when the device type
/// is gpu
template <>
struct DeviceSelector<device::gpu> {
  using Type = cl::sycl::gpu_selector;
};
/// \brief specialisation of the device_selector for sycl when the device type
/// is host
template <>
struct DeviceSelector<device::host> {
  using Type = cl::sycl::host_selector;
};
}  // end internal
}  // end visioncpp
#include "extract_accessors.hpp"
#include "sycl_device.hpp"
