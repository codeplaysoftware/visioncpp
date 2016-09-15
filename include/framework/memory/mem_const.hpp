// This file is part of VisionCpp, a lightweight C++ template library
// for computer vision and image processing.
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

/// \file mem_const.hpp
/// \brief specialisations of SyclMem type when we are passing constant variable
/// to the device. In this case we are not using syclbuffer which can be stored
/// on device memory. we have created struct supporting c++ standard layout.
/// Therefore, the constant variable can be detectable by sycl compiler on the
/// device side and each thread can have a copy of it on the private memory.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_CONST_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_CONST_HPP_

namespace visioncpp {
namespace internal {
/// \struct ConstMemory
/// \brief represents a c++ standard layout for detecting constant variable on
/// the device.
/// template parameters:
/// \tparam T: the type of the constant variable
template <typename T>
struct ConstMemory {
  ConstMemory(T r) : r(r) {}
  template <typename T2>
  ConstMemory(T r, T2 &x)
      : ConstMemory(r) {}
  using value_type = T;
  const T r;
  const value_type &operator[](cl::sycl::nd_item<2> itemID) const { return r; }
  const value_type &operator[](cl::sycl::nd_item<1> itemID) const { return r; }
  const value_type &operator[](cl::sycl::id<2> itemID) const { return r; }
  const value_type &operator[](int itemID) const { return r; }
  const value_type &operator[](size_t itemID) const { return r; }
  /// this function is used to mimic the getpointer function used by evaluator
  /// expression, in order to access a ConstMemory Node.
  /// \return ConstMemory
  const ConstMemory<T> &get_pointer() const { return *this; }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_CONST_HPP_