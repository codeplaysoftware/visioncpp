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

/// \file mem_coordinate.hpp
/// \brief This files contains the Coordinate struct which is used to specify
/// local/global offset for local/global access to the local/global memory for
/// each thread on the device.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_COORDINATE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_COORDINATE_HPP_

// these two must be switched for row-major
namespace visioncpp {
namespace internal {
/// \brief Color / Row-major option
namespace mem_dim {
/// \brief when the sycl change the dim this should be applied and work
/// ColDim=1 and RowDim==0;
static constexpr size_t ColDim = 0;
static constexpr size_t RowDim = 1;
};
/// \struct Coordinate
/// \brief Coordinate is used to specify
/// local/global offset for local/global access to the local/global memory for
/// each thread on the device.
/// template parameters:
/// \tparam LC The column size for local memory
/// \tparam LR The Row size for the local memory
/// \tparam ItemID provided by sycl
template <size_t LC, size_t LR, typename ItemID>
struct Coordinate {
  Coordinate(ItemID itemID)
      : itemID(itemID),
        cLRng(itemID.get_local_range()[mem_dim::ColDim]),
        rLRng(itemID.get_local_range()[mem_dim::RowDim]),
        pointOp_gc(0),
        pointOp_gr(0),
        g_c(itemID.get_local(mem_dim::ColDim) +
            (itemID.get_group(mem_dim::ColDim) * ((LC / cLRng) * cLRng))),
        g_r(itemID.get_local(mem_dim::RowDim) +
            itemID.get_group(mem_dim::RowDim) * ((LR / rLRng) * rLRng)),
        l_c(itemID.get_local(mem_dim::ColDim)),
        l_r(itemID.get_local(mem_dim::RowDim)) {}

  /// function barrier is used to call sycl local barrier for local threads
  /// \return void
  inline void barrier() {
    itemID.barrier(cl::sycl::access::fence_space::local_space);
  }
  // function global_barrier is used to call sycl local barrier for global
  // threads
  /// \return void
  inline void global_barrier() {
    itemID.barrier(cl::sycl::access::fence_space::global_space);
  }

  ItemID itemID;
  size_t cLRng;
  size_t rLRng;
  size_t pointOp_gc;
  size_t pointOp_gr;
  size_t g_c;
  size_t g_r;
  size_t l_c;
  size_t l_r;
};
/// deduction function for Coordinate
template <size_t LC, size_t LR, typename ItemID>
Coordinate<LC, LR, ItemID> memLocation(ItemID itemID) {
  return Coordinate<LC, LR, ItemID>(itemID);
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_COORDINATE_HPP_
