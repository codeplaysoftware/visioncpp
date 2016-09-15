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

/// \file mem_vision.hpp
/// \brief this file contains the memory struct used to store {Sycl buffer ,
/// Host pointer or Sycl image}.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VISION_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VISION_HPP_

namespace visioncpp {
namespace internal {

/// \struct VisionMemory
/// \brief VisionMemory is the memory container of visioncpp. It has been
/// defined general enough that can be used for not only sycl buffer and image,
/// but also for host backend such as openmp.
/// template parameters:
/// \tparam MapAllocator:  a boolean value representing whether or not the buffer
/// has map allocator
/// \tparam ScalarType: represent whether the element of the memory is Basic or
/// Struct.
/// \tparam MemoryType: represent the type of the memory.(e.g. Buffer2D,
/// Buffer1D, Image, Host)
/// \tparam Sclr: represent the type of each channel of the elements of the
/// memory
/// \tparam Col: represents the column size of the memory
/// \tparam Row: represents the Row size of the memory
/// \tparam ElementTp: represents the types of each element inside the memory
/// \tparam Elements: represents the number of channels for each element
/// \tparam Sc: represents the memory target on the device (global memory or
/// constant memory)- Default is global
// LVL : represent the level of the memory in the expression tree
template <bool MapAllocator, size_t ScalarType, size_t MemoryType,
          typename Sclr, size_t Col, size_t Row, typename ElementTp,
          size_t Elements, size_t Sc, size_t LVL>
struct VisionMemory {
 public:
  static constexpr size_t Rows = Row;
  static constexpr size_t Cols = Col;
  static constexpr size_t Channels = Elements;
  static constexpr size_t LeafType = MemoryType;
  static constexpr size_t Dim = MemDimension<MemoryType>::Dim;
  static constexpr bool SubExpressionEvaluationNeeded = false;
  static constexpr size_t MemoryCategory = ScalarType;
  static constexpr bool HasMapAllocator = MapAllocator;
  static constexpr size_t scope = Sc;
  using ElementType = ElementTp;
  using Scalar = Sclr;
  using AccessType = ElementType;  // it should be changed later on when it is
  // going to be used for image. This is the reason it has been added now
  template <cl::sycl::access::mode acMd>
  using Accessor = typename SyclAccessor<LeafType, Dim, acMd, ElementType,
                                         Scalar, scope>::Accessor;
  // FIXME:: we have to make it work for image as well , the cl::sycl must be
  // removed as well
  template <cl::sycl::access::mode acMd>
  using HostAccessor =
      typename SyclAccessor<LeafType, Dim, acMd, ElementType, Scalar,
                            scope::Host_Buffer>::Accessor;
  using Type = VisionMemory<HasMapAllocator, ScalarType, LeafType, Scalar, Cols,
                            Rows, ElementTp, Elements, scope, LVL>;
  static constexpr size_t Level = LVL;
  using syclBuffer =
      typename SyclMem<HasMapAllocator, LeafType, Dim, ElementType>::Type;
  std::shared_ptr<syclBuffer> syclData;
  std::shared_ptr<HostAccessor<cl::sycl::access::mode::read>> hostAcc;

  static constexpr size_t used_memory() {
    return (Rows * Cols * Channels * sizeof(Scalar));
  }

  static constexpr size_t get_size() { return (Type::used_memory()); }

  VisionMemory(Scalar *dt) {
    create_sycl_buffer<LeafType, ElementType, Scalar>(
        syclData, dt, get_range<Dim>(Rows, Cols));
  }
  /// buffer copy is lightweight no need to pass by ref
  VisionMemory(syclBuffer dt) { syclData = std::make_shared<syclBuffer>(dt); }

  VisionMemory() {
    create_sycl_buffer<LeafType, ElementType, Scalar>(
        syclData, get_range<Dim>(Rows, Cols));
  }
  /// sub_expression_evaluation
  /// \brief This function is used to break the expression tree whenever
  /// necessary. The decision for breaking the tree will be determined based on
  /// the static parameter called SubExpressionEvaluationNeeded. When this is
  /// set to true, the sub_expression_evaluation is called recursively from the
  /// root of the tree. Each node based on their parent decision will decide to
  /// launch a kernel for itself. Also, they decide for each of their children
  /// whether or not to launch a kernel separately.
  /// template parameters:
  ///\tparam ForcedToExec : a boolean value representing the decision made by
  /// the parent of this node for launching a kernel.
  /// \tparam LC: is the column size of local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LR: is the row size of local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LCT: is the column size of workgroup
  /// \tparam LRT: is the row size of workgroup
  /// function parameters:
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  internal::LeafNode<Type, Level> inline sub_expression_evaluation(
      const DeviceT &) {
    return internal::LeafNode<Type, Level>(*this);
  }
  /// \brief This function is used to get the device access on the memory.
  /// The mode of the accessor represents the type of the access.
  /// template parameters:
  /// \tparam accMode : represents the sycl type of access
  /// function parameters:
  /// \param cgh: sycl command group handler
  /// \return Accessor
  template <cl::sycl::access::mode accMode>
  Accessor<accMode> get_device_accessor(cl::sycl::handler &cgh) {
    return Accessor<accMode>(*syclData, cgh);
  }
  /// \brief reset_input is used to manually reset the input value of an input
  /// sycl buffer. This can be used when we are dealing with video streaming and
  /// we want to pass different input stream to the expression.
  /// \return void
  void reset_input(Scalar *dt) {
    buffer_update<LeafType, Rows, Cols, ElementType, Scalar>(syclData, dt);
  }

  /// \brief set_output function is used to destroy the sycl buffer and manually
  /// allocated the data to the provided pointer. This is used when we needed to
  /// return the value of the device-only buffer.
  /// \param ptr: the pointer for manually allocating the data
  /// \return void
  void set_output(std::shared_ptr<Scalar> &ptr) {
    syclData.get()->set_final_data(ptr);
    syclData.reset();  // that my needed to be added
  }
  /// \brief lock function is used to access the sycl buffer on the host using
  /// a host pointer. Because the host accessor is blocking we are creating it
  /// dynamically so by calling the lock function. It is the responsibility of
  /// a user to call unlock function in order to destroy the accessor once his
  /// work is finished. This is useful when we are dealing with video and we
  /// want to display each frame of the video at the end of each iteration.
  /// \return void
  void lock() {
    hostAcc = std::make_shared<HostAccessor<cl::sycl::access::mode::read>>(
        HostAccessor<cl::sycl::access::mode::read>(*syclData));
  }
  /// \brief The unlock function for destroying the host accessor. It must be
  /// called by user if the lock has been called in order to destroy the host
  /// accessor and release the execution.
  /// \return void
  void unlock() { hostAcc.reset(); }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VISION_HPP_
