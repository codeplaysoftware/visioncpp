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

/// \file leaf_node.hpp
/// \brief This file contains the LeafNode struct which is a general representation
/// of our terminal node in the expression tree.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_LEAF_NODE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_LEAF_NODE_HPP_

namespace visioncpp {
namespace internal {
/// \struct LeafNode
/// \brief This file contains LeafNode struct which is a general representation
/// of our terminal node in the expression tree.
/// template parameters:
/// \tparam RHS is the visionMemory allocated to our terminal node. It can be
/// Sycl memory(Sycl memory with map allocator and device only sycl memory );
/// Host Memory or VirtualMemory.
/// \tparam LVL: the level of the node in the expression tree
template <typename RHS, size_t LVL>
struct LeafNode {
  static constexpr bool has_out = false;
  using Type =
      typename internal::OutputMemory<typename RHS::ElementType, RHS::LeafType,
                                      RHS::Cols, RHS::Rows, RHS::Level>::Type;
  static constexpr size_t Level = LVL;
  using RHSExpr = RHS;
  using Scalar = typename Type::Scalar;
  static constexpr size_t LeafType = RHS::LeafType;
  static constexpr size_t RThread = RHS::Rows;
  static constexpr size_t CThread = RHS::Cols;
  using OutType = typename RHS::ElementType;
  static constexpr size_t ND_Category = internal::expr_category::Unary;
  RHS vilibMemory;
  static constexpr bool SubExpressionEvaluationNeeded =
      RHS::SubExpressionEvaluationNeeded;
  static constexpr size_t Operation_type = internal::ops_category::PointOP;
  LeafNode(RHS dt) : vilibMemory(dt), subexpr_execution_reseter(false) {}
  LeafNode() : LeafNode(Type()) {}  // at this point RHS =Type must hold
  /// buffer copy is lightweight no need to pass by ref
  bool subexpr_execution_reseter;
  LeafNode(typename RHS::syclBuffer dt) : LeafNode(RHS(dt)) {}

  void reset(bool reset) { subexpr_execution_reseter = reset; }

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
  /// \tparam DeviceT: type representing the device
  /// function parameters:
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  LeafNode<typename RHS::Type, Level> inline sub_expression_evaluation(
      const DeviceT &dev) {
    return vilibMemory
        .template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev);
  }
  /// \brief set_output function is used to destroy the sycl buffer and manually
  /// allocate the data to the provided pointer. This is used when we needed to
  /// return the value of the device only buffer.
  /// \param ptr: the pointer for manually allocating the data
  /// \return void
  inline void set_output(std::shared_ptr<Scalar> &ptr) {
    (vilibMemory.set_output(ptr));
  }

  /// \brief reset_input is used to manually reset the input value of an input
  /// sycl buffer. This can be used when we are dealing with video streaming and
  /// we want to pass different input stream to the expression.
  /// \return void
  inline void reset_input(Scalar *dt) { vilibMemory.reset_input(dt); }

  /// \brief lock function is used to access the sycl buffer on the host using
  /// a host pointer. Because the host accessor is blocking. We are creating it
  /// dynamically so by calling the lock function. It is the responsibility of
  /// a user to call the unlock function in order to destroy the accessor once its
  /// work is finished. This is useful when we are dealing with video and we
  /// want to display each frame of the video at the end of each iteration.
  /// \return void
  inline void lock() { vilibMemory.lock(); }
  /// \brief The unlock function for destroying the host accessor. It must be
  /// called by user if the lock has been called in order to destroy the host
  /// accessor and release the execution.
  /// \return void
  inline void unlock() { vilibMemory.unlock(); }
};
}  // internal

/// \brief template deduction of LeafNode for buffer/image/host 2d  where the
/// element_category is Struct
template <typename ElemTp, size_t Cols, size_t Rows, size_t MemoryType,
          size_t Sc = scope::Global>
auto terminal(typename internal::MemoryProperties<ElemTp>::ChannelType *dt)
    -> internal::LeafNode<
        internal::VisionMemory<
            true, internal::MemoryProperties<ElemTp>::ElementCategory,
            MemoryType,
            typename internal::MemoryProperties<ElemTp>::ChannelType, Cols,
            Rows, ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize, Sc,
            0>,
        0> {
  return internal::LeafNode<
      internal::VisionMemory<
          true, internal::MemoryProperties<ElemTp>::ElementCategory, MemoryType,
          typename internal::MemoryProperties<ElemTp>::ChannelType, Cols, Rows,
          ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize, Sc, 0>,
      0>(internal::VisionMemory<
      true, internal::MemoryProperties<ElemTp>::ElementCategory, MemoryType,
      typename internal::MemoryProperties<ElemTp>::ChannelType, Cols, Rows,
      ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize, Sc, 0>(dt));
}

/// \brief creation of the device only memory when the type is struct
// device only memory
template <typename ElemTp, size_t Cols, size_t Rows, size_t MemoryType,
          size_t Sc = scope::Global>
auto terminal() -> internal::LeafNode<
    internal::VisionMemory<
        false, internal::MemoryProperties<ElemTp>::ElementCategory, MemoryType,
        typename internal::MemoryProperties<ElemTp>::ChannelType, Cols, Rows,
        ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize, Sc, 0>,
    0> {
  return internal::LeafNode<
      internal::VisionMemory<
          false, internal::MemoryProperties<ElemTp>::ElementCategory,
          MemoryType, typename internal::MemoryProperties<ElemTp>::ChannelType,
          Cols, Rows, ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize,
          Sc, 0>,
      0>(internal::VisionMemory<
      false, internal::MemoryProperties<ElemTp>::ElementCategory, MemoryType,
      typename internal::MemoryProperties<ElemTp>::ChannelType, Cols, Rows,
      ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize, Sc, 0>());
}

/// \brief template deduction of LeafNode where the memory_type is a constant
/// variable and element_category is Struct
template <typename ElemTp, size_t LeafType>
auto terminal(typename internal::MemoryProperties<ElemTp>::ChannelType dt)
    -> internal::LeafNode<
        internal::VisionMemory<
            true, internal::MemoryProperties<ElemTp>::ElementCategory, LeafType,
            typename internal::MemoryProperties<ElemTp>::ChannelType, 1, 1,
            ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize,
            scope::Global, 0>,
        0> {
  return internal::LeafNode<
      internal::VisionMemory<
          true, internal::MemoryProperties<ElemTp>::ElementCategory, LeafType,
          typename internal::MemoryProperties<ElemTp>::ChannelType, 1, 1,
          ElemTp, internal::MemoryProperties<ElemTp>::ChannelSize,
          scope::Global, 0>,
      0>(internal::VisionMemory<
      true, internal::MemoryProperties<ElemTp>::ElementCategory, LeafType,
      typename internal::MemoryProperties<ElemTp>::ChannelType, 1, 1, ElemTp,
      internal::MemoryProperties<ElemTp>::ChannelSize, scope::Global, 0>(dt));
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_LEAF_NODE_HPP_
