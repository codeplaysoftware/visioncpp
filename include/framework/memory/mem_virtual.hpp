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

/// \file mem_virtual.hpp
/// This file contains the VirtualMemory struct. VirtualMemory struct is nothing
/// but a future LeafNode representing the result of the subexpression passed to
/// it to be executed. It is used by user to schedule any arbitrary
/// subexpression to be executed in a separate kernel. This will allow a user to
/// manually break the expression tree at compile time.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VIRTUAL_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VIRTUAL_HPP_

namespace visioncpp {
namespace internal {
/// \struct VirtualMemory
/// \brief VirtualMemory struct is nothing but a future LeafNode representing
/// the result of the subexpression passed to it to be executed. It is used by
/// the user to schedule any arbitrary subexpression to be executed in a separate
/// kernel. This will allow a user to manually break the expression tree at
/// compile time.
/// template parameters
/// \tparam PlcType: represent the policyType for executing the subexpression
/// \tparam Node: the subexpression tree needed to be executed
/// template parameters:
/// \tparam LC: is the column size of local memory
/// \tparam LR: is the row size of local memory
/// \tparam LCT: is the column size of workgroup
/// \tparam LRT: is the row size of workgroup
template <bool PlcType, typename Node, size_t LC, size_t LR, size_t LCT,
          size_t LRT>
struct VirtualMemory {
  static constexpr bool policyType = PlcType;
  using Type = typename Node::Type;
  using Scalar = typename Type::Scalar;
  using ElementType = typename Type::ElementType;
  template <cl::sycl::access::mode acMd>
  using Accessor = typename Type::template Accessor<acMd>;
  static constexpr size_t Rows = Type::Rows;
  static constexpr size_t Cols = Type::Cols;
  static constexpr size_t Channels = Type::Channels;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr bool SubExpressionEvaluationNeeded = true;
  static constexpr size_t Level = Node::Level;
  Node subTree;
  using syclBuffer = Node;
  VirtualMemory(Node nd) : subTree(nd){};
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
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC1, size_t LR1, size_t LRT1, size_t LCT1,
            typename DeviceT>
  internal::LeafNode<Type, Level> sub_expression_evaluation(
      const DeviceT &dev) {
    // this is manually breaking so we have to break and we cannot use the
    // condition used in the subtree for evalifneeded
    auto lhs = internal::LeafNode<Type, Level>();
    auto rhs =
        subTree.template sub_expression_evaluation<false, LC1, LR1, LRT1, LCT1>(
            dev);
    auto a =
        internal::Assign<decltype(lhs), decltype(rhs),
                         decltype(lhs)::Type::Cols, decltype(lhs)::Type::Rows,
                         decltype(lhs)::Type::LeafType,
                         1 + internal::tools::StaticIf<
                                 (decltype(lhs)::Level > decltype(rhs)::Level),
                                 decltype(lhs), decltype(rhs)>::Type::Level>(
            lhs, rhs);
    execute<PlcType, LC, LR, LCT, LRT>(a, dev);
    return lhs;
  }
};
}
/// brief function schedule is a template deduction function for VirtualMemory
/// when the local memory and workgroup size is defined by a user.
template <size_t plcType, size_t LWV, size_t LHV, size_t LCTV, size_t LRTV,
          typename Type>
auto schedule(Type dt) -> decltype(internal::LeafNode<
    internal::VirtualMemory<plcType, Type, LWV, LHV, LCTV, LRTV>, Type::Level>(
    dt)) {
  return internal::LeafNode<
      internal::VirtualMemory<plcType, Type, LWV, LHV, LCTV, LRTV>,
      Type::Level>(
      internal::VirtualMemory<plcType, Type, LWV, LHV, LCTV, LRTV>(dt));
}
/// brief function schedule is a template deduction function for VirtualMemory
/// when the local memory and workgroup size is default
template <size_t plcType, typename Type>
auto schedule(Type dt)
    -> decltype(internal::LeafNode<internal::VirtualMemory<plcType, Type>,
                                   Type::Level>(dt)) {
  return internal::LeafNode<internal::VirtualMemory<plcType, Type>,
                            Type::Level>(
      internal::VirtualMemory<plcType, Type>(dt));
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_VIRTUAL_HPP_