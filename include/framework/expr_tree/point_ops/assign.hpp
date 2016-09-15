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

/// \file assign.hpp
/// \brief This file contains the Assign struct which is used to allocate the result
/// of the right hand side expression (RHS) to the left hand side expression
/// LHS. LHS is always a leaf node.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_ASSIGN_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_ASSIGN_HPP_

namespace visioncpp {
namespace internal {
/// \struct Assign
/// \brief Assign is used to allocate the result of the right hand side
/// expression (RHS) to the left hand side expression LHS. LHS is always a
/// leaf node. It can be used for PointOP, NeighbourOP, and GlobalNeighbourOP.
/// template parameters:
/// \tparam LHS is the output leafNode
/// \tparam RHS is the right hand side expression
/// \tparam Cols: determines the column size of the output
/// \tparam Rows: determines the row size of the output
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree
template <typename LHS, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL>
struct Assign {
  static constexpr bool has_out = true;
  using OutType = typename LHS::OutType;
  using Type = typename LHS::Type;
  using LHSExpr = LHS;
  using RHSExpr = RHS;
  static constexpr size_t Level = LVL;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr size_t RThread = RHS::RThread;
  static constexpr size_t CThread = RHS::CThread;
  static constexpr size_t ND_Category = expr_category::Binary;
  static constexpr bool SubExpressionEvaluationNeeded =
      LHS::SubExpressionEvaluationNeeded || RHS::SubExpressionEvaluationNeeded;
  // may be this can be passed based on the shape of the cope . If
  // source is equal to dest that can be passed as false
  static constexpr size_t Operation_type = RHS::Operation_type;
  template <typename TmpLHS, typename TmpRHS>
  using ExprExchange = Assign<TmpLHS, TmpRHS, Cols, Rows, LfType, LVL>;

  LHS lhs;
  RHS rhs;
  bool subexpr_execution_reseter;
  Assign(LHS lhsArg, RHS rhsArg)
      : lhs(lhsArg), rhs(rhsArg), subexpr_execution_reseter(false) {}
  void reset(bool reset) {
    lhs.reset(reset);
    rhs.reset(reset);
    subexpr_execution_reseter = reset;
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
  /// \tparam DeviceT: type representing the device
  /// function parameters:
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  auto inline sub_expression_evaluation(const DeviceT &dev) -> decltype(
      execute_expr<false, false, ExprExchange<LHS, RHS>, LC, LR, LCT, LRT>(
          lhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev),
          rhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev),
          dev)) {
    return execute_expr<false, false, ExprExchange<LHS, RHS>, LC, LR, LCT, LRT>(
        lhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev),
        rhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev),
        dev);
  }
};
}  // internal

/// assign function
/// \brief This function is used to deduce the Assign struct.
template <typename LHS, typename RHS>
auto assign(LHS lhs, RHS rhs)
    -> internal::Assign<LHS, RHS, LHS::Type::Cols, LHS::Type::Rows,
                        LHS::Type::LeafType,
                        1 + internal::tools::StaticIf<(LHS::Level > RHS::Level),
                                                      LHS, RHS>::Type::Level> {
  return internal::Assign<
      LHS, RHS, LHS::Type::Cols, LHS::Type::Rows, LHS::Type::LeafType,
      1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                    RHS>::Type::Level>(lhs, rhs);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_ASSIGN_HPP_
