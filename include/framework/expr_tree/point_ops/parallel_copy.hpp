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

/// \file parallel_copy.hpp
/// \brief This file contains the ParallelCopy struct which is used to allocate the
/// partial result of the right-hand side expression (RHS) to the (partial block
/// of) left-hand side expression LHS is always a leaf node.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_PARALLEL_COPY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_PARALLEL_COPY_HPP_

namespace visioncpp {
namespace internal {
/// \struct ParallelCopy
/// \brief parallel copy is used to allocate the partial result of the right-
/// hand side expression (RHS) to the (partial block of) left hand side
/// expression LHS is always a leaf node. It can be used for PointOP, NeighbourOP,
/// and GlobalNeighbourOP.
/// template parameters:
/// \tparam LHS is the output leafNode
/// \tparam RHS is the right-hand side expression
/// \tparam Cols: determines the column size of the output
/// \tparam Rows: determines the row size of the output
/// \tparam OffsetColIn: starting column offset for RHS result
/// \tparam OffsetRowIn: starting Row offset for RHS result
/// \tparam OffsetColOut: starting column offset for LHS node
/// \tparam OffsetRowOut: starting Row offset for LHS node
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree

template <typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t OffsetColIn, size_t OffsetRowIn, size_t OffsetColOut,
          size_t OffsetRowOut, size_t LfType, size_t LVL>
struct ParallelCopy {
  static constexpr bool has_out = true;
  using OutType = typename LHS::OutType;
  using Type = typename LHS::Type;
  static constexpr size_t Level = LVL;
  static constexpr size_t RThread =
      internal::tools::IfConst<internal::tools::IfNode<RHS>::Is_LeafNode, Rows,
                               RHS::RThread>::Value;  // Rows;
  static constexpr size_t CThread =
      internal::tools::IfConst<internal::tools::IfNode<RHS>::Is_LeafNode, Cols,
                               RHS::CThread>::Value;  // Cols;
  static constexpr size_t ND_Category = expr_category::Binary;
  static constexpr size_t LeafType = Type::LeafType;
  using RHSExpr = RHS;
  using LHSExpr = LHS;
  template <typename TmpLHS, typename TmpRHS>
  using ExprExchange =
      internal::ParallelCopy<TmpLHS, TmpRHS, Cols, Rows, OffsetColIn,
                             OffsetRowIn, OffsetColOut, OffsetRowOut, LfType,
                             LVL>;

  // Maybe this can be passed based on the shape of the copy. If
  // source is equal to dest that can be passed as false,
  // if the type is assign and neighbour op
  // cols and rows should be the same size as cols and rows of the RHS output.
  // and the start offset for input should be 0,0

  static constexpr size_t Operation_type =
      RHS::Operation_type;  // internal::ops_category::PointOP;
  static constexpr bool SubExpressionEvaluationNeeded =
      (Operation_type == internal::ops_category::NeighbourOP &&
       (Cols != RHS::Type::Cols || Rows != RHS::Type::Rows ||
        OffsetColIn != 0 || OffsetRowIn != 0)) ||
      RHS::SubExpressionEvaluationNeeded;
  LHS lhs;
  RHS rhs;
  bool subexpr_execution_reseter;
  ParallelCopy(LHS lhsArg, RHS rhsArg)
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
  /// function parameters:
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  LHS inline sub_expression_evaluation(const DeviceT &dev) {
    // Here, again, we cannot use the general subexpr. For two reasons:
    // partial assign is always the root of a tree. So it does not come here and
    // there is no eval_expr provided for that.
    // Secondly even if we come here the lhs is passed as a type and there is no
    // intermediate type for that. We have to run it differently from the
    // intermediate root node.
    auto eval_sub =
        rhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev);
    // through template instantiation it executes the kernel when it is a leaf node.
    auto intermediate_output =
        SubExprRes<LC, LR, LCT, LRT, 1 + LVL, decltype(eval_sub)>::get(eval_sub,
                                                                       dev);
    internal::fuse<LC, LR, LCT, LRT>(
        internal::ParallelCopy<
            LHS, decltype(intermediate_output), Cols, Rows, OffsetColIn,
            OffsetRowIn, OffsetColOut, OffsetRowOut, LHS::LeafType,
            1 + internal::tools::StaticIf<
                    (LHS::Level > decltype(intermediate_output)::Level), LHS,
                    decltype(intermediate_output)>::Type::Level>(
            lhs, intermediate_output),
        dev);
    return lhs;
  }
};
}  // internal

/// partial_assign function
/// \brief This function is used to deduce the ParallelCopy struct.
template <size_t Cols, size_t Rows, size_t OffsetColIn, size_t OffsetRowIn,
          size_t OffsetColOut, size_t OffsetRowOut, typename LHS, typename RHS>
auto partial_assign(LHS lhs, RHS rhs) -> internal::ParallelCopy<
    LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn, OffsetColOut, OffsetRowOut,
    LHS::LeafType, 1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                                 RHS>::Type::Level> {
  return internal::ParallelCopy<
      LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn, OffsetColOut,
      OffsetRowOut, LHS::LeafType,
      1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                    RHS>::Type::Level>(lhs, rhs);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_PARALLEL_COPY_HPP_
