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

/// \file resizable_unary.hpp
/// \brief This file contains RUnOP (Unary Operation) struct which is used to
/// apply Unary Operation on the right hand side(RHS) node.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_UNARY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_UNARY_HPP_

namespace visioncpp {
namespace internal {
/// \struct RUnOP
/// \brief RUnOP is used to
/// apply a unary operation on the right hand side(RHS) operand.
/// template parameters:
/// \tparam RHS is the right hand side expression
/// \tparam Cols: determines the column size of the output
/// \tparam Rows: determines the row size of the output
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree
template <typename UN_OP, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL>
struct RUnOP {
 public:
  static constexpr bool has_out = false;
  using OutType = typename UN_OP::OutType;
  using OPType = UN_OP;
  using RHSExpr = RHS;
  using Type = typename OutputMemory<OutType, LfType, Cols, Rows, LVL>::Type;
  static constexpr size_t Level = LVL;
  static constexpr size_t RThread = Rows;
  static constexpr size_t CThread = Cols;
  static constexpr size_t ND_Category = internal::expr_category::Unary;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr bool Unary_Conds =
      (RHS::LeafType != memory_type::Const &&
       ((RThread != RHS::RThread) || (CThread != RHS::CThread)));
  static constexpr bool SubExpressionEvaluationNeeded =
      Unary_Conds || RHS::SubExpressionEvaluationNeeded;
  static constexpr size_t Operation_type = RHS::Operation_type;

  template <typename TmpRHS>
  using ExprExchange = RUnOP<UN_OP, TmpRHS, Cols, Rows, LfType, LVL>;

  RHS rhs;
  bool subexpr_execution_reseter;
  RUnOP(RHS rhsArg) : rhs(rhsArg), subexpr_execution_reseter(false) {}

  void reset(bool reset) {
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
  /// \tparam LC: is the column size of the local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LR: is the row size of the local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LCT: is the column size of workgroup
  /// \tparam LRT: is the row size of workgroup
  /// \tparam DeviceT: type representing the device
  /// function parameters:
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  auto inline sub_expression_evaluation(const DeviceT &dev)
      -> decltype(execute_expr<Unary_Conds, ForcedToExec, ExprExchange<RHS>, LC,
                               LR, LCT, LRT>(
          rhs.template sub_expression_evaluation<Unary_Conds, LC, LR, LCT, LRT>(
              dev),
          dev)) {
    return execute_expr<Unary_Conds, ForcedToExec, ExprExchange<RHS>, LC, LR,
                        LCT, LRT>(
        rhs.template sub_expression_evaluation<Unary_Conds, LC, LR, LCT, LRT>(
            dev),
        dev);
  }
};
}  // internal

/// \brief template deduction for RUnOP struct where user determines the Column,
/// Row, and memory_type of the output.
template <typename OP, size_t Cols, size_t Rows, size_t LeafType, typename RHS>
internal::RUnOP<internal::PixelUnaryOp<OP, typename RHS::OutType>, RHS, Cols,
                Rows, LeafType, 1 + RHS::Level>
point_operation(RHS rhs) {
  return internal::RUnOP<internal::PixelUnaryOp<OP, typename RHS::OutType>, RHS,
                         Cols, Rows, LeafType, 1 + RHS::Level>(rhs);
}

/// \brief template deduction for RUnOP struct where the Column, Row,
/// and memory_type of the output has been automatically deduced from LHS and
/// RHS operands
template <typename OP, typename RHS>
auto point_operation(RHS rhs)
    -> internal::RUnOP<internal::PixelUnaryOp<OP, typename RHS::OutType>, RHS,
                       RHS::Type::Cols, RHS::Type::Rows, RHS::Type::LeafType,
                       1 + RHS::Level> {
  return internal::RUnOP<internal::PixelUnaryOp<OP, typename RHS::OutType>, RHS,
                         RHS::Type::Cols, RHS::Type::Rows, RHS::Type::LeafType,
                         1 + RHS::Level>(rhs);
}

/// \brief template deduction for RUnOP when it is used to broadcast one const
/// value to all the channels of its parent node.
template <typename OP, size_t Cols, size_t Rows, size_t LeafType, size_t Sc,
          typename ElementTp, typename Scalar>
internal::RUnOP<internal::PixelUnaryOp<OP, ElementTp>,
                internal::VisionMemory<true, Sc, LeafType, Scalar, 1, 1,
                                       ElementTp, 1, scope::Global, 0>,
                Cols, Rows, LeafType, 1>
    broadcast_value(internal::VisionMemory<
        true, Sc, LeafType, Scalar, 1, 1, ElementTp, 1, scope::Global, 0> rhs) {
  return internal::RUnOP<
      internal::PixelUnaryOp<OP, ElementTp>,
      internal::VisionMemory<true, Sc, LeafType, Scalar, 1, 1, ElementTp, 1,
                             scope::Global, 0>,
      Cols, Rows, LeafType, 1>(rhs);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_UNARY_HPP_
