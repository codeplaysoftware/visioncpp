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

/// \file resizable_binary.hpp
/// \brief This file contains RBiOP (Binary Operation) struct which is used to
/// apply Binary Operation on left hand side(LHS) and right hand side(RHS)
/// operands.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_BINARY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_BINARY_HPP_

namespace visioncpp {
namespace internal {
/// \struct InheritTypeBinary
/// \brief This struct is used to extract the output type of the binary
/// operation from both input. This is useful when one of the operands passed is
/// a constant variable and the other one is a buffer. This inheritance allows to
/// swap the place of the constant variable in the node construction
/// template parameters
/// \tparam LHSExpr left hand side expression
/// \tparam RHSExpr right hand side expression
template <typename LHSExpr, typename RHSExpr>
struct InheritTypeBinary {
  using LHSTypeDetector = typename LHSExpr::Type;
  using RHSTypeDetector = typename RHSExpr::Type;
  using Type =
      typename tools::StaticIf<LHSTypeDetector::LeafType != memory_type::Const,
                               LHSTypeDetector, RHSTypeDetector>::Type;
};

/// \struct OpTP
/// \brief This struct is used to determine the operation type for binary
/// operation based on the operation type of its children.
/// always pointop << neighbourop
/// template parameters
/// \tparam Conds : boolean value determines whether or not both types are equal
/// \tparam LhsOp : operation type for left-hand size expression
/// \tparam RhsOp : operation type for right-hand size expression
template <bool Conds, size_t LhsOP, size_t RhsOP>
struct OpTP;
/// \brief specialisation of the PointOP category where the two operation types
/// are not equal and the lhs operation type is PointOp
template <size_t RhsOP>
struct OpTP<false, internal::ops_category::PointOP, RhsOP> {
  static constexpr size_t Operation_type = RhsOP;
};
/// \brief specialisation of the PointOP category where the two operation types
/// are not equal and the rhs operation type is PointOp
template <size_t LhsOP>
struct OpTP<false, LhsOP, internal::ops_category::PointOP> {
  static constexpr size_t Operation_type = LhsOP;
};
/// \brief specialisation of the PointOP category where the two operation types
/// are equal
template <size_t OPType>
struct OpTP<true, OPType, OPType> {
  static constexpr size_t Operation_type = OPType;
};

/// \struct RBiOP
/// \brief RBiOP is used to
/// apply Binary Operation on left-hand side(LHS) and right-hand side(RHS)
/// operands.
/// template parameters:
/// \tparam LHS is the left-hand- side expression
/// \tparam RHS is the right-hand side expression
/// \tparam Cols: determines the column size of the output
/// \tparam Rows: determines the row size of the output
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree
template <typename BI_OP, typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL>
struct RBiOP {
 public:
  static constexpr bool has_out = false;
  using OutType = typename BI_OP::OutType;
  using OPType = BI_OP;
  using RHSExpr = RHS;
  using LHSExpr = LHS;
  using Type = typename visioncpp::internal::OutputMemory<OutType, LfType, Cols,
                                                          Rows, LVL>::Type;
  static constexpr size_t Level = LVL;
  static constexpr size_t RThread = Rows;
  static constexpr size_t CThread = Cols;
  static constexpr size_t ND_Category = internal::expr_category::Binary;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr bool Binary_Conds =
      (LHS::LeafType != memory_type::Const &&
       ((RThread != LHS::RThread) || (CThread != LHS::CThread))) ||
      (RHS::LeafType != memory_type::Const &&
       ((RThread != RHS::RThread) || (CThread != RHS::CThread)));
  static constexpr bool SubExpressionEvaluationNeeded =
      Binary_Conds || LHS::SubExpressionEvaluationNeeded ||
      RHS::SubExpressionEvaluationNeeded;
  static constexpr size_t Operation_type =
      internal::OpTP<(LHS::Operation_type == RHS::Operation_type),
                     LHS::Operation_type, RHS::Operation_type>::Operation_type;

  template <typename TmpLHS, typename TmpRHS>
  using ExprExchange = RBiOP<BI_OP, TmpLHS, TmpRHS, Cols, Rows, LfType, LVL>;

  LHS lhs;
  RHS rhs;
  bool subexpr_execution_reseter;
  RBiOP(LHS lhsArg, RHS rhsArg)
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
  auto inline sub_expression_evaluation(const DeviceT& dev)
      -> decltype(execute_expr<Binary_Conds, ForcedToExec,
                               ExprExchange<LHS, RHS>, LC, LR, LCT, LRT>(
          lhs.template sub_expression_evaluation<Binary_Conds, LC, LR, LRT,
                                                 LCT>(dev),
          rhs.template sub_expression_evaluation<Binary_Conds, LC, LR, LRT,
                                                 LCT>(dev),
          dev)) {
    return execute_expr<Binary_Conds, ForcedToExec, ExprExchange<LHS, RHS>, LC,
                        LR, LCT, LRT>(
        lhs.template sub_expression_evaluation<Binary_Conds, LC, LR, LCT, LRT>(
            dev),
        rhs.template sub_expression_evaluation<Binary_Conds, LC, LR, LCT, LRT>(
            dev),
        dev);
  }
};

/// \brief  template deduction for the RBiOP struct where user determines the
/// Column, Row, and memory_type of the output.
template <typename OP, size_t Cols, size_t Rows, size_t LeafType, typename LHS,
          typename RHS>
auto point_operation(LHS lhs, RHS rhs) -> RBiOP<
    internal::PixelBinaryOp<OP, typename LHS::OutType, typename RHS::OutType>,
    LHS, RHS, Cols, Rows, LeafType,
    1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                  RHS>::Type::Level> {
  return RBiOP<
      internal::PixelBinaryOp<OP, typename LHS::OutType, typename RHS::OutType>,
      LHS, RHS, Cols, Rows, LeafType,
      1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                    RHS>::Type::Level>(lhs, rhs);
}
}  // internal

/// \brief  template deduction for RBiOP struct where the Column, Row,
/// and memory_type of the output has been automatically deduced from LHS and
/// RHS operands
template <typename OP, typename LHS, typename RHS>
auto point_operation(LHS lhs, RHS rhs) -> internal::RBiOP<
    internal::PixelBinaryOp<OP, typename LHS::OutType, typename RHS::OutType>,
    LHS, RHS, internal::InheritTypeBinary<LHS, RHS>::Type::Cols,
    internal::InheritTypeBinary<LHS, RHS>::Type::Rows,
    internal::InheritTypeBinary<LHS, RHS>::Type::LeafType,
    1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                  RHS>::Type::Level> {
  return internal::RBiOP<
      internal::PixelBinaryOp<OP, typename LHS::OutType, typename RHS::OutType>,
      LHS, RHS, internal::InheritTypeBinary<LHS, RHS>::Type::Cols,
      internal::InheritTypeBinary<LHS, RHS>::Type::Rows,
      internal::InheritTypeBinary<LHS, RHS>::Type::LeafType,
      1 + internal::tools::StaticIf<(LHS::Level > RHS::Level), LHS,
                                    RHS>::Type::Level>(lhs, rhs);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_POINT_OPS_RESIZABLE_BINARY_HPP_
