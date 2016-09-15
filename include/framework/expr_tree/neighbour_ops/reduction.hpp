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

/// \file reduction.hpp
/// \brief This file contains RDCN struct which is used to construct a
/// node to shrink an image.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_NEIGHBOUR_OPS_REDUCTION_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_NEIGHBOUR_OPS_REDUCTION_HPP_

namespace visioncpp {
namespace internal {
/// \struct RDCN
/// \brief RDCN is used to shrink the size of an input. It can be used both for
/// NeighbourOP and GlobalNeighbourOP.
/// \tparam DownSmplOP: downsampling function to shrink an input
/// \tparam LHS is the input
/// \tparam RHS is the filter2d node
/// \tparam Cols: determines the column size of the output
/// \tparam Rows: determines the row size of the output
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree
template <typename DownSmplOP, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL>
struct RDCN {
 public:
  static constexpr size_t LC_Ratio =
      tools::IfConst<DownSmplOP::Operation_type ==
                         ops_category::GlobalNeighbourOP,
                     1, RHS::CThread / Cols>::Value;
  static constexpr size_t LR_Ratio =
      tools::IfConst<DownSmplOP::Operation_type ==
                         ops_category::GlobalNeighbourOP,
                     1, RHS::RThread / Rows>::Value;
  using RHSExpr = RHS;
  static constexpr bool has_out = false;
  using OutType = typename DownSmplOP::OutType;
  using OPType = DownSmplOP;
  using Type = typename OutputMemory<OutType, LfType, Cols, Rows, LVL>::Type;
  static constexpr size_t Level = LVL;
  static constexpr size_t RThread = Rows * LR_Ratio;
  static constexpr size_t CThread = Cols * LC_Ratio;
  static constexpr size_t ND_Category = expr_category::Unary;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr bool SubExpressionEvaluationNeeded =
      DownSmplOP::Operation_type == ops_category::GlobalNeighbourOP ||
      RHS::SubExpressionEvaluationNeeded;
  // if the coordinates are not the same
  // with your child you have to break for others except reduction
  static constexpr size_t Operation_type = DownSmplOP::Operation_type;
  template <typename TmpRHS>
  using ExprExchange = RDCN<DownSmplOP, TmpRHS, Cols, Rows, LfType, LVL>;

  RHS rhs;
  bool subexpr_execution_reseter;
  RDCN(RHS rhsArg) : rhs(rhsArg), subexpr_execution_reseter(false) {}

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
  auto inline sub_expression_evaluation(const DeviceT &dev)
      -> decltype(execute_expr<
          DownSmplOP::Operation_type == ops_category::GlobalNeighbourOP,
          ForcedToExec, ExprExchange<RHS>, LC, LR, LCT, LRT>(
          rhs.template sub_expression_evaluation<  // decide for your child
              DownSmplOP::Operation_type == ops_category::GlobalNeighbourOP, LC,
              LR, LCT, LRT>(dev),
          dev)) {
    return execute_expr<DownSmplOP::Operation_type ==
                            ops_category::GlobalNeighbourOP,
                        ForcedToExec, ExprExchange<RHS>, LC, LR, LCT, LRT>(
        rhs.template sub_expression_evaluation<  // decide for your child
            DownSmplOP::Operation_type == ops_category::GlobalNeighbourOP, LC,
            LR, LCT, LRT>(dev),
        dev);
  }
};
}  // internal

/// \brief template deduction function for RDCN when it is used for
/// GlobalNeighbourOP.
template <typename OP, size_t Cols, size_t Rows, size_t LeafType, typename RHS>
auto global_operation(RHS rhs)
    -> internal::RDCN<internal::GlobalUnaryOp<OP, typename RHS::OutType>, RHS,
                      Cols, Rows, LeafType, 1 + RHS::Level> {
  return internal::RDCN<internal::GlobalUnaryOp<OP, typename RHS::OutType>, RHS,
                        Cols, Rows, LeafType, 1 + RHS::Level>(rhs);
}
/// \brief template deduction function for RDCN when it is used for
/// NeighbourOP.
template <typename OP, size_t Cols, size_t Rows, size_t LeafType, typename RHS>
auto neighbour_operation(RHS rhs)
    -> internal::RDCN<internal::LocalUnaryOp<OP, typename RHS::OutType>, RHS,
                      Cols, Rows, LeafType, 1 + RHS::Level> {
  return internal::RDCN<internal::LocalUnaryOp<OP, typename RHS::OutType>, RHS,
                        Cols, Rows, LeafType, 1 + RHS::Level>(rhs);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_NEIGHBOUR_OPS_REDUCTION_HPP_
