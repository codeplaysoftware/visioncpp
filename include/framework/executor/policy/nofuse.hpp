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

/// \file nofuse.hpp
/// \brief This file contains the specialisation of the NoFuse for
/// different nodes.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_NOFUSE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_NOFUSE_HPP_

namespace visioncpp {
namespace internal {
/// \brief The specialisation of the NoFuse for LeafNode.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename RHS,
          size_t LVL>
struct NoFuse<LC, LR, LCT, LRT, internal::expr_category::Unary,
              LeafNode<RHS, LVL>> {
  using ALHS = LeafNode<RHS, LVL>;
  /// \brief the no_fuse function to execute a leafNode. It does nothing but to
  /// return the node.
  /// \param rhs : the leafNode passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// \return the leafNode passed to the function
  template <typename DeviceT>
  static ALHS no_fuse(LeafNode<RHS, LVL> rhs, const DeviceT &dev) {
    return rhs;
  }
};
/// \brief The specialisation of the NoFuse for Expression node with one
/// operand.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr>
struct NoFuse<LC, LR, LCT, LRT, internal::expr_category::Unary, Expr> {
  using ALHS = LeafNode<typename Expr::Type, Expr::Level>;
  /// \brief the no_fuse function to execute an expression node with one
  /// operand. It recursively calls the no_fuse function for its RHS; collects
  /// the result; launch a device kernel for the current expr with the new
  /// collected result; and returns a leafNode representing the output result of
  /// the expression.
  /// \param expr : the expression passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// \return the leafNode representing the result of the expression
  /// execution.
  template <typename DeviceT>
  static ALHS no_fuse(Expr expr, const DeviceT &dev) {
    auto iOutput = NoFuse<LC, LR, LCT, LRT, decltype(expr.rhs)::ND_Category,
                          decltype(expr.rhs)>::no_fuse(expr.rhs, dev);
    using IOutput = decltype(iOutput);
    auto lhs = ALHS();
    using ARHS = typename Expr::template ExprExchange<IOutput>;
    fuse<LC, LR, LCT, LRT>(
        Assign<ALHS, ARHS, ALHS::Type::Cols, ALHS::Type::Rows,
               ALHS::Type::LeafType,
               1 + tools::StaticIf<(ALHS::Level > ARHS::Level), ALHS,
                                   ARHS>::Type::Level>(lhs, ARHS(iOutput)),
        dev);
    return lhs;
  }
};
/// \brief The specialisation of the NoFuse for Expression node with two
/// operands.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr>
struct NoFuse<LC, LR, LCT, LRT, internal::expr_category::Binary, Expr> {
  using ALHS = LeafNode<typename Expr::Type, Expr::Level>;
  /// \brief the no_fuse function to execute an expression node with two
  /// operands. It recursively calls the no_fuse function for its LHS and RHS;
  /// collects the results; launch a device kernel for the current expr with the
  /// expression with new collected results; and returns a leafNode
  /// representing the output result of the expression.
  /// \param expr : the expression passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// \return the leafNode representing the result of the expression
  /// execution.
  template <typename DeviceT>
  static ALHS no_fuse(Expr expr, const DeviceT &dev) {
    auto i_lhs_output =
        NoFuse<LC, LR, LCT, LRT, decltype(expr.lhs)::ND_Category,
               decltype(expr.lhs)>::no_fuse(expr.lhs, dev);

    auto i_rhs_output =
        NoFuse<LC, LR, LCT, LRT, decltype(expr.rhs)::ND_Category,
               decltype(expr.rhs)>::no_fuse(expr.rhs, dev);

    using ARHS = typename Expr::template ExprExchange<decltype(i_lhs_output),
                                                      decltype(i_rhs_output)>;
    auto lhs = ALHS();
    fuse<LC, LR, LCT, LRT>(
        Assign<ALHS, ARHS, ALHS::Type::Cols, ALHS::Type::Rows,
               ALHS::Type::LeafType,
               1 + tools::StaticIf<(ALHS::Level > ARHS::Level), ALHS,
                                   ARHS>::Type::Level>(
            lhs, ARHS(i_lhs_output, i_rhs_output)),
        dev);
    return lhs;
  }
};

/// \brief The specialisation of the NoFuse for Assign where it is a root
/// node and has its own lhs leaf node
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t LeafType, size_t LVL>
struct NoFuse<LC, LR, LCT, LRT, internal::expr_category::Binary,
              Assign<LHS, RHS, Cols, Rows, LeafType, LVL>> {
  using out_Type = LHS;
  using Expr = Assign<LHS, RHS, Cols, Rows, LeafType, LVL>;
  /// \brief the no_fuse function to execute an Assign expression. It
  /// recursively
  /// calls the no_fuse function for its RHS; collects the result; launch a
  /// device kernel for the current expr with the new collected result; and
  /// returns a leafNode representing the output result of the expression.
  /// \param expr : the expression passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// \return the leafNode representing the result of the expression
  /// execution.
  template <typename DeviceT>
  static LHS no_fuse(Assign<LHS, RHS, Cols, Rows, LeafType, LVL> expr,
                     const DeviceT &dev) {
    auto i_rhs_output =
        NoFuse<LC, LR, LCT, LRT, RHS::ND_Category, RHS>::no_fuse(expr.rhs, dev);

    using ARHS =
        typename Expr::template ExprExchange<LHS, decltype(i_rhs_output)>;
    fuse<LC, LR, LCT, LRT>(ARHS(expr.lhs, i_rhs_output), dev);
    return expr.lhs;
  }
};

/// \brief template deduction function for no_fuse expression
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr,
          typename DeviceT>
void no_fuse(Expr expr, const DeviceT &dev) {
  NoFuse<LC, LR, LCT, LRT, Expr::ND_Category, Expr>::no_fuse(expr, dev);
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_NOFUSE_HPP_
