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

/// \file executor_subexpr_if_needed.hpp
/// \brief This file contains required classes for break an expression tree into
/// subexpression trees for particular nodes when it is needed.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_SUBEXPR_IF_NEEDED_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_SUBEXPR_IF_NEEDED_HPP_

namespace visioncpp {
namespace internal {

/// \struct SubExprRes
/// \brief The SubExprRes is used to specialise the get_subexpr_executor
/// function for terminal and non-terminal node. The get function in SubExprRes
/// executes the expression and returns a leaf node representing the output
/// result of the kernel execution, when it is a non-terminal node.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, size_t LVL,
          typename Expr>
struct SubExprRes {
  using Type = internal::LeafNode<typename Expr::Type, LVL>;
  /// \brief The get function in SubExprRes
  /// executes the expression and returns a leaf node representing the output.
  /// \param eval_sub: the SubExprRes passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode: the leafNode containing the result of the
  /// subexpression execution.
  template <typename DeviceT>
  static Type get(Expr &eval_sub, const DeviceT &dev) {
    using Intermediate_Output = internal::LeafNode<typename Expr::Type, LVL>;
    auto intermediate_output = Intermediate_Output();
    internal::fuse<LC, LR, LCT, LRT>(
        internal::Assign<Intermediate_Output, Expr,
                         Intermediate_Output::Type::Cols,
                         Intermediate_Output::Type::Rows,
                         Intermediate_Output::Type::LeafType,
                         1 + internal::tools::StaticIf<
                                 (Intermediate_Output::Level > Expr::Level),
                                 Intermediate_Output, Expr>::Type::Level>(
            intermediate_output, eval_sub),
        dev);
    return intermediate_output;
  }
};

/// \brief specialisation of the SubExprRes when it is a LeafNode. It does
/// nothing but returns the given input leaf node as there is no operation to
/// apply. This specialisation is used to speed up the process and to prevent
/// extra memory creation and execution.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename RHS,
          size_t LVL1, size_t LVL2>
struct SubExprRes<LC, LR, LCT, LRT, LVL1, internal::LeafNode<RHS, LVL2>> {
  using Type = internal::LeafNode<RHS, LVL2>;
  /// \brief The get function in SubExprRes do nothing but return the leaf node
  /// when the input expression type is a leafNode.
  /// \param eval_sub: the SubExprRes passed to be executed on the device
  /// \return LeafNode: the leafNode containing the result of the
  /// subexpression execution.
  template <typename DeviceT>
  static inline Type get(Type &eval_sub, const DeviceT &) {
    return eval_sub;
  }
};

/// \brief template deduction for SubExprRes. This is used when we manually need
/// to pass the level in order to avoid double naming
template <size_t LC, size_t LR, size_t LCT, size_t LRT, size_t LVL,
          typename Expr, typename DeviceT>
auto get_subexpr_executor(Expr &expr, const DeviceT &dev) ->
    typename internal::SubExprRes<LC, LR, LCT, LRT, LVL, Expr>::Type {
  return internal::SubExprRes<LC, LR, LCT, LRT, LVL, Expr>::get(expr, dev);
};

// \brief template deduction for SubExprRes. This one will automatically level
// the kernel and create a name
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr,
          typename DeviceT>
auto get_subexpr_executor(Expr &expr, const DeviceT &dev) ->
    typename internal::SubExprRes<LC, LR, LCT, LRT, 1 + Expr::Level,
                                  Expr>::Type {
  return internal::SubExprRes<LC, LR, LCT, LRT, 1 + Expr::Level, Expr>::get(
      expr, dev);
};

/// \brief \struct ParentForcedExecute is used to check whether or not the
/// parent of the Expr requires that the expr to be executed and the LeafNode as
/// a result to be returned. True means that execute the expression and return
/// the LeafNode; false means return the execute as it is. This is an extra
/// check making sure that we wont break the tree when it is not necessary. Such
/// example would be when you have reduction as a root of a subexpression tree
/// in the middle of an expression, it is possible to minimise the number of
/// subexpression to be creating by 2.
template <bool ParentConds, typename Expr>
struct ParentForcedExecute;

/// \brief specialisation of ParentForcedExecute when the condition is true
template <typename Expr>
struct ParentForcedExecute<true, Expr> {
  using Type = internal::LeafNode<typename Expr::Type, Expr::Level>;
  /// \brief forced_exec here executes the expr and return the leafNode
  /// template parameters
  /// \tparam LC: suggested column size for the local memory
  /// \tparam LR: suggested row size for the local memory
  /// \tparam LRT: suggested workgroup row size
  /// \tparam LCT: suggested workgroup column size
  /// \tparam Expr: the expression type
  /// function parameters:
  /// \param expr: the expression needed to be executed
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename DeviceT>
  static inline Type forced_exec(Expr &expr, const DeviceT &dev) {
    auto lhs = Type();
    internal::fuse<LC, LR, LCT, LRT>(
        internal::Assign<
            Type, Expr, Type::Type::Cols, Type::Type::Rows,
            Type::Type::LeafType,
            1 + internal::tools::StaticIf<(Type::Level > Expr::Level), Type,
                                          Expr>::Type::Level>(lhs, expr),
        dev);
    return lhs;
  }
};
/// \brief specialisation of ParentForcedExecute when the condition is false
template <typename Expr>
struct ParentForcedExecute<false, Expr> {
  using Type = Expr;
  /// \brief forced_exec here does nothing but return the expression as the parent
  // of the expression does not require to break the expression further more.
  /// template parameters
  /// \tparam LC: suggested column size for the local memory
  /// \tparam LR: suggested row size for the local memory
  /// \tparam LRT: suggested workgroup row size
  /// \tparam LCT: suggested workgroup column size
  /// \tparam Expr: the expression type
  /// function parameters:
  /// \param expr: the input expression
  /// \return Expr the input expression
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename DeviceT>
  static inline Type forced_exec(Expr &expr, const DeviceT &) {
    return expr;
  }
};

/// \brief the specialisation of the IfExprExecNeeded when the decision for
/// executing the children of the expression is false and the expression
/// category is unary.
template <bool ParentConds, typename Expr>
struct IfExprExecNeeded<false, ParentConds, internal::expr_category::Unary,
                        Expr> {
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename NestedExpr,
            typename DeviceT>
  static inline auto execute_expr(NestedExpr &nestedExpr, const DeviceT &dev) ->
      typename ParentForcedExecute<
          ParentConds, typename Expr::template ExprExchange<NestedExpr>>::Type {
    using NestedType = typename Expr::template ExprExchange<NestedExpr>;
    auto res = NestedType(nestedExpr);

    return ParentForcedExecute<ParentConds, NestedType>::template forced_exec<
        LC, LR, LCT, LRT>(res, dev);
  }
};

/// \brief the specialisation of the IfExprExecNeeded when the decision for
/// executing the children of the expression is true and the expression
/// category is unary.
template <bool ParentConds, typename Expr>
struct IfExprExecNeeded<true, ParentConds, internal::expr_category::Unary,
                        Expr> {
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename NestedExpr,
            typename DeviceT>
  static inline auto execute_expr(NestedExpr &sub_expr, const DeviceT &dev) ->
      typename ParentForcedExecute<
          ParentConds,
          typename Expr::template ExprExchange<decltype(
              get_subexpr_executor<LC, LR, LCT, LRT>(sub_expr, dev))>>::Type {
    auto nested_output = get_subexpr_executor<LC, LR, LCT, LRT>(sub_expr, dev);
    using NestedType =
        typename Expr::template ExprExchange<decltype(nested_output)>;
    auto res = NestedType(nested_output);
    return ParentForcedExecute<ParentConds, NestedType>::template forced_exec<
        LC, LR, LCT, LRT>(res, dev);
  }
};

/// \brief the specialisation of the IfExprExecNeeded when the decision for
/// executing the children of the expression is false and the expression
/// category is binary.
template <bool ParentConds, typename Expr>
struct IfExprExecNeeded<false, ParentConds, internal::expr_category::Binary,
                        Expr> {
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename LHSExpr,
            typename RHSExpr, typename DeviceT>
  static inline auto execute_expr(LHSExpr &lhsExpr, RHSExpr &rhsExpr,
                                  const DeviceT &dev) ->
      typename ParentForcedExecute<
          ParentConds,
          typename Expr::template ExprExchange<LHSExpr, RHSExpr>>::Type {
    using SubExprType = typename Expr::template ExprExchange<LHSExpr, RHSExpr>;
    auto res = SubExprType(lhsExpr, rhsExpr);
    return ParentForcedExecute<ParentConds, SubExprType>::template forced_exec<
        LC, LR, LCT, LRT>(res, dev);
  }
};

/// \brief the specialisation of the IfExprExecNeeded when the decision for
/// executing the children of the expression is true and the expression
/// category is binary.
template <bool ParentConds, typename Expr>
struct IfExprExecNeeded<true, ParentConds, internal::expr_category::Binary,
                        Expr> {
  template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename LHSExpr,
            typename RHSExpr, typename DeviceT>
  static inline auto execute_expr(LHSExpr &lhsExpr, RHSExpr &rhsExpr,
                                  const DeviceT &dev) ->

      typename ParentForcedExecute<
          ParentConds,
          typename Expr::template ExprExchange<
              decltype(get_subexpr_executor<LC, LR, LCT, LRT>(lhsExpr, dev)),
              decltype(get_subexpr_executor<LC, LR, LCT, LRT>(rhsExpr,
                                                              dev))>>::Type {
    auto lhs_output = get_subexpr_executor<LC, LR, LCT, LRT>(lhsExpr, dev);
    auto rhs_output = get_subexpr_executor<LC, LR, LCT, LRT>(rhsExpr, dev);
    using SubExprType =
        typename Expr::template ExprExchange<decltype(lhs_output),
                                             decltype(rhs_output)>;
    auto res = SubExprType(lhs_output, rhs_output);
    return ParentForcedExecute<ParentConds, SubExprType>::template forced_exec<
        LC, LR, LCT, LRT>(res, dev);
  }
};

/// \brief template deduction for IfExprExecNeeded when the expression category
/// is unary
template <bool Conds, bool ParentConds, typename Expr, size_t LC, size_t LR,
          size_t LCT, size_t LRT, typename NestedExpr, typename DeviceT>
inline auto execute_expr(NestedExpr nestedExpr, const DeviceT &dev)
    -> decltype(internal::IfExprExecNeeded<
        Conds, ParentConds, internal::expr_category::Unary,
        Expr>::template execute_expr<LC, LR, LCT, LRT>(nestedExpr, dev)) {
  return internal::IfExprExecNeeded<
      Conds, ParentConds, internal::expr_category::Unary,
      Expr>::template execute_expr<LC, LR, LCT, LRT>(nestedExpr, dev);
}

/// \brief template deduction for IfExprExecNeeded when the expression
/// category is Binary
template <bool Conds, bool ParentConds, typename Expr, size_t LC, size_t LR,
          size_t LCT, size_t LRT, typename LHSExpr, typename RHSExpr,
          typename DeviceT>
inline auto execute_expr(LHSExpr lhsExpr, RHSExpr rhsExpr, const DeviceT &dev)
    -> decltype(internal::IfExprExecNeeded<
        Conds, ParentConds, internal::expr_category::Binary,
        Expr>::template execute_expr<LC, LR, LCT, LRT>(lhsExpr, rhsExpr, dev)) {
  return internal::IfExprExecNeeded<
      Conds, ParentConds, internal::expr_category::Binary,
      Expr>::template execute_expr<LC, LR, LCT, LRT>(lhsExpr, rhsExpr, dev);
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_SUBEXPR_IF_NEEDED_HPP_