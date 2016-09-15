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

/// \file executor.hpp
/// \brief This files contains a series of forward declaration and include files
/// required for executing an expression tree.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_HPP_

namespace visioncpp {
namespace internal {

/// \struct FuseExpr
/// \brief The FuseExpr struct is used to generate one device kernel for the
/// expression which is not a terminal node (leafNode).
/// It is used to specialise the fuse function for terminal and
/// non-terminal nodes
/// template parameters:
/// \tparam LC: is the column size of local memory
/// \tparam LR: is the row size of local memory
/// \tparam LCT: is the column size of workgroup
/// \tparam LRT: is the row size of workgroup
/// \tparam Expr : the expression tree needed to be executed
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr>
struct FuseExpr;

// \struct NoFuse
/// \brief The NoFuse struct is used to generate one device kernel per each
/// non-terminal node in the expression. It is used to specialise the no_fuse
/// function for different non-terminal nodes
/// template parameters:
/// \tparam LC: is the column size of local memory
/// \tparam LR: is the row size of local memory
/// \tparam LCT: is the column size of workgroup
/// \tparam LRT: is the row size of workgroup
/// \tparam Expr : the node needed to be executed
///\tparam Category: the Category type of the node (Binary, Unary, Nullary)
template <size_t LC, size_t LR, size_t LCT, size_t LRT, size_t Category,
          typename Expr>
struct NoFuse;

/// \brief IfExprExecNeeded is used to decide:
/// 1) the expression should force its children to launch a separate kernel and
/// execute the code
/// 2) the expression should create a new subexpression tree with the result of
/// its children execution; execute the new sub expression and return the
/// LeafNode as a result.
/// template parameters
/// \tparam Conds: the decision made by Expr for its children stating whether or
/// not the child of Expr to be executed.
/// \tparam ParentConds: the decision mad by Expr parent stating that whether or
/// not the expr itself needed to be executed.
/// \tparam Category: representing to what expression Category the Expr
/// besize_ts
/// \tparam Expr: the type of the expression
template <bool Conds, bool ParentConds, size_t Category, typename Expr>
struct IfExprExecNeeded;

/// \struct Executor
/// \brief The Executor struct is used to specialise the execute function for
/// different avaiable policies at compile time.
template <bool ExecPolicy, size_t LC, size_t LR, size_t LCT, size_t LRT,
          typename Expr>
struct Executor;

/// \brief specialisaton of Execute function when the policy is fuse.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr>
struct Executor<policy::Fuse, LC, LR, LCT, LRT, Expr> {
  /// \brief  executing the expression
  /// parameters:
  /// \param expr : the expression needed to be executed
  /// \param dev : the selected device for executing the expression
  /// return void
  template <typename DeviceT>
  static inline void execute(Expr expr, const DeviceT &dev) {
    fuse<LC, LR, LCT, LRT>(expr, dev);
  }
};

template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr>
struct Executor<policy::NoFuse, LC, LR, LCT, LRT, Expr> {
  /// \brief  executing the expression
  /// parameters:
  /// \param expr : the expression needed to be executed
  /// \param dev : the selected device for executing the expression
  /// return void
  template <typename DeviceT>
  static inline void execute(Expr expr, const DeviceT &dev) {
    no_fuse<LC, LR, LCT, LRT>(expr, dev);
  }
};

/// \struct SubExprExecute
/// \brief it is used to statically determine whether or not a subexpression
/// execution is needed. It increases the execution time by avoiding executing
/// the subexpression when it is not needed. Using this
/// struct with sub_expression_evaluation parameter in every non-terminal node,
/// it is possible to determine such condition at compile time.
template <bool Val, bool ExecPolicy, size_t LC, size_t LR, size_t LCT,
          size_t LRT, typename Expr>
struct SubExprExecute {
  template <typename DeviceT>
  static void inline execute(Expr &expr, const DeviceT &dev) {
    expr.reset(true);
    Executor<ExecPolicy, LC, LR, LCT, LRT,
             decltype(expr.template sub_expression_evaluation<false, LC, LR,
                                                              LCT, LRT>(dev))>::
        execute(
            expr.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(
                dev),
            dev);
  }
};
/// \brief specialisation of the status of when there is no need for
/// subexpression execution
template <bool ExecPolicy, size_t LC, size_t LR, size_t LCT, size_t LRT,
          typename Expr>
struct SubExprExecute<false, ExecPolicy, LC, LR, LCT, LRT, Expr> {
  template <typename DeviceT>
  static void inline execute(Expr &expr, const DeviceT &dev) {
    Executor<ExecPolicy, LC, LR, LCT, LRT, Expr>::execute(expr, dev);
  }
};

}  // internal

/// \brief execute function is called by user in order to execute an expression
/// template parameters:
/// \tparam ExecPolicy: determining which policy to be used for executing an
/// expression. this can be Fuse or NoFuse
/// \tparam LC the column size for local memory when needed
/// \tparam LR the row size for column memory when needed
/// \tparam LCT the size of the workgroup column.
/// \tparam LRT the size of the workgroup row.
/// \tparam Expr the expression type to be executed.
/// function parameters:
/// \param expr the expression to be executed
/// \param dev the selected device for executing the expression
/// \return void
template <bool ExecPolicy, size_t LC, size_t LR, size_t LCT, size_t LRT,
          typename Expr, typename DeviceT>
void inline execute(Expr &expr, const DeviceT &dev) {
  internal::SubExprExecute<Expr::SubExpressionEvaluationNeeded, ExecPolicy, LC,
                           LR, LCT, LRT, Expr>::execute(expr, dev);
}

/// \brief special case of the execute function with default value for local
/// memory and workgroup size
/// template parameters:
/// \tparam ExecPolicy: determining which policy to be used for executing an
/// expression. this can be Fuse or NoFuse
/// \tparam Expr: the expression type to be executed.
/// function parameters:
/// \param expr: the expression to be executed
/// \param dev : the selected device for executing the expression
/// \return void
template <bool ExecPolicy, typename Expr, typename DeviceT>
void inline execute(Expr &expr, const DeviceT &dev) {
  internal::SubExprExecute<Expr::SubExpressionEvaluationNeeded, ExecPolicy, 8,
                           8, 8, 8, Expr>::execute(expr, dev);
}
}  // visioncpp
#include "executor_subexpr_if_needed.hpp"
#include "policy/fuse.hpp"
#include "policy/nofuse.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_EXECUTOR_HPP_
