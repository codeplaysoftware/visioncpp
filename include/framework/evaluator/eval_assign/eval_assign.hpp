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

/// \file eval_assign.hpp
/// \brief This file contains the specialisation of the Evaluator struct for
/// assign when it is a root struct

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the Evaluator when the expression is an
/// internal::Assign expression and the internal::ops_category is
/// GlobalNeighbourOP.
template <size_t OutputIndex, size_t Offset, size_t LC, size_t LR, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t LfType, size_t LVL,
          typename Loc, typename... Params>
struct Evaluator<internal::ops_category::GlobalNeighbourOP, OutputIndex, Offset,
                 LC, LR, internal::Assign<LHS, RHS, Cols, Rows, LfType, LVL>,
                 Loc, Params...> {
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    EvalExpr<RHS, Loc, Params...>::template eval_global_neighbour<
        true, Offset, OutputIndex, LC, LR>(cOffset, t);
  }
};

/// \brief Partial specialisation of the Evaluator when the expression is an
/// internal::Assign expression and the internal::ops_category is NeighbourOP.
template <size_t OutputIndex, size_t Offset, size_t LC, size_t LR, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t LfType, size_t LVL,
          typename Loc, typename... Params>
struct Evaluator<internal::ops_category::NeighbourOP, OutputIndex, Offset, LC,
                 LR, internal::Assign<LHS, RHS, Cols, Rows, LfType, LVL>, Loc,
                 Params...> {
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    EvalExpr<RHS, Loc, Params...>::template eval_neighbour<
        true, 0, 0, 0, 0, Offset, OutputIndex, LC, LR>(cOffset, t);
  }
};

/// \brief Partial specialisation of the Evaluator when the expression is an
/// internal::Assign expression and the internal::ops_category is PointOP.
template <size_t Output_Index, size_t Offset, size_t LC, size_t LR,
          typename LHS, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL, typename Loc, typename... Params>
struct Evaluator<internal::ops_category::PointOP, Output_Index, Offset, LC, LR,
                 internal::Assign<LHS, RHS, Cols, Rows, LfType, LVL>, Loc,
                 Params...> {
  using Expr = internal::Assign<LHS, RHS, Cols, Rows, LfType, LVL>;
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    using RHS_Eval_Expr = EvalExpr<RHS, Loc, Params...>;
    using LHS_Eval_Expr = EvalExpr<LHS, Loc, Params...>;

    using ElementType =
        typename MemoryTrait<Expr::LeafType,
                             decltype(tools::tuple::get<0>(t))>::Type;
    for (int i = 0; i < LC; i += cOffset.cLRng)
      if (cOffset.g_c + i < Expr::Type::Cols)
        for (int j = 0; j < LR; j += cOffset.rLRng)
          if (cOffset.g_r + j < Expr::Type::Rows) {
            cOffset.pointOp_gc = cOffset.g_c + i;
            cOffset.pointOp_gr = cOffset.g_r + j;
            LHS_Eval_Expr::get_accessor(t).get_pointer()[calculate_index(
                cOffset.g_c + i, cOffset.g_r + j, Expr::Type::Cols,
                Expr::Type::Rows)] =
                tools::convert<ElementType>(
                    RHS_Eval_Expr::eval_point(cOffset, t));
          }
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_HPP_
