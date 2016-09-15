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

/// \file eval_expr_assign.hpp
/// this file contains the specialisation of EvalExpr for Assign when it
/// is not a root

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_ASSIGN_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_ASSIGN_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an Assign expression. This node is not a root node.
template <typename LHS, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL, typename Loc, typename... Params>
struct EvalExpr<Assign<LHS, RHS, Cols, Rows, LfType, LVL>, Loc, Params...> {
  using LHS_Eval_Expr = EvalExpr<LHS, Loc, Params...>;
  using RHS_Eval_Expr = EvalExpr<RHS, Loc, Params...>;
  static auto get_accessor(const tools::tuple::Tuple<Params...> &t)
      -> decltype(LHS_Eval_Expr::get_accessor(t)) {
    return LHS_Eval_Expr::get_accessor(t);
  }
  /// \brief evaluate function when the internal::ops_category is PointOP.
  static auto eval_point(Loc &cOffset, const tools::tuple::Tuple<Params...> &t)
      -> typename MemoryTrait<LfType,
                              decltype(LHS_Eval_Expr::get_accessor(t))>::Type {
    auto x = RHS_Eval_Expr::eval_point(cOffset, t);
    LHS_Eval_Expr::eval_point(cOffset, t) = x;
    return x;
  }
  /// \brief evaluate function when the internal::ops_category is NeighbourOP.
  template <bool IsRoot, size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
            size_t Halo_Right, size_t Offset, size_t Index, size_t LC,
            size_t LR>
  static auto eval_neighbour(Loc &cOffset,
                             const tools::tuple::Tuple<Params...> &t)
      -> decltype(
          tools::tuple::get<OutputLocation<IsRoot, Offset + Index - 1>::ID>(
              t)) {
    constexpr bool isLocal =
        Trait<typename tools::RemoveAll<decltype(
            LHS_Eval_Expr::get_accessor(t))>::Type>::scope == scope::Local;
    constexpr size_t LC_Ratio = RHS::CThread / Cols;
    constexpr size_t LR_Ratio = RHS::RThread / Rows;
    // lhs expression shared mem
    auto nested_accessor =
        RHS_Eval_Expr::template eval_neighbour<IsRoot, Halo_Top, Halo_Left,
                                               Halo_Butt, Halo_Right, Offset,
                                               Index, LC, LR>(cOffset, t);
    constexpr bool isLocal_nested =
        Trait<typename tools::RemoveAll<decltype(
            nested_accessor)>::Type>::scope == scope::Local;
    auto rhs_acc = nested_accessor.get_pointer();
    auto lhs_acc = LHS_Eval_Expr::get_accessor(t).get_pointer();
    if ((cOffset.l_c < (cOffset.cLRng / LC_Ratio)) &&
        (cOffset.l_r < (cOffset.rLRng / LR_Ratio))) {
      size_t g_c = ((cOffset.g_c - cOffset.l_c) / LC_Ratio) + cOffset.l_c;
      size_t g_r = ((cOffset.g_r - cOffset.l_r) / LR_Ratio) + cOffset.l_r;
      for (int i = 0; i < LC / LC_Ratio; i += (cOffset.cLRng / LC_Ratio)) {
        if (get_compare<isLocal, LC / LC_Ratio, Cols>(cOffset.l_c, i, g_c)) {
          for (size_t j = 0; j < LR / LR_Ratio;
               j += (cOffset.rLRng / LR_Ratio)) {
            if (get_compare<isLocal, LR / LR_Ratio, Rows>(cOffset.l_r, j,
                                                          g_r)) {
              lhs_acc[calculate_index(id_val<isLocal>(cOffset.l_c, g_c) + i,
                                      id_val<isLocal>(cOffset.l_r, g_r) + j,
                                      id_val<isLocal>(LC / LC_Ratio, Cols),
                                      id_val<isLocal>(LR / LR_Ratio, Rows))] =
                  tools::convert<typename MemoryTrait<
                      LfType, decltype(nested_accessor)>::Type>(
                      rhs_acc[calculate_index(
                          id_val<isLocal_nested>(cOffset.l_c, g_c) + i,
                          id_val<isLocal_nested>(cOffset.l_r, g_r) + j,
                          id_val<isLocal_nested>(LC / LC_Ratio, Cols),
                          id_val<isLocal_nested>(LR / LR_Ratio, Rows))]);
            }
          }
        }
      }
    }
    // here you need to put local barrier
    cOffset.barrier();

    return tools::tuple::get<OutputLocation<IsRoot, Offset + Index - 1>::ID>(t);
  }
  /// \brief evaluate function when the internal::ops_category is
  /// GlobalNeighbour.
  template <bool IsRoot, size_t Offset, size_t Index, size_t LC, size_t LR>
  static auto eval_global_neighbour(Loc &cOffset,
                                    const tools::tuple::Tuple<Params...> &t)
      -> decltype(
          tools::tuple::get<OutputLocation<IsRoot, Offset + Index - 1>::ID>(
              t)) {
    // lhs expression shared mem
    auto rhs_acc = RHS_Eval_Expr::template eval_global_neighbour<IsRoot, Offset,
                                                                 Index, LC, LR>(
                       cOffset, t).get_pointer();
    auto lhs_acc = LHS_Eval_Expr::get_accessor(t).get_pointer();
    // here the neighbour is the entire output
    size_t index = 0;
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (get_compare<false, LC, Cols>(cOffset.l_c, i, cOffset.g_c)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (get_compare<false, LR, Rows>(cOffset.l_r, j, cOffset.g_r)) {
            index =
                calculate_index(cOffset.g_c + i, cOffset.g_r + j, Cols, Rows);
            lhs_acc[index] = rhs_acc[index];
          }
        }
      }
    }

    // here you need to put a local barrier
    cOffset.barrier();

    return RHS_Eval_Expr::template eval_global_neighbour<IsRoot, Offset,
                                                         Index - 1>(cOffset, t);
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_ASSIGN_HPP_
