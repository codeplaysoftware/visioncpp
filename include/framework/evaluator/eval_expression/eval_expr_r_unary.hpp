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

/// \file eval_expr_r_unary.hpp
/// \brief This file contains the specialisation of the EvalExpr
/// for RUnOP( pointwise Unary operation node).

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_UNARY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_UNARY_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an RUnOP(unary operation) expression.
template <typename UN_OP, typename Nested, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct EvalExpr<RUnOP<UN_OP, Nested, Cols, Rows, LfType, LVL>, Loc, Params...> {
  /// \brief evaluate function when the internal::ops_category is PointOP.
  static typename UN_OP::OutType eval_point(
      Loc &cOffset, const tools::tuple::Tuple<Params...> &t) {
    auto nested_acc = EvalExpr<Nested, Loc, Params...>::eval_point(cOffset, t);
    return typename UN_OP::OP()(
        tools::convert<typename UN_OP::InType>(nested_acc));
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
    constexpr size_t OutOffset = OutputLocation<IsRoot, Offset + Index - 1>::ID;
    constexpr bool isLocal =
        Trait<typename tools::RemoveAll<decltype(
            tools::tuple::get<OutOffset>(t))>::Type>::scope == scope::Local;
    auto nested_acc = EvalExpr<Nested, Loc, Params...>::template eval_neighbour<
                          false, Halo_Top, Halo_Left, Halo_Butt, Halo_Right,
                          Offset, Index - 1, LC, LR>(cOffset, t).get_pointer();
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (get_compare<isLocal, LC, Cols>(cOffset.l_c, i, cOffset.g_c)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (get_compare<isLocal, LR, Rows>(cOffset.l_r, j, cOffset.g_r)) {
            tools::tuple::get<OutOffset>(t).get_pointer()[calculate_index(
                id_val<isLocal>(cOffset.l_c, cOffset.g_c) + i,
                id_val<isLocal>(cOffset.l_r, cOffset.g_r) + j,
                id_val<isLocal>(LC, Cols), id_val<isLocal>(LR, Rows))] =
                tools::convert<typename MemoryTrait<
                    LfType, decltype(tools::tuple::get<OutOffset>(t))>::Type>(
                    typename UN_OP::OP()(nested_acc[calculate_index(
                        cOffset.l_c + i, cOffset.l_r + j, LC, LR)]));
          }
        }
      }
    }

    // here you need to put a local barrier
    cOffset.barrier();
    // return the the valid neighbour area for parent
    return tools::tuple::get<OutOffset>(t);
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_UNARY_HPP_