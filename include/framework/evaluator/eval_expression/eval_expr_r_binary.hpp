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

/// \file eval_expr_r_binary.hpp
/// \brief This file contains the specialisation of the EvalExpr
/// for RBiOP( pointwise Binary operation node).

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_BINARY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_BINARY_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an RBiOP(binary operation) expression.
template <typename BI_OP, typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct EvalExpr<RBiOP<BI_OP, LHS, RHS, Cols, Rows, LfType, LVL>, Loc,
                Params...> {
  /// \brief evaluate function when the internal::ops_category is PointOP.
  static typename BI_OP::OutType eval_point(
      Loc &cOffset, const tools::tuple::Tuple<Params...> &t) {
    auto lhs_acc = EvalExpr<LHS, Loc, Params...>::eval_point(cOffset, t);
    auto rhs_acc = EvalExpr<RHS, Loc, Params...>::eval_point(cOffset, t);
    return
        typename BI_OP::OP()(tools::convert<typename BI_OP::InType1>(lhs_acc),
                             tools::convert<typename BI_OP::InType2>(rhs_acc));
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
    static constexpr size_t RHSCount =
        LocalMemCount<RHS::ND_Category, RHS>::Count;

    auto lhs_acc =
        EvalExpr<LHS, Loc, Params...>::template eval_neighbour<
            false, Halo_Top, Halo_Left, Halo_Butt, Halo_Right, Offset,
            Index - 1 - RHSCount, LC, LR>(cOffset, t).get_pointer();
    auto rhs_acc = EvalExpr<RHS, Loc, Params...>::template eval_neighbour<
                       false, Halo_Top, Halo_Left, Halo_Butt, Halo_Right,
                       Offset, Index - 1, LC, LR>(cOffset, t).get_pointer();
    // eval the RBiOP
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (get_compare<isLocal, LC, Cols>(cOffset.l_c, i, cOffset.g_c)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (get_compare<isLocal, LR, Rows>(cOffset.l_r, j, cOffset.g_r)) {
            size_t child_index =
                calculate_index(cOffset.l_c + i, cOffset.l_r + j, LC, LR);
            tools::tuple::get<OutOffset>(t).get_pointer()[calculate_index(
                id_val<isLocal>(cOffset.l_c, cOffset.g_c) + i,
                id_val<isLocal>(cOffset.l_r, cOffset.g_r) + j,
                id_val<isLocal>(LC, Cols), id_val<isLocal>(LR, Rows))] =
                tools::convert<typename MemoryTrait<
                    LfType, decltype(tools::tuple::get<OutOffset>(t))>::Type>(
                    typename BI_OP::OP()(lhs_acc[child_index],
                                         rhs_acc[child_index]));
          }
        }
      }
    }

    // here you need to put a local barrier
    cOffset.barrier();
    // return the valid neighbour area for your parent
    return tools::tuple::get<OutOffset>(t);
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_R_BINARY_HPP_
