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

/// \file eval_expr_stn_filt.hpp
/// \brief This file contains the specialisation of the EvalExpr
/// for StnFilt( stencil node with filter ).

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_STN_FILT_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_STN_FILT_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an StnFilt(stencil with filter operation) expression.
template <typename C_OP, size_t Halo_T, size_t Halo_L, size_t Halo_B,
          size_t Halo_R, typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct EvalExpr<StnFilt<C_OP, Halo_T, Halo_L, Halo_B, Halo_R, LHS, RHS, Cols,
                        Rows, LfType, LVL>,
                Loc, Params...> {
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
    // lhs expression shared mem
    static constexpr size_t RHSCount =
        LocalMemCount<RHS::ND_Category, RHS>::Count;
    auto lhs_acc =
        EvalExpr<LHS, Loc, Params...>::template eval_neighbour<
            false, Halo_Top + Halo_T, Halo_Left + Halo_L, Halo_Butt + Halo_B,
            Halo_Right + Halo_R, Offset, Index - 1 - RHSCount,
            LC + Halo_L + Halo_R, LR + Halo_T + Halo_B>(cOffset, t)
            .get_pointer();
    // rhs expression shared mem
    auto rhs_acc = EvalExpr<RHS, Loc, Params...>::template eval_neighbour<
                       false, Halo_Top, Halo_Left, Halo_Butt, Halo_Right,
                       Offset, Index - 1, LC, LR>(cOffset, t).get_pointer();

    auto neighbour = LocalNeighbour<typename C_OP::InType1>(
        lhs_acc, LC + Halo_L + Halo_R, LR + Halo_T + Halo_B);
    // filter for StnFilt
    auto filter = ConstNeighbour<typename C_OP::InType2>(
        rhs_acc, RHS::Type::Cols, RHS::Type::Rows);
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (get_compare<isLocal, LC, Cols>(cOffset.l_c, i, cOffset.g_c)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (get_compare<isLocal, LR, Rows>(cOffset.l_r, j, cOffset.g_r)) {
            neighbour.set_offset(cOffset.l_c + Halo_L + i,
                                 cOffset.l_r + Halo_T + j);
            tools::tuple::get<OutOffset>(t).get_pointer()[calculate_index(
                id_val<isLocal>(cOffset.l_c, cOffset.g_c) + i,
                id_val<isLocal>(cOffset.l_r, cOffset.g_r) + j,
                id_val<isLocal>(LC, Cols), id_val<isLocal>(LR, Rows))] =
                tools::convert<typename MemoryTrait<
                    LfType, decltype(tools::tuple::get<OutOffset>(t))>::Type>(
                    typename C_OP::OP()(neighbour, filter));
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
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_STN_FILT_HPP_