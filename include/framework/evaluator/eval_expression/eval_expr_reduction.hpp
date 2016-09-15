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

/// \file eval_expr_reduction.hpp
/// \brief This file contains the specialisation of the EvalExpr
/// for RDCN( reduction operation node).

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_REDUCTION_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_REDUCTION_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an RDCN(reduction operation) expression.
template <typename C_OP, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL, typename Loc, typename... Params>
struct EvalExpr<RDCN<C_OP, RHS, Cols, Rows, LfType, LVL>, Loc, Params...> {
  // no eval point
  //-- -
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
    constexpr size_t LC_Ratio = RHS::CThread / Cols;
    constexpr size_t LR_Ratio = RHS::RThread / Rows;
    // lhs expression shared mem
    auto nested_acc = EvalExpr<RHS, Loc, Params...>::template eval_neighbour<
                          false, Halo_Top, Halo_Left, Halo_Butt, Halo_Right,
                          Offset, Index - 1, LC, LR>(cOffset, t).get_pointer();

    if ((cOffset.l_c < (cOffset.cLRng / LC_Ratio)) &&
        (cOffset.l_r < (cOffset.rLRng / LR_Ratio))) {
      static constexpr size_t Neighbour_LC_Ratio =
          LC_Ratio / (RHS::Type::Cols / Cols);
      static constexpr size_t Neighbour_LR_Ratio =
          LR_Ratio / (RHS::Type::Rows / Rows);
      auto neighbour = LocalNeighbour<typename C_OP::InType>(
          nested_acc, LC / Neighbour_LC_Ratio, LR / Neighbour_LR_Ratio);
      size_t g_c = ((cOffset.g_c - cOffset.l_c) / LC_Ratio) + cOffset.l_c;
      size_t g_r = ((cOffset.g_r - cOffset.l_r) / LR_Ratio) + cOffset.l_r;

      for (int i = 0; i < LC / LC_Ratio; i += (cOffset.cLRng / LC_Ratio)) {
        if (get_compare<isLocal, LC / LC_Ratio, Cols>(cOffset.l_c, i, g_c)) {
          for (size_t j = 0; j < LR / LR_Ratio;
               j += (cOffset.rLRng / LR_Ratio)) {
            if (get_compare<isLocal, LR / LR_Ratio, Rows>(cOffset.l_r, j,
                                                          g_r)) {
              neighbour.set_offset((cOffset.l_c + i), (cOffset.l_r + j));
              tools::tuple::get<OutOffset>(t).get_pointer()[calculate_index(
                  id_val<isLocal>(cOffset.l_c, g_c) + i,
                  id_val<isLocal>(cOffset.l_r, g_r) + j,
                  id_val<isLocal>(LC / LC_Ratio, Cols),
                  id_val<isLocal>(LR / LR_Ratio, Rows))] =
                  tools::convert<typename MemoryTrait<
                      LfType, decltype(tools::tuple::get<OutOffset>(t))>::Type>(
                      typename C_OP::OP()(neighbour));
            }
          }
        }
      }
    }
    // here you need to put a local barrier
    cOffset.barrier();
    // return the valid neighbour area for your parent
    return tools::tuple::get<OutOffset>(t);
  }
  // eval global
  template <bool IsRoot, size_t Offset, size_t Index, size_t LC, size_t LR>
  static auto eval_global_neighbour(Loc &cOffset,
                                    const tools::tuple::Tuple<Params...> &t)
      -> decltype(
          tools::tuple::get<OutputLocation<IsRoot, Offset + Index - 1>::ID>(
              t)) {
    constexpr size_t OutOffset = OutputLocation<IsRoot, Offset + Index - 1>::ID;
    constexpr bool isLocal =
        Trait<typename tools::RemoveAll<decltype(
            tools::tuple::get<OutOffset>(t))>::Type>::scope == scope::Local;
    // lhs expression shared mem
    auto nested_acc =
        EvalExpr<RHS, Loc, Params...>::template eval_global_neighbour<
            false, Offset, Index - 1, LC, LR>(cOffset, t).get_pointer();
    // here the neighbour is the entire output
    auto reduction = GlobalNeighbour<typename C_OP::InType>(
        nested_acc, RHS::Type::Cols, RHS::Type::Rows);
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (get_compare<isLocal, LC, Cols>(cOffset.l_c, i, cOffset.g_c)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (get_compare<isLocal, LR, Rows>(cOffset.l_r, j, cOffset.g_r)) {
            reduction.set_offset(cOffset.g_c + i, cOffset.g_r + j);
            tools::tuple::get<OutOffset>(t).get_pointer()[calculate_index(
                id_val<isLocal>(cOffset.l_c, cOffset.g_c) + i,
                id_val<isLocal>(cOffset.l_r, cOffset.g_r) + j,
                id_val<isLocal>(LC, Cols), id_val<isLocal>(LR, Rows))] =
                ((tools::convert<typename MemoryTrait<
                    LfType, decltype(tools::tuple::get<OutOffset>(t))>::Type>(
                    typename C_OP::OP()(reduction))));
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
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_REDUCTION_HPP_
