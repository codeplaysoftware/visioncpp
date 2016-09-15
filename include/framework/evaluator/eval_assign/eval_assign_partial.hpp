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

/// \file eval_assign_partial.hpp
/// This file contains the specialisation of the Evaluator struct for assign
/// when it is a root struct

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_PARTIAL_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_PARTIAL_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the Evaluator when the expression is an
/// ParallelCopy expression and the ops_category is PointOP.
template <size_t OutputIndex, size_t Offset, size_t LC, size_t LR, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t OffsetColIn,
          size_t OffsetRowIn, size_t OffsetColOut, size_t OffsetRowOut,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct Evaluator<ops_category::PointOP, OutputIndex, Offset, LC, LR,
                 ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                              OffsetColOut, OffsetRowOut, LfType, LVL>,
                 Loc, Params...> {
  using Expr = ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                            OffsetColOut, OffsetRowOut, LfType, LVL>;
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    using RHS_Eval_Expr = EvalExpr<RHS, Loc, Params...>;
    using LHS_Eval_Expr = EvalExpr<LHS, Loc, Params...>;

    using ElementType =
        typename MemoryTrait<Expr::LeafType,
                             decltype(tools::tuple::get<0>(t))>::Type;

    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (((cOffset.g_c + OffsetColIn + i) < RHS::Type::Cols) &&
          ((cOffset.g_c + OffsetColOut + i) < LHS::Type::Cols)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (((cOffset.g_r + OffsetRowIn + j) < RHS::Type::Rows) &&
              ((cOffset.g_r + OffsetRowOut + j) < LHS::Type::Rows)) {
            cOffset.pointOp_gc = cOffset.g_c + i + OffsetColIn;
            cOffset.pointOp_gr = cOffset.g_r + j + OffsetRowIn;

            LHS_Eval_Expr::get_accessor(t).get_pointer()[calculate_index(
                cOffset.g_c + i + OffsetColOut, cOffset.g_r + j + OffsetRowOut,
                LHS::Type::Cols, LHS::Type::Rows)] =
                tools::convert<ElementType>(
                    RHS_Eval_Expr::eval_point(cOffset, t));
          }
        }
      }
    }
  }
};

/// \brief Partial specialisation of the Evaluator when the expression is an
/// ParallelCopy expression and the ops_category is NeighbourOP.
template <size_t OutputIndex, size_t Offset, size_t LC, size_t LR, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t OffsetColIn,
          size_t OffsetRowIn, size_t OffsetColOut, size_t OffsetRowOut,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct Evaluator<ops_category::NeighbourOP, OutputIndex, Offset, LC, LR,
                 ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                              OffsetColOut, OffsetRowOut, LfType, LVL>,
                 Loc, Params...> {
  using Expr = ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                            OffsetColOut, OffsetRowOut, LfType, LVL>;
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    using RHS_Eval_Expr = EvalExpr<RHS, Loc, Params...>;
    using LHS_Eval_Expr = EvalExpr<LHS, Loc, Params...>;

    constexpr size_t LC_Ratio = Expr::CThread / Cols;
    constexpr size_t LR_Ratio = Expr::RThread / Rows;
    constexpr size_t RHS_LC_Ratio = RHS::CThread / RHS::Type::Cols;
    constexpr size_t RHS_LR_Ratio = RHS::RThread / RHS::Type::Rows;
    auto rhs_acc2 = EvalExpr<RHS, Loc, Params...>::template eval_neighbour<
        false, 0, 0, 0, 0, Offset, OutputIndex, LC, LR>(cOffset, t);
    constexpr bool isLocal =
        Trait<typename tools::RemoveAll<decltype(rhs_acc2)>::Type>::scope ==
        scope::Local;
    cOffset.global_barrier();
    auto rhs_acc1 = RHS_Eval_Expr::get_accessor(t);
    auto lhs_acc1 = LHS_Eval_Expr::get_accessor(t);
    auto rhs_acc = rhs_acc2.get_pointer();
    auto lhs_acc = LHS_Eval_Expr::get_accessor(t).get_pointer();
    static_assert(RHS_LR_Ratio == LR_Ratio && RHS_LC_Ratio == LC_Ratio,
                  "You made a programing mistake. The kernel must break when "
                  "the two are not equal");
    if ((cOffset.l_c < (cOffset.cLRng / LC_Ratio)) &&
        (cOffset.l_r < (cOffset.rLRng / LR_Ratio))) {
      size_t g_c = ((cOffset.g_c - cOffset.l_c) / LC_Ratio) + cOffset.l_c;
      size_t g_r = ((cOffset.g_r - cOffset.l_r) / LR_Ratio) + cOffset.l_r;

      for (int i = 0; i < LC / LC_Ratio; i += (cOffset.cLRng / LC_Ratio)) {
        if (get_compare<isLocal, LC / RHS_LC_Ratio, RHS::Type::Cols>(
                cOffset.l_c, i, g_c) &&
            (g_c + i + OffsetColOut < LHS::Type::Cols)) {
          for (size_t j = 0; j < LR / LR_Ratio;
               j += (cOffset.rLRng / LR_Ratio)) {
            if (get_compare<isLocal, LR / LR_Ratio, RHS::Type::Rows>(
                    cOffset.l_r, j, g_r) &&
                (g_r + j + OffsetRowOut < LHS::Type::Rows)) {
              lhs_acc[calculate_index(g_c + i + OffsetColOut,
                                      g_r + j + OffsetRowOut, LHS::Type::Cols,
                                      LHS::Type::Rows)] =
                  rhs_acc[calculate_index(cOffset.l_c + i, cOffset.l_r + j,
                                          LC / LC_Ratio, LR / LR_Ratio)];
            }
          }
        }
      }
    }
  }
};

/// \brief Partial specialisation of the Evaluator when the expression
/// is an ParallelCopy expression and the ops_category is
/// GlobalNeighbourOP.

template <size_t OutputIndex, size_t Offset, size_t LC, size_t LR, typename LHS,
          typename RHS, size_t Cols, size_t Rows, size_t OffsetColIn,
          size_t OffsetRowIn, size_t OffsetColOut, size_t OffsetRowOut,
          size_t LfType, size_t LVL, typename Loc, typename... Params>
struct Evaluator<ops_category::GlobalNeighbourOP, OutputIndex, Offset, LC, LR,
                 ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                              OffsetColOut, OffsetRowOut, LfType, LVL>,
                 Loc, Params...> {
  using Expr = ParallelCopy<LHS, RHS, Cols, Rows, OffsetColIn, OffsetRowIn,
                            OffsetColOut, OffsetRowOut, LfType, LVL>;
  static inline void eval(Loc &cOffset,
                          const tools::tuple::Tuple<Params...> &t) {
    // the reason for calling like that is to avoid the last shared
    // memory to be created for the RHS child of the root
    using RHS_Eval_Expr = EvalExpr<RHS, Loc, Params...>;
    using LHS_Eval_Expr = EvalExpr<LHS, Loc, Params...>;
    EvalExpr<RHS, Loc, Params...>::template eval_global_neighbour<
        true, Offset, OutputIndex, LC, LR>(cOffset, t);
    auto rhs_acc = RHS_Eval_Expr::get_accessor(t).get_pointer();
    auto lhs_acc = LHS_Eval_Expr::get_accessor(t).get_pointer();

    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if (((cOffset.g_c + OffsetColIn + i) < RHS::Type::Cols) &&
          ((cOffset.g_c + OffsetColOut + i) < LHS::Type::Cols)) {
        for (int j = 0; j < LR; j += cOffset.rLRng) {
          if (((cOffset.g_r + OffsetRowIn + j) < RHS::Type::Rows) &&
              ((cOffset.g_r + OffsetRowOut + j) < LHS::Type::Rows)) {
            cOffset.pointOp_gc = cOffset.g_c + i + OffsetColIn;
            cOffset.pointOp_gr = cOffset.g_r + j + OffsetRowIn;
            lhs_acc[(cOffset.g_c + i + OffsetColOut) +
                    (LHS::Type::Cols * (cOffset.g_r + j + OffsetRowOut))] =
                rhs_acc[(cOffset.g_c + i + OffsetColIn) +
                        (RHS::Type::Cols * (cOffset.g_r + j + OffsetRowIn))];
          }
        }
      }
    }
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_ASSIGN_EVAL_ASSIGN_PARTIAL_HPP_
