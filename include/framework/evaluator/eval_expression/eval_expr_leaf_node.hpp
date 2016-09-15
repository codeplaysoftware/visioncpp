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

/// \file eval_expr_leaf_node.hpp
/// \brief This file contains the specialisation of the EvalExpr
/// for LeafNode.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_LEAF_NODE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_LEAF_NODE_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the EvalExpr when the expression is
/// an LeafNode expression and the internal::ops_category is NeighbourOP.
template <size_t Memory_Type, size_t N, size_t Cols, size_t Rows, size_t LVL,
          size_t Sc, typename Loc, typename... Params>
struct EvalExpr<LeafNode<PlaceHolder<Memory_Type, N, Cols, Rows, Sc>, LVL>, Loc,
                Params...> {
  using Expr = LeafNode<PlaceHolder<Memory_Type, N, Cols, Rows, Sc>, LVL>;

  static auto get_accessor(const tools::tuple::Tuple<Params...> &t)
      -> decltype(tools::tuple::get<N>(t)) {
    return tools::tuple::get<N>(t);
  }
  /// \brief evaluate function when the internal::ops_category is PointOP.
  static inline auto eval_point(Loc &cOffset,
                                const tools::tuple::Tuple<Params...> &t)
      -> decltype(tools::tuple::get<N>(t)
                      .get_pointer()[cOffset.pointOp_gc +
                                     (Cols * cOffset.pointOp_gr)]) {
    return tools::tuple::get<N>(t).get_pointer()[calculate_index(
        cOffset.pointOp_gc, cOffset.pointOp_gr, Cols, Rows)];
  }
  /// \brief evaluate function when the internal::ops_category is NeighbourOP.
  template <bool IsRoot, size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
            size_t Halo_Right, size_t Offset, size_t Index, size_t LC,
            size_t LR>
  static inline auto eval_neighbour(Loc &cOffset,
                                    const tools::tuple::Tuple<Params...> &t)
      -> decltype(tools::tuple::get<Index_Finder<
          N, OutputLocation<IsRoot, Offset + Index - 1>::ID, Memory_Type,
          Trait<typename tools::RemoveAll<
              decltype(tools::tuple::get<N>(t))>::Type>::scope>::Index>(t)) {
    fill_local_neighbour<
        Halo_Top, Halo_Left, Halo_Butt, Halo_Right,
        Index_Finder<N, OutputLocation<IsRoot, Offset + Index - 1>::ID,
                     Memory_Type, Sc>::Index,
        LC, LR, Expr>(cOffset, t);
    return tools::tuple::get<
        Index_Finder<N, OutputLocation<IsRoot, Offset + Index - 1>::ID,
                     Memory_Type, Sc>::Index>(t);
  }
  /// \brief evaluate function when the internal::ops_category is
  /// GlobalNeighbour;
  template <bool IsRoot, size_t Offset, size_t Index, size_t LC, size_t LR>
  static inline auto eval_global_neighbour(
      Loc &, const tools::tuple::Tuple<Params...> &t)
      -> decltype(tools::tuple::get<N>(t)) {
    return tools::tuple::get<N>(t);
  }
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVAL_EXPRESSION_EVAL_EXPR_LEAF_NODE_HPP_
