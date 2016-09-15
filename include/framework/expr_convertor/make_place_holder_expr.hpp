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

/// \file make_place_holder_expr.hpp
/// \brief PlaceHolder expression helper is used to create an expression in
/// which the leafNodes of visionMemories has been replaced by leafNodes of
/// PlaceHolders.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_MAKE_PLACE_HOLDER_EXPR_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_MAKE_PLACE_HOLDER_EXPR_HPP_

namespace visioncpp {
namespace internal {
/// \brief specialisation of MakePlaceHolderExprHelper where the node is
/// leaf node.
template <typename RHS, size_t LVL, size_t N>
struct MakePlaceHolderExprHelper<internal::expr_category::Unary,
                                 LeafNode<RHS, LVL>, N> {
  using Type =
      LeafNode<PlaceHolder<RHS::LeafType, N, RHS::Cols, RHS::Rows, RHS::scope>,
               LVL>;
};
/// \brief specialisation of MakePlaceHolderExprHelper where the operation of
/// the node is unary (the node has one child).
template <typename Expr, size_t N>
struct MakePlaceHolderExprHelper<internal::expr_category::Unary, Expr, N> {
  using RHSPlaceHolderType =
      typename MakePlaceHolderExprHelper<Expr::RHSExpr::ND_Category,
                                         typename Expr::RHSExpr, N>::Type;
  using Type = typename Expr::template ExprExchange<RHSPlaceHolderType>;
};
/// \brief specialisation of MakePlaceHolderExprHelper where the operation of
/// the node is binary (the node has two children).
template <typename Expr, size_t N>
struct MakePlaceHolderExprHelper<internal::expr_category::Binary, Expr, N> {
  static constexpr size_t RHSLeafCount =
      LeafCount<Expr::RHSExpr::ND_Category, typename Expr::RHSExpr>::Count;

  using LHSPlaceHolderType =
      typename MakePlaceHolderExprHelper<Expr::LHSExpr::ND_Category,
                                         typename Expr::LHSExpr,
                                         N - RHSLeafCount>::Type;
  using RHSPlaceHolderType =
      typename MakePlaceHolderExprHelper<Expr::RHSExpr::ND_Category,
                                         typename Expr::RHSExpr, N>::Type;

  using Type = typename Expr::template ExprExchange<LHSPlaceHolderType,
                                                    RHSPlaceHolderType>;
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_MAKE_PLACE_HOLDER_EXPR_HPP_